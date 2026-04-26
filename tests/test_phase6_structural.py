"""Tests for Phase 6 structural enrichments: per-regime graph + signal stability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autosignalx.config import settings
from autosignalx.signal import stability

# ---------- Walk-forward signal stability ---------- #


@pytest.fixture
def synthetic_features():
    rng = np.random.default_rng(0)
    n = 1500
    ts = pd.date_range("2020-01-01", periods=n, freq="B")
    feat_df = pd.DataFrame({
        "timestamp": ts,
        "rsi": rng.normal(0, 1, n),
        "macd": rng.normal(0, 1, n),
        "macro_lag1": rng.normal(0, 1, n),
        "noise": rng.normal(0, 1, n),
    })
    # Target depends on rsi in regime 0 and macd in regime 1.
    regime = np.where(np.arange(n) < n // 2, 0, 1)
    target = np.where(
        regime == 0,
        (feat_df["rsi"] > 0).astype(int),
        (feat_df["macd"] > 0).astype(int),
    )
    feat_df["target_direction"] = target
    regime_df = pd.DataFrame({"timestamp": ts, "regime_id": regime})
    return feat_df, regime_df


def test_walk_forward_rank_returns_per_window_ranking(synthetic_features):
    feat_df, regime_df = synthetic_features
    wf = stability.walk_forward_rank(
        features_df=feat_df,
        regime_labels=regime_df,
        feature_cols=["rsi", "macd", "macro_lag1", "noise"],
        n_windows=4,
        min_window_size=200,
    )
    assert not wf.empty
    assert {"window_idx", "regime_id", "feature", "importance", "rank"} <= set(wf.columns)
    # Multiple windows produced
    assert wf["window_idx"].nunique() >= 2


def test_summarise_stability_produces_metrics(synthetic_features):
    feat_df, regime_df = synthetic_features
    wf = stability.walk_forward_rank(
        feat_df, regime_df,
        feature_cols=["rsi", "macd", "macro_lag1", "noise"],
        n_windows=4, min_window_size=200,
    )
    summary = stability.summarise_stability(wf, top_k=2)
    assert {"mean_importance", "mean_rank", "rank_std", "top2_share", "stability"} <= set(summary.columns)
    assert summary["stability"].between(0.0, 1.0).all()


def test_build_and_save_writes_parquets(synthetic_features, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    feat_df, regime_df = synthetic_features
    out = stability.build_and_save(
        features_df=feat_df,
        regime_labels=regime_df,
        feature_cols=["rsi", "macd", "macro_lag1", "noise"],
        n_windows=4,
    )
    assert not out["walk_forward"].empty
    assert (tmp_path / "reports" / "signals" / "walk_forward_ranking.parquet").exists()
    assert (tmp_path / "reports" / "signals" / "signal_stability.parquet").exists()


# ---------- Per-regime graph ---------- #


@pytest.fixture
def synthetic_returns_with_regimes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "repo_root", tmp_path)
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")

    rng = np.random.default_rng(7)
    n = 800
    ts = pd.date_range("2020-01-01", periods=n, freq="B")
    # 4 assets; in regime 0 SPY drives QQQ; in regime 1 TLT drives GLD.
    regime = np.where(np.arange(n) < n // 2, 0, 1)
    spy = rng.normal(0, 1, n)
    tlt = rng.normal(0, 1, n)
    qqq = np.where(regime == 0, 0.7 * spy + 0.3 * rng.normal(0, 1, n), rng.normal(0, 1, n))
    gld = np.where(regime == 1, 0.7 * tlt + 0.3 * rng.normal(0, 1, n), rng.normal(0, 1, n))

    rd = tmp_path / "reports"
    (rd / "regimes").mkdir(parents=True)
    (rd / "graph").mkdir(parents=True)

    pd.DataFrame({"timestamp": ts, "regime_id": regime}).to_parquet(
        rd / "regimes" / "kmeans.parquet"
    )

    # The data layer's loader.load_returns_wide reads from data cache; the
    # per_regime build calls loader.load_returns_wide directly. Patch it.
    return tmp_path, ts, regime, spy, qqq, tlt, gld


def test_per_regime_build_produces_regime_specific_artifacts(synthetic_returns_with_regimes, monkeypatch):
    tmp_path, ts, regime, spy, qqq, tlt, gld = synthetic_returns_with_regimes
    df = pd.DataFrame({
        "SPY": spy, "QQQ": qqq, "TLT": tlt, "GLD": gld,
    }, index=ts)
    df.index.name = "timestamp"

    # Patch the loader so the module sees our synthetic returns frame.
    from autosignalx.data import loader

    monkeypatch.setattr(loader, "load_returns_wide", lambda: df)

    from autosignalx.graph import per_regime as pr

    out = pr.build_per_regime(p_threshold=0.10, max_lag=2, pcorr_threshold=1e-3, min_samples=100)
    assert 0 in out and 1 in out
    for rid, payload in out.items():
        assert not payload["centrality"].empty
        assert payload["centrality"]["regime_id"].iloc[0] == rid

    # Persisted files
    assert (tmp_path / "reports" / "graph" / "per_regime" / "regime_0").exists()
    sens = pr.load_regime_sensitivity()
    assert not sens.empty
    assert "betweenness_centrality_range" in sens.columns


def test_load_per_regime_round_trip(synthetic_returns_with_regimes, monkeypatch):
    tmp_path, ts, regime, spy, qqq, tlt, gld = synthetic_returns_with_regimes
    df = pd.DataFrame({"SPY": spy, "QQQ": qqq, "TLT": tlt, "GLD": gld}, index=ts)
    df.index.name = "timestamp"
    from autosignalx.data import loader

    monkeypatch.setattr(loader, "load_returns_wide", lambda: df)

    from autosignalx.graph import per_regime as pr

    pr.build_per_regime(p_threshold=0.10, max_lag=2, min_samples=100)
    loaded = pr.load_per_regime()
    assert set(loaded.keys()) == {0, 1}
