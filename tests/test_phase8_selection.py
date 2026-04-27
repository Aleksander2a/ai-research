"""Phase 8 tests: CPCV / PBO / Deflated Sharpe / Romano-Wolf / pre-registration / vault."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autosignalx.eval import (
    cpcv,
    deflated_sharpe,
    holdout_vault,
    pbo,
    preregistration,
    romano_wolf,
)

# ---------- CPCV ---------- #


def test_cpcv_paths_sizes():
    origins = pd.date_range("2024-01-01", periods=60).tolist()
    paths = cpcv.cpcv_paths(origins, n_folds=6, k_test=2, embargo=1)
    assert len(paths) == 15  # C(6, 2)
    for p in paths:
        assert len(p.test_origins) > 0
        assert len(p.train_origins) > 0
        assert set(p.train_origins).isdisjoint(set(p.test_origins))


def test_cpcv_paths_rejects_too_few_origins():
    origins = pd.date_range("2024-01-01", periods=4).tolist()
    with pytest.raises(ValueError):
        cpcv.cpcv_paths(origins, n_folds=6, k_test=2)


def _two_method_forecasts(seed: int = 0, n_origins: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_origins):
        origin = pd.Timestamp("2024-01-01") + pd.Timedelta(days=k)
        for h in range(1, 4):
            ts = origin + pd.Timedelta(days=h)
            target = float(100 + rng.normal(0, 1))
            rows.append({
                "asset": "SPY", "timestamp": ts, "forecast_origin": origin,
                "horizon": h, "method": "naive",
                "prediction": float(100 + rng.normal(0, 1.5)),
                "origin_value": 100.0, "target": target,
            })
            rows.append({
                "asset": "SPY", "timestamp": ts, "forecast_origin": origin,
                "horizon": h, "method": "good",
                "prediction": target + float(rng.normal(0, 0.4)),
                "origin_value": 100.0, "target": target,
            })
    return pd.DataFrame(rows)


def test_cpcv_skill_distribution_runs():
    df = _two_method_forecasts(seed=42, n_origins=120)
    res = cpcv.cpcv_skill_distribution(df, method="good", baseline_method="naive", n_folds=6)
    assert res["n_paths"] > 0
    assert res["skill_mean"] > 0  # genuinely good method should be positive on average


# ---------- PBO ---------- #


def test_pbo_close_to_zero_when_one_strategy_dominates():
    rng = np.random.default_rng(0)
    n_periods, n_strats = 200, 5
    M = rng.normal(0, 1, size=(n_periods, n_strats))
    # Make strategy 0 reliably best by adding a constant
    M[:, 0] += 2.0
    res = pbo.probability_of_backtest_overfitting(M, s=10)
    # Strategy 0 always wins IS and OOS -> PBO should be very low
    assert res.pbo < 0.2


def test_pbo_around_half_when_pure_noise():
    rng = np.random.default_rng(0)
    M = rng.normal(0, 1, size=(200, 8))
    res = pbo.probability_of_backtest_overfitting(M, s=10)
    assert 0.2 < res.pbo < 0.8  # noise: rank flips frequently


def test_pbo_from_forecasts_empty_input_does_not_crash():
    res = pbo.pbo_from_forecasts(pd.DataFrame(), methods=[], baseline="naive")
    assert np.isnan(res.pbo)


# ---------- Deflated Sharpe ---------- #


def test_dsr_zero_returns_returns_nan():
    res = deflated_sharpe.deflated_sharpe_ratio(np.zeros(20), n_trials=5)
    assert np.isnan(res.deflated_sharpe)


def test_dsr_with_real_signal_high_under_few_trials():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.005, size=252)
    res_few = deflated_sharpe.deflated_sharpe_ratio(returns, n_trials=1)
    res_many = deflated_sharpe.deflated_sharpe_ratio(returns, n_trials=10000)
    # More trials should deflate the Sharpe more aggressively
    assert res_many.deflated_sharpe < res_few.deflated_sharpe


def test_expected_max_sharpe_under_null_grows_with_n_trials():
    a = deflated_sharpe.expected_max_sharpe_under_null(10)
    b = deflated_sharpe.expected_max_sharpe_under_null(1000)
    assert b > a


# ---------- Romano-Wolf ---------- #


def test_romano_wolf_reports_per_hypothesis_q():
    rng = np.random.default_rng(0)
    n = 200
    # 3 hypotheses: 1 with real positive effect, 2 null
    h1 = rng.normal(0.05, 0.05, size=n)
    h2 = rng.normal(0.0, 0.05, size=n)
    h3 = rng.normal(0.0, 0.05, size=n)
    D = np.column_stack([h1, h2, h3])
    res = romano_wolf.romano_wolf(D, alpha=0.10, n_bootstrap=200, block_size=10)
    assert len(res.p_adjusted) == 3
    # h1 should be the most significant
    assert res.p_adjusted[0] <= res.p_adjusted[1]
    assert res.p_adjusted[0] <= res.p_adjusted[2]


def test_romano_wolf_handles_degenerate_input():
    res = romano_wolf.romano_wolf(np.zeros((10, 2)), alpha=0.05, n_bootstrap=100, block_size=5)
    # Should not error; survives flag is False or NaN
    assert all(s is False for s in res.survives)


# ---------- Pre-registration ---------- #


def test_preregistration_hash_is_deterministic():
    p = preregistration.PreRegistration(
        hypothesis="H", method="m", baseline="naive",
        filters={"asset": "SPY"}, decision_rule={"p": 0.05},
        predicted_effect={}, falsifier="F",
    )
    p2 = preregistration.PreRegistration(
        hypothesis="H", method="m", baseline="naive",
        filters={"asset": "SPY"}, decision_rule={"p": 0.05},
        predicted_effect={}, falsifier="F",
    )
    assert p.hash() == p2.hash()


def test_preregistration_roundtrip(tmp_path: Path):
    path = tmp_path / "pre.jsonl"
    p = preregistration.PreRegistration(
        hypothesis="H", method="m", baseline="naive",
        filters={}, decision_rule={"p_threshold": 0.05},
        predicted_effect={}, falsifier="F",
    )
    rec = preregistration.register(p, path=path)
    assert rec["id"].startswith("p_")
    loaded = preregistration.load(path=path)
    assert len(loaded) == 1
    # Idempotent re-register
    preregistration.register(p, path=path)
    assert len(preregistration.load(path=path)) == 1


def test_preregistration_resolve(tmp_path: Path):
    p_path = tmp_path / "pre.jsonl"
    r_path = tmp_path / "res.jsonl"
    p = preregistration.PreRegistration(
        hypothesis="H", method="m", baseline="naive",
        filters={}, decision_rule={"p_threshold": 0.05},
        predicted_effect={}, falsifier="F",
    )
    rec = preregistration.register(p, path=p_path)
    preregistration.resolve(rec["id"], promoted=True, evidence={"p": 0.01}, path=r_path)
    res = preregistration.load_resolutions(path=r_path)
    assert len(res) == 1
    assert res[0]["promoted"] is True


def test_preregistration_from_hypothesis_dict():
    h = {
        "hypothesis": "TLT chronos beats naive in regime 3",
        "experiment": {
            "type": "slice_forecasts",
            "params": {"method": "chronos2_multivariate", "asset": "TLT", "regime_id": 3},
        },
    }
    p = preregistration.from_hypothesis_dict(h, session_id="s1", round=0)
    assert p.method == "chronos2_multivariate"
    assert p.filters["asset"] == "TLT"
    assert p.filters["regime_id"] == 3


# ---------- Holdout vault ---------- #


@pytest.fixture
def vault_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr("autosignalx.eval.holdout_vault.VAULT_DIR", tmp_path / "vault")
    monkeypatch.setattr("autosignalx.eval.holdout_vault.VAULT_META", tmp_path / "vault" / "vault.json")
    monkeypatch.setattr("autosignalx.eval.holdout_vault.VAULT_RESULTS", tmp_path / "vault" / "results.json")
    return tmp_path


def test_vault_initialize_and_status(vault_fixture):
    rec = holdout_vault.initialize_vault("2025-01-01", "2025-12-31", description="final")
    assert rec["start"] == "2025-01-01"
    s = holdout_vault.vault_status()
    assert s["initialized"] is True
    assert s["opened"] is False


def test_vault_assert_no_leakage_passes_when_disjoint(vault_fixture):
    holdout_vault.initialize_vault("2025-01-01", "2025-12-31")
    df = pd.DataFrame({
        "forecast_origin": pd.to_datetime(["2024-06-01", "2024-12-31"]),
        "prediction": [1.0, 1.0], "target": [1.0, 1.0],
    })
    holdout_vault.assert_no_vault_leakage(df)  # should not raise


def test_vault_assert_no_leakage_blocks_overlap(vault_fixture):
    holdout_vault.initialize_vault("2025-01-01", "2025-12-31")
    df = pd.DataFrame({
        "forecast_origin": pd.to_datetime(["2025-06-01"]),
        "prediction": [1.0], "target": [1.0],
    })
    with pytest.raises(RuntimeError, match="leakage"):
        holdout_vault.assert_no_vault_leakage(df)


def test_vault_open_records_results(vault_fixture):
    holdout_vault.initialize_vault("2024-01-01", "2024-12-31")
    df = _two_method_forecasts(seed=0, n_origins=40)
    df["forecast_origin"] = pd.to_datetime(df["forecast_origin"])
    res = holdout_vault.open_vault(df, methods=["good", "naive"], baseline="naive")
    assert res["n_rows"] > 0
    # Second open: returns already_opened
    res2 = holdout_vault.open_vault(df, methods=["good", "naive"], baseline="naive")
    assert res2.get("already_opened") is True
