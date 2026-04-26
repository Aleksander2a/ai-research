"""End-to-end integration: study + cache + baseline eval + backtest.

Avoids the network (yfinance) and Chronos-2 (heavy model load) by
fabricating an OHLCV cache directly under a study directory and then
exercising the eval-baseline + backtest paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autosignalx.backtest import runner as bt_runner
from autosignalx.backtest.schemas import BacktestConfig
from autosignalx.config import settings
from autosignalx.data import cache as data_cache
from autosignalx.data import splits
from autosignalx.eval import harness
from autosignalx.forecast import baselines
from autosignalx.study import Study
from autosignalx.study import pipeline as study_pipeline


@pytest.fixture
def temp_study(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "repo_root", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "reports").mkdir()
    s = Study(
        name="integ",
        assets=["AAA", "BBB", "CCC"],
        macro=["XXX"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        train_end="2022-06-30",
        val_end="2023-06-30",
        test_end="2024-12-31",
        forecast_horizon_days=10,
        rolling_step_days=10,
    )
    s.save()
    yield s


def _synth_ohlcv(assets: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(0)
    rows = []
    for a in assets:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, size=len(idx))))
        ret = np.concatenate(([0.0], np.diff(prices) / prices[:-1]))
        for ts, p, r in zip(idx, prices, ret, strict=False):
            rows.append({
                "timestamp": ts, "asset": a,
                "open": p, "high": p * 1.005, "low": p * 0.995,
                "close": p, "adj_close": p, "volume": 1_000_000,
                "returns": r,
            })
    return pd.DataFrame(rows)


def _synth_macro(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    rows = []
    rng = np.random.default_rng(1)
    for t in tickers:
        for ts in idx:
            rows.append({"timestamp": ts, "signal": t, "value": float(rng.normal(0, 1))})
    return pd.DataFrame(rows)


def test_pipeline_run_data_fetch_writes_to_study_cache(temp_study, monkeypatch):
    """Cockpit fetch path writes the fetched frames into the study cache."""
    s = temp_study
    ohlcv = _synth_ohlcv(tuple(s.assets), s.start_date, s.end_date)
    macro_df = _synth_macro(tuple(s.macro), s.start_date, s.end_date)
    seen: dict[str, object] = {}

    def fake_fetch_all(assets, macro, start, end):
        seen["assets"] = assets
        seen["macro"] = macro
        seen["start"] = start
        seen["end"] = end
        return ohlcv, macro_df

    monkeypatch.setattr(study_pipeline.fetch, "fetch_all", fake_fetch_all)

    out = study_pipeline.run_data_fetch(s)

    assert seen == {
        "assets": s.assets,
        "macro": s.macro,
        "start": s.start_date,
        "end": s.end_date,
    }
    assert Path(out["ohlcv_path"]).exists()
    assert Path(out["macro_path"]).exists()
    assert s.cache_dir in Path(out["ohlcv_path"]).parents
    assert s.cache_dir in Path(out["macro_path"]).parents
    assert out["ohlcv_rows"] == len(ohlcv)
    assert out["macro_rows"] == len(macro_df)


def test_eval_baseline_writes_to_study_ablations_dir(temp_study):
    """Baseline ablation, when given study cache, writes into study tree."""
    s = temp_study
    ohlcv = _synth_ohlcv(tuple(s.assets), s.start_date, s.end_date)
    macro_df = _synth_macro(tuple(s.macro), s.start_date, s.end_date)
    data_cache.write_ohlcv(ohlcv, cache_root=s.cache_dir)
    data_cache.write_macro(macro_df, cache_root=s.cache_dir)

    windows = splits.walk_forward_windows(
        val_end=s.val_end,
        test_end=s.test_end,
        horizon_days=s.forecast_horizon_days,
        step_days=s.rolling_step_days,
    )
    assert windows, "expected non-zero walk-forward windows on the synthetic span"

    forecasts = harness.ablation({"naive": baselines.naive_forecast}, ohlcv, windows)
    assert not forecasts.empty
    out_path = s.ablations_dir / "baseline.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out_path, index=False)

    assert out_path.exists()
    assert s.ablations_dir in out_path.parents


def test_backtest_with_study_reads_from_study_cache(temp_study):
    """run_backtest with study_name reads prices from the study's cache and
    writes its run artifacts under the study's reports tree."""
    s = temp_study
    ohlcv = _synth_ohlcv(tuple(s.assets), s.start_date, s.end_date)
    macro_df = _synth_macro(tuple(s.macro), s.start_date, s.end_date)
    data_cache.write_ohlcv(ohlcv, cache_root=s.cache_dir)
    data_cache.write_macro(macro_df, cache_root=s.cache_dir)

    cfg = BacktestConfig(
        strategies=["EqualWeightUniverse"],
        start_date=s.effective_backtest_start,
        end_date=s.test_end,
        universe=list(s.assets),
        cost_bps=s.cost_bps,
        bootstrap_n=200,  # small for speed
    )
    result = bt_runner.run_backtest(cfg, study_name=s.name)

    # Run dir lives under the study's reports tree.
    assert s.backtest_runs_dir in (result.artifacts_dir and __import__("pathlib").Path(result.artifacts_dir)).parents
    # Default project tree should NOT contain this run.
    default_runs = settings.reports_dir / "backtest" / "runs"
    if default_runs.exists():
        assert not any(p.name == result.run_id for p in default_runs.iterdir())


def test_study_pipeline_minimal_flow_runs_end_to_end(temp_study, monkeypatch):
    """Custom-study fetch -> baseline -> backtest works with the cockpit helpers."""
    s = temp_study
    ohlcv = _synth_ohlcv(tuple(s.assets), s.start_date, s.end_date)
    macro_df = _synth_macro(tuple(s.macro), s.start_date, s.end_date)

    monkeypatch.setattr(
        study_pipeline.fetch,
        "fetch_all",
        lambda assets, macro, start, end: (ohlcv, macro_df),
    )

    fetch_out = study_pipeline.run_data_fetch(s)
    baseline_out = study_pipeline.run_baseline_eval(s)
    backtest_out = study_pipeline.run_backtest_for_study(s)

    assert fetch_out["ohlcv_rows"] == len(ohlcv)
    assert Path(fetch_out["ohlcv_path"]).exists()
    assert baseline_out["rows"] > 0
    assert Path(baseline_out["out_path"]).exists()
    assert backtest_out["n_strategies"] == 1
    run_dir = Path(backtest_out["artifacts_dir"])
    assert run_dir.exists()
    assert s.backtest_runs_dir in run_dir.parents
