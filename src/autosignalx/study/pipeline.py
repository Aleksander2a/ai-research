"""Pure-Python pipeline entry points for a Study.

These functions wrap the same work the CLI does (data fetch, baseline
ablation, backtest) but expose plain Python signatures so the cockpit
can call them inline (with a spinner) without subprocessing the CLI.
Each function writes to the study's artifact tree and returns a small
summary dict the caller can render.

Heavy steps (Chronos-2 in particular) are deliberately *not* exposed
here: those should be launched from the CLI so logs stream to the
terminal. The cockpit panel surfaces the CLI command as copyable text.
"""

from __future__ import annotations

from typing import Any

from autosignalx.backtest import runner as bt_runner
from autosignalx.backtest.schemas import BacktestConfig
from autosignalx.backtest.strategy_selection import (
    default_study_strategies,
    ensure_strategy_prerequisites,
)
from autosignalx.data import cache as data_cache
from autosignalx.data import fetch, splits
from autosignalx.eval import harness
from autosignalx.forecast import baselines
from autosignalx.study.config import Study


def run_data_fetch(study: Study) -> dict[str, Any]:
    """Pull OHLCV + macro for the study from yfinance and write to its cache."""
    ohlcv, macro_df = fetch.fetch_all(
        study.assets, study.macro, study.start_date, study.end_date
    )
    ohlcv_path = data_cache.write_ohlcv(ohlcv, cache_root=study.cache_dir)
    macro_path = data_cache.write_macro(macro_df, cache_root=study.cache_dir)
    return {
        "ohlcv_rows": int(len(ohlcv)),
        "macro_rows": int(len(macro_df)),
        "ohlcv_path": str(ohlcv_path),
        "macro_path": str(macro_path),
    }


def run_baseline_eval(study: Study) -> dict[str, Any]:
    """Run naive + seasonal_naive + arima ablation, write to study ablations."""
    ohlcv = data_cache.read_ohlcv(cache_root=study.cache_dir)
    windows = splits.walk_forward_windows(
        val_end=study.val_end,
        test_end=study.test_end,
        horizon_days=study.forecast_horizon_days,
        step_days=study.rolling_step_days,
    )
    method_map = {
        "naive": baselines.naive_forecast,
        "seasonal_naive": baselines.seasonal_naive_forecast,
        "arima": baselines.arima_forecast,
    }
    forecasts = harness.ablation(method_map, ohlcv, windows)
    if forecasts.empty:
        return {"rows": 0, "windows": len(windows), "out_path": None}
    out_dir = study.ablations_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline.parquet"
    forecasts.to_parquet(out_path, index=False)
    return {
        "rows": int(len(forecasts)),
        "windows": len(windows),
        "out_path": str(out_path),
    }


def run_backtest_for_study(
    study: Study,
    strategies: list[str] | None = None,
    cost_bps: float | None = None,
) -> dict[str, Any]:
    """Run a backtest using the study's config."""
    if strategies is None:
        selected_strategies = default_study_strategies(study)
    else:
        selected_strategies = list(strategies)
        if not selected_strategies:
            raise ValueError("At least one strategy must be provided.")
        ensure_strategy_prerequisites(
            selected_strategies, study=study, universe=list(study.assets)
        )

    cfg = BacktestConfig(
        strategies=selected_strategies,
        start_date=study.effective_backtest_start,
        end_date=study.test_end,
        cost_bps=cost_bps if cost_bps is not None else study.cost_bps,
        universe=list(study.assets),
        bootstrap_n=2000,
    )
    result = bt_runner.run_backtest(cfg, study_name=study.name)
    return {
        "run_id": result.run_id,
        "artifacts_dir": result.artifacts_dir,
        "n_strategies": len(result.strategies),
    }


def pipeline_status(study: Study) -> dict[str, Any]:
    """Inventory which artifacts already exist for this study."""
    ohlcv_path = study.cache_dir / "ohlcv.parquet"
    macro_path = study.cache_dir / "macro.parquet"
    baseline_path = study.ablations_dir / "baseline.parquet"
    chronos_path = study.ablations_dir / "chronos2.parquet"
    backtest_runs = (
        sorted([p for p in study.backtest_runs_dir.iterdir() if p.is_dir()])
        if study.backtest_runs_dir.exists()
        else []
    )
    return {
        "ohlcv": ohlcv_path.exists(),
        "macro": macro_path.exists(),
        "baseline": baseline_path.exists(),
        "chronos": chronos_path.exists(),
        "n_backtest_runs": len(backtest_runs),
        "latest_run": backtest_runs[-1].name if backtest_runs else None,
    }
