"""Top-level orchestrator: run a configured set of strategies, persist
artifacts, return a typed result.

Loads adjusted-close prices from the OHLCV cache, slices to the backtest
window, and runs each strategy through ``engine.run_engine``. Asserts
the temporal-disjointness invariant before any computation.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from autosignalx.backtest import engine, metrics, signals, strategies
from autosignalx.backtest.schemas import (
    DISCOVERY_END,
    BacktestConfig,
    BacktestResult,
    StrategyResult,
)
from autosignalx.config import settings
from autosignalx.data import loader

DEFAULT_UNIVERSE = ("SPY", "QQQ", "IWM", "GLD", "TLT", "EFA", "EEM", "HYG")


def _load_prices(
    universe: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    wide = loader.load_close_wide()
    missing = [a for a in universe if a not in wide.columns]
    if missing:
        raise ValueError(
            f"OHLCV cache missing assets {missing}; run `autosignalx data fetch`"
        )
    px = wide[list(universe)].loc[start:end].dropna(how="all")
    if px.empty:
        raise ValueError(
            f"no price rows in window [{start.date()}, {end.date()}]; check dates"
        )
    return px


def _build_context(cfg: BacktestConfig) -> dict:
    """Best-effort load of forecast/regime/finding inputs.

    Each load is wrapped: if the artifact is missing, the corresponding
    context key is ``None`` and a strategy that needs it will raise a
    clear error when invoked. Strategies that don't need it are
    unaffected.
    """
    ctx: dict = {}
    try:
        ctx["forecast_signals"] = signals.load_forecast_signals()
    except FileNotFoundError:
        ctx["forecast_signals"] = None
    try:
        ctx["regimes"] = signals.load_regime_series()
    except FileNotFoundError:
        ctx["regimes"] = None
    ctx["findings"] = signals.load_promoted_findings()
    ctx["config"] = cfg
    return ctx


def _assert_no_leakage(start: pd.Timestamp) -> None:
    """The backtest window must start strictly after discovery ended."""
    if start <= pd.Timestamp(DISCOVERY_END):
        raise ValueError(
            f"backtest start {start.date()} must be strictly after the discovery "
            f"window end {DISCOVERY_END.date()}; otherwise look-ahead bias is "
            f"possible since the regime model and agent findings were fit on "
            f"data through {DISCOVERY_END.date()}"
        )


def run_backtest(
    config: BacktestConfig | None = None,
    artifacts_root: Path | None = None,
) -> BacktestResult:
    """Run the configured strategies and persist artifacts to disk."""
    cfg = config or BacktestConfig()
    universe = tuple(cfg.universe) if cfg.universe else DEFAULT_UNIVERSE
    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    _assert_no_leakage(start)

    prices = _load_prices(universe, start, end)
    context = _build_context(cfg)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    base = artifacts_root or (settings.reports_dir / "backtest" / "runs")
    out_dir = base / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    portfolio_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    summaries: list[StrategyResult] = []
    metrics_payload: dict[str, dict] = {}

    for spec in cfg.strategies:
        strat = strategies.parse_strategy_spec(spec)
        display_name = strat.name
        weights = strat.weights(prices, context=context)
        weights = weights.reindex(prices.index).fillna(0.0)
        out = engine.run_engine(weights, prices, cost_bps=cfg.cost_bps)

        m = metrics.compute_all(
            out["equity"], out["returns"], out["turnover"], out["cost"]
        )
        sanitized = metrics.sanitize_metrics(m)
        regime_series = context.get("regimes")
        if regime_series is not None:
            sanitized["per_regime"] = metrics.compute_per_regime(
                out["returns"], out["turnover"], out["cost"], regime_series
            )
        metrics_payload[display_name] = sanitized
        summaries.append(StrategyResult(name=display_name, **m))

        per_day = pd.DataFrame(
            {
                "timestamp": out["equity"].index,
                "strategy": display_name,
                "return": out["returns"].values,
                "equity": out["equity"].values,
                "turnover": out["turnover"].values,
                "cost": out["cost"].values,
            }
        )
        portfolio_rows.append(per_day)

        trades = engine.assemble_trades(weights, cost_bps=cfg.cost_bps)
        trades.insert(1, "strategy", display_name)
        trade_rows.append(trades)

    portfolio_df = pd.concat(portfolio_rows, ignore_index=True)
    trades_df = (
        pd.concat(trade_rows, ignore_index=True)
        if trade_rows
        else pd.DataFrame(columns=["timestamp", "strategy", "asset", "dweight", "cost"])
    )

    portfolio_path = out_dir / "portfolio_daily.parquet"
    trades_path = out_dir / "trades.parquet"
    metrics_path = out_dir / "metrics.json"
    meta_path = out_dir / "meta.json"

    portfolio_df.to_parquet(portfolio_path, index=False)
    trades_df.to_parquet(trades_path, index=False)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    meta_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": cfg.model_dump(),
                "universe": list(universe),
                "n_periods": int(len(prices)),
                "discovery_end": DISCOVERY_END.isoformat(),
                "generated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )

    return BacktestResult(
        run_id=run_id,
        config=cfg,
        strategies=summaries,
        artifacts_dir=str(out_dir),
        portfolio_path=str(portfolio_path),
        trades_path=str(trades_path),
        metrics_path=str(metrics_path),
        meta_path=str(meta_path),
    )


def list_runs(artifacts_root: Path | None = None) -> list[Path]:
    base = artifacts_root or (settings.reports_dir / "backtest" / "runs")
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])
