"""Pydantic models for backtest configuration and result artifacts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Hard temporal floor for any backtest in this project. The discovery
# pipeline (Chronos-2, regime model, signal weights, agent findings) was
# fit on data with timestamp <= 2020-12-31, so the backtest window must
# start strictly after that to avoid look-ahead.
DISCOVERY_END = datetime(2020, 12, 31)


class BacktestConfig(BaseModel):
    """User-facing knobs for a backtest run."""

    strategies: list[str] = Field(
        default_factory=lambda: ["BuyAndHoldSPY", "EqualWeightUniverse"],
        description="Strategy class names to run.",
    )
    start_date: str = Field(
        default="2021-01-01",
        description="Backtest window start (must be after the discovery window).",
    )
    end_date: str = Field(
        default="2025-12-31", description="Backtest window end."
    )
    cost_bps: float = Field(
        default=5.0, ge=0.0, description="One-way transaction cost in basis points."
    )
    benchmark: str = Field(
        default="SPY", description="Benchmark ticker for relative metrics."
    )
    seed: int = Field(default=42)
    universe: list[str] | None = Field(
        default=None,
        description="Asset list. None = use the project default 8-ETF universe.",
    )


class StrategyResult(BaseModel):
    """Per-strategy summary metrics. Series live in parquet, not here."""

    name: str
    n_periods: int
    total_return: float
    cagr: float
    annual_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    avg_turnover: float
    cost_drag: float


class BacktestResult(BaseModel):
    """Top-level run artifact. Paths point at parquet/JSON files on disk."""

    run_id: str
    config: BacktestConfig
    strategies: list[StrategyResult]
    artifacts_dir: str
    portfolio_path: str
    trades_path: str
    metrics_path: str
    meta_path: str
