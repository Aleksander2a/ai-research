"""Backtested simulation layer (Phase 1).

Translates AutoSignal-X's discovered structure (forecasts, regimes,
agent-promoted findings) into simulated trading strategies and evaluates
their out-of-sample performance under strict walk-forward discipline.

Public API:
    from autosignalx.backtest import run_backtest, BacktestConfig

Artifacts: reports/backtest/runs/<run_id>/{portfolio_daily.parquet,
trades.parquet, metrics.json, meta.json}.

Cockpit reader: app.streamlit_app.render_backtest_arena.
"""

from autosignalx.backtest.runner import run_backtest
from autosignalx.backtest.schemas import BacktestConfig, BacktestResult, StrategyResult

__all__ = ["BacktestConfig", "BacktestResult", "StrategyResult", "run_backtest"]
