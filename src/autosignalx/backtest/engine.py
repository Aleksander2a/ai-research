"""Vectorized portfolio engine.

Trade-timing convention: weights set at the close of bar ``t`` apply to
the return earned from close ``t`` to close ``t+1``. We enforce this by
shifting the weights matrix forward by one bar before multiplying with
the returns matrix. This is the single line of code that prevents
look-ahead bias in the simulation; do not change it without updating
``tests/test_no_backtest_leakage.py`` accordingly.

Costs: a one-way bps charge applied to per-asset weight changes
(``|w_t - w_{t-1}|``). Realistic enough for an MVP; richer slippage
models are out of scope for Phase 1.
"""

from __future__ import annotations

import pandas as pd


def run_engine(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Run a single strategy through the engine.

    Args:
        weights: target weights, indexed by timestamp, columns = assets.
            Values are fractions of NAV; rows do not need to sum to 1
            (cash is implicit when sum < 1; leverage when > 1).
        prices: adjusted close prices, indexed by timestamp, columns = assets.
            Must align (or be a superset of) weights' index/columns.
        cost_bps: one-way transaction cost in basis points applied per
            unit of |Δweight| at each rebalance.

    Returns:
        Dict with keys:
            - ``returns``: Series of net daily portfolio returns
            - ``equity``: Series of cumulative equity (starts at 1.0)
            - ``turnover``: Series of per-bar turnover (sum |Δw|)
            - ``cost``: Series of per-bar cost drag (return units)
            - ``weights``: aligned weights actually used (fwd-filled)
    """
    if weights.empty or prices.empty:
        raise ValueError("weights and prices must be non-empty")

    common_index = weights.index.intersection(prices.index)
    if len(common_index) < 2:
        raise ValueError("weights and prices must share at least 2 timestamps")
    common_assets = weights.columns.intersection(prices.columns)
    if len(common_assets) == 0:
        raise ValueError("no overlapping assets between weights and prices")

    w = weights.loc[common_index, common_assets].astype(float).fillna(0.0)
    p = prices.loc[common_index, common_assets].astype(float)

    asset_returns = p.pct_change().fillna(0.0)
    cost_rate = cost_bps / 10_000.0

    # Trade-timing invariant: shift weights forward by one bar so the
    # weight chosen at close(t) earns the close(t)->close(t+1) return.
    held_weights = w.shift(1).fillna(0.0)
    gross_return = (held_weights * asset_returns).sum(axis=1)

    delta = w.subtract(w.shift(1).fillna(0.0))
    turnover = delta.abs().sum(axis=1)
    cost = turnover * cost_rate

    net_return = gross_return - cost
    equity = (1.0 + net_return).cumprod()

    return {
        "returns": net_return.rename("return"),
        "equity": equity.rename("equity"),
        "turnover": turnover.rename("turnover"),
        "cost": cost.rename("cost"),
        "weights": w,
    }


def assemble_trades(weights: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Long-format ledger of weight changes (one row per non-zero Δw)."""
    delta = weights.subtract(weights.shift(1).fillna(0.0))
    long = delta.stack().rename("dweight").reset_index()
    long.columns = ["timestamp", "asset", "dweight"]
    long = long[long["dweight"] != 0.0].copy()
    long["cost"] = long["dweight"].abs() * (cost_bps / 10_000.0)
    return long
