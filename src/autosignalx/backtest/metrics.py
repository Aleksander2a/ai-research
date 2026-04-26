"""Performance metrics for backtest results.

All inputs are pandas Series indexed by timestamp. ``periods_per_year``
is 252 for daily trading data; the runner can override for other
frequencies. Everything is computed from net (post-cost) returns.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    """Compounded return from first to last bar."""
    if equity.empty:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised compound growth rate."""
    if equity.empty or len(equity) < 2:
        return float("nan")
    n_periods = len(equity) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return float("nan")
    growth = float(equity.iloc[-1] / equity.iloc[0])
    if growth <= 0:
        return float("nan")
    return growth ** (1.0 / years) - 1.0


def annual_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return float("nan")
    return float(returns.std(ddof=0) * math.sqrt(periods_per_year))


def sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio of net returns (zero risk-free by default)."""
    if returns.empty:
        return float("nan")
    excess = returns - rf / periods_per_year
    sd = excess.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Sortino ratio: mean / downside-deviation, annualised."""
    if returns.empty:
        return float("nan")
    downside = returns.clip(upper=0.0)
    dd = math.sqrt((downside**2).mean())
    if dd == 0 or math.isnan(dd):
        return float("nan")
    return float(returns.mean() / dd * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough drawdown as a non-positive fraction."""
    if equity.empty:
        return float("nan")
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def calmar(equity: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR / |max drawdown|."""
    mdd = max_drawdown(equity)
    if mdd == 0 or math.isnan(mdd):
        return float("nan")
    return float(cagr(equity, periods_per_year) / abs(mdd))


def hit_rate(returns: pd.Series) -> float:
    """Fraction of bars with strictly positive return."""
    if returns.empty:
        return float("nan")
    return float((returns > 0).sum() / len(returns))


def avg_turnover(turnover: pd.Series) -> float:
    """Mean per-bar turnover (sum of |Δw|)."""
    if turnover.empty:
        return float("nan")
    return float(turnover.mean())


def cost_drag(cost: pd.Series) -> float:
    """Total cost as a fraction of NAV over the run."""
    if cost.empty:
        return float("nan")
    return float(cost.sum())


def compute_all(
    equity: pd.Series,
    returns: pd.Series,
    turnover: pd.Series,
    cost: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    return {
        "n_periods": int(len(returns)),
        "total_return": total_return(equity),
        "cagr": cagr(equity, periods_per_year),
        "annual_vol": annual_vol(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year),
        "sortino": sortino(returns, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity, periods_per_year),
        "hit_rate": hit_rate(returns),
        "avg_turnover": avg_turnover(turnover),
        "cost_drag": cost_drag(cost),
    }


def _sanitize(value: float) -> float:
    """Replace NaN/Inf with 0.0 for JSON serialization safety."""
    if value is None or math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def sanitize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {k: _sanitize(v) if isinstance(v, float | np.floating) else v
            for k, v in metrics.items()}


def compute_per_regime(
    returns: pd.Series,
    turnover: pd.Series,
    cost: pd.Series,
    regimes: pd.Series,
    periods_per_year: int = 252,
) -> dict[int, dict[str, float]]:
    """Recompute the metric block on each regime's subset of bars.

    The regime-conditional equity curve is rebuilt from the filtered
    returns, so the per-regime Sharpe/Calmar reflect only the bars in
    that regime (compounding contiguously). Bars where the regime label
    is missing are skipped.
    """
    aligned = regimes.reindex(returns.index).ffill()
    out: dict[int, dict[str, float]] = {}
    for r in sorted(aligned.dropna().unique()):
        mask = aligned == r
        sub_ret = returns[mask]
        if sub_ret.empty:
            continue
        sub_turn = turnover[mask]
        sub_cost = cost[mask]
        sub_eq = (1.0 + sub_ret).cumprod()
        block = compute_all(sub_eq, sub_ret, sub_turn, sub_cost, periods_per_year)
        out[int(r)] = sanitize_metrics(block)
    return out
