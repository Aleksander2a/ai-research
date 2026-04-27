"""Phase 8 -- Probability of Backtest Overfitting (Bailey & Lopez de Prado).

Given a matrix M of N strategies evaluated over T time periods, PBO
asks: across all 2^S combinatorially symmetric splits of the time
series into in-sample (IS) and out-of-sample (OOS), what is the
probability that the strategy ranked best IS has below-median rank
OOS?

If PBO ≈ 0, the IS-best strategy is robust. PBO ≈ 0.5 means the IS
ranking has no predictive power for OOS performance -- which is what
the literature finds for over-tuned trading rules. The metric directly
captures the "selection bias from search-space size" that the AutoSignal-X
agent's promotion gate doesn't see.

We compute it on the per-strategy per-period skill matrix produced by
the harness (rows = periods, columns = strategies). Periods can be
walk-forward windows, regimes, or assets -- any axis along which we
can split IS vs OOS.

Reference: Bailey, Borwein, Lopez de Prado, Zhu (2014), "The
Probability of Backtest Overfitting", Journal of Computational Finance.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_combinations: int
    n_strategies: int
    n_periods: int
    is_oos_logits: list[float]  # log(rank_oos / (1 - rank_oos)) for diagnostic plots


def probability_of_backtest_overfitting(
    metric_matrix: np.ndarray,
    s: int = 16,
) -> PBOResult:
    """Compute PBO on a (n_periods x n_strategies) metric matrix.

    Args:
        metric_matrix: rows = periods, columns = strategies; higher is
            better (e.g. Sharpe, skill_vs_naive). NaN rows are dropped.
        s: even integer; number of equal-size sub-periods. The
            combinatorial procedure picks every S/2-subset of those S
            sub-periods as IS, the complement as OOS. With S=16 there
            are C(16,8)=12,870 splits -- the canonical reference value.

    Returns the PBO and per-split logits."""
    M = np.asarray(metric_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"metric_matrix must be 2D, got shape {M.shape}")
    M = M[~np.isnan(M).any(axis=1)]
    n_periods, n_strategies = M.shape
    if n_strategies < 2:
        return PBOResult(pbo=float("nan"), n_combinations=0, n_strategies=n_strategies,
                          n_periods=n_periods, is_oos_logits=[])
    # Round s to an even number that divides cleanly into the periods we have
    s = max(2, min(s, n_periods))
    if s % 2 != 0:
        s -= 1
    if s < 2:
        return PBOResult(pbo=float("nan"), n_combinations=0, n_strategies=n_strategies,
                          n_periods=n_periods, is_oos_logits=[])

    period_size = n_periods // s
    if period_size == 0:
        return PBOResult(pbo=float("nan"), n_combinations=0, n_strategies=n_strategies,
                          n_periods=n_periods, is_oos_logits=[])
    sub_period_idx = [
        list(range(i * period_size, (i + 1) * period_size if i < s - 1 else n_periods))
        for i in range(s)
    ]

    half = s // 2
    splits = list(combinations(range(s), half))
    logits: list[float] = []
    for is_subs in splits:
        is_set = set(is_subs)
        is_idx: list[int] = []
        oos_idx: list[int] = []
        for i, subset in enumerate(sub_period_idx):
            (is_idx if i in is_set else oos_idx).extend(subset)
        if not is_idx or not oos_idx:
            continue
        is_mean = np.nanmean(M[is_idx], axis=0)
        oos_mean = np.nanmean(M[oos_idx], axis=0)
        # Skip splits where everything is NaN
        if np.isnan(is_mean).any() or np.isnan(oos_mean).any():
            continue
        n_star = int(np.argmax(is_mean))
        oos_ranks = pd.Series(oos_mean).rank(method="average")
        # Fractional rank of the IS-best strategy in OOS, in (0, 1)
        rank_oos = float(oos_ranks.iloc[n_star]) / (n_strategies + 1)
        rank_oos = float(np.clip(rank_oos, 1e-6, 1 - 1e-6))
        logits.append(float(np.log(rank_oos / (1 - rank_oos))))

    if not logits:
        return PBOResult(pbo=float("nan"), n_combinations=0, n_strategies=n_strategies,
                          n_periods=n_periods, is_oos_logits=[])

    arr = np.asarray(logits, dtype=float)
    pbo = float(np.mean(arr < 0))  # PBO = P(rank_OOS < 0.5) = P(logit < 0)
    return PBOResult(
        pbo=pbo,
        n_combinations=len(arr),
        n_strategies=n_strategies,
        n_periods=n_periods,
        is_oos_logits=[float(x) for x in arr],
    )


def pbo_from_forecasts(
    forecasts: pd.DataFrame,
    methods: list[str],
    baseline: str = "naive",
    s: int = 16,
) -> PBOResult:
    """Build a periods-x-strategies matrix of skill-vs-baseline and call PBO.

    Periods are walk-forward forecast origins. For each (origin, method),
    skill = 1 - method_mae / baseline_mae over rows sharing the origin."""
    if forecasts.empty:
        return PBOResult(pbo=float("nan"), n_combinations=0, n_strategies=0, n_periods=0, is_oos_logits=[])
    f = forecasts.copy()
    f["forecast_origin"] = pd.to_datetime(f["forecast_origin"])
    f["abs_err"] = (f["prediction"] - f["target"]).abs()
    pivoted = (
        f.groupby(["forecast_origin", "method"], observed=True)["abs_err"]
        .mean()
        .unstack("method")
    )
    if baseline not in pivoted.columns:
        return PBOResult(pbo=float("nan"), n_combinations=0, n_strategies=0,
                          n_periods=0, is_oos_logits=[])
    baseline_col = pivoted[baseline].replace({0: np.nan})
    skills: dict[str, np.ndarray] = {}
    for m in methods:
        if m == baseline or m not in pivoted.columns:
            continue
        skills[m] = (1.0 - pivoted[m] / baseline_col).to_numpy()
    if len(skills) < 2:
        return PBOResult(pbo=float("nan"), n_combinations=0, n_strategies=len(skills),
                          n_periods=len(pivoted), is_oos_logits=[])
    matrix = np.column_stack([skills[m] for m in skills])
    return probability_of_backtest_overfitting(matrix, s=s)
