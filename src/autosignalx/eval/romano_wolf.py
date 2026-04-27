"""Phase 8 -- Romano-Wolf step-down (multiple-testing under correlation).

Romano-Wolf (2005) is more powerful than BH-FDR when test statistics
are correlated. It iteratively bootstraps the joint distribution of
the (sorted) maxima of test statistics under the null and rejects
hypotheses whose statistics exceed the bootstrap critical value.

We implement the studentized version: each hypothesis has a per-bar
loss-difference series d_i (method - baseline). The studentized
statistic is t_i = mean(d_i) / sqrt(var(d_i) / n). The bootstrap
distribution of max_i |t_i| under the null is computed by resampling
the full block-bootstrap of d, recentering each series, and tracking
the maximum across hypotheses on each bootstrap draw.

Returns per-hypothesis adjusted p-values that strongly control FWER
in finite samples for arbitrary dependence structure -- the gold
standard for joint testing in the trading-rule literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RomanoWolfResult:
    p_values: list[float]       # raw t-test p-values per hypothesis
    p_adjusted: list[float]     # RW step-down adjusted p-values
    survives: list[bool]        # adjusted p < alpha
    alpha: float
    n_bootstrap: int
    block_size: int


def _bootstrap_max_t(
    d_matrix: np.ndarray,
    n_bootstrap: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    """Block-bootstrap max-t distribution under the null (E[d_i]=0).

    d_matrix: (T, k) per-bar loss differences for k hypotheses."""
    T, k = d_matrix.shape
    rng = np.random.default_rng(seed)
    centred = d_matrix - d_matrix.mean(axis=0, keepdims=True)
    n_blocks = max(1, T // block_size)
    max_starts = T - block_size + 1
    max_t_draws = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        starts = rng.integers(0, max_starts, size=n_blocks)
        sample_idx = np.concatenate(
            [np.arange(s, s + block_size) for s in starts]
        )
        sample = centred[sample_idx]
        means = sample.mean(axis=0)
        sds = sample.std(axis=0, ddof=1)
        sds = np.where(sds > 0, sds, np.inf)
        t = means / (sds / np.sqrt(len(sample)))
        max_t_draws[b] = float(np.max(np.abs(t)))
    return max_t_draws


def romano_wolf(
    diff_matrix: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    block_size: int = 20,
    seed: int = 42,
) -> RomanoWolfResult:
    """Stepdown adjusted p-values across k hypotheses.

    Args:
        diff_matrix: (T, k) per-bar loss differences. Positive value at
            (t, i) = method i predicts better than its baseline at bar t.
        alpha: target FWER level.
        n_bootstrap: bootstrap iterations for the max-t null.
        block_size: moving-block-bootstrap block length (handles
            autocorrelation from h-step-ahead overlapping forecasts).
    """
    D = np.asarray(diff_matrix, dtype=float)
    if D.ndim == 1:
        D = D.reshape(-1, 1)
    T, k = D.shape
    if block_size + 1 > T or k == 0:
        return RomanoWolfResult(
            p_values=[float("nan")] * k,
            p_adjusted=[float("nan")] * k,
            survives=[False] * k,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
        )

    means = D.mean(axis=0)
    sds = D.std(axis=0, ddof=1)
    sds_safe = np.where(sds > 0, sds, np.inf)
    t_stats = means / (sds_safe / np.sqrt(T))
    abs_t = np.abs(t_stats)

    # Use a bootstrap distribution of max-|t| under H0 (centred D)
    max_t_dist = _bootstrap_max_t(D, n_bootstrap, block_size, seed)

    # Step-down: order hypotheses by |t| descending; for each, recompute
    # max-|t| over the still-active set
    order = np.argsort(-abs_t)
    adjusted = np.full(k, np.nan, dtype=float)
    active = list(range(k))
    for _step, idx in enumerate(order):
        if idx not in active:
            continue
        # Recompute max-|t| under null over the active hypotheses only
        # by re-bootstrapping the relevant columns
        if len(active) == k:
            # For step 1, the full max-t distribution applies directly
            p_adj = float(np.mean(max_t_dist >= abs_t[idx]))
        else:
            sub_max = _bootstrap_max_t(D[:, active], n_bootstrap, block_size, seed + _step)
            p_adj = float(np.mean(sub_max >= abs_t[idx]))
        # Enforce monotonicity: adjusted p must be >= previous step's
        prev = np.nanmax(adjusted) if np.any(np.isfinite(adjusted)) else 0.0
        adjusted[idx] = max(p_adj, prev)
        active.remove(idx)

    raw_p = 2.0 * (1.0 - _safe_cdf_norm(abs_t))
    survives = (adjusted <= alpha) & np.isfinite(adjusted)
    return RomanoWolfResult(
        p_values=[float(p) for p in raw_p],
        p_adjusted=[float(p) for p in adjusted],
        survives=[bool(s) for s in survives],
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
    )


def _safe_cdf_norm(x: np.ndarray) -> np.ndarray:
    from scipy import stats

    return stats.norm.cdf(x)
