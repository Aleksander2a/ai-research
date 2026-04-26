"""Block-bootstrap significance tests on Sharpe-difference between
strategies.

Pairs the bootstrap samples (same indices for both return series) so
the correlation structure between the two strategies is preserved -- a
naive independent resample would understate the CI when both
strategies are exposed to overlapping market drivers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _sharpe_pure(values: np.ndarray, periods_per_year: int = 252) -> float:
    if values.size == 0:
        return float("nan")
    mu = values.mean()
    sd = values.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        return float("nan")
    return float(mu / sd * math.sqrt(periods_per_year))


def bootstrap_sharpe_diff(
    returns_a: pd.Series,
    returns_b: pd.Series,
    n_bootstrap: int = 5000,
    block_size: int = 5,
    seed: int = 42,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Paired moving-block bootstrap of Sharpe(A) - Sharpe(B).

    Args:
        returns_a, returns_b: net per-bar return series. Aligned on the
            inner intersection of their indices before resampling.
        n_bootstrap: number of bootstrap iterations.
        block_size: contiguous-block length (bars). 5 is reasonable for
            daily data with weekly autocorrelation.
        seed: PRNG seed.
        periods_per_year: 252 for daily, 12 for monthly, etc.

    Returns:
        dict with keys
          - sharpe_a, sharpe_b
          - sharpe_diff (observed)
          - ci_low, ci_high (2.5/97.5 percentiles)
          - p_value (two-sided, fraction of resamples on the wrong side
            of zero)
          - n_bootstrap, block_size, n_periods
    """
    df = pd.concat({"a": returns_a, "b": returns_b}, axis=1).dropna()
    if df.empty:
        return _nan_result(n_bootstrap, block_size, 0)
    a = df["a"].to_numpy()
    b = df["b"].to_numpy()
    n = len(a)
    if n < block_size * 2:
        return _nan_result(n_bootstrap, block_size, n)

    rng = np.random.default_rng(seed)
    n_blocks = (n + block_size - 1) // block_size
    max_start = n - block_size + 1

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, max_start, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        sa = a[idx]
        sb = b[idx]
        diffs[i] = _sharpe_pure(sa, periods_per_year) - _sharpe_pure(sb, periods_per_year)

    observed_a = _sharpe_pure(a, periods_per_year)
    observed_b = _sharpe_pure(b, periods_per_year)
    observed_diff = observed_a - observed_b

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # Two-sided p-value: probability of observing a diff at least as
    # extreme as zero under the bootstrap distribution.
    p_low = float((diffs <= 0).mean())
    p_high = float((diffs >= 0).mean())
    p_value = float(min(2.0 * min(p_low, p_high), 1.0))

    return {
        "sharpe_a": observed_a,
        "sharpe_b": observed_b,
        "sharpe_diff": float(observed_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
        "n_periods": n,
    }


def _nan_result(n_bootstrap: int, block_size: int, n: int) -> dict[str, float]:
    return {
        "sharpe_a": float("nan"),
        "sharpe_b": float("nan"),
        "sharpe_diff": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_value": float("nan"),
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
        "n_periods": n,
    }


def is_significant(result: dict[str, float], alpha: float = 0.05) -> bool:  # noqa: ARG001
    """Significant if both CI ends sit on the same side of zero."""
    lo = result.get("ci_low", float("nan"))
    hi = result.get("ci_high", float("nan"))
    if math.isnan(lo) or math.isnan(hi):
        return False
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
