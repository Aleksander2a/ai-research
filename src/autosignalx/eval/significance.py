"""Statistical significance for forecast comparisons.

The Diebold-Mariano test asks whether two forecast methods have
significantly different prediction accuracy. Block-bootstrap CIs
quantify the size of the loss difference under serial correlation
typical of financial time series.

Together they form the **promotion gate**: a hypothesis becomes a
'finding' only if its method beats the baseline with DM p < threshold
AND a positive bootstrap CI on the loss difference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _newey_west_variance(d: np.ndarray, h: int) -> float:
    """Newey-West HAC variance estimator for h-step-ahead forecast losses.

    Required because forecast errors at horizon h are auto-correlated
    up to lag h-1 (overlap-induced)."""
    n = len(d)
    d_centered = d - d.mean()
    gamma_0 = float(np.dot(d_centered, d_centered) / n)
    var = gamma_0
    for k in range(1, h):
        gamma_k = float(np.dot(d_centered[:-k], d_centered[k:]) / n)
        var += 2.0 * (1.0 - k / h) * gamma_k
    return max(var, 1e-12)


def dm_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
) -> tuple[float, float]:
    """Diebold-Mariano test on aligned per-observation losses.

    Returns (dm_statistic, p_value). p < 0.05 rejects H0 that the two
    methods have equal expected loss; the sign of dm_statistic indicates
    which is better (positive => method A is worse, B is better)."""
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    if loss_a.shape != loss_b.shape:
        raise ValueError(f"Shape mismatch: {loss_a.shape} vs {loss_b.shape}")
    mask = np.isfinite(loss_a) & np.isfinite(loss_b)
    a = loss_a[mask]
    b = loss_b[mask]
    n = len(a)
    if n < 5:
        return float("nan"), float("nan")
    d = a - b
    var_d = _newey_west_variance(d, max(horizon, 1))
    dm_stat = float(np.mean(d) / np.sqrt(var_d / n))
    p = float(2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=max(n - 1, 1))))
    return dm_stat, p


def block_bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int = 20,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean of a correlated time series.

    Returns (low, high) quantiles of the bootstrap distribution at the
    requested confidence level."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < block_size + 1:
        m = float(np.mean(values)) if n > 0 else float("nan")
        return m, m
    rng = np.random.default_rng(seed)
    n_blocks = max(1, n // block_size)
    means = np.empty(n_bootstrap, dtype=float)
    max_start = n - block_size + 1
    for i in range(n_bootstrap):
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([values[s : s + block_size] for s in starts])
        means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def is_promotable(
    forecasts: pd.DataFrame,
    method: str,
    baseline_method: str = "naive",
    p_threshold: float = 0.05,
    min_samples: int = 30,
    horizon: int = 21,
) -> tuple[bool, dict[str, Any]]:
    """Promotion gate: does ``method`` significantly beat ``baseline_method``?

    Aligns per-row predictions on ``(timestamp, asset, forecast_origin)``,
    runs DM on the absolute-error losses, computes a bootstrap CI on the
    loss difference, and returns ``(promotable, evidence_dict)``."""
    if forecasts.empty:
        return False, {"reason": "empty"}
    methods_in_frame = set(forecasts["method"].unique())
    if method not in methods_in_frame:
        return False, {"reason": f"method '{method}' not in frame"}
    if baseline_method not in methods_in_frame:
        return False, {"reason": f"baseline '{baseline_method}' not in frame"}

    keys = ["timestamp", "asset", "forecast_origin"]
    a = forecasts[forecasts["method"] == method][[*keys, "prediction", "target"]]
    b = forecasts[forecasts["method"] == baseline_method][[*keys, "prediction"]]
    merged = a.merge(b, on=keys, suffixes=("_method", "_baseline"))
    if len(merged) < min_samples:
        return False, {"reason": "insufficient_samples", "n": int(len(merged))}

    loss_method = (merged["prediction_method"] - merged["target"]).abs().to_numpy()
    loss_baseline = (merged["prediction_baseline"] - merged["target"]).abs().to_numpy()
    dm_stat, p_value = dm_test(loss_method, loss_baseline, horizon=horizon)
    method_mae = float(np.mean(loss_method))
    baseline_mae = float(np.mean(loss_baseline))
    skill = float(1.0 - method_mae / baseline_mae) if baseline_mae > 0 else float("nan")
    diff = loss_baseline - loss_method  # positive = method is better
    ci_low, ci_hi = block_bootstrap_ci(diff)

    promotable = bool(
        np.isfinite(p_value)
        and p_value < p_threshold
        and skill > 0
        and ci_low > 0  # bootstrap CI strictly above zero (method consistently better)
    )

    return promotable, {
        "n": int(len(merged)),
        "method": method,
        "baseline_method": baseline_method,
        "method_mae": method_mae,
        "baseline_mae": baseline_mae,
        "skill_vs_baseline": skill,
        "dm_statistic": float(dm_stat) if np.isfinite(dm_stat) else None,
        "p_value": float(p_value) if np.isfinite(p_value) else None,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_hi,
        "p_threshold": p_threshold,
        "horizon": horizon,
    }
