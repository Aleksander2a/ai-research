"""Forecast metrics.

Point metrics (MAE, MAPE, directional accuracy) operate on aligned
prediction/target arrays. Skill score expresses improvement over a
baseline (positive = better than baseline). CRPS and probabilistic
metrics land in Iter 3 alongside Chronos-2."""

from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error."""
    pred, target = _align(pred, target)
    if len(pred) == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - target)))


def mape(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute percentage error. Targets equal to zero are masked out."""
    pred, target = _align(pred, target)
    mask = target != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((pred[mask] - target[mask]) / target[mask])))


def directional_accuracy(
    pred: np.ndarray, target: np.ndarray, origin_value: np.ndarray
) -> float:
    """Fraction of forecasts whose predicted change-direction matches the realized change.

    A forecast is correct if ``sign(pred - origin_value) == sign(target - origin_value)``,
    counting flat (zero change) as a single category."""
    pred, target = _align(pred, target)
    origin_value = np.asarray(origin_value, dtype=float)
    if len(pred) == 0:
        return float("nan")
    pred_dir = np.sign(pred - origin_value)
    actual_dir = np.sign(target - origin_value)
    return float(np.mean(pred_dir == actual_dir))


def skill_score(method_mae: float, baseline_mae: float) -> float:
    """``1 - method_mae / baseline_mae``. Positive => better than baseline,
    zero => same, negative => worse. Returns NaN if baseline_mae is zero."""
    if baseline_mae == 0 or not np.isfinite(baseline_mae):
        return float("nan")
    return float(1.0 - method_mae / baseline_mae)


def crps_from_quantiles(
    quantile_predictions: np.ndarray,
    quantile_levels: np.ndarray,
    target: np.ndarray,
) -> float:
    """Approximate CRPS from a set of quantile forecasts per sample.

    Uses the relationship CRPS = 2 * mean over q of pinball loss, where
    pinball loss at level q is ``max(q * err, (q - 1) * err)`` with
    ``err = target - quantile``. Lower is better; perfect = 0.

    Args:
        quantile_predictions: shape (n_samples, n_quantiles).
        quantile_levels: shape (n_quantiles,), values in (0, 1).
        target: shape (n_samples,).
    """
    quantile_predictions = np.asarray(quantile_predictions, dtype=float)
    quantile_levels = np.asarray(quantile_levels, dtype=float)
    target = np.asarray(target, dtype=float)
    if quantile_predictions.ndim != 2:
        raise ValueError(
            f"quantile_predictions must be 2-D, got shape {quantile_predictions.shape}"
        )
    if quantile_predictions.shape[0] != target.shape[0]:
        raise ValueError(
            f"Sample-count mismatch: predictions {quantile_predictions.shape[0]} vs "
            f"target {target.shape[0]}"
        )
    if quantile_predictions.shape[1] != quantile_levels.shape[0]:
        raise ValueError(
            f"Quantile-count mismatch: predictions {quantile_predictions.shape[1]} vs "
            f"levels {quantile_levels.shape[0]}"
        )
    # Mask out rows with any NaN
    finite = (
        np.all(np.isfinite(quantile_predictions), axis=1) & np.isfinite(target)
    )
    if not finite.any():
        return float("nan")
    err = target[finite, None] - quantile_predictions[finite]
    pinball = np.maximum(quantile_levels[None, :] * err, (quantile_levels[None, :] - 1) * err)
    return float(2.0 * pinball.mean())


def _align(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    mask = np.isfinite(pred) & np.isfinite(target)
    return pred[mask], target[mask]
