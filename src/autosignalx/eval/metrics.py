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


def _align(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    mask = np.isfinite(pred) & np.isfinite(target)
    return pred[mask], target[mask]
