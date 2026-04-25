"""Metric correctness tests against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from autosignalx.eval import metrics


def test_mae_matches_hand_computation() -> None:
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([1.5, 2.5, 2.5, 5.0])
    assert metrics.mae(pred, target) == pytest.approx(0.625)


def test_mape_matches_hand_computation() -> None:
    pred = np.array([100.0, 200.0])
    target = np.array([110.0, 180.0])
    expected = (10 / 110 + 20 / 180) / 2
    assert metrics.mape(pred, target) == pytest.approx(expected)


def test_mape_masks_zero_targets() -> None:
    pred = np.array([1.0, 2.0])
    target = np.array([0.0, 4.0])
    assert metrics.mape(pred, target) == pytest.approx(0.5)


def test_directional_accuracy_full_match() -> None:
    pred = np.array([105.0, 95.0, 100.0])
    target = np.array([110.0, 90.0, 100.0])
    origin = np.array([100.0, 100.0, 100.0])
    assert metrics.directional_accuracy(pred, target, origin) == pytest.approx(1.0)


def test_directional_accuracy_full_miss() -> None:
    pred = np.array([105.0, 105.0])
    target = np.array([95.0, 95.0])
    origin = np.array([100.0, 100.0])
    assert metrics.directional_accuracy(pred, target, origin) == pytest.approx(0.0)


def test_skill_score_zero_when_equal() -> None:
    assert metrics.skill_score(1.0, 1.0) == pytest.approx(0.0)


def test_skill_score_positive_when_better() -> None:
    assert metrics.skill_score(0.5, 1.0) == pytest.approx(0.5)


def test_skill_score_negative_when_worse() -> None:
    assert metrics.skill_score(2.0, 1.0) == pytest.approx(-1.0)


def test_skill_score_nan_when_baseline_zero() -> None:
    assert math.isnan(metrics.skill_score(0.5, 0.0))


def test_metrics_handle_nan_input() -> None:
    pred = np.array([1.0, np.nan, 3.0])
    target = np.array([1.5, 2.5, 2.5])
    assert metrics.mae(pred, target) == pytest.approx((0.5 + 0.5) / 2)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.mae(np.array([1.0, 2.0]), np.array([1.0]))


# CRPS tests


def test_crps_perfect_forecast_is_zero() -> None:
    target = np.array([10.0, 20.0, 30.0])
    quantiles = np.column_stack([target, target, target])  # all q's exactly = target
    levels = np.array([0.1, 0.5, 0.9])
    assert metrics.crps_from_quantiles(quantiles, levels, target) == pytest.approx(0.0)


def test_crps_constant_offset_positive() -> None:
    target = np.array([10.0, 20.0, 30.0])
    # quantiles all 1 below target -- pinball loss is symmetric in q for this case
    quantiles = np.column_stack([target - 1, target - 1, target - 1])
    levels = np.array([0.1, 0.5, 0.9])
    crps = metrics.crps_from_quantiles(quantiles, levels, target)
    # 2 * mean over q of q*(target-q_y) since target > q_y everywhere
    # = 2 * mean(0.1*1, 0.5*1, 0.9*1) = 2 * 0.5 = 1.0
    assert crps == pytest.approx(1.0)


def test_crps_handles_nans_by_masking() -> None:
    target = np.array([10.0, 20.0, 30.0])
    quantiles = np.array([
        [9.0, 10.0, 11.0],
        [np.nan, 20.0, 21.0],
        [29.0, 30.0, 31.0],
    ])
    levels = np.array([0.1, 0.5, 0.9])
    # The NaN row is masked; result should equal CRPS on rows 0 and 2 only
    crps = metrics.crps_from_quantiles(quantiles, levels, target)
    assert np.isfinite(crps)


def test_crps_2d_shape_required() -> None:
    target = np.array([10.0, 20.0])
    with pytest.raises(ValueError, match="2-D"):
        metrics.crps_from_quantiles(np.array([10.0, 20.0]), np.array([0.5]), target)


def test_crps_all_nan_returns_nan() -> None:
    target = np.array([np.nan, np.nan])
    quantiles = np.full((2, 3), np.nan)
    levels = np.array([0.1, 0.5, 0.9])
    assert math.isnan(metrics.crps_from_quantiles(quantiles, levels, target))
