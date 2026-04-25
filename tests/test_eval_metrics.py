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
