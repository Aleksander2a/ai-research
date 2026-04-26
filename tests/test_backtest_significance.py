"""Tests for the paired moving-block bootstrap of Sharpe-difference."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from autosignalx.backtest import significance


def _series(n: int, mean: float, std: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.Series(rng.normal(mean, std, size=n), index=idx)


def test_bootstrap_diff_zero_when_strategies_identical():
    """Same series in both slots -> diff = 0, CI tight around 0."""
    a = _series(500, 0.001, 0.01, seed=0)
    res = significance.bootstrap_sharpe_diff(a, a.copy(), n_bootstrap=500)
    assert res["sharpe_diff"] == pytest.approx(0.0, abs=1e-9)
    assert res["ci_low"] == pytest.approx(0.0, abs=1e-9)
    assert res["ci_high"] == pytest.approx(0.0, abs=1e-9)


def test_bootstrap_picks_up_clear_difference():
    """A has clearly higher Sharpe than B."""
    a = _series(1000, 0.002, 0.01, seed=1)
    b = _series(1000, -0.001, 0.01, seed=2)
    res = significance.bootstrap_sharpe_diff(a, b, n_bootstrap=500)
    assert res["sharpe_diff"] > 0
    # CI should be entirely above 0 with this much signal.
    assert res["ci_low"] > 0
    assert significance.is_significant(res)


def test_bootstrap_returns_nan_when_too_few_periods():
    a = _series(3, 0.0, 0.01, seed=0)
    b = _series(3, 0.0, 0.01, seed=1)
    res = significance.bootstrap_sharpe_diff(a, b, block_size=5, n_bootstrap=100)
    assert math.isnan(res["sharpe_diff"])


def test_bootstrap_aligns_indexes():
    """Mismatched indices -> intersection used; result still valid."""
    idx_a = pd.bdate_range("2021-01-04", periods=300)
    idx_b = pd.bdate_range("2021-02-01", periods=300)
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 0.01, size=300), index=idx_a)
    b = pd.Series(rng.normal(0, 0.01, size=300), index=idx_b)
    res = significance.bootstrap_sharpe_diff(a, b, n_bootstrap=200)
    # Intersection has ~234 bars; result should be finite.
    assert not math.isnan(res["sharpe_diff"])
    assert res["n_periods"] > 0
    assert res["n_periods"] < 300


def test_is_significant_rejects_ci_straddling_zero():
    res = {"ci_low": -0.5, "ci_high": 0.5}
    assert not significance.is_significant(res)


def test_is_significant_accepts_one_sided_ci():
    res = {"ci_low": 0.1, "ci_high": 0.5}
    assert significance.is_significant(res)
    res = {"ci_low": -0.5, "ci_high": -0.1}
    assert significance.is_significant(res)


def test_is_significant_handles_nan():
    res = {"ci_low": float("nan"), "ci_high": float("nan")}
    assert not significance.is_significant(res)
