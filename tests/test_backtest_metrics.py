"""Metric-level tests using analytically known sequences."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from autosignalx.backtest import metrics


def _series(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2021-01-04", periods=len(values))
    return pd.Series(values, index=idx)


def test_total_return_on_known_equity():
    eq = _series([1.0, 1.05, 1.10, 1.21])
    assert metrics.total_return(eq) == pytest.approx(0.21)


def test_max_drawdown_on_known_equity():
    eq = _series([1.0, 1.5, 0.75, 1.0, 2.0])
    # Peak 1.5 -> 0.75 = -50%
    assert metrics.max_drawdown(eq) == pytest.approx(-0.5)


def test_max_drawdown_zero_when_monotonic():
    eq = _series([1.0, 1.1, 1.2, 1.3])
    assert metrics.max_drawdown(eq) == 0.0


def test_sharpe_constant_returns():
    """Constant returns -> std=0 -> Sharpe is NaN by definition."""
    rets = _series([0.001, 0.001, 0.001, 0.001])
    assert math.isnan(metrics.sharpe(rets))


def test_sharpe_known_value():
    rets = _series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
    expected = rets.mean() / rets.std(ddof=0) * math.sqrt(252)
    assert metrics.sharpe(rets) == pytest.approx(expected, rel=1e-9)


def test_sortino_uses_downside_only():
    """Series with no negative returns has zero downside dev -> NaN."""
    rets = _series([0.01, 0.005, 0.02, 0.001])
    assert math.isnan(metrics.sortino(rets))


def test_hit_rate():
    rets = _series([0.01, -0.005, 0.0, 0.02, -0.01])
    assert metrics.hit_rate(rets) == pytest.approx(2 / 5)


def test_compute_all_returns_expected_keys():
    eq = _series([1.0, 1.01, 1.02, 1.03])
    rets = eq.pct_change().fillna(0.0)
    turnover = _series([0.0, 0.5, 0.0, 0.0])
    cost = turnover * 0.0001
    out = metrics.compute_all(eq, rets, turnover, cost)
    assert {
        "n_periods", "total_return", "cagr", "annual_vol", "sharpe", "sortino",
        "max_drawdown", "calmar", "hit_rate", "avg_turnover", "cost_drag",
    } == set(out)


def test_sanitize_metrics_replaces_nan_inf():
    raw = {"a": float("nan"), "b": float("inf"), "c": 1.5, "d": 3}
    clean = metrics.sanitize_metrics(raw)
    assert clean["a"] == 0.0
    assert clean["b"] == 0.0
    assert clean["c"] == 1.5
    assert clean["d"] == 3


def test_cagr_rejects_negative_growth():
    eq2 = pd.Series([1.0, -0.1], index=pd.bdate_range("2021-01-04", periods=2))
    assert math.isnan(metrics.cagr(eq2))


def test_calmar_zero_drawdown_returns_nan():
    eq = _series([1.0, 1.01, 1.02, 1.03])
    assert math.isnan(metrics.calmar(eq))


def test_compute_per_regime_partitions_returns():
    idx = pd.bdate_range("2021-01-04", periods=10)
    rets = pd.Series([0.01, 0.02, -0.01, 0.0, 0.03, -0.02, 0.0, 0.01, -0.005, 0.0],
                     index=idx)
    turn = pd.Series(0.0, index=idx)
    cost = pd.Series(0.0, index=idx)
    regimes = pd.Series([0] * 5 + [1] * 5, index=idx, name="regime_id")
    out = metrics.compute_per_regime(rets, turn, cost, regimes)
    assert set(out) == {0, 1}
    assert out[0]["n_periods"] == 5
    assert out[1]["n_periods"] == 5


def test_annual_vol_close_to_known():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0, 0.01, size=10000),
                     index=pd.bdate_range("2010-01-04", periods=10000))
    expected = 0.01 * math.sqrt(252)
    assert metrics.annual_vol(rets) == pytest.approx(expected, rel=0.1)
