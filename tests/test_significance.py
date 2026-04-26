"""Tests for DM test, block bootstrap CI, and the promotion gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autosignalx.eval import significance


def test_dm_identical_losses_zero_stat() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    stat, p = significance.dm_test(a, a.copy())
    assert stat == pytest.approx(0.0, abs=1e-9)


def test_dm_clearly_different_losses_low_p() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(loc=2.0, scale=0.5, size=300)
    b = rng.normal(loc=1.0, scale=0.5, size=300)
    _, p = significance.dm_test(a, b)
    assert p < 0.01


def test_dm_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        significance.dm_test(np.array([1.0, 2.0]), np.array([1.0]))


def test_block_bootstrap_ci_brackets_mean() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(loc=5.0, scale=1.0, size=500)
    lo, hi = significance.block_bootstrap_ci(values, n_bootstrap=400, block_size=10, seed=0)
    assert lo < 5.0 < hi


def test_block_bootstrap_ci_too_few_returns_mean() -> None:
    lo, hi = significance.block_bootstrap_ci(np.array([1.0, 2.0]), block_size=20)
    assert lo == hi == pytest.approx(1.5)


def _synthetic_forecasts(n_per_method: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keys = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_per_method, freq="B"),
            "asset": "SPY",
            "forecast_origin": pd.date_range("2023-12-01", periods=n_per_method, freq="B"),
        }
    )
    target = 100 + np.cumsum(rng.normal(0, 1, n_per_method))
    naive_pred = np.roll(target, 1)
    naive_pred[0] = target[0]
    method_pred = target + rng.normal(0, 0.1, n_per_method)  # nearly perfect
    rows = []
    for method, pred in [("naive", naive_pred), ("hot", method_pred)]:
        sub = keys.copy()
        sub["method"] = method
        sub["prediction"] = pred
        sub["target"] = target
        sub["origin_value"] = naive_pred
        sub["horizon"] = 1
        sub["lower"] = pd.NA
        sub["upper"] = pd.NA
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def test_is_promotable_clearly_better_passes() -> None:
    df = _synthetic_forecasts()
    promotable, evidence = significance.is_promotable(df, method="hot", baseline_method="naive")
    assert promotable is True
    assert evidence["p_value"] is not None and evidence["p_value"] < 0.05
    assert evidence["skill_vs_baseline"] > 0


def test_is_promotable_same_method_not_promotable() -> None:
    df = _synthetic_forecasts()
    promotable, evidence = significance.is_promotable(df, method="naive", baseline_method="naive")
    # method == baseline => skill ~= 0, not promotable
    assert promotable is False


def test_is_promotable_missing_method() -> None:
    df = _synthetic_forecasts()
    promotable, evidence = significance.is_promotable(df, method="ghost")
    assert promotable is False
    assert "not in frame" in evidence["reason"]


def test_is_promotable_insufficient_samples() -> None:
    df = _synthetic_forecasts(n_per_method=10)
    promotable, evidence = significance.is_promotable(df, method="hot", min_samples=50)
    assert promotable is False
    assert evidence["reason"] == "insufficient_samples"
