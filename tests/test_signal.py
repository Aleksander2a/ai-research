"""Signal layer tests: feature engineering correctness, ranking contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autosignalx.signal import features


def test_compute_rsi_in_zero_to_hundred() -> None:
    rng = np.random.default_rng(0)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)))
    rsi = features.compute_rsi(prices, window=14)
    valid = rsi.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_compute_macd_signal_finite() -> None:
    rng = np.random.default_rng(0)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)))
    macd_sig = features.compute_macd_signal(prices)
    assert macd_sig.notna().any()
    assert np.isfinite(macd_sig.dropna()).all()


def test_build_features_target_returns_binary() -> None:
    rng = np.random.default_rng(0)
    n = 300
    ts = pd.date_range("2023-01-01", periods=n)
    log_ret = rng.normal(0.0003, 0.01, n)
    prices = np.exp(np.cumsum(log_ret) + np.log(100.0))
    ohlcv = pd.DataFrame(
        {
            "timestamp": ts,
            "asset": "TEST",
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "adj_close": prices,
            "volume": 1.0,
        }
    )
    macro = pd.DataFrame({"VIX": rng.normal(20, 5, n)}, index=ts)
    out = features.build_features_target(ohlcv, macro, horizon_days=21)
    assert "target_direction" in out.columns
    assert out["target_direction"].isin([0, 1]).all()
    # We dropna -- output should be shorter than input
    assert 0 < len(out) < n


def test_feature_columns_excludes_ids_and_target() -> None:
    rng = np.random.default_rng(0)
    n = 100
    ts = pd.date_range("2024-01-01", periods=n)
    prices = np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ohlcv = pd.DataFrame(
        {
            "timestamp": ts,
            "asset": "X",
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "adj_close": prices,
            "volume": 1.0,
        }
    )
    macro = pd.DataFrame()
    df = features.build_features_target(ohlcv, macro, horizon_days=21)
    cols = features.feature_columns(df)
    for excluded in (
        "timestamp",
        "asset",
        "target_direction",
        "future_close",
        "returns_1d",
        "adj_close",
    ):
        assert excluded not in cols
    # Should include at least the technical features
    assert "rolling_mean_5" in cols
    assert "rsi_14" in cols
