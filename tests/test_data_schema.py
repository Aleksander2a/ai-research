"""Schema tests: synthetic OHLCV / macro DataFrames must satisfy the
contract; deviations are caught at the schema-assertion boundary."""

from __future__ import annotations

import pandas as pd
import pytest

from autosignalx.data.schema import assert_macro_schema, assert_ohlcv_schema


def _ohlcv_frame(n: int = 10, asset: str = "SPY") -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "asset": pd.array([asset] * n, dtype="string"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "adj_close": 100.5,
            "volume": 1_000_000.0,
            "returns": 0.005,
        }
    )


def _macro_frame(n: int = 10, signal: str = "^VIX") -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "signal": pd.array([signal] * n, dtype="string"),
            "value": 18.0,
        }
    )


def test_valid_ohlcv_passes() -> None:
    assert_ohlcv_schema(_ohlcv_frame())


def test_missing_column_raises() -> None:
    df = _ohlcv_frame().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing columns"):
        assert_ohlcv_schema(df)


def test_non_monotonic_timestamps_raise() -> None:
    df = _ohlcv_frame()
    df = pd.concat([df.iloc[5:], df.iloc[:5]], ignore_index=True)  # scramble
    with pytest.raises(ValueError, match="monotonic"):
        assert_ohlcv_schema(df)


def test_two_assets_each_monotonic_passes() -> None:
    df = pd.concat(
        [_ohlcv_frame(asset="SPY"), _ohlcv_frame(asset="QQQ")],
        ignore_index=True,
    )
    assert_ohlcv_schema(df)


def test_valid_macro_passes() -> None:
    assert_macro_schema(_macro_frame())


def test_macro_missing_column_raises() -> None:
    df = _macro_frame().drop(columns=["value"])
    with pytest.raises(ValueError, match="missing columns"):
        assert_macro_schema(df)
