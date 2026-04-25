"""Forecast schema contract tests."""

from __future__ import annotations

import pandas as pd
import pytest

from autosignalx.eval.contracts import FORECAST_COLUMNS_REQUIRED, assert_forecast_schema


def _valid_forecasts(n: int = 5) -> pd.DataFrame:
    origin = pd.Timestamp("2024-01-01")
    target = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "timestamp": target,
            "asset": "SPY",
            "forecast_origin": origin,
            "horizon": (target - origin).days.astype(int),
            "method": "naive",
            "prediction": 100.0,
            "origin_value": 100.0,
            "target": 100.5,
        }
    )


def test_valid_forecasts_pass() -> None:
    assert_forecast_schema(_valid_forecasts())


def test_empty_dataframe_passes() -> None:
    df = pd.DataFrame(columns=list(FORECAST_COLUMNS_REQUIRED))
    assert_forecast_schema(df)


def test_missing_column_raises() -> None:
    df = _valid_forecasts().drop(columns=["target"])
    with pytest.raises(ValueError, match="missing columns"):
        assert_forecast_schema(df)


def test_leakage_when_origin_equal_to_timestamp_raises() -> None:
    df = _valid_forecasts()
    df.loc[0, "forecast_origin"] = df.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="Leakage"):
        assert_forecast_schema(df)


def test_leakage_when_origin_after_timestamp_raises() -> None:
    df = _valid_forecasts()
    df.loc[0, "forecast_origin"] = df.loc[0, "timestamp"] + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Leakage"):
        assert_forecast_schema(df)


def test_negative_horizon_raises() -> None:
    df = _valid_forecasts()
    df.loc[0, "horizon"] = -1
    with pytest.raises(ValueError, match="Negative horizon"):
        assert_forecast_schema(df)
