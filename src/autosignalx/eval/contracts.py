"""Forecast DataFrame contract.

Every forecasting method (baselines, Chronos-2, signal-enhanced ensembles)
produces a DataFrame matching this contract. The harness, metrics, and
cockpit all read against it -- this is the seam that lets layers compose."""

from __future__ import annotations

import pandas as pd

FORECAST_COLUMNS_REQUIRED: tuple[str, ...] = (
    "timestamp",        # target time the forecast is for (trading day)
    "asset",            # asset ticker
    "forecast_origin",  # when the forecast was made; must be < timestamp
    "horizon",          # int days from forecast_origin to timestamp
    "method",           # string identifier of the forecasting method
    "prediction",       # point forecast in adj_close units
    "origin_value",     # adj_close at forecast_origin (for directional metrics)
    "target",           # realized adj_close at timestamp
)

FORECAST_COLUMNS_OPTIONAL: tuple[str, ...] = (
    "lower",            # interval lower bound (None for point-only)
    "upper",            # interval upper bound (None for point-only)
    "regime_id",        # regime label (filled in Iter 4+)
)


def assert_forecast_schema(df: pd.DataFrame) -> None:
    """Validate a forecasts DataFrame.

    Required columns are present; for every row, ``forecast_origin < timestamp``
    (no leakage); horizons are non-negative."""
    missing = set(FORECAST_COLUMNS_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Forecast DataFrame missing columns: {sorted(missing)}")
    if df.empty:
        return
    leakage = df["forecast_origin"] >= df["timestamp"]
    if leakage.any():
        n = int(leakage.sum())
        raise ValueError(
            f"Leakage: {n} rows have forecast_origin >= timestamp "
            f"(first offending: {df.loc[leakage, ['method', 'asset', 'forecast_origin', 'timestamp']].iloc[0].to_dict()})"
        )
    if (df["horizon"] < 0).any():
        raise ValueError("Negative horizon found; horizon must be >= 0")
