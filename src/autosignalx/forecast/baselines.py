"""Forecasting baselines: naive, seasonal-naive, ARIMA.

All baselines satisfy the harness contract:
    ``forecast_fn(asset_train: pd.DataFrame, origin: pd.Timestamp,
                  target_dates: list[pd.Timestamp]) -> pd.DataFrame``
returning a frame with ``timestamp`` and ``prediction`` columns aligned to
``target_dates``. Predictions are in adj_close units."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def naive_forecast(
    asset_train: pd.DataFrame,
    origin: pd.Timestamp,  # noqa: ARG001
    target_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Predict the most recent training-set adj_close for every target date.

    The classical naive forecast for an integrated time series. Strong
    baseline for asset prices since they are approximately a random walk."""
    last_value = float(asset_train["adj_close"].iloc[-1])
    return pd.DataFrame({"timestamp": target_dates, "prediction": last_value})


def seasonal_naive_forecast(
    asset_train: pd.DataFrame,
    origin: pd.Timestamp,  # noqa: ARG001
    target_dates: list[pd.Timestamp],
    season_days: int = 252,
) -> pd.DataFrame:
    """For each target date ``ts``, predict the adj_close from approximately
    one calendar year earlier (``ts - 252 calendar days``).

    A simple seasonality baseline. For prices it is rarely competitive but
    is a useful sanity check against models that overfit recent dynamics."""
    history = asset_train.set_index("timestamp")["adj_close"].sort_index()
    fallback = float(history.iloc[-1])
    predictions = []
    for ts in target_dates:
        lookup = ts - pd.Timedelta(days=season_days)
        prior = history[history.index <= lookup]
        predictions.append(float(prior.iloc[-1]) if len(prior) else fallback)
    return pd.DataFrame({"timestamp": target_dates, "prediction": predictions})


def arima_forecast(
    asset_train: pd.DataFrame,
    origin: pd.Timestamp,  # noqa: ARG001
    target_dates: list[pd.Timestamp],
    order: tuple[int, int, int] = (1, 1, 1),
) -> pd.DataFrame:
    """ARIMA on log(adj_close) with default order (1,1,1).

    Fits on the full training-set series, forecasts ``len(target_dates)``
    steps ahead, and exponentiates back to price space. Convergence
    warnings are suppressed -- failures bubble up to the harness, which
    catches them and skips the (window, asset) pair."""
    from statsmodels.tsa.arima.model import ARIMA

    series = asset_train.sort_values("timestamp")["adj_close"].to_numpy(dtype=float)
    log_series = np.log(series)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(log_series, order=order)
        fit = model.fit()
        log_forecast = np.asarray(fit.forecast(steps=len(target_dates)))

    return pd.DataFrame(
        {
            "timestamp": target_dates,
            "prediction": np.exp(log_forecast),
        }
    )
