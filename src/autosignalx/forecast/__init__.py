"""Forecasting layer (L1) -- baselines today, Chronos-2 in Iter 3.

Public API:
- ``baselines.naive_forecast`` / ``seasonal_naive_forecast`` / ``arima_forecast``
- (Iter 3+) ``chronos.chronos_forecast`` -- Chronos-2 multivariate with covariates

All forecasting methods satisfy the eval-harness contract; see
``autosignalx.eval.harness.ForecastFn``."""

from autosignalx.forecast import baselines  # noqa: F401
from autosignalx.forecast.baselines import (  # noqa: F401
    arima_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)
