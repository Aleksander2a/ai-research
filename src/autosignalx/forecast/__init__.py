"""Forecasting layer (L1) -- baselines and Chronos-2.

Public API:
- ``baselines.naive_forecast`` / ``seasonal_naive_forecast`` / ``arima_forecast``
- ``chronos2.chronos2_univariate`` -- single-asset Chronos-2 forecast
- ``chronos2.make_chronos2_multivariate(macro)`` -- closure forecast_fn with
  macro past-covariates
- ``chronos2.batched_ablation(method_specs, ohlcv, macro, windows, horizon_days)``
  -- fast bulk runner for chronos variants

All forecasting methods satisfy the eval-harness contract; see
``autosignalx.eval.harness.ForecastFn``."""

from autosignalx.forecast import baselines, chronos2  # noqa: F401
from autosignalx.forecast.baselines import (  # noqa: F401
    arima_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)
from autosignalx.forecast.chronos2 import (  # noqa: F401
    chronos2_univariate,
    make_chronos2_multivariate,
)
