"""Returns-target forecasting baselines (Phase 7).

When the eval target is log-return rather than price, the natural
baselines change:

* ``zero_return`` -- predict 0.0 for every horizon (the random-walk
  hypothesis on returns; the analogue of price-level naive).
* ``mean_return`` -- predict the trailing-window mean log return.
* ``momentum`` -- predict the trailing-window cumulative log-return,
  scaled to the forecast horizon.

These are fed into ``eval/targets.to_log_return``-converted forecasts so
downstream DM / bootstrap gates compare apples to apples (returns vs
returns), not returns to a price-level naive.

The functions return price-level forecasts that are subsequently
converted via ``to_log_return``; this keeps the harness contract single
(it always materialises price-level predictions) and the target-type
conversion explicit and auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zero_return_forecast(
    asset_train: pd.DataFrame,
    origin: pd.Timestamp,  # noqa: ARG001
    target_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Predict zero log-return -> price equals last training value.

    Identical to ``naive_forecast`` for prices, but registered as a
    distinct method when running a returns-target ablation so cockpit
    panels can show both naive (price) and zero_return (returns) cleanly."""
    last_value = float(asset_train["adj_close"].iloc[-1])
    return pd.DataFrame({"timestamp": target_dates, "prediction": last_value})


def mean_return_forecast(
    asset_train: pd.DataFrame,
    origin: pd.Timestamp,  # noqa: ARG001
    target_dates: list[pd.Timestamp],
    lookback: int = 60,
) -> pd.DataFrame:
    """Predict cumulative drift from the training-window mean log return.

    For target horizon h trading days from origin, predicted price is
    last_price * exp(mean_return * h). h is approximated as the row
    index in target_dates (1-based)."""
    series = asset_train.sort_values("timestamp")["adj_close"].to_numpy(dtype=float)
    if len(series) < lookback + 2:
        last = float(series[-1])
        return pd.DataFrame({"timestamp": target_dates, "prediction": last})
    log_ret = np.diff(np.log(series))
    mean_lr = float(np.mean(log_ret[-lookback:]))
    last = float(series[-1])
    horizons = np.arange(1, len(target_dates) + 1, dtype=float)
    preds = last * np.exp(mean_lr * horizons)
    return pd.DataFrame({"timestamp": target_dates, "prediction": preds})


def momentum_forecast(
    asset_train: pd.DataFrame,
    origin: pd.Timestamp,  # noqa: ARG001
    target_dates: list[pd.Timestamp],
    lookback: int = 60,
    horizon_scale: float = 0.5,
) -> pd.DataFrame:
    """Trailing-window cumulative-log-return projected forward, scaled.

    Cumulative log-return over the last ``lookback`` trading days is
    projected onto the forecast horizon at ``horizon_scale``. With
    ``horizon_scale=0.5`` we damp by half (a standard mean-reverting
    correction; full carry-over is too aggressive for daily ETFs)."""
    series = asset_train.sort_values("timestamp")["adj_close"].to_numpy(dtype=float)
    if len(series) < lookback + 2:
        last = float(series[-1])
        return pd.DataFrame({"timestamp": target_dates, "prediction": last})
    log_ret = np.diff(np.log(series))
    cum_lr = float(np.sum(log_ret[-lookback:]))
    last = float(series[-1])
    n = len(target_dates)
    horizons = np.arange(1, n + 1, dtype=float)
    forward_lr = cum_lr / lookback * horizons * horizon_scale
    preds = last * np.exp(forward_lr)
    return pd.DataFrame({"timestamp": target_dates, "prediction": preds})
