"""Baseline forecaster contract tests.

Each baseline must:
- Return a DataFrame with the right columns and length matching target_dates.
- Produce finite, positive predictions in adj_close units.
- Not look at the future (no row from after origin should influence outputs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autosignalx.forecast import baselines


def _synthetic_train(n: int = 300, seed: int = 0, start: str = "2023-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    drift = 0.0003
    sigma = 0.01
    log_ret = rng.normal(drift, sigma, size=n)
    log_price = np.cumsum(log_ret) + np.log(100.0)
    price = np.exp(log_price)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "asset": "SYN",
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "adj_close": price,
            "volume": 1.0,
            "returns": np.r_[np.nan, np.diff(price) / price[:-1]],
        }
    )


def _target_dates(origin: pd.Timestamp, n: int = 21) -> list[pd.Timestamp]:
    return list(pd.bdate_range(origin + pd.Timedelta(days=1), periods=n))


@pytest.mark.parametrize(
    "fn", [baselines.naive_forecast, baselines.seasonal_naive_forecast, baselines.arima_forecast]
)
def test_baseline_returns_aligned_predictions(fn) -> None:
    train = _synthetic_train()
    origin = train["timestamp"].iloc[-1]
    targets = _target_dates(origin, n=21)
    out = fn(train, origin, targets)
    assert list(out.columns)[:2] == ["timestamp", "prediction"]
    assert len(out) == len(targets)
    assert (out["prediction"] > 0).all()
    assert np.isfinite(out["prediction"]).all()


def test_naive_forecast_predicts_last_close() -> None:
    train = _synthetic_train()
    last = float(train["adj_close"].iloc[-1])
    origin = train["timestamp"].iloc[-1]
    out = baselines.naive_forecast(train, origin, _target_dates(origin, n=5))
    assert (out["prediction"] == last).all()


def test_seasonal_naive_falls_back_when_history_too_short() -> None:
    """When season_days exceeds available history, fall back to last value."""
    train = _synthetic_train(n=10)  # too short for 252-day lookback
    origin = train["timestamp"].iloc[-1]
    out = baselines.seasonal_naive_forecast(
        train, origin, _target_dates(origin, n=3), season_days=252
    )
    last = float(train["adj_close"].iloc[-1])
    assert (out["prediction"] == last).all()


def test_arima_does_not_explode() -> None:
    train = _synthetic_train()
    origin = train["timestamp"].iloc[-1]
    out = baselines.arima_forecast(train, origin, _target_dates(origin, n=21))
    last = float(train["adj_close"].iloc[-1])
    # ARIMA forecasts should stay within 50% of the last value over a short horizon
    # for synthetic random-walk data
    assert ((out["prediction"] > 0.5 * last) & (out["prediction"] < 1.5 * last)).all()
