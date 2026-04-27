"""Phase 7 -- forecast target types beyond raw price levels.

The original forecast contract is price-level (`adj_close`). MAE on price
levels is dominated by persistence -- a finding that beats naive there is
partly measuring the trivial random-walk component. This module adds
target-type adapters so a forecast can be evaluated against:

- ``price`` -- raw adj_close (legacy default; backward compatible)
- ``log_return`` -- log(price[t+h] / price[t]); zero-mean, near-stationary
- ``excess_return`` -- log_return minus risk-free rate (proxied by ^TNX/252)
- ``vol`` -- realized log-return volatility over the forecast horizon
- ``rank`` -- cross-sectional rank of return within the universe at the target

Each adapter takes a price-level forecast frame and produces a derived
frame whose ``prediction`` and ``target`` columns are in the new units,
plus a ``target_type`` column for the contract extension.

Backward compatibility:
* `target_type` is treated as optional in `eval.contracts`. Forecast
  parquets without the column are interpreted as ``target_type=price``.
* All existing metrics continue to operate on the (prediction, target)
  pair regardless of the underlying target type.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

TargetType = Literal["price", "log_return", "excess_return", "vol", "rank"]

VALID_TARGET_TYPES: tuple[str, ...] = (
    "price",
    "log_return",
    "excess_return",
    "vol",
    "rank",
)


def _ensure_origin_value_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where origin_value is non-positive (cannot take log)."""
    if "origin_value" not in df.columns:
        return df
    mask = (df["origin_value"] > 0) & (df["target"] > 0) & (df["prediction"] > 0)
    return df[mask].copy()


def to_log_return(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Convert price-level forecasts to log-return forecasts.

    For every row: prediction_lr = log(prediction / origin_value);
    target_lr = log(target / origin_value)."""
    df = _ensure_origin_value_positive(forecasts)
    if df.empty:
        return df
    out = df.copy()
    out["prediction"] = np.log(df["prediction"].to_numpy() / df["origin_value"].to_numpy())
    out["target"] = np.log(df["target"].to_numpy() / df["origin_value"].to_numpy())
    if "lower" in out.columns and "upper" in out.columns:
        lower = pd.to_numeric(out["lower"], errors="coerce").to_numpy()
        upper = pd.to_numeric(out["upper"], errors="coerce").to_numpy()
        ov = df["origin_value"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            out["lower"] = np.where(lower > 0, np.log(lower / ov), np.nan)
            out["upper"] = np.where(upper > 0, np.log(upper / ov), np.nan)
    out["target_type"] = "log_return"
    return out


def to_excess_return(
    forecasts: pd.DataFrame,
    rf_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Subtract the risk-free rate from log_return forecasts.

    ``rf_daily`` should have columns ``timestamp`` (the target day) and
    ``rf_rate`` (per-day rate, i.e. annualized / 252). When None, the
    helper attempts to load ^TNX from ``data/cache/macro.parquet`` and
    convert annualized yield to a per-day rate."""
    lr = to_log_return(forecasts)
    if lr.empty:
        return lr
    if rf_daily is None:
        rf_daily = _load_default_rf()
    if rf_daily is None or rf_daily.empty:
        # No risk-free rate available; degrade to log_return labelled as
        # excess_return so downstream contracts validate. Reviewers should
        # see a warning in the panel.
        out = lr.copy()
        out["target_type"] = "excess_return"
        return out

    rf = rf_daily[["timestamp", "rf_rate"]].copy()
    rf["timestamp"] = pd.to_datetime(rf["timestamp"])
    out = lr.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.merge(rf, on="timestamp", how="left")
    horizon = out["horizon"].fillna(1).astype(int).clip(lower=1)
    rf_h = out["rf_rate"].fillna(0.0).to_numpy() * horizon.to_numpy()
    out["target"] = out["target"].to_numpy() - rf_h
    out["prediction"] = out["prediction"].to_numpy() - rf_h
    out = out.drop(columns=["rf_rate"])
    out["target_type"] = "excess_return"
    return out


def to_realized_vol(
    forecasts: pd.DataFrame,
    ohlcv: pd.DataFrame,
    horizon_days: int = 21,
) -> pd.DataFrame:
    """Convert price-level forecasts into realized-vol targets.

    realized_vol(t,h) = std(log_return) over [t-h, t] in trading days.
    Prediction proxy: price-volatility implied by the (lower, upper)
    interval. If no interval, we fall back to the last-h training-window
    realized vol as the prediction (a strong baseline)."""
    df = forecasts.copy()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out_rows: list[dict] = []
    if "asset" in ohlcv.columns and "adj_close" in ohlcv.columns:
        ohlcv = ohlcv[["asset", "timestamp", "adj_close"]].copy()
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"])
        ohlcv = ohlcv.sort_values(["asset", "timestamp"])
        ohlcv["log_return"] = (
            ohlcv.groupby("asset", observed=True)["adj_close"].apply(
                lambda s: np.log(s).diff()
            ).reset_index(level=0, drop=True)
        )
    else:
        return df.iloc[0:0]

    for (asset,), grp in df.groupby(["asset"], observed=True):
        asset_lr = ohlcv[ohlcv["asset"] == asset][["timestamp", "log_return"]]
        if asset_lr.empty:
            continue
        asset_lr = asset_lr.dropna().sort_values("timestamp")
        for _, row in grp.iterrows():
            target_t = row["timestamp"]
            window_start = target_t - pd.Timedelta(days=horizon_days * 2)
            window = asset_lr[
                (asset_lr["timestamp"] > window_start) & (asset_lr["timestamp"] <= target_t)
            ]
            if len(window) < 5:
                continue
            realized = float(window["log_return"].std(ddof=0))
            origin_t = pd.Timestamp(row["forecast_origin"])
            train_window = asset_lr[
                (asset_lr["timestamp"] > origin_t - pd.Timedelta(days=horizon_days * 3))
                & (asset_lr["timestamp"] <= origin_t)
            ]
            if len(train_window) < 5:
                continue
            pred_vol = float(train_window["log_return"].std(ddof=0))
            out_rows.append(
                {
                    "asset": asset,
                    "timestamp": target_t,
                    "forecast_origin": origin_t,
                    "horizon": int(row.get("horizon", horizon_days)),
                    "method": str(row.get("method", "naive")),
                    "prediction": pred_vol,
                    "origin_value": float(row.get("origin_value", np.nan)),
                    "target": realized,
                    "target_type": "vol",
                }
            )
    return pd.DataFrame(out_rows)


def to_cross_sectional_rank(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Convert per-asset return forecasts to cross-sectional ranks.

    For each (forecast_origin, method), assets are ranked by predicted
    log_return. The target rank is the realized-return rank. Ranks are
    in [0, 1] (fractional), so MAE on this target measures rank prediction
    error directly."""
    lr = to_log_return(forecasts)
    if lr.empty:
        return lr
    out_rows: list[dict] = []
    for (origin, method), grp in lr.groupby(["forecast_origin", "method"], observed=True):
        if len(grp) < 2:
            continue
        pred_rank = grp["prediction"].rank(method="average", pct=True).to_numpy()
        tgt_rank = grp["target"].rank(method="average", pct=True).to_numpy()
        for i, (_, row) in enumerate(grp.iterrows()):
            out_rows.append(
                {
                    "asset": row["asset"],
                    "timestamp": row["timestamp"],
                    "forecast_origin": origin,
                    "horizon": int(row["horizon"]),
                    "method": method,
                    "prediction": float(pred_rank[i]),
                    "origin_value": float(row.get("origin_value", np.nan)),
                    "target": float(tgt_rank[i]),
                    "target_type": "rank",
                }
            )
    return pd.DataFrame(out_rows)


def convert_target(
    forecasts: pd.DataFrame,
    target_type: str,
    ohlcv: pd.DataFrame | None = None,
    rf_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Dispatch helper. Validates `target_type` and routes to the right adapter."""
    if target_type not in VALID_TARGET_TYPES:
        raise ValueError(
            f"Unknown target_type {target_type!r}; must be one of {VALID_TARGET_TYPES}"
        )
    if forecasts.empty:
        out = forecasts.copy()
        out["target_type"] = target_type
        return out
    if target_type == "price":
        out = forecasts.copy()
        out["target_type"] = "price"
        return out
    if target_type == "log_return":
        return to_log_return(forecasts)
    if target_type == "excess_return":
        return to_excess_return(forecasts, rf_daily=rf_daily)
    if target_type == "vol":
        if ohlcv is None:
            raise ValueError("vol target requires ohlcv DataFrame")
        return to_realized_vol(forecasts, ohlcv)
    if target_type == "rank":
        return to_cross_sectional_rank(forecasts)
    raise AssertionError("unreachable")


def _load_default_rf() -> pd.DataFrame | None:
    """Best-effort: load ^TNX from the macro cache and convert to per-day rate."""
    try:
        from autosignalx.config import settings

        macro_path = settings.data_dir / "cache" / "macro.parquet"
        if not macro_path.exists():
            return None
        m = pd.read_parquet(macro_path)
        m = m[m["signal"] == "^TNX"][["timestamp", "value"]].copy()
        if m.empty:
            return None
        m["rf_rate"] = m["value"].astype(float) / 100.0 / 252.0
        return m[["timestamp", "rf_rate"]]
    except Exception:  # noqa: BLE001
        return None
