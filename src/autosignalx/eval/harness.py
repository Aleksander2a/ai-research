"""Walk-forward evaluation harness.

The harness is the contract between forecasting methods and metrics. It:
1. Iterates over walk-forward windows produced by ``data.splits``.
2. For each (window, asset), slices training data up to ``window.train_end``,
   gathers the realized target trading days from the cache, and calls the
   method's forecasting function.
3. Joins predictions with realized targets to produce a unified forecast
   DataFrame matching ``eval.contracts.FORECAST_COLUMNS_REQUIRED``.
4. Provides ``summarize()`` to aggregate per-(method, asset) metrics.

The contract for a forecasting function is:
    ``forecast_fn(asset_train: pd.DataFrame, origin: pd.Timestamp,
                  target_dates: list[pd.Timestamp]) -> pd.DataFrame``
returning a frame with at least ``timestamp`` and ``prediction`` columns
(optionally ``lower`` and ``upper`` for probabilistic methods)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from autosignalx.data.splits import WalkForwardWindow
from autosignalx.eval import metrics
from autosignalx.eval.contracts import assert_forecast_schema

ForecastFn = Callable[[pd.DataFrame, pd.Timestamp, list[pd.Timestamp]], pd.DataFrame]


def run_walk_forward(
    method_name: str,
    forecast_fn: ForecastFn,
    ohlcv: pd.DataFrame,
    windows: list[WalkForwardWindow],
    min_train_rows: int = 30,
) -> pd.DataFrame:
    """Run one forecasting method across walk-forward windows and assets.

    Returns a forecasts DataFrame matching the eval contract. Skips
    (window, asset) pairs whose training set is too small or whose forecast
    function raises -- one bad asset shouldn't kill the run."""
    rows: list[pd.DataFrame] = []
    for window in windows:
        for asset, asset_full in ohlcv.groupby("asset", observed=True):
            asset_full = asset_full.sort_values("timestamp")
            train = asset_full[asset_full["timestamp"] <= window.train_end]
            test = asset_full[
                (asset_full["timestamp"] > window.train_end)
                & (asset_full["timestamp"] <= window.forecast_end)
            ]
            if len(train) < min_train_rows or test.empty:
                continue

            target_dates = test["timestamp"].tolist()
            origin_value = float(train["adj_close"].iloc[-1])

            try:
                preds = forecast_fn(train, window.train_end, target_dates)
            except Exception:  # noqa: BLE001
                continue

            merged = preds.merge(
                test[["timestamp", "adj_close"]],
                on="timestamp",
                how="inner",
            )
            if merged.empty:
                continue

            merged = merged.assign(
                asset=asset,
                method=method_name,
                forecast_origin=window.train_end,
                horizon=(merged["timestamp"] - window.train_end).dt.days.astype(int),
                origin_value=origin_value,
            ).rename(columns={"adj_close": "target"})

            rows.append(merged)

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "asset",
                "forecast_origin",
                "horizon",
                "method",
                "prediction",
                "origin_value",
                "target",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    # Promote optional columns to None if the method didn't supply them
    for col in ("lower", "upper"):
        if col not in out.columns:
            out[col] = pd.NA
    assert_forecast_schema(out)
    return out


def ablation(
    methods: dict[str, ForecastFn],
    ohlcv: pd.DataFrame,
    windows: list[WalkForwardWindow],
    min_train_rows: int = 30,
) -> pd.DataFrame:
    """Run multiple forecasting methods and concatenate their forecast frames."""
    frames = [
        run_walk_forward(name, fn, ohlcv, windows, min_train_rows)
        for name, fn in methods.items()
    ]
    return pd.concat(frames, ignore_index=True)


def summarize(
    forecasts: pd.DataFrame, by: list[str] | None = None
) -> pd.DataFrame:
    """Per-group MAE / MAPE / directional accuracy / CRPS.

    CRPS is computed only for methods that supply quantile intervals
    (lower / upper present and finite); for point-only methods the CRPS
    column is NaN. Quantile levels assumed: 0.1 (lower), 0.5 (prediction
    = median), 0.9 (upper)."""
    import numpy as np

    by = by or ["method", "asset"]
    rows: list[dict] = []
    quantile_levels = np.array([0.1, 0.5, 0.9])

    for keys, grp in forecasts.groupby(by, observed=True):
        keys_tup = keys if isinstance(keys, tuple) else (keys,)
        row: dict = dict(zip(by, keys_tup, strict=False))
        row["n"] = len(grp)
        row["mae"] = metrics.mae(grp["prediction"].to_numpy(), grp["target"].to_numpy())
        row["mape"] = metrics.mape(grp["prediction"].to_numpy(), grp["target"].to_numpy())
        row["dir_acc"] = metrics.directional_accuracy(
            grp["prediction"].to_numpy(),
            grp["target"].to_numpy(),
            grp["origin_value"].to_numpy(),
        )
        if "lower" in grp.columns and "upper" in grp.columns:
            mask = grp["lower"].notna() & grp["upper"].notna()
            if mask.any():
                q_array = np.column_stack(
                    [
                        grp.loc[mask, "lower"].to_numpy(),
                        grp.loc[mask, "prediction"].to_numpy(),
                        grp.loc[mask, "upper"].to_numpy(),
                    ]
                )
                row["crps"] = metrics.crps_from_quantiles(
                    q_array, quantile_levels, grp.loc[mask, "target"].to_numpy()
                )
            else:
                row["crps"] = float("nan")
        else:
            row["crps"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def add_skill_score(
    summary: pd.DataFrame, baseline_method: str = "naive"
) -> pd.DataFrame:
    """Append ``skill_vs_<baseline>`` per asset, computed from MAE."""
    out = summary.copy()
    if "asset" not in out.columns:
        baseline_mae = out.loc[out["method"] == baseline_method, "mae"]
        if baseline_mae.empty:
            return out
        baseline_value = float(baseline_mae.iloc[0])
        out[f"skill_vs_{baseline_method}"] = out["mae"].apply(
            lambda m: metrics.skill_score(m, baseline_value)
        )
        return out

    baseline_per_asset = (
        out[out["method"] == baseline_method].set_index("asset")["mae"].to_dict()
    )
    out[f"skill_vs_{baseline_method}"] = [
        metrics.skill_score(mae_, baseline_per_asset.get(asset, float("nan")))
        for mae_, asset in zip(out["mae"], out["asset"], strict=False)
    ]
    return out
