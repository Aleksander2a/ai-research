"""Chronos-2 forecasting layer (L1).

Wraps amazon/chronos-2 from chronos-forecasting. Two flavors of forecast,
both producing a point forecast plus an 80%% interval (10/50/90 quantiles):

- ``chronos2_univariate(asset_train, origin, target_dates)`` -- per-asset,
  satisfies the harness ForecastFn contract.
- ``make_chronos2_multivariate(macro)`` -- returns a closure forecast_fn
  that threads the macro DataFrame through as past covariates.

For bulk ablations across many (window, asset) pairs, ``batched_ablation``
calls the pipeline once with a list of inputs -- the GPU/CPU is amortized
across ~hundreds of calls in the same forward pass."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from functools import lru_cache

import numpy as np
import pandas as pd

from autosignalx.data.splits import WalkForwardWindow

DEFAULT_MODEL = "amazon/chronos-2"
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


@lru_cache(maxsize=2)
def _load_pipeline(model_id: str = DEFAULT_MODEL):
    """Lazy-load and cache Chronos-2 pipeline. First call downloads ~150-300 MB."""
    import torch
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(
        model_id,
        device_map="cpu",
        dtype=torch.float32,
    )


def _align_covariates(
    macro_train: pd.DataFrame, asset_dates: pd.Series
) -> dict[str, np.ndarray]:
    """Forward-fill each macro signal onto the asset's training dates.

    Macro signals (e.g., ^VIX) trade on a slightly different calendar than
    ETFs; ffill plus bfill gives a clean, asset-aligned series with no gaps."""
    cov: dict[str, np.ndarray] = {}
    for sig, sub in macro_train.groupby("signal", observed=True):
        idx = sub.set_index("timestamp")["value"]
        aligned = idx.reindex(asset_dates).ffill().bfill()
        if aligned.isna().all():
            continue
        cov[str(sig)] = aligned.to_numpy(dtype=np.float32)
    return cov


def _predict_quantiles(inputs: list, prediction_length: int) -> tuple[list, list]:
    """Run pipeline.predict_quantiles, suppressing convergence warnings."""
    pipeline = _load_pipeline()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipeline.predict_quantiles(
            inputs,
            prediction_length=prediction_length,
            quantile_levels=DEFAULT_QUANTILES,
        )


def _to_predictions_df(
    quantile_tensor, mean_tensor, target_dates: list[pd.Timestamp]
) -> pd.DataFrame:
    n = len(target_dates)
    q = quantile_tensor.squeeze(0).cpu().numpy()[:n]  # (n, 3)
    mu = mean_tensor.squeeze(0).cpu().numpy()[:n]
    return pd.DataFrame(
        {
            "timestamp": target_dates,
            "prediction": mu,
            "lower": q[:, 0],
            "upper": q[:, 2],
        }
    )


# ---- per-call ForecastFn implementations (harness-compatible) ----


def chronos2_univariate(
    asset_train: pd.DataFrame,
    origin: pd.Timestamp,  # noqa: ARG001
    target_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Single-asset Chronos-2 forecast on the asset's adj_close history."""
    series = asset_train.sort_values("timestamp")["adj_close"].to_numpy(dtype=np.float32)
    quantiles, mean = _predict_quantiles([series], prediction_length=len(target_dates))
    return _to_predictions_df(quantiles[0], mean[0], target_dates)


def make_chronos2_multivariate(macro: pd.DataFrame):
    """Build a forecast_fn that uses macro signals as past covariates.

    The closure captures the macro DataFrame; the harness calls the returned
    function as ``fn(asset_train, origin, target_dates)`` like any other
    forecast_fn."""

    def fn(
        asset_train: pd.DataFrame,
        origin: pd.Timestamp,
        target_dates: list[pd.Timestamp],
    ) -> pd.DataFrame:
        asset_train = asset_train.sort_values("timestamp")
        series = asset_train["adj_close"].to_numpy(dtype=np.float32)
        macro_train = macro[macro["timestamp"] <= origin]
        cov = _align_covariates(macro_train, asset_train["timestamp"])
        inputs = (
            [series] if not cov else [{"target": series, "past_covariates": cov}]
        )
        quantiles, mean = _predict_quantiles(inputs, prediction_length=len(target_dates))
        return _to_predictions_df(quantiles[0], mean[0], target_dates)

    return fn


# ---- batched bulk runner (faster than per-call harness loop for ablation) ----


def batched_ablation(
    method_specs: dict[str, dict],
    ohlcv: pd.DataFrame,
    macro: pd.DataFrame,
    windows: Iterable[WalkForwardWindow],
    horizon_days: int,
    min_train_rows: int = 30,
) -> pd.DataFrame:
    """Run one or more chronos variants across (window, asset) pairs in a
    single batched call per variant.

    ``method_specs`` is a dict like
        ``{"chronos2_univariate": {"use_covariates": False},
           "chronos2_multivariate": {"use_covariates": True}}``"""
    rows: list[dict] = []
    windows_list = list(windows)

    for method_name, spec in method_specs.items():
        use_cov = bool(spec.get("use_covariates", False))
        all_inputs: list = []
        meta: list[dict] = []

        for window in windows_list:
            for asset, asset_full in ohlcv.groupby("asset", observed=True):
                asset_full = asset_full.sort_values("timestamp")
                train = asset_full[asset_full["timestamp"] <= window.train_end]
                test = asset_full[
                    (asset_full["timestamp"] > window.train_end)
                    & (asset_full["timestamp"] <= window.forecast_end)
                ]
                if len(train) < min_train_rows or test.empty:
                    continue
                series = train["adj_close"].to_numpy(dtype=np.float32)
                if use_cov:
                    macro_train = macro[macro["timestamp"] <= window.train_end]
                    cov = _align_covariates(macro_train, train["timestamp"])
                    item = {"target": series, "past_covariates": cov} if cov else series
                else:
                    item = series

                all_inputs.append(item)
                meta.append(
                    {
                        "asset": asset,
                        "forecast_origin": window.train_end,
                        "target_dates": test["timestamp"].tolist(),
                        "origin_value": float(train["adj_close"].iloc[-1]),
                        "test": test,
                    }
                )

        if not all_inputs:
            continue

        quantiles_list, mean_list = _predict_quantiles(
            all_inputs, prediction_length=horizon_days
        )

        for q_t, m_t, m in zip(quantiles_list, mean_list, meta, strict=False):
            preds = _to_predictions_df(q_t, m_t, m["target_dates"])
            merged = preds.merge(
                m["test"][["timestamp", "adj_close"]],
                on="timestamp",
                how="inner",
            )
            if merged.empty:
                continue
            for _, p in merged.iterrows():
                rows.append(
                    {
                        "timestamp": p["timestamp"],
                        "asset": m["asset"],
                        "forecast_origin": m["forecast_origin"],
                        "horizon": int((p["timestamp"] - m["forecast_origin"]).days),
                        "method": method_name,
                        "prediction": float(p["prediction"]),
                        "lower": float(p["lower"]),
                        "upper": float(p["upper"]),
                        "origin_value": m["origin_value"],
                        "target": float(p["adj_close"]),
                    }
                )

    return pd.DataFrame(rows)
