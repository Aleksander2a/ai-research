"""Constrained code-spec DSL for agent-authored forecasting methods.

The agent cannot execute arbitrary Python (that's deferred to Iter 20).
Instead, it emits a structured **method spec** -- a small, validated
JSON object that this module translates into a real ForecastFn and
executes through the standard walk-forward harness. The output is a
``reports/ablations/<name>.parquet`` indistinguishable from a
human-authored method's results, so it shows up in the Forecast Arena
panel and is selectable for downstream slicing / promotion.

Spec schema (validated at execute time):

    {
        "name": "chronos2_dxyonly",          # required, becomes method label
        "base": "chronos2_multivariate",     # one of: naive, arima,
                                             #   chronos2_univariate,
                                             #   chronos2_multivariate
        "covariate_subset": ["DX-Y.NYB"],    # optional; only used when
                                             #   base == chronos2_multivariate
        "ensemble_naive_weight": 0.0,        # float in [0, 1];
                                             #   0 = pure base; 1 = pure naive
        "max_windows": 12,                   # cap on windows for fast
                                             #   iteration (None = use all)
        "asset_subset": ["SPY", "EFA"]       # optional; restrict to these
    }

The ``max_windows`` and ``asset_subset`` fields keep agent-authored
experiments fast (~1-2 minutes for a typical chronos-based spec)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from autosignalx.config import settings
from autosignalx.data import cache, loader, splits
from autosignalx.eval import harness

ALLOWED_BASES = {"naive", "arima", "chronos2_univariate", "chronos2_multivariate"}
ABLATIONS_DIR = settings.reports_dir / "ablations"


def validate_spec(spec: dict[str, Any]) -> tuple[bool, str]:
    """Return (ok, error_message). Validates every field before execution."""
    if not isinstance(spec, dict):
        return False, "spec must be a dict"
    name = spec.get("name", "")
    if not isinstance(name, str) or not name:
        return False, "spec.name (str, non-empty) is required"
    if not name.replace("_", "").replace("-", "").isalnum():
        return False, "spec.name must be alphanumeric (with underscores / dashes)"
    base = spec.get("base", "")
    if base not in ALLOWED_BASES:
        return False, f"spec.base must be one of {sorted(ALLOWED_BASES)}"
    cov = spec.get("covariate_subset")
    if cov is not None and not (isinstance(cov, list) and all(isinstance(c, str) for c in cov)):
        return False, "spec.covariate_subset must be list[str] or null"
    weight = spec.get("ensemble_naive_weight", 0.0)
    if not isinstance(weight, (int, float)) or not (0.0 <= weight <= 1.0):
        return False, "spec.ensemble_naive_weight must be in [0, 1]"
    mw = spec.get("max_windows")
    if mw is not None and (not isinstance(mw, int) or mw <= 0):
        return False, "spec.max_windows must be a positive int or null"
    asub = spec.get("asset_subset")
    if asub is not None and not (isinstance(asub, list) and all(isinstance(a, str) for a in asub)):
        return False, "spec.asset_subset must be list[str] or null"
    return True, ""


def _build_forecast_fn(spec: dict[str, Any]):
    """Translate a validated spec into a ForecastFn, possibly with covariate
    subsetting and naive-ensembling baked in."""
    base = spec["base"]
    covariate_subset = spec.get("covariate_subset")
    ensemble_w = float(spec.get("ensemble_naive_weight", 0.0))

    from autosignalx.forecast import baselines

    # Build the base forecast function
    if base == "naive":
        base_fn = baselines.naive_forecast
    elif base == "arima":
        base_fn = baselines.arima_forecast
    elif base == "chronos2_univariate":
        from autosignalx.forecast import chronos2

        base_fn = chronos2.chronos2_univariate
    elif base == "chronos2_multivariate":
        from autosignalx.forecast import chronos2

        macro_full = loader.load_macro_wide()
        if covariate_subset is not None:
            keep = [c for c in covariate_subset if c in macro_full.columns]
            macro_filtered = macro_full[keep] if keep else macro_full.iloc[:, :0]
        else:
            macro_filtered = macro_full
        base_fn = chronos2.make_chronos2_multivariate(
            macro_filtered.reset_index().melt(
                id_vars="timestamp", var_name="signal", value_name="value"
            )
        )
    else:
        raise ValueError(f"unknown base: {base}")

    if ensemble_w <= 0.0:
        return base_fn

    def ensemble_fn(asset_train, origin, target_dates):  # noqa: ANN001
        base_pred = base_fn(asset_train, origin, target_dates)
        naive_pred = baselines.naive_forecast(asset_train, origin, target_dates)
        out = base_pred.copy()
        out["prediction"] = (
            (1.0 - ensemble_w) * base_pred["prediction"].to_numpy()
            + ensemble_w * naive_pred["prediction"].to_numpy()
        )
        # Intervals collapse if blending; if base provided them, keep them
        # as-is so the user can see uncertainty inherited from base
        return out

    return ensemble_fn


def execute(spec: dict[str, Any], config_name: str = "default") -> dict[str, Any]:
    """Validate, build, run, persist. Returns a result dict with status,
    output path, and a per-method summary computed via ``harness.summarize``."""
    ok, err = validate_spec(spec)
    if not ok:
        return {"status": "error", "error": err}

    from autosignalx.config import load_config

    cfg = load_config(config_name)
    eval_cfg = cfg["eval"]
    splits_cfg = eval_cfg["splits"]

    ohlcv = cache.read_ohlcv()
    asset_subset: Iterable[str] | None = spec.get("asset_subset")
    if asset_subset:
        ohlcv = ohlcv[ohlcv["asset"].isin(asset_subset)]
        if ohlcv.empty:
            return {"status": "error", "error": "asset_subset filtered out all data"}

    windows = splits.walk_forward_windows(
        val_end=splits_cfg["val_end"],
        test_end=splits_cfg["test_end"],
        horizon_days=eval_cfg["forecast_horizon_days"],
        step_days=eval_cfg["rolling_step_days"],
    )
    max_windows = spec.get("max_windows")
    if max_windows is not None:
        windows = windows[: int(max_windows)]

    fn = _build_forecast_fn(spec)
    name = spec["name"]
    forecasts = harness.run_walk_forward(name, fn, ohlcv, windows)
    if forecasts.empty:
        return {"status": "error", "error": "no forecasts produced"}

    ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ABLATIONS_DIR / f"{name}.parquet"
    forecasts.to_parquet(out_path, index=False)

    summary = harness.summarize(forecasts, by=["method"])
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    return {
        "status": "ok",
        "name": name,
        "output_path": str(out_path),
        "n_rows": int(len(forecasts)),
        "n_windows": int(len(windows)),
        "summary": {
            "mae": _safe_float(row.get("mae")),
            "mape": _safe_float(row.get("mape")),
            "dir_acc": _safe_float(row.get("dir_acc")),
            "crps": _safe_float(row.get("crps")),
        },
    }


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if np.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None
