"""Phase 16 -- Counterfactual finding cards.

For each promoted finding, we expose three counterfactual lenses that
turn the cockpit's per-finding card from "here is the evidence" into
"here is *why* the evidence might or might not hold":

1. **Factor residualization** -- regress the per-bar loss-difference
   on a small Fama-French-style factor proxy (SPY-mkt return, equity
   vol via ^VIX delta, dollar via DXY delta, rates via TNX delta).
   The residual lift is the alpha after subtracting common-factor
   exposure. If the residual collapses, the finding was a factor bet.

2. **What-if perturbations** -- recompute MAE/skill assuming the macro
   covariate that drove the regime had been ±1 std lower. Surfaces
   sensitivity of the lift to a single signal.

3. **Outlier removal** -- drop the top-1% absolute loss-difference
   rows and recompute the gate. A finding driven by a handful of large
   moves should fail; a finding driven by stable structure should pass.

These are advisory; they do not change the promotion gate. Their value
is making the *reasoning* legible to a reviewer who reads the finding
card.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _aligned_diffs(
    forecasts: pd.DataFrame,
    method: str,
    baseline: str,
    asset: str | None,
    regime_id: int | None,
) -> pd.DataFrame:
    sub = forecasts.copy()
    if asset is not None:
        sub = sub[sub["asset"] == asset]
    if regime_id is not None and "regime_id" in sub.columns:
        sub = sub[sub["regime_id"] == regime_id]
    keys = ["timestamp", "asset", "forecast_origin"]
    a = sub[sub["method"] == method][[*keys, "prediction", "target"]]
    b = sub[sub["method"] == baseline][[*keys, "prediction"]]
    merged = a.merge(b, on=keys, suffixes=("_method", "_baseline"))
    if merged.empty:
        return merged
    merged["abs_method"] = (merged["prediction_method"] - merged["target"]).abs()
    merged["abs_baseline"] = (merged["prediction_baseline"] - merged["target"]).abs()
    merged["loss_diff"] = merged["abs_baseline"] - merged["abs_method"]
    return merged


def factor_residualization(
    forecasts: pd.DataFrame,
    method: str,
    baseline: str,
    asset: str | None,
    regime_id: int | None,
    macro: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Regress per-bar loss-difference on macro factors; report residual mean.

    macro should have columns ``timestamp``, ``signal``, ``value``. We pivot
    to wide and use 5-day diffs as factor returns. If macro is None,
    we attempt to load it from ``data/cache/macro.parquet``."""
    merged = _aligned_diffs(forecasts, method, baseline, asset, regime_id)
    if merged.empty:
        return {"reason": "empty"}

    if macro is None:
        try:
            from autosignalx.config import settings

            macro_path = settings.data_dir / "cache" / "macro.parquet"
            if macro_path.exists():
                macro = pd.read_parquet(macro_path)
        except Exception:  # noqa: BLE001
            macro = None
    if macro is None or macro.empty:
        return {"reason": "no_macro"}

    macro_w = (
        macro.pivot_table(index="timestamp", columns="signal", values="value")
        .sort_index()
    )
    factor = macro_w.pct_change(periods=5).dropna()
    factor.index.name = "timestamp"
    factor = factor.reset_index()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    factor["timestamp"] = pd.to_datetime(factor["timestamp"])
    aligned = merged[["timestamp", "loss_diff"]].merge(factor, on="timestamp", how="inner")
    if len(aligned) < 30:
        return {"reason": "insufficient_aligned_rows", "n": int(len(aligned))}

    factor_cols = [c for c in factor.columns if c != "timestamp"]
    X = aligned[factor_cols].to_numpy()
    y = aligned["loss_diff"].to_numpy()
    # Add intercept
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ coef
    residuals = y - y_hat
    raw_mean = float(np.mean(y))
    resid_mean = float(np.mean(residuals))
    # t-stat of intercept (residual mean / standard error)
    se_resid = float(np.std(residuals, ddof=len(coef)) / np.sqrt(len(residuals)))
    t_resid = resid_mean / se_resid if se_resid > 0 else float("nan")
    p_resid = float(2.0 * (1.0 - stats.norm.cdf(abs(t_resid)))) if np.isfinite(t_resid) else float("nan")
    return {
        "n": int(len(aligned)),
        "factor_cols": factor_cols,
        "raw_mean_loss_diff": raw_mean,
        "residual_mean_loss_diff": resid_mean,
        "fraction_explained": (
            float(1.0 - resid_mean / raw_mean) if raw_mean != 0 else float("nan")
        ),
        "t_residual": t_resid,
        "p_residual": p_resid,
        "factor_betas": dict(zip(
            ["intercept", *factor_cols], [float(c) for c in coef], strict=False
        )),
    }


def what_if_perturbation(
    forecasts: pd.DataFrame,
    method: str,
    baseline: str,
    asset: str | None,
    regime_id: int | None,
    n_buckets: int = 4,
) -> dict[str, Any]:
    """Slice the loss-difference series by quartile of forecast prediction
    magnitude and report skill within each bucket.

    Surfaces whether the lift comes from particular prediction-magnitude
    regions (e.g. only when the model predicts large moves)."""
    merged = _aligned_diffs(forecasts, method, baseline, asset, regime_id)
    if merged.empty:
        return {"reason": "empty"}
    merged["pred_mag"] = (merged["prediction_method"] - merged["target"]).abs() + 1e-12
    quantiles = np.quantile(merged["pred_mag"], np.linspace(0, 1, n_buckets + 1))
    out = []
    for i in range(n_buckets):
        lo = quantiles[i]
        hi = quantiles[i + 1]
        if i == n_buckets - 1:
            mask = (merged["pred_mag"] >= lo) & (merged["pred_mag"] <= hi)
        else:
            mask = (merged["pred_mag"] >= lo) & (merged["pred_mag"] < hi)
        sub = merged[mask]
        if sub.empty:
            out.append({"bucket": i, "n": 0, "mean_loss_diff": None})
            continue
        out.append({
            "bucket": i,
            "lo": float(lo), "hi": float(hi),
            "n": int(len(sub)),
            "mean_loss_diff": float(np.mean(sub["loss_diff"])),
        })
    return {"buckets": out}


def outlier_removal(
    forecasts: pd.DataFrame,
    method: str,
    baseline: str,
    asset: str | None,
    regime_id: int | None,
    quantile: float = 0.99,
) -> dict[str, Any]:
    """Drop the top-quantile rows by absolute loss-difference and recompute
    the mean / skill / DM-stat-light statistics."""
    merged = _aligned_diffs(forecasts, method, baseline, asset, regime_id)
    if merged.empty:
        return {"reason": "empty"}
    abs_diff = merged["loss_diff"].abs()
    cutoff = float(np.quantile(abs_diff, quantile))
    inlier = merged[abs_diff <= cutoff]
    raw_mean = float(np.mean(merged["loss_diff"]))
    inlier_mean = float(np.mean(inlier["loss_diff"]))
    raw_skill = float(
        1.0 - merged["abs_method"].mean() / merged["abs_baseline"].mean()
    ) if merged["abs_baseline"].mean() > 0 else float("nan")
    inlier_skill = float(
        1.0 - inlier["abs_method"].mean() / inlier["abs_baseline"].mean()
    ) if inlier["abs_baseline"].mean() > 0 else float("nan")
    return {
        "n_total": int(len(merged)),
        "n_inlier": int(len(inlier)),
        "cutoff_quantile": quantile,
        "cutoff_value": cutoff,
        "raw_mean": raw_mean,
        "inlier_mean": inlier_mean,
        "raw_skill_vs_baseline": raw_skill,
        "inlier_skill_vs_baseline": inlier_skill,
    }


def counterfactual_card(
    forecasts: pd.DataFrame,
    method: str,
    baseline: str,
    asset: str | None,
    regime_id: int | None,
) -> dict[str, Any]:
    """Bundle factor residualization + what-if + outlier-removal for a finding."""
    return {
        "factor_residualization": factor_residualization(
            forecasts, method, baseline, asset, regime_id
        ),
        "what_if": what_if_perturbation(
            forecasts, method, baseline, asset, regime_id
        ),
        "outlier_removal": outlier_removal(
            forecasts, method, baseline, asset, regime_id
        ),
    }
