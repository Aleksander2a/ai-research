"""Phase 16 -- Statistical-power dashboard.

For each (asset, regime, method) cell, we compute:

* n_samples -- how many forecast rows live there
* observed_effect_size -- Cohen's d on per-bar loss differences
* power_at_alpha_05 -- approximate power of the existing DM gate at
  alpha=0.05 given the observed effect and n
* min_n_for_80pct_power -- sample-size needed to achieve 80% power
  given the observed effect size

A cell with low n and small observed d is *under-powered* -- the
agent's failure to promote there is not informative. A cell with high
n and small d is genuinely null. The dashboard makes the difference
visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cohen_d(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return float("nan")
    sd = float(np.std(diffs, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(np.mean(diffs) / sd)


def power_at_alpha(d: float, n: int, alpha: float = 0.05) -> float:
    """Approximate one-sided power of a t-test on n observations with effect d.

    For large n we fall back to the normal approximation: scipy's noncentral
    t-distribution can return NaN at large degrees of freedom + moderate ncp.
    The normal approximation is exact in the limit and very close for n>=200."""
    if not np.isfinite(d) or n < 2:
        return float("nan")
    df = n - 1
    if n >= 200:
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ncp = d * np.sqrt(n)
        power = 1.0 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
        return float(np.clip(power, 0.0, 1.0))
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    ncp = d * np.sqrt(n)
    power = 1.0 - stats.nct.cdf(t_crit, df=df, nc=ncp) + stats.nct.cdf(-t_crit, df=df, nc=ncp)
    return float(np.clip(power, 0.0, 1.0))


def min_n_for_power(d: float, target_power: float = 0.8, alpha: float = 0.05) -> int:
    """Solve for n such that power(d, n) ≈ target_power. Bisection on log-n."""
    if not np.isfinite(d) or d == 0:
        return -1
    lo, hi = 4, 100000
    if power_at_alpha(d, hi, alpha) < target_power:
        return -1
    for _ in range(40):
        mid = (lo + hi) // 2
        if power_at_alpha(d, mid, alpha) < target_power:
            lo = mid + 1
        else:
            hi = mid
    return int(hi)


def cell_power(
    forecasts: pd.DataFrame,
    method: str,
    baseline: str = "naive",
    asset: str | None = None,
    regime_id: int | None = None,
) -> dict:
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
        return {"n": 0, "d": float("nan"), "power": float("nan"), "min_n_80": -1}
    la = (merged["prediction_method"] - merged["target"]).abs().to_numpy()
    lb = (merged["prediction_baseline"] - merged["target"]).abs().to_numpy()
    d = cohen_d(lb - la)
    n = int(len(merged))
    p = power_at_alpha(d, n)
    min_n = min_n_for_power(d)
    return {"n": n, "d": d, "power": p, "min_n_80": min_n}


def power_grid(
    forecasts: pd.DataFrame,
    methods: list[str],
    baseline: str = "naive",
    assets: list[str] | None = None,
    regimes: list[int] | None = None,
) -> pd.DataFrame:
    """Compute power statistics for every (asset, regime, method) cell."""
    if forecasts.empty:
        return pd.DataFrame()
    if assets is None:
        assets = sorted(forecasts["asset"].unique())
    if regimes is None and "regime_id" in forecasts.columns:
        regimes = sorted(forecasts["regime_id"].dropna().unique().tolist())
    if regimes is None:
        regimes = [None]
    rows = []
    for m in methods:
        if m == baseline:
            continue
        for a in assets:
            for r in regimes:
                stats_dict = cell_power(forecasts, method=m, baseline=baseline, asset=a, regime_id=r)
                rows.append({
                    "method": m, "asset": a, "regime_id": r, **stats_dict,
                })
    return pd.DataFrame(rows)
