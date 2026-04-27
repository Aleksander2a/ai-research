"""Phase 7 -- returns-target forecast metrics.

When the target is log-return rather than price level, the headline
metrics change. Per-row absolute error is still informative, but several
returns-specific metrics become the primary lens:

* ``forecast_sharpe`` -- the Sharpe of a long/flat strategy that goes
  long when prediction > 0. A *forecasting* metric (no costs, no full
  portfolio) that asks "if I traded the sign of this signal, would it
  produce a positive risk-adjusted excess return?"
* ``hit_rate_returns`` -- fraction of rows where sign(prediction) ==
  sign(target). The directional-accuracy analogue for returns.
* ``ic_pearson`` / ``ic_spearman`` -- information coefficient: linear
  and rank correlation between prediction and target. Quant industry's
  standard alpha-quality summary.
* ``returns_skill`` -- 1 - method_mae / baseline_mae on returns; positive
  means the method beats baseline on the returns target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def forecast_sharpe(prediction: np.ndarray, target: np.ndarray) -> float:
    """Sharpe of a sign-following strategy on per-row returns.

    Strategy: take a unit long position when prediction > 0, unit short
    when prediction < 0, flat at 0. Returns equals sign(prediction) *
    target. Annualises by sqrt(252) assuming daily returns; for other
    frequencies, scale externally."""
    pred = np.asarray(prediction, dtype=float)
    tgt = np.asarray(target, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(tgt)
    if not mask.any():
        return float("nan")
    sig = np.sign(pred[mask])
    pnl = sig * tgt[mask]
    sd = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else float("nan")
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(np.mean(pnl) / sd * np.sqrt(252))


def hit_rate_returns(prediction: np.ndarray, target: np.ndarray) -> float:
    """Fraction of rows where sign(prediction) == sign(target)."""
    pred = np.asarray(prediction, dtype=float)
    tgt = np.asarray(target, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(tgt)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(pred[mask]) == np.sign(tgt[mask])))


def ic_pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation between prediction and target."""
    pred = np.asarray(prediction, dtype=float)
    tgt = np.asarray(target, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(tgt)
    if mask.sum() < 3:
        return float("nan")
    r, _ = stats.pearsonr(pred[mask], tgt[mask])
    return float(r)


def ic_spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation between prediction and target."""
    pred = np.asarray(prediction, dtype=float)
    tgt = np.asarray(target, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(tgt)
    if mask.sum() < 3:
        return float("nan")
    r, _ = stats.spearmanr(pred[mask], tgt[mask])
    return float(r)


def summarise_returns(forecasts: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Per-group MAE + returns-specific metrics."""
    by = by or ["method"]
    rows: list[dict] = []
    for keys, grp in forecasts.groupby(by, observed=True):
        keys_tup = keys if isinstance(keys, tuple) else (keys,)
        row: dict = dict(zip(by, keys_tup, strict=False))
        pred = grp["prediction"].to_numpy()
        tgt = grp["target"].to_numpy()
        row["n"] = int(len(grp))
        diffs = pred - tgt
        row["mae"] = float(np.mean(np.abs(diffs[np.isfinite(diffs)]))) if len(diffs) else float("nan")
        row["forecast_sharpe"] = forecast_sharpe(pred, tgt)
        row["hit_rate"] = hit_rate_returns(pred, tgt)
        row["ic_pearson"] = ic_pearson(pred, tgt)
        row["ic_spearman"] = ic_spearman(pred, tgt)
        rows.append(row)
    return pd.DataFrame(rows)
