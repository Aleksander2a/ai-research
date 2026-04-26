"""Helpers that turn forecast/regime/finding artifacts into the
per-rebalance signals consumed by signal-driven strategies.

The forecast layer writes ``reports/ablations/chronos2.parquet`` with one
row per (timestamp, asset, method, forecast_origin, horizon). Strategies
need the *predicted return over the holding period* for each origin and
asset; this module extracts that from the raw forecast parquet without
duplicating any model logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from autosignalx.config import settings


def _ablations_path(
    filename: str = "chronos2.parquet", reports_root: Path | None = None
) -> Path:
    base = reports_root if reports_root is not None else settings.reports_dir
    return base / "ablations" / filename


def load_forecast_signals(
    method: str = "chronos2_multivariate",
    holding_horizon: int = 20,
    filename: str = "chronos2.parquet",
    reports_root: Path | None = None,
) -> pd.DataFrame:
    """Per-origin predicted-return panel.

    Args:
        method: forecast-method name to filter on (must match the
            ``method`` column in the ablations parquet).
        holding_horizon: number of bars between rebalances. The signal
            for each (origin, asset) pair is the prediction at the row
            whose ``horizon`` is closest to ``holding_horizon``.
        filename: ablation file under ``reports/ablations/``.

    Returns:
        DataFrame with columns ``forecast_origin``, ``asset``,
        ``predicted_return`` (= prediction / origin_value - 1).
    """
    path = _ablations_path(filename, reports_root=reports_root)
    if not path.exists():
        raise FileNotFoundError(
            f"forecast ablations not found at {path}; "
            f"run `autosignalx eval chronos` first"
        )
    df = pd.read_parquet(path)
    df = df[df["method"] == method].copy()
    if df.empty:
        raise ValueError(
            f"no rows for method {method!r} in {path}; "
            f"available: {sorted(pd.read_parquet(path)['method'].unique())}"
        )

    df["_h_diff"] = (df["horizon"] - holding_horizon).abs()
    idx = df.groupby(["forecast_origin", "asset"], observed=True)["_h_diff"].idxmin()
    sel = df.loc[idx, ["forecast_origin", "asset", "prediction", "origin_value"]].copy()
    sel["predicted_return"] = sel["prediction"] / sel["origin_value"] - 1.0
    sel["forecast_origin"] = pd.to_datetime(sel["forecast_origin"])
    return sel[["forecast_origin", "asset", "predicted_return"]].reset_index(drop=True)


def load_regime_series(
    method: str = "kmeans_contrastive",
    filename: str = "kmeans.parquet",
    reports_root: Path | None = None,
) -> pd.Series:
    """Daily regime ID series indexed by timestamp.

    Regime labels for the test window are predicted by a model fit on
    data through the val cutoff, so reading these labels does not
    introduce look-ahead.
    """
    base = reports_root if reports_root is not None else settings.reports_dir
    path = base / "regimes" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"regime artifacts not found at {path}; "
            f"run `autosignalx regime fit` first"
        )
    df = pd.read_parquet(path)
    df = df[df["method"] == method].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")["regime_id"].sort_index()


def load_promoted_findings(reports_root: Path | None = None) -> list[dict]:
    """Return all rows from ``reports/agent/findings.jsonl`` as dicts."""
    import json

    base = reports_root if reports_root is not None else settings.reports_dir
    path = base / "agent" / "findings.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
