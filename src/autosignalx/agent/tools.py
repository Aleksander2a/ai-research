"""Deterministic experiment tools the agent calls.

Each tool reads from the persisted artifacts (forecasts, regimes,
signals, graph) under ``reports/`` and returns a small JSON-serializable
result. Keeping these deterministic means the agent's research loop is
fast (no model retraining per hypothesis) and reproducible."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from autosignalx.config import settings
from autosignalx.eval import harness


def _load_all_forecasts() -> pd.DataFrame:
    """Combine every forecast parquet in reports/ablations/."""
    ablations_dir = settings.reports_dir / "ablations"
    paths = list(ablations_dir.glob("*.parquet")) if ablations_dir.exists() else []
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _load_regime_labels() -> pd.DataFrame:
    p = settings.reports_dir / "regimes" / "kmeans.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _load_signal_ranking() -> pd.DataFrame:
    """Most recently modified ranking parquet under reports/signals/."""
    signals_dir = settings.reports_dir / "signals"
    paths: list[Path] = (
        list(signals_dir.glob("*.parquet")) if signals_dir.exists() else []
    )
    if not paths:
        return pd.DataFrame()
    p = max(paths, key=lambda x: x.stat().st_mtime)
    return pd.read_parquet(p)


def _load_centrality() -> pd.DataFrame:
    p = settings.reports_dir / "graph" / "centrality.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def list_methods() -> list[str]:
    f = _load_all_forecasts()
    return sorted(f["method"].unique().tolist()) if not f.empty else []


def list_assets() -> list[str]:
    f = _load_all_forecasts()
    return sorted(f["asset"].unique().tolist()) if not f.empty else []


def list_regimes() -> list[int]:
    rl = _load_regime_labels()
    if rl.empty:
        return []
    return sorted(rl["regime_id"].unique().tolist())


def slice_forecasts(
    method: str | None = None,
    asset: str | None = None,
    regime_id: int | None = None,
) -> dict[str, Any]:
    """Compute MAE / MAPE / dir-acc / CRPS / skill_vs_naive on a slice of
    the forecast cache, optionally filtered by ``method``, ``asset``,
    and/or ``regime_id`` (joined on ``forecast_origin``)."""
    f = _load_all_forecasts()
    if f.empty:
        return {"error": "no forecasts cached"}

    if regime_id is not None:
        rl = _load_regime_labels()
        if rl.empty:
            return {"error": "no regime labels cached"}
        rl_join = rl[["timestamp", "regime_id"]].rename(columns={"timestamp": "forecast_origin"})
        f = f.merge(rl_join, on="forecast_origin", how="left")
        f = f[f["regime_id"] == regime_id]

    if method is not None:
        f = f[f["method"] == method]
    if asset is not None:
        f = f[f["asset"] == asset]

    if f.empty:
        return {"n": 0}

    summary_overall = harness.summarize(f, by=["method"])
    summary_overall = harness.add_skill_score(summary_overall, baseline_method="naive")

    rows = []
    for _, row in summary_overall.iterrows():
        rows.append(
            {
                "method": str(row["method"]),
                "n": int(row["n"]),
                "mae": _safe_float(row["mae"]),
                "mape": _safe_float(row["mape"]),
                "dir_acc": _safe_float(row["dir_acc"]),
                "crps": _safe_float(row.get("crps")),
                "skill_vs_naive": _safe_float(row.get("skill_vs_naive")),
            }
        )
    return {
        "filters": {"method": method, "asset": asset, "regime_id": regime_id},
        "n_total_rows": int(len(f)),
        "per_method": rows,
    }


def get_top_features(regime_id: int, top_k: int = 5) -> list[dict[str, Any]]:
    """Top-K features for a regime from the signal ranking."""
    r = _load_signal_ranking()
    if r.empty:
        return []
    sub = r[r["regime_id"] == regime_id].sort_values("rank").head(top_k)
    return [
        {
            "feature": row["feature"],
            "importance": _safe_float(row["importance"]),
            "rank": int(row["rank"]),
        }
        for _, row in sub.iterrows()
    ]


def get_centrality_summary() -> dict[str, dict[str, float]]:
    """Per-asset centrality dictionary."""
    c = _load_centrality()
    if c.empty:
        return {}
    return {
        str(row["node"]): {
            "degree": _safe_float(row["degree_centrality"]),
            "eigenvector": _safe_float(row["eigenvector_centrality"]),
            "betweenness": _safe_float(row["betweenness_centrality"]),
        }
        for _, row in c.iterrows()
    }


def context_snapshot() -> dict[str, Any]:
    """Compact snapshot of all available artifact summaries -- used to
    seed the agent's prompts at the start of a run."""
    return {
        "methods": list_methods(),
        "assets": list_assets(),
        "regimes": list_regimes(),
        "top_features_per_regime": {
            r: get_top_features(r, top_k=3) for r in list_regimes()
        },
        "centrality": get_centrality_summary(),
        "overall_metrics": slice_forecasts().get("per_method", []),
    }


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None
