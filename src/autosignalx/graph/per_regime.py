"""Regime-conditioned cross-asset graph.

The default graph layer (`graph.build`) computes a single global
partial-correlation + Granger graph over the full returns history. That
hides regime-specific structural changes: hubs that only matter in
risk-off conditions, edges that flip sign in inflationary regimes,
asset clusters that fragment in crisis. This module re-runs the same
machinery once per regime, on the subset of timesteps with that
regime's label, and persists the results in a per-regime tree.

Output (under ``reports/graph/per_regime/<regime_id>/``):

* ``edges.parquet`` -- combined partial-corr + Granger edge list
* ``centrality.parquet`` -- centrality on the partial-corr graph

Plus a top-level ``reports/graph/per_regime/regime_diff.parquet``
summarising how each asset's centrality shifts across regimes -- the
panel that surfaces "TLT becomes the bridge in regime 3 but is
peripheral in regime 1" type observations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from autosignalx.config import settings
from autosignalx.data import loader
from autosignalx.graph import causality, centrality, correlation


def _regime_root() -> Path:
    p = settings.reports_dir / "graph" / "per_regime"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_regime_labels() -> pd.DataFrame | None:
    path = settings.reports_dir / "regimes" / "kmeans.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if {"timestamp", "regime_id"} <= set(df.columns):
        return df[["timestamp", "regime_id"]].copy()
    return None


def build_per_regime(
    p_threshold: float = 0.05,
    max_lag: int = 5,
    pcorr_threshold: float = 1e-3,
    min_samples: int = 60,
) -> dict[int, dict[str, pd.DataFrame]]:
    """Build one (edges, centrality) pair per regime; return them and persist.

    Regimes with fewer than ``min_samples`` rows are skipped (any GLASSO
    estimator would be too unstable). Returns ``{regime_id: {edges, centrality}}``.
    """
    returns = loader.load_returns_wide().dropna()
    rl = _load_regime_labels()
    if rl is None:
        raise RuntimeError(
            "Per-regime graph requires reports/regimes/kmeans.parquet. "
            "Run `autosignalx regime fit` first."
        )

    rl = rl.set_index("timestamp")["regime_id"]
    aligned = returns.join(rl, how="inner")
    asset_cols = [c for c in aligned.columns if c != "regime_id"]
    out: dict[int, dict[str, pd.DataFrame]] = {}

    centrality_rows: list[dict] = []

    for regime_id, group in aligned.groupby("regime_id", observed=True):
        if len(group) < min_samples:
            continue
        sub = group[asset_cols]

        pcorr_edges = correlation.partial_correlation_edges(sub, threshold=pcorr_threshold)
        granger_edges = causality.granger_edges(sub, max_lag=max_lag, p_threshold=p_threshold)
        all_edges = pd.concat([pcorr_edges, granger_edges], ignore_index=True, sort=False)
        all_edges["regime_id"] = int(regime_id)

        cent = centrality.compute_centrality(pcorr_edges, asset_cols, directed=False)
        cent["regime_id"] = int(regime_id)
        cent["n_samples"] = int(len(sub))

        regime_dir = _regime_root() / f"regime_{int(regime_id)}"
        regime_dir.mkdir(parents=True, exist_ok=True)
        all_edges.to_parquet(regime_dir / "edges.parquet", index=False)
        cent.to_parquet(regime_dir / "centrality.parquet", index=False)

        out[int(regime_id)] = {"edges": all_edges, "centrality": cent}
        centrality_rows.extend(cent.to_dict("records"))

    if centrality_rows:
        diff_df = pd.DataFrame(centrality_rows)
        diff_df.to_parquet(_regime_root() / "centrality_by_regime.parquet", index=False)

        # Compute per-asset centrality dispersion across regimes -- the
        # "regime sensitivity" of each asset's role in the network.
        if "node" in diff_df.columns and "betweenness_centrality" in diff_df.columns:
            sens = (
                diff_df.groupby("node")[["degree_centrality", "eigenvector_centrality", "betweenness_centrality"]]
                .agg(["mean", "std", "max", "min"])
            )
            sens.columns = [f"{a}_{b}" for a, b in sens.columns]
            sens = sens.reset_index()
            sens["betweenness_centrality_range"] = (
                sens["betweenness_centrality_max"] - sens["betweenness_centrality_min"]
            )
            sens = sens.sort_values("betweenness_centrality_range", ascending=False)
            sens.to_parquet(_regime_root() / "regime_sensitivity.parquet", index=False)

    return out


def load_per_regime() -> dict[int, dict[str, pd.DataFrame]]:
    """Read every persisted per-regime graph from disk."""
    root = settings.reports_dir / "graph" / "per_regime"
    if not root.exists():
        return {}
    out: dict[int, dict[str, pd.DataFrame]] = {}
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not p.name.startswith("regime_"):
            continue
        try:
            rid = int(p.name.split("_", 1)[1])
        except ValueError:
            continue
        edges_p = p / "edges.parquet"
        cent_p = p / "centrality.parquet"
        if edges_p.exists() and cent_p.exists():
            out[rid] = {
                "edges": pd.read_parquet(edges_p),
                "centrality": pd.read_parquet(cent_p),
            }
    return out


def load_regime_sensitivity() -> pd.DataFrame:
    """Load the asset-level cross-regime centrality dispersion summary."""
    p = settings.reports_dir / "graph" / "per_regime" / "regime_sensitivity.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)
