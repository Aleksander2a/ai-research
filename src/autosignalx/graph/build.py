"""Orchestrate the cross-asset graph build.

Loads the wide returns matrix, computes partial-correlation edges
(undirected) + Granger-causality edges (directed), computes centrality
on the partial-correlation graph, and persists everything under
``reports/graph/``."""

from __future__ import annotations

import pandas as pd

from autosignalx.config import settings
from autosignalx.data import loader
from autosignalx.graph import causality, centrality, correlation

GRAPH_DIR = settings.reports_dir / "graph"


def build_and_save(
    p_threshold: float = 0.05,
    max_lag: int = 5,
    pcorr_threshold: float = 1e-3,
) -> dict[str, pd.DataFrame]:
    """Build the full graph and write artifacts.

    Outputs (under ``reports/graph/``):
    - ``edges.parquet`` -- combined partial-corr + Granger edge list
    - ``centrality.parquet`` -- per-asset centrality on the partial-corr graph"""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    returns = loader.load_returns_wide().dropna()
    asset_names = list(returns.columns)

    pcorr_edges = correlation.partial_correlation_edges(returns, threshold=pcorr_threshold)
    granger_edges = causality.granger_edges(returns, max_lag=max_lag, p_threshold=p_threshold)

    all_edges = pd.concat([pcorr_edges, granger_edges], ignore_index=True, sort=False)
    all_edges.to_parquet(GRAPH_DIR / "edges.parquet", index=False)

    cent = centrality.compute_centrality(pcorr_edges, asset_names, directed=False)
    cent.to_parquet(GRAPH_DIR / "centrality.parquet", index=False)

    return {
        "edges": all_edges,
        "centrality": cent,
        "partial_corr": pcorr_edges,
        "granger": granger_edges,
    }
