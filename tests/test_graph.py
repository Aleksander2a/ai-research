"""Graph layer tests: edge list contracts and centrality basics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autosignalx.graph import causality, centrality, correlation


def _synthetic_returns(n: int = 200, n_assets: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, n)
    cols = {}
    for i in range(n_assets):
        idiosyncratic = rng.normal(0, 0.005, n)
        cols[f"A{i}"] = common + idiosyncratic  # all assets share a factor
    return pd.DataFrame(cols, index=pd.date_range("2024-01-01", periods=n))


def test_partial_correlation_returns_edge_list() -> None:
    returns = _synthetic_returns(n=300, n_assets=5)
    edges = correlation.partial_correlation_edges(returns)
    assert set(edges.columns) >= {"source", "target", "edge_type", "weight"}
    assert (edges["edge_type"] == "partial_corr").all()


def test_partial_correlation_too_few_rows_raises() -> None:
    returns = _synthetic_returns(n=20, n_assets=4)
    try:
        correlation.partial_correlation_edges(returns)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_granger_edges_smoke() -> None:
    returns = _synthetic_returns(n=300, n_assets=4)
    edges = causality.granger_edges(returns, max_lag=3, p_threshold=0.5)
    assert set(edges.columns) >= {
        "source",
        "target",
        "edge_type",
        "weight",
        "p_value",
        "best_lag",
    }
    if not edges.empty:
        assert (edges["edge_type"] == "granger").all()
        assert (edges["p_value"] >= 0).all()
        assert (edges["p_value"] <= 1).all()


def test_compute_centrality_returns_one_row_per_node() -> None:
    edges = pd.DataFrame(
        [
            {"source": "A", "target": "B", "edge_type": "x", "weight": 0.5},
            {"source": "A", "target": "C", "edge_type": "x", "weight": 0.3},
        ]
    )
    cent = centrality.compute_centrality(edges, ["A", "B", "C", "D"], directed=False)
    assert len(cent) == 4
    assert set(cent.columns) >= {
        "node",
        "degree_centrality",
        "eigenvector_centrality",
        "betweenness_centrality",
    }
    # Centrality is in [0, 1]
    for col in (
        "degree_centrality",
        "eigenvector_centrality",
        "betweenness_centrality",
    ):
        assert (cent[col] >= 0).all()


def test_centrality_isolated_node_zero() -> None:
    edges = pd.DataFrame(
        [{"source": "A", "target": "B", "edge_type": "x", "weight": 1.0}]
    )
    cent = centrality.compute_centrality(edges, ["A", "B", "C"], directed=False)
    isolated = cent[cent["node"] == "C"].iloc[0]
    assert isolated["degree_centrality"] == 0.0
    assert isolated["betweenness_centrality"] == 0.0
