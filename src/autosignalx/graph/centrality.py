"""Centrality metrics on a graph built from an edge list.

Three classical centrality measures from NetworkX:
- degree_centrality: fraction of nodes connected to (normalized degree)
- eigenvector_centrality: a node is central if its neighbors are central
- betweenness_centrality: fraction of shortest paths that pass through a node

For trading: high eigenvector centrality marks 'hub' assets whose moves
propagate widely; high betweenness marks bridge assets between sectors."""

from __future__ import annotations

import pandas as pd


def compute_centrality(
    edges: pd.DataFrame, node_set: list[str], directed: bool = False
) -> pd.DataFrame:
    """Build a NetworkX graph from the edge list and compute centrality per node.

    Returns DataFrame with columns
    ``(node, degree_centrality, eigenvector_centrality, betweenness_centrality)``,
    sorted by eigenvector centrality descending."""
    import networkx as nx

    g = nx.DiGraph() if directed else nx.Graph()
    g.add_nodes_from(node_set)
    for _, row in edges.iterrows():
        if row["source"] not in node_set or row["target"] not in node_set:
            continue
        g.add_edge(row["source"], row["target"], weight=abs(float(row["weight"])))

    degree = nx.degree_centrality(g)
    try:
        eigen = nx.eigenvector_centrality_numpy(g, weight="weight")
    except Exception:  # noqa: BLE001
        eigen = {n: 0.0 for n in node_set}
    between = nx.betweenness_centrality(g, weight="weight")

    rows = [
        {
            "node": node,
            "degree_centrality": float(degree.get(node, 0.0)),
            "eigenvector_centrality": float(eigen.get(node, 0.0)),
            "betweenness_centrality": float(between.get(node, 0.0)),
        }
        for node in node_set
    ]
    return pd.DataFrame(rows).sort_values(
        "eigenvector_centrality", ascending=False
    )
