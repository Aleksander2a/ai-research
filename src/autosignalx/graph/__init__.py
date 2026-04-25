"""Relational layer (L4) -- partial-correlation graph + Granger causality + centrality.

Public API:
- ``correlation.partial_correlation_edges(returns, threshold)`` -- undirected
- ``causality.granger_edges(returns, max_lag, p_threshold)`` -- directed
- ``centrality.compute_centrality(edges, node_set, directed)`` -- NetworkX-based
- ``build.build_and_save(p_threshold, max_lag, pcorr_threshold)`` -- end-to-end

Output schema (under ``reports/graph/``):
- ``edges.parquet``: ``(source, target, edge_type, weight, p_value?, best_lag?)``
- ``centrality.parquet``: ``(node, degree_centrality, eigenvector_centrality, betweenness_centrality)``"""

from autosignalx.graph import build, causality, centrality, correlation  # noqa: F401
from autosignalx.graph.build import build_and_save  # noqa: F401
from autosignalx.graph.causality import granger_edges  # noqa: F401
from autosignalx.graph.centrality import compute_centrality  # noqa: F401
from autosignalx.graph.correlation import partial_correlation_edges  # noqa: F401
