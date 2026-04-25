"""GLASSO partial-correlation graph between asset return series.

Partial correlations measure direct statistical relationships between
two assets after controlling for all other assets in the panel -- a
sharper signal than raw Pearson correlations, which conflate direct
links with indirect ones via common factors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def partial_correlation_edges(
    returns: pd.DataFrame,
    threshold: float = 1e-3,
) -> pd.DataFrame:
    """Sparse partial-correlation edges via ``sklearn.covariance.GraphicalLassoCV``.

    Returns an undirected edge list DataFrame with columns
    ``(source, target, edge_type='partial_corr', weight)`` where ``weight`` is
    the partial correlation in ``[-1, 1]``. Edges with absolute weight below
    ``threshold`` are dropped."""
    from sklearn.covariance import GraphicalLassoCV

    X = returns.dropna()
    asset_names = list(X.columns)
    Xa = X.to_numpy()
    if Xa.shape[0] < 50:
        raise ValueError(
            f"Not enough rows for GLASSO ({Xa.shape[0]}); need >= 50"
        )
    Xa = (Xa - Xa.mean(axis=0)) / (Xa.std(axis=0) + 1e-12)

    glasso = GraphicalLassoCV(cv=3)
    glasso.fit(Xa)
    theta = glasso.precision_
    diag = np.sqrt(np.diag(theta))
    pcorr = -theta / np.outer(diag, diag)
    np.fill_diagonal(pcorr, 0.0)

    n = pcorr.shape[0]
    rows: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(pcorr[i, j])
            if abs(w) >= threshold:
                rows.append(
                    {
                        "source": asset_names[i],
                        "target": asset_names[j],
                        "edge_type": "partial_corr",
                        "weight": w,
                    }
                )
    return pd.DataFrame(
        rows, columns=["source", "target", "edge_type", "weight"]
    )
