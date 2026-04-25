"""Granger causality between asset return series.

For each ordered pair ``(X, Y)``, we test the hypothesis that lagged
values of X help predict Y beyond Y's own lags. The minimum p-value
across lags ``[1, max_lag]`` is taken; pairs below the threshold are
emitted as directed edges with weight ``-log10(p)``."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def granger_edges(
    returns: pd.DataFrame,
    max_lag: int = 5,
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """Directed Granger-causality edges below ``p_threshold``.

    Returns DataFrame with columns
    ``(source, target, edge_type='granger', weight=-log10(p), p_value, best_lag)``.
    ``source`` Granger-causes ``target``; weight is higher = stronger
    statistical evidence."""
    from statsmodels.tsa.stattools import grangercausalitytests

    X = returns.dropna()
    asset_names = list(X.columns)
    rows: list[dict] = []
    for src in asset_names:
        for tgt in asset_names:
            if src == tgt:
                continue
            # statsmodels convention: data[:, 0] is the target, data[:, 1] is the predictor
            data = X[[tgt, src]].to_numpy()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                p_values = {
                    lag: result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)
                }
            except Exception:  # noqa: BLE001
                continue
            best_lag = min(p_values, key=p_values.get)
            min_p = p_values[best_lag]
            if min_p < p_threshold:
                rows.append(
                    {
                        "source": src,
                        "target": tgt,
                        "edge_type": "granger",
                        "weight": float(-np.log10(min_p + 1e-12)),
                        "p_value": float(min_p),
                        "best_lag": int(best_lag),
                    }
                )
    return pd.DataFrame(
        rows, columns=["source", "target", "edge_type", "weight", "p_value", "best_lag"]
    )
