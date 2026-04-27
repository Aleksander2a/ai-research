"""Phase 8 -- Combinatorial Purged Cross-Validation (Lopez de Prado).

Walk-forward is a single train/test trajectory. CPCV constructs k(k-1)/2
*paths* by combinatorially holding out 2 of N folds at a time, with a
purge buffer between train and test to prevent label-overlap leakage on
overlapping h-step-ahead targets.

Two outputs every consumer needs:

* ``CPCVPath`` -- a single (train_origins, test_origins) split with a
  purged middle.
* ``cpcv_paths(origins, n_folds, k_test, embargo)`` -- yields the full
  set of paths.

Combined with the existing DM gate, this gives an honest distribution
of out-of-sample skill estimates rather than a single number.

Mathematically, with N folds and k=2 test folds per path, you get
C(N, 2) = N(N-1)/2 paths. Each origin is in test exactly (N-1)
times. The mean across paths estimates expected out-of-sample skill;
the std is the true uncertainty around that estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CPCVPath:
    """One CPCV path: train and test origins, with a purge buffer between."""

    train_origins: tuple[pd.Timestamp, ...]
    test_origins: tuple[pd.Timestamp, ...]
    fold_indices: tuple[int, ...]  # which folds are in test


def cpcv_paths(
    origins: list[pd.Timestamp],
    n_folds: int = 6,
    k_test: int = 2,
    embargo: int = 1,
) -> list[CPCVPath]:
    """Generate CPCV paths over a sorted list of forecast origins.

    Args:
        origins: sorted list of unique ``forecast_origin`` timestamps.
        n_folds: total folds to partition origins into.
        k_test: number of test folds per path.
        embargo: number of origins to purge from train at each
            train/test boundary (handles h-step-ahead overlap).

    Returns C(n_folds, k_test) paths."""
    if k_test < 1 or k_test >= n_folds:
        raise ValueError(f"k_test must be in [1, {n_folds - 1}]")
    origins = sorted(set(pd.Timestamp(o) for o in origins))
    if len(origins) < n_folds * 2:
        raise ValueError(
            f"Insufficient origins ({len(origins)}) for {n_folds}-fold CPCV"
        )

    fold_size = len(origins) // n_folds
    fold_bounds: list[tuple[int, int]] = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(origins)
        fold_bounds.append((start, end))

    paths: list[CPCVPath] = []
    for test_folds in combinations(range(n_folds), k_test):
        test_idx_set: set[int] = set()
        for f in test_folds:
            s, e = fold_bounds[f]
            test_idx_set.update(range(s, e))

        # Train = everything not in test, with embargo neighbourhoods purged
        purge_idx_set: set[int] = set()
        for idx in test_idx_set:
            for delta in range(-embargo, embargo + 1):
                p = idx + delta
                if 0 <= p < len(origins):
                    purge_idx_set.add(p)

        train_idx = [i for i in range(len(origins)) if i not in purge_idx_set]
        test_idx = sorted(test_idx_set)
        if not train_idx or not test_idx:
            continue

        paths.append(
            CPCVPath(
                train_origins=tuple(origins[i] for i in train_idx),
                test_origins=tuple(origins[i] for i in test_idx),
                fold_indices=tuple(test_folds),
            )
        )
    return paths


def cpcv_skill_distribution(
    forecasts: pd.DataFrame,
    method: str,
    baseline_method: str = "naive",
    n_folds: int = 6,
    k_test: int = 2,
    embargo: int = 1,
) -> dict:
    """Compute skill-vs-baseline on each CPCV path; return mean/std/distribution."""
    if forecasts.empty:
        return {"n_paths": 0, "skill_mean": float("nan"), "skill_std": float("nan")}
    forecasts = forecasts.copy()
    forecasts["forecast_origin"] = pd.to_datetime(forecasts["forecast_origin"])
    origins = sorted(forecasts["forecast_origin"].unique())
    try:
        paths = cpcv_paths(origins, n_folds=n_folds, k_test=k_test, embargo=embargo)
    except ValueError as e:
        return {"n_paths": 0, "skill_mean": float("nan"), "skill_std": float("nan"), "error": str(e)}

    skills: list[float] = []
    for path in paths:
        test_set = set(path.test_origins)
        sub = forecasts[forecasts["forecast_origin"].isin(test_set)]
        a = sub[sub["method"] == method]
        b = sub[sub["method"] == baseline_method]
        if a.empty or b.empty:
            continue
        keys = ["timestamp", "asset", "forecast_origin"]
        merged = a.merge(b, on=keys, suffixes=("_method", "_baseline"))
        if merged.empty:
            continue
        m_mae = float(np.mean(np.abs(merged["prediction_method"] - merged["target_method"])))
        b_mae = float(np.mean(np.abs(merged["prediction_baseline"] - merged["target_method"])))
        if b_mae <= 0:
            continue
        skills.append(1.0 - m_mae / b_mae)

    if not skills:
        return {"n_paths": len(paths), "skill_mean": float("nan"), "skill_std": float("nan")}
    arr = np.asarray(skills, dtype=float)
    return {
        "n_paths": len(paths),
        "n_evaluated": int(len(arr)),
        "skill_mean": float(np.mean(arr)),
        "skill_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "skill_min": float(np.min(arr)),
        "skill_max": float(np.max(arr)),
        "skill_median": float(np.median(arr)),
        "skills": [float(s) for s in arr],
    }
