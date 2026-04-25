"""Per-regime feature ranking via tabular classifier + permutation importance.

For each regime, fit a classifier on (features, target_direction) sampled
from that regime's timesteps; then shuffle one feature at a time and
measure the drop in accuracy. The shuffled-vs-baseline accuracy delta is
the feature's importance score.

Originally targeted TabPFN-v2 (Prior Labs) per the project plan, but
TabPFN's >=2.x packages require an interactive browser-based license
acceptance that cannot complete in the reviewer's `make demo` flow or in
CI. We pivoted to ``HistGradientBoostingClassifier`` -- a strong,
license-free, sklearn-native classifier -- to preserve the project's
'reviewers run the demo without provisioning anything' guarantee. The
ranking methodology is identical."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def _permutation_importance(
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Importance = base_accuracy - mean accuracy when feature j is shuffled."""
    rng = np.random.default_rng(seed)
    base_pred = predict_fn(X)
    base_acc = float((base_pred == y).mean())

    n_features = X.shape[1]
    importances = np.zeros((n_repeats, n_features))
    for r in range(n_repeats):
        for j in range(n_features):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            pred = predict_fn(X_perm)
            importances[r, j] = base_acc - (pred == y).mean()
    mean = importances.mean(axis=0)
    std = importances.std(axis=0)
    return mean, std


def rank_features_per_regime(
    features_df: pd.DataFrame,
    regime_labels: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_direction",
    n_samples_per_regime: int = 2000,
    n_repeats: int = 2,
    min_samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """For each regime, fit a classifier and rank features by permutation importance.

    Returns DataFrame with columns
    ``(regime_id, feature, importance, importance_std, n_samples, rank)``.
    Rank 1 = most important within the regime."""
    rl = regime_labels[["timestamp", "regime_id"]].copy()
    df = features_df.merge(rl, on="timestamp", how="inner")

    rows: list[dict] = []
    for regime_id, group in df.groupby("regime_id", observed=True):
        if len(group) < min_samples:
            continue
        if len(group) > n_samples_per_regime:
            group = group.sample(n=n_samples_per_regime, random_state=seed)

        X = group[feature_cols].to_numpy(dtype=np.float32)
        y = group[target_col].to_numpy(dtype=np.int32)

        if len(np.unique(y)) < 2:
            continue

        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=seed
        )
        clf.fit(X, y)
        importances, importances_std = _permutation_importance(
            clf.predict, X, y, n_repeats=n_repeats, seed=seed
        )

        for i, fname in enumerate(feature_cols):
            rows.append(
                {
                    "regime_id": int(regime_id),
                    "feature": fname,
                    "importance": float(importances[i]),
                    "importance_std": float(importances_std[i]),
                    "n_samples": int(len(group)),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank"] = (
            out.groupby("regime_id")["importance"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )
        out = out.sort_values(["regime_id", "rank"]).reset_index(drop=True)
    return out
