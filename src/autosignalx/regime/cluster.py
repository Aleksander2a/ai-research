"""Regime clustering: KMeans on contrastive embeddings (primary) and HMM
on raw features (sanity-check baseline).

KMeans gives crisp, hard assignments; HMM models temporal dynamics
(transitions) and gives probabilistic assignments. Where the two agree,
confidence in the regime structure rises; where they disagree, the
disagreement is itself a research signal."""

from __future__ import annotations

import numpy as np


def kmeans_regimes(
    embeddings: np.ndarray, n_regimes: int = 4, seed: int = 42
) -> tuple[np.ndarray, object]:
    """KMeans on encoder embeddings. Returns ``(labels, fitted_model)``."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km


def hmm_regimes(
    features: np.ndarray, n_regimes: int = 4, seed: int = 42, n_iter: int = 100
) -> np.ndarray:
    """Gaussian HMM on raw market features. Returns one regime label per timestep."""
    from hmmlearn import hmm

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=seed,
    )
    model.fit(features)
    return model.predict(features)
