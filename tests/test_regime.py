"""Tests for the regime layer: encoder shape contract and clustering basics.

Heavy training is exercised by the actual `regime fit` CLI run during
verification; these unit tests use tiny synthetic data and short runs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autosignalx.regime import cluster, encoder


def test_make_windows_shape() -> None:
    features = np.zeros((100, 4), dtype=np.float32)
    windows = encoder.make_windows(features, window_days=10)
    assert windows.shape == (91, 4, 10)


def test_make_windows_too_short_raises() -> None:
    features = np.zeros((5, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds available timesteps"):
        encoder.make_windows(features, window_days=10)


def test_encoder_forward_shape() -> None:
    model = encoder.RegimeEncoder(n_features=4, embedding_dim=16, window_days=20)
    x = torch.zeros(8, 4, 20)
    out = model(x)
    assert out.shape == (8, 16)


def test_train_encoder_runs_and_returns_embeddings() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3)).astype(np.float32)
    enc, embeddings = encoder.train_encoder(
        features,
        embedding_dim=8,
        window_days=20,
        epochs=2,
        batch_size=16,
        seed=0,
    )
    assert isinstance(enc, encoder.RegimeEncoder)
    assert embeddings.shape == (181, 8)
    assert np.isfinite(embeddings).all()


def test_kmeans_regimes_returns_labels_per_input() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(200, 8))
    labels, model = cluster.kmeans_regimes(embeddings, n_regimes=3, seed=0)
    assert labels.shape == (200,)
    assert set(labels.tolist()).issubset({0, 1, 2})
    assert model is not None


def test_hmm_regimes_returns_labels_per_input() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 4))
    labels = cluster.hmm_regimes(features, n_regimes=3, seed=0, n_iter=5)
    assert labels.shape == (200,)
    assert set(labels.tolist()).issubset({0, 1, 2})
