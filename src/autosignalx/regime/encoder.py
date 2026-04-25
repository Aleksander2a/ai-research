"""Contrastive 1D-CNN encoder for time-series windows.

A small PyTorch model that learns embeddings from rolling windows of
market features. Trained via triplet loss with positive=adjacent window,
negative=distant window. Embeddings feed into KMeans for regime clustering."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RegimeEncoder(nn.Module):
    """Tiny 1D-CNN: 2 conv blocks + global pool + linear projection."""

    def __init__(self, n_features: int, embedding_dim: int, window_days: int):
        super().__init__()
        self.window_days = window_days
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape: (batch, n_features, window_days)."""
        return self.net(x)


def make_windows(features: np.ndarray, window_days: int) -> np.ndarray:
    """Sliding window: features (T, n_features) -> windows (T - window + 1, n_features, window).

    Window i covers timesteps [i, i + window_days - 1] inclusive."""
    n_timesteps, n_features = features.shape
    n_windows = n_timesteps - window_days + 1
    if n_windows <= 0:
        raise ValueError(
            f"window_days ({window_days}) exceeds available timesteps ({n_timesteps})"
        )
    out = np.empty((n_windows, n_features, window_days), dtype=np.float32)
    for i in range(n_windows):
        out[i] = features[i : i + window_days].T
    return out


def train_encoder(
    features: np.ndarray,
    embedding_dim: int = 16,
    window_days: int = 60,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    margin: float = 1.0,
    pos_offset_max: int = 3,
    neg_offset_min: int = 60,
    seed: int = 42,
) -> tuple[RegimeEncoder, np.ndarray]:
    """Train the contrastive encoder via triplet loss; return ``(encoder, embeddings)``.

    embeddings has shape ``(n_windows, embedding_dim)`` with one row per
    window in temporal order."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    windows = make_windows(features, window_days)
    n_windows = len(windows)
    n_features = features.shape[1]

    encoder = RegimeEncoder(
        n_features=n_features,
        embedding_dim=embedding_dim,
        window_days=window_days,
    )
    optim = torch.optim.Adam(encoder.parameters(), lr=lr)
    triplet_fn = nn.TripletMarginLoss(margin=margin)

    encoder.train()
    for _ in range(epochs):
        perm = rng.permutation(n_windows)
        for i in range(0, n_windows, batch_size):
            anchor_idx = perm[i : i + batch_size]
            valid_mask = (anchor_idx >= pos_offset_max) & (
                anchor_idx + pos_offset_max < n_windows
            )
            anchor_idx = anchor_idx[valid_mask]
            if len(anchor_idx) == 0:
                continue
            pos_offset = rng.integers(1, pos_offset_max + 1, size=len(anchor_idx))
            pos_offset *= rng.choice([-1, 1], size=len(anchor_idx))
            pos_idx = anchor_idx + pos_offset
            neg_offset_high = max(neg_offset_min + 1, n_windows - neg_offset_min)
            neg_idx = (
                anchor_idx + rng.integers(neg_offset_min, neg_offset_high, size=len(anchor_idx))
            ) % n_windows

            a = torch.from_numpy(windows[anchor_idx])
            p = torch.from_numpy(windows[pos_idx])
            n = torch.from_numpy(windows[neg_idx])

            ea = encoder(a)
            ep = encoder(p)
            en = encoder(n)
            loss = triplet_fn(ea, ep, en)

            optim.zero_grad()
            loss.backward()
            optim.step()

    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(torch.from_numpy(windows)).cpu().numpy()
    return encoder, embeddings
