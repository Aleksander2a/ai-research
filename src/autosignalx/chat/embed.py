"""Embedding provider for the chat layer.

Live mode calls DeepInfra's OpenAI-compatible ``/embeddings`` endpoint
with ``BAAI/bge-large-en-v1.5`` (1024-dim) and caches each text by
content hash on disk. Replay/no-key mode falls back to a deterministic,
dependency-free hashed-bag embedding so the panel still works without
network access.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from autosignalx.config import settings

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
HASHED_DIM = 256
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _hashed_embed_one(text: str, dim: int = HASHED_DIM) -> np.ndarray:
    """Deterministic hashed-bag embedding -- works without any model."""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _TOKEN_RE.findall(text.lower()):
        h = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] & 1) == 0 else -1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


def hashed_embed(texts: list[str], dim: int = HASHED_DIM) -> np.ndarray:
    return np.vstack([_hashed_embed_one(t, dim) for t in texts]) if texts else np.zeros((0, dim), dtype=np.float32)


class EmbeddingProvider:
    """Routes to DeepInfra in live mode, hashed-bag in replay mode.

    Caches every (model, text) pair by content hash under
    ``reports/agent/embed_cache/`` so re-indexing is free and
    deterministic across runs."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
        force_hashed: bool = False,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir or (settings.reports_dir / "agent" / "embed_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_hashed = force_hashed or settings.use_replay or not settings.deepinfra_api_key
        self.mode = "hashed" if self.force_hashed else "live"
        self.dim = HASHED_DIM if self.force_hashed else 1024

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256(f"{self.model}|{text}".encode()).hexdigest()[:24]
        return f"{self.mode}_{h}"

    def _read_cache(self, key: str) -> np.ndarray | None:
        p = self.cache_dir / f"{key}.json"
        if not p.exists():
            return None
        try:
            arr = json.loads(p.read_text(encoding="utf-8"))
            return np.asarray(arr, dtype=np.float32)
        except Exception:  # noqa: BLE001
            return None

    def _write_cache(self, key: str, vec: np.ndarray) -> None:
        p = self.cache_dir / f"{key}.json"
        p.write_text(json.dumps(vec.tolist()), encoding="utf-8")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.force_hashed:
            return hashed_embed(texts, dim=HASHED_DIM)

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        to_call: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            cached = self._read_cache(self._cache_key(t))
            if cached is not None and cached.shape[0] == self.dim:
                out[i] = cached
            else:
                to_call.append((i, t))

        if to_call:
            try:
                from openai import OpenAI

                client = OpenAI(
                    api_key=settings.deepinfra_api_key,
                    base_url=settings.deepinfra_base_url,
                )
                # Batch the API call.
                resp = client.embeddings.create(
                    model=self.model,
                    input=[t for _, t in to_call],
                )
                for (i, t), item in zip(to_call, resp.data, strict=False):
                    vec = np.asarray(item.embedding, dtype=np.float32)
                    out[i] = vec
                    self._write_cache(self._cache_key(t), vec)
            except Exception:  # noqa: BLE001
                # Live call failed -- fall back to hashed for the missing rows
                # so the panel never hard-fails on a transient network issue.
                for i, t in to_call:
                    out[i] = _hashed_embed_one(t, dim=self.dim) if self.dim == HASHED_DIM else np.pad(
                        _hashed_embed_one(t, dim=HASHED_DIM), (0, self.dim - HASHED_DIM)
                    )
        return out
