"""On-disk vector index for the chat corpus.

The index is a dataclass holding parallel arrays: a list of ``Chunk``
records and an ``(N, D)`` ``np.ndarray`` of L2-normalized embeddings.
Persisted as a sibling pair under ``reports/chat/``: ``chunks.jsonl``
(human-inspectable) + ``vectors.npy`` (binary). Top-K retrieval is a
single matmul -- the corpus is small enough that no FAISS/ANN is
warranted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autosignalx.chat.corpus import Chunk, build_corpus, chunks_from_jsonl, chunks_to_jsonl
from autosignalx.chat.embed import EmbeddingProvider
from autosignalx.config import settings


def _normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray
    mode: str
    model: str

    def search(self, query_vec: np.ndarray, k: int = 6) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = query_vec.reshape(-1)
        qn = float(np.linalg.norm(q))
        if qn > 0:
            q = q / qn
        sims = self.vectors @ q
        k = min(k, len(self.chunks))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(self.chunks[int(i)], float(sims[int(i)])) for i in top_idx]


def index_dir() -> Path:
    return settings.reports_dir / "chat"


def build_index(
    reports_dir: Path | None = None,
    force_hashed: bool = False,
) -> Index:
    """Walk artifacts, embed every chunk, persist + return the index."""
    chunks = build_corpus(reports_dir=reports_dir)
    provider = EmbeddingProvider(force_hashed=force_hashed)
    if chunks:
        vectors = provider.embed([c.text for c in chunks])
        vectors = _normalize(vectors)
    else:
        vectors = np.zeros((0, provider.dim), dtype=np.float32)
    out = index_dir()
    out.mkdir(parents=True, exist_ok=True)
    chunks_to_jsonl(chunks, out / "chunks.jsonl")
    np.save(out / "vectors.npy", vectors)
    (out / "meta.json").write_text(
        f'{{"mode": "{provider.mode}", "model": "{provider.model}", '
        f'"n_chunks": {len(chunks)}, "dim": {provider.dim}}}',
        encoding="utf-8",
    )
    return Index(chunks=chunks, vectors=vectors, mode=provider.mode, model=provider.model)


def load_index() -> Index | None:
    out = index_dir()
    cpath = out / "chunks.jsonl"
    vpath = out / "vectors.npy"
    if not cpath.exists() or not vpath.exists():
        return None
    chunks = chunks_from_jsonl(cpath)
    vectors = np.load(vpath)
    meta_path = out / "meta.json"
    mode = "hashed"
    model = "?"
    if meta_path.exists():
        import json as _json

        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        mode = meta.get("mode", mode)
        model = meta.get("model", model)
    return Index(chunks=chunks, vectors=vectors, mode=mode, model=model)
