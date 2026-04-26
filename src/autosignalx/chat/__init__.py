"""Phase 3: conversational explainability over the run corpus.

A small RAG stack: chunk on-disk artifacts into citable units, embed them
(DeepInfra in live mode, deterministic hash-based fallback in replay
mode), retrieve top-K by cosine similarity, and ask the chat LLM to
answer questions citing the retrieved chunk IDs verbatim. The "cite or
refuse" prompt + the deterministic replay path keep the panel honest
without a DeepInfra key.
"""

from autosignalx.chat.answer import answer_question
from autosignalx.chat.corpus import Chunk, build_corpus
from autosignalx.chat.index import Index, build_index, load_index

__all__ = [
    "Chunk",
    "Index",
    "answer_question",
    "build_corpus",
    "build_index",
    "load_index",
]
