"""Grounded answer generation for the chat panel.

Pipeline: embed the question -> top-K retrieve from the index -> ask
the chat LLM (or replay) with a strict cite-or-refuse system prompt.
The returned ``ChatAnswer`` carries the citation list so the UI can
render artifact links next to the answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from autosignalx.chat.corpus import Chunk
from autosignalx.chat.embed import EmbeddingProvider
from autosignalx.chat.index import Index, load_index

SYSTEM_PROMPT = (
    "You are AutoSignal-X's research assistant. You answer questions strictly "
    "from the provided EVIDENCE chunks, each tagged with a [citation_id]. "
    "Rules:\n"
    "1. Every factual claim must be followed by one or more [citation_id] in "
    "square brackets, copied verbatim from the EVIDENCE section.\n"
    "2. If the evidence does not support an answer, say exactly: "
    "'I don't have evidence for that in the run artifacts.' and stop.\n"
    "3. Do NOT speculate, extrapolate, or invent numbers, methods, or "
    "asset/regime details that aren't in the EVIDENCE.\n"
    "4. Keep answers concise (3-6 sentences)."
)

REFUSAL_TEXT = "I don't have evidence for that in the run artifacts."


@dataclass
class ChatAnswer:
    text: str
    citations: list[str]
    retrieved: list[tuple[Chunk, float]]
    mode: str


def _format_evidence(retrieved: list[tuple[Chunk, float]]) -> str:
    lines = []
    for c, score in retrieved:
        lines.append(f"[{c.citation_id}] (score={score:.3f}, kind={c.kind})\n{c.text}")
    return "\n\n".join(lines)


_CITE_RE = re.compile(r"\[([a-zA-Z_]+:[^\]\s]+)\]")


def _extract_citations(text: str, valid: set[str]) -> list[str]:
    seen: list[str] = []
    for m in _CITE_RE.finditer(text):
        cid = m.group(1)
        if cid in valid and cid not in seen:
            seen.append(cid)
    return seen


def _replay_answer(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    """Deterministic, no-LLM answer for replay mode.

    Returns a short rendering of the top retrieved chunks with their
    citations -- not a true natural-language answer, but always grounded
    and reproducible without a key."""
    if not retrieved:
        return REFUSAL_TEXT
    head = retrieved[0]
    parts = [
        f"Top evidence for: \"{question.strip()[:200]}\"",
        "",
        f"- {head[0].text} [{head[0].citation_id}]",
    ]
    for c, _ in retrieved[1:4]:
        parts.append(f"- {c.text} [{c.citation_id}]")
    return "\n".join(parts)


def answer_question(
    question: str,
    index: Index | None = None,
    k: int = 6,
    record_replay: bool = False,
) -> ChatAnswer:
    """Run the full retrieve+generate pipeline for one question."""
    idx = index if index is not None else load_index()
    if idx is None or not idx.chunks:
        return ChatAnswer(
            text="The chat index is empty. Run `autosignalx chat index` first.",
            citations=[],
            retrieved=[],
            mode="empty",
        )

    embedder = EmbeddingProvider(model=idx.model, force_hashed=(idx.mode == "hashed"))
    qvec = embedder.embed([question])[0]
    retrieved = idx.search(qvec, k=k)
    valid_ids = {c.citation_id for c, _ in retrieved}

    from autosignalx.agent.llm import get_provider
    from autosignalx.config import settings

    if settings.use_replay or not settings.deepinfra_api_key:
        text = _replay_answer(question, retrieved)
        citations = _extract_citations(text, valid_ids)
        return ChatAnswer(text=text, citations=citations, retrieved=retrieved, mode="replay")

    provider = get_provider(record_replay=record_replay, role="chat")
    user_msg = (
        f"## EVIDENCE\n{_format_evidence(retrieved)}\n\n"
        f"## QUESTION\n{question.strip()}\n\n"
        "Answer in 3-6 sentences. Every factual claim must end with a "
        "[citation_id] copied from EVIDENCE."
    )
    try:
        text = provider.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            step="chat_rag",
            round=-1,
        )
    except Exception as e:  # noqa: BLE001
        return ChatAnswer(
            text=f"LLM call failed: {e}",
            citations=[],
            retrieved=retrieved,
            mode="error",
        )
    citations = _extract_citations(text, valid_ids)
    if not citations and text.strip() and REFUSAL_TEXT not in text:
        # Model produced an unsourced claim -- enforce refusal.
        text = REFUSAL_TEXT + "\n\n_(Model returned an answer without valid citations.)_"
    return ChatAnswer(text=text, citations=citations, retrieved=retrieved, mode="live")


def answer_to_jsonable(a: ChatAnswer) -> dict:
    return {
        "text": a.text,
        "citations": a.citations,
        "retrieved": [
            {"citation_id": c.citation_id, "kind": c.kind, "score": s}
            for c, s in a.retrieved
        ],
        "mode": a.mode,
    }


def jsonable_dumps(a: ChatAnswer) -> str:
    return json.dumps(answer_to_jsonable(a), default=str)
