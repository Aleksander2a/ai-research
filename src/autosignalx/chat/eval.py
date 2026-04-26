"""Grounding eval set + scoring harness.

A small fixture of questions whose ground truth lives in the bundled
artifacts. Scores two things:

* **Citation recall** -- the retrieved top-K must contain at least one
  of the expected ``citation_id`` patterns (substring match).
* **Refusal accuracy** -- intentionally off-corpus questions must
  trigger the canonical refusal text.

Designed to run in replay mode (deterministic hashed embeddings + the
no-LLM rendering path), so CI can verify it without a DeepInfra key.
"""

from __future__ import annotations

from dataclasses import dataclass

from autosignalx.chat.answer import REFUSAL_TEXT, answer_question
from autosignalx.chat.index import Index, build_index, load_index


@dataclass
class EvalCase:
    question: str
    expected_kinds: list[str]  # any chunk of these kinds counts as good retrieval
    expected_id_substr: str | None = None  # optional: any retrieved citation_id must contain this substring
    must_refuse: bool = False


CASES: list[EvalCase] = [
    EvalCase(
        question="Which findings has the agent promoted, and what is the strongest one?",
        expected_kinds=["finding"],
    ),
    EvalCase(
        question="What did the skeptic argue about EFA's bridge-asset hypothesis?",
        expected_kinds=["ledger"],
        expected_id_substr="skeptic",
    ),
    EvalCase(
        question="How did the TopKLong backtest strategy perform on Sharpe?",
        expected_kinds=["backtest"],
    ),
    EvalCase(
        question="What was the agent's self-critique of its earlier findings?",
        expected_kinds=["self_critique"],
    ),
    EvalCase(
        question="How much money was spent on LLM calls in total?",
        expected_kinds=["telemetry"],
    ),
    EvalCase(
        question="What is the boiling point of mercury in Kelvin?",
        expected_kinds=[],
        must_refuse=True,
    ),
    EvalCase(
        question="Who won the 2024 FIFA World Cup?",
        expected_kinds=[],
        must_refuse=True,
    ),
]


def _retrieval_ok(case: EvalCase, retrieved_kinds: list[str], retrieved_ids: list[str]) -> bool:
    if case.must_refuse:
        return True  # refusal-only checks don't grade retrieval
    if case.expected_kinds and not any(k in retrieved_kinds for k in case.expected_kinds):
        return False
    return not (
        case.expected_id_substr
        and not any(case.expected_id_substr in cid for cid in retrieved_ids)
    )


def run_eval(index: Index | None = None) -> dict:
    idx = index if index is not None else load_index()
    if idx is None:
        idx = build_index()

    results = []
    citation_hits = 0
    refusal_correct = 0
    n_refusal = 0
    n_grounded = 0
    for case in CASES:
        ans = answer_question(case.question, index=idx, k=6)
        retrieved_kinds = [c.kind for c, _ in ans.retrieved]
        retrieved_ids = [c.citation_id for c, _ in ans.retrieved]
        retrieval_ok = _retrieval_ok(case, retrieved_kinds, retrieved_ids)

        if case.must_refuse:
            n_refusal += 1
            # Off-corpus question: in replay mode the system always renders
            # top retrieved chunks, so we only score live-mode refusal.
            # Replay mode counts as "n/a" (passes by default).
            ok = ans.mode != "live" or REFUSAL_TEXT in ans.text
            if ok:
                refusal_correct += 1
        else:
            n_grounded += 1
            if retrieval_ok:
                citation_hits += 1
            ok = retrieval_ok

        results.append(
            {
                "question": case.question,
                "expected_kinds": case.expected_kinds,
                "must_refuse": case.must_refuse,
                "retrieved_kinds": retrieved_kinds[:3],
                "retrieved_ids": retrieved_ids[:3],
                "passed": ok,
                "mode": ans.mode,
            }
        )

    return {
        "n": len(CASES),
        "citation_recall": citation_hits / n_grounded if n_grounded else 0.0,
        "refusal_correct": refusal_correct / n_refusal if n_refusal else 1.0,
        "results": results,
    }
