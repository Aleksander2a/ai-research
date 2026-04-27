"""Phase 15 -- Long-horizon coherence eval.

Beyond per-round trace quality, we also score *across rounds and
sessions* the coherence of the agent's research arc. Specifically:

* Did the agent revisit settled questions unnecessarily (low coherence)?
* Did the agent's lessons in earlier sessions drive later proposals (high
  coherence)?
* Did the lineage DAG converge on a useful direction or branch noisily?

We compute several scalar coherence proxies that don't require an LLM:

* **lessons_uptake** -- fraction of proposals in session N that mention
  themes named in lessons.md from sessions <N (substring match on key
  phrases like "regime 3", "TLT", "DXY").
* **lineage_branching_factor** -- mean out-degree per node in the
  lineage DAG; close to 0 means the agent didn't refine; very high
  means it scattered.
* **theme_persistence** -- entropy of the (asset, regime) cells visited
  across rounds; lower entropy = more focused; higher = more exploratory.

These proxies are written to ``reports/agent/coherence.jsonl`` as a
session-level record so the cockpit can plot trends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from autosignalx.config import settings

COHERENCE_PATH = settings.reports_dir / "agent" / "coherence.jsonl"


@dataclass(frozen=True)
class CoherenceRecord:
    session_id: str
    n_rounds: int
    lessons_uptake: float
    lineage_branching_factor: float
    theme_persistence_entropy: float
    coherence_score: float


def _ledger_for_session(ledger: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    return [e for e in ledger if e.get("session_id") == session_id]


def lessons_uptake(
    proposals: list[dict[str, Any]],
    prior_lessons: str,
    minimum_phrase_len: int = 4,
) -> float:
    """Fraction of proposals that include at least one substring from prior lessons.

    Crude but informative: picks out the multi-word noun phrases the lessons doc
    introduces and counts how often the agent mentions them in subsequent rounds."""
    if not proposals or not prior_lessons:
        return 0.0
    phrases = []
    for line in prior_lessons.splitlines():
        words = [w for w in line.strip().split() if len(w) >= minimum_phrase_len]
        for i in range(len(words) - 1):
            phrase = (words[i] + " " + words[i + 1]).lower()
            if phrase.isascii():
                phrases.append(phrase)
    if not phrases:
        return 0.0
    matches = 0
    for p in proposals:
        text = json.dumps(p, default=str).lower()
        if any(phrase in text for phrase in phrases):
            matches += 1
    return float(matches / len(proposals))


def lineage_branching_factor(lineage: dict[str, Any]) -> float:
    nodes = lineage.get("nodes", [])
    edges = lineage.get("edges", [])
    if not nodes:
        return 0.0
    out_degree: dict[str, int] = {}
    for e in edges:
        out_degree[e.get("source", "")] = out_degree.get(e.get("source", ""), 0) + 1
    if not out_degree:
        return 0.0
    return float(np.mean(list(out_degree.values())))


def theme_persistence_entropy(proposals: list[dict[str, Any]]) -> float:
    """Shannon entropy over the (asset, regime) cells visited in proposals."""
    counts: dict[tuple, int] = {}
    for p in proposals:
        params = (p.get("experiment") or {}).get("params") or {}
        key = (params.get("asset"), params.get("regime_id"))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return 0.0
    total = sum(counts.values())
    probs = np.asarray([c / total for c in counts.values()])
    nz = probs[probs > 0]
    return float(-np.sum(nz * np.log(nz)))


def score_session(
    session_id: str,
    ledger: list[dict[str, Any]] | None = None,
    prior_lessons: str | None = None,
) -> CoherenceRecord:
    """Compute coherence proxies for a single session."""
    from autosignalx.agent import ledger as ledger_mod
    from autosignalx.agent import lineage as lineage_mod
    from autosignalx.agent import memory as memory_mod

    if ledger is None:
        ledger = ledger_mod.load()
    session_entries = _ledger_for_session(ledger, session_id)
    proposals = [
        e.get("content") or {}
        for e in session_entries
        if e.get("step") in ("propose", "theorist") and isinstance(e.get("content"), dict)
    ]
    rounds = sorted({int(e.get("round", 0)) for e in session_entries})
    n_rounds = len(rounds)

    if prior_lessons is None:
        prior_lessons = memory_mod.load_lessons(max_chars=8000)

    lup = lessons_uptake(proposals, prior_lessons)
    lin = lineage_mod.build_lineage(session_entries)
    bf = lineage_branching_factor(lin)
    ent = theme_persistence_entropy(proposals)

    # Composite score: high uptake, moderate branching (~1), moderate entropy
    # are all good. Penalise both 0 and >>1 branching factor.
    bf_penalty = abs(bf - 1.0) if bf > 0 else 1.0
    score = float(0.5 * lup + 0.3 * (1.0 / (1.0 + bf_penalty)) + 0.2 * min(ent, 2.0) / 2.0)

    return CoherenceRecord(
        session_id=session_id,
        n_rounds=n_rounds,
        lessons_uptake=lup,
        lineage_branching_factor=bf,
        theme_persistence_entropy=ent,
        coherence_score=score,
    )


def append_coherence(rec: CoherenceRecord, path: Path | None = None) -> None:
    p = path or COHERENCE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**rec.__dict__, "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds")}
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def load_coherence(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or COHERENCE_PATH
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
