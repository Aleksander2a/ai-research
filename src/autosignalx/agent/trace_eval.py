"""Per-round trace quality scoring -- LLM-as-judge in the openevals style.

For each agent round, an evaluator LLM reads the round's entries
(propose / theorist / skeptic / experiment / adjudicator / decide) and
scores it on four research-quality rubrics (1-5 scale):

- **clarity**: was the hypothesis specific enough to be tested?
- **novelty**: did this round explore a (regime, asset, method)
  combination not yet seen in the ledger?
- **falsifiability**: was the prediction concrete enough that the
  experiment could in principle refute it?
- **evidence_citing**: did the critique / adjudication cite specific
  ledger / artifact entries (not just generic concerns)?

Scores persist to ``reports/agent/trace_quality.jsonl``. The cockpit's
Agent Console renders the trend over rounds, so reviewers can see
whether the agent's reasoning quality is improving as the session
progresses.

Conceptually this follows ``openevals.create_llm_as_judge`` and
``agentevals``'s trajectory-evaluation pattern; we route the judge
through our existing LLMProvider so live (DeepInfra) and replay modes
both work."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autosignalx.agent.graph import _safe_parse_json
from autosignalx.agent.llm import LLMProvider, get_provider
from autosignalx.config import settings

QUALITY_DIR = settings.reports_dir / "agent"


def _quality_path() -> Path:
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    return QUALITY_DIR / "trace_quality.jsonl"


JUDGE_SYSTEM = """You are an evaluator scoring the quality of one round of an
AI research agent's reasoning. The agent is exploring conditional forecasting
strategies for ETFs across latent market regimes.

Score the round on these four rubrics on a 1-5 integer scale:

- clarity: was the hypothesis specific enough to be tested? (1=vague, 5=razor-sharp)
- novelty: does this round explore a (regime/asset/method) combination not in
  the ledger so far? (1=duplicate; 5=genuinely new direction)
- falsifiability: is the prediction concrete enough that the experiment could
  refute it? (1=unfalsifiable claim; 5=clean go/no-go)
- evidence_citing: do the critique / adjudication cite specific ledger or
  artifact entries (not generic concerns)? (1=hand-waving; 5=specific citation
  of round-N evidence)

Respond with a JSON object exactly matching:

{
  "clarity": <int 1..5>,
  "novelty": <int 1..5>,
  "falsifiability": <int 1..5>,
  "evidence_citing": <int 1..5>,
  "rationale": "one sentence summarizing the round's main weakness or strength"
}"""


def _round_summary(round_entries: list[dict[str, Any]]) -> str:
    lines = []
    for e in round_entries:
        step = e.get("step", "?")
        content = e.get("content", "")
        if isinstance(content, dict):
            content_str = json.dumps(content, default=str)[:400]
        else:
            content_str = str(content)[:400]
        lines.append(f"- {step}: {content_str}")
    return "\n".join(lines)


def score_round(
    round_number: int,
    round_entries: list[dict[str, Any]],
    ledger_summary: str = "",
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    """Run the LLM judge on one round's entries; return the scores dict
    plus the rationale and timestamp. Defaults to the 'critic' role provider."""
    if provider is None:
        provider = get_provider(role="critic")
    user = (
        f"## Round {round_number}\n{_round_summary(round_entries)}\n\n"
        f"## Ledger context (preceding rounds, summarized)\n{ledger_summary[:1500]}\n\n"
        "Respond with the JSON evaluation."
    )
    raw = provider.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        step="trace_eval",
        round=round_number,
    )
    parsed = _safe_parse_json(raw)
    # Coerce to ints where possible
    out = {"round": round_number, "raw": raw[:400] if not parsed else None}
    for k in ("clarity", "novelty", "falsifiability", "evidence_citing"):
        v = parsed.get(k)
        try:
            out[k] = int(v) if v is not None else None
        except (ValueError, TypeError):
            out[k] = None
    out["rationale"] = parsed.get("rationale", "")
    out["ts"] = datetime.now(UTC).isoformat(timespec="seconds")
    return out


def score_session(
    ledger_entries: list[dict[str, Any]],
    session_id: str,
    provider: LLMProvider | None = None,
) -> list[dict[str, Any]]:
    """Score every round in a session. Persists results and returns them."""
    by_round: dict[int, list[dict[str, Any]]] = {}
    for e in ledger_entries:
        rd = int(e.get("round", 0))
        by_round.setdefault(rd, []).append(e)

    scores = []
    running_summary_parts = []
    for rd in sorted(by_round.keys()):
        ledger_summary = "\n".join(running_summary_parts[-10:])
        score = score_round(rd, by_round[rd], ledger_summary=ledger_summary, provider=provider)
        score["session_id"] = session_id
        scores.append(score)
        running_summary_parts.append(f"round {rd}: {by_round[rd][0].get('step', '?')}")

    with _quality_path().open("a", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s, default=str) + "\n")
    return scores


def load() -> list[dict[str, Any]]:
    """Read all persisted trace-quality records."""
    path = _quality_path()
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def clear() -> None:
    p = _quality_path()
    if p.exists():
        p.unlink()
