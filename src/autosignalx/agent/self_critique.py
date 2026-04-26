"""Self-critique -- the agent grades its own past hypotheses.

After enough sessions accumulate, the agent's own claims may have been
either reinforced or undermined by later evidence. Self-critique runs
an LLM judge over each promoted finding, asking 'given the current
state of the ledger and findings store, is this claim still well-
supported, weakened, or refuted?'

Each entry records ``(finding_id, current_state, rationale, ts)``.
The cockpit Self-Critique panel renders the verdict per finding so
reviewers see the agent's calibration over time."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autosignalx.agent import findings as findings_mod
from autosignalx.agent import ledger as ledger_mod
from autosignalx.agent.graph import _safe_parse_json
from autosignalx.agent.llm import LLMProvider, get_provider
from autosignalx.config import settings

CRITIQUE_DIR = settings.reports_dir / "agent"


def _critique_path() -> Path:
    CRITIQUE_DIR.mkdir(parents=True, exist_ok=True)
    return CRITIQUE_DIR / "self_critique.jsonl"


SELF_CRITIQUE_SYSTEM = """You are reviewing one of your own (the AI agent's)
past promoted findings against the current state of evidence in the ledger.

Given:
- The original finding (hypothesis, method, filters, evidence at time of promotion)
- A summary of subsequent ledger activity and other promoted findings

Decide whether the finding is now:
- 'reinforced' (later evidence further supports it; replications, related findings)
- 'unchanged' (no new evidence either way)
- 'weakened' (some related evidence cuts against it but not enough to refute)
- 'refuted' (subsequent evidence or refute verdicts contradict it)

Respond with a JSON object exactly matching:

{
  "current_state": "reinforced" | "unchanged" | "weakened" | "refuted",
  "rationale": "one or two sentences citing specific later evidence"
}"""


def critique_finding(
    finding: dict[str, Any],
    ledger_summary: str,
    other_findings_summary: str,
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    """Run the self-critique LLM over a single finding. Returns the
    record (also persisted)."""
    if provider is None:
        provider = get_provider(role="adjudicator")
    user = (
        f"## Original finding\n```json\n{json.dumps(finding, indent=2, default=str)[:1500]}\n```\n\n"
        f"## Subsequent ledger activity\n{ledger_summary[:1500]}\n\n"
        f"## Other promoted findings\n{other_findings_summary[:1500]}\n\n"
        "Respond with the JSON verdict."
    )
    raw = provider.chat(
        [
            {"role": "system", "content": SELF_CRITIQUE_SYSTEM},
            {"role": "user", "content": user},
        ],
        step="self_critique",
        round=-1,
    )
    parsed = _safe_parse_json(raw)
    record = {
        "finding_id": finding.get("id"),
        "current_state": parsed.get("current_state", "unchanged"),
        "rationale": parsed.get("rationale", ""),
        "raw": raw[:300] if not parsed else None,
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with _critique_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def critique_all_findings(provider: LLMProvider | None = None) -> list[dict[str, Any]]:
    """Run self-critique over every promoted finding currently in the store."""
    findings = findings_mod.load()
    ledger_entries = ledger_mod.load()
    ledger_summary = ledger_mod.summarize_for_prompt(ledger_entries, limit=30)
    other_summary = json.dumps(
        [
            {"id": f.get("id"), "method": f.get("method"), "filters": f.get("filters")}
            for f in findings
        ],
        indent=2,
        default=str,
    )[:1200]
    out = []
    for f in findings:
        out.append(critique_finding(f, ledger_summary, other_summary, provider))
    return out


def load() -> list[dict[str, Any]]:
    path = _critique_path()
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
    p = _critique_path()
    if p.exists():
        p.unlink()
