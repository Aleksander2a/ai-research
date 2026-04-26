"""Cross-session memory consolidation -- the long-horizon memory cell.

At the end of each agent session, an LLM consolidates what just
happened into a Markdown 'lessons learned' document. The doc grows
session by session under ``reports/agent/lessons.md``; the next
session reads the most recent N lessons as additional context, so the
agent's first round of session N is informed by sessions 1..N-1.

This is the long-horizon memory cell that Deeter explicitly asks for:
unbounded growing context, summarized periodically into a structured
form the agent can re-consume."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autosignalx.agent import findings as findings_mod
from autosignalx.agent import ledger as ledger_mod
from autosignalx.agent.llm import LLMProvider, get_provider
from autosignalx.config import settings

LESSONS_DIR = settings.reports_dir / "agent"


def _lessons_path() -> Path:
    LESSONS_DIR.mkdir(parents=True, exist_ok=True)
    return LESSONS_DIR / "lessons.md"


CONSOLIDATOR_SYSTEM = """You are summarizing one session of an AI research
agent's work into a 'lessons learned' note that will be appended to a long-
horizon memory document and re-read at the start of the next session.

Write a Markdown section in this exact structure (be concise; total under 350
words):

## Session <session_id> -- <iso_date>

**What was tried**: 1-3 sentences naming the (regime, asset, method)
combinations the agent explored.

**What worked**: list the promoted findings (by ID, with one-line summary),
or "(none)" if nothing passed the gate.

**What was refuted**: list hypotheses the adjudicator marked as refute, or
"(none)".

**Patterns observed**: 1-2 sentences on the cross-cutting insight (e.g., a
specific macro consistently dominating one regime, an asset pair that
co-moves predictably, etc.).

**Open directions for next session**: 1-3 specific (regime/asset/method)
slices that look promising but weren't fully explored.

Be specific. Cite hypothesis content rather than generic statements."""


def consolidate(
    session_id: str,
    ledger_entries: list[dict[str, Any]] | None = None,
    finding_records: list[dict[str, Any]] | None = None,
    provider: LLMProvider | None = None,
) -> str:
    """Run the consolidator LLM on the session's ledger + findings;
    return the Markdown section."""
    if ledger_entries is None:
        ledger_entries = ledger_mod.load()
    if finding_records is None:
        finding_records = findings_mod.load()
    if provider is None:
        provider = get_provider(role="adjudicator")

    session_findings = [f for f in finding_records if f.get("session_id") == session_id]
    ledger_summary = ledger_mod.summarize_for_prompt(ledger_entries, limit=40)
    findings_summary = json.dumps(
        [
            {
                "id": f.get("id"),
                "method": f.get("method"),
                "filters": f.get("filters"),
                "skill": f.get("evidence", {}).get("skill_vs_baseline"),
                "p_value": f.get("evidence", {}).get("p_value"),
            }
            for f in session_findings
        ],
        indent=2,
        default=str,
    )[:1500]

    user = (
        f"## Session ID\n{session_id}\n\n"
        f"## Date\n{datetime.now(UTC).strftime('%Y-%m-%d')}\n\n"
        f"## Ledger (last 40 entries)\n{ledger_summary}\n\n"
        f"## Promoted findings this session\n{findings_summary}\n\n"
        "Write the Markdown lessons section now."
    )
    raw = provider.chat(
        [
            {"role": "system", "content": CONSOLIDATOR_SYSTEM},
            {"role": "user", "content": user},
        ],
        step="consolidate",
        round=-1,
    )
    return raw.strip()


def append_to_lessons(section: str) -> Path:
    """Append a freshly-consolidated section to ``reports/agent/lessons.md``."""
    path = _lessons_path()
    with path.open("a", encoding="utf-8") as f:
        if path.stat().st_size > 0:
            f.write("\n\n---\n\n")
        f.write(section)
        f.write("\n")
    return path


def load_lessons(max_chars: int = 8000) -> str:
    """Read the lessons document, capped at ``max_chars`` (most recent first
    via tail). Returns empty string if no lessons exist yet."""
    path = _lessons_path()
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    # Tail to roughly the last max_chars; keep section breaks intact
    tail = text[-max_chars:]
    sep_idx = tail.find("\n## Session ")
    if sep_idx >= 0:
        return "(... earlier sessions truncated ...)\n" + tail[sep_idx:]
    return "(... earlier sessions truncated ...)\n" + tail


def consolidate_and_append(
    session_id: str,
    ledger_entries: list[dict[str, Any]] | None = None,
    finding_records: list[dict[str, Any]] | None = None,
    provider: LLMProvider | None = None,
) -> tuple[Path, str]:
    """Convenience: consolidate the session and append the result."""
    section = consolidate(session_id, ledger_entries, finding_records, provider)
    path = append_to_lessons(section)
    return path, section


def clear() -> None:
    p = _lessons_path()
    if p.exists():
        p.unlink()
