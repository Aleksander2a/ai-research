"""Append-only JSONL ledger for agent steps -- the persistent memory cell.

Every agent step (proposed hypothesis, critique, experiment result,
decision) is written as one JSON line. The cockpit and the agent itself
read this file to reconstruct context across sessions."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autosignalx.config import settings

LEDGER_DIR = settings.reports_dir / "agent"


def _ledger_path() -> Path:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    return LEDGER_DIR / "ledger.jsonl"


def append(entry: dict[str, Any]) -> None:
    """Append an entry to the ledger with an ISO timestamp."""
    record = {**entry, "ts": datetime.now(UTC).isoformat(timespec="seconds")}
    with _ledger_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load() -> list[dict[str, Any]]:
    """Read all ledger entries (oldest first)."""
    path = _ledger_path()
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def clear() -> None:
    """Wipe the ledger (used by the CLI for fresh runs)."""
    p = _ledger_path()
    if p.exists():
        p.unlink()


def summarize_for_prompt(entries: Iterable[dict[str, Any]], limit: int = 20) -> str:
    """Compact ledger summary suitable for stuffing into an LLM prompt."""
    items = list(entries)[-limit:]
    if not items:
        return "(empty -- this is the first round)"
    lines = []
    for e in items:
        rd = e.get("round", "?")
        step = e.get("step", "?")
        content = e.get("content", {})
        if isinstance(content, dict):
            content_str = json.dumps(content, default=str)[:300]
        else:
            content_str = str(content)[:300]
        lines.append(f"  round {rd} {step}: {content_str}")
    return "\n".join(lines)
