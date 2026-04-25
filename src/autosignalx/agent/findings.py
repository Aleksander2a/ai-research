"""Promoted findings store -- separate from the raw experiment ledger.

The raw ledger (``reports/agent/ledger.jsonl``) contains every step the
agent took: hypotheses, experiments, critiques, decisions. The findings
store (``reports/agent/findings.jsonl``) contains only **promoted**
findings: hypotheses that passed the statistical promotion gate
(``eval.significance.is_promotable``).

Each finding carries full provenance: the originating hypothesis, the
DM/bootstrap evidence, the agent's confidence statement, the session
ID, and parent-hypothesis IDs (used by the lineage layer in Iter 14)."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autosignalx.config import settings

FINDINGS_DIR = settings.reports_dir / "agent"


def _findings_path() -> Path:
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return FINDINGS_DIR / "findings.jsonl"


def _finding_id(content: dict[str, Any]) -> str:
    """Deterministic short ID derived from content."""
    payload = json.dumps(
        {
            "hypothesis": content.get("hypothesis"),
            "method": content.get("method"),
            "filters": content.get("filters"),
        },
        sort_keys=True,
        default=str,
    )
    return "f_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def promote(
    hypothesis: str,
    method: str,
    filters: dict[str, Any],
    evidence: dict[str, Any],
    agent_confidence: str,
    round: int,
    session_id: str,
    parent_hypothesis_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Append a promoted finding to ``reports/agent/findings.jsonl``.

    Idempotent on (hypothesis, method, filters): re-promoting the same
    finding bumps its replication_count rather than duplicating."""
    record_core = {
        "hypothesis": hypothesis,
        "method": method,
        "filters": filters,
        "evidence": evidence,
        "agent_confidence": agent_confidence,
    }
    fid = _finding_id(record_core)
    existing = load()
    for prior in existing:
        if prior.get("id") == fid:
            return _update_replication(fid, round=round, session_id=session_id)
    record = {
        **record_core,
        "id": fid,
        "session_id": session_id,
        "round": round,
        "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "parent_hypothesis_ids": parent_hypothesis_ids or [],
        "replication_count": 1,
        "replications": [{"session_id": session_id, "round": round}],
    }
    with _findings_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def _update_replication(fid: str, round: int, session_id: str) -> dict[str, Any]:
    """Bump replication count for an existing finding by rewriting the file."""
    rows = load()
    out = []
    updated_record = None
    for r in rows:
        if r.get("id") == fid:
            r.setdefault("replications", []).append({"session_id": session_id, "round": round})
            r["replication_count"] = len(r["replications"])
            updated_record = r
        out.append(r)
    if updated_record is None:
        return {}
    with _findings_path().open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, default=str) + "\n")
    return updated_record


def load() -> list[dict[str, Any]]:
    """Read all promoted findings (oldest first)."""
    path = _findings_path()
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
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
    p = _findings_path()
    if p.exists():
        p.unlink()


def make_session_id() -> str:
    """Short, sortable session ID prefixed with ISO date."""
    today = datetime.now(UTC).strftime("%Y%m%d")
    return f"{today}-{uuid.uuid4().hex[:8]}"
