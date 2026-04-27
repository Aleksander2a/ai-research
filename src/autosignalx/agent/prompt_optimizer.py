"""Phase 15 -- Prompt versioning and quality tracking.

Treat each role's system prompt as a versioned artifact. When a prompt
is edited, append the new version to ``reports/agent/prompts/<role>.jsonl``
with a content hash and the session(s) it was used in. Each session's
trace quality + finding-replication rate is then attributable to a
specific prompt version.

A simple "best prompt" picker reads the prompt history + the per-session
trace_quality.jsonl + findings.jsonl and reports which version produced
the highest mean clarity / falsifiability / replication.

This is the lightweight, no-extra-deps cousin of full DSPy / TextGrad
optimization. It enables:

* See which Theorist prompt produced the highest-quality reasoning.
* Roll back to a prompt version that worked better.
* Run a fixed prompt for N sessions, then compare versions.

A future iteration can wire this into an actual optimizer that mutates
prompts and runs the agent against a fixed eval set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autosignalx.config import settings

PROMPTS_DIR = settings.reports_dir / "agent" / "prompts"


@dataclass(frozen=True)
class PromptVersion:
    role: str
    version_id: str
    text: str
    registered_at: str


def _path_for_role(role: str) -> Path:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROMPTS_DIR / f"{role}.jsonl"


def _version_id(text: str) -> str:
    return "v_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def register_prompt(role: str, text: str, path: Path | None = None) -> PromptVersion:
    """Append a new version of a role's prompt. Idempotent on hash."""
    p = path or _path_for_role(role)
    p.parent.mkdir(parents=True, exist_ok=True)
    vid = _version_id(text)
    existing = load_versions(role, path=path)
    if any(v.get("version_id") == vid for v in existing):
        return PromptVersion(
            role=role, version_id=vid, text=text, registered_at=existing[-1].get("registered_at", "")
        )
    rec = {
        "role": role,
        "version_id": vid,
        "text": text,
        "registered_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return PromptVersion(role=role, version_id=vid, text=text, registered_at=rec["registered_at"])


def load_versions(role: str, path: Path | None = None) -> list[dict[str, Any]]:
    p = path or _path_for_role(role)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
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


def score_versions(
    role: str,
    trace_quality: list[dict[str, Any]],
    sessions_per_version: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate trace-quality scores per prompt version.

    Args:
        trace_quality: rows from trace_quality.jsonl with ``round`` and
            scoring keys (clarity / novelty / falsifiability / evidence_citing).
        sessions_per_version: mapping ``version_id -> [session_id, ...]``;
            if not provided we treat the most-recent version as covering
            every session in trace_quality (so the report is informational
            rather than discriminative until the user starts versioning).
    """
    versions = load_versions(role)
    if not versions:
        return []
    if sessions_per_version is None:
        # Naive: assume the most-recent version covers every session.
        sessions_per_version = {versions[-1]["version_id"]: list({
            s.get("session_id") for s in trace_quality if s.get("session_id")
        })}

    def _avg(rows: list[dict[str, Any]], field: str) -> float | None:
        vals = [r.get(field) for r in rows if isinstance(r.get(field), (int, float))]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    out: list[dict[str, Any]] = []
    for v in versions:
        vid = v["version_id"]
        sids = set(sessions_per_version.get(vid, []))
        rows = [r for r in trace_quality if r.get("session_id") in sids]
        if not rows:
            continue
        out.append({
            "version_id": vid,
            "n_sessions": len(sids),
            "n_rounds": len(rows),
            "avg_clarity": _avg(rows, "clarity"),
            "avg_novelty": _avg(rows, "novelty"),
            "avg_falsifiability": _avg(rows, "falsifiability"),
            "avg_evidence_citing": _avg(rows, "evidence_citing"),
        })
    return out


def best_version(role: str, trace_quality: list[dict[str, Any]]) -> str | None:
    """Pick the version with highest geometric mean across quality rubrics."""
    scored = score_versions(role, trace_quality)
    if not scored:
        return None
    def _score(s: dict[str, Any]) -> float:
        keys = ("avg_clarity", "avg_novelty", "avg_falsifiability", "avg_evidence_citing")
        vals = [s[k] for k in keys if s.get(k) is not None]
        if not vals:
            return -1.0
        prod = 1.0
        for v in vals:
            prod *= max(v, 1e-3)
        return float(prod ** (1.0 / len(vals)))
    return max(scored, key=_score)["version_id"]
