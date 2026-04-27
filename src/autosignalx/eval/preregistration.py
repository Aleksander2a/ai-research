"""Phase 8 -- Pre-registration ledger.

Each hypothesis the agent (or a human) wants to test must be hash-committed
*before* it runs. The pre-registration record fixes:

* the hypothesis (natural language)
* the experiment spec (method/asset/regime/baseline/horizon)
* the decision rule (p_threshold, skill_threshold, ci_must_be_positive)
* the predicted effect size (so we can score calibration later)
* the falsifiability statement (what evidence would refute the
  hypothesis)

The hash uniquely identifies the registration. When the experiment
runs, the result is linked back to the registration; reviewers can
audit the *open registrations* (still untested) vs *resolved
registrations* (tested + verdict) without re-running the agent.

Stored under ``reports/agent/preregistrations.jsonl``. The file is
append-only; resolved status is a separate record so we never rewrite
history.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autosignalx.config import settings

PREREG_PATH = settings.reports_dir / "agent" / "preregistrations.jsonl"
RESOLUTION_PATH = settings.reports_dir / "agent" / "preregistration_resolutions.jsonl"


@dataclass
class PreRegistration:
    hypothesis: str
    method: str
    baseline: str
    filters: dict[str, Any]
    decision_rule: dict[str, Any]
    predicted_effect: dict[str, Any]
    falsifier: str
    registered_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    session_id: str | None = None
    round: int | None = None
    proposer_role: str | None = None

    def hash(self) -> str:
        payload = json.dumps(
            {
                "hypothesis": self.hypothesis,
                "method": self.method,
                "baseline": self.baseline,
                "filters": self.filters,
                "decision_rule": self.decision_rule,
                "falsifier": self.falsifier,
            },
            sort_keys=True,
            default=str,
        )
        return "p_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.hash(), **self.__dict__}


def register(prereg: PreRegistration, path: Path | None = None) -> dict[str, Any]:
    """Append a pre-registration to the ledger. Idempotent on hash."""
    p = path or PREREG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = prereg.to_dict()
    existing_ids = {r["id"] for r in load(p)}
    if rec["id"] in existing_ids:
        return rec
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def resolve(
    prereg_id: str,
    promoted: bool,
    evidence: dict[str, Any],
    notes: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    """Append a resolution record (does not mutate the registration itself)."""
    p = path or RESOLUTION_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "preregistration_id": prereg_id,
        "promoted": bool(promoted),
        "evidence": evidence,
        "notes": notes,
        "resolved_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def load(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or PREREG_PATH
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


def load_resolutions(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or RESOLUTION_PATH
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


def trial_count(path: Path | None = None) -> int:
    """Total registered trials. Used as ``n_trials`` for Deflated Sharpe."""
    return len(load(path))


def open_count(path: Path | None = None, resolutions_path: Path | None = None) -> int:
    """Registered but not yet resolved -- the agent's open hypotheses."""
    regs = load(path)
    res = {r.get("preregistration_id") for r in load_resolutions(resolutions_path)}
    return sum(1 for r in regs if r.get("id") not in res)


def from_hypothesis_dict(
    h: dict[str, Any],
    session_id: str | None = None,
    round: int | None = None,
    proposer_role: str | None = None,
) -> PreRegistration:
    """Best-effort conversion from the agent's hypothesis JSON to a PreReg.

    Default decision rule mirrors ``eval.significance.is_promotable``:
    p<0.05, skill>0, bootstrap CI strictly above zero. Override per-call
    by passing a richer hypothesis dict with explicit ``decision_rule``
    and ``predicted_effect`` keys."""
    exp = h.get("experiment") or {}
    params = exp.get("params") or {}
    spec = (params.get("spec") or {}) if isinstance(params, dict) else {}
    method = (
        params.get("method")
        or spec.get("name")
        or h.get("method")
        or "unknown"
    )
    baseline = h.get("baseline") or "naive"
    filters = {
        "asset": params.get("asset") or (spec.get("asset_subset") or [None])[0],
        "regime_id": params.get("regime_id"),
    }
    decision_rule = h.get("decision_rule") or {
        "p_threshold": 0.05,
        "skill_threshold": 0.0,
        "ci_must_be_positive": True,
    }
    predicted_effect = h.get("predicted_effect") or {
        "expected_skill": None,
        "expected_p": None,
    }
    falsifier = h.get("falsifier") or (
        f"Refuted iff DM p>=0.05 OR skill<=0 OR bootstrap CI includes 0 "
        f"on the {filters} slice."
    )
    return PreRegistration(
        hypothesis=h.get("hypothesis", ""),
        method=str(method),
        baseline=str(baseline),
        filters=filters,
        decision_rule=decision_rule,
        predicted_effect=predicted_effect,
        falsifier=falsifier,
        session_id=session_id,
        round=round,
        proposer_role=proposer_role,
    )
