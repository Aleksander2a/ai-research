"""Phase 14 -- Pre-registration verifier.

Before any experiment runs, the Verifier checks that the agent's
hypothesis carries a complete pre-registration: a decision rule,
a predicted effect size, a falsifiability statement. Hypotheses
without these are downgraded (not blocked) and flagged in the ledger.

A complete pre-registration enables Phase-15 calibration: at session
end the agent's predicted effect can be compared against the observed
effect, producing a per-role calibration plot that scores the
Theorist's prior accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerifierResult:
    ok: bool
    missing: list[str]
    downgrades: list[str]


REQUIRED_FIELDS = ("decision_rule", "falsifier")
SOFT_FIELDS = ("predicted_effect",)


def verify_hypothesis(h: dict[str, Any]) -> VerifierResult:
    """Check that a hypothesis dict contains a real pre-registration block.

    Required: decision_rule, falsifier. Soft: predicted_effect (its
    absence triggers a calibration-disabled flag, not a hard reject)."""
    missing: list[str] = []
    downgrades: list[str] = []
    for k in REQUIRED_FIELDS:
        v = h.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(k)
    for k in SOFT_FIELDS:
        v = h.get(k)
        if v is None or (isinstance(v, dict) and not any(v.values())):
            downgrades.append(k)
    decision_rule = h.get("decision_rule") or {}
    if isinstance(decision_rule, dict) and "p_threshold" not in decision_rule:
        missing.append("decision_rule.p_threshold")
    return VerifierResult(
        ok=not missing,
        missing=missing,
        downgrades=downgrades,
    )
