"""Phase 15 -- Agent confidence calibration.

For every promoted finding, we asked the Theorist to predict the
expected effect size (Phase 14's pre-registration). Calibration scores
the Theorist's predictions against observed outcomes:

* **Brier score** = mean squared error between predicted prob and observed
  binary outcome (here: did the finding survive hardening?).
* **Reliability diagram** -- bin the predicted-confidence score and plot
  observed survival rate per bin. Perfect calibration: identity line.
* **Expected Calibration Error (ECE)** -- weighted-mean absolute deviation
  from the identity line.

Output is a per-role calibration record under
``reports/agent/calibration.jsonl``. Cockpit Phase-16 panel renders
the reliability diagram.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CalibrationRecord:
    role: str
    n: int
    brier: float
    ece: float
    bins: list[dict[str, Any]] = field(default_factory=list)


def _coerce_confidence(v: Any) -> float | None:
    """Map various confidence encodings to a [0, 1] scalar."""
    if v is None:
        return None
    try:
        x = float(v)
        if 0.0 <= x <= 1.0:
            return x
        if 0.0 <= x <= 100.0:
            return x / 100.0
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        v_low = v.strip().lower()
        mapping = {
            "low": 0.2, "medium": 0.5, "high": 0.8, "very high": 0.95,
            "yes": 0.85, "no": 0.15, "auto-promoted by experiment gate": 0.7,
        }
        for key, val in mapping.items():
            if key in v_low:
                return val
    return None


def calibration_for_role(
    findings: list[dict[str, Any]],
    survival_records: list[dict[str, Any]],
    role: str = "theorist",
    n_bins: int = 5,
    survival_field: str = "survives_all_strict",
) -> CalibrationRecord:
    """Compute calibration of agent confidence vs survival outcome.

    ``findings`` rows must contain ``id`` and a confidence-like field
    (we accept ``agent_confidence`` text or ``predicted_effect.expected_p``
    or ``predicted_effect.expected_skill``). ``survival_records`` is the
    output of Phase-5 hardening; we look up the matching ``finding_id``.
    """
    surv_by_id = {s.get("finding_id"): s for s in survival_records}
    confs: list[float] = []
    outcomes: list[int] = []
    for f in findings:
        fid = f.get("id")
        if not fid or fid not in surv_by_id:
            continue
        # Confidence: try predicted_effect.expected_p first, then text
        pe = f.get("predicted_effect") or {}
        c = _coerce_confidence(pe.get("expected_skill")) or _coerce_confidence(
            pe.get("expected_p")
        ) or _coerce_confidence(f.get("agent_confidence"))
        if c is None:
            continue
        outcome = surv_by_id[fid].get(survival_field)
        if outcome is None:
            continue
        confs.append(float(c))
        outcomes.append(int(bool(outcome)))

    n = len(confs)
    if n == 0:
        return CalibrationRecord(role=role, n=0, brier=float("nan"), ece=float("nan"), bins=[])

    confs_arr = np.asarray(confs, dtype=float)
    outcomes_arr = np.asarray(outcomes, dtype=float)
    brier = float(np.mean((confs_arr - outcomes_arr) ** 2))

    # Equal-width binning
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_records: list[dict[str, Any]] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confs_arr >= lo) & (confs_arr <= hi)
        else:
            mask = (confs_arr >= lo) & (confs_arr < hi)
        nb = int(mask.sum())
        if nb == 0:
            bin_records.append({"bin_lo": lo, "bin_hi": hi, "n": 0, "mean_confidence": None, "obs_rate": None})
            continue
        mean_conf = float(confs_arr[mask].mean())
        obs_rate = float(outcomes_arr[mask].mean())
        ece += (nb / n) * abs(mean_conf - obs_rate)
        bin_records.append({
            "bin_lo": lo, "bin_hi": hi, "n": nb,
            "mean_confidence": mean_conf, "obs_rate": obs_rate,
        })

    return CalibrationRecord(
        role=role,
        n=n,
        brier=brier,
        ece=float(ece),
        bins=bin_records,
    )
