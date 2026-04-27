"""One-shot backfill: add `role` and `session_id` to historical telemetry records.

Older live sessions wrote ``role="unknown"`` and ``session_id=None`` because
the LLM provider didn't yet thread either through. This script fixes the
committed records in-place by:

* Inferring ``role`` from the ``step`` field (e.g. step=theorist -> role=theorist).
* Aligning each telemetry timestamp with the closest preceding ``ledger.jsonl``
  entry and copying its ``session_id`` over.

The script is idempotent: rows that already have a non-trivial role or
session_id are left untouched. Future records produced by the patched
``LiveProvider`` carry both fields natively, so this runs once.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY = REPO_ROOT / "reports" / "agent" / "telemetry.jsonl"
LEDGER = REPO_ROOT / "reports" / "agent" / "ledger.jsonl"


STEP_TO_ROLE = {
    "propose": "proposer",
    "theorist": "theorist",
    "skeptic": "skeptic",
    "critique": "critic",
    "adjudicator": "adjudicator",
    "decide": "adjudicator",
    "consolidate": "adjudicator",
    "trace_eval": "critic",
    "self_critique": "adjudicator",
    "principal_investigator": "adjudicator",
    "verifier": "adjudicator",
    "kg_writer": "adjudicator",
}


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _read_jsonl(p: Path) -> list[dict]:
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


def main() -> int:
    if not TELEMETRY.exists():
        print(f"[backfill] no telemetry file at {TELEMETRY}; nothing to do.")
        return 0
    rows = _read_jsonl(TELEMETRY)
    if not rows:
        print("[backfill] telemetry empty; nothing to do.")
        return 0

    ledger = _read_jsonl(LEDGER)
    # Build a sorted index of (timestamp, session_id) from ledger entries that
    # have both fields. Use it to attribute each telemetry call to the
    # nearest-in-time session that was active at that wall clock.
    ledger_idx: list[tuple[datetime, str]] = []
    for e in ledger:
        sid = e.get("session_id")
        ets = _parse_ts(e.get("ts"))
        if sid and ets is not None:
            ledger_idx.append((ets, str(sid)))
    ledger_idx.sort(key=lambda x: x[0])

    def _session_for(ts: datetime | None) -> str | None:
        if ts is None or not ledger_idx:
            return None
        # linear scan: pick the latest ledger entry at or before ts
        chosen: str | None = None
        for ets, sid in ledger_idx:
            if ets <= ts:
                chosen = sid
            else:
                break
        # if all ledger entries are after the telemetry record, fall back to
        # the earliest known session_id
        return chosen or ledger_idx[0][1]

    n_role = 0
    n_session = 0
    for r in rows:
        if r.get("role") in (None, "", "unknown"):
            step = str(r.get("step", "")).strip().lower()
            inferred = STEP_TO_ROLE.get(step) or (
                "specialist" if step.startswith("specialist:") else None
            )
            if inferred:
                r["role"] = inferred
                n_role += 1
        if not r.get("session_id"):
            ts = _parse_ts(r.get("ts"))
            sid = _session_for(ts)
            if sid:
                r["session_id"] = sid
                n_session += 1

    with TELEMETRY.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print(
        f"[backfill] rewrote {len(rows)} telemetry rows; "
        f"backfilled role on {n_role}, session_id on {n_session}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
