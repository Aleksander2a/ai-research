"""Multi-session aggregation -- the productivity dashboard.

Reads ledger / findings / telemetry / trace_quality and groups
everything by ``session_id``, producing per-session summaries and
cross-session productivity trends. The cockpit Sessions panel renders
this for the long-term view of agent operation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from autosignalx.agent import findings as findings_mod
from autosignalx.agent import ledger as ledger_mod
from autosignalx.agent import telemetry as telemetry_mod
from autosignalx.agent import trace_eval as trace_eval_mod


def list_sessions() -> list[str]:
    """Distinct session IDs across all stores, sorted lexicographically
    (which is also chronological for our YYYYMMDD-prefixed IDs)."""
    seen: set[str] = set()
    for entries in (ledger_mod.load(), findings_mod.load(), telemetry_mod.load(), trace_eval_mod.load()):
        for e in entries:
            sid = e.get("session_id")
            if sid:
                seen.add(str(sid))
    return sorted(seen)


def session_summary(session_id: str) -> dict[str, Any]:
    """Aggregate everything for one session into a single summary dict."""
    ledger_entries = [e for e in ledger_mod.load() if e.get("session_id") == session_id]
    findings = [f for f in findings_mod.load() if f.get("session_id") == session_id]
    telemetry = [t for t in telemetry_mod.load() if t.get("session_id") == session_id]
    trace = [s for s in trace_eval_mod.load() if s.get("session_id") == session_id]

    rounds = sorted({int(e.get("round", 0)) for e in ledger_entries})
    propose_count = sum(1 for e in ledger_entries if e.get("step") in ("propose", "theorist"))
    refute_count = sum(
        1
        for e in ledger_entries
        if e.get("step") == "adjudicator" and "verdict: refute" in str(e.get("content", "")).lower()
    )
    cost_usd = sum(float(t.get("cost_usd", 0.0)) for t in telemetry)
    total_tokens = sum(int(t.get("total_tokens", 0)) for t in telemetry)
    latency_total = sum(float(t.get("latency_ms", 0.0)) for t in telemetry)

    avg_quality = None
    if trace:
        scores = [s.get("clarity") for s in trace if s.get("clarity") is not None]
        if scores:
            avg_quality = sum(scores) / len(scores)

    return {
        "session_id": session_id,
        "n_rounds": len(rounds),
        "n_propose": propose_count,
        "n_findings": len(findings),
        "n_refuted": refute_count,
        "cost_usd": cost_usd,
        "total_tokens": total_tokens,
        "latency_total_ms": latency_total,
        "avg_clarity": avg_quality,
        "promotion_rate": (len(findings) / propose_count) if propose_count else 0.0,
        "cost_per_finding": (cost_usd / len(findings)) if findings else None,
    }


def all_summaries() -> pd.DataFrame:
    """One row per session, sorted by session_id (chronological)."""
    rows = [session_summary(sid) for sid in list_sessions()]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def productivity_trend() -> pd.DataFrame:
    """Cumulative findings / cost across sessions."""
    df = all_summaries()
    if df.empty:
        return df
    df = df.sort_values("session_id").reset_index(drop=True)
    df["cum_findings"] = df["n_findings"].cumsum()
    df["cum_cost_usd"] = df["cost_usd"].cumsum()
    return df
