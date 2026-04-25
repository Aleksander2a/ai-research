"""Hypothesis lineage DAG.

Each hypothesis the agent proposes gets a stable ID derived from its
content. ``parent_ids`` are inferred heuristically (overlap on
method/asset/regime with prior round hypotheses) so the DAG can be
constructed entirely from the existing ledger -- no extra agent prompt
turns required.

The DAG is rendered in the cockpit's Lineage panel: nodes colored by
status (open / refuted / promoted), edges showing which prior thinking
each hypothesis built on. Reviewers can trace any promoted finding back
to its initial brainstorm and see the chain of refinements."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from autosignalx.agent import findings as findings_mod
from autosignalx.agent import ledger as ledger_mod


def hypothesis_id(content: dict[str, Any]) -> str:
    """Stable short ID from hypothesis text + experiment params."""
    payload = json.dumps(
        {
            "hypothesis": content.get("hypothesis", "")[:200],
            "experiment": content.get("experiment", {}),
        },
        sort_keys=True,
        default=str,
    )
    return "h_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _params(content: dict[str, Any]) -> dict[str, Any]:
    return content.get("experiment", {}).get("params", {}) or {}


def _overlap_score(a_params: dict[str, Any], b_params: dict[str, Any]) -> int:
    """Count how many of (method, asset, regime_id) match between two
    hypotheses. Used as a coarse parent-inference heuristic."""
    score = 0
    for k in ("method", "asset", "regime_id"):
        av = a_params.get(k)
        bv = b_params.get(k)
        if av is not None and av == bv:
            score += 1
    return score


def build_lineage(
    ledger_entries: list[dict[str, Any]] | None = None,
    finding_records: list[dict[str, Any]] | None = None,
    parent_lookback: int = 5,
    overlap_threshold: int = 1,
) -> dict[str, Any]:
    """Construct the lineage DAG from the ledger.

    Returns a dict with ``nodes`` (one per unique hypothesis) and
    ``edges`` (parent -> child). Status is derived from the findings
    store: a hypothesis whose ID matches a promoted finding's
    ``parent_hypothesis_ids`` or a directly-promoted hypothesis ID is
    marked ``promoted``; hypotheses whose adjudicator returned a
    refute verdict are ``refuted``; everything else is ``open``."""
    entries = ledger_entries if ledger_entries is not None else ledger_mod.load()
    if finding_records is None:
        finding_records = findings_mod.load()

    propose_steps = [e for e in entries if e.get("step") in ("propose", "theorist")]
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    id_for_round: dict[int, str] = {}

    for entry in propose_steps:
        content = entry.get("content", {}) or {}
        if not isinstance(content, dict):
            continue
        hid = hypothesis_id(content)
        rd = int(entry.get("round", 0))
        id_for_round[rd] = hid
        if hid in seen:
            # second occurrence of same hypothesis -- skip duplicate node
            continue
        seen.add(hid)
        nodes.append(
            {
                "id": hid,
                "round": rd,
                "step": entry.get("step", "propose"),
                "hypothesis": (content.get("hypothesis") or "")[:200],
                "params": _params(content),
                "ts": entry.get("ts", ""),
                "status": "open",  # filled below
            }
        )

    # Edges: for each propose step, look back up to parent_lookback rounds for
    # the closest hypothesis with sufficient (method/asset/regime) overlap.
    edges: list[dict[str, str]] = []
    propose_by_round = {int(e.get("round", 0)): e for e in propose_steps}
    for rd in sorted(propose_by_round.keys()):
        cur_content = propose_by_round[rd].get("content", {}) or {}
        if not isinstance(cur_content, dict):
            continue
        cur_id = hypothesis_id(cur_content)
        cur_params = _params(cur_content)
        for prev_rd in range(rd - 1, max(-1, rd - parent_lookback - 1), -1):
            prev_entry = propose_by_round.get(prev_rd)
            if not prev_entry:
                continue
            prev_content = prev_entry.get("content", {}) or {}
            if not isinstance(prev_content, dict):
                continue
            prev_id = hypothesis_id(prev_content)
            if prev_id == cur_id:
                continue
            if _overlap_score(cur_params, _params(prev_content)) >= overlap_threshold:
                edges.append({"source": prev_id, "target": cur_id})
                break  # only the closest parent per current node

    # Promotion: any hypothesis ID matching a promoted finding's parent IDs
    # (or its own derived ID) is marked promoted.
    promoted_ids: set[str] = set()
    for f in finding_records:
        for pid in f.get("parent_hypothesis_ids", []) or []:
            promoted_ids.add(str(pid))
    # Also: a finding's source round can be matched back to a hypothesis ID.
    for f in finding_records:
        rd = f.get("round")
        if rd is not None:
            hid = id_for_round.get(int(rd))
            if hid:
                promoted_ids.add(hid)

    # Refutation: an adjudicator entry whose content contains "VERDICT: refute"
    # marks the hypothesis from the same round as refuted.
    refuted_ids: set[str] = set()
    for e in entries:
        if e.get("step") == "adjudicator":
            content = str(e.get("content", "")).lower()
            if "verdict: refute" in content:
                hid = id_for_round.get(int(e.get("round", 0)))
                if hid:
                    refuted_ids.add(hid)

    for n in nodes:
        if n["id"] in promoted_ids:
            n["status"] = "promoted"
        elif n["id"] in refuted_ids:
            n["status"] = "refuted"
    return {"nodes": nodes, "edges": edges}


def lineage_dataframe(lineage: dict[str, Any]) -> pd.DataFrame:
    """Convenience: return a DataFrame of (id, round, status, hypothesis,
    parents) suitable for tabular rendering in the cockpit."""
    edges_by_target: dict[str, list[str]] = {}
    for e in lineage.get("edges", []):
        edges_by_target.setdefault(e["target"], []).append(e["source"])
    rows = []
    for n in lineage.get("nodes", []):
        rows.append(
            {
                "id": n["id"],
                "round": n["round"],
                "status": n["status"],
                "hypothesis": n["hypothesis"],
                "parents": ", ".join(edges_by_target.get(n["id"], [])) or "(root)",
            }
        )
    return pd.DataFrame(rows)
