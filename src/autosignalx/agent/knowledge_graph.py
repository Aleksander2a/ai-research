"""Phase 14 -- Persistent knowledge graph as agent memory.

`lessons.md` is good but flat. A research lab needs a structured graph:

* **Nodes**: findings, hypotheses, methods, regimes, assets, mechanisms
* **Edges**: refines, refutes, generalizes, complements, attacks,
  cites_paper

The KG persists as two JSONL files under ``reports/agent/kg/``:

    nodes.jsonl  -- one JSON per node: id, kind, label, attrs, ts
    edges.jsonl  -- one JSON per edge: source, target, relation, weight, ts

Both append-only (idempotent on (id) for nodes and (source, target,
relation) for edges). The agent's Historian role queries the graph at
the start of every round so the next proposal can sit on top of prior
work; the Adjudicator writes new nodes/edges at the end of every round.

The graph is *complementary* to lessons.md: lessons.md is the
human-readable narrative, the KG is the queryable structure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autosignalx.config import settings

KG_DIR = settings.reports_dir / "agent" / "kg"
NODES_PATH = KG_DIR / "nodes.jsonl"
EDGES_PATH = KG_DIR / "edges.jsonl"

NODE_KINDS = (
    "finding",
    "hypothesis",
    "method",
    "regime",
    "asset",
    "mechanism",
    "session",
    "ticket",
)

EDGE_RELATIONS = (
    "refines",
    "refutes",
    "generalizes",
    "complements",
    "attacks",
    "cites",
    "implements",
    "promoted_by",
    "discovered_in",
    "applies_to",
)


def _ensure_dirs() -> None:
    KG_DIR.mkdir(parents=True, exist_ok=True)


def _node_id(kind: str, label: str) -> str:
    """Deterministic short ID from (kind, label)."""
    payload = json.dumps({"kind": kind, "label": label}, sort_keys=True).encode("utf-8")
    return f"n_{hashlib.sha256(payload).hexdigest()[:10]}"


def _edge_id(source: str, target: str, relation: str) -> str:
    payload = json.dumps(
        {"s": source, "t": target, "r": relation}, sort_keys=True
    ).encode("utf-8")
    return f"e_{hashlib.sha256(payload).hexdigest()[:10]}"


@dataclass
class Node:
    kind: str
    label: str
    attrs: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise ValueError(f"Unknown node kind {self.kind!r}; must be in {NODE_KINDS}")
        if not self.id:
            self.id = _node_id(self.kind, self.label)


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if self.relation not in EDGE_RELATIONS:
            raise ValueError(
                f"Unknown edge relation {self.relation!r}; must be in {EDGE_RELATIONS}"
            )
        if not self.id:
            self.id = _edge_id(self.source, self.target, self.relation)


def add_node(node: Node, path: Path | None = None) -> Node:
    """Idempotent on (kind, label)."""
    _ensure_dirs()
    p = path or NODES_PATH
    existing_ids = {n.get("id") for n in load_nodes(p)}
    if node.id in existing_ids:
        return node
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(node.__dict__, default=str) + "\n")
    return node


def add_edge(edge: Edge, path: Path | None = None) -> Edge:
    """Idempotent on (source, target, relation)."""
    _ensure_dirs()
    p = path or EDGES_PATH
    existing_ids = {e.get("id") for e in load_edges(p)}
    if edge.id in existing_ids:
        return edge
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(edge.__dict__, default=str) + "\n")
    return edge


def load_nodes(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or NODES_PATH
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


def load_edges(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or EDGES_PATH
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


def neighbors(
    node_id: str,
    relation: str | None = None,
    direction: str = "out",
    nodes_path: Path | None = None,
    edges_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return neighbor nodes connected to ``node_id``.

    direction: "out" follows source->target; "in" follows target->source;
    "both" follows either."""
    edges = load_edges(edges_path)
    nodes = {n["id"]: n for n in load_nodes(nodes_path)}
    out_ids: list[str] = []
    for e in edges:
        if relation is not None and e.get("relation") != relation:
            continue
        if direction in ("out", "both") and e.get("source") == node_id:
            out_ids.append(e.get("target", ""))
        if direction in ("in", "both") and e.get("target") == node_id:
            out_ids.append(e.get("source", ""))
    return [nodes[i] for i in out_ids if i in nodes]


def query(
    kind: str | None = None,
    label_contains: str | None = None,
    nodes_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Filter nodes by kind and/or label substring."""
    out = load_nodes(nodes_path)
    if kind is not None:
        out = [n for n in out if n.get("kind") == kind]
    if label_contains is not None:
        needle = label_contains.lower()
        out = [n for n in out if needle in str(n.get("label", "")).lower()]
    return out


def ingest_findings(
    findings: list[dict[str, Any]],
    nodes_path: Path | None = None,
    edges_path: Path | None = None,
) -> dict[str, int]:
    """Idempotently ingest a list of finding records into the KG.

    Creates: finding nodes, method nodes, asset nodes, regime nodes;
    edges: finding -applies_to-> asset, finding -implements-> method,
    finding -applies_to-> regime."""
    n_nodes = 0
    n_edges = 0
    for f in findings:
        fid = f.get("id")
        if not fid:
            continue
        f_node = Node(
            kind="finding",
            label=fid,
            attrs={
                "hypothesis": (f.get("hypothesis") or "")[:300],
                "method": f.get("method"),
                "filters": f.get("filters", {}),
                "skill": (f.get("evidence") or {}).get("skill_vs_baseline"),
                "p_value": (f.get("evidence") or {}).get("p_value"),
                "session_id": f.get("session_id"),
            },
        )
        before = len(load_nodes(nodes_path))
        add_node(f_node, path=nodes_path)
        if len(load_nodes(nodes_path)) > before:
            n_nodes += 1

        method = f.get("method")
        if method:
            m_node = Node(kind="method", label=str(method))
            before = len(load_nodes(nodes_path))
            add_node(m_node, path=nodes_path)
            if len(load_nodes(nodes_path)) > before:
                n_nodes += 1
            edge = Edge(source=f_node.id, target=m_node.id, relation="implements")
            before = len(load_edges(edges_path))
            add_edge(edge, path=edges_path)
            if len(load_edges(edges_path)) > before:
                n_edges += 1

        filters = f.get("filters") or {}
        asset = filters.get("asset")
        if asset:
            a_node = Node(kind="asset", label=str(asset))
            before = len(load_nodes(nodes_path))
            add_node(a_node, path=nodes_path)
            if len(load_nodes(nodes_path)) > before:
                n_nodes += 1
            edge = Edge(source=f_node.id, target=a_node.id, relation="applies_to")
            before = len(load_edges(edges_path))
            add_edge(edge, path=edges_path)
            if len(load_edges(edges_path)) > before:
                n_edges += 1

        regime_id = filters.get("regime_id")
        if regime_id is not None:
            r_node = Node(kind="regime", label=f"regime_{regime_id}")
            before = len(load_nodes(nodes_path))
            add_node(r_node, path=nodes_path)
            if len(load_nodes(nodes_path)) > before:
                n_nodes += 1
            edge = Edge(source=f_node.id, target=r_node.id, relation="applies_to")
            before = len(load_edges(edges_path))
            add_edge(edge, path=edges_path)
            if len(load_edges(edges_path)) > before:
                n_edges += 1
    return {"nodes_added": n_nodes, "edges_added": n_edges}


def kg_summary(
    nodes_path: Path | None = None,
    edges_path: Path | None = None,
) -> dict[str, Any]:
    """Summary of the KG suitable for prompt seeding."""
    nodes = load_nodes(nodes_path)
    edges = load_edges(edges_path)
    by_kind: dict[str, int] = {}
    for n in nodes:
        by_kind[n.get("kind", "?")] = by_kind.get(n.get("kind", "?"), 0) + 1
    by_relation: dict[str, int] = {}
    for e in edges:
        by_relation[e.get("relation", "?")] = by_relation.get(e.get("relation", "?"), 0) + 1
    return {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "nodes_by_kind": by_kind,
        "edges_by_relation": by_relation,
    }
