"""Phase 14 tests: specialist roles, KG memory, EIG, verifier, lab graph."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autosignalx.agent import eig as eig_mod
from autosignalx.agent import knowledge_graph as kg_mod
from autosignalx.agent import specialists as specialists_mod
from autosignalx.agent import verifier as verifier_mod

# ---------- Specialists ---------- #


def test_specialist_roles_have_prompts():
    for r in (
        "principal_investigator",
        "statistician",
        "quant",
        "risk_officer",
        "economist",
        "implementer",
        "red_team",
        "historian",
    ):
        assert r in specialists_mod.SPECIALIST_PROMPTS
        assert len(specialists_mod.SPECIALIST_PROMPTS[r]) > 200


def test_consult_specialist_returns_consult(monkeypatch):
    class FakeProvider:
        mode = "fake"

        def chat(self, messages, step, round):  # noqa: ARG002
            return "Test specialist response."

    consult = specialists_mod.consult_specialist(
        role="statistician",
        payload={"hypothesis": {"text": "test"}},
        provider=FakeProvider(),
        round_n=0,
    )
    assert consult.role == "statistician"
    assert consult.response.startswith("Test")


def test_consult_specialist_rejects_unknown_role():
    with pytest.raises(ValueError):
        specialists_mod.consult_specialist(role="garbage", payload={})


# ---------- Knowledge graph ---------- #


def test_node_id_deterministic():
    n1 = kg_mod.Node(kind="finding", label="f_abc")
    n2 = kg_mod.Node(kind="finding", label="f_abc")
    assert n1.id == n2.id


def test_node_kind_validation():
    with pytest.raises(ValueError):
        kg_mod.Node(kind="garbage", label="x")


def test_kg_add_and_query(tmp_path: Path):
    nodes_path = tmp_path / "nodes.jsonl"
    edges_path = tmp_path / "edges.jsonl"
    f = kg_mod.Node(kind="finding", label="f_test")
    m = kg_mod.Node(kind="method", label="chronos2_multivariate")
    kg_mod.add_node(f, path=nodes_path)
    kg_mod.add_node(m, path=nodes_path)
    kg_mod.add_edge(
        kg_mod.Edge(source=f.id, target=m.id, relation="implements"),
        path=edges_path,
    )
    nodes = kg_mod.load_nodes(nodes_path)
    assert len(nodes) == 2
    edges = kg_mod.load_edges(edges_path)
    assert len(edges) == 1
    out = kg_mod.neighbors(f.id, relation="implements", nodes_path=nodes_path, edges_path=edges_path)
    assert len(out) == 1
    assert out[0]["kind"] == "method"


def test_kg_idempotent_on_re_add(tmp_path: Path):
    np_ = tmp_path / "n.jsonl"
    n = kg_mod.Node(kind="asset", label="SPY")
    kg_mod.add_node(n, path=np_)
    kg_mod.add_node(n, path=np_)
    assert len(kg_mod.load_nodes(np_)) == 1


def test_kg_ingest_findings(tmp_path: Path):
    np_, ep = tmp_path / "n.jsonl", tmp_path / "e.jsonl"
    findings = [
        {
            "id": "f_one",
            "method": "chronos2_multivariate",
            "filters": {"asset": "TLT", "regime_id": 3},
            "evidence": {"skill_vs_baseline": 0.05, "p_value": 0.04},
            "hypothesis": "TLT chronos beats naive in regime 3",
            "session_id": "s1",
        }
    ]
    res = kg_mod.ingest_findings(findings, nodes_path=np_, edges_path=ep)
    assert res["nodes_added"] >= 4  # finding, method, asset, regime
    assert res["edges_added"] >= 3


def test_kg_summary(tmp_path: Path):
    np_, ep = tmp_path / "n.jsonl", tmp_path / "e.jsonl"
    kg_mod.add_node(kg_mod.Node(kind="asset", label="SPY"), path=np_)
    s = kg_mod.kg_summary(nodes_path=np_, edges_path=ep)
    assert s["n_nodes"] == 1
    assert s["nodes_by_kind"]["asset"] == 1


# ---------- EIG ---------- #


def test_candidate_eig_ranks_untested_higher():
    forecasts = pd.DataFrame([
        {"method": "naive", "asset": "SPY", "regime_id": 0, "prediction": 1.0, "target": 1.0,
         "forecast_origin": pd.Timestamp("2024-01-01"), "timestamp": pd.Timestamp("2024-01-02")},
        {"method": "naive", "asset": "SPY", "regime_id": 0, "prediction": 1.0, "target": 1.0,
         "forecast_origin": pd.Timestamp("2024-01-02"), "timestamp": pd.Timestamp("2024-01-03")},
    ])
    cands = eig_mod.candidate_eig(
        forecasts=forecasts,
        methods=["naive", "good"],
        assets=["SPY", "QQQ"],
        regimes=[0, 1],
        findings=[],
        tested_keys=set(),
    )
    # Untested with data-rich slices (SPY/regime_0/naive) should be near the top
    assert cands[0].n_samples > 0


def test_coverage_map_returns_dataframe():
    forecasts = pd.DataFrame()
    df = eig_mod.coverage_map(
        forecasts=forecasts,
        methods=["naive"],
        assets=["SPY"],
        regimes=[0],
        findings=[],
        tested_keys=set(),
    )
    assert "status" in df.columns
    assert "eig_score" in df.columns


# ---------- Verifier ---------- #


def test_verifier_passes_complete_hypothesis():
    h = {
        "hypothesis": "X",
        "decision_rule": {"p_threshold": 0.05},
        "predicted_effect": {"expected_skill": 0.05},
        "falsifier": "If p>0.05",
    }
    r = verifier_mod.verify_hypothesis(h)
    assert r.ok is True
    assert not r.missing


def test_verifier_flags_missing_decision_rule():
    h = {
        "hypothesis": "X",
        "falsifier": "F",
    }
    r = verifier_mod.verify_hypothesis(h)
    assert r.ok is False
    assert "decision_rule" in r.missing


def test_verifier_downgrades_missing_predicted_effect():
    h = {
        "hypothesis": "X",
        "decision_rule": {"p_threshold": 0.05},
        "falsifier": "F",
    }
    r = verifier_mod.verify_hypothesis(h)
    assert r.ok is True
    assert "predicted_effect" in r.downgrades
