"""Tests for the promoted-findings store."""

from __future__ import annotations

from autosignalx.agent import findings as findings_mod


def test_promote_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(findings_mod, "FINDINGS_DIR", tmp_path)
    findings_mod.clear()
    rec = findings_mod.promote(
        hypothesis="naive is beatable on regime 3",
        method="chronos2_multivariate",
        filters={"asset": "EFA", "regime_id": 3},
        evidence={"p_value": 0.01, "skill_vs_baseline": 0.05},
        agent_confidence="strong",
        round=4,
        session_id="s1",
    )
    assert rec["id"].startswith("f_")
    assert rec["replication_count"] == 1
    rows = findings_mod.load()
    assert len(rows) == 1
    assert rows[0]["method"] == "chronos2_multivariate"


def test_promote_idempotent_bumps_replication(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(findings_mod, "FINDINGS_DIR", tmp_path)
    findings_mod.clear()
    args = dict(
        hypothesis="x",
        method="m",
        filters={"asset": "A", "regime_id": 1},
        evidence={"p_value": 0.01},
        agent_confidence="ok",
    )
    findings_mod.promote(round=1, session_id="s1", **args)
    findings_mod.promote(round=2, session_id="s2", **args)
    rows = findings_mod.load()
    assert len(rows) == 1
    assert rows[0]["replication_count"] == 2
    assert len(rows[0]["replications"]) == 2


def test_make_session_id_unique() -> None:
    a = findings_mod.make_session_id()
    b = findings_mod.make_session_id()
    assert a != b
    assert len(a) > 5
