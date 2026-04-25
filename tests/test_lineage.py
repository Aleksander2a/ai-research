"""Tests for the hypothesis lineage DAG (Iter 14)."""

from __future__ import annotations

from autosignalx.agent import lineage


def _propose(round_n: int, method: str, asset: str | None = None, regime_id: int | None = None, hypothesis: str | None = None):
    return {
        "round": round_n,
        "step": "propose",
        "content": {
            "hypothesis": hypothesis or f"hypothesis at round {round_n}",
            "experiment": {
                "type": "slice_forecasts",
                "params": {"method": method, "asset": asset, "regime_id": regime_id},
            },
        },
        "ts": "2026-04-26T00:00:00+00:00",
    }


def test_hypothesis_id_stable() -> None:
    a = _propose(0, "naive")["content"]
    b = _propose(0, "naive")["content"]
    assert lineage.hypothesis_id(a) == lineage.hypothesis_id(b)


def test_hypothesis_id_changes_with_method() -> None:
    a = _propose(0, "naive")["content"]
    b = _propose(0, "arima")["content"]
    assert lineage.hypothesis_id(a) != lineage.hypothesis_id(b)


def test_build_lineage_no_entries_empty() -> None:
    out = lineage.build_lineage(ledger_entries=[], finding_records=[])
    assert out == {"nodes": [], "edges": []}


def test_build_lineage_chains_overlapping_methods() -> None:
    entries = [
        _propose(0, "chronos2_univariate", asset="SPY", regime_id=2),
        _propose(1, "chronos2_univariate", asset="SPY", regime_id=2, hypothesis="refined"),
    ]
    out = lineage.build_lineage(ledger_entries=entries, finding_records=[])
    assert len(out["nodes"]) == 2
    assert len(out["edges"]) == 1
    assert out["edges"][0]["source"] != out["edges"][0]["target"]


def test_build_lineage_no_edge_when_no_overlap() -> None:
    entries = [
        _propose(0, "naive", asset="SPY", regime_id=0),
        _propose(1, "arima", asset="GLD", regime_id=3),
    ]
    out = lineage.build_lineage(ledger_entries=entries, finding_records=[])
    assert len(out["edges"]) == 0


def test_promoted_status_via_finding_round_match() -> None:
    entries = [_propose(0, "chronos2_univariate", asset="SPY", regime_id=2)]
    findings = [
        {
            "id": "f_xxx",
            "round": 0,
            "session_id": "s1",
            "parent_hypothesis_ids": [],
        }
    ]
    out = lineage.build_lineage(ledger_entries=entries, finding_records=findings)
    assert out["nodes"][0]["status"] == "promoted"


def test_refuted_status_via_adjudicator_verdict() -> None:
    entries = [
        _propose(0, "naive"),
        {"round": 0, "step": "adjudicator", "content": "Indeed problematic. VERDICT: refute"},
    ]
    out = lineage.build_lineage(ledger_entries=entries, finding_records=[])
    assert out["nodes"][0]["status"] == "refuted"


def test_lineage_dataframe_columns() -> None:
    entries = [_propose(0, "naive")]
    out = lineage.build_lineage(ledger_entries=entries, finding_records=[])
    df = lineage.lineage_dataframe(out)
    assert set(df.columns) >= {"id", "round", "status", "hypothesis", "parents"}
    assert df.iloc[0]["parents"] == "(root)"
