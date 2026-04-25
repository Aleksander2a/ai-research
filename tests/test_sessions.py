"""Tests for cross-session aggregation (Iter 18)."""

from __future__ import annotations

from pathlib import Path

from autosignalx.agent import findings as findings_mod
from autosignalx.agent import ledger as ledger_mod
from autosignalx.agent import sessions
from autosignalx.agent import telemetry as telemetry_mod
from autosignalx.agent import trace_eval as trace_eval_mod


def _setup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ledger_mod, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(findings_mod, "FINDINGS_DIR", tmp_path / "findings")
    monkeypatch.setattr(telemetry_mod, "TELEMETRY_DIR", tmp_path / "telemetry")
    monkeypatch.setattr(trace_eval_mod, "QUALITY_DIR", tmp_path / "trace")
    ledger_mod.clear()
    findings_mod.clear()
    telemetry_mod.clear()
    trace_eval_mod.clear()


def test_list_sessions_empty(tmp_path: Path, monkeypatch) -> None:
    _setup(tmp_path, monkeypatch)
    assert sessions.list_sessions() == []


def test_list_sessions_dedupe_across_stores(tmp_path: Path, monkeypatch) -> None:
    _setup(tmp_path, monkeypatch)
    ledger_mod.append({"round": 0, "step": "propose", "content": "x", "session_id": "20260426-aaaa"})
    ledger_mod.append({"round": 0, "step": "experiment", "content": "y", "session_id": "20260427-bbbb"})
    telemetry_mod.record_call(
        model="x", role="r", step="s", round_n=0,
        prompt_tokens=10, completion_tokens=20, latency_ms=100, session_id="20260426-aaaa",
    )
    out = sessions.list_sessions()
    assert out == ["20260426-aaaa", "20260427-bbbb"]


def test_session_summary_aggregates_correctly(tmp_path: Path, monkeypatch) -> None:
    _setup(tmp_path, monkeypatch)
    sid = "20260426-aaaa"
    # Two propose entries, one finding, one telemetry record
    ledger_mod.append({"round": 0, "step": "propose", "content": "h1", "session_id": sid})
    ledger_mod.append({"round": 1, "step": "propose", "content": "h2", "session_id": sid})
    ledger_mod.append({"round": 1, "step": "adjudicator", "content": "VERDICT: refute", "session_id": sid})
    findings_mod.promote(
        hypothesis="x", method="m", filters={"asset": None, "regime_id": None},
        evidence={"p_value": 0.01}, agent_confidence="ok", round=0, session_id=sid,
    )
    telemetry_mod.record_call(
        model="kimi", role="r", step="s", round_n=0,
        prompt_tokens=1000, completion_tokens=500, latency_ms=2000, session_id=sid,
    )
    s = sessions.session_summary(sid)
    assert s["session_id"] == sid
    assert s["n_propose"] == 2
    assert s["n_findings"] == 1
    assert s["n_refuted"] == 1
    assert s["total_tokens"] == 1500
    assert s["promotion_rate"] == 0.5
    assert s["cost_per_finding"] is not None
    assert s["cost_per_finding"] > 0


def test_all_summaries_returns_dataframe(tmp_path: Path, monkeypatch) -> None:
    _setup(tmp_path, monkeypatch)
    findings_mod.promote(
        hypothesis="x", method="m", filters={}, evidence={}, agent_confidence="",
        round=0, session_id="20260426-aaaa",
    )
    df = sessions.all_summaries()
    assert len(df) == 1
    assert "session_id" in df.columns


def test_productivity_trend_cumulative(tmp_path: Path, monkeypatch) -> None:
    _setup(tmp_path, monkeypatch)
    findings_mod.promote(
        hypothesis="a", method="m", filters={"asset": "A"}, evidence={}, agent_confidence="",
        round=0, session_id="20260426-aaaa",
    )
    findings_mod.promote(
        hypothesis="b", method="m", filters={"asset": "B"}, evidence={}, agent_confidence="",
        round=0, session_id="20260427-bbbb",
    )
    trend = sessions.productivity_trend()
    assert "cum_findings" in trend.columns
    assert trend["cum_findings"].tolist() == [1, 2]
