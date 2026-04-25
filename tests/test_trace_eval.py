"""Tests for the LLM-as-judge trace quality evaluator (Iter 15)."""

from __future__ import annotations

import json
from pathlib import Path

from autosignalx.agent import trace_eval
from autosignalx.agent.llm import ReplayProvider


def _round(rd: int):
    return [
        {"round": rd, "step": "propose", "content": {"hypothesis": "x", "experiment": {}}},
        {"round": rd, "step": "experiment", "content": {"n": 50}},
        {"round": rd, "step": "critique", "content": "ok"},
    ]


def test_score_round_with_replay_provider_returns_keys(tmp_path: Path) -> None:
    rec = tmp_path / "agent_steps.jsonl"
    rec.write_text(
        json.dumps(
            {
                "round": 0,
                "step": "trace_eval",
                "content": json.dumps(
                    {
                        "clarity": 4,
                        "novelty": 3,
                        "falsifiability": 5,
                        "evidence_citing": 2,
                        "rationale": "specific but lacks citations",
                    }
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provider = ReplayProvider(path=rec)
    out = trace_eval.score_round(0, _round(0), provider=provider)
    assert out["clarity"] == 4
    assert out["novelty"] == 3
    assert out["falsifiability"] == 5
    assert out["evidence_citing"] == 2
    assert "rationale" in out
    assert "ts" in out


def test_score_round_handles_unparseable_response(tmp_path: Path) -> None:
    rec = tmp_path / "agent_steps.jsonl"
    rec.write_text(
        json.dumps(
            {"round": 0, "step": "trace_eval", "content": "not valid json"}
        )
        + "\n",
        encoding="utf-8",
    )
    provider = ReplayProvider(path=rec)
    out = trace_eval.score_round(0, _round(0), provider=provider)
    # All score keys should still exist (None-valued)
    for k in ("clarity", "novelty", "falsifiability", "evidence_citing"):
        assert k in out


def test_score_session_persists_and_returns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(trace_eval, "QUALITY_DIR", tmp_path)
    trace_eval.clear()
    rec = tmp_path / "replay.jsonl"
    rec.write_text(
        json.dumps(
            {
                "round": 0,
                "step": "trace_eval",
                "content": json.dumps(
                    {
                        "clarity": 3,
                        "novelty": 3,
                        "falsifiability": 3,
                        "evidence_citing": 3,
                        "rationale": "average",
                    }
                ),
            }
        )
        + "\n"
        + json.dumps(
            {
                "round": 1,
                "step": "trace_eval",
                "content": json.dumps(
                    {
                        "clarity": 4,
                        "novelty": 4,
                        "falsifiability": 4,
                        "evidence_citing": 4,
                        "rationale": "improving",
                    }
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provider = ReplayProvider(path=rec)
    entries = _round(0) + _round(1)
    scores = trace_eval.score_session(entries, session_id="s1", provider=provider)
    assert len(scores) == 2
    persisted = trace_eval.load()
    assert len(persisted) == 2
    assert persisted[0]["session_id"] == "s1"


def test_round_summary_truncates_long_content() -> None:
    long_content = "x" * 1000
    out = trace_eval._round_summary(
        [{"round": 0, "step": "test", "content": long_content}]
    )
    assert "test:" in out
    assert len(out) < 600  # truncated to ~400 chars per entry
