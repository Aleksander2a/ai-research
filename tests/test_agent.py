"""Agent layer tests: ledger persistence, replay provider, graph compile."""

from __future__ import annotations

import json
from pathlib import Path

from autosignalx.agent import ledger as ledger_mod
from autosignalx.agent.llm import ReplayProvider
from autosignalx.agent.state import AgentState


def test_ledger_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ledger_mod, "LEDGER_DIR", tmp_path)
    ledger_mod.clear()
    ledger_mod.append({"round": 0, "step": "propose", "content": {"x": 1}})
    ledger_mod.append({"round": 0, "step": "experiment", "content": "ok"})
    rows = ledger_mod.load()
    assert len(rows) == 2
    assert rows[0]["step"] == "propose"
    assert rows[1]["content"] == "ok"
    assert "ts" in rows[0]


def test_ledger_summary_empty_message() -> None:
    summary = ledger_mod.summarize_for_prompt([])
    assert "empty" in summary.lower()


def test_ledger_summary_truncates_per_entry() -> None:
    long = "x" * 1000
    summary = ledger_mod.summarize_for_prompt(
        [{"round": 0, "step": "test", "content": long}]
    )
    # Each entry's content is truncated to 300 chars (per ledger.summarize_for_prompt)
    assert len(summary) < 600


def test_replay_provider_returns_recorded_response(tmp_path: Path) -> None:
    rec = tmp_path / "agent_steps.jsonl"
    rec.write_text(
        json.dumps({"round": 0, "step": "propose", "content": "hello"}) + "\n",
        encoding="utf-8",
    )
    p = ReplayProvider(path=rec)
    out = p.chat([{"role": "user", "content": "ignored"}], step="propose", round=0)
    assert out == "hello"


def test_replay_provider_falls_back_when_unrecorded(tmp_path: Path) -> None:
    rec = tmp_path / "missing.jsonl"
    p = ReplayProvider(path=rec)
    out = p.chat([{"role": "user", "content": "x"}], step="propose", round=0)
    # Falls back to a synthetic-but-structured response
    assert isinstance(out, str)
    assert len(out) > 0


def test_agent_state_typed_dict_keys() -> None:
    s: AgentState = {
        "round": 0,
        "max_rounds": 3,
        "seed": 42,
        "ledger": [],
        "context": {},
        "current_hypothesis": None,
        "current_critique": None,
        "current_experiment": None,
        "next_action": "continue",
    }
    assert s["round"] == 0
    assert s["next_action"] == "continue"
