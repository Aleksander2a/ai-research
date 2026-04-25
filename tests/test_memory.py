"""Tests for the long-horizon memory consolidation (Iter 16)."""

from __future__ import annotations

import json
from pathlib import Path

from autosignalx.agent import memory
from autosignalx.agent.llm import ReplayProvider


def test_load_lessons_when_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(memory, "LESSONS_DIR", tmp_path)
    assert memory.load_lessons() == ""


def test_append_then_load_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(memory, "LESSONS_DIR", tmp_path)
    memory.clear()
    memory.append_to_lessons("## Session s1 -- 2026-04-26\n\n**What was tried**: x.")
    memory.append_to_lessons("## Session s2 -- 2026-04-27\n\n**What was tried**: y.")
    text = memory.load_lessons()
    assert "Session s1" in text
    assert "Session s2" in text
    # Sections separated
    assert text.count("---") >= 1


def test_load_lessons_truncates_to_max_chars(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(memory, "LESSONS_DIR", tmp_path)
    memory.clear()
    big = "x" * 5000
    memory.append_to_lessons(f"## Session s -- 2026-04-26\n\n{big}")
    out = memory.load_lessons(max_chars=2000)
    assert len(out) <= 3000  # truncation prelude + tail


def test_consolidate_with_replay_provider(tmp_path: Path) -> None:
    rec = tmp_path / "agent_steps.jsonl"
    rec.write_text(
        json.dumps(
            {
                "round": -1,
                "step": "consolidate",
                "content": "## Session s1 -- 2026-04-26\n\n**What was tried**: chronos2 on EFA.\n\n**What worked**: f_abc.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provider = ReplayProvider(path=rec)
    section = memory.consolidate(
        session_id="s1",
        ledger_entries=[],
        finding_records=[],
        provider=provider,
    )
    assert "Session s1" in section
    assert "chronos2" in section


def test_consolidate_and_append_persists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(memory, "LESSONS_DIR", tmp_path)
    memory.clear()
    rec = tmp_path / "agent_steps.jsonl"
    rec.write_text(
        json.dumps(
            {
                "round": -1,
                "step": "consolidate",
                "content": "## Session s1 -- 2026-04-26\n\n**Body**: lorem.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provider = ReplayProvider(path=rec)
    path, section = memory.consolidate_and_append(
        session_id="s1",
        ledger_entries=[],
        finding_records=[],
        provider=provider,
    )
    assert path.exists()
    assert "Session s1" in path.read_text(encoding="utf-8")
