"""Tests for the self-critique module (Iter 19)."""

from __future__ import annotations

import json
from pathlib import Path

from autosignalx.agent import self_critique
from autosignalx.agent.llm import ReplayProvider


def test_critique_finding_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(self_critique, "CRITIQUE_DIR", tmp_path)
    self_critique.clear()
    rec = tmp_path / "agent_steps.jsonl"
    rec.write_text(
        json.dumps(
            {
                "round": -1,
                "step": "self_critique",
                "content": json.dumps(
                    {
                        "current_state": "reinforced",
                        "rationale": "later sessions replicated this finding twice",
                    }
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provider = ReplayProvider(path=rec)
    out = self_critique.critique_finding(
        {"id": "f_xxx", "method": "m"},
        "ledger blah",
        "other findings blah",
        provider=provider,
    )
    assert out["finding_id"] == "f_xxx"
    assert out["current_state"] == "reinforced"
    rows = self_critique.load()
    assert len(rows) == 1


def test_critique_finding_unparseable_response_defaults_to_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(self_critique, "CRITIQUE_DIR", tmp_path)
    self_critique.clear()
    rec = tmp_path / "agent_steps.jsonl"
    rec.write_text(
        json.dumps({"round": -1, "step": "self_critique", "content": "garbage"}) + "\n",
        encoding="utf-8",
    )
    provider = ReplayProvider(path=rec)
    out = self_critique.critique_finding(
        {"id": "f_yyy"}, "", "", provider=provider
    )
    # Default fallback when JSON parse fails
    assert out["current_state"] == "unchanged"
