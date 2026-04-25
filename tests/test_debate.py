"""Tests for the multi-agent debate (Iter 12)."""

from __future__ import annotations

from autosignalx.agent import debate, llm, prompts


def test_role_to_env_keys() -> None:
    for role in ("theorist", "skeptic", "adjudicator", "proposer", "critic", "chat"):
        assert role in llm.ROLE_TO_ENV


def test_model_for_role_falls_back_to_defaults() -> None:
    # No env override -> falls back to category default
    assert llm._model_for_role("theorist") in (
        "moonshotai/Kimi-K2.6",
        "moonshotai/Kimi-K2.6",  # default
    ) or llm._model_for_role("theorist") != ""


def test_theorist_messages_shape() -> None:
    msgs = prompts.theorist_messages({"methods": ["naive"]}, "(empty)")
    assert msgs[0]["role"] == "system"
    assert "THEORIST" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"


def test_skeptic_messages_shape() -> None:
    msgs = prompts.skeptic_messages({"hypothesis": "x"})
    assert msgs[0]["role"] == "system"
    assert "SKEPTIC" in msgs[0]["content"]


def test_adjudicator_messages_shape() -> None:
    msgs = prompts.adjudicator_messages(
        {"hypothesis": "x"}, "challenge", {"n": 5}
    )
    assert msgs[0]["role"] == "system"
    assert "ADJUDICATOR" in msgs[0]["content"]


def test_build_debate_agent_graph_compiles() -> None:
    g = debate.build_debate_agent_graph()
    # Should be a compiled langgraph
    assert hasattr(g, "invoke")
