"""Multi-agent debate -- the deepagents-pattern (Iter 12).

A round of debate-mode is structured as:
    Theorist proposes -> Skeptic challenges -> experiment runs -> Adjudicator decides

Each role uses its own DeepInfra model (configurable via env), so the
debate exposes genuinely different "voices" rather than a single LLM
arguing with itself. The Theorist is creative (default Kimi-K2.6),
the Skeptic is critical (default GLM-5.1), the Adjudicator is decisive
(default DeepSeek-V4-Pro).

The library ``deepagents`` provides the SubAgent / planner abstractions
that inspired this pattern; we implement the explicit round flow inside
our existing LangGraph state machine so each phase is observable in the
ledger and replayable in the cockpit."""

from __future__ import annotations

from typing import Any

from autosignalx.agent import ledger as ledger_mod
from autosignalx.agent import prompts
from autosignalx.agent.graph import _safe_parse_json
from autosignalx.agent.llm import get_provider
from autosignalx.agent.state import AgentState


def make_theorist_node(record_replay: bool = False):
    provider = get_provider(record_replay=record_replay, role="theorist")

    def theorist(state: AgentState) -> AgentState:
        rd = state["round"]
        ctx = state.get("context", {})
        ledger_summary = ledger_mod.summarize_for_prompt(state.get("ledger", []))
        msgs = prompts.theorist_messages(ctx, ledger_summary)
        raw = provider.chat(msgs, step="theorist", round=rd, session_id=state.get("session_id"))
        h = _safe_parse_json(raw)
        if not h:
            h = {"hypothesis": "(parse failed)", "experiment": {}, "raw": raw[:500]}
        h["proposer_role"] = "theorist"
        entry = {"round": rd, "step": "theorist", "content": h, "session_id": state.get("session_id")}
        ledger_mod.append(entry)
        state["current_hypothesis"] = h
        state["ledger"] = state.get("ledger", []) + [entry]
        return state

    return theorist


def make_skeptic_node(record_replay: bool = False):
    provider = get_provider(record_replay=record_replay, role="skeptic")

    def skeptic(state: AgentState) -> AgentState:
        rd = state["round"]
        h = state.get("current_hypothesis") or {}
        msgs = prompts.skeptic_messages(h)
        challenge = provider.chat(msgs, step="skeptic", round=rd, session_id=state.get("session_id")).strip()
        entry = {"round": rd, "step": "skeptic", "content": challenge, "session_id": state.get("session_id")}
        ledger_mod.append(entry)
        # Attach challenge to the hypothesis for downstream visibility
        if "skeptic_challenge" not in h:
            h["skeptic_challenge"] = challenge
            state["current_hypothesis"] = h
        state["current_critique"] = challenge  # backward-compat with old field
        state["ledger"] = state.get("ledger", []) + [entry]
        return state

    return skeptic


def make_adjudicator_node(record_replay: bool = False):
    provider = get_provider(record_replay=record_replay, role="adjudicator")

    def adjudicator(state: AgentState) -> AgentState:
        rd = state["round"]
        h = state.get("current_hypothesis") or {}
        challenge = h.get("skeptic_challenge", "")
        experiment = state.get("current_experiment") or {}
        msgs = prompts.adjudicator_messages(h, challenge, experiment)
        verdict = provider.chat(msgs, step="adjudicator", round=rd, session_id=state.get("session_id")).strip()
        entry = {"round": rd, "step": "adjudicator", "content": verdict, "session_id": state.get("session_id")}
        ledger_mod.append(entry)
        state["ledger"] = state.get("ledger", []) + [entry]
        # Routing: stop if max_rounds; else continue
        max_rounds = state.get("max_rounds", 5)
        if rd + 1 >= max_rounds:
            state["next_action"] = "stop"
        else:
            # Adjudicator's verdict text doesn't dictate continue/stop;
            # the round-cap does. Future iters could parse "VERDICT:" for
            # branching strategies (skip / refine / promote-and-continue).
            state["next_action"] = "continue"
        state["round"] = rd + 1
        return state

    return adjudicator


def build_debate_agent_graph(record_replay: bool = False):
    """Compile the debate-mode LangGraph state machine.

    Nodes: theorist -> skeptic -> experiment -> adjudicator -> [theorist | END]
    Each LLM-touching node uses its role-specific provider."""
    from langgraph.graph import END, START, StateGraph

    from autosignalx.agent.graph import experiment_node

    graph = StateGraph(AgentState)
    graph.add_node("theorist", make_theorist_node(record_replay=record_replay))
    graph.add_node("skeptic", make_skeptic_node(record_replay=record_replay))
    graph.add_node("experiment", experiment_node)
    graph.add_node("adjudicator", make_adjudicator_node(record_replay=record_replay))
    graph.add_edge(START, "theorist")
    graph.add_edge("theorist", "skeptic")
    graph.add_edge("skeptic", "experiment")
    graph.add_edge("experiment", "adjudicator")
    graph.add_conditional_edges(
        "adjudicator",
        lambda s: s.get("next_action", "stop"),
        {"continue": "theorist", "stop": END},
    )
    return graph.compile()


def run_debate(
    max_rounds: int = 5,
    seed: int = 42,
    record_replay: bool = False,
    session_id: str | None = None,
) -> list[dict]:
    """Top-level entry for debate mode."""
    from autosignalx.agent import tools
    from autosignalx.agent.findings import make_session_id

    app = build_debate_agent_graph(record_replay=record_replay)
    initial: AgentState = {
        "round": 0,
        "max_rounds": max_rounds,
        "seed": seed,
        "ledger": [],
        "context": tools.context_snapshot(),
        "current_hypothesis": None,
        "current_critique": None,
        "current_experiment": None,
        "next_action": "continue",
        "session_id": session_id or make_session_id(),
    }
    final = app.invoke(initial, {"recursion_limit": max_rounds * 8 + 4})
    return final.get("ledger", [])


def _safe_parse_json_(text: str) -> dict[str, Any]:  # noqa: ARG001
    """Local re-export used by tests; parses JSON with markdown-fence stripping."""
    return _safe_parse_json(text)
