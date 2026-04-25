"""LangGraph state machine for the agent's research loop.

Nodes: propose -> critique -> experiment -> decide -> [propose | END]

Each round writes to the persistent ledger (``reports/agent/ledger.jsonl``)
so cross-session memory works for free. The 'experiment' node is
deterministic (slices cached forecasts via ``agent.tools``); the
'propose', 'critique', and 'decide' nodes call the LLM provider
(``agent.llm``), which routes to live DeepInfra or replay fallback
based on settings."""

from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph

from autosignalx.agent import ledger, prompts, tools
from autosignalx.agent.llm import LLMProvider
from autosignalx.agent.state import AgentState


def _safe_parse_json(text: str) -> dict[str, Any]:
    """Best-effort JSON parse: strip markdown fences if present."""
    s = text.strip()
    if s.startswith("```"):
        # remove first line (```json or ```) and trailing ```
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def make_propose_node(provider: LLMProvider):
    def propose(state: AgentState) -> AgentState:
        rd = state["round"]
        ctx = state.get("context", {})
        ledger_summary = ledger.summarize_for_prompt(state.get("ledger", []))
        msgs = prompts.proposer_messages(ctx, ledger_summary)
        raw = provider.chat(msgs, step="propose", round=rd)
        h = _safe_parse_json(raw)
        if not h:
            h = {"hypothesis": "(parse failed)", "experiment": {}, "raw": raw[:500]}
        entry = {"round": rd, "step": "propose", "content": h, "session_id": state.get("session_id")}
        ledger.append(entry)
        state["current_hypothesis"] = h
        state["ledger"] = state.get("ledger", []) + [entry]
        return state

    return propose


def experiment_node(state: AgentState) -> AgentState:
    rd = state["round"]
    h = state.get("current_hypothesis") or {}
    exp_spec = h.get("experiment", {})
    exp_type = exp_spec.get("type", "slice_forecasts")
    params = exp_spec.get("params", {}) or {}

    if exp_type == "slice_forecasts":
        result = tools.slice_forecasts(
            method=params.get("method"),
            asset=params.get("asset"),
            regime_id=params.get("regime_id"),
        )
        method = params.get("method")
        target_filters = {
            "asset": params.get("asset"),
            "regime_id": params.get("regime_id"),
        }
    elif exp_type == "spawn_method":
        # Agent-authored method via the Iter 13 constrained code-spec DSL.
        spec = params.get("spec", {}) or {}
        result = tools.spawn_method(spec)
        method = spec.get("name") if result.get("status") == "ok" else None
        target_filters = {"asset": None, "regime_id": None}
    else:
        result = {"error": f"unknown experiment type: {exp_type}"}
        method = None
        target_filters = {}

    # Auto-promotion attempt: if a non-naive method now exists, run the
    # significance gate against naive on the same slice and persist a
    # finding when it passes.
    if method and method != "naive":
        sig = tools.test_significance(
            method=method,
            asset=target_filters.get("asset"),
            regime_id=target_filters.get("regime_id"),
        )
        result["significance"] = sig
        if sig.get("promotable"):
            from autosignalx.agent import findings as findings_mod

            finding = findings_mod.promote(
                hypothesis=h.get("hypothesis", ""),
                method=method,
                filters=target_filters,
                evidence=sig["evidence"],
                agent_confidence="auto-promoted by experiment gate",
                round=rd,
                session_id=state.get("session_id", "unknown"),
                parent_hypothesis_ids=h.get("parent_hypothesis_ids", []),
            )
            result["promoted_finding_id"] = finding.get("id")

    entry = {"round": rd, "step": "experiment", "content": result, "session_id": state.get("session_id")}
    ledger.append(entry)
    state["current_experiment"] = result
    state["ledger"] = state.get("ledger", []) + [entry]
    return state


def make_critique_node(provider: LLMProvider):
    def critique(state: AgentState) -> AgentState:
        rd = state["round"]
        h = state.get("current_hypothesis") or {}
        exp = state.get("current_experiment") or {}
        msgs = prompts.critic_messages(h, exp)
        raw = provider.chat(msgs, step="critique", round=rd)
        entry = {"round": rd, "step": "critique", "content": raw.strip(), "session_id": state.get("session_id")}
        ledger.append(entry)
        state["current_critique"] = raw.strip()
        state["ledger"] = state.get("ledger", []) + [entry]
        return state

    return critique


def make_decide_node(provider: LLMProvider):
    def decide(state: AgentState) -> AgentState:
        rd = state["round"]
        max_rounds = state.get("max_rounds", 5)
        if rd + 1 >= max_rounds:
            entry = {"round": rd, "step": "decide", "content": {"action": "stop", "reason": "max_rounds reached"}, "session_id": state.get("session_id")}
            state["next_action"] = "stop"
        else:
            ledger_summary = ledger.summarize_for_prompt(state.get("ledger", []))
            msgs = prompts.decider_messages(ledger_summary, rd, max_rounds)
            raw = provider.chat(msgs, step="decide", round=rd)
            decision = _safe_parse_json(raw)
            action = str(decision.get("action", "continue"))
            if action not in {"continue", "stop"}:
                action = "continue"
            entry = {"round": rd, "step": "decide", "content": decision or {"action": action}, "session_id": state.get("session_id")}
            state["next_action"] = action
        ledger.append(entry)
        state["ledger"] = state.get("ledger", []) + [entry]
        state["round"] = rd + 1
        return state

    return decide


def _route(state: AgentState) -> str:
    return state.get("next_action", "stop")


def build_agent_graph(provider: LLMProvider):
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)
    graph.add_node("propose", make_propose_node(provider))
    graph.add_node("experiment", experiment_node)
    graph.add_node("critique", make_critique_node(provider))
    graph.add_node("decide", make_decide_node(provider))
    graph.add_edge(START, "propose")
    graph.add_edge("propose", "experiment")
    graph.add_edge("experiment", "critique")
    graph.add_edge("critique", "decide")
    graph.add_conditional_edges(
        "decide",
        _route,
        {"continue": "propose", "stop": END},
    )
    return graph.compile()


def run(
    max_rounds: int = 5,
    seed: int = 42,
    record_replay: bool = False,
    session_id: str | None = None,
) -> list[dict]:
    """Top-level entry point. Returns the full ledger after the run."""
    from autosignalx.agent.findings import make_session_id
    from autosignalx.agent.llm import get_provider

    provider = get_provider(record_replay=record_replay)
    app = build_agent_graph(provider)
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
    final = app.invoke(initial, {"recursion_limit": max_rounds * 6 + 4})
    return final.get("ledger", [])
