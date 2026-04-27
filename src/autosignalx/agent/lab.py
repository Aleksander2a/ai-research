"""Phase 14 -- Specialist research lab orchestration.

Composes the specialist roles into a planner-driven research loop:

    START -> Theorist -> Verifier -> [Skeptic + Specialist consults]
            -> experiment -> Adjudicator -> [Theorist | END]

Each round runs at most ``n_specialist_consults`` consultations beyond
the existing Theorist/Skeptic/Adjudicator triad. The PrincipalInvestigator
LLM picks which specialist to consult given the round's state.

Every consult and KG update is recorded in the ledger so the cockpit
can render the full multi-specialist trace.
"""

from __future__ import annotations

from typing import Any

from autosignalx.agent import (
    findings as findings_mod,
)
from autosignalx.agent import (
    knowledge_graph as kg_mod,
)
from autosignalx.agent import (
    ledger as ledger_mod,
)
from autosignalx.agent import (
    specialists as specialists_mod,
)
from autosignalx.agent import (
    verifier as verifier_mod,
)
from autosignalx.agent.debate import (
    make_adjudicator_node,
    make_skeptic_node,
    make_theorist_node,
)
from autosignalx.agent.graph import _safe_parse_json, experiment_node
from autosignalx.agent.llm import get_provider
from autosignalx.agent.state import AgentState

DEFAULT_SPECIALISTS = ("statistician", "quant", "economist")


def make_verifier_node():
    """Verifier checks the latest hypothesis carries a real pre-registration."""

    def verifier(state: AgentState) -> AgentState:
        rd = state["round"]
        h = state.get("current_hypothesis") or {}
        result = verifier_mod.verify_hypothesis(h)
        entry = {
            "round": rd,
            "step": "verifier",
            "content": {
                "ok": result.ok,
                "missing": result.missing,
                "downgrades": result.downgrades,
            },
            "session_id": state.get("session_id"),
        }
        ledger_mod.append(entry)
        state["ledger"] = state.get("ledger", []) + [entry]
        # Augment the hypothesis with the verifier verdict so downstream
        # roles can see it.
        h["_verifier"] = entry["content"]
        state["current_hypothesis"] = h
        # Phase 8: register the hypothesis in the pre-registration ledger
        try:
            from autosignalx.eval import preregistration as prereg

            p = prereg.from_hypothesis_dict(
                h, session_id=state.get("session_id"), round=rd, proposer_role="theorist"
            )
            prereg.register(p)
        except Exception:  # noqa: BLE001
            pass
        return state

    return verifier


def make_planner_node(specialists: tuple[str, ...] = DEFAULT_SPECIALISTS, record_replay: bool = False):
    """Picks the next specialist to consult or chooses to skip the consult phase."""
    provider = get_provider(record_replay=record_replay, role="adjudicator")

    def planner(state: AgentState) -> AgentState:
        rd = state["round"]
        h = state.get("current_hypothesis") or {}
        ledger_summary = ledger_mod.summarize_for_prompt(state.get("ledger", [])[-12:])
        kg_sum = kg_mod.kg_summary()
        payload = {
            "current_hypothesis": h,
            "open_specialist_pool": list(specialists),
            "ledger_tail": ledger_summary,
            "kg_summary": kg_sum,
        }
        import json

        msgs = [
            {"role": "system", "content": specialists_mod.PRINCIPAL_INVESTIGATOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    "## State\n"
                    f"```json\n{json.dumps(payload, indent=2, default=str)[:2500]}\n```\n\n"
                    "Pick exactly one specialist."
                ),
            },
        ]
        raw = provider.chat(msgs, step="principal_investigator", round=rd)
        decision = _safe_parse_json(raw)
        next_role = str(decision.get("next_specialist", "")).strip().lower()
        if next_role not in specialists:
            # Fallback rotation across the default pool by round
            next_role = specialists[rd % len(specialists)]

        state["next_specialist"] = next_role
        entry = {
            "round": rd,
            "step": "principal_investigator",
            "content": {"next_specialist": next_role, "rationale": decision.get("rationale", "")},
            "session_id": state.get("session_id"),
        }
        ledger_mod.append(entry)
        state["ledger"] = state.get("ledger", []) + [entry]
        return state

    return planner


def make_specialist_node(record_replay: bool = False):
    """Runs the specialist chosen by the planner."""

    def specialist(state: AgentState) -> AgentState:
        rd = state["round"]
        role = state.get("next_specialist", "statistician")
        h = state.get("current_hypothesis") or {}
        # Pull the matching evidence (if any) from the latest ledger entries
        recent = state.get("ledger", [])[-8:]
        payload = {"hypothesis": h, "ledger_tail": recent}
        consult = specialists_mod.consult_specialist(
            role=role,
            payload=payload,
            record_replay=record_replay,
            round_n=rd,
        )
        entry = {
            "round": rd,
            "step": f"specialist:{role}",
            "content": consult.response,
            "session_id": state.get("session_id"),
        }
        ledger_mod.append(entry)
        state["ledger"] = state.get("ledger", []) + [entry]
        return state

    return specialist


def make_kg_writer_node():
    """After adjudication, write any newly-promoted finding to the KG."""

    def kg_writer(state: AgentState) -> AgentState:
        rd = state["round"]
        # Re-read the entire findings store; idempotent ingest handles duplicates.
        all_findings = findings_mod.load()
        result = kg_mod.ingest_findings(all_findings)
        entry = {
            "round": rd,
            "step": "kg_writer",
            "content": result,
            "session_id": state.get("session_id"),
        }
        ledger_mod.append(entry)
        state["ledger"] = state.get("ledger", []) + [entry]
        return state

    return kg_writer


def build_lab_agent_graph(
    record_replay: bool = False,
    specialists: tuple[str, ...] = DEFAULT_SPECIALISTS,
):
    """Lab-mode LangGraph state machine.

    Theorist -> Verifier -> Planner -> Specialist -> Skeptic ->
    experiment -> Adjudicator -> KG-writer -> [Theorist | END].

    Every node writes its own ledger entry; specialist consult and
    KG-writer entries appear in the cockpit's Specialist Council
    panel."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    graph.add_node("theorist", make_theorist_node(record_replay=record_replay))
    graph.add_node("verifier", make_verifier_node())
    graph.add_node("planner", make_planner_node(specialists=specialists, record_replay=record_replay))
    graph.add_node("specialist", make_specialist_node(record_replay=record_replay))
    graph.add_node("skeptic", make_skeptic_node(record_replay=record_replay))
    graph.add_node("experiment", experiment_node)
    graph.add_node("adjudicator", make_adjudicator_node(record_replay=record_replay))
    graph.add_node("kg_writer", make_kg_writer_node())

    graph.add_edge(START, "theorist")
    graph.add_edge("theorist", "verifier")
    graph.add_edge("verifier", "planner")
    graph.add_edge("planner", "specialist")
    graph.add_edge("specialist", "skeptic")
    graph.add_edge("skeptic", "experiment")
    graph.add_edge("experiment", "adjudicator")
    graph.add_edge("adjudicator", "kg_writer")
    graph.add_conditional_edges(
        "kg_writer",
        lambda s: s.get("next_action", "stop"),
        {"continue": "theorist", "stop": END},
    )
    return graph.compile()


def run_lab(
    max_rounds: int = 5,
    seed: int = 42,
    record_replay: bool = False,
    session_id: str | None = None,
    specialists: tuple[str, ...] = DEFAULT_SPECIALISTS,
) -> list[dict[str, Any]]:
    """Top-level entry point for lab mode."""
    from autosignalx.agent import tools
    from autosignalx.agent.findings import make_session_id

    app = build_lab_agent_graph(record_replay=record_replay, specialists=specialists)
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
    final = app.invoke(initial, {"recursion_limit": max_rounds * 14 + 8})
    return final.get("ledger", [])
