"""Agent state schema for the LangGraph state machine."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State carried through the agent graph.

    The ledger is the persistent memory cell -- it grows monotonically and
    is the substrate for the cockpit's 'Ask the Memory' panel."""

    round: int
    max_rounds: int
    seed: int
    ledger: list[dict[str, Any]]
    context: dict[str, Any]  # snapshot of regime/signal/graph/forecasts info
    current_hypothesis: dict[str, Any] | None
    current_critique: str | None
    current_experiment: dict[str, Any] | None
    next_action: str  # "continue" or "stop"
