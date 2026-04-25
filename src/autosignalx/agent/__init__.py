"""Agentic layer (L5) -- LangGraph state machine + persistent ledger.

Public API:
- ``state.AgentState`` -- TypedDict for the LangGraph state
- ``ledger.append(entry)`` / ``load()`` / ``clear()`` -- persistent JSONL memory
- ``tools.slice_forecasts(method, asset, regime_id)`` -- the agent's primary
  experiment tool
- ``tools.context_snapshot()`` -- compact snapshot of all artifact summaries
- ``llm.get_provider(record_replay)`` -- LiveProvider (DeepInfra) or
  ReplayProvider (no-key fallback)
- ``graph.build_agent_graph(provider)`` -- compiles the LangGraph
- ``graph.run(max_rounds, seed, record_replay)`` -- runs the loop end-to-end

The ledger is the persistent memory cell. The cockpit's 'Ask the Memory'
panel reads from it to answer questions about past agent reasoning."""

from autosignalx.agent import (  # noqa: F401
    debate,
    findings,
    graph,
    ledger,
    lineage,
    llm,
    memory,
    prompts,
    specs,
    state,
    telemetry,
    tools,
    trace_eval,
)
from autosignalx.agent.graph import build_agent_graph, run  # noqa: F401
from autosignalx.agent.ledger import append, clear, load, summarize_for_prompt  # noqa: F401
from autosignalx.agent.llm import LiveProvider, ReplayProvider, get_provider  # noqa: F401
from autosignalx.agent.tools import context_snapshot, slice_forecasts  # noqa: F401
