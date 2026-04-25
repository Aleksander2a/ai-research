"""Agentic layer (L5) — LangGraph state machine using the deepagents pattern.

Implementation lands in **Iter 7**. The agent reads the experiment ledger,
proposes hypotheses, runs experiments via the eval harness, critiques its
own results, and decides what to try next. The ledger is the system's
persistent memory; it is also exposed via an "Ask the Memory" panel in
the cockpit. See README for the iteration plan."""
