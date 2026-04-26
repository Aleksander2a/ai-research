"""CLI subcommand for the agentic layer."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from autosignalx.agent import debate as debate_mod
from autosignalx.agent import graph, ledger
from autosignalx.config import settings

agent_app = typer.Typer(
    name="agent",
    help="Agentic research loop -- LangGraph state machine + persistent ledger.",
    no_args_is_help=True,
)
console = Console()


@agent_app.command("run")
def run_cmd(
    max_rounds: int = typer.Option(5, help="Number of propose/critique/experiment rounds."),
    seed: int = typer.Option(42, help="Random seed."),
    fresh: bool = typer.Option(False, help="Wipe the ledger before starting."),
    record_replay: bool = typer.Option(
        False, help="Append live LLM responses to replay/agent_steps.jsonl."
    ),
    mode: str = typer.Option(
        "single",
        help="'single' (one LLM does propose/critique/decide) or 'debate' "
        "(Theorist/Skeptic/Adjudicator multi-role debate per round, Iter 12).",
    ),
) -> None:
    """Run the agent's research loop for ``max_rounds`` rounds.

    In live mode (DEEPINFRA_API_KEY set, AUTOSIGNALX_REPLAY != true) the
    agent uses DeepInfra-hosted open-source LLMs. In replay mode the agent
    plays back recorded responses from replay/agent_steps.jsonl, falling
    back to deterministic plausible responses if the file is incomplete.
    Either way, the experiment step is deterministic (slices cached
    forecasts) and the ledger captures every step."""
    if fresh:
        ledger.clear()
        console.print("Ledger cleared.")

    runtime_mode = "replay" if settings.use_replay else "live"
    console.print(
        f"Starting agent loop ({max_rounds} rounds, mode={mode}, "
        f"runtime={runtime_mode}, record_replay={record_replay})..."
    )
    if mode == "debate":
        entries = debate_mod.run_debate(
            max_rounds=max_rounds, seed=seed, record_replay=record_replay
        )
    else:
        entries = graph.run(
            max_rounds=max_rounds, seed=seed, record_replay=record_replay
        )
    console.print(f"Agent finished. Ledger now has {len(entries)} entries.")

    table = Table(title="Agent ledger summary", header_style="bold")
    table.add_column("Round", justify="right")
    table.add_column("Step", style="cyan")
    table.add_column("Content (head)", overflow="fold")
    for e in entries[-min(20, len(entries)) :]:
        c = e.get("content", "")
        c_str = (
            json.dumps(c, default=str)[:120] if isinstance(c, dict) else str(c)[:120]
        )
        table.add_row(str(e.get("round", "")), str(e.get("step", "")), c_str)
    console.print(table)


@agent_app.command("consolidate")
def consolidate_cmd(
    session_id: str = typer.Option(
        "current", help="Session ID for the lessons section header."
    ),
) -> None:
    """Consolidate the current ledger + findings into a Markdown lessons
    section, appended to reports/agent/lessons.md."""
    from autosignalx.agent import memory as memory_mod

    path, section = memory_mod.consolidate_and_append(session_id=session_id)
    console.print(f"Wrote {len(section)} chars to {path}")
    console.print(section[:600] + ("..." if len(section) > 600 else ""))


@agent_app.command("score-traces")
def score_traces_cmd(
    session_id: str = typer.Option(
        "current",
        help="Session ID to score; 'current' = the entire current ledger.",
    ),
) -> None:
    """Score every round of the current ledger via LLM-as-judge."""
    from autosignalx.agent import trace_eval

    entries = ledger.load()
    if not entries:
        console.print("Ledger is empty.")
        return
    console.print(
        f"Scoring {max(int(e.get('round', 0)) for e in entries) + 1} rounds "
        f"({len(entries)} ledger entries)..."
    )
    scores = trace_eval.score_session(entries, session_id=session_id)
    for s in scores:
        console.print(
            f"  round {s['round']}: clarity={s.get('clarity')} "
            f"novelty={s.get('novelty')} falsifiability={s.get('falsifiability')} "
            f"evidence_citing={s.get('evidence_citing')}  --  {s.get('rationale', '')[:80]}"
        )


@agent_app.command("self-critique")
def self_critique_cmd() -> None:
    """Run self-critique over every promoted finding."""
    from autosignalx.agent import self_critique as sc

    findings = __import__("autosignalx.agent", fromlist=["findings"]).findings.load()
    if not findings:
        console.print("No promoted findings to critique.")
        return
    console.print(f"Self-critiquing {len(findings)} promoted findings...")
    out = sc.critique_all_findings()
    for r in out:
        console.print(f"  {r['finding_id']}: {r['current_state']}  --  {r['rationale'][:80]}")


@agent_app.command("harden")
def harden_cmd(
    fdr_alpha: float = typer.Option(0.10, help="BH-FDR target rate."),
) -> None:
    """Re-evaluate every promoted finding under FDR + adversarial replication.

    Writes ``reports/agent/survival.jsonl`` with the per-finding survival
    record. Intended to run after a session completes (or as part of the
    scheduled-session pipeline)."""
    from autosignalx.eval.survival import harden_findings

    records = harden_findings(fdr_alpha=fdr_alpha)
    if not records:
        console.print("[yellow]No promoted findings to harden.[/yellow]")
        return

    n = len(records)
    n_fdr = sum(1 for r in records if r.get("survives_fdr"))
    n_full = sum(1 for r in records if r.get("survives_full_test"))
    n_placebo = sum(1 for r in records if r.get("survives_placebo"))
    n_block = sum(1 for r in records if r.get("survives_block_holdout"))
    n_all = sum(1 for r in records if r.get("survives_all"))
    console.print(
        f"Hardened [bold]{n}[/bold] findings -- "
        f"FDR (q<={fdr_alpha}): {n_fdr}/{n}, "
        f"full-test: {n_full}/{n}, "
        f"placebo-rejected: {n_placebo}/{n}, "
        f"block-holdout: {n_block}/{n}, "
        f"survives all: [bold]{n_all}/{n}[/bold]."
    )


@agent_app.command("status")
def status_cmd() -> None:
    """Print ledger size and last-round summary."""
    entries = ledger.load()
    console.print(f"Ledger entries: {len(entries)}")
    if not entries:
        return
    last_round = max(e.get("round", 0) for e in entries)
    console.print(f"Last round: {last_round}")
    p = settings.reports_dir / "agent" / "ledger.jsonl"
    console.print(f"Path: {p}")
