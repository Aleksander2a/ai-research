"""Typer subcommands for the chat layer."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from autosignalx.chat import answer as answer_mod
from autosignalx.chat import index as index_mod

chat_app = typer.Typer(name="chat", help="Conversational explainability over run artifacts.")
console = Console()


@chat_app.command("index")
def index_cmd(
    force_hashed: bool = typer.Option(
        False, "--force-hashed", help="Use deterministic hashed-bag embeddings even if a key is set."
    ),
) -> None:
    """(Re)build the chat index over the current artifacts."""
    idx = index_mod.build_index(force_hashed=force_hashed)
    console.print(
        f"Indexed [bold]{len(idx.chunks)}[/bold] chunks "
        f"(mode={idx.mode}, model={idx.model}). Saved to "
        f"{index_mod.index_dir()}."
    )


@chat_app.command("status")
def status_cmd() -> None:
    """Print index inventory."""
    idx = index_mod.load_index()
    if idx is None:
        console.print("[yellow]No chat index found. Run `autosignalx chat index`.[/yellow]")
        return
    console.print(f"Index: {len(idx.chunks)} chunks, mode={idx.mode}, model={idx.model}")
    counts: dict[str, int] = {}
    for c in idx.chunks:
        counts[c.kind] = counts.get(c.kind, 0) + 1
    t = Table(title="By kind")
    t.add_column("kind")
    t.add_column("count", justify="right")
    for k, v in sorted(counts.items()):
        t.add_row(k, str(v))
    console.print(t)


@chat_app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="Question to ask the agent's memory."),
    k: int = typer.Option(6, "--k"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Ask one question against the chat index."""
    a = answer_mod.answer_question(question, k=k)
    if json_output:
        console.print(json.dumps(answer_mod.answer_to_jsonable(a), default=str))
        return
    console.print(f"[bold]Mode:[/bold] {a.mode}")
    console.print(a.text)
    if a.citations:
        console.print(f"\n[dim]Citations: {', '.join(a.citations)}[/dim]")


@chat_app.command("eval")
def eval_cmd(json_output: bool = typer.Option(False, "--json")) -> None:
    """Run the grounding eval set."""
    from autosignalx.chat import eval as eval_mod

    summary = eval_mod.run_eval()
    if json_output:
        console.print(json.dumps(summary, default=str))
        return
    console.print(
        f"[bold]Eval:[/bold] {summary['n']} questions, "
        f"recall={summary['citation_recall']:.2f}, "
        f"refusal_correct={summary['refusal_correct']:.2f}"
    )
    for r in summary["results"]:
        ok = "OK" if r["passed"] else "FAIL"
        console.print(f"  [{ok}] {r['question'][:80]}")
