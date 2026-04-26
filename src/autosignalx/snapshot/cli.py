"""Typer subcommands for the snapshot layer."""

from __future__ import annotations

import typer
from rich.console import Console

from autosignalx.snapshot import builder

snapshot_app = typer.Typer(name="snapshot", help="Static HTML snapshot of the cockpit.")
console = Console()


@snapshot_app.command("build")
def build_cmd() -> None:
    """Render the multi-page HTML snapshot from current artifacts."""
    result = builder.build_snapshot()
    console.print(
        f"Wrote [bold]{len(result.pages_written)}[/bold] pages "
        f"({result.figures} figures) to {result.out_dir}."
    )
    for p in result.pages_written:
        console.print(f"  - {p}")
