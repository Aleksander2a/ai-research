"""Typer-based CLI dispatcher.

Each layer registers its subcommands as the iteration that builds it lands.
At Iter 1 the CLI exposes ``version``, ``status``, and the ``data`` sub-app
(``fetch``, ``status``). Later iterations add ``forecast``, ``regime``,
``signal``, ``graph``, ``agent``, and ``report``."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from autosignalx import __version__
from autosignalx.config import settings
from autosignalx.data.cli import data_app

app = typer.Typer(
    name="autosignalx",
    help="AutoSignal-X — modular AI research instrument.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")
console = Console()


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"autosignalx [bold]{__version__}[/bold]")


@app.command()
def status() -> None:
    """Show iteration progress, data cache state, and layer availability."""
    console.print(f"[bold]AutoSignal-X[/bold] v{__version__}")
    console.print(f"  repo root:   {settings.repo_root}")
    console.print(f"  data dir:    {settings.data_dir}")
    console.print(f"  replay mode: {settings.use_replay}")

    # Data cache state -- gracefully degrade if the data layer hasn't landed yet.
    try:
        from autosignalx.data import cache

        info = cache.cache_status()
        if info["ohlcv"].get("exists"):
            ohlcv_summary = (
                f"OHLCV {info['ohlcv']['rows']:,} rows "
                f"({info['ohlcv'].get('earliest')} -> {info['ohlcv'].get('latest')})"
            )
            macro_rows = info["macro"].get("rows", 0) if info["macro"].get("exists") else 0
            console.print(f"  data cache:  {ohlcv_summary}, macro {macro_rows:,} rows")
        else:
            console.print("  data cache:  empty -- run 'autosignalx data fetch'")
    except Exception as e:  # noqa: BLE001
        console.print(f"  data cache:  unavailable ({e})")

    console.print()

    table = Table(title="Layer status", show_lines=False, header_style="bold")
    table.add_column("Layer", style="cyan")
    table.add_column("Status")
    table.add_column("Lands in")

    layers = [
        ("L1 Forecasting", "pending", "Iter 3 -- chronos-2"),
        ("L2 Representation", "pending", "Iter 4 -- regime"),
        ("L3 Reasoning", "pending", "Iter 5 -- signal"),
        ("L4 Relational", "pending", "Iter 6 -- graph"),
        ("L5 Agentic", "pending", "Iter 7 -- agent"),
    ]
    for name, status_, lands_in in layers:
        table.add_row(name, status_, lands_in)

    console.print(table)


if __name__ == "__main__":
    app()
