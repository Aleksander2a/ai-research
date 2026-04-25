"""Typer-based CLI dispatcher.

Each layer registers its subcommands as the iteration that builds it lands.
At Iter 2 the CLI exposes ``version``, ``status``, the ``data`` sub-app
(``fetch``, ``status``), and the ``eval`` sub-app (``baseline``, ``status``).
Later iterations add ``forecast`` (Chronos-2), ``regime``, ``signal``,
``graph``, ``agent``, and ``report``."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from autosignalx import __version__
from autosignalx.config import settings
from autosignalx.data.cli import data_app
from autosignalx.eval.cli import eval_app
from autosignalx.regime.cli import regime_app
from autosignalx.signal.cli import signal_app

app = typer.Typer(
    name="autosignalx",
    help="AutoSignal-X — modular AI research instrument.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(regime_app, name="regime")
app.add_typer(signal_app, name="signal")
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

    # Data cache state
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

    # Ablation cache state
    ablations_dir = settings.reports_dir / "ablations"
    if ablations_dir.exists():
        files = sorted(ablations_dir.glob("*.parquet"))
        if files:
            names = ", ".join(p.stem for p in files)
            console.print(f"  ablations:   {names}")
        else:
            console.print("  ablations:   empty -- run 'autosignalx eval baseline'")
    else:
        console.print("  ablations:   none yet")

    console.print()

    table = Table(title="Layer status", show_lines=False, header_style="bold")
    table.add_column("Layer", style="cyan")
    table.add_column("Status")
    table.add_column("Lands in")

    layers = [
        ("L1 Forecasting", "ok (baselines + chronos-2)", "complete in Iter 3"),
        ("L2 Representation", "ok (contrastive + KMeans + HMM)", "complete in Iter 4"),
        ("L3 Reasoning", "ok (TabPFN + permutation importance)", "complete in Iter 5"),
        ("L4 Relational", "pending", "Iter 6 -- graph"),
        ("L5 Agentic", "pending", "Iter 7 -- agent"),
    ]
    for name, status_, lands_in in layers:
        table.add_row(name, status_, lands_in)

    console.print(table)


if __name__ == "__main__":
    app()
