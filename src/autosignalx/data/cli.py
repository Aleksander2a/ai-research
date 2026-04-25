"""CLI subcommand for the data layer.

``autosignalx data fetch`` pulls ETF + macro data per the active config and
writes parquet caches to ``data/cache/``.
``autosignalx data status`` prints what's currently cached."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from autosignalx.config import load_config
from autosignalx.data import cache, fetch

data_app = typer.Typer(
    name="data",
    help="Data pipeline -- fetch and inspect cached OHLCV / macro.",
    no_args_is_help=True,
)
console = Console()


@data_app.command("fetch")
def fetch_cmd(
    config: str = typer.Option("default", help="Config name under configs/."),
    start: str = typer.Option("", help="Override start date (YYYY-MM-DD)."),
    end: str = typer.Option("", help="Override end date (YYYY-MM-DD)."),
) -> None:
    """Pull OHLCV and macro data per the named config and cache to parquet."""
    cfg = load_config(config)["data"]
    start_date = start or cfg["start_date"]
    end_date = end or cfg["end_date"]

    console.print(
        f"Fetching {len(cfg['assets'])} assets and {len(cfg['macro'])} macro signals "
        f"from {start_date} to {end_date}..."
    )
    ohlcv, macro = fetch.fetch_all(cfg["assets"], cfg["macro"], start_date, end_date)
    ohlcv_path = cache.write_ohlcv(ohlcv)
    macro_path = cache.write_macro(macro)
    console.print(f"  wrote {len(ohlcv):>7,} OHLCV rows  -> {ohlcv_path}")
    console.print(f"  wrote {len(macro):>7,} macro rows  -> {macro_path}")


@data_app.command("status")
def status_cmd() -> None:
    """Print what's currently cached under data/cache/."""
    info = cache.cache_status()
    table = Table(title="Data cache", show_lines=False, header_style="bold")
    table.add_column("Frame", style="cyan")
    table.add_column("Status")
    table.add_column("Rows", justify="right")
    table.add_column("Range")
    for name, meta in info.items():
        if not meta.get("exists"):
            table.add_row(name, "missing", "-", "-")
            continue
        rng = f"{meta.get('earliest')} -> {meta.get('latest')}"
        table.add_row(name, "ok", f"{meta.get('rows', 0):,}", rng)
    console.print(table)
