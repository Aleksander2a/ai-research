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


def _resolve_universe(config: str, study: str, start: str, end: str):
    """Common resolver for fetch_cmd: returns (assets, macro, start, end, cache_root)."""
    if study:
        from autosignalx.study import Study

        s = Study.load(study)
        return (
            list(s.assets),
            list(s.macro),
            start or s.start_date,
            end or s.end_date,
            s.cache_dir,
        )
    cfg = load_config(config)["data"]
    return (
        list(cfg["assets"]),
        list(cfg["macro"]),
        start or cfg["start_date"],
        end or cfg["end_date"],
        None,
    )


@data_app.command("fetch")
def fetch_cmd(
    config: str = typer.Option("default", help="Config name under configs/."),
    study: str = typer.Option("", help="Study name (overrides --config)."),
    start: str = typer.Option("", help="Override start date (YYYY-MM-DD)."),
    end: str = typer.Option("", help="Override end date (YYYY-MM-DD)."),
) -> None:
    """Pull OHLCV and macro data per the named config or study."""
    assets, macro_tickers, start_date, end_date, cache_root = _resolve_universe(
        config, study, start, end
    )
    label = f"study={study}" if study else f"config={config}"
    console.print(
        f"Fetching {len(assets)} assets and {len(macro_tickers)} macro signals "
        f"({label}) from {start_date} to {end_date}..."
    )
    ohlcv, macro_df = fetch.fetch_all(assets, macro_tickers, start_date, end_date)
    ohlcv_path = cache.write_ohlcv(ohlcv, cache_root=cache_root)
    macro_path = cache.write_macro(macro_df, cache_root=cache_root)
    console.print(f"  wrote {len(ohlcv):>7,} OHLCV rows  -> {ohlcv_path}")
    console.print(f"  wrote {len(macro_df):>7,} macro rows  -> {macro_path}")


@data_app.command("status")
def status_cmd(
    study: str = typer.Option("", help="Inspect a study's cache instead of default."),
) -> None:
    """Print what's currently cached."""
    cache_root = None
    if study:
        from autosignalx.study import Study

        cache_root = Study.load(study).cache_dir
    info = cache.cache_status(cache_root=cache_root)
    title = f"Data cache (study={study})" if study else "Data cache"
    table = Table(title=title, show_lines=False, header_style="bold")
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
