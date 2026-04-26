"""CLI subcommand for the study layer.

``autosignalx study create``  Create a new study config on disk.
``autosignalx study list``    List existing studies.
``autosignalx study show``    Print a study's resolved config + paths.
``autosignalx study delete``  Remove a study and all its artifacts.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from autosignalx.study.config import (
    DEFAULT_ASSETS,
    DEFAULT_MACRO,
    Study,
    StudyExistsError,
    StudyNotFoundError,
    list_studies,
)

study_app = typer.Typer(
    name="study",
    help="User-defined studies -- run the pipeline on your own assets and dates.",
    no_args_is_help=True,
)
console = Console()


@study_app.command("create")
def create_cmd(
    name: str = typer.Option(..., help="Study name (alphanumeric, _, -)."),
    assets: str = typer.Option(
        ",".join(DEFAULT_ASSETS),
        help="Comma-separated ticker list.",
    ),
    macro: str = typer.Option(
        ",".join(DEFAULT_MACRO),
        help="Comma-separated macro covariate tickers.",
    ),
    start: str = typer.Option("2010-01-01", help="Universe start date."),
    end: str = typer.Option("2025-12-31", help="Universe end date."),
    train_end: str = typer.Option("2018-12-31"),
    val_end: str = typer.Option("2020-12-31"),
    test_end: str = typer.Option("2025-12-31"),
    n_regimes: int = typer.Option(4),
    horizon: int = typer.Option(21, help="Forecast horizon (bars)."),
    step: int = typer.Option(21, help="Walk-forward step (bars)."),
    cost_bps: float = typer.Option(5.0),
    description: str = typer.Option("", help="Optional free-form description."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing study."),
) -> None:
    """Create a new study and write its config to disk."""
    s = Study(
        name=name,
        description=description,
        assets=[a.strip() for a in assets.split(",") if a.strip()],
        macro=[m.strip() for m in macro.split(",") if m.strip()],
        start_date=start,
        end_date=end,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        n_regimes=n_regimes,
        forecast_horizon_days=horizon,
        rolling_step_days=step,
        cost_bps=cost_bps,
    )
    try:
        path = s.save(overwrite=overwrite)
    except StudyExistsError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Created study[/green] [cyan]{name}[/cyan] -> {path}")
    console.print(f"  cache:    {s.cache_dir}")
    console.print(f"  reports:  {s.reports_root}")
    console.print(
        "  Next: [bold]autosignalx data fetch --study " f"{name}[/bold]"
    )


@study_app.command("list")
def list_cmd() -> None:
    """List all studies on disk."""
    names = list_studies()
    if not names:
        console.print("No studies yet. Create one with `autosignalx study create`.")
        return
    table = Table(title="Studies", show_lines=False, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Assets")
    table.add_column("Range")
    table.add_column("Description")
    for n in names:
        s = Study.load(n)
        assets_summary = ", ".join(s.assets[:5]) + (
            f" +{len(s.assets) - 5}" if len(s.assets) > 5 else ""
        )
        table.add_row(
            n,
            assets_summary,
            f"{s.start_date} -> {s.end_date}",
            s.description or "-",
        )
    console.print(table)


@study_app.command("show")
def show_cmd(name: str = typer.Argument(..., help="Study name.")) -> None:
    """Print a study's full config and resolved paths."""
    try:
        s = Study.load(name)
    except StudyNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[bold cyan]{s.name}[/bold cyan]")
    if s.description:
        console.print(f"  {s.description}")
    console.print(f"  assets:        {', '.join(s.assets)}")
    console.print(f"  macro:         {', '.join(s.macro)}")
    console.print(f"  range:         {s.start_date} -> {s.end_date}")
    console.print(
        f"  splits:        train<={s.train_end}, val<={s.val_end}, test<={s.test_end}"
    )
    console.print(
        f"  forecast:      horizon={s.forecast_horizon_days}d, step={s.rolling_step_days}d"
    )
    console.print(f"  regimes:       n={s.n_regimes}")
    console.print(f"  cost (bps):    {s.cost_bps}")
    console.print(f"  backtest from: {s.effective_backtest_start}")
    console.print(f"  root:          {s.root}")
    console.print(f"  cache:         {s.cache_dir}")
    console.print(f"  reports:       {s.reports_root}")


@study_app.command("validate")
def validate_cmd(
    name: str = typer.Argument(..., help="Study name."),
    check_tickers: bool = typer.Option(
        False, "--check-tickers",
        help="Also probe yfinance availability for each ticker (network).",
    ),
) -> None:
    """Run pre-flight checks on a study config (date ordering, window count,
    universe size, optional ticker availability)."""
    from autosignalx.study import validation

    try:
        s = Study.load(name)
    except StudyNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None
    report = validation.validate(s, check_tickers=check_tickers)

    for msg in report.info:
        console.print(f"  [dim]info:[/dim]  {msg}")
    for msg in report.warnings:
        console.print(f"  [yellow]warn:[/yellow]  {msg}")
    for msg in report.errors:
        console.print(f"  [red]error:[/red] {msg}")
    if report.ok:
        console.print(f"[green]Validation OK[/green] for study [cyan]{name}[/cyan].")
    else:
        console.print(
            f"[red]Validation failed[/red] for study [cyan]{name}[/cyan] "
            f"({len(report.errors)} error(s))."
        )
        raise typer.Exit(code=1)


@study_app.command("delete")
def delete_cmd(
    name: str = typer.Argument(..., help="Study name."),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
) -> None:
    """Remove a study's directory tree (data and reports)."""
    try:
        s = Study.load(name)
    except StudyNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None
    if not yes:
        confirm = typer.confirm(
            f"Delete {s.root} and {s.reports_root}?", default=False
        )
        if not confirm:
            console.print("Aborted.")
            return
    s.delete()
    console.print(f"[green]Deleted[/green] study [cyan]{name}[/cyan].")
