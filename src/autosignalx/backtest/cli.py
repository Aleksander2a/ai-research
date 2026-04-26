"""CLI subcommand for the backtest layer.

``autosignalx backtest run`` executes the configured strategies over the
test window and writes artifacts under ``reports/backtest/runs/<run_id>/``.
The cockpit Backtest Arena panel reads from this directory.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from autosignalx.backtest import runner
from autosignalx.backtest.schemas import BacktestConfig

backtest_app = typer.Typer(
    name="backtest",
    help="Backtested simulation -- strategies driven by discovered structure.",
    no_args_is_help=True,
)
console = Console()


@backtest_app.command("run")
def run_cmd(
    strategies: str = typer.Option(
        "BuyAndHoldSPY;EqualWeightUniverse;TopKLong:k=3;LongShortKK:k=2;RegimeGated:k=3;FindingDriven",
        help="Semicolon-separated strategy specs (e.g. 'TopKLong:k=3;LongShortKK:k=2').",
    ),
    start: str = typer.Option("2021-01-01", help="Backtest start (YYYY-MM-DD)."),
    end: str = typer.Option("2025-12-31", help="Backtest end (YYYY-MM-DD)."),
    cost_bps: float = typer.Option(5.0, help="One-way transaction cost in bps."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Run the backtest ablation and write artifacts."""
    strat_list = [s.strip() for s in strategies.split(";") if s.strip()]
    cfg = BacktestConfig(
        strategies=strat_list,
        start_date=start,
        end_date=end,
        cost_bps=cost_bps,
        seed=seed,
    )
    result = runner.run_backtest(cfg)

    console.print(f"Wrote artifacts -> [cyan]{result.artifacts_dir}[/cyan]")
    table = Table(
        title=f"Backtest results (run_id={result.run_id})",
        show_lines=False,
        header_style="bold",
    )
    table.add_column("Strategy", style="cyan")
    table.add_column("CAGR", justify="right")
    table.add_column("Vol", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("Calmar", justify="right")
    table.add_column("Hit", justify="right")
    table.add_column("Turnover", justify="right")
    for s in result.strategies:
        table.add_row(
            s.name,
            f"{s.cagr:.2%}",
            f"{s.annual_vol:.2%}",
            f"{s.sharpe:+.2f}",
            f"{s.max_drawdown:.2%}",
            f"{s.calmar:+.2f}",
            f"{s.hit_rate:.1%}",
            f"{s.avg_turnover:.4f}",
        )
    console.print(table)


@backtest_app.command("status")
def status_cmd() -> None:
    """List available backtest runs."""
    runs = runner.list_runs()
    if not runs:
        console.print("reports/backtest/runs/ is empty (run `autosignalx backtest run`).")
        return
    table = Table(title="Backtest runs", show_lines=False, header_style="bold")
    table.add_column("Run ID", style="cyan")
    table.add_column("Path")
    for r in runs:
        table.add_row(r.name, str(r))
    console.print(table)
