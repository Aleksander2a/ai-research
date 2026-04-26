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
from autosignalx.backtest.strategy_selection import (
    default_cli_strategies,
    default_study_strategies,
    ensure_strategy_prerequisites,
)

backtest_app = typer.Typer(
    name="backtest",
    help="Backtested simulation -- strategies driven by discovered structure.",
    no_args_is_help=True,
)
console = Console()


@backtest_app.command("run")
def run_cmd(
    study: str = typer.Option(
        "", help="Study name; reads prices/forecasts/regimes from study tree."
    ),
    strategies: str = typer.Option(
        "",
        help=(
            "Semicolon-separated strategy specs (e.g. "
            "'TopKLong:k=3;LongShortKK:k=2'). When omitted for study runs, "
            "AutoSignal-X picks a compatible bundle from the study artifacts."
        ),
    ),
    start: str = typer.Option("", help="Backtest start (default: study.effective_backtest_start or 2021-01-01)."),
    end: str = typer.Option("", help="Backtest end (default: study.test_end or 2025-12-31)."),
    cost_bps: float = typer.Option(-1.0, help="One-way transaction cost in bps; -1 = use study/default."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Run the backtest ablation and write artifacts."""
    active_study = None
    if study:
        from autosignalx.study import Study

        active_study = Study.load(study)
        eff_start = start or active_study.effective_backtest_start
        eff_end = end or active_study.test_end
        eff_cost = cost_bps if cost_bps >= 0 else active_study.cost_bps
        eff_universe = list(active_study.assets)
    else:
        eff_start = start or "2021-01-01"
        eff_end = end or "2025-12-31"
        eff_cost = cost_bps if cost_bps >= 0 else 5.0
        eff_universe = None

    if strategies.strip():
        strat_list = [s.strip() for s in strategies.split(";") if s.strip()]
        if not strat_list:
            raise typer.BadParameter("No valid strategies were provided.")
        ensure_strategy_prerequisites(
            strat_list, study=active_study, universe=eff_universe
        )
    elif active_study is not None:
        strat_list = default_study_strategies(active_study)
    else:
        strat_list = default_cli_strategies()

    console.print(f"Strategies: {', '.join(strat_list)}")
    cfg = BacktestConfig(
        strategies=strat_list,
        start_date=eff_start,
        end_date=eff_end,
        cost_bps=eff_cost,
        seed=seed,
        universe=eff_universe,
    )
    result = runner.run_backtest(cfg, study_name=study)

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

    # Significance table
    import json
    from pathlib import Path

    metrics_json = json.loads(Path(result.metrics_path).read_text())
    sig = metrics_json.get("__significance__", {})
    if sig:
        sig_table = Table(
            title=f"Sharpe-diff vs {cfg.benchmark_strategy} "
                  f"(paired block-bootstrap, n={cfg.bootstrap_n}, B={cfg.bootstrap_block_size})",
            show_lines=False,
            header_style="bold",
        )
        sig_table.add_column("Strategy", style="cyan")
        sig_table.add_column("Diff", justify="right")
        sig_table.add_column("95% CI", justify="right")
        sig_table.add_column("p-value", justify="right")
        sig_table.add_column("Significant", justify="center")
        for name, s in sig.items():
            sig_table.add_row(
                name,
                f"{s['sharpe_diff']:+.3f}",
                f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]",
                f"{s['p_value']:.3f}",
                "yes" if s["significant"] else "no",
            )
        console.print(sig_table)


@backtest_app.command("status")
def status_cmd(
    study: str = typer.Option("", help="Inspect a study's backtest runs."),
) -> None:
    """List available backtest runs."""
    runs = runner.list_runs(study_name=study)
    if not runs:
        scope = f"study={study}" if study else "default"
        console.print(f"No backtest runs ({scope}). Run `autosignalx backtest run`.")
        return
    title = f"Backtest runs (study={study})" if study else "Backtest runs"
    table = Table(title=title, show_lines=False, header_style="bold")
    table.add_column("Run ID", style="cyan")
    table.add_column("Path")
    for r in runs:
        table.add_row(r.name, str(r))
    console.print(table)
