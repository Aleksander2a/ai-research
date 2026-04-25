"""CLI subcommand for the eval layer.

``autosignalx eval baseline`` runs the naive / seasonal-naive / ARIMA
ablation over walk-forward windows, writes the forecasts to
``reports/ablations/<filename>.parquet``, and prints a per-method summary.
The cockpit Forecast Arena panel reads from this directory."""

from __future__ import annotations

import math
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from autosignalx.config import load_config, settings
from autosignalx.data import cache, splits
from autosignalx.eval import harness
from autosignalx.forecast import baselines

eval_app = typer.Typer(
    name="eval",
    help="Evaluation -- walk-forward ablations and metrics.",
    no_args_is_help=True,
)
console = Console()

ABLATIONS_DIR = settings.reports_dir / "ablations"


def _ensure_ablations_dir() -> Path:
    ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return ABLATIONS_DIR


def _fmt_skill(val: object) -> str:
    if val is None:
        return "  n/a"
    if isinstance(val, float) and math.isnan(val):
        return "  n/a"
    return f"{float(val):+.3f}"


@eval_app.command("baseline")
def baseline_cmd(
    config: str = typer.Option("default", help="Config name under configs/."),
    test_end: str = typer.Option("", help="Override test_end (YYYY-MM-DD)."),
    methods: str = typer.Option(
        "naive,seasonal_naive,arima",
        help="Comma-separated subset of: naive, seasonal_naive, arima.",
    ),
    output: str = typer.Option(
        "baseline.parquet",
        help="Filename under reports/ablations/.",
    ),
) -> None:
    """Run the baseline ablation across walk-forward windows."""
    cfg = load_config(config)
    eval_cfg = cfg["eval"]
    splits_cfg = eval_cfg["splits"]
    test_end_value = test_end or splits_cfg["test_end"]

    method_map = {
        "naive": baselines.naive_forecast,
        "seasonal_naive": baselines.seasonal_naive_forecast,
        "arima": baselines.arima_forecast,
    }
    selected = {name: method_map[name] for name in methods.split(",") if name in method_map}
    if not selected:
        raise typer.BadParameter(f"No valid methods in {methods!r}")

    ohlcv = cache.read_ohlcv()
    windows = splits.walk_forward_windows(
        val_end=splits_cfg["val_end"],
        test_end=test_end_value,
        horizon_days=eval_cfg["forecast_horizon_days"],
        step_days=eval_cfg["rolling_step_days"],
    )
    n_assets = ohlcv["asset"].nunique()
    console.print(
        f"Running {len(selected)} method(s) "
        f"({', '.join(selected)}) "
        f"across {len(windows)} walk-forward windows x {n_assets} assets..."
    )

    forecasts = harness.ablation(selected, ohlcv, windows)
    if forecasts.empty:
        console.print("[yellow]No forecasts produced.[/yellow]")
        raise typer.Exit(code=1)

    out_path = _ensure_ablations_dir() / output
    forecasts.to_parquet(out_path, index=False)
    console.print(f"  wrote {len(forecasts):>7,} forecast rows -> {out_path}")

    overall = harness.add_skill_score(
        harness.summarize(forecasts, by=["method"]),
        baseline_method="naive",
    )

    table = Table(
        title="Per-method overall (skill > 0 = better than naive)",
        show_lines=False,
        header_style="bold",
    )
    table.add_column("Method", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("MAPE", justify="right")
    table.add_column("Dir-acc", justify="right")
    table.add_column("Skill vs naive", justify="right")
    for _, row in overall.iterrows():
        table.add_row(
            str(row["method"]),
            f"{row['n']:,}",
            f"{row['mae']:.3f}",
            f"{row['mape']:.3%}",
            f"{row['dir_acc']:.1%}",
            _fmt_skill(row.get("skill_vs_naive")),
        )
    console.print(table)


@eval_app.command("status")
def status_cmd() -> None:
    """List ablation files currently under reports/ablations/."""
    if not ABLATIONS_DIR.exists():
        console.print("reports/ablations/ does not exist (no ablations yet).")
        return
    files = sorted(ABLATIONS_DIR.glob("*.parquet"))
    if not files:
        console.print("reports/ablations/ is empty (no ablations yet).")
        return
    table = Table(title="Ablations cache", show_lines=False, header_style="bold")
    table.add_column("File", style="cyan")
    table.add_column("Size (KB)", justify="right")
    for p in files:
        size_kb = p.stat().st_size / 1024
        table.add_row(p.name, f"{size_kb:,.1f}")
    console.print(table)
