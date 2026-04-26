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


def _ensure_ablations_dir(study_name: str = "") -> Path:
    """Resolve the ablations directory: default project tree or per-study."""
    if study_name:
        from autosignalx.study import Study

        s = Study.load(study_name)
        d = s.ablations_dir
    else:
        d = ABLATIONS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_eval_inputs(study_name: str, config: str, test_end_override: str):
    """Return (cache_root, splits, eval_cfg, test_end_value).

    For studies, all knobs come from study.yaml; for the default flow,
    they come from configs/<config>.yaml.
    """
    if study_name:
        from autosignalx.study import Study

        s = Study.load(study_name)
        splits_cfg = {
            "train_end": s.train_end,
            "val_end": s.val_end,
            "test_end": s.test_end,
        }
        eval_cfg = {
            "forecast_horizon_days": s.forecast_horizon_days,
            "rolling_step_days": s.rolling_step_days,
        }
        return s.cache_dir, splits_cfg, eval_cfg, test_end_override or s.test_end
    cfg = load_config(config)
    eval_cfg = cfg["eval"]
    splits_cfg = eval_cfg["splits"]
    return None, splits_cfg, eval_cfg, test_end_override or splits_cfg["test_end"]


def _fmt_skill(val: object) -> str:
    if val is None:
        return "  n/a"
    if isinstance(val, float) and math.isnan(val):
        return "  n/a"
    return f"{float(val):+.3f}"


def _fmt_value(val: object) -> str:
    """Plain absolute-value formatter (no +/-) for non-skill metrics like CRPS."""
    if val is None:
        return "  n/a"
    if isinstance(val, float) and math.isnan(val):
        return "  n/a"
    return f"{float(val):.3f}"


@eval_app.command("baseline")
def baseline_cmd(
    config: str = typer.Option("default", help="Config name under configs/."),
    study: str = typer.Option("", help="Study name (overrides --config)."),
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
    cache_root, splits_cfg, eval_cfg, test_end_value = _resolve_eval_inputs(
        study, config, test_end
    )

    method_map = {
        "naive": baselines.naive_forecast,
        "seasonal_naive": baselines.seasonal_naive_forecast,
        "arima": baselines.arima_forecast,
    }
    selected = {name: method_map[name] for name in methods.split(",") if name in method_map}
    if not selected:
        raise typer.BadParameter(f"No valid methods in {methods!r}")

    ohlcv = cache.read_ohlcv(cache_root=cache_root)
    windows = splits.walk_forward_windows(
        val_end=splits_cfg["val_end"],
        test_end=test_end_value,
        horizon_days=eval_cfg["forecast_horizon_days"],
        step_days=eval_cfg["rolling_step_days"],
    )
    n_assets = ohlcv["asset"].nunique()
    label = f"study={study}" if study else f"config={config}"
    console.print(
        f"Running {len(selected)} method(s) ({', '.join(selected)}) "
        f"across {len(windows)} walk-forward windows x {n_assets} assets ({label})..."
    )

    forecasts = harness.ablation(selected, ohlcv, windows)
    if forecasts.empty:
        console.print("[yellow]No forecasts produced.[/yellow]")
        raise typer.Exit(code=1)

    out_path = _ensure_ablations_dir(study) / output
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


@eval_app.command("chronos")
def chronos_cmd(
    config: str = typer.Option("default", help="Config name under configs/."),
    study: str = typer.Option("", help="Study name (overrides --config)."),
    test_end: str = typer.Option("", help="Override test_end (YYYY-MM-DD)."),
    methods: str = typer.Option(
        "univariate,multivariate",
        help="Comma-separated subset of: univariate, multivariate.",
    ),
    output: str = typer.Option(
        "chronos2.parquet",
        help="Filename under reports/ablations/.",
    ),
) -> None:
    """Run the Chronos-2 ablation (univariate and/or multivariate with covariates)."""
    cache_root, splits_cfg, eval_cfg, test_end_value = _resolve_eval_inputs(
        study, config, test_end
    )

    method_specs: dict[str, dict] = {}
    method_set = {m.strip() for m in methods.split(",") if m.strip()}
    if "univariate" in method_set:
        method_specs["chronos2_univariate"] = {"use_covariates": False}
    if "multivariate" in method_set:
        method_specs["chronos2_multivariate"] = {"use_covariates": True}
    if not method_specs:
        raise typer.BadParameter(f"No valid methods in {methods!r}")

    ohlcv = cache.read_ohlcv(cache_root=cache_root)
    macro = cache.read_macro(cache_root=cache_root)
    windows = splits.walk_forward_windows(
        val_end=splits_cfg["val_end"],
        test_end=test_end_value,
        horizon_days=eval_cfg["forecast_horizon_days"],
        step_days=eval_cfg["rolling_step_days"],
    )

    label = f"study={study}" if study else f"config={config}"
    console.print(
        f"Running {len(method_specs)} chronos variant(s) ({', '.join(method_specs)}) "
        f"across {len(windows)} windows x {ohlcv['asset'].nunique()} assets ({label}, "
        "model load takes ~40s first time)..."
    )

    from autosignalx.forecast import chronos2

    forecasts = chronos2.batched_ablation(
        method_specs, ohlcv, macro, windows, eval_cfg["forecast_horizon_days"]
    )
    if forecasts.empty:
        console.print("[yellow]No forecasts produced.[/yellow]")
        raise typer.Exit(code=1)

    out_path = _ensure_ablations_dir(study) / output
    forecasts.to_parquet(out_path, index=False)
    console.print(f"  wrote {len(forecasts):>7,} forecast rows -> {out_path}")

    overall = harness.summarize(forecasts, by=["method"])
    table = Table(
        title="Per-method overall (lower MAE / CRPS = better)",
        show_lines=False,
        header_style="bold",
    )
    table.add_column("Method", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("MAPE", justify="right")
    table.add_column("Dir-acc", justify="right")
    table.add_column("CRPS", justify="right")
    for _, row in overall.iterrows():
        table.add_row(
            str(row["method"]),
            f"{row['n']:,}",
            f"{row['mae']:.3f}",
            f"{row['mape']:.3%}",
            f"{row['dir_acc']:.1%}",
            _fmt_value(row.get("crps")),
        )
    console.print(table)


@eval_app.command("status")
def status_cmd(
    study: str = typer.Option("", help="Inspect a study's ablations dir."),
) -> None:
    """List ablation files."""
    if study:
        from autosignalx.study import Study

        target = Study.load(study).ablations_dir
        title = f"Ablations cache (study={study})"
    else:
        target = ABLATIONS_DIR
        title = "Ablations cache"
    if not target.exists():
        console.print(f"{target} does not exist (no ablations yet).")
        return
    files = sorted(target.glob("*.parquet"))
    if not files:
        console.print(f"{target} is empty (no ablations yet).")
        return
    table = Table(title=title, show_lines=False, header_style="bold")
    table.add_column("File", style="cyan")
    table.add_column("Size (KB)", justify="right")
    for p in files:
        size_kb = p.stat().st_size / 1024
        table.add_row(p.name, f"{size_kb:,.1f}")
    console.print(table)
