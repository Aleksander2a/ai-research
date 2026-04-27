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


@eval_app.command("returns")
def returns_cmd(
    config: str = typer.Option("default", help="Config name under configs/."),
    study: str = typer.Option("", help="Study name (overrides --config)."),
    test_end: str = typer.Option("", help="Override test_end (YYYY-MM-DD)."),
    methods: str = typer.Option(
        "zero_return,mean_return,momentum",
        help="Comma-separated subset of: zero_return, mean_return, momentum.",
    ),
    target: str = typer.Option(
        "log_return",
        help="Target type: log_return | excess_return | rank.",
    ),
    output: str = typer.Option(
        "returns_baseline.parquet",
        help="Filename under reports/ablations/.",
    ),
) -> None:
    """Phase 7: run a returns-target ablation.

    Produces price-level forecasts via simple returns-style baselines,
    then converts to the requested return-type before persisting. Cockpit
    panels can filter by ``target_type`` to stratify rigorous returns
    findings from price-level findings."""
    from autosignalx.eval import targets as targets_mod
    from autosignalx.forecast import returns_baselines

    cache_root, splits_cfg, eval_cfg, test_end_value = _resolve_eval_inputs(
        study, config, test_end
    )
    method_map = {
        "zero_return": returns_baselines.zero_return_forecast,
        "mean_return": returns_baselines.mean_return_forecast,
        "momentum": returns_baselines.momentum_forecast,
    }
    selected = {n: method_map[n] for n in methods.split(",") if n in method_map}
    if not selected:
        raise typer.BadParameter(f"No valid methods in {methods!r}")

    if target not in targets_mod.VALID_TARGET_TYPES:
        raise typer.BadParameter(
            f"target must be one of {targets_mod.VALID_TARGET_TYPES}"
        )

    ohlcv = cache.read_ohlcv(cache_root=cache_root)
    windows = splits.walk_forward_windows(
        val_end=splits_cfg["val_end"],
        test_end=test_end_value,
        horizon_days=eval_cfg["forecast_horizon_days"],
        step_days=eval_cfg["rolling_step_days"],
    )
    label = f"study={study}" if study else f"config={config}"
    console.print(
        f"Phase-7 returns ablation: {len(selected)} method(s), target={target} ({label}); "
        f"{len(windows)} windows x {ohlcv['asset'].nunique()} assets..."
    )
    price_forecasts = harness.ablation(selected, ohlcv, windows)
    if price_forecasts.empty:
        console.print("[yellow]No forecasts produced.[/yellow]")
        raise typer.Exit(code=1)

    converted = targets_mod.convert_target(price_forecasts, target_type=target, ohlcv=ohlcv)
    if converted.empty:
        console.print(
            "[yellow]Target conversion produced an empty frame "
            "(insufficient data for vol/rank/excess?).[/yellow]"
        )
        raise typer.Exit(code=1)

    out_path = _ensure_ablations_dir(study) / output
    converted.to_parquet(out_path, index=False)
    console.print(f"  wrote {len(converted):>7,} {target} forecast rows -> {out_path}")

    from autosignalx.eval.metrics_returns import summarise_returns

    summary = summarise_returns(converted, by=["method"])
    table = Table(
        title=f"Per-method overall ({target} target)",
        show_lines=False,
        header_style="bold",
    )
    table.add_column("Method", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("Hit-rate", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("IC (Pearson)", justify="right")
    table.add_column("IC (Spearman)", justify="right")
    for _, row in summary.iterrows():
        table.add_row(
            str(row["method"]),
            f"{int(row['n']):,}",
            f"{row['mae']:.5f}",
            _fmt_value(row.get("hit_rate")),
            _fmt_value(row.get("forecast_sharpe")),
            _fmt_value(row.get("ic_pearson")),
            _fmt_value(row.get("ic_spearman")),
        )
    console.print(table)


@eval_app.command("pbo")
def pbo_cmd(
    methods: str = typer.Option(
        "",
        help="Comma-separated method names to include in the PBO matrix; "
        "default = every non-naive method in the cache.",
    ),
    baseline: str = typer.Option("naive", help="Baseline method."),
    s: int = typer.Option(16, help="Sub-period count for combinatorial split."),
) -> None:
    """Phase 8: Probability of Backtest Overfitting across cached methods."""
    from autosignalx.eval.pbo import pbo_from_forecasts

    abl_dir = settings.reports_dir / "ablations"
    if not abl_dir.exists():
        console.print("[yellow]No ablations cached.[/yellow]")
        raise typer.Exit(code=1)
    import pandas as pd

    frames = []
    for fp in abl_dir.glob("*.parquet"):
        try:
            frames.append(pd.read_parquet(fp))
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        console.print("[yellow]No ablations cached.[/yellow]")
        raise typer.Exit(code=1)
    forecasts = pd.concat(frames, ignore_index=True)
    if methods:
        method_list = [m.strip() for m in methods.split(",") if m.strip()]
    else:
        method_list = sorted(forecasts["method"].unique())
    res = pbo_from_forecasts(forecasts, methods=method_list, baseline=baseline, s=s)
    console.print(
        f"PBO over {res.n_strategies} strategies x {res.n_periods} periods, "
        f"{res.n_combinations} combinatorial splits: [bold]{res.pbo:.3f}[/bold]"
    )
    if res.pbo > 0.5:
        console.print(
            "[red]>0.5: IS-best ranking has worse-than-random OOS predictive power; "
            "the search overfit the cache.[/red]"
        )
    elif res.pbo < 0.1:
        console.print("[green]<0.1: rankings transfer to OOS; methodology is robust.[/green]")
    out_path = settings.reports_dir / "agent" / "pbo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        __import__("json").dumps(
            {
                "pbo": res.pbo,
                "n_strategies": res.n_strategies,
                "n_periods": res.n_periods,
                "n_combinations": res.n_combinations,
                "methods": method_list,
                "baseline": baseline,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"  wrote {out_path}")


@eval_app.command("synthetic")
def synthetic_cmd(
    n_trials: int = typer.Option(5, help="Number of independent synthetic universes."),
    seed: int = typer.Option(0, help="Base seed; trial t uses seed + 1000*t."),
    planted_cells: int = typer.Option(3, help="Number of planted (asset, regime, method) edges."),
    distractor_cells: int = typer.Option(9, help="Number of distractor (no-signal) methods."),
    planted_skill: float = typer.Option(0.20, help="Planted skill-vs-naive lift on truth cells."),
    n_origins: int = typer.Option(80, help="Forecast origins per universe."),
    n_assets: int = typer.Option(4, help="Synthetic universe size."),
    n_regimes: int = typer.Option(2, help="Synthetic regime count."),
) -> None:
    """Synthetic-known-answer benchmark for the methodology stack.

    Plants a known-true (asset, regime, method) edge in N synthetic
    universes; surrounds it with distractor cells; runs every gate
    (DM -> +FDR -> +adversarial -> +Romano-Wolf -> +Bayesian -> strict);
    reports per-gate recall and FDR averaged across trials. Output is
    written to ``reports/agent/synthetic_benchmark.json`` and rendered
    by the cockpit's Synthetic Benchmark panel + the static snapshot."""
    from autosignalx.eval.synthetic_benchmark import run_benchmark

    console.print(
        f"Running synthetic benchmark: n_trials={n_trials}, "
        f"planted={planted_cells}, distractors={distractor_cells}, "
        f"planted_skill={planted_skill:.2f}, n_origins={n_origins}, "
        f"n_assets={n_assets}, n_regimes={n_regimes}"
    )
    summary = run_benchmark(
        n_trials=n_trials, seed=seed,
        planted_cells=planted_cells, distractor_cells=distractor_cells,
        planted_skill=planted_skill, n_origins=n_origins,
        n_assets=n_assets, n_regimes=n_regimes,
    )
    table = Table(title="Synthetic benchmark -- per-gate recall / FDR", header_style="bold")
    table.add_column("Gate", style="cyan")
    table.add_column("Mean recall", justify="right")
    table.add_column("Mean FDR", justify="right")
    table.add_column("Mean n_promoted", justify="right")
    for row in summary["ablations"]:
        table.add_row(
            str(row["gate"]),
            f"{row['mean_recall']:.2f}",
            f"{row['mean_fdr']:.2f}",
            f"{row['mean_n_promoted']:.1f}",
        )
    console.print(table)


@eval_app.command("ablate-capability")
def ablate_capability_cmd(
    methods_per_layer: str = typer.Option(
        "naive,arima,chronos2_univariate,chronos2_multivariate",
        help="Methods that represent each layer's marginal contribution.",
    ),
) -> None:
    """Phase 16 -- smallest-capability-preserving ablation.

    For each candidate layer (L1 forecasting / L2 regime / L3 signal /
    L4 graph / L5 agent), drop that layer's contribution to the gating
    pipeline and re-grade promoted findings. The resulting marginal-
    skill column tells the user which layers are load-bearing and
    which can be compressed/distilled without losing signal.
    Persists ``reports/agent/capability_ablation.json``."""
    from autosignalx.eval.capability_ablation import run_capability_ablation

    summary = run_capability_ablation(methods=[
        m.strip() for m in methods_per_layer.split(",") if m.strip()
    ])
    if not summary.get("rows"):
        console.print("[yellow]No ablation rows produced (need cached forecasts + findings).[/yellow]")
        return
    table = Table(title="Capability-preserving ablation", header_style="bold")
    table.add_column("Variant", style="cyan")
    table.add_column("Layers", overflow="fold")
    table.add_column("# findings", justify="right")
    table.add_column("Mean MAE", justify="right")
    table.add_column("Marginal skill", justify="right")
    table.add_column("Cost proxy", justify="right")
    for row in summary["rows"]:
        table.add_row(
            str(row.get("variant")), str(row.get("layers", "")),
            str(row.get("n_findings", "")),
            f"{row.get('mean_mae', float('nan')):.4f}",
            f"{row.get('marginal_skill', float('nan')):+.4f}",
            f"{row.get('cost_proxy', 0):.0f}",
        )
    console.print(table)


@eval_app.command("vault-init")
def vault_init_cmd(
    start: str = typer.Argument(..., help="Vault start (YYYY-MM-DD)."),
    end: str = typer.Argument(..., help="Vault end (YYYY-MM-DD)."),
    description: str = typer.Option("", help="Why this vault exists."),
) -> None:
    """Phase 8: declare a never-touched holdout vault."""
    from autosignalx.eval.holdout_vault import initialize_vault

    rec = initialize_vault(start=start, end=end, description=description)
    console.print(
        f"Vault locked: {rec['start']} -> {rec['end']} "
        f"(hash {rec['lock_hash']}, locked at {rec['locked_at']})"
    )


@eval_app.command("vault-open")
def vault_open_cmd(
    methods: str = typer.Option("", help="Comma-separated methods to evaluate."),
    baseline: str = typer.Option("naive", help="Baseline method."),
) -> None:
    """Phase 8: one-time vault open and final evaluation."""
    from autosignalx.eval.holdout_vault import open_vault

    abl_dir = settings.reports_dir / "ablations"
    import pandas as pd

    frames = []
    for fp in abl_dir.glob("*.parquet"):
        try:
            frames.append(pd.read_parquet(fp))
        except Exception:  # noqa: BLE001
            continue
    forecasts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    method_list = (
        [m.strip() for m in methods.split(",") if m.strip()]
        if methods
        else sorted(forecasts["method"].unique()) if not forecasts.empty else []
    )
    res = open_vault(forecasts, methods=method_list, baseline=baseline)
    if res.get("already_opened"):
        console.print("[yellow]Vault already opened.[/yellow]")
    elif res.get("empty"):
        console.print("[yellow]No rows in vault window.[/yellow]")
    else:
        console.print(f"Vault opened. n_rows={res['n_rows']}")
        for m, mae in res["per_method_mae"].items():
            skill = res["skill_vs_baseline"].get(m)
            sk_str = f"{skill:+.3f}" if skill is not None else "n/a"
            console.print(f"  {m}: MAE={mae:.4f} skill_vs_{baseline}={sk_str}")


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
