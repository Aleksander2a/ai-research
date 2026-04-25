"""CLI subcommand for the regime layer."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from autosignalx.config import load_config, settings
from autosignalx.regime import labels

regime_app = typer.Typer(
    name="regime",
    help="Regime detection -- contrastive encoder + KMeans + HMM.",
    no_args_is_help=True,
)
console = Console()


@regime_app.command("fit")
def fit_cmd(
    config: str = typer.Option("default", help="Config name under configs/."),
    seed: int = typer.Option(42, help="Random seed for encoder + clusterer."),
) -> None:
    """Train the contrastive encoder, cluster (KMeans + HMM), and save labels."""
    cfg = load_config(config)["regime"]
    enc_cfg = cfg.get("encoder", {})

    console.print(
        f"Fitting regime detection: {cfg['n_regimes']} regimes, "
        f"window {enc_cfg.get('window_days', 60)}d, "
        f"embedding_dim {enc_cfg.get('embedding_dim', 16)}, "
        f"epochs {enc_cfg.get('epochs', 25)}..."
    )
    out = labels.fit_and_save(
        n_regimes=cfg["n_regimes"],
        embedding_dim=enc_cfg.get("embedding_dim", 16),
        window_days=enc_cfg.get("window_days", 60),
        epochs=enc_cfg.get("epochs", 25),
        batch_size=enc_cfg.get("batch_size", 64),
        seed=seed,
    )

    table = Table(title="Regime label distribution", header_style="bold")
    table.add_column("Method", style="cyan")
    table.add_column("N labels", justify="right")
    table.add_column("Counts per regime")
    for name in ("kmeans", "hmm"):
        df = out[name]
        counts = df["regime_id"].value_counts().sort_index().to_dict()
        table.add_row(name, f"{len(df):,}", str(counts))
    console.print(table)


@regime_app.command("status")
def status_cmd() -> None:
    """List cached regime label files."""
    out_dir = settings.reports_dir / "regimes"
    if not out_dir.exists() or not list(out_dir.glob("*.parquet")):
        console.print("reports/regimes/ is empty -- run 'autosignalx regime fit'")
        return
    table = Table(title="Regimes cache", header_style="bold")
    table.add_column("File", style="cyan")
    table.add_column("Size (KB)", justify="right")
    for p in sorted(out_dir.glob("*.parquet")):
        table.add_row(p.name, f"{p.stat().st_size / 1024:,.1f}")
    console.print(table)
