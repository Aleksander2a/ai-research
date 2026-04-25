"""CLI subcommand for the signal layer.

``autosignalx signal rank`` builds features for every asset, joins regime
labels, fits TabPFN per regime, and ranks features by permutation
importance. Output: ``reports/signals/tabpfn_ranking.parquet``."""

from __future__ import annotations

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from autosignalx.config import load_config, settings
from autosignalx.data import cache, loader
from autosignalx.regime.labels import load_regime_labels
from autosignalx.signal import features, ranking

signal_app = typer.Typer(
    name="signal",
    help="Signal discovery -- TabPFN per-regime feature ranking.",
    no_args_is_help=True,
)
console = Console()

SIGNALS_DIR = settings.reports_dir / "signals"


@signal_app.command("rank")
def rank_cmd(
    config: str = typer.Option("default", help="Config name under configs/."),
    regime_method: str = typer.Option("kmeans_contrastive", help="Regime detector."),
    n_samples: int = typer.Option(2000, help="Max samples per regime."),
    n_repeats: int = typer.Option(2, help="Permutation repeats per feature."),
    seed: int = typer.Option(42, help="Random seed."),
    output: str = typer.Option(
        "signal_ranking.parquet", help="Filename under reports/signals/."
    ),
) -> None:
    """Rank features per regime via TabPFN + permutation importance."""
    load_config(config)  # validate config exists
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    console.print("Loading data and regime labels...")
    ohlcv = cache.read_ohlcv()
    macro_wide = loader.load_macro_wide()
    regime_labels = load_regime_labels(regime_method)

    console.print(
        f"Building features across {ohlcv['asset'].nunique()} assets x "
        f"{len(ohlcv):,} OHLCV rows..."
    )
    parts = []
    for asset, asset_ohlcv in ohlcv.groupby("asset", observed=True):
        feats = features.build_features_target(asset_ohlcv, macro_wide)
        feats["asset"] = asset
        parts.append(feats)
    feat_df = pd.concat(parts, ignore_index=True)

    feature_cols = features.feature_columns(feat_df)
    console.print(
        f"  {len(feat_df):,} feature rows; {len(feature_cols)} candidate features."
    )

    console.print(
        f"Ranking via HistGradientBoostingClassifier "
        f"({n_samples} samples/regime, {n_repeats} permutation repeats)..."
    )
    rankings = ranking.rank_features_per_regime(
        feat_df,
        regime_labels,
        feature_cols,
        n_samples_per_regime=n_samples,
        n_repeats=n_repeats,
        seed=seed,
    )

    if rankings.empty:
        console.print("[yellow]No rankings produced (no regime had enough samples).[/yellow]")
        raise typer.Exit(code=1)

    out_path = SIGNALS_DIR / output
    rankings.to_parquet(out_path, index=False)
    console.print(f"  wrote {len(rankings):,} ranking rows -> {out_path}")

    for regime_id, group in rankings.groupby("regime_id", observed=True):
        table = Table(
            title=f"Regime {regime_id} -- top 5 features",
            show_lines=False,
            header_style="bold",
        )
        table.add_column("Rank", justify="right")
        table.add_column("Feature", style="cyan")
        table.add_column("Importance", justify="right")
        table.add_column("± Std", justify="right")
        for _, row in group.head(5).iterrows():
            table.add_row(
                str(row["rank"]),
                str(row["feature"]),
                f"{row['importance']:+.3f}",
                f"{row['importance_std']:.3f}",
            )
        console.print(table)


@signal_app.command("status")
def status_cmd() -> None:
    """List cached signal-ranking files."""
    if not SIGNALS_DIR.exists() or not list(SIGNALS_DIR.glob("*.parquet")):
        console.print("reports/signals/ is empty -- run 'autosignalx signal rank'")
        return
    table = Table(title="Signals cache", header_style="bold")
    table.add_column("File", style="cyan")
    table.add_column("Size (KB)", justify="right")
    for p in sorted(SIGNALS_DIR.glob("*.parquet")):
        table.add_row(p.name, f"{p.stat().st_size / 1024:,.1f}")
    console.print(table)
