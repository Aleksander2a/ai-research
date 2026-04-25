"""CLI subcommand for the graph layer."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from autosignalx.config import settings
from autosignalx.graph import build

graph_app = typer.Typer(
    name="graph",
    help="Cross-asset relational structure -- GLASSO + Granger + centrality.",
    no_args_is_help=True,
)
console = Console()


@graph_app.command("build")
def build_cmd(
    p_threshold: float = typer.Option(0.05, help="Granger p-value threshold for edges."),
    max_lag: int = typer.Option(5, help="Max lag for Granger causality test."),
    pcorr_threshold: float = typer.Option(1e-3, help="Min |partial corr| to emit an edge."),
) -> None:
    """Compute the partial-correlation + Granger graph + centrality and save."""
    console.print(
        f"Building cross-asset graph: GLASSO + Granger (p<{p_threshold}, "
        f"max_lag={max_lag}) + centrality..."
    )
    out = build.build_and_save(
        p_threshold=p_threshold, max_lag=max_lag, pcorr_threshold=pcorr_threshold
    )
    n_pcorr = len(out["partial_corr"])
    n_granger = len(out["granger"])
    console.print(
        f"  partial-corr edges: {n_pcorr}; Granger edges (p<{p_threshold}): {n_granger}"
    )
    console.print(
        f"  wrote {len(out['edges'])} total edges -> "
        f"{settings.reports_dir / 'graph' / 'edges.parquet'}"
    )

    cent = out["centrality"]
    table = Table(title="Centrality (sorted by eigenvector)", header_style="bold")
    table.add_column("Node", style="cyan")
    table.add_column("Degree", justify="right")
    table.add_column("Eigenvector", justify="right")
    table.add_column("Betweenness", justify="right")
    for _, row in cent.iterrows():
        table.add_row(
            str(row["node"]),
            f"{row['degree_centrality']:.3f}",
            f"{row['eigenvector_centrality']:.3f}",
            f"{row['betweenness_centrality']:.3f}",
        )
    console.print(table)


@graph_app.command("status")
def status_cmd() -> None:
    """List cached graph artifacts."""
    out_dir = settings.reports_dir / "graph"
    if not out_dir.exists() or not list(out_dir.glob("*.parquet")):
        console.print("reports/graph/ is empty -- run 'autosignalx graph build'")
        return
    table = Table(title="Graph cache", header_style="bold")
    table.add_column("File", style="cyan")
    table.add_column("Size (KB)", justify="right")
    for p in sorted(out_dir.glob("*.parquet")):
        table.add_row(p.name, f"{p.stat().st_size / 1024:,.1f}")
    console.print(table)
