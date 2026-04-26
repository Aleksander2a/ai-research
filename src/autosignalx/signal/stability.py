"""Walk-forward signal-importance stability.

The base ``rank_features_per_regime`` fits a HistGradientBoosting
classifier once per regime on the union of in-regime samples. That
gives a per-regime importance ranking but says nothing about how
*stable* those rankings are over time -- a feature ranked #1 averaged
over the whole period may oscillate wildly across sub-periods, in
which case its #1 status is an averaging artefact.

This module slides a fixed-length window across the timeline,
re-fitting the per-regime ranker on each window, and persists the
per-(window, regime, feature) importance + rank. From those we derive
**stability metrics** per feature × regime:

* ``mean_rank`` -- average rank across windows
* ``rank_std`` -- standard deviation of rank
* ``top5_share`` -- fraction of windows where the feature was in the top-5
* ``stability`` -- 1 - (rank_std / max_rank), clipped to [0, 1]

A feature with high mean importance *and* high stability is
research-grade; a feature with high mean importance but low stability
is a candidate for deprioritisation.
"""

from __future__ import annotations

import pandas as pd

from autosignalx.config import settings
from autosignalx.signal.ranking import rank_features_per_regime


def walk_forward_rank(
    features_df: pd.DataFrame,
    regime_labels: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_direction",
    n_windows: int = 6,
    min_window_size: int = 250,
    n_samples_per_regime: int = 1500,
    n_repeats: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Slide ``n_windows`` walk-forward windows across the timeline and run
    the per-regime ranker inside each. Returns a long-format frame keyed by
    ``(window_idx, window_start, regime_id, feature, importance, rank)``.

    Walk-forward integrity: each window's ranker only sees samples whose
    timestamp falls inside that window. No future leakage."""
    df = features_df.merge(regime_labels[["timestamp", "regime_id"]], on="timestamp", how="inner")
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_total = len(df)
    if n_total < min_window_size * 2:
        return pd.DataFrame()

    window_size = max(min_window_size, n_total // n_windows)
    step = max(1, (n_total - window_size) // max(n_windows - 1, 1))

    rows: list[pd.DataFrame] = []
    for w in range(n_windows):
        start_idx = w * step
        end_idx = start_idx + window_size
        if end_idx > n_total:
            break
        sub = df.iloc[start_idx:end_idx]
        ranking = rank_features_per_regime(
            features_df=sub.drop(columns=["regime_id"]),
            regime_labels=sub[["timestamp", "regime_id"]],
            feature_cols=feature_cols,
            target_col=target_col,
            n_samples_per_regime=n_samples_per_regime,
            n_repeats=n_repeats,
            seed=seed,
        )
        if ranking.empty:
            continue
        ranking["window_idx"] = w
        ranking["window_start"] = sub["timestamp"].iloc[0]
        ranking["window_end"] = sub["timestamp"].iloc[-1]
        rows.append(ranking)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarise_stability(walk_forward_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Aggregate the walk-forward ranking into per-(regime, feature) stability."""
    if walk_forward_df.empty:
        return pd.DataFrame()
    g = walk_forward_df.groupby(["regime_id", "feature"], observed=True)
    summary = g.agg(
        mean_importance=("importance", "mean"),
        std_importance=("importance", "std"),
        mean_rank=("rank", "mean"),
        rank_std=("rank", "std"),
        n_windows=("window_idx", "nunique"),
    ).reset_index()
    # Top-K share: in how many windows was this feature ranked <= top_k?
    in_top = walk_forward_df[walk_forward_df["rank"] <= top_k]
    top_share = (
        in_top.groupby(["regime_id", "feature"], observed=True).size()
        / walk_forward_df.groupby(["regime_id", "feature"], observed=True).size()
    ).rename(f"top{top_k}_share").reset_index()
    summary = summary.merge(top_share, on=["regime_id", "feature"], how="left")
    summary[f"top{top_k}_share"] = summary[f"top{top_k}_share"].fillna(0.0)

    max_rank = walk_forward_df["rank"].max() or 1
    summary["stability"] = 1.0 - (summary["rank_std"].fillna(0.0) / float(max_rank))
    summary["stability"] = summary["stability"].clip(0.0, 1.0)

    return summary.sort_values(
        ["regime_id", "mean_importance"], ascending=[True, False]
    ).reset_index(drop=True)


def build_and_save(
    features_df: pd.DataFrame,
    regime_labels: pd.DataFrame,
    feature_cols: list[str],
    n_windows: int = 6,
) -> dict[str, pd.DataFrame]:
    """Run walk-forward ranking + stability summary, persist both."""
    out_dir = settings.reports_dir / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)

    wf = walk_forward_rank(
        features_df=features_df,
        regime_labels=regime_labels,
        feature_cols=feature_cols,
        n_windows=n_windows,
    )
    if wf.empty:
        return {"walk_forward": wf, "stability": pd.DataFrame()}

    wf.to_parquet(out_dir / "walk_forward_ranking.parquet", index=False)
    summary = summarise_stability(wf)
    summary.to_parquet(out_dir / "signal_stability.parquet", index=False)
    return {"walk_forward": wf, "stability": summary}


def load_walk_forward() -> pd.DataFrame:
    p = settings.reports_dir / "signals" / "walk_forward_ranking.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_stability() -> pd.DataFrame:
    p = settings.reports_dir / "signals" / "signal_stability.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()
