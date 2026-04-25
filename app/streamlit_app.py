"""AutoSignal-X — Streamlit research cockpit.

A reviewer-journey UI that surfaces, panel by panel, what the system has
done and what it has discovered. Panels light up as their iterations land."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from autosignalx import __version__
from autosignalx.config import settings

st.set_page_config(
    page_title="AutoSignal-X",
    layout="wide",
)


def render_overview() -> None:
    st.title("AutoSignal-X")
    st.caption(
        "A modular AI research instrument for discovering predictive structure "
        "in dynamic markets."
    )

    st.markdown(
        """
        **Thesis.** Can a multi-model AI pipeline outperform standalone forecasting
        systems by explicitly modeling latent regimes, structured signal relevance,
        and relational dependencies?

        This cockpit is a *research artifact*: every layer's output is inspectable,
        every experiment is logged into a persistent ledger, and the agent's
        reasoning is rendered as you watch it.
        """
    )

    st.subheader("Layers")
    layer_specs = [
        ("L1 Forecasting", "Iter 3", "Chronos-2"),
        ("L2 Representation", "Iter 4", "Contrastive + KMeans"),
        ("L3 Reasoning", "Iter 5", "TabPFN"),
        ("L4 Relational", "Iter 6", "Graph + Granger"),
        ("L5 Agentic", "Iter 7", "LangGraph + deepagents"),
    ]
    cols = st.columns(len(layer_specs))
    for col, (name, iter_label, impl) in zip(cols, layer_specs, strict=False):
        with col:
            st.metric(label=name, value=iter_label, delta=impl, delta_color="off")

    st.divider()
    st.subheader("System")
    st.code(
        f"version:        {__version__}\n"
        f"replay mode:    {settings.use_replay}\n"
        f"repo root:      {settings.repo_root}\n"
        f"data dir:       {settings.data_dir}\n"
        f"reports dir:    {settings.reports_dir}\n",
        language="yaml",
    )

    st.divider()
    st.caption(
        "Pipeline layers register their own panels here as their iterations land. "
        "See README.md for the iteration plan."
    )


def render_data() -> None:
    st.title("Data")
    st.caption("ETF OHLCV and macro signal cache backing every experiment.")

    try:
        from autosignalx.data import cache, loader
    except ImportError as e:
        st.error(f"Data layer not available: {e}")
        return

    info = cache.cache_status()
    if not info["ohlcv"].get("exists"):
        st.warning(
            "No data cached yet. Run `make data` (or `uv run autosignalx data fetch`) "
            "to populate the cache."
        )
        return

    cols = st.columns(2)
    with cols[0]:
        st.metric("OHLCV rows", f"{info['ohlcv']['rows']:,}")
        st.caption(
            f"{info['ohlcv'].get('earliest')} -> {info['ohlcv'].get('latest')}"
        )
    with cols[1]:
        if info["macro"].get("exists"):
            st.metric("Macro rows", f"{info['macro']['rows']:,}")
            st.caption(
                f"{info['macro'].get('earliest')} -> {info['macro'].get('latest')}"
            )
        else:
            st.metric("Macro rows", "0")
            st.caption("Macro cache empty.")

    st.divider()
    st.subheader("Adjusted close (normalized to 100 at start of cache)")
    try:
        prices = loader.load_close_wide().dropna(how="all")
        if not prices.empty:
            base = prices.iloc[0].replace(0, pd.NA)
            normalized = prices.divide(base) * 100
            st.line_chart(normalized, height=400)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not render prices: {e}")

    st.divider()
    st.subheader("Macro signals")
    try:
        macro_w = loader.load_macro_wide().dropna(how="all")
        if not macro_w.empty:
            st.line_chart(macro_w, height=300)
    except FileNotFoundError:
        st.caption("Macro cache empty.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not render macro: {e}")


def render_forecast_arena() -> None:
    st.title("Forecast Arena")
    st.caption(
        "Per-method, per-asset forecast comparison on walk-forward windows. "
        "New methods are appended to reports/ablations/ as their iterations land."
    )

    from pathlib import Path

    from autosignalx.config import settings
    from autosignalx.eval import harness

    ablations_dir = settings.reports_dir / "ablations"
    parquets: list[Path] = sorted(ablations_dir.glob("*.parquet")) if ablations_dir.exists() else []
    if not parquets:
        st.info(
            "No ablation results cached yet. Run `make baseline` (or "
            "`uv run autosignalx eval baseline`) to populate."
        )
        return

    forecasts = pd.concat(
        [pd.read_parquet(p) for p in parquets], ignore_index=True
    )

    st.caption(
        f"Loaded {len(forecasts):,} forecast rows from {len(parquets)} file(s): "
        + ", ".join(p.name for p in parquets)
    )

    overall = harness.add_skill_score(
        harness.summarize(forecasts, by=["method"]), baseline_method="naive"
    )
    st.subheader("Per-method (overall)")
    st.dataframe(
        overall.style.format(
            {
                "mae": "{:.3f}",
                "mape": "{:.3%}",
                "dir_acc": "{:.1%}",
                "skill_vs_naive": "{:+.3f}",
            }
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Per-method, per-asset")
    per_asset = harness.add_skill_score(harness.summarize(forecasts), baseline_method="naive")
    st.dataframe(
        per_asset.style.format(
            {
                "mae": "{:.3f}",
                "mape": "{:.3%}",
                "dir_acc": "{:.1%}",
                "skill_vs_naive": "{:+.3f}",
            }
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Forecast trajectory")
    asset_choice = st.selectbox("Asset", sorted(forecasts["asset"].unique()))
    asset_forecasts = forecasts[forecasts["asset"] == asset_choice].copy()
    if asset_forecasts.empty:
        st.warning("No forecasts for this asset.")
        return
    asset_forecasts = asset_forecasts.sort_values("timestamp")
    pivot = asset_forecasts.pivot_table(
        index="timestamp", columns="method", values="prediction", aggfunc="mean"
    )
    target_series = (
        asset_forecasts.drop_duplicates("timestamp").set_index("timestamp")["target"]
    )
    pivot["target"] = target_series
    st.line_chart(pivot, height=400)


PANELS = {
    "Overview": render_overview,
    "Data": render_data,
    "Forecast Arena": render_forecast_arena,
}


# Streamlit executes the script top-to-bottom on every interaction.
panel_name = st.sidebar.radio("Panel", list(PANELS.keys()))
st.sidebar.divider()
st.sidebar.caption(f"AutoSignal-X v{__version__}")
PANELS[panel_name]()
