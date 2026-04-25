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


PANELS = {
    "Overview": render_overview,
    "Data": render_data,
}


# Streamlit executes the script top-to-bottom on every interaction.
panel_name = st.sidebar.radio("Panel", list(PANELS.keys()))
st.sidebar.divider()
st.sidebar.caption(f"AutoSignal-X v{__version__}")
PANELS[panel_name]()
