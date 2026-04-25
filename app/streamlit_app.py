"""AutoSignal-X — Streamlit research cockpit.

A reviewer-journey UI that surfaces, panel by panel, what the system has
done and what it has discovered. Panels light up as their iterations land."""

from __future__ import annotations

import json

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

    st.success(
        "**Reviewer journey.** Walk the panels left-to-right in the sidebar: "
        "**Data** -> **Forecast Arena** -> **Regime Explorer** -> "
        "**Signal Discovery Lab** -> **Cross-Asset Graph** -> **Agent Console** -> "
        "**Ask the Memory**. The story builds layer by layer."
    )

    st.subheader("Layers (all 5 live)")
    layer_specs = [
        ("L1 Forecasting", "Chronos-2 + baselines", "ok"),
        ("L2 Representation", "Contrastive + KMeans", "ok"),
        ("L3 Reasoning", "Per-regime ranking", "ok"),
        ("L4 Relational", "GLASSO + Granger", "ok"),
        ("L5 Agentic", "LangGraph + DeepInfra", "ok"),
    ]
    cols = st.columns(len(layer_specs))
    for col, (name, impl, status) in zip(cols, layer_specs, strict=False):
        with col:
            st.metric(label=name, value=status.upper(), delta=impl, delta_color="off")

    st.divider()
    st.subheader("Headline findings")
    st.markdown(
        """
        - **Iter 3 (negative result, calibrated)**: Chronos-2 underperforms naive on
          daily ETFs by 5-6% MAE; 80% intervals well-calibrated (CRPS ≈ 2.9). Macro
          covariates do not help unconditionally.
        - **Iter 5 (signals)**: Macros dominate every regime's top-5 features for
          direction prediction, but the **dominant macro depends on the regime**
          (TNX in Regime 0, DXY in Regimes 1+3, CL=F in Regime 2).
        - **Iter 6 (graph)**: SPY is the structural hub (eigenvector 0.532); GLD is
          statistically isolated; TLT is the bridge (highest betweenness 0.429).
        - **Iter 7 (agent)**: by Round 4 the live agent composes findings from
          every prior layer into a single mechanistic, falsifiable hypothesis --
          the conditional-improvement search opened by Iter 3's negative result.
        """
    )

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
        "See REPORT.md in the repo for the full layer-by-layer findings narrative."
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
                "crps": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    # Regime-stratified view (optional; only if regime labels exist)
    try:
        from autosignalx.regime.labels import add_regime_to_forecasts

        forecasts_r = add_regime_to_forecasts(forecasts, method="kmeans_contrastive")
        if forecasts_r["regime_id"].notna().any():
            st.subheader("Per-method, per-regime (KMeans regimes)")
            by_regime = harness.summarize(
                forecasts_r.dropna(subset=["regime_id"]),
                by=["method", "regime_id"],
            )
            st.dataframe(
                by_regime.style.format(
                    {
                        "mae": "{:.3f}",
                        "mape": "{:.3%}",
                        "dir_acc": "{:.1%}",
                        "crps": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )
    except FileNotFoundError:
        st.caption(
            "Regime labels not available -- run `make regime` to enable per-regime stratification."
        )
    except Exception as e:  # noqa: BLE001
        st.caption(f"Regime stratification failed: {e}")

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
    col_a, col_b = st.columns([1, 1])
    with col_a:
        asset_choice = st.selectbox("Asset", sorted(forecasts["asset"].unique()))
    with col_b:
        method_choice = st.selectbox(
            "Method (interval bands shown for the selected method)",
            sorted(forecasts["method"].unique()),
        )

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

    # Uncertainty bands for the selected method (only if it produced intervals)
    method_data = (
        asset_forecasts[asset_forecasts["method"] == method_choice]
        .drop_duplicates("timestamp")
        .set_index("timestamp")
        .sort_index()
    )
    if "lower" in method_data.columns and method_data["lower"].notna().any():
        st.caption(
            f"80% prediction interval for {method_choice} on {asset_choice}"
        )
        band = method_data[["lower", "prediction", "upper", "target"]].astype(float)
        st.line_chart(band, height=300)
    else:
        st.caption(
            f"{method_choice} is a point-only method (no interval bands)."
        )


def render_regime_explorer() -> None:
    st.title("Regime Explorer")
    st.caption(
        "Latent market regimes from a contrastive temporal encoder + KMeans, "
        "with a Gaussian HMM as a sanity-check baseline."
    )

    from autosignalx.config import settings

    regime_dir = settings.reports_dir / "regimes"
    if not regime_dir.exists() or not list(regime_dir.glob("*.parquet")):
        st.warning(
            "No regime labels yet. Run `make regime` (or "
            "`uv run autosignalx regime fit`) to populate."
        )
        return

    km = pd.read_parquet(regime_dir / "kmeans.parquet")
    hmm_path = regime_dir / "hmm.parquet"
    embed_path = regime_dir / "embeddings.parquet"
    hmm = pd.read_parquet(hmm_path) if hmm_path.exists() else None
    embed = pd.read_parquet(embed_path) if embed_path.exists() else None

    cols = st.columns(3)
    cols[0].metric("KMeans regimes", km["regime_id"].nunique())
    cols[0].caption(f"{len(km):,} labeled timesteps")
    if hmm is not None:
        cols[1].metric("HMM regimes", hmm["regime_id"].nunique())
        cols[1].caption(f"{len(hmm):,} labeled timesteps")

    st.divider()
    st.subheader("Regime timeline (KMeans on contrastive embeddings)")
    timeline = (
        km.set_index("timestamp")["regime_id"]
        .sort_index()
        .astype("int")
    )
    st.line_chart(timeline, height=200)

    if hmm is not None:
        st.subheader("Regime timeline (Gaussian HMM, sanity check)")
        st.line_chart(
            hmm.set_index("timestamp")["regime_id"].sort_index().astype("int"),
            height=200,
        )

    if embed is not None and len(embed) > 0:
        st.divider()
        st.subheader("Embedding visualization (PCA-2D, colored by KMeans regime)")
        import plotly.express as px
        from sklearn.decomposition import PCA

        embedding_cols = [c for c in embed.columns if c != "timestamp"]
        embed_aligned = embed.merge(km[["timestamp", "regime_id"]], on="timestamp", how="inner")
        if len(embed_aligned) >= 2 and len(embedding_cols) >= 2:
            pca = PCA(n_components=2)
            xy = pca.fit_transform(embed_aligned[embedding_cols].to_numpy())
            scatter_df = pd.DataFrame(
                {
                    "PC1": xy[:, 0],
                    "PC2": xy[:, 1],
                    "regime": embed_aligned["regime_id"].astype("string"),
                    "timestamp": embed_aligned["timestamp"].astype("string"),
                }
            )
            fig = px.scatter(
                scatter_df,
                x="PC1",
                y="PC2",
                color="regime",
                hover_data=["timestamp"],
                opacity=0.6,
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_signal_lab() -> None:
    st.title("Signal Discovery Lab")
    st.caption(
        "Per-regime feature relevance ranking via HistGradientBoosting + "
        "permutation importance. Higher importance => shuffling the feature "
        "hurt accuracy more."
    )

    from autosignalx.config import settings

    signals_dir = settings.reports_dir / "signals"
    ranking_files = sorted(signals_dir.glob("*.parquet")) if signals_dir.exists() else []
    if not ranking_files:
        st.warning(
            "No signal ranking yet. Run `make signal` "
            "(or `uv run autosignalx signal rank`) to populate."
        )
        return

    # Use the most recently modified ranking file
    ranking_path = max(ranking_files, key=lambda p: p.stat().st_mtime)
    st.caption(f"Loaded ranking from `{ranking_path.name}`.")
    rankings = pd.read_parquet(ranking_path)
    regimes = sorted(rankings["regime_id"].unique())

    sel = st.selectbox("Regime", regimes)
    sub = rankings[rankings["regime_id"] == sel].sort_values("rank")

    cols = st.columns(2)
    with cols[0]:
        st.metric("Features ranked", len(sub))
    with cols[1]:
        if "n_samples" in sub.columns and len(sub) > 0:
            st.metric("Samples in regime", f"{int(sub['n_samples'].iloc[0]):,}")

    st.divider()
    st.subheader(f"Regime {sel} -- ranked features")
    st.bar_chart(
        sub.set_index("feature")["importance"].sort_values(ascending=True),
        height=500,
    )
    st.dataframe(
        sub[["rank", "feature", "importance", "importance_std", "n_samples"]].style.format(
            {
                "importance": "{:+.3f}",
                "importance_std": "{:.3f}",
                "n_samples": "{:,}",
            }
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Top features across regimes (importance heatmap)")
    top_per_regime = (
        rankings.sort_values("rank").groupby("regime_id").head(8)["feature"].unique()
    )
    pivoted = (
        rankings[rankings["feature"].isin(top_per_regime)]
        .pivot_table(
            index="feature", columns="regime_id", values="importance", aggfunc="mean"
        )
        .sort_index()
    )
    if not pivoted.empty:
        try:
            styled = pivoted.style.background_gradient(cmap="RdYlGn", axis=None).format("{:+.3f}")
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(pivoted, use_container_width=True)


def render_cross_asset_graph() -> None:
    st.title("Cross-Asset Graph")
    st.caption(
        "Direct (partial) correlations and Granger-causal edges between ETFs. "
        "Hubs (high eigenvector centrality) are assets whose moves propagate widely."
    )

    from autosignalx.config import settings

    graph_dir = settings.reports_dir / "graph"
    if not graph_dir.exists() or not (graph_dir / "edges.parquet").exists():
        st.warning(
            "No graph yet. Run `make graph` (or `uv run autosignalx graph build`)."
        )
        return

    edges = pd.read_parquet(graph_dir / "edges.parquet")
    cent_path = graph_dir / "centrality.parquet"
    cent = pd.read_parquet(cent_path) if cent_path.exists() else None

    pcorr = edges[edges["edge_type"] == "partial_corr"]
    granger = edges[edges["edge_type"] == "granger"]

    cols = st.columns(3)
    cols[0].metric("Partial-corr edges", len(pcorr))
    cols[1].metric("Granger edges", len(granger))
    cols[2].metric("Nodes", cent["node"].nunique() if cent is not None else "-")

    if cent is not None:
        st.divider()
        st.subheader("Centrality (sorted by eigenvector centrality)")
        st.dataframe(
            cent.style.format(
                {
                    "degree_centrality": "{:.3f}",
                    "eigenvector_centrality": "{:.3f}",
                    "betweenness_centrality": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Partial-correlation matrix")
    if not pcorr.empty:
        nodes = sorted(set(pcorr["source"]).union(pcorr["target"]))
        mat = pd.DataFrame(0.0, index=nodes, columns=nodes)
        for _, row in pcorr.iterrows():
            mat.at[row["source"], row["target"]] = row["weight"]
            mat.at[row["target"], row["source"]] = row["weight"]
        try:
            styled = mat.style.background_gradient(cmap="RdBu_r", axis=None, vmin=-1, vmax=1).format("{:+.2f}")
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(mat, use_container_width=True)

    if not granger.empty:
        st.divider()
        st.subheader("Top Granger-causal edges (lower p = stronger)")
        top = granger.sort_values("p_value").head(20)
        st.dataframe(
            top[["source", "target", "best_lag", "p_value", "weight"]].style.format(
                {"p_value": "{:.4f}", "weight": "{:.2f}"}
            ),
            use_container_width=True,
        )


def render_agent_console() -> None:
    st.title("Agent Console")
    st.caption(
        "LangGraph state machine over DeepInfra LLMs (or replay mode). "
        "Every step is appended to a persistent JSONL ledger -- the system's "
        "long-horizon memory cell."
    )

    from autosignalx.agent import ledger as ledger_mod
    from autosignalx.config import settings

    mode = "replay" if settings.use_replay else "live"
    cols = st.columns(3)
    cols[0].metric("Mode", mode)
    entries = ledger_mod.load()
    cols[1].metric("Ledger entries", len(entries))
    if entries:
        last_round = max(e.get("round", 0) for e in entries)
        cols[2].metric("Last round", int(last_round))

    if not entries:
        st.warning(
            "No ledger entries yet. Run `make agent` (or "
            "`uv run autosignalx agent run`) to populate."
        )
        return

    st.divider()
    st.subheader("Trace timeline")
    role_icon = {
        "propose": "🧠",
        "theorist": "💡",
        "skeptic": "🔍",
        "experiment": "🧪",
        "critique": "📝",
        "adjudicator": "⚖️",
        "decide": "🎯",
    }
    for e in entries:
        rd = e.get("round", "?")
        step = e.get("step", "?")
        ts = e.get("ts", "")
        content = e.get("content", "")
        icon = role_icon.get(step, "•")
        is_user_role = step == "experiment"
        chat_role = "user" if is_user_role else "assistant"
        with st.chat_message(chat_role):
            st.markdown(f"{icon} **round {rd} -- {step}**  *({ts})*")
            if step in ("propose", "theorist"):
                if isinstance(content, dict):
                    st.markdown(f"_Hypothesis_: {content.get('hypothesis', '')}")
                    exp = content.get("experiment", {})
                    if exp:
                        st.code(json.dumps(exp, indent=2), language="json")
            elif step == "experiment":
                st.json(content)
                if isinstance(content, dict) and content.get("promoted_finding_id"):
                    st.success(f"✓ Promoted finding: `{content['promoted_finding_id']}`")
            elif step in ("critique", "skeptic", "adjudicator"):
                st.markdown(content if isinstance(content, str) else str(content))
            elif step == "decide":
                st.json(content)
            else:
                st.write(content)


def render_ask_the_memory() -> None:
    st.title("Ask the Memory")
    st.caption(
        "Free-form question against the agent's experiment ledger. In live "
        "mode, the LLM answers using ledger context. In replay mode, simple "
        "keyword search returns matching ledger entries."
    )

    from autosignalx.agent import ledger as ledger_mod
    from autosignalx.config import settings

    entries = ledger_mod.load()
    if not entries:
        st.warning("Ledger empty. Run `make agent` first.")
        return

    if "memory_history" not in st.session_state:
        st.session_state.memory_history = []

    for q, a in st.session_state.memory_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    question = st.chat_input("Ask the agent's memory...")
    if not question:
        return

    if settings.use_replay or not settings.deepinfra_api_key:
        # Deterministic fallback: keyword search
        ql = question.lower()
        hits = [
            e for e in entries
            if any(
                ql_term in json.dumps(e, default=str).lower()
                for ql_term in ql.split()
                if len(ql_term) > 2
            )
        ][:8]
        if hits:
            answer_parts = ["**Replay-mode keyword search** (no LLM call):"]
            for e in hits:
                answer_parts.append(
                    f"- round {e.get('round')} {e.get('step')}: "
                    f"{json.dumps(e.get('content', ''), default=str)[:200]}"
                )
            answer = "\n".join(answer_parts)
        else:
            answer = "_No ledger entries matched. Try different keywords._"
    else:
        # Live mode: stuff ledger into the prompt
        try:
            from autosignalx.agent.ledger import summarize_for_prompt
            from autosignalx.agent.llm import get_provider

            provider = get_provider(record_replay=False)
            ledger_summary = summarize_for_prompt(entries, limit=40)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You answer questions about an experiment ledger from a "
                        "quantitative ML research agent. Cite specific rounds when relevant."
                    ),
                },
                {
                    "role": "user",
                    "content": f"## Ledger\n{ledger_summary}\n\n## Question\n{question}",
                },
            ]
            answer = provider.chat(messages, step="ask_memory", round=-1)
        except Exception as e:  # noqa: BLE001
            answer = f"LLM call failed: {e}"

    st.session_state.memory_history.append((question, answer))
    st.rerun()


def render_findings() -> None:
    st.title("Findings")
    st.caption(
        "Promoted findings -- hypotheses that passed the statistical "
        "promotion gate (Diebold-Mariano p < 0.05 AND positive bootstrap CI). "
        "Each card carries the full evidence trail."
    )

    from autosignalx.agent import findings as findings_mod

    rows = findings_mod.load()
    if not rows:
        st.info(
            "No promoted findings yet. Run the agent (`make agent`); it "
            "automatically attempts to promote each experiment that names "
            "a non-naive method."
        )
        return

    rows_sorted = sorted(
        rows,
        key=lambda r: r.get("evidence", {}).get("skill_vs_baseline", -1.0),
        reverse=True,
    )
    cols = st.columns(3)
    cols[0].metric("Total findings", len(rows_sorted))
    cols[1].metric("Sessions producing findings", len({r.get("session_id") for r in rows_sorted}))
    cols[2].metric("Best skill vs naive", f"{rows_sorted[0].get('evidence', {}).get('skill_vs_baseline', 0.0):+.3f}")

    st.divider()
    for r in rows_sorted:
        ev = r.get("evidence", {})
        with st.expander(
            f"{r.get('id', '?')}  -  skill +{ev.get('skill_vs_baseline', 0):.3f}  "
            f"(p={ev.get('p_value', float('nan')):.4f}, "
            f"replications={r.get('replication_count', 1)})  -  "
            f"{r.get('method', '?')}",
            expanded=False,
        ):
            st.markdown(f"**Hypothesis** ({r.get('session_id')} round {r.get('round')}):")
            st.markdown(f"> {r.get('hypothesis', '')}")
            st.markdown("**Filters**:")
            st.code(json.dumps(r.get("filters", {}), indent=2), language="json")
            st.markdown("**Evidence**:")
            st.code(json.dumps(ev, indent=2, default=str), language="json")
            st.markdown(f"**Agent confidence**: _{r.get('agent_confidence', '')}_")
            if r.get("replication_count", 1) > 1:
                st.markdown(f"**Replications**: {r['replication_count']}")
                st.code(json.dumps(r.get("replications", []), indent=2), language="json")


def render_lineage() -> None:
    st.title("Hypothesis Lineage")
    st.caption(
        "DAG of hypotheses across rounds: nodes are unique hypotheses "
        "(deduped by content hash), edges show inferred parent->child "
        "refinements. Status colors: green=promoted, red=refuted, gray=open."
    )

    from autosignalx.agent import lineage as lineage_mod

    lineage = lineage_mod.build_lineage()
    nodes = lineage.get("nodes", [])
    if not nodes:
        st.warning("No hypotheses in the ledger yet. Run `make agent`.")
        return

    cols = st.columns(3)
    cols[0].metric("Hypotheses", len(nodes))
    cols[1].metric("Promoted", sum(1 for n in nodes if n["status"] == "promoted"))
    cols[2].metric("Refuted", sum(1 for n in nodes if n["status"] == "refuted"))

    st.divider()
    df = lineage_mod.lineage_dataframe(lineage)
    st.dataframe(df, use_container_width=True)

    if not lineage.get("edges"):
        st.caption("No parent->child edges inferred yet (need >=2 hypotheses with overlapping method/asset/regime).")
        return

    # Render the DAG with networkx + plotly
    try:
        import networkx as nx
        import plotly.graph_objects as go

        g = nx.DiGraph()
        for n in nodes:
            g.add_node(n["id"], **n)
        for e in lineage["edges"]:
            g.add_edge(e["source"], e["target"])

        # Layered layout by round
        pos = {}
        rounds = sorted({n["round"] for n in nodes})
        for r in rounds:
            ns_at = [n for n in nodes if n["round"] == r]
            for i, n in enumerate(ns_at):
                pos[n["id"]] = (r, i - len(ns_at) / 2)

        edge_x, edge_y = [], []
        for src, tgt in g.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line={"width": 1, "color": "#888"},
            hoverinfo="none",
            mode="lines",
        )
        color_map = {"promoted": "#2ca02c", "refuted": "#d62728", "open": "#7f7f7f"}
        node_x, node_y, node_color, node_text, node_hover = [], [], [], [], []
        for n in nodes:
            x, y = pos[n["id"]]
            node_x.append(x)
            node_y.append(y)
            node_color.append(color_map.get(n["status"], "#7f7f7f"))
            node_text.append(n["id"])
            params = json.dumps(n["params"], default=str)[:80]
            node_hover.append(f"{n['id']} ({n['status']})<br>round {n['round']}<br>{n['hypothesis'][:120]}<br>params: {params}")
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="bottom center",
            hovertext=node_hover,
            hoverinfo="text",
            marker={"size": 18, "color": node_color, "line": {"width": 1, "color": "#333"}},
        )
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Lineage DAG (round on x-axis)",
                showlegend=False,
                xaxis={"title": "round", "showgrid": False, "zeroline": False},
                yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                height=500,
                margin={"l": 20, "r": 20, "t": 60, "b": 40},
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"DAG rendering failed: {e}")


PANELS = {
    "Overview": render_overview,
    "Data": render_data,
    "Forecast Arena": render_forecast_arena,
    "Regime Explorer": render_regime_explorer,
    "Signal Discovery Lab": render_signal_lab,
    "Cross-Asset Graph": render_cross_asset_graph,
    "Agent Console": render_agent_console,
    "Findings": render_findings,
    "Lineage": render_lineage,
    "Ask the Memory": render_ask_the_memory,
}


# Streamlit executes the script top-to-bottom on every interaction.
panel_name = st.sidebar.radio("Panel", list(PANELS.keys()))
st.sidebar.divider()
st.sidebar.caption(f"AutoSignal-X v{__version__}")
PANELS[panel_name]()
