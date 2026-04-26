"""AutoSignal-X — Streamlit research cockpit.

15 panels in the sidebar, each a read-only viewer over a typed artifact
written by one of the system's layers. Every panel includes a standardized
'About this panel' expander documenting its inputs, operations / algorithms,
goal, and how to interpret the results."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from autosignalx import __version__
from autosignalx.config import settings

st.set_page_config(
    page_title="AutoSignal-X",
    layout="wide",
)


def _load_runtime_checks_module():
    """Load runtime checks from the current repo checkout, not site-packages."""
    helper_path = Path(__file__).resolve().parents[1] / "src" / "autosignalx" / "runtime_checks.py"
    spec = importlib.util.spec_from_file_location(
        "autosignalx_repo_runtime_checks", helper_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runtime checks from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_supported_runtime() -> None:
    expected_repo_root = Path(__file__).resolve().parents[1]
    checks = _load_runtime_checks_module()
    result = checks.validate_repo_runtime(expected_repo_root)
    if result["ok"]:
        return

    st.title("Runtime setup required")
    st.error(
        "This cockpit is not running against the current AutoSignal-X checkout. "
        "It was likely launched from a stale global or Conda environment, so "
        "study-aware code is out of sync with the app."
    )
    st.markdown("Refresh the repo-local environment and relaunch from the repo root:")
    st.code("uv sync --all-extras\nuv run streamlit run app/streamlit_app.py")
    st.caption(f"Expected repo root: {expected_repo_root}")
    with st.expander("Diagnostics"):
        st.markdown("**Errors**")
        for err in result["errors"]:
            st.markdown(f"- {err}")
        st.markdown("**Loaded modules**")
        for detail in result["details"]:
            st.markdown(f"- {detail}")
    st.stop()


_require_supported_runtime()


def _panel_doc(
    inputs: str,
    operations: str,
    goal: str,
    interpretation: str,
    expanded: bool = False,
) -> None:
    """Standardized 'About this panel' expander rendering 4 structured fields."""
    with st.expander("About this panel", expanded=expanded):
        st.markdown(f"**Inputs (data this panel reads).** {inputs}")
        st.markdown(f"**Operations / algorithms.** {operations}")
        st.markdown(f"**Goal.** {goal}")
        st.markdown(f"**How to interpret.** {interpretation}")


def render_overview() -> None:
    st.title("AutoSignal-X")
    st.caption(
        "A 5-layer modular AI research system for discovering predictive structure "
        "in liquid daily ETF prices, paired with a multi-agent research loop."
    )

    st.markdown(
        """
        **Research question.** For which (regime, asset, method) combinations does a
        layered forecasting stack outperform the naive baseline on daily ETF prices,
        and is that outperformance statistically significant under both Diebold-Mariano
        (p < 0.05) and a positive bootstrap CI on the loss difference?

        **Architecture.** Five model layers (L1-L5) plus an agent that orchestrates
        them. Each layer persists outputs as typed parquet/JSONL artifacts under
        `reports/`; the agent reads everything through a shared tool surface and
        writes its own structured outputs (ledger, findings, lessons, telemetry,
        trace quality, self-critique). Every cockpit panel is a read-only viewer
        over those artifacts.
        """
    )

    st.subheader("Model layers")
    layer_rows = [
        ("L1 Forecasting", "Probabilistic point + interval forecasts", "Chronos-2 (multivariate, with covariates) + baselines (naive, seasonal-naive, ARIMA(1,1,1) on log-prices)"),
        ("L2 Representation", "Per-timestep latent regime labels", "Contrastive 1D-CNN encoder (16-dim, 60-day windows, triplet loss) + KMeans on embeddings; Gaussian HMM as parallel detector"),
        ("L3 Reasoning", "Per-regime feature importance", "HistGradientBoostingClassifier per regime + custom permutation importance"),
        ("L4 Relational", "Cross-asset dependency structure", "GLASSO partial correlations + Granger causality + NetworkX centrality"),
        ("L5 Agentic", "Hypothesis generation, experimentation, statistical promotion", "LangGraph; debate mode = Theorist (Kimi-K2.6) / Skeptic (GLM-5.1) / Adjudicator (DeepSeek-V4-Pro); 3 ways to author experiments (slice / DSL / sandboxed Python)"),
    ]
    for name, purpose, impl in layer_rows:
        st.markdown(f"- **{name}** — *{purpose}.*  {impl}")

    st.divider()
    st.subheader("Cockpit panels (sidebar order)")
    panel_rows = [
        ("Overview", "This page. System pitch, layer summary, panel index, system status."),
        ("Data", "Cache inventory; ETF and macro time series. Reads `data/cache/*.parquet`."),
        ("Forecast Arena", "Per-method overall metrics; per-(method, regime) stratified metrics; per-asset trajectory chart with 80% interval bands. Reads `reports/ablations/*.parquet`."),
        ("Regime Explorer", "KMeans + HMM regime timelines; PCA-2D scatter of contrastive embeddings colored by regime. Reads `reports/regimes/*.parquet`."),
        ("Signal Discovery Lab", "Per-regime feature importance bar chart; ranking table; cross-regime importance heatmap. Reads `reports/signals/signal_ranking.parquet`."),
        ("Cross-Asset Graph", "Centrality table; partial-correlation matrix; top Granger edges. Reads `reports/graph/{edges,centrality}.parquet`."),
        ("Backtest Arena", "Simulated trading on the test window driven by discovered structure (Phase 1). Equity curves, drawdowns, Sharpe/Sortino/Calmar; strict no-look-ahead. Reads `reports/backtest/runs/<run_id>/`."),
        ("Agent Console", "Chat-style ledger timeline; per-round trace-quality chart at the bottom. Reads `reports/agent/ledger.jsonl`."),
        ("Auto-Play Replay", "Playback controls (play / pause / reset, 0.5x-4x speed) over the ledger."),
        ("Findings", "Promoted findings (passed DM + bootstrap gate) sorted by skill-vs-naive; full statistical evidence per card. Reads `reports/agent/findings.jsonl`."),
        ("Lineage", "Plotly DAG of hypothesis evolution; nodes colored by status (promoted / refuted / open). Inferred via `agent/lineage.py`."),
        ("Self-Critique", "Agent's verdicts on its own past findings against current evidence. Reads `reports/agent/self_critique.jsonl`."),
        ("Lessons & Memory", "Accumulating Markdown of consolidated session notes (long-horizon memory). Reads `reports/agent/lessons.md`."),
        ("Telemetry", "Cost / tokens / latency per LLM call; per-model and per-step breakdown; cumulative cost. Reads `reports/agent/telemetry.jsonl`."),
        ("Sessions", "Per-session productivity (rounds, findings, cost-per-finding); cumulative trend. Aggregates all stores by `session_id`."),
        ("Ask the Memory", "Free-form chat against the ledger (LLM in live mode, keyword search in replay mode)."),
    ]
    for name, desc in panel_rows:
        st.markdown(f"- **{name}** — {desc}")

    st.divider()
    st.subheader("Inputs and outputs")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            """
            **Inputs**
            - **yfinance API**: 8 ETFs (SPY, QQQ, IWM, GLD, TLT, EFA, EEM, HYG) + 4 macro signals (^TNX, ^VIX, DX-Y.NYB, CL=F), daily 2010-01-01 → 2025-12-31.
            - **DeepInfra API key** (optional): OpenAI-compatible endpoint for the agent layer. Without a key, the agent runs deterministically against `replay/agent_steps.jsonl`.
            - **`configs/default.yaml`**: date splits, horizon, per-layer hyperparameters.
            """
        )
    with cols[1]:
        st.markdown(
            """
            **Outputs (all under `reports/`)**
            - `ablations/*.parquet` — per-method walk-forward forecasts.
            - `regimes/*.parquet` — KMeans + HMM regime labels and embeddings.
            - `signals/signal_ranking.parquet` — per-regime feature importance.
            - `graph/{edges,centrality}.parquet` — partial-corr + Granger; centrality.
            - `agent/{ledger,findings,telemetry,trace_quality,self_critique}.jsonl` — agent state and observability.
            - `agent/lessons.md` — long-horizon memory.
            - `agent/generated_methods/` — sandboxed Python authored by the agent.
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
        "See **REPORT.md** for research questions, methodology, and results. "
        "See **docs/ARCHITECTURE.md** for data flow, contracts, per-layer wiring, "
        "agent loop, and the sandbox model."
    )


def render_data() -> None:
    st.title("Data")
    st.caption("ETF OHLCV and macro signal cache backing every experiment.")

    _panel_doc(
        inputs="`data/cache/ohlcv.parquet` (8 ETFs, daily 2010-01-01 → 2025-12-31, ~32k rows) and `data/cache/macro.parquet` (^TNX, ^VIX, DX-Y.NYB, CL=F, ~16k rows). Both pulled from yfinance via `make data`.",
        operations="Long-format parquet I/O with schema enforcement at the persistence boundary (`data/schema.py:assert_*`). Wide-format pivots via `data/loader.py` for visualization (close, returns, macro). Per-asset timestamps strictly monotonic increasing (asserted at write).",
        goal="Show the substrate every other layer reads from. Reviewers see at a glance what data is in scope, the time range covered, and the qualitative shape of price and macro series.",
        interpretation="Row counts and date range should match the configured window. The normalized-close chart shows relative ETF performance; the macro chart shows the level evolution of the four signals (yield, vol, dollar, oil). If any cache is missing, run `make data`.",
    )

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

    _panel_doc(
        inputs="All `reports/ablations/*.parquet` (concatenated). Each file is one method's forecasts on the walk-forward harness, conforming to `eval/contracts.py:FORECAST_COLUMNS_REQUIRED`. Optionally joined with `reports/regimes/kmeans.parquet` on `forecast_origin` for stratification.",
        operations="`eval/harness.py:summarize` aggregates by method (or by method × regime); `eval/metrics.py` computes MAE, MAPE, directional accuracy, skill-vs-naive (`1 − method_mae / naive_mae`), and CRPS (from the (lower, prediction, upper) triple via pinball-loss for methods with intervals). The trajectory chart pivots predictions by method and overlays the realized target.",
        goal="Identify which forecasting methods (and which method × regime combinations) outperform the naive baseline on walk-forward forecasts. CRPS shows whether probabilistic methods produce calibrated intervals.",
        interpretation="`skill_vs_naive > 0` means the method beats naive on MAE; positive but small (<1%) is unlikely to be statistically significant. CRPS is in `adj_close` units; compare across probabilistic methods. Per-(method, regime) rows surface conditional improvements that the overall view masks. The trajectory chart's interval bands show the model's stated uncertainty.",
    )

    from autosignalx.config import settings
    from autosignalx.eval import harness

    active_study = st.session_state.get("active_study")
    if active_study:
        from autosignalx.study import Study

        try:
            ablations_dir = Study.load(active_study).ablations_dir
            st.info(f"Reading from study scope: **{active_study}**")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not load study {active_study!r}: {e}")
            return
    else:
        ablations_dir = settings.reports_dir / "ablations"

    parquets: list[Path] = sorted(ablations_dir.glob("*.parquet")) if ablations_dir.exists() else []
    if not parquets:
        scope = f" (study={active_study})" if active_study else ""
        st.info(
            f"No ablation results cached yet{scope}. Run `make baseline` (or "
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

    _panel_doc(
        inputs="`reports/regimes/kmeans.parquet` (window-aligned KMeans labels), `reports/regimes/hmm.parquet` (per-day HMM labels), `reports/regimes/embeddings.parquet` (16-dim contrastive embeddings, one per 60-day window). Produced by `make regime` from market features = SPY+QQQ daily returns + 4 macro signals (standardized).",
        operations="**KMeans branch**: 1D-CNN encoder (Conv → GELU × 2 → AdaptiveAvgPool1d → Linear) trained 25 epochs with `nn.TripletMarginLoss` (positive: ±3-day adjacent windows; negative: ≥60-day distant windows); KMeans (n_init=10) on 16-dim embeddings. **HMM branch**: `hmmlearn.GaussianHMM(n_components=4, covariance_type='diag', n_iter=100)` directly on standardized features. PCA scatter via `sklearn.decomposition.PCA(n_components=2)` on the embeddings.",
        goal="Surface the latent market states downstream layers condition on. Regime labels feed the signal layer (per-regime ranking) and the agent's hypothesis space (per-regime, per-asset slices).",
        interpretation="The two timelines should *broadly* agree in segment structure if the regimes are real (exact alignment is not expected — different methods, different units). The PCA scatter should show visible clusters; if it looks like a uniform blob, the contrastive training failed to separate states. Each regime's relative size determines how much data the signal layer has to fit per regime.",
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

    _panel_doc(
        inputs="Most recently modified `reports/signals/*.parquet`. Produced by `make signal`. Each row is `(regime_id, feature, importance, importance_std, n_samples, rank)`.",
        operations="For each regime: subsample up to 2,000 (asset, timestamp) rows whose KMeans label = R; build features (8 technical: rolling mean/std, momentum, RSI-14, MACD signal + 8 macro: level + 5-day change for each of ^TNX/^VIX/DX-Y.NYB/CL=F); fit `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4)` on (features, binary direction at horizon=21); custom permutation importance with `n_repeats=2` (shuffle one feature at a time, measure accuracy drop, average).",
        goal="Identify which features carry the most signal for predicting next-21-day price direction within each regime. Feeds the agent's per-regime hypothesis generation.",
        interpretation="Within each regime, rank 1 is the most important feature (largest accuracy drop when shuffled). Compare across regimes via the heatmap: features with consistently high importance are universal; features that dominate one regime and not others are conditionally important. Importance std gives a coarse sense of stability across the 2 permutation repeats.",
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

    _panel_doc(
        inputs="`reports/graph/edges.parquet` (combined partial-correlation + Granger edges) and `reports/graph/centrality.parquet` (per-node degree / eigenvector / betweenness). Built from daily returns of all 8 ETFs via `make graph`.",
        operations="**Partial correlations**: `sklearn.covariance.GraphicalLassoCV(cv=3)` fits a sparse precision matrix on standardized returns; off-diagonals are normalized to partial correlations (direct relationships after controlling for all other assets). **Granger causality**: `statsmodels.tsa.stattools.grangercausalitytests(max_lag=5)` for every ordered pair; min p-value across lags compared to threshold 0.05; weight = -log10(p). **Centrality**: NetworkX `degree_centrality`, `eigenvector_centrality_numpy`, `betweenness_centrality` on the partial-correlation graph (treated as undirected, |weight|).",
        goal="Identify hub assets (whose moves propagate widely), bridge assets (that connect distinct clusters), and isolated assets (that diversify) in the cross-asset structure. The agent uses these typed roles when proposing per-asset hypotheses.",
        interpretation="**Eigenvector centrality** measures connectedness to other connected nodes (hubs). **Betweenness** measures how often a node lies on shortest paths between other pairs (bridges). The partial-correlation matrix is a diverging-color heatmap: dark red / blue cells indicate strong direct relationships. Granger edges are descriptive ranking under common-factor confounding (treat them as candidate information-flow paths, not causal claims).",
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

    _panel_doc(
        inputs="`reports/agent/ledger.jsonl` (append-only, one JSON per agent step), and `reports/agent/trace_quality.jsonl` for the per-round quality scores at the bottom.",
        operations="Renders ledger entries chronologically as `st.chat_message` rows, one per step. Steps come from two graph topologies: **single mode** (`propose / experiment / critique / decide`, all from one LLM) and **debate mode** (`theorist / skeptic / experiment / adjudicator`, three different DeepInfra models per round). Auto-promotion runs the DM + bootstrap gate on every non-naive method's experiment slice; passing experiments are tagged with a `promoted_finding_id` and rendered with a green checkmark.",
        goal="Make the agent's reasoning chain inspectable. Reviewers see exactly what was hypothesized, what evidence was tested, what the critic / adjudicator said, and what was decided.",
        interpretation="Each chat row is one step in one round. Hypotheses are JSON with `hypothesis` text + `experiment` spec. Experiment results show metrics on the slice. Adjudicator messages end with `VERDICT: support | refute | inconclusive`. The trace-quality chart at the bottom plots per-round LLM-judge scores (clarity / novelty / falsifiability / evidence-citing, 1-5 scale); upward trend = the agent is asking sharper questions over time.",
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

    st.divider()
    render_trace_quality_chart()


def render_ask_the_memory() -> None:
    st.title("Ask the Memory")
    st.caption(
        "Free-form question against the agent's experiment ledger. In live "
        "mode, the LLM answers using ledger context. In replay mode, simple "
        "keyword search returns matching ledger entries."
    )

    _panel_doc(
        inputs="`reports/agent/ledger.jsonl` (full agent step history). Free-form text query from `st.chat_input`.",
        operations="**Live mode** (DEEPINFRA_API_KEY set, AUTOSIGNALX_REPLAY != true): summarize the most recent 40 ledger entries; send `(system: 'answer questions about an experiment ledger; cite specific rounds') + (user: ledger summary + question)` to the chat-role LLM. **Replay mode** (no key): split the question into terms (length > 2), filter ledger entries whose JSON-stringified content contains any term, return the first 8 matches.",
        goal="Let reviewers query the agent's memory in natural language without browsing the raw JSONL.",
        interpretation="Live answers cite specific rounds (e.g., 'in round 3 the agent proposed X'); replay answers list matching ledger entries with their round / step / content excerpt. Prior chat history is preserved within the session via `st.session_state.memory_history`.",
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


def render_trace_quality_chart() -> None:
    """Sub-renderer used by Agent Console to show quality trends per session."""
    from autosignalx.agent import trace_eval

    rows = trace_eval.load()
    if not rows:
        return
    st.subheader("Trace quality over rounds (LLM-as-judge, 1-5)")
    df = pd.DataFrame(rows)
    df = df[["round", "clarity", "novelty", "falsifiability", "evidence_citing"]].copy()
    df = df.set_index("round")
    st.line_chart(df, height=250)
    st.caption(
        "Higher = better. Run `autosignalx agent score-traces` after each "
        "session to refresh."
    )


def render_auto_play() -> None:
    st.title("Auto-Play Replay")
    st.caption(
        "Press play to watch the agent's recorded research session unfold "
        "round by round. The trace is read from `reports/agent/ledger.jsonl` "
        "(or the committed replay) -- live and replay sessions both render here."
    )

    _panel_doc(
        inputs="`reports/agent/ledger.jsonl`. Three `st.session_state` keys: `playback_idx`, `playback_speed`, `is_playing`.",
        operations="Manual controls: play / pause / reset buttons; speed slider (0.5x / 1x / 2x / 4x); step slider for direct jump. While `is_playing`, `playback_idx` advances on each Streamlit rerun via `st.rerun()` after `time.sleep(1.0 / speed)`. Each visible step is rendered as a chat-style message with a step-letter icon.",
        goal="Pace the agent's reasoning visually. Reviewers can stop on any step, scrub back and forth, and inspect specific rounds without scrolling through the full Agent Console.",
        interpretation="The progress bar shows `current_step / total_steps`. Promoted findings during playback show a green badge with the finding ID — click through to the Findings panel for the full evidence trail.",
    )

    from autosignalx.agent import ledger as ledger_mod

    entries = ledger_mod.load()
    if not entries:
        st.info("No ledger entries yet. Run `make agent` to record a session.")
        return

    if "playback_idx" not in st.session_state:
        st.session_state.playback_idx = 0
    if "playback_speed" not in st.session_state:
        st.session_state.playback_speed = 1.0
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        if st.button("Play"):
            st.session_state.is_playing = True
    with col2:
        if st.button("Pause"):
            st.session_state.is_playing = False
    with col3:
        if st.button("Reset"):
            st.session_state.playback_idx = 0
            st.session_state.is_playing = False
    with col4:
        st.session_state.playback_speed = st.select_slider(
            "Speed",
            options=[0.5, 1.0, 2.0, 4.0],
            value=st.session_state.playback_speed,
            label_visibility="collapsed",
        )

    st.session_state.playback_idx = st.slider(
        "Round",
        min_value=0,
        max_value=len(entries),
        value=st.session_state.playback_idx,
    )
    progress = st.session_state.playback_idx / max(1, len(entries))
    st.progress(progress, text=f"Step {st.session_state.playback_idx} / {len(entries)}")

    visible = entries[: st.session_state.playback_idx]
    role_icon = {
        "propose": "P",
        "theorist": "T",
        "skeptic": "S",
        "experiment": "E",
        "critique": "C",
        "adjudicator": "A",
        "decide": "D",
    }
    for e in visible:
        rd = e.get("round", "?")
        step = e.get("step", "?")
        content = e.get("content", "")
        with st.chat_message("user" if step == "experiment" else "assistant"):
            st.markdown(f"**[{role_icon.get(step, '*')}] round {rd} -- {step}**")
            if isinstance(content, dict):
                if step in ("propose", "theorist"):
                    st.markdown(f"_Hypothesis_: {content.get('hypothesis', '')}")
                else:
                    st.json(content)
            else:
                st.markdown(str(content)[:800])

    if st.session_state.is_playing and st.session_state.playback_idx < len(entries):
        import time

        time.sleep(max(0.2, 1.0 / st.session_state.playback_speed))
        st.session_state.playback_idx += 1
        st.rerun()


def render_self_critique() -> None:
    st.title("Self-Critique")
    st.caption(
        "The agent re-reads its own promoted findings against the current "
        "state of the ledger and other findings. Verdicts: reinforced "
        "(later evidence supports), unchanged, weakened (some doubt), "
        "refuted (later evidence contradicts)."
    )

    _panel_doc(
        inputs="`reports/agent/self_critique.jsonl` (one record per finding × critique run). Generated by `autosignalx agent self-critique`.",
        operations="For each finding in `findings.jsonl`, send `(system: SELF_CRITIQUE_SYSTEM with 4-state rubric) + (user: original finding + ledger summary + summary of other findings)` to the adjudicator-role LLM. Parse JSON `{current_state, rationale}`.",
        goal="Detect findings whose support has weakened over time as new sessions add evidence. Counters confirmation bias by forcing periodic re-evaluation against fresh data.",
        interpretation="`reinforced` = later evidence supports; `unchanged` = no new evidence either way (likely if only one session has run); `weakened` = some related evidence cuts against; `refuted` = subsequent evidence or refute verdicts contradict. Cards are sorted most-recent first; rationales should cite specific later evidence (not generic concerns).",
    )

    from autosignalx.agent import self_critique as sc

    rows = sc.load()
    if not rows:
        st.info(
            "No self-critiques yet. Run `autosignalx agent self-critique` "
            "after enough findings have accumulated."
        )
        return

    state_counts = pd.Series([r.get("current_state", "unknown") for r in rows]).value_counts()
    cols = st.columns(min(4, len(state_counts) + 1))
    cols[0].metric("Total critiques", len(rows))
    for i, (state, cnt) in enumerate(state_counts.items(), start=1):
        if i < len(cols):
            cols[i].metric(state.capitalize(), int(cnt))

    st.divider()
    state_emoji = {"reinforced": "++", "unchanged": "==", "weakened": "--", "refuted": "XX"}
    for r in rows[::-1]:  # most recent first
        st.markdown(
            f"**[{state_emoji.get(r.get('current_state', ''), '?')}] "
            f"{r.get('finding_id', '?')}**  -  *{r.get('current_state', '?')}*"
        )
        st.markdown(f"_{r.get('rationale', '')}_")
        st.caption(f"recorded {r.get('ts', '')}")
        st.divider()


def render_sessions() -> None:
    st.title("Sessions")
    st.caption(
        "Multi-session view. Each row aggregates ledger, findings, "
        "telemetry, and trace-quality records by session_id. Sorted "
        "chronologically (session IDs are YYYYMMDD-prefixed)."
    )

    _panel_doc(
        inputs="All four agent stores (`ledger.jsonl`, `findings.jsonl`, `telemetry.jsonl`, `trace_quality.jsonl`) under `reports/agent/`. `agent/sessions.py` aggregates them by `session_id`.",
        operations="`list_sessions()` collects distinct IDs across stores. `session_summary(sid)` computes per-session aggregates: `n_rounds, n_propose, n_findings, n_refuted, cost_usd, total_tokens, latency_total_ms, avg_clarity, promotion_rate, cost_per_finding`. `productivity_trend()` adds cumulative `cum_findings` and `cum_cost_usd` columns.",
        goal="Long-horizon productivity view. Tracks how many DM-significant findings the agent produces per session, what each finding costs, and how the rate of new findings evolves over time.",
        interpretation="**Cost per finding** is the operational ROI metric. **Promotion rate** = findings / propose count; high rate suggests easy hypotheses or weak gates, low rate suggests stringent gates or hard problem. The cumulative trend chart should ideally show findings growing roughly linearly with cost (constant marginal cost per finding).",
    )

    from autosignalx.agent import sessions as sessions_mod

    df = sessions_mod.all_summaries()
    if df.empty:
        st.info(
            "No sessions yet. Run `make agent` (or schedule daily runs via "
            "`scripts/run_session.sh`) to populate."
        )
        return

    cols = st.columns(4)
    cols[0].metric("Sessions", len(df))
    cols[1].metric("Total findings", int(df["n_findings"].sum()))
    cols[2].metric("Total cost (USD)", f"${df['cost_usd'].sum():.4f}")
    finds = df["n_findings"].sum()
    cost = df["cost_usd"].sum()
    cols[3].metric("Cost per finding", f"${cost / finds:.4f}" if finds > 0 else "n/a")

    st.divider()
    st.subheader("Per-session summary")
    st.dataframe(
        df.style.format(
            {
                "cost_usd": "${:.4f}",
                "total_tokens": "{:,}",
                "latency_total_ms": "{:,.0f}",
                "promotion_rate": "{:.1%}",
                "cost_per_finding": "${:.4f}",
                "avg_clarity": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Productivity trend (cumulative)")
    trend = sessions_mod.productivity_trend()
    if not trend.empty:
        chart_df = trend[["session_id", "cum_findings", "cum_cost_usd"]].set_index("session_id")
        st.line_chart(chart_df, height=260)


def render_telemetry() -> None:
    st.title("Telemetry")
    st.caption(
        "Cost / latency / token usage for live LLM calls. "
        "Cached and replay-mode calls don't generate records. "
        "Cost is estimated from a per-model price table (override via "
        "`DEEPINFRA_PRICE_<MODEL>_IN/_OUT` env vars)."
    )

    _panel_doc(
        inputs="`reports/agent/telemetry.jsonl`. Written by `agent/llm.py:LiveProvider.chat` after every non-cached LLM call.",
        operations="Per-call records carry `(ts, model, role, step, round, prompt_tokens, completion_tokens, total_tokens, latency_ms, cost_usd, session_id)`. Token counts come from `response.response_metadata.token_usage` (or `usage_metadata`); fallback is a character-count estimate (~4 chars per token). Cost = `(prompt / 1M) × in_price + (completion / 1M) × out_price`, with prices from `agent/telemetry.py:DEFAULT_PRICES` overridable via env vars.",
        goal="Operational observability. Make the cost / latency footprint of agent autonomy visible. Cost-per-finding (in the Sessions panel) is the headline metric this panel feeds.",
        interpretation="**Per-model breakdown** shows which agent role (Theorist / Skeptic / Adjudicator / chat / consolidate) consumes most of the budget. **Per-step breakdown** shows whether the cost concentrates in propose-time or critique-time. **Cumulative cost chart** is a single-line view of total spend over the session — should be roughly linear in number of LLM calls. Cached and replay-mode calls don't appear (they're free).",
    )

    from autosignalx.agent import telemetry as telemetry_mod

    rows = telemetry_mod.load()
    if not rows:
        st.info(
            "No telemetry records yet. Live LLM calls (non-cached, "
            "non-replay) automatically record to "
            "`reports/agent/telemetry.jsonl`."
        )
        return

    df = pd.DataFrame(rows)
    cols = st.columns(4)
    cols[0].metric("Total calls", f"{len(df):,}")
    cols[1].metric("Total cost (USD)", f"${df['cost_usd'].sum():.4f}")
    cols[2].metric("Total tokens", f"{int(df['total_tokens'].sum()):,}")
    cols[3].metric("Median latency (ms)", f"{int(df['latency_ms'].median()):,}")

    st.divider()
    st.subheader("Per-model breakdown")
    by_model = (
        df.groupby("model")
        .agg(
            calls=("model", "count"),
            tokens=("total_tokens", "sum"),
            cost=("cost_usd", "sum"),
            latency_p50=("latency_ms", "median"),
            latency_p95=("latency_ms", lambda s: s.quantile(0.95)),
        )
        .sort_values("cost", ascending=False)
    )
    st.dataframe(
        by_model.style.format(
            {
                "tokens": "{:,}",
                "cost": "${:.4f}",
                "latency_p50": "{:,.0f}",
                "latency_p95": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Per-step breakdown")
    by_step = (
        df.groupby("step")
        .agg(
            calls=("step", "count"),
            tokens=("total_tokens", "sum"),
            cost=("cost_usd", "sum"),
        )
        .sort_values("cost", ascending=False)
    )
    st.dataframe(
        by_step.style.format({"tokens": "{:,}", "cost": "${:.4f}"}),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Cost per round (cumulative)")
    df_sorted = df.sort_values("ts").reset_index(drop=True)
    df_sorted["cum_cost"] = df_sorted["cost_usd"].cumsum()
    st.line_chart(df_sorted[["cum_cost"]], height=250)


def render_lessons() -> None:
    st.title("Lessons & Memory")
    st.caption(
        "The long-horizon memory cell. After each session, the agent "
        "consolidates the ledger + promoted findings into a Markdown "
        "'lessons' section that the next session reads as context. "
        "(Run `autosignalx agent consolidate` to update.)"
    )

    _panel_doc(
        inputs="`reports/agent/lessons.md` (Markdown, append-only across sessions, `---` separators between sessions).",
        operations="`agent/memory.py:consolidate(session_id, ledger, findings, provider)` sends `(system: CONSOLIDATOR_SYSTEM with strict 5-section structure under 350 words) + (user: session ID + date + last 40 ledger entries + summary of promoted findings)` to the adjudicator-role LLM. Output is appended to lessons.md by `append_to_lessons`. The next session's `tools.context_snapshot()` includes the most recent ~4000 chars of lessons under the key `prior_sessions_lessons`.",
        goal="Long-horizon memory. Compress each session's raw ledger into a short structured summary the agent can re-consume in subsequent sessions, so cross-session continuity does not require re-stuffing the full ledger into context.",
        interpretation="Each section follows the same 5-block structure: **What was tried** | **What worked** | **What was refuted** | **Patterns observed** | **Open directions for next session**. 'Open directions' becomes natural seeds for the agent's first hypothesis in the next session.",
    )

    from autosignalx.agent import memory as memory_mod

    text = memory_mod.load_lessons(max_chars=20000)
    if not text:
        st.info(
            "No lessons recorded yet. After a session, run "
            "`autosignalx agent consolidate` to summarize and persist."
        )
        return
    st.markdown(text)


def render_findings() -> None:
    st.title("Findings")
    st.caption(
        "Promoted findings -- hypotheses that passed the statistical "
        "promotion gate (Diebold-Mariano p < 0.05 AND positive bootstrap CI). "
        "Each card carries the full evidence trail."
    )

    _panel_doc(
        inputs="`reports/agent/findings.jsonl`. Each record: `(id, hypothesis, method, filters, evidence, agent_confidence, round, session_id, promoted_at, parent_hypothesis_ids, replication_count, replications)`.",
        operations="Findings are produced by the auto-promotion path in `agent/graph.py:experiment_node`: every non-naive method's experiment slice is run through `eval/significance.py:is_promotable(method, baseline='naive', asset, regime_id, p_threshold=0.05)`. A method passes when **all three** of (DM p < 0.05, skill > 0, bootstrap CI strictly above zero). Persisting is idempotent on hypothesis content — re-promotion bumps `replication_count` instead of duplicating.",
        goal="Surface the agent's discoveries that meet the statistical bar. Findings are the system's research output; everything else (ledger, telemetry, etc.) is supporting infrastructure.",
        interpretation="Sort order is descending by skill-vs-naive (largest improvements first). Per-card evidence: `n` (sample size), `method_mae` vs `baseline_mae` (lower better), `skill_vs_baseline` (positive = better than naive), `dm_statistic` and `p_value` (test of equal predictive accuracy), `bootstrap_ci_low` / `_high` (loss difference distribution). `replication_count > 1` means the same finding has been independently re-discovered across sessions — strongest signal of robustness.",
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

    _panel_doc(
        inputs="`reports/agent/ledger.jsonl` (propose / theorist entries) + `reports/agent/findings.jsonl` (promoted findings with parent IDs).",
        operations="`agent/lineage.py:build_lineage` walks the ledger, dedupes propose / theorist entries by content hash (`h_<hash10>` derived from `hypothesis text + experiment params`), and infers parent edges by **method/asset/regime overlap** with prior rounds (within `parent_lookback`). Status assignment: `promoted` = matches a finding's round-of-promotion or appears in `parent_hypothesis_ids`; `refuted` = same-round adjudicator content contains `VERDICT: refute`; `open` otherwise. Plotly graph layout: x-axis = round number, vertical jitter for collision avoidance.",
        goal="Trace any promoted finding back to its initial brainstorm and see the chain of refinements / refutations that led to it.",
        interpretation="Green nodes = promoted (finding with that hypothesis was DM-significant). Red nodes = refuted by the adjudicator. Gray = open / inconclusive. Edges point parent → child (predecessor with overlapping params → current refinement). Hover over a node for full hypothesis preview + experiment params. The tabular view above the graph lists every node with its `parents` column (or `(root)` for orphans).",
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


def render_backtest_arena() -> None:
    st.title("Backtest Arena")
    st.write(
        "Simulated trading on the test window using strategies driven by the "
        "system's discoveries. Strict no-look-ahead: backtest start is "
        "after the discovery window end (2020-12-31)."
    )
    _panel_doc(
        inputs="Reads `data/cache/ohlcv.parquet` for adjusted close prices and "
        "`reports/backtest/runs/<run_id>/{portfolio_daily,trades,metrics,meta}` "
        "for completed runs. Strategy `BuyAndHoldSPY` and `EqualWeightUniverse` "
        "(passive) need only prices; signal-driven strategies (Phase 1.2+) "
        "additionally consume `reports/ablations/*.parquet` and "
        "`reports/agent/findings.jsonl`.",
        operations="Vectorized portfolio engine: weights set at close(t) earn "
        "the close(t)->close(t+1) return (one-bar shift prevents look-ahead). "
        "Per-bar turnover charges a configurable bps cost. Metrics: CAGR, "
        "annualised vol, Sharpe, Sortino, max drawdown, Calmar, hit rate, "
        "average turnover, total cost drag.",
        goal="Translate the abstract research output (forecasts, regimes, "
        "agent-promoted findings) into concrete simulated trading performance "
        "to test whether the discoveries are economically actionable, not just "
        "statistically significant.",
        interpretation="A strategy beats the SPY benchmark if its Sharpe is "
        "higher *net of costs*. Negative Sharpe means the strategy lost money "
        "on a risk-adjusted basis. Drawdown depth and Calmar ratio matter as "
        "much as headline returns. Phase 1.1 ships the two passive "
        "benchmarks; signal-driven strategies arrive in subsequent "
        "sub-iterations.",
    )

    active_study = st.session_state.get("active_study")
    if active_study:
        from autosignalx.study import Study

        try:
            runs_dir = Study.load(active_study).backtest_runs_dir
            st.info(f"Reading from study scope: **{active_study}**")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not load study {active_study!r}: {e}")
            return
    else:
        runs_dir = settings.reports_dir / "backtest" / "runs"

    if not runs_dir.exists():
        scope = f"(study={active_study})" if active_study else ""
        st.info(f"No backtest runs yet {scope}. Run `autosignalx backtest run`.")
        return
    runs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], reverse=True)
    if not runs:
        st.info("No backtest runs yet. Run `autosignalx backtest run`.")
        return

    selected = st.selectbox("Run", [r.name for r in runs])
    run_dir = runs_dir / selected

    metrics_path = run_dir / "metrics.json"
    portfolio_path = run_dir / "portfolio_daily.parquet"
    meta_path = run_dir / "meta.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        col1, col2, col3 = st.columns(3)
        col1.metric("Strategies", len(meta.get("config", {}).get("strategies", [])))
        col2.metric("Period (bars)", meta.get("n_periods", 0))
        col3.metric("Cost (bps)", meta.get("config", {}).get("cost_bps", 0.0))
        with st.expander("Run config"):
            st.json(meta)

    if not metrics_path.exists() or not portfolio_path.exists():
        st.warning("Run is missing artifacts.")
        return

    metrics = json.loads(metrics_path.read_text())
    sig_payload = metrics.pop("__significance__", {})
    rows = []
    for name, m in metrics.items():
        # Strip nested per_regime block before flattening for the table.
        flat = {k: v for k, v in m.items() if k != "per_regime"}
        rows.append({"strategy": name, **flat})
    metrics_df = pd.DataFrame(rows)
    pretty_cols = {
        "strategy": "Strategy",
        "total_return": "Total Return",
        "cagr": "CAGR",
        "annual_vol": "Annual Vol",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "max_drawdown": "Max DD",
        "calmar": "Calmar",
        "hit_rate": "Hit Rate",
        "avg_turnover": "Turnover",
        "cost_drag": "Cost Drag",
    }
    metrics_view = metrics_df.rename(columns=pretty_cols)
    st.subheader("Per-strategy metrics")
    st.dataframe(metrics_view, hide_index=True, use_container_width=True)

    portfolio = pd.read_parquet(portfolio_path)
    portfolio["timestamp"] = pd.to_datetime(portfolio["timestamp"])

    st.subheader("Equity curves")
    pivot = portfolio.pivot(index="timestamp", columns="strategy", values="equity")
    st.line_chart(pivot)

    st.subheader("Drawdown")
    eq = pivot
    peak = eq.cummax()
    drawdown = eq / peak - 1.0
    st.area_chart(drawdown)

    st.caption(
        "Equity normalised to 1.0 at run start; drawdown is the fraction "
        "below the running peak. All series include realised costs."
    )

    if sig_payload:
        st.subheader("Sharpe-difference significance")
        bench_strat = next(iter(sig_payload.values())).get("benchmark", "benchmark")
        st.caption(
            f"Paired moving-block bootstrap of Sharpe(strategy) - Sharpe({bench_strat}). "
            f"'Significant' = 95% CI excludes 0."
        )
        sig_rows = []
        for name, s in sig_payload.items():
            sig_rows.append({
                "Strategy": name,
                "Sharpe diff": s.get("sharpe_diff", 0.0),
                "CI low": s.get("ci_low", 0.0),
                "CI high": s.get("ci_high", 0.0),
                "p-value": s.get("p_value", 1.0),
                "Significant": "yes" if s.get("significant") else "no",
            })
        st.dataframe(pd.DataFrame(sig_rows), hide_index=True, use_container_width=True)

    # Per-regime breakdown
    per_regime_blocks = {
        name: m["per_regime"] for name, m in metrics.items() if "per_regime" in m
    }
    if per_regime_blocks:
        st.subheader("Per-regime metrics")
        st.caption(
            "Sharpe / CAGR conditional on regime ID; the equity curve "
            "is recompounded over each regime's bars in isolation."
        )
        chosen = st.selectbox("Strategy", list(per_regime_blocks.keys()))
        block = per_regime_blocks[chosen]
        regime_rows = []
        for r, m in block.items():
            regime_rows.append({
                "Regime": r,
                "N bars": m.get("n_periods", 0),
                "CAGR": m.get("cagr", 0.0),
                "Vol": m.get("annual_vol", 0.0),
                "Sharpe": m.get("sharpe", 0.0),
                "Max DD": m.get("max_drawdown", 0.0),
                "Hit Rate": m.get("hit_rate", 0.0),
            })
        st.dataframe(pd.DataFrame(regime_rows), hide_index=True, use_container_width=True)


def render_custom_study() -> None:
    st.title("Custom Study")
    st.write(
        "Run AutoSignal-X on your own asset universe and date range. "
        "Each study has its own data cache and reports tree under "
        "`data/studies/<name>/` and `reports/studies/<name>/` so multiple "
        "studies coexist without collision."
    )
    st.caption(
        "Supported local launch: `uv sync --all-extras` once, then "
        "`uv run streamlit run app/streamlit_app.py` from the repo root."
    )
    _panel_doc(
        inputs="Reads `data/studies/<name>/study.yaml` for each study's "
        "config; the pipeline buttons read `data/studies/<name>/cache/` and "
        "write to `reports/studies/<name>/{ablations,backtest/runs}/`.",
        operations="Form-based create/validate; pipeline buttons call the "
        "pure-Python entry points in `study.pipeline` (data fetch -> "
        "yfinance; baseline eval -> walk-forward naive/seasonal_naive/arima; "
        "backtest -> the same vectorized engine as the default flow). "
        "Heavy steps (Chronos-2 forecasting, agent runs) surface the CLI "
        "command for the user to run from the terminal where logs stream.",
        goal="Make AutoSignal-X usable on user-relevant data without "
        "editing config files or touching the codebase.",
        interpretation="The Pipeline status table shows which artifacts "
        "exist for each study. The backtest run id at the bottom is the "
        "newest output; switch the Sidebar 'Study scope' to that study to "
        "make Backtest Arena and Forecast Arena read from it.",
    )

    from autosignalx.study import (
        Study,
        StudyExistsError,
        StudyNotFoundError,
        list_studies,
        validation,
    )
    from autosignalx.study import pipeline as pipe

    # ---- create new study form ------------------------------------------
    with st.expander("Create new study", expanded=not list_studies()):
        with st.form("create_study"):
            col_a, col_b = st.columns(2)
            with col_a:
                name = st.text_input(
                    "Name", help="Alphanumeric, _, - only; not 'default'."
                )
                assets_text = st.text_area(
                    "Assets (one per line or comma-separated)",
                    value="SPY\nQQQ\nIWM\nGLD\nTLT",
                    height=120,
                )
                macro_text = st.text_area(
                    "Macro covariates",
                    value="^TNX\n^VIX\nDX-Y.NYB\nCL=F",
                    height=80,
                )
                description = st.text_input("Description (optional)")
            with col_b:
                start_date = st.text_input("Start date (YYYY-MM-DD)", value="2015-01-01")
                end_date = st.text_input("End date (YYYY-MM-DD)", value="2024-12-31")
                train_end = st.text_input("Train end", value="2021-12-31")
                val_end = st.text_input("Val end", value="2022-12-31")
                test_end = st.text_input("Test end", value="2024-12-31")
                horizon = st.number_input("Forecast horizon (bars)", min_value=1, value=21)
                step = st.number_input("Walk-forward step (bars)", min_value=1, value=21)
                n_regimes = st.number_input("Number of regimes", min_value=2, max_value=10, value=4)
                cost_bps = st.number_input("Cost (bps)", min_value=0.0, value=5.0)
                overwrite = st.checkbox("Overwrite if exists", value=False)
            submitted = st.form_submit_button("Create study")
        if submitted:
            try:
                tickers = [
                    t.strip() for chunk in assets_text.replace(",", "\n").splitlines()
                    for t in [chunk.strip()] if t
                ]
                macro_tickers = [
                    t.strip() for chunk in macro_text.replace(",", "\n").splitlines()
                    for t in [chunk.strip()] if t
                ]
                s = Study(
                    name=name, description=description,
                    assets=tickers, macro=macro_tickers,
                    start_date=start_date, end_date=end_date,
                    train_end=train_end, val_end=val_end, test_end=test_end,
                    forecast_horizon_days=int(horizon),
                    rolling_step_days=int(step),
                    n_regimes=int(n_regimes),
                    cost_bps=float(cost_bps),
                )
                s.save(overwrite=overwrite)
                st.success(f"Created study `{name}` at {s.config_path}")
            except StudyExistsError as e:
                st.error(str(e))
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not create study: {e}")

    # ---- existing studies ------------------------------------------------
    studies = list_studies()
    if not studies:
        st.info(
            "No studies yet. Create one above, or from the CLI: "
            "`autosignalx study create --name X --assets ... --start ... --end ...`"
        )
        return

    chosen = st.selectbox("Study", studies)
    try:
        study = Study.load(chosen)
    except StudyNotFoundError as e:
        st.error(str(e))
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Assets", len(study.assets))
    col2.metric("Macro", len(study.macro))
    col3.metric("Span", f"{study.start_date[:4]}-{study.end_date[:4]}")
    col4.metric("Cost (bps)", study.cost_bps)
    with st.expander("Full config"):
        st.json(study.model_dump())

    # ---- validation ------------------------------------------------------
    st.subheader("Pre-flight validation")
    check_tickers = st.checkbox("Also probe yfinance for ticker availability (network call)")
    if st.button("Run validation"):
        report = validation.validate(study, check_tickers=check_tickers)
        for msg in report.info:
            st.info(msg)
        for msg in report.warnings:
            st.warning(msg)
        for msg in report.errors:
            st.error(msg)
        if report.ok:
            st.success("Validation OK")

    # ---- pipeline status -------------------------------------------------
    st.subheader("Pipeline status")
    status = pipe.pipeline_status(study)
    status_rows = [
        {"Step": "Data cache (OHLCV)", "Status": "ok" if status["ohlcv"] else "missing"},
        {"Step": "Data cache (macro)", "Status": "ok" if status["macro"] else "missing"},
        {"Step": "Baseline ablation", "Status": "ok" if status["baseline"] else "missing"},
        {"Step": "Chronos-2 ablation", "Status": "ok" if status["chronos"] else "missing"},
        {"Step": "Backtest runs", "Status": f"{status['n_backtest_runs']} run(s)"
                                              + (f" (latest: {status['latest_run']})"
                                                 if status['latest_run'] else "")},
    ]
    st.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)

    # ---- pipeline actions ------------------------------------------------
    st.subheader("Run pipeline (synchronous)")
    cols = st.columns(3)
    with cols[0]:
        if st.button("1. Fetch data"):
            with st.spinner(f"Pulling {len(study.assets)} assets + "
                            f"{len(study.macro)} macro from yfinance..."):
                try:
                    out = pipe.run_data_fetch(study)
                    st.success(
                        f"Wrote {out['ohlcv_rows']:,} OHLCV rows + "
                        f"{out['macro_rows']:,} macro rows"
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Fetch failed: {e}")
    with cols[1]:
        if st.button("2. Run baseline eval"):
            with st.spinner("Running naive + seasonal_naive + arima ablation..."):
                try:
                    out = pipe.run_baseline_eval(study)
                    st.success(
                        f"Wrote {out['rows']:,} forecast rows across "
                        f"{out['windows']} windows -> {out['out_path']}"
                    )
                except FileNotFoundError as e:
                    st.error(f"Baseline eval failed: {e}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Baseline eval failed: {e}")
    with cols[2]:
        if st.button("3. Run backtest"):
            with st.spinner("Running backtest..."):
                try:
                    out = pipe.run_backtest_for_study(study)
                    st.success(
                        f"Run {out['run_id']} ({out['n_strategies']} strategies) "
                        f"-> {out['artifacts_dir']}"
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Backtest failed: {e}")

    st.caption(
        "Heavy steps (Chronos-2, agent run) are best launched from the CLI "
        f"so logs stream to the terminal: \n\n"
        f"```\n"
        f"autosignalx eval chronos --study {study.name}\n"
        f"autosignalx agent run --max-rounds 5\n"
        f"```"
    )

    st.caption(
        "Tip: pick this study in the sidebar **Study scope** selector to "
        "make Backtest Arena and Forecast Arena read its artifacts."
    )


PANELS = {
    "Overview": render_overview,
    "Data": render_data,
    "Forecast Arena": render_forecast_arena,
    "Regime Explorer": render_regime_explorer,
    "Signal Discovery Lab": render_signal_lab,
    "Cross-Asset Graph": render_cross_asset_graph,
    "Backtest Arena": render_backtest_arena,
    "Custom Study": render_custom_study,
    "Agent Console": render_agent_console,
    "Auto-Play Replay": render_auto_play,
    "Findings": render_findings,
    "Lineage": render_lineage,
    "Self-Critique": render_self_critique,
    "Lessons & Memory": render_lessons,
    "Telemetry": render_telemetry,
    "Sessions": render_sessions,
    "Ask the Memory": render_ask_the_memory,
}


# Streamlit executes the script top-to-bottom on every interaction.
panel_name = st.sidebar.radio("Panel", list(PANELS.keys()))
st.sidebar.divider()

# Phase 2: study-scope selector. When set, study-aware panels (Forecast
# Arena, Backtest Arena) read from that study's reports tree instead of
# the project default. Stored in session_state so panels can look it up.
try:
    from autosignalx.study import list_studies as _list_studies

    _study_names = _list_studies()
except Exception:  # noqa: BLE001
    _study_names = []
_scope_options = ["(default)"] + _study_names
_scope_choice = st.sidebar.selectbox(
    "Study scope",
    _scope_options,
    index=0,
    help="Study whose artifacts the panels read. '(default)' uses the project tree.",
)
st.session_state["active_study"] = (
    None if _scope_choice == "(default)" else _scope_choice
)

st.sidebar.caption(f"AutoSignal-X v{__version__}")
PANELS[panel_name]()
