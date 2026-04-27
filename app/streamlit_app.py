"""AutoSignal-X — Streamlit research cockpit.

31 panels in the sidebar (5 model layers + agent loop + Phase-1 backtest +
Phase-2 custom studies + Phase-3 grounded chat + Phases 7/8/12/14/15/16
research-lab observability). Each panel is a read-only viewer over a typed
artifact written by one of the system's layers; expensive computation
lives in CLI commands, the cockpit only visualises persisted state. Every
panel includes a standardized 'About this panel' expander documenting its
inputs, operations / algorithms, goal, and how to interpret the results."""

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
        ("L1 Forecasting", "Probabilistic point + interval forecasts", "Chronos-2 (multivariate, with covariates) + classical baselines (naive, seasonal-naive, ARIMA(1,1,1) on log-prices). Phase 7 adds returns-target baselines (zero_return / mean_return / momentum) and a `target_type ∈ {price, log_return, excess_return, vol, rank}` contract column."),
        ("L2 Representation", "Per-timestep latent regime labels", "Contrastive 1D-CNN encoder (16-dim, 60-day windows, triplet loss) + KMeans on embeddings; Gaussian HMM as parallel detector."),
        ("L3 Reasoning", "Per-regime feature importance", "HistGradientBoostingClassifier per regime + custom permutation importance. Phase 6 walk-forward stability layer (per-(regime, feature) mean rank, rank std, top-K share)."),
        ("L4 Relational", "Cross-asset dependency structure", "GLASSO partial correlations + Granger causality + NetworkX centrality. Phase 6 regime-conditioned variant rebuilds the graph per regime + cross-regime sensitivity."),
        ("L5 Agentic", "Hypothesis generation, experimentation, statistical promotion", "LangGraph state machine in three modes: `single` (one LLM), `debate` (Theorist/Skeptic/Adjudicator), `lab` (Phase 14: planner → specialist consult → KG writer + 11 specialist roles). Three experiment surfaces: cached-slice, JSON DSL, sandboxed Python."),
    ]
    for name, purpose, impl in layer_rows:
        st.markdown(f"- **{name}** — *{purpose}.*  {impl}")

    st.markdown(
        "**Statistical methodology.** Promotion gate: Diebold-Mariano (Newey-West HAC) "
        "+ block-bootstrap CI. Hardening (Phase 5): BH-FDR + adversarial replication "
        "(full-test / placebo / block-holdout). Selection-bias-aware (Phase 8): "
        "Combinatorial Purged CV, Probability of Backtest Overfitting, Deflated Sharpe, "
        "Romano-Wolf stepdown, hash-committed pre-registration ledger, never-touched "
        "holdout vault. Bayesian (Phase 12): hierarchical Normal-Normal model with "
        "empirical-Bayes shrinkage, Bayes factors, posterior-predictive checks. The "
        "strict survival bar (`survives_all_strict`) is the conjunction of every gate."
    )

    st.divider()
    st.subheader("Cockpit panels (31, in sidebar order)")
    panel_rows = [
        ("Overview", "This page. System pitch, model layers, statistical methodology, panel index, inputs/outputs."),
        ("Data", "Cache inventory; ETF and macro time series. Reads `data/cache/*.parquet`."),
        ("Forecast Arena", "Per-method overall metrics; per-(method, regime) stratified metrics; per-asset trajectory chart with 80% interval bands. Reads `reports/ablations/*.parquet`."),
        ("Regime Explorer", "KMeans + HMM regime timelines; PCA-2D scatter of contrastive embeddings colored by regime. Reads `reports/regimes/*.parquet`."),
        ("Signal Discovery Lab", "Per-regime feature importance bar chart; ranking table; cross-regime importance heatmap. Reads `reports/signals/signal_ranking.parquet`."),
        ("Cross-Asset Graph", "Centrality table; partial-correlation matrix; top Granger edges. Reads `reports/graph/{edges,centrality}.parquet`."),
        ("Regime-Conditioned Graph", "Phase 6: same machinery rebuilt per regime; surfaces hubs and bridges that flip role across regimes; per-asset regime-sensitivity table."),
        ("Signal Stability", "Phase 6: walk-forward feature-importance rankings; per-(regime, feature) stability metrics; rank-trajectory chart across windows."),
        ("Backtest Arena", "Phase 1: simulated trading on the test window driven by discovered structure. Equity curves, drawdowns, Sharpe/Sortino/Calmar; paired bootstrap CI on Sharpe-difference vs benchmark; strict no-look-ahead."),
        ("Custom Study", "Phase 2: form-based per-study workspace (universe / dates / splits). Pre-flight validation + pipeline buttons. Each study has its own `data/studies/<name>/` and `reports/studies/<name>/` tree."),
        ("Coverage Map", "Phase 14/16: 4D heatmap of (method × asset × regime) coloured by EIG (expected information gain). Reviewer sees exactly where the agent has hunted, what's promoted, what's open."),
        ("Statistical Power", "Phase 16: per-cell Cohen's d, observed power at α=0.05, sample-size required for 80% power. Distinguishes underpowered failures from genuine nulls."),
        ("Counterfactual Cards", "Phase 16: per-finding factor residualization (against macro factors), what-if perturbations across prediction-magnitude buckets, outlier-removal stability. Makes the *reasoning* behind each finding interrogable."),
        ("Bayesian Evidence", "Phase 12: hierarchical Normal-Normal posterior over each finding's true skill. Reports posterior mean / sd, P(θ>0), Bayes factor BF_10 vs the null, posterior predictive intervals."),
        ("Specialist Council", "Phase 14: multi-role consultation feed (Statistician / Quant / RiskOfficer / Economist / Implementer / RedTeam / Historian); PrincipalInvestigator routing decisions; persistent KG explorer."),
        ("Pre-Registration", "Phase 8: hash-committed hypothesis ledger. Open / resolved (promoted / refuted) counts. Makes the agent's commitment-before-evidence auditable."),
        ("Holdout Vault", "Phase 8: never-touched final test slice. Locked / opened state, leakage assertions, one-time evaluation results."),
        ("Agent Calibration", "Phase 15: reliability diagram of Theorist confidence vs finding-survival rate. Brier score and Expected Calibration Error."),
        ("RedTeam Attacks", "Phase 15: per-finding asset-shuffle (re-test on every other asset in the same regime) + time-shift attacks (shift forecast_origin by 5 days)."),
        ("Agent Coherence", "Phase 15: per-session lessons-uptake, lineage branching factor, theme-persistence entropy, composite coherence score across sessions."),
        ("Agent Console", "Chat-style ledger timeline; per-round trace-quality chart at the bottom. Reads `reports/agent/ledger.jsonl`."),
        ("Auto-Play Replay", "Playback controls (play / pause / reset, 0.5x-4x speed) over the ledger."),
        ("Findings", "Promoted findings (passed DM + bootstrap gate) sorted by skill-vs-naive; full statistical evidence per card."),
        ("Lineage", "Plotly DAG of hypothesis evolution; nodes colored by status (promoted / refuted / open). Inferred via `agent/lineage.py`."),
        ("Self-Critique", "Agent's verdicts on its own past findings against current evidence."),
        ("Survival Analysis", "Phase 5/8/12 hardening grid: BH-FDR + adversarial (full-test / placebo / block-holdout) + Romano-Wolf + Deflated Sharpe + CPCV + Bayesian. The strict bar `survives_all_strict` is the conjunction."),
        ("Lessons & Memory", "Accumulating Markdown of consolidated session notes (long-horizon memory)."),
        ("Telemetry", "Cost / tokens / latency per LLM call; per-model and per-step breakdown; cumulative cost."),
        ("Sessions", "Per-session productivity (rounds, findings, cost-per-finding); cumulative trend. Aggregates by `session_id`."),
        ("Reproducibility", "Phase 16: git hash + dirty flag, Python env, library versions, replay-mode flag, per-artifact SHA-256, single bundle hash for the current cockpit state."),
        ("Ask the Memory", "Phase 3: grounded RAG chat over the run corpus (ledger / findings / lessons / trace_quality / self_critique / telemetry / backtests / regime-graph / signal-stability / survival). Cite-or-refuse system prompt; off-corpus questions trigger refusal."),
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
            - **Optional study workspace** (`data/studies/<name>/study.yaml`): user-defined universe + date range + walk-forward boundaries (Phase 2).
            """
        )
    with cols[1]:
        st.markdown(
            """
            **Outputs (all under `reports/`)**
            - `ablations/*.parquet` — per-method walk-forward forecasts (with optional `target_type` from Phase 7).
            - `regimes/*.parquet` — KMeans + HMM regime labels and contrastive embeddings.
            - `signals/{signal_ranking,walk_forward_ranking,signal_stability}.parquet` — per-regime feature importance + walk-forward stability summary (Phase 6).
            - `graph/{edges,centrality}.parquet` + `graph/per_regime/` — global + regime-conditioned cross-asset graph (Phase 6).
            - `backtest/runs/<run_id>/` — strategy P&L, trades, metrics, paired bootstrap (Phase 1).
            - `agent/{ledger,findings,telemetry,trace_quality,self_critique,survival}.jsonl` — agent state, evidence, hardening output.
            - `agent/{preregistrations,preregistration_resolutions,red_team,calibration,coherence}.jsonl` — Phase-8 / Phase-15 artifacts.
            - `agent/kg/{nodes,edges}.jsonl` — Phase-14 persistent knowledge graph.
            - `agent/holdout_vault/{vault,results}.json` — Phase-8 final test slice.
            - `agent/eval_summary.json`, `pbo.json`, `reproducibility_badge.json` — Phase-15 / Phase-16 rollups.
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


def render_regime_graph() -> None:
    st.title("Regime-Conditioned Graph")
    st.caption(
        "Cross-asset structure recomputed within each regime's data subset. "
        "Surfaces hubs that only matter in one regime, bridge assets that flip "
        "role across regimes, and clusters that fragment under stress -- "
        "structural information the global graph averages away."
    )

    _panel_doc(
        inputs="`reports/graph/per_regime/regime_<id>/{edges,centrality}.parquet` plus `regime_sensitivity.parquet`. Built by `autosignalx graph build-per-regime`. Requires both the kmeans regime artifact and the OHLCV cache.",
        operations="For each regime's timestep subset: (1) GLASSO partial-correlation edges (`graph.correlation`); (2) Granger causality edges (`graph.causality`); (3) NetworkX centrality on the partial-correlation graph (`graph.centrality`). Then aggregate per-asset centrality across regimes into `regime_sensitivity.parquet` -- the dispersion (max - min betweenness) ranks how *regime-sensitive* each asset's structural role is.",
        goal="Expose regime-conditional structural changes that the global graph hides. The strongest research signal here is when a hub or bridge asset *only* plays that role in a specific regime -- evidence that the regime label captures real structural state, not just a noise cluster.",
        interpretation="Compare the per-regime hub/bridge tables: an asset that is a bridge in every regime (e.g., TLT often is) plays a stable role; an asset whose betweenness ranges from ~0 to ~0.6 (look at the sensitivity table) is regime-conditional. Both are valuable signals -- the first for stable cross-asset risk modelling, the second for regime-aware allocation.",
    )

    from autosignalx.graph import per_regime as pr

    loaded = pr.load_per_regime()
    if not loaded:
        st.warning(
            "No per-regime graph artifacts. Run `autosignalx graph build-per-regime`."
        )
        return

    # Headline summary
    cols = st.columns(len(loaded))
    for col, (rid, payload) in zip(cols, sorted(loaded.items()), strict=False):
        cent = payload["centrality"].sort_values("eigenvector_centrality", ascending=False)
        bridge = payload["centrality"].sort_values("betweenness_centrality", ascending=False)
        with col:
            st.markdown(f"**Regime {rid}**")
            st.caption(f"n={int(cent['n_samples'].iloc[0]) if 'n_samples' in cent.columns else '?'} samples")
            st.metric("Top hub (eig.)", str(cent.iloc[0]["node"]) if not cent.empty else "?")
            st.metric("Top bridge (betw.)", str(bridge.iloc[0]["node"]) if not bridge.empty else "?")
            st.metric("Edges", len(payload["edges"]))

    st.divider()
    st.subheader("Regime-sensitivity ranking")
    sens = pr.load_regime_sensitivity()
    if not sens.empty:
        st.caption(
            "Assets ranked by how much their betweenness centrality varies across regimes. "
            "Larger range = role flips more dramatically with regime."
        )
        display_cols = [
            c for c in [
                "node", "betweenness_centrality_min", "betweenness_centrality_max",
                "betweenness_centrality_range", "eigenvector_centrality_mean",
                "degree_centrality_mean",
            ] if c in sens.columns
        ]
        st.dataframe(sens[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No sensitivity summary available (single-regime data?).")

    st.divider()
    st.subheader("Per-regime centrality detail")
    for rid, payload in sorted(loaded.items()):
        with st.expander(f"Regime {rid}"):
            st.dataframe(
                payload["centrality"].sort_values("eigenvector_centrality", ascending=False),
                use_container_width=True, hide_index=True,
            )


def render_signal_stability() -> None:
    st.title("Signal Stability")
    st.caption(
        "Walk-forward feature-importance rankings across rolling windows. "
        "Distinguishes features whose high importance is *stable* from those "
        "whose ranking is an averaging artefact of one favourable sub-period."
    )

    _panel_doc(
        inputs="`reports/signals/walk_forward_ranking.parquet` (per-window-per-regime importance + rank) and `reports/signals/signal_stability.parquet` (per-(regime, feature) summary). Built by `autosignalx signal stability`.",
        operations="Slide N walk-forward windows across the timeline, refit the per-regime HistGradientBoosting + permutation importance ranker inside each, persist per-window rankings. Aggregate to per-(regime, feature) metrics: mean importance, mean rank, rank std, top-K share, and a composite stability = 1 - (rank_std / max_rank).",
        goal="Cross-validate the per-regime signal ranking. A feature with high mean importance AND high stability AND high top-K share is research-grade; high importance + low stability flags an averaging artefact; high stability with low importance flags a consistently-mediocre feature.",
        interpretation="Stability ∈ [0,1]; >0.85 is robust. Top-K share = fraction of windows the feature was in the top-K within its regime. The combination 'mean importance > 0.05 AND stability > 0.85 AND top-5 share > 0.75' is a useful research-grade gate.",
    )

    from autosignalx.signal import stability as stab

    summary = stab.load_stability()
    wf = stab.load_walk_forward()
    if summary.empty:
        st.warning("No stability summary. Run `autosignalx signal stability`.")
        return

    regimes = sorted(summary["regime_id"].unique())
    selected = st.selectbox("Regime", regimes)
    sub = summary[summary["regime_id"] == selected].sort_values("mean_importance", ascending=False)

    st.subheader(f"Regime {selected} -- top features")
    cols = [c for c in [
        "feature", "mean_importance", "mean_rank", "rank_std", "stability",
    ] + [c for c in sub.columns if c.startswith("top")] for c in [c]]
    cols = [c for c in cols if c in sub.columns]
    st.dataframe(sub[cols].head(15), use_container_width=True, hide_index=True)

    if not wf.empty:
        wf_sub = wf[wf["regime_id"] == selected]
        try:
            import plotly.express as px

            top_features = sub.head(8)["feature"].tolist()
            wf_top = wf_sub[wf_sub["feature"].isin(top_features)]
            if not wf_top.empty:
                fig = px.line(
                    wf_top.sort_values("window_idx"),
                    x="window_idx", y="rank", color="feature",
                    markers=True, title=f"Rank trajectory across windows (regime {selected})",
                    template="plotly_white",
                )
                fig.update_yaxes(autorange="reversed", title="Rank (1 = best)")
                fig.update_xaxes(title="Walk-forward window index")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.info(f"Could not render rank trajectory: {e}")


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
        "Grounded RAG chat over the run corpus (ledger, findings, lessons, "
        "trace quality, self-critique, telemetry, backtests). Every claim "
        "carries a citation; off-corpus questions are refused."
    )

    _panel_doc(
        inputs="`reports/chat/chunks.jsonl` + `vectors.npy` (built by `autosignalx chat index`). User question via `st.chat_input`.",
        operations="Embed the question (DeepInfra `bge-large-en-v1.5` in live mode; deterministic hashed-bag in replay/no-key mode). Top-K cosine retrieval over the index. Live mode: send retrieved chunks + cite-or-refuse system prompt to the chat-role LLM. Replay mode: render the top-K chunks with their citation IDs (no LLM call).",
        goal="Let reviewers query AutoSignal-X's discoveries in natural language with verifiable citations back to the underlying artifacts.",
        interpretation="The answer is followed by a citation chip row (e.g. `finding:f_9395cd1bd1be`, `ledger:r3/skeptic`, `backtest:<run_id>/TopKLong`). When evidence is absent the assistant refuses rather than hallucinating.",
    )

    from autosignalx.chat import answer as answer_mod
    from autosignalx.chat import index as index_mod
    from autosignalx.config import settings

    idx = index_mod.load_index()
    if idx is None or not idx.chunks:
        st.warning(
            "Chat index not built. Run `autosignalx chat index` (or click below)."
        )
        if st.button("Build chat index now"):
            with st.spinner("Indexing artifacts..."):
                idx = index_mod.build_index()
            st.success(f"Indexed {len(idx.chunks)} chunks (mode={idx.mode}).")
            st.rerun()
        return

    cols = st.columns([3, 1])
    cols[0].caption(
        f"Index: {len(idx.chunks)} chunks · mode={idx.mode} · model={idx.model}"
    )
    if cols[1].button("Rebuild index"):
        with st.spinner("Re-indexing..."):
            index_mod.build_index()
        st.success("Rebuilt.")
        st.rerun()

    if "memory_history" not in st.session_state:
        st.session_state.memory_history = []

    for q, a, cites in st.session_state.memory_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
            if cites:
                st.markdown(
                    " ".join(f"`{c}`" for c in cites),
                    help="Citation IDs reference on-disk artifacts.",
                )

    question = st.chat_input("Ask the agent's memory...")
    if not question:
        return

    spinner_text = (
        "Retrieving + asking LLM..."
        if (not settings.use_replay and settings.deepinfra_api_key)
        else "Retrieving (replay mode, no LLM call)..."
    )
    with st.spinner(spinner_text):
        result = answer_mod.answer_question(question, index=idx)
    st.session_state.memory_history.append((question, result.text, result.citations))
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


def render_survival_analysis() -> None:
    st.title("Survival Analysis")
    st.caption(
        "Every promoted finding re-evaluated under three layered attacks: "
        "Benjamini–Hochberg FDR correction (across all promoted findings), "
        "full-test replication (drops the 8-window cap), placebo "
        "regime-label shuffle, and 50/50 block holdout. A finding "
        "'survives' only if it passes every attack independently."
    )

    _panel_doc(
        inputs="`reports/agent/survival.jsonl` (one record per finding) generated by `autosignalx agent harden`. Joins findings.jsonl with the union of ablation parquets + the kmeans regime artifact.",
        operations=(
            "For each finding: (1) BH-FDR step-up over the family of original p-values at α=0.10. "
            "(2) Full-test: re-run the gate on the entire ablation slice, ignoring agent's max-windows cap. "
            "(3) Placebo: shuffle regime labels uniformly, re-run the gate; finding 'survives placebo' iff the shuffled gate fails. "
            "(4) Block-holdout: split the slice 50/50 by forecast_origin time and require both halves to pass independently."
        ),
        goal="Expose findings whose original promotion was driven by multiple-comparison luck, single-window overfit, regime-label artefacts, or sub-period concentration. Honest reporting beats a fragile headline.",
        interpretation="A green check in every column means the finding is robust to all four attacks — strong evidence the discovered structure is real. A red X in any column is a *research insight*, not a failure: it tells you precisely how the original gate over-promoted. Block-holdout failures are especially diagnostic: they mean the lift is concentrated in one sub-period of the test window.",
    )

    from autosignalx.eval.survival import load_survival

    records = load_survival()
    if not records:
        st.warning(
            "No survival records yet. Run `autosignalx agent harden` to "
            "evaluate every promoted finding under FDR + adversarial attacks."
        )
        if st.button("Run hardening now"):
            with st.spinner("Running FDR + adversarial replication..."):
                from autosignalx.eval.survival import harden_findings

                harden_findings()
            st.rerun()
        return

    # Headline survival counts
    n = len(records)
    n_fdr = sum(1 for r in records if r.get("survives_fdr"))
    n_full = sum(1 for r in records if r.get("survives_full_test"))
    n_placebo = sum(1 for r in records if r.get("survives_placebo") is True)
    n_block = sum(1 for r in records if r.get("survives_block_holdout"))
    n_all = sum(1 for r in records if r.get("survives_all"))

    cols = st.columns(5)
    cols[0].metric("Total promoted", n)
    cols[1].metric("Survive FDR", f"{n_fdr}/{n}")
    cols[2].metric("Full-test", f"{n_full}/{n}")
    cols[3].metric("Placebo", f"{n_placebo}/{n}")
    cols[4].metric("Block holdout", f"{n_block}/{n}")

    if n_all == 0 and n > 0:
        st.warning(
            f"**Zero of {n} promoted findings survive every attack.** "
            "This is what an honest research methodology looks like: the "
            "system flagged candidates, the hardening exposed which were "
            "fragile. The methodology — not the finding count — is the "
            "research artifact."
        )
    elif n_all > 0:
        st.success(
            f"**{n_all} of {n} promoted findings survive every attack.** "
            "These are robust to FDR correction, full-test replication, "
            "regime-placebo, and block-holdout."
        )

    # Per-finding pass/fail grid
    grid_rows = []
    for r in records:
        def _mark(v):
            if v is True:
                return "✅"
            if v is False:
                return "❌"
            return "—"

        grid_rows.append({
            "finding_id": r.get("finding_id"),
            "method": r.get("method"),
            "filters": str(r.get("filters")),
            "p (orig)": f"{r.get('original_p', float('nan')):.4f}" if r.get("original_p") is not None else "—",
            "q (FDR)": f"{r.get('fdr_q', float('nan')):.4f}" if r.get("fdr_q") is not None else "—",
            "FDR": _mark(r.get("survives_fdr")),
            "full-test": _mark(r.get("survives_full_test")),
            "placebo": _mark(r.get("survives_placebo")),
            "block-holdout": _mark(r.get("survives_block_holdout")),
            "all": _mark(r.get("survives_all")),
        })
    st.dataframe(pd.DataFrame(grid_rows), use_container_width=True, hide_index=True)

    # Detailed expander per finding
    st.subheader("Per-finding evidence")
    for r in records:
        title = f"{r.get('finding_id')} · {r.get('method')} · {r.get('filters')}"
        with st.expander(title):
            st.markdown(f"**Hypothesis:** {r.get('hypothesis', '')}")
            adv_d = r.get("adversarial", {})
            ft = adv_d.get("full_test", {})
            pl = adv_d.get("placebo", {})
            bh = adv_d.get("block_holdout", {})
            cc = st.columns(3)
            with cc[0]:
                st.markdown("**Full-test replication**")
                if ft.get("reason"):
                    st.caption(ft.get("reason"))
                else:
                    st.write({"n": ft.get("n"), "skill": ft.get("skill_vs_baseline"),
                              "p": ft.get("p_value"), "ci": [ft.get("bootstrap_ci_low"), ft.get("bootstrap_ci_high")]})
            with cc[1]:
                st.markdown("**Placebo (shuffled regimes)**")
                if pl.get("reason"):
                    st.caption(pl.get("reason"))
                else:
                    st.write({"placebo_promotable": pl.get("promotable"),
                              "placebo_p": pl.get("p_value"), "placebo_skill": pl.get("skill_vs_baseline")})
            with cc[2]:
                st.markdown("**Block holdout (50/50)**")
                if bh.get("reason"):
                    st.caption(bh.get("reason"))
                else:
                    st.write({"split_at": bh.get("split_at"),
                              "first_half_p": bh.get("first_half", {}).get("p_value"),
                              "second_half_p": bh.get("second_half", {}).get("p_value")})


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


def render_coverage_map() -> None:
    """Phase 16: hypothesis search-space coverage map."""
    st.title("Coverage Map")
    st.caption(
        "Where has the agent looked? Each cell of (asset × regime × method) is "
        "colored by status: open / tested / promoted / no-data. EIG ranks the "
        "highest-information next experiment."
    )

    _panel_doc(
        inputs="Concatenated `reports/ablations/*.parquet`, joined to `reports/regimes/kmeans.parquet` for regime labels; `reports/agent/findings.jsonl` for the 'promoted' overlay; `reports/agent/ledger.jsonl` for the 'tested' overlay (parsed by `agent.eig._build_tested_keys_from_ledger`).",
        operations="Builds a candidate grid over Cartesian product (methods × assets × regimes), computes a Phase-14 EIG proxy = α·novelty + β·sqrt(n_samples) − 0.5·already_tested, and renders the coverage status. The proxy avoids re-testing settled cases.",
        goal="Make the agent's exploration legible. Reviewers see what's been examined, what hasn't, and where new effort should go.",
        interpretation="High EIG with `status=open` and `n_samples > 200` is the best next experiment slot. Many `tested` cells without promotion is a sign the slice is genuinely null. `no_data` cells need a forecast pass first.",
    )

    from autosignalx.agent import eig as eig_mod
    from autosignalx.agent import findings as findings_mod
    from autosignalx.agent import tools as tools_mod

    forecasts = tools_mod._load_all_forecasts()
    if forecasts.empty:
        st.warning("No forecasts cached. Run `autosignalx eval baseline` first.")
        return
    rl_path = settings.reports_dir / "regimes" / "kmeans.parquet"
    if rl_path.exists() and "forecast_origin" in forecasts.columns:
        try:
            rl = pd.read_parquet(rl_path)
            rl_join = rl[["timestamp", "regime_id"]].rename(columns={"timestamp": "forecast_origin"})
            rl_join["forecast_origin"] = pd.to_datetime(rl_join["forecast_origin"])
            forecasts["forecast_origin"] = pd.to_datetime(forecasts["forecast_origin"])
            forecasts = forecasts.merge(rl_join, on="forecast_origin", how="left")
        except Exception:  # noqa: BLE001
            pass

    methods = sorted(forecasts["method"].unique())
    assets = sorted(forecasts["asset"].unique())
    regimes = (
        sorted(forecasts["regime_id"].dropna().unique().astype(int).tolist())
        if "regime_id" in forecasts.columns
        else []
    )
    if not regimes:
        st.warning("No regime labels available. Run `autosignalx regime fit` first.")
        return
    findings_records = findings_mod.load()
    df = eig_mod.coverage_map(
        forecasts=forecasts,
        methods=methods,
        assets=assets,
        regimes=regimes,
        findings=findings_records,
    )
    st.subheader("Coverage table")
    st.caption(
        f"{len(df)} (method × asset × regime) cells; {sum(df['status']=='promoted')} promoted, "
        f"{sum(df['status']=='tested')} previously tested, {sum(df['status']=='open')} open."
    )
    st.dataframe(df.sort_values("eig_score", ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Coverage heatmap (per method)")
    method_sel = st.selectbox("Method", methods, index=min(2, len(methods) - 1))
    sub = df[df["method"] == method_sel]
    pivot = sub.pivot_table(index="asset", columns="regime_id", values="eig_score")
    try:
        import plotly.express as px

        fig = px.imshow(
            pivot, aspect="auto", color_continuous_scale="Viridis",
            labels={"color": "EIG score", "x": "regime_id", "y": "asset"},
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:  # noqa: BLE001
        st.dataframe(pivot)


def render_statistical_power() -> None:
    """Phase 16: per-cell sample-size and power dashboard."""
    st.title("Statistical Power Dashboard")
    st.caption(
        "For every (method, asset, regime) cell, what is the observed effect "
        "size, the implied power at α=0.05, and the sample size required "
        "for 80% power? Distinguishes 'under-powered failure' from 'genuine null'."
    )

    _panel_doc(
        inputs="Concatenated `reports/ablations/*.parquet` joined to `reports/regimes/kmeans.parquet`.",
        operations="Per cell: align method vs naive on (timestamp, asset, forecast_origin), compute Cohen's d on per-bar loss differences, approximate one-sided t-test power via noncentral t, and bisect on n for 80% power.",
        goal="Tell the reviewer whether a non-promotable cell was *underpowered* (n too small) or *genuinely null* (large n with d ≈ 0).",
        interpretation="Cells with d > 0.1 and power > 0.8 that didn't promote are real failures of the candidate method. Cells with low power are uninformative -- the agent hasn't yet collected enough data to render a verdict.",
    )

    from autosignalx.agent import tools as tools_mod
    from autosignalx.eval.power import power_grid

    forecasts = tools_mod._load_all_forecasts()
    if forecasts.empty:
        st.warning("No forecasts cached.")
        return
    rl_path = settings.reports_dir / "regimes" / "kmeans.parquet"
    if rl_path.exists() and "forecast_origin" in forecasts.columns:
        try:
            rl = pd.read_parquet(rl_path)
            rl_join = rl[["timestamp", "regime_id"]].rename(columns={"timestamp": "forecast_origin"})
            rl_join["forecast_origin"] = pd.to_datetime(rl_join["forecast_origin"])
            forecasts["forecast_origin"] = pd.to_datetime(forecasts["forecast_origin"])
            forecasts = forecasts.merge(rl_join, on="forecast_origin", how="left")
        except Exception:  # noqa: BLE001
            pass

    methods = sorted(forecasts["method"].unique())
    grid = power_grid(forecasts, methods=methods, baseline="naive")
    if grid.empty:
        st.warning("Power grid is empty.")
        return

    cols = st.columns(3)
    cols[0].metric("Cells", len(grid))
    cols[1].metric("Mean power", f"{grid['power'].mean():.2f}")
    cols[2].metric("Mean Cohen d", f"{grid['d'].mean():+.3f}")

    st.subheader("Per-cell power")
    st.dataframe(
        grid.sort_values(["method", "asset", "regime_id"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Power vs effect size (scatter)")
    try:
        import plotly.express as px

        fig = px.scatter(
            grid, x="d", y="power", color="method", size="n",
            hover_data=["asset", "regime_id"],
            labels={"d": "Cohen's d", "power": "Power at α=0.05"},
        )
        fig.add_hline(y=0.8, line_dash="dash", annotation_text="80% target")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:  # noqa: BLE001
        st.write(grid)


def render_counterfactual_cards() -> None:
    """Phase 16: counterfactual interrogation per finding."""
    st.title("Counterfactual Cards")
    st.caption(
        "For every promoted finding: factor-residualised lift (after subtracting "
        "macro-factor exposure), what-if perturbations across prediction-magnitude "
        "buckets, and outlier-removal stability."
    )

    _panel_doc(
        inputs="`reports/agent/findings.jsonl`; concatenated `reports/ablations/*.parquet`; `data/cache/macro.parquet` for the factor regression.",
        operations="`eval.counterfactual.factor_residualization` regresses per-bar loss-difference on 5-day-diff macro factors and reports the residual mean. `what_if_perturbation` slices by prediction-magnitude quartile. `outlier_removal` drops the top 1% absolute-difference rows and recomputes skill.",
        goal="Make a finding's evidence interrogable: does the lift survive after subtracting common factor exposure? Is it concentrated in extreme predictions? Is it dominated by a handful of outlier days?",
        interpretation="If `fraction_explained` is high, the finding is partly a factor bet. If `inlier_skill` collapses below `raw_skill`, the lift is outlier-driven. If both stay positive across what-if buckets, the structure is robust.",
    )

    from autosignalx.agent import findings as findings_mod
    from autosignalx.agent import tools as tools_mod
    from autosignalx.eval.counterfactual import counterfactual_card

    findings_records = findings_mod.load()
    if not findings_records:
        st.info("No promoted findings yet.")
        return
    forecasts = tools_mod._load_all_forecasts()
    if forecasts.empty:
        st.warning("No forecasts cached.")
        return
    rl_path = settings.reports_dir / "regimes" / "kmeans.parquet"
    if rl_path.exists() and "forecast_origin" in forecasts.columns:
        try:
            rl = pd.read_parquet(rl_path)
            rl_join = rl[["timestamp", "regime_id"]].rename(columns={"timestamp": "forecast_origin"})
            rl_join["forecast_origin"] = pd.to_datetime(rl_join["forecast_origin"])
            forecasts["forecast_origin"] = pd.to_datetime(forecasts["forecast_origin"])
            forecasts = forecasts.merge(rl_join, on="forecast_origin", how="left")
        except Exception:  # noqa: BLE001
            pass

    for f in findings_records:
        with st.expander(f"**{f.get('id', '?')}** — {(f.get('hypothesis') or '')[:120]}", expanded=False):
            method = f.get("method")
            ev = f.get("evidence", {}) or {}
            baseline = ev.get("baseline_method", "naive")
            filters = f.get("filters") or {}
            card = counterfactual_card(
                forecasts=forecasts, method=method, baseline=baseline,
                asset=filters.get("asset"), regime_id=filters.get("regime_id"),
            )

            fr = card.get("factor_residualization", {}) or {}
            st.markdown("**Factor residualization**")
            if "reason" in fr:
                st.caption(f"(skipped: {fr['reason']})")
            else:
                cols = st.columns(3)
                cols[0].metric("Raw mean diff", f"{fr.get('raw_mean_loss_diff', 0):.5f}")
                cols[1].metric("Residual mean", f"{fr.get('residual_mean_loss_diff', 0):.5f}")
                cols[2].metric("Fraction explained", f"{fr.get('fraction_explained', 0):.1%}")
                st.caption(f"t_residual = {fr.get('t_residual', 0):.2f}, p_residual = {fr.get('p_residual', 0):.3f}")
                st.json(fr.get("factor_betas", {}))

            wi = card.get("what_if", {}) or {}
            st.markdown("**What-if (per prediction-magnitude bucket)**")
            buckets = wi.get("buckets", [])
            if buckets:
                st.dataframe(pd.DataFrame(buckets), use_container_width=True, hide_index=True)

            ol = card.get("outlier_removal", {}) or {}
            st.markdown("**Outlier removal**")
            if "reason" in ol:
                st.caption(f"(skipped: {ol['reason']})")
            else:
                cols = st.columns(2)
                cols[0].metric("Raw skill", f"{ol.get('raw_skill_vs_baseline', 0):+.4f}")
                cols[1].metric("Inlier skill", f"{ol.get('inlier_skill_vs_baseline', 0):+.4f}")
                st.caption(
                    f"Dropped top {(1 - ol.get('cutoff_quantile', 1.0)) * 100:.0f}% absolute-diff rows; "
                    f"n_total={ol.get('n_total')} -> n_inlier={ol.get('n_inlier')}."
                )


def render_bayesian_evidence() -> None:
    """Phase 12 + 16: hierarchical Bayesian posterior + Bayes factors."""
    st.title("Bayesian Evidence")
    st.caption(
        "Per-finding posterior over the true skill (Normal-Normal hierarchical "
        "model with empirical-Bayes shrinkage), Bayes factor vs the null, and "
        "posterior-predictive intervals."
    )

    _panel_doc(
        inputs="`reports/agent/findings.jsonl`; concatenated `reports/ablations/*.parquet` joined to `reports/regimes/kmeans.parquet`.",
        operations="`eval.bayesian.hierarchical_findings` fits a Normal-Normal hierarchical model: d_i ~ N(theta_i, sigma_i^2/n_i), theta_i ~ N(mu, tau^2), with mu and tau^2 fit by method of moments. Reports posterior mean, sd, P(theta>0), Bayes factor BF_10 vs theta=0.",
        goal="Provide decision-relevant evidence the frequentist DM gate doesn't expose. Bayes factors directly answer 'how much should I update?'; posterior P(theta>0) gives a calibrated probability statement.",
        interpretation="BF > 10 = 'strong evidence' for the alternative; BF > 30 = 'very strong'; BF < 1 = data favours the null. P(theta>0) > 0.95 plus BF > 10 is the lab-grade Bayesian bar.",
    )

    from autosignalx.agent import findings as findings_mod
    from autosignalx.agent import tools as tools_mod
    from autosignalx.eval.bayesian import hierarchical_findings, posterior_predictive_check

    findings_records = findings_mod.load()
    if not findings_records:
        st.info("No promoted findings yet.")
        return
    forecasts = tools_mod._load_all_forecasts()
    if forecasts.empty:
        st.warning("No forecasts cached.")
        return
    rl_path = settings.reports_dir / "regimes" / "kmeans.parquet"
    if rl_path.exists() and "forecast_origin" in forecasts.columns:
        try:
            rl = pd.read_parquet(rl_path)
            rl_join = rl[["timestamp", "regime_id"]].rename(columns={"timestamp": "forecast_origin"})
            rl_join["forecast_origin"] = pd.to_datetime(rl_join["forecast_origin"])
            forecasts["forecast_origin"] = pd.to_datetime(forecasts["forecast_origin"])
            forecasts = forecasts.merge(rl_join, on="forecast_origin", how="left")
        except Exception:  # noqa: BLE001
            pass

    summary = hierarchical_findings(findings_records, forecasts)
    if summary.n_findings == 0:
        st.info("Not enough aligned data for Bayesian inference yet.")
        return

    cols = st.columns(3)
    cols[0].metric("Findings", summary.n_findings)
    cols[1].metric("Population mean μ", f"{summary.mu_pop:.5f}")
    cols[2].metric("Population variance τ²", f"{summary.tau2_pop:.6f}")

    rows = []
    for bf in summary.findings:
        rows.append({
            "id": bf.finding_id,
            "n": bf.n,
            "data_mean": bf.d_mean,
            "posterior_mean": bf.posterior_mean,
            "posterior_sd": bf.posterior_sd,
            "P(θ>0)": bf.prob_positive,
            "Bayes factor (BF_10)": bf.bayes_factor,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if st.button("Run posterior predictive check"):
        ppc = posterior_predictive_check(findings_records, forecasts, n_simulations=500)
        st.json(ppc)


def render_specialist_council() -> None:
    """Phase 14 + 16: specialist consultations and KG explorer."""
    st.title("Specialist Council")
    st.caption(
        "Multi-role consultation feed (Statistician / Quant / RiskOfficer / "
        "Economist / Implementer / RedTeam / Historian) and the persistent "
        "knowledge-graph explorer."
    )

    _panel_doc(
        inputs="`reports/agent/ledger.jsonl` filtered for `step` starting with `specialist:`; `reports/agent/kg/{nodes,edges}.jsonl`.",
        operations="Filters the ledger for specialist consultations and groups by role. Loads the KG nodes/edges and renders kind / relation distributions plus a node search.",
        goal="Make Phase-14's multi-specialist debate visible and the persistent KG queryable from the cockpit.",
        interpretation="Each specialist's consultation is one LLM call recorded in the ledger; their advice frames the Adjudicator's verdict. The KG accumulates structural knowledge across sessions: nodes by kind, edges by relation, and a search box for 'what's been said about regime 3 / TLT / chronos2_multivariate'.",
    )

    from autosignalx.agent import knowledge_graph as kg_mod
    from autosignalx.agent import ledger as ledger_mod

    entries = ledger_mod.load()
    consult_entries = [e for e in entries if str(e.get("step", "")).startswith("specialist:")]
    pi_entries = [e for e in entries if e.get("step") == "principal_investigator"]
    st.subheader("Recent specialist consultations")
    if not consult_entries:
        st.info(
            "No specialist consults yet. Run `autosignalx agent run --mode lab` "
            "to invoke the Phase-14 multi-specialist orchestration."
        )
    else:
        cols = st.columns(2)
        with cols[0]:
            roles = sorted({str(e.get("step", "")).split(":")[1] for e in consult_entries})
            role_pick = st.selectbox("Role filter", ["(all)"] + roles)
        with cols[1]:
            n_show = st.slider("Show last N", min_value=5, max_value=200, value=30)
        sub = consult_entries
        if role_pick != "(all)":
            sub = [e for e in consult_entries if e.get("step") == f"specialist:{role_pick}"]
        for e in sub[-n_show:]:
            with st.expander(
                f"R{e.get('round')} · {e.get('step')} · {(e.get('content') or '')[:80]}",
                expanded=False,
            ):
                st.write(e.get("content"))

    if pi_entries:
        with st.expander("PrincipalInvestigator decisions", expanded=False):
            st.dataframe(
                pd.DataFrame([
                    {
                        "round": e.get("round"),
                        "next_specialist": (e.get("content") or {}).get("next_specialist"),
                        "rationale": (e.get("content") or {}).get("rationale"),
                    }
                    for e in pi_entries
                ]),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("Knowledge graph")
    summary = kg_mod.kg_summary()
    cols = st.columns(2)
    cols[0].metric("Nodes", summary.get("n_nodes", 0))
    cols[1].metric("Edges", summary.get("n_edges", 0))
    if summary.get("n_nodes", 0) > 0:
        st.write("**Nodes by kind:**", summary.get("nodes_by_kind", {}))
        st.write("**Edges by relation:**", summary.get("edges_by_relation", {}))
        kind = st.selectbox("Filter nodes by kind", ["(any)"] + list(summary.get("nodes_by_kind", {}).keys()))
        needle = st.text_input("Label contains", "")
        nodes = kg_mod.query(
            kind=None if kind == "(any)" else kind,
            label_contains=needle or None,
        )
        st.dataframe(pd.DataFrame(nodes)[:200], use_container_width=True, hide_index=True)
    else:
        st.info("KG empty. Run `autosignalx agent run --mode lab` to ingest findings into the KG.")


def render_preregistration() -> None:
    """Phase 8 + 16: pre-registration ledger."""
    st.title("Pre-Registration")
    st.caption(
        "Hypotheses hash-committed BEFORE running. Open registrations are the "
        "agent's outstanding tests; resolved registrations link a registered "
        "hypothesis to its eventual evidence."
    )

    _panel_doc(
        inputs="`reports/agent/preregistrations.jsonl` (registrations) and `reports/agent/preregistration_resolutions.jsonl` (resolutions).",
        operations="Loads both files via `eval.preregistration.load` / `load_resolutions`, joins on `id`, surfaces open vs resolved.",
        goal="Make the agent's commitment-before-evidence auditable. The ratio resolved / open is a coverage-of-obligation metric.",
        interpretation="A registered hypothesis whose resolution is `promoted=True` is one that survived the gate as predicted. Registered without resolution = open obligation. Resolved with `promoted=False` = honest negative result.",
    )

    from autosignalx.eval.preregistration import load, load_resolutions

    regs = load()
    resols = load_resolutions()
    by_id = {r.get("preregistration_id"): r for r in resols}

    cols = st.columns(3)
    cols[0].metric("Registered", len(regs))
    cols[1].metric("Resolved", len(by_id))
    cols[2].metric("Open", len(regs) - len(by_id))

    rows = []
    for r in regs:
        res = by_id.get(r.get("id"))
        rows.append({
            "id": r.get("id"),
            "method": r.get("method"),
            "filters": r.get("filters"),
            "registered_at": r.get("registered_at"),
            "status": "open" if res is None else ("promoted" if res.get("promoted") else "refuted"),
            "p_value": (res or {}).get("evidence", {}).get("p_value"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_holdout_vault() -> None:
    """Phase 8 + 16: holdout vault status."""
    st.title("Holdout Vault")
    st.caption(
        "Never-touched final test slice. Reviewers see whether the vault is "
        "locked, whether discovery has accidentally consulted it, and (after "
        "explicit open) the final headline metric."
    )

    _panel_doc(
        inputs="`reports/agent/holdout_vault/{vault.json,results.json}`.",
        operations="`eval.holdout_vault.vault_status` reports lock state and bounds. `assert_no_vault_leakage` raises if any forecast row's `forecast_origin` falls inside the locked range -- intended to be wired into the agent's experiment node.",
        goal="Distinguish the discovery phase (forbidden inside the vault) from the publication phase (one-time vault open).",
        interpretation="A locked vault with a non-zero lock-hash and zero leakage events is a healthy state. After publication, `vault.json` records `opened: True` and `results.json` carries the headline numbers — a one-time, unalterable record.",
    )

    from autosignalx.eval.holdout_vault import VAULT_RESULTS, vault_status

    s = vault_status()
    if not s.get("initialized"):
        st.info(
            "Vault not initialized. Run `autosignalx eval vault-init <start> <end>` "
            "to lock a final test slice before publication."
        )
        return
    cols = st.columns(3)
    cols[0].metric("Start", s.get("start"))
    cols[1].metric("End", s.get("end"))
    cols[2].metric("Status", "OPENED" if s.get("opened") else "LOCKED")
    st.caption(f"Lock hash: `{s.get('lock_hash')}` (locked at {s.get('locked_at')})")
    if s.get("opened") and VAULT_RESULTS.exists():
        st.subheader("Vault evaluation (final headline numbers)")
        st.json(json.loads(VAULT_RESULTS.read_text(encoding="utf-8")))


def render_calibration_panel() -> None:
    """Phase 15 + 16: agent confidence calibration."""
    st.title("Agent Calibration")
    st.caption(
        "Reliability diagram of the agent's predicted confidence vs the survival "
        "rate of its findings. Brier score and ECE summarise how well-calibrated "
        "the agent's intuition is."
    )

    _panel_doc(
        inputs="`reports/agent/findings.jsonl` (predicted_effect / agent_confidence) and `reports/agent/survival.jsonl` (survives_all_strict).",
        operations="`agent.calibration.calibration_for_role` coerces confidence (numeric or text) to [0,1], bins predictions, and reports per-bin observed-survival rate plus Brier score and ECE.",
        goal="Score the Theorist's calibration over time. A well-calibrated agent's 0.8-confidence findings should survive hardening 80% of the time.",
        interpretation="Brier closer to 0 = better; ECE closer to 0 = bins agree with their predictions; the reliability curve should sit on the y=x identity. Consistent over-confidence (curve below y=x) is a sign the Theorist's prompt should be retuned.",
    )

    from autosignalx.agent import calibration as calibration_mod
    from autosignalx.agent import findings as findings_mod
    from autosignalx.eval.survival import load_survival

    findings_records = findings_mod.load()
    survival = load_survival()
    if not findings_records or not survival:
        st.info("Need at least one promoted finding plus its survival record.")
        return

    rec = calibration_mod.calibration_for_role(
        findings=findings_records,
        survival_records=survival,
        role="theorist",
    )
    cols = st.columns(3)
    cols[0].metric("N", rec.n)
    cols[1].metric("Brier score", f"{rec.brier:.3f}")
    cols[2].metric("ECE", f"{rec.ece:.3f}")
    if rec.bins:
        st.subheader("Reliability bins")
        st.dataframe(pd.DataFrame(rec.bins), use_container_width=True, hide_index=True)

        try:
            import plotly.graph_objects as go

            xs = [b.get("mean_confidence") for b in rec.bins if b.get("mean_confidence") is not None]
            ys = [b.get("obs_rate") for b in rec.bins if b.get("obs_rate") is not None]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+lines", name="Observed"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line={"dash": "dash"}))
            fig.update_layout(xaxis_title="Predicted confidence", yaxis_title="Observed survival rate")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:  # noqa: BLE001
            pass


def render_red_team_panel() -> None:
    """Phase 15 + 16: RedTeam attack outcomes."""
    st.title("RedTeam Attacks")
    st.caption(
        "Per-finding asset-shuffle and time-shift adversarial attacks beyond "
        "the existing FDR / full-test / placebo / block-holdout suite."
    )

    _panel_doc(
        inputs="`reports/agent/red_team.jsonl` produced by `autosignalx agent eval-suite`.",
        operations="Asset-shuffle: re-run the gate on every other asset in the same regime; finding survives iff no other asset is also promotable. Time-shift: shift forecast_origin by 5 days; finding survives the shift if the lift wasn't a date-specific coincidence.",
        goal="Catch findings the existing hardening misses: asset-non-specific lifts and date-coincidence artefacts.",
        interpretation="If a finding is `promotable_elsewhere=[…]`, the regime alone explains the lift -- the asset specificity was spurious. If `promotable_after_shift=False` and original is True, the lift may have been driven by a single date.",
    )

    from autosignalx.agent.red_team import load_red_team

    records = load_red_team()
    if not records:
        st.info(
            "No RedTeam records yet. Run `autosignalx agent eval-suite` to "
            "generate them."
        )
        return
    rows = []
    for r in records:
        a = r.get("asset_shuffle", {}) or {}
        t = r.get("time_shift", {}) or {}
        rows.append({
            "finding_id": r.get("finding_id"),
            "asset_shuffle_survives": a.get("survives"),
            "promotable_elsewhere": a.get("promotable_elsewhere"),
            "time_shift_promotable": t.get("promotable_after_shift"),
            "time_shift_skill": t.get("skill"),
            "survives_red_team": r.get("survives_red_team"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_coherence_panel() -> None:
    """Phase 15 + 16: long-horizon agent coherence."""
    st.title("Agent Coherence")
    st.caption(
        "Per-session coherence proxies: lessons uptake, lineage branching factor, "
        "theme persistence entropy, composite coherence score."
    )

    _panel_doc(
        inputs="`reports/agent/ledger.jsonl`, `reports/agent/lessons.md`, plus the lineage DAG built from the ledger.",
        operations="`agent.coherence.score_session` computes lessons-uptake (substring match against prior lessons), lineage branching factor (mean out-degree), and Shannon entropy of (asset, regime) cells visited.",
        goal="Make multi-session research arc coherence visible -- did each session build on prior lessons or drift?",
        interpretation="High lessons_uptake means later sessions cite earlier insight. Branching factor near 1 is healthy; very high = scattered, very low = no refinement. Entropy ~ ln(distinct cells visited): too low = stuck on one slice; too high = no focus.",
    )

    from autosignalx.agent.coherence import load_coherence

    records = load_coherence()
    if not records:
        st.info(
            "No coherence records yet. Run `autosignalx agent eval-suite` to "
            "score every session in the ledger."
        )
        return
    df = pd.DataFrame(records)
    if "evaluated_at" in df.columns:
        df = df.sort_values("evaluated_at")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if "coherence_score" in df.columns and len(df) > 1:
        try:
            import plotly.express as px

            fig = px.line(df, x="evaluated_at", y="coherence_score",
                          color="session_id", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:  # noqa: BLE001
            pass


def render_reproducibility_panel() -> None:
    """Phase 16: reproducibility badge."""
    st.title("Reproducibility Badge")
    st.caption(
        "Git hash, Python env, key library versions, replay-mode flag, and "
        "content hashes of every parquet/JSONL artifact under `reports/`. "
        "Bundle hash is the deterministic identifier for the current state."
    )

    _panel_doc(
        inputs="Live computed: `git rev-parse`, `git status --porcelain`, package metadata, `reports/` artifact bytes.",
        operations="`autosignalx.reproducibility.reproducibility_badge` collects git+env+per-file SHA-256 hashes, then derives a single artifacts_bundle_hash from the sorted file map.",
        goal="Make any cockpit screenshot or finding citation reproducible: paste the bundle hash and the reader can verify the same state.",
        interpretation="If git is `dirty`, the displayed numbers may include uncommitted changes. The bundle hash changes whenever any artifact changes -- it is the cryptographic fingerprint of 'what the cockpit is currently showing'.",
    )

    from autosignalx.reproducibility import reproducibility_badge, write_badge

    if st.button("Compute / refresh badge"):
        path = write_badge()
        st.success(f"Wrote {path}")

    badge = reproducibility_badge()
    cols = st.columns(3)
    cols[0].metric("Bundle hash", badge.get("artifacts_bundle_hash", "?"))
    cols[1].metric("Artifacts", badge.get("n_artifacts", 0))
    cols[2].metric("Replay mode", str(badge.get("replay_mode", False)))
    st.subheader("Git")
    st.json(badge.get("git", {}))
    st.subheader("Environment")
    st.json(badge.get("env", {}))
    with st.expander("Artifact hashes"):
        rows = [
            {"path": k, "sha256_16": v}
            for k, v in (badge.get("artifact_hashes") or {}).items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_headline() -> None:
    """Default-landing panel: a self-contained system overview.

    Describes what AutoSignal-X is, why it exists, how the pieces fit
    together, and what the bundled artifacts say about the apparatus'
    behaviour. Every other panel in the cockpit shows the math behind a
    number rendered here."""
    st.title("AutoSignal-X")
    st.caption(
        "AI research system for discovering predictive structure in liquid "
        "daily ETF prices, with an autonomous research loop that grades its "
        "own discoveries through a multi-stage statistical methodology."
    )

    rd = settings.reports_dir
    findings_path = rd / "agent" / "findings.jsonl"
    survival_path = rd / "agent" / "survival.jsonl"
    synth_path = rd / "agent" / "synthetic_benchmark.json"
    cap_path = rd / "agent" / "capability_ablation.json"
    badge_path = rd / "reproducibility_badge.json"

    # ---- Section 1: what the system is + why ---------------------------
    st.markdown(
        """
        AutoSignal-X discovers conditional predictive structure in liquid
        daily ETF prices and grades every claim through a defendable
        statistical methodology before any finding ships. Five model layers
        — **L1 forecasting** (Chronos-2 + classical baselines), **L2 latent
        regimes** (contrastive 1D-CNN encoder + KMeans + parallel HMM),
        **L3 per-regime feature ranking** (HistGradientBoosting + permutation
        importance + walk-forward stability), **L4 cross-asset structure**
        (GLASSO partial correlations + Granger causality + per-regime
        graphs), **L5 agentic discovery** (LangGraph state machine in
        `single` / `debate` / `lab` modes) — feed a hardening pipeline that
        runs every promoted finding through Diebold-Mariano + block
        bootstrap → BH-FDR → adversarial replication → Combinatorial
        Purged CV → Probability of Backtest Overfitting → Deflated Sharpe →
        Romano-Wolf joint stepdown → hierarchical Bayesian shrinkage →
        a strict bar that is the conjunction of every gate.

        The contribution is the **methodology and the agent that operates
        it**, not any single trade. The same apparatus runs on the bundled
        8-ETF universe (the default) or any user-defined universe via the
        Phase-2 Custom Study workspace.
        """
    )

    # ---- Section 2: bundled-result metric row ---------------------------
    n_findings = 0
    n_strict = 0
    if findings_path.exists():
        with findings_path.open("r", encoding="utf-8") as fh:
            n_findings = sum(1 for line in fh if line.strip())
    survival_records: list[dict] = []
    if survival_path.exists():
        with survival_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("survives_all_strict"):
                    n_strict += 1
                survival_records.append(rec)

    synth_summary: dict = {}
    if synth_path.exists():
        import contextlib

        with contextlib.suppress(Exception):
            synth_summary = json.loads(synth_path.read_text(encoding="utf-8"))
    dm_recall = strict_recall = strict_fdr = None
    if synth_summary:
        for row in synth_summary.get("ablations", []):
            if row.get("gate") == "dm_only":
                dm_recall = row.get("mean_recall")
            if row.get("gate") == "strict":
                strict_recall = row.get("mean_recall")
                strict_fdr = row.get("mean_fdr")

    st.subheader("Bundled artifacts at a glance")
    cols = st.columns(4)
    cols[0].metric(
        "Promoted findings",
        n_findings,
        help="Count of hypotheses that passed the initial DM + bootstrap promotion gate during the bundled session(s). Stored in reports/agent/findings.jsonl.",
    )
    cols[1].metric(
        "Survive every gate (strict bar)",
        f"{n_strict}/{n_findings}" if n_findings else "—",
        help="Subset of promoted findings that survive FDR + adversarial replication + CPCV + Romano-Wolf + Deflated Sharpe + hierarchical Bayesian (BF₁₀ ≥ 10 and P(θ>0) ≥ 0.95). Stored in reports/agent/survival.jsonl as `survives_all_strict`.",
    )
    if strict_recall is not None:
        cols[2].metric(
            "Synthetic-benchmark recall (strict)",
            f"{strict_recall:.0%}",
            delta=(
                f"DM-only: {dm_recall:.0%}"
                if dm_recall is not None
                else None
            ),
            help="Fraction of deliberately-planted truth cells the apparatus recovers at the strict bar on a controlled synthetic universe. Lower than DM-only by design — the gates trade recall for FDR control.",
        )
    else:
        cols[2].metric("Synthetic-benchmark recall (strict)", "—")
    if strict_fdr is not None:
        cols[3].metric(
            "Synthetic-benchmark FDR (strict)",
            f"{strict_fdr:.0%}",
            help="Fraction of strict-bar promotions on the synthetic universe that are distractor cells (no planted signal). Should be near zero — that's the gates' contract.",
        )
    else:
        cols[3].metric("Synthetic-benchmark FDR (strict)", "—")

    # ---- Section 3: strict-bar verdict + finding card -------------------
    st.divider()
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Strict-bar verdict on the promoted findings")
        if n_findings == 0:
            st.info(
                "No findings yet. Run `autosignalx agent run --mode lab` to "
                "drive a fresh research session, then `autosignalx agent harden` "
                "to grade everything through the methodology stack."
            )
        elif n_strict == 0:
            st.warning(
                f"**0 of {n_findings} promoted findings survive every gate.** "
                "The hardening exposed exactly the fragility each finding has — "
                "for example, the bundled finding `f_9395cd1bd1be` (TLT, regime 3, "
                "chronos2_multivariate) passes BH-FDR + full-test + placebo but "
                "fails 50/50 block-holdout: the lift is concentrated in the "
                "first half of the test window and does not corroborate in the "
                "second half. This is the apparatus correctly grading its own "
                "discovery, not a failure mode — the gates exist precisely to "
                "expose this kind of fragility before any trade is placed."
            )
        else:
            st.success(
                f"**{n_strict} of {n_findings} promoted findings survive every gate.** "
                "These passed BH-FDR + adversarial replication + Combinatorial "
                "Purged CV + Romano-Wolf + Deflated Sharpe + hierarchical "
                "Bayesian (BF₁₀ ≥ 10 and P(θ>0) ≥ 0.95)."
            )
        if survival_records:
            shown_cols = ["finding_id", "method", "filters", "original_p",
                          "fdr_q", "survives_block_holdout", "survives_rw",
                          "survives_dsr", "survives_bayes", "survives_all_strict"]
            sdf = pd.DataFrame(survival_records)
            cols_present = [c for c in shown_cols if c in sdf.columns]
            st.dataframe(sdf[cols_present], use_container_width=True, hide_index=True)

    with cols[1]:
        st.subheader("Layer-by-layer marginal contribution")
        st.caption(
            "Each variant adds one layer's worth of methods to the pool the "
            "promotion pipeline can draw from. *Mean MAE* is computed on the "
            "union of methods the variant has access to. *Marginal skill* is "
            "the MAE drop vs the previous variant. *Cost* is the on-disk "
            "size of the precomputed forecast parquets the variant consumes."
        )
        if cap_path.exists():
            try:
                cap = json.loads(cap_path.read_text(encoding="utf-8"))
                rows = cap.get("rows") or []
                if rows:
                    df = pd.DataFrame([
                        {
                            "variant": r.get("variant"),
                            "Mean MAE": r.get("mean_mae"),
                            "Marg. skill": r.get("marginal_skill"),
                            "n findings": r.get("n_findings"),
                            "Cost (KB)": int(r.get("cost_proxy", 0) / 1024),
                        }
                        for r in rows
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("Run `autosignalx eval ablate-capability`.")
            except Exception:  # noqa: BLE001
                st.info("Run `autosignalx eval ablate-capability`.")
        else:
            st.info("Run `autosignalx eval ablate-capability` to populate this card.")

    # ---- Section 4: pipeline at a glance --------------------------------
    st.divider()
    st.subheader("Pipeline at a glance")
    st.markdown(
        """
        ```
        yfinance OHLCV (data/cache/)            ── L1 walk-forward forecast harness ──> reports/ablations/*.parquet
                                                         │
                                                         ├── L2 contrastive encoder + KMeans + HMM ──> reports/regimes/*.parquet
                                                         ├── L3 per-regime feature ranking + walk-forward stability ──> reports/signals/*.parquet
                                                         └── L4 GLASSO + Granger + per-regime graphs ──> reports/graph/*.parquet

                                                         ▼
        L5 agent loop (single | debate | lab)   ── pre-registers each hypothesis ──> reports/agent/preregistrations.jsonl
        Theorist → (lab) Verifier → Planner →   ── runs the experiment ──────────> reports/ablations/<agent-authored>.parquet
        Specialist → Skeptic → experiment →     ── persists state ────────────────> reports/agent/{ledger,findings,kg,...}.jsonl
        Adjudicator → KG-writer

                                                         ▼
        Hardening (autosignalx agent harden)    ── DM + bootstrap → BH-FDR → adversarial → CPCV → PBO → DSR → Romano-Wolf → Bayes
                                                         ──> reports/agent/survival.jsonl (strict bar = conjunction)

        Capability evals (eval-suite + synthetic + ablate-capability)
                                                         ──> reports/agent/{calibration,red_team,coherence,synthetic_benchmark,capability_ablation}*
        ```
        """
    )

    # ---- Section 5: where to look next ----------------------------------
    st.subheader("Where to look next")
    st.markdown(
        """
        - **Survival Analysis** — per-finding pass/fail across every gate, with
          full per-attack evidence cards.
        - **Synthetic Benchmark** — the same gates run on a controlled universe
          with deliberately planted causal structure; the recall/FDR pair is
          the apparatus' own audited discriminative power.
        - **Capability Ablation** — the table above with the full Pareto plot
          of MAE vs cost-proxy across all variants.
        - **Bayesian Evidence** — per-finding posterior mean / sd, P(θ>0), and
          Bayes factor BF₁₀ from the Phase-12 hierarchical Normal-Normal model.
        - **Specialist Council** — multi-role consultation feed (Statistician /
          Quant / RiskOfficer / Economist / Implementer / RedTeam / Historian)
          plus the persistent knowledge graph that survives across sessions.
        - **Reproducibility** — git commit, library versions, replay-mode flag,
          per-artifact SHA-256, and a single bundle hash for this cockpit state.
        - **Forecast Arena, Backtest Arena, Custom Study** — the underlying
          forecast cache, the simulated trading layer, and the per-study
          workspace for running the apparatus on user-supplied data.
        """
    )

    # ---- Section 6: reproducibility chip --------------------------------
    if badge_path.exists():
        try:
            badge = json.loads(badge_path.read_text(encoding="utf-8"))
            st.caption(
                f"Reproducibility bundle hash: "
                f"`{badge.get('artifacts_bundle_hash', '?')}` "
                f"(git: `{(badge.get('git') or {}).get('commit', '?')[:12]}`, "
                f"replay mode: {badge.get('replay_mode', False)})."
            )
        except Exception:  # noqa: BLE001
            pass


def render_synthetic_benchmark() -> None:
    """Phase 7 (added): synthetic-known-answer benchmark results."""
    st.title("Synthetic Benchmark")
    st.caption(
        "Per-gate recall + FDR on a controlled universe with planted causal "
        "structure. Grades the apparatus' own discriminative power."
    )

    _panel_doc(
        inputs="`reports/agent/synthetic_benchmark.json` produced by `autosignalx eval synthetic`.",
        operations=(
            "For each of N independent universes, plants P (asset, regime, method) "
            "edges with a chosen MAE-vs-naive lift, surrounds them with D distractor "
            "(no-signal) methods, then runs every gate (DM → +FDR → +adversarial "
            "→ +Romano-Wolf → +Bayesian → strict) and reports recall (planted "
            "truths recovered) and FDR (distractors promoted)."
        ),
        goal="Make the apparatus' own statistical sensitivity an audited number.",
        interpretation=(
            "Strict-bar **recall** should be substantially below DM-only recall (gates "
            "are conservative); strict-bar **FDR** should be near zero (the gates' "
            "promise). Gap between the two columns measures how much real signal you "
            "trade off for selection-bias safety."
        ),
    )

    p = settings.reports_dir / "agent" / "synthetic_benchmark.json"
    if not p.exists():
        st.warning(
            "No synthetic benchmark results. Run "
            "`autosignalx eval synthetic --n-trials 6 --planted-skill 0.18`."
        )
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not parse synthetic_benchmark.json: {e}")
        return

    cols = st.columns(4)
    cols[0].metric("Trials", data.get("n_trials", 0))
    cols[1].metric("Planted truths / trial", data.get("planted_truths", 0))
    cols[2].metric("Distractors / trial", data.get("distractors", 0))
    cols[3].metric("Generated at", str(data.get("generated_at", ""))[:19])

    rows = data.get("ablations") or []
    df = pd.DataFrame(rows)
    if not df.empty:
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Recall", x=df["gate"], y=df["mean_recall"], marker_color="#00b894"))
            fig.add_trace(go.Bar(name="FDR", x=df["gate"], y=df["mean_fdr"], marker_color="#d63031"))
            fig.update_layout(
                barmode="group",
                yaxis_title="Mean across trials",
                xaxis_title="Gate",
                template="plotly_white",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:  # noqa: BLE001
            pass
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_capability_ablation() -> None:
    """Phase 16: layer-by-layer marginal-contribution ablation."""
    st.title("Capability Ablation")
    st.caption(
        "Layer-by-layer marginal contribution. Each variant adds one model "
        "layer's worth of methods to the pool the promotion pipeline can "
        "draw from; the table reports the resulting Mean MAE, the marginal "
        "MAE drop vs the previous variant, and a cost proxy in bytes."
    )

    _panel_doc(
        inputs="`reports/agent/capability_ablation.json` produced by `autosignalx eval ablate-capability`.",
        operations=(
            "Concatenates the cached ablation parquets, slices by `method` column, "
            "then constructs progressively richer variants (`baseline_only` -> "
            "`+arima` -> `+chronos_univ` -> `+multivariate` -> `+regime` -> `+graph` -> "
            "`full_stack`). Each variant's Mean MAE is computed on the union of "
            "methods it has access to; *marginal skill* = previous-variant-MAE − "
            "this-variant-MAE; *cost proxy* = total bytes of the bundled ablation "
            "parquets the variant consumes (each parquet counted once even if it "
            "carries multiple methods)."
        ),
        goal=(
            "Quantify each layer's marginal predictive contribution and the "
            "byte cost of keeping it in the pipeline, so the resulting "
            "capability-vs-cost frontier is explicit and auditable."
        ),
        interpretation=(
            "Variants with positive marginal-skill are load-bearing — that layer "
            "moves the headline MAE. Variants with zero or negative marginal-skill "
            "add cost without lifting the metric. The `n findings` column shows "
            "whether the layer contributes through the conditional gate (regime / "
            "graph / agent) even when raw MAE doesn't move: a finding can only "
            "appear once enough layers are in scope to express its filter."
        ),
    )

    p = settings.reports_dir / "agent" / "capability_ablation.json"
    if not p.exists():
        st.warning(
            "No ablation results. Run `autosignalx eval ablate-capability`."
        )
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not parse capability_ablation.json: {e}")
        return

    rows = data.get("rows") or []
    if not rows:
        st.info("Ablation produced no rows -- ensure `reports/ablations/` is populated.")
        return
    df = pd.DataFrame(rows)
    df["cost_proxy_kb"] = (df["cost_proxy"] / 1024).round(1)
    st.dataframe(
        df[["variant", "layers", "n_findings", "mean_mae",
            "marginal_skill", "cost_proxy_kb"]],
        use_container_width=True,
        hide_index=True,
    )

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["cost_proxy_kb"], y=df["mean_mae"], mode="lines+markers+text",
            text=df["variant"], textposition="top right", line={"color": "#0984e3"},
        ))
        fig.update_layout(
            xaxis_title="Cost proxy (KB)",
            yaxis_title="Mean MAE (lower = better)",
            template="plotly_white",
            height=420,
            title="Capability vs cost",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:  # noqa: BLE001
        pass


# ---- Sidebar grouping ----------------------------------------------------

PANEL_SECTIONS = [
    ("Headline", [
        ("Headline", render_headline),
    ]),
    ("Data & Forecasts", [
        ("Overview", render_overview),
        ("Data", render_data),
        ("Forecast Arena", render_forecast_arena),
    ]),
    ("Discovery (L2-L4)", [
        ("Regime Explorer", render_regime_explorer),
        ("Signal Discovery Lab", render_signal_lab),
        ("Cross-Asset Graph", render_cross_asset_graph),
        ("Regime-Conditioned Graph", render_regime_graph),
        ("Signal Stability", render_signal_stability),
    ]),
    ("Strategy & Studies", [
        ("Backtest Arena", render_backtest_arena),
        ("Custom Study", render_custom_study),
    ]),
    ("Methodology", [
        ("Survival Analysis", lambda: render_survival_analysis()),
        ("Bayesian Evidence", render_bayesian_evidence),
        ("Synthetic Benchmark", render_synthetic_benchmark),
        ("Capability Ablation", render_capability_ablation),
        ("Coverage Map", render_coverage_map),
        ("Statistical Power", render_statistical_power),
        ("Counterfactual Cards", render_counterfactual_cards),
        ("Pre-Registration", render_preregistration),
        ("Holdout Vault", render_holdout_vault),
        ("RedTeam Attacks", render_red_team_panel),
        ("Agent Calibration", render_calibration_panel),
        ("Agent Coherence", render_coherence_panel),
    ]),
    ("Agent activity", [
        ("Agent Console", render_agent_console),
        ("Specialist Council", render_specialist_council),
        ("Auto-Play Replay", render_auto_play),
        ("Findings", render_findings),
        ("Lineage", render_lineage),
        ("Self-Critique", render_self_critique),
        ("Lessons & Memory", render_lessons),
        ("Telemetry", render_telemetry),
        ("Sessions", render_sessions),
    ]),
    ("Reviewer", [
        ("Reproducibility", render_reproducibility_panel),
        ("Ask the Memory", render_ask_the_memory),
    ]),
]

# Flat dict (preserved for compatibility with any external code that imported PANELS)
PANELS = {name: fn for _, panels in PANEL_SECTIONS for name, fn in panels}


# Streamlit executes the script top-to-bottom on every interaction.
section_names = [s[0] for s in PANEL_SECTIONS]
section_name = st.sidebar.radio(
    "Section", section_names, index=0,
    help="Sidebar groups -- start at Headline, walk down to Reviewer.",
)
panels_in_section = next(p for s, p in PANEL_SECTIONS if s == section_name)
panel_name = st.sidebar.radio(
    "Panel", [p[0] for p in panels_in_section], index=0, key=f"panel_for_{section_name}",
)
st.sidebar.divider()

# Phase 2: study-scope selector. When set, study-aware panels (Forecast
# Arena, Backtest Arena) read from that study's reports tree instead of
# the project default. Stored in session_state so panels can look it up.
try:
    from autosignalx.study import list_studies as _list_studies

    _study_names = _list_studies()
except Exception:  # noqa: BLE001
    _study_names = []
_scope_options = ["(default)", *_study_names]
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
dict(panels_in_section)[panel_name]()
