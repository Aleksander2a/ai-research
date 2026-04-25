# AutoSignal-X

> A modular AI research instrument for discovering predictive structure in dynamic markets.

AutoSignal-X is a 5-layer research stack that combines foundation forecasting (**Chronos-2**), learned temporal representations (contrastive encoder + clustering), per-regime feature ranking (**HistGradientBoosting + permutation importance**), relational analysis (partial-correlation graphs + Granger causality), and an agentic discovery loop (**LangGraph + DeepInfra**) into a unified workflow for studying *what makes signals predictive, when, and why* under regime shifts.

This is a research artifact, not a product. The goal is to make scientific discovery *legible*: every layer's output is inspectable, every experiment is logged into a persistent ledger, and the agent's reasoning is rendered in a Streamlit cockpit.

## Status

**All 5 layers complete.** L1 Forecasting (Chronos-2 + baselines), L2 Representation (contrastive encoder + KMeans + HMM), L3 Reasoning (per-regime feature ranking), L4 Relational (GLASSO + Granger + centrality), L5 Agentic (LangGraph + DeepInfra + persistent ledger). See [REPORT.md](REPORT.md) for the full layer-by-layer findings; see [iteration plan](#iteration-plan) below for branch structure.

## Headline findings

- **Foundation models alone don't beat naive on liquid daily ETF prices** -- Chronos-2 underperforms naive by 5-6% MAE; macro covariates don't help unconditionally. 80% intervals are well-calibrated (CRPS ≈ 2.9). This is a calibrated negative result, not a bug -- daily ETF prices are very close to martingales.
- **Macros dominate every regime's top features for direction prediction, but the dominant macro depends on the regime** -- 10Y yields in Regime 0, dollar index in Regimes 1+3, crude oil in Regime 2. *Conditional* macro selection is the right structure, not unconditional multi-covariate input.
- **Cross-asset graph reveals typed roles**: SPY is the hub (eigenvector centrality 0.532), GLD is statistically isolated, TLT is the bridge (highest betweenness 0.429). These typed roles become the agent's hypothesis-space inputs.
- **The live LangGraph agent composes findings from every prior layer** -- by Round 4 it proposes a mechanistic, falsifiable hypothesis: *"in Regime 3 (USD strength + elevated VIX), chronos2_multivariate will outperform naive on EFA because EFA's high betweenness centrality positions it as a bridge between US equity and international markets, allowing the multivariate transformer to encode cross-asset flight-to-quality and USD-transmission dynamics that a random-walk baseline ignores."* The conditional-improvement search opened by Iter 3's negative result.

## Architecture

| Layer | Purpose | Implementation |
|---|---|---|
| **L1 Forecasting** | Probabilistic point + interval forecasts | Chronos-2 (frozen), multivariate with covariates |
| **L2 Representation** | Latent regime discovery | Small contrastive 1D-CNN encoder + KMeans (HMM as sanity-check baseline) |
| **L3 Reasoning** | Per-regime feature relevance | HistGradientBoosting + permutation importance over technical + macro features |
| **L4 Relational** | Cross-asset dependency structure | NetworkX + statsmodels: GLASSO partial-correlation + Granger causality + centrality |
| **L5 Agentic** | Hypothesis generation, experiment orchestration | LangGraph state machine + deepagents; openevals/agentevals for trace quality |

All layers feed a shared evaluation harness (walk-forward, regime-stratified) and an append-only experiment ledger (`ledger.jsonl`) that serves as the system's persistent memory.

## Data sources

All free, all reproducible:

- **ETF universe** via [yfinance](https://pypi.org/project/yfinance/): SPY, QQQ, IWM, GLD, TLT, EFA, EEM, HYG
- **Macro signals** via yfinance: 10Y yield (`^TNX`), VIX (`^VIX`), Dollar index (`DX-Y.NYB`), Crude (`CL=F`)
- **Window**: 2010-01-01 → 2025-12-31, daily frequency

## Quick start

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. Install it first if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

Then:

```bash
git clone https://github.com/Aleksander2a/ai-research.git
cd ai-research
uv sync --all-extras
uv run streamlit run app/streamlit_app.py
```

Or, with `make`:

```bash
make sync
make demo
```

### Reviewer journey (5-minute walk through the cockpit)

The Streamlit cockpit's sidebar lists 8 panels in the order to walk:

1. **Overview** -- thesis, headline findings, layer status grid.
2. **Data** -- ETF + macro substrate; cache inventory; normalized price chart.
3. **Forecast Arena** -- 4 methods (naive, ARIMA, Chronos-2 univariate, Chronos-2 multivariate) on walk-forward; per-method overall + per-regime stratified metrics; uncertainty bands per asset.
4. **Regime Explorer** -- contrastive encoder + KMeans regimes vs Gaussian HMM baseline; PCA-2D scatter colored by regime.
5. **Signal Discovery Lab** -- per-regime feature importance ranking via permutation importance; cross-regime importance heatmap.
6. **Cross-Asset Graph** -- partial-correlation matrix, Granger edges, NetworkX centrality (degree / eigenvector / betweenness).
7. **Agent Console** -- chat-style timeline of the LangGraph agent's research session: propose → experiment → critique → decide, round after round, reading from the recorded live trace.
8. **Ask the Memory** -- free-form query against the ledger; LLM-answered in live mode, deterministic keyword search in replay mode.

### LLM provider (optional)

The agentic layer (Iter 7+) uses [DeepInfra](https://deepinfra.com/) (OpenAI-compatible) for open-source LLM inference. Without an API key, the system runs in **deterministic replay mode** -- the agent panel plays back pre-recorded traces from `replay/agent_steps.jsonl` (committed to the repo from a live recorded session), so reviewers can experience the full cockpit without provisioning an account.

To use live mode, copy `.env.example` to `.env` and set:

```bash
DEEPINFRA_API_KEY=<your key>
DEEPINFRA_MODEL_PROPOSER=moonshotai/Kimi-K2.6     # or any OpenAI-compatible model on DeepInfra
DEEPINFRA_MODEL_CRITIC=zai-org/GLM-4.7-Flash
DEEPINFRA_MODEL_CHAT=deepseek-ai/DeepSeek-V4-Pro
```

Then `make agent` (or `uv run autosignalx agent run --record-replay`) runs the loop live and appends to the replay file.

### Windows note

If `uv` reports a cross-drive cache error during install, set the cache to a directory on the same drive as the repo:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
uv sync --all-extras
```

## Documentation map

- **README.md** (this file) — project framing, headline findings, reviewer journey, quick start.
- **[REPORT.md](REPORT.md)** — layer-by-layer findings narrative; executive summary at top, per-iteration sections that grew with the codebase.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — factual implementation reference: data-flow diagram, contracts between layers, per-layer wiring, the agent loop.

## Repository layout

```
src/autosignalx/        # Library — one module per layer
  data/                 # Pullers, splits, leakage tests           (Iter 1)
  eval/                 # Walk-forward harness, metrics, ablations (Iter 2)
  forecast/             # Chronos-2 + classical baselines          (Iter 3)
  regime/               # Contrastive encoder + clustering         (Iter 4)
  signal/               # Feature engineering + per-regime ranking (Iter 5)
  graph/                # Partial-corr + Granger + centrality      (Iter 6)
  agent/                # LangGraph state machine + ledger         (Iter 7)
  cli.py                # Typer entrypoint dispatching to each layer
  config.py             # Pydantic settings (paths, env, flags)
app/                    # Streamlit research cockpit (one panel per layer)
configs/                # YAML experiment configs
tests/                  # Pytest — leakage, contracts, smoke
data/                   # (gitignored) cached parquet
reports/                # Per-run artifacts; runs/ is gitignored
replay/                 # Pre-recorded agent traces for no-LLM-key mode
REPORT.md               # Running research report — findings as iterations land
```

## Iteration plan

Each iteration ships a runnable system and merges into the integration branch with `--no-ff`.

| # | Branch | Theme |
|---|---|---|
| 0 | `iter-0-scaffold` | Repository structure, tooling, cockpit shell |
| 1 | `iter-1-data` | Reproducible ETF + macro pipeline with leakage tests |
| 2 | `iter-2-baselines` | Walk-forward harness with naive + ARIMA |
| 3 | `iter-3-chronos2` | Chronos-2 multivariate forecasting with covariates |
| 4 | `iter-4-regime` | Contrastive temporal encoder + regime clustering |
| 5 | `iter-5-signal` | Regime-aware feature ranking (HistGradientBoosting + permutation importance) |
| 6 | `iter-6-graph` | Partial-correlation + Granger cross-asset graph |
| 7 | `iter-7-agent` | LangGraph agentic research loop + persistent ledger |
| 8 | `iter-8-cockpit` | Polished cockpit with reviewer-journey navigation |
| 9 | `iter-9-report` | Consolidated report + reproducibility check |

If time runs short, later iterations are skippable — each commit on the integration branch leaves a complete, demonstrable system.

## License

MIT
