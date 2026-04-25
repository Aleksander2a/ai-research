# AutoSignal-X

> A modular AI research instrument for discovering predictive structure in dynamic markets.

AutoSignal-X is a 5-layer research stack that combines foundation forecasting (**Chronos-2**), learned temporal representations (contrastive encoder + clustering), structured signal reasoning (**TabPFN**), relational analysis (partial-correlation graphs + Granger causality), and an agentic discovery loop (**LangGraph + deepagents**) into a unified workflow for studying *what makes signals predictive, when, and why* under regime shifts.

This is a research artifact, not a product. The goal is to make scientific discovery *legible*: every layer's output is inspectable, every experiment is logged into a persistent ledger, and the agent's reasoning is rendered in a Streamlit cockpit.

## Status

**Iteration 0** — repository scaffold, packaging, test infrastructure, cockpit shell. Subsequent iterations land on their own branches and merge into the integration branch with `--no-ff` to preserve boundaries in history. See [iteration plan](#iteration-plan) below.

## Architecture

| Layer | Purpose | Implementation |
|---|---|---|
| **L1 Forecasting** | Probabilistic point + interval forecasts | Chronos-2 (frozen), multivariate with covariates |
| **L2 Representation** | Latent regime discovery | Small contrastive 1D-CNN encoder + KMeans (HMM as sanity-check baseline) |
| **L3 Reasoning** | Per-regime feature relevance | TabPFN ranking over technical + macro features |
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

### LLM provider (optional)

The agentic layer (Iter 7+) uses [DeepInfra](https://deepinfra.com/) (OpenAI-compatible) for open-source LLM inference. Without an API key, the system runs in **deterministic replay mode** — the agent panel plays back pre-recorded traces from `replay/`, so reviewers can experience the full cockpit without provisioning an account.

To use live mode, copy `.env.example` to `.env` and set `DEEPINFRA_API_KEY`.

### Windows note

If `uv` reports a cross-drive cache error during install, set the cache to a directory on the same drive as the repo:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
uv sync --all-extras
```

## Repository layout

```
src/autosignalx/        # Library — one module per layer
  data/                 # Pullers, splits, leakage tests           (Iter 1)
  eval/                 # Walk-forward harness, metrics, ablations (Iter 2)
  forecast/             # Chronos-2 + classical baselines          (Iter 3)
  regime/               # Contrastive encoder + clustering         (Iter 4)
  signal/               # Feature engineering + TabPFN ranking     (Iter 5)
  graph/                # Partial-corr + Granger + centrality      (Iter 6)
  agent/                # LangGraph state machine + ledger         (Iter 7)
  cli.py                # Typer entrypoint dispatching to each layer
  config.py             # Pydantic settings (paths, env, flags)
app/                    # Streamlit research cockpit (one panel per layer)
configs/                # YAML experiment configs
tests/                  # Pytest — leakage, contracts, smoke
docs/proposals/         # Original research and UI blueprints
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
| 5 | `iter-5-signal` | Regime-aware signal ranking via TabPFN |
| 6 | `iter-6-graph` | Partial-correlation + Granger cross-asset graph |
| 7 | `iter-7-agent` | LangGraph agentic research loop + persistent ledger |
| 8 | `iter-8-cockpit` | Polished cockpit with reviewer-journey navigation |
| 9 | `iter-9-report` | Consolidated report + reproducibility check |

If time runs short, later iterations are skippable — each commit on the integration branch leaves a complete, demonstrable system.

## License

MIT
