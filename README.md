# AutoSignal-X

[![Live cockpit](https://img.shields.io/badge/cockpit-live-ff4b4b?logo=streamlit&logoColor=white)](https://autosignal-x.streamlit.app)
[![Static snapshot](https://img.shields.io/badge/snapshot-github_pages-222?logo=github&logoColor=white)](https://aleksander2a.github.io/ai-research/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A 5-layer modular AI research system for discovering predictive structure in liquid daily ETF prices, paired with a multi-agent research loop that proposes hypotheses, designs and runs experiments, evaluates them under statistical rigor, and persists findings with full provenance.

The system answers one question: **for which (regime, asset, method) combinations does a layered forecasting stack outperform the naive baseline on daily ETF prices, and is that outperformance statistically significant under Diebold–Mariano + a positive bootstrap CI on the loss difference?**

## Inputs and outputs

### Inputs

| Source | What | Required | When |
|---|---|---|---|
| yfinance API | 8 ETFs (SPY, QQQ, IWM, GLD, TLT, EFA, EEM, HYG) and 4 macro signals (^TNX, ^VIX, DX-Y.NYB, CL=F), daily 2010-01-01 → 2025-12-31 | Yes | At `make data` (one-time per refresh; cached as parquet) |
| DeepInfra API key | OpenAI-compatible LLM endpoint for the agent layer | No (replay mode works without it) | Per agent run |
| `replay/agent_steps.jsonl` | Pre-recorded LLM responses keyed by `(round, step)` | No | Auto-used when no DeepInfra key is set |
| `configs/default.yaml` | Date splits, horizon, regime / signal / agent hyperparameters | Yes | Read by every CLI subcommand |

### Outputs (all persisted under `reports/` and committed to the repo)

| Artifact | What |
|---|---|
| `reports/ablations/*.parquet` | One file per forecasting method; long-format forecasts with the full forecast contract (target, prediction, intervals, origin, etc.) |
| `reports/regimes/{kmeans,hmm,embeddings}.parquet` | Per-timestep regime labels from two detectors; raw contrastive embeddings |
| `reports/signals/signal_ranking.parquet` | Per-(regime, feature) importance rankings |
| `reports/signals/walk_forward_ranking.parquet` | Per-(window, regime, feature) walk-forward importance with rank |
| `reports/signals/signal_stability.parquet` | Per-(regime, feature) stability summary: mean rank, rank std, top-K share, stability score |
| `reports/graph/{edges,centrality}.parquet` | Global cross-asset partial-correlation + Granger edges; per-node centrality |
| `reports/graph/per_regime/regime_<id>/{edges,centrality}.parquet` | Same machinery rebuilt within each regime's data subset |
| `reports/graph/per_regime/regime_sensitivity.parquet` | Per-asset cross-regime centrality dispersion (max - min betweenness, etc.) |
| `reports/agent/survival.jsonl` | Every promoted finding re-evaluated under BH-FDR + adversarial replication (full-test, placebo, block-holdout) |
| `reports/agent/ledger.jsonl` | Append-only record of every agent step (propose / theorist / skeptic / experiment / critique / adjudicator / decide) |
| `reports/agent/findings.jsonl` | Promoted findings (passed the DM + bootstrap gate); idempotent on hypothesis content with replication tracking |
| `reports/agent/lessons.md` | Markdown summary appended per session, used as long-horizon memory for the next session |
| `reports/agent/telemetry.jsonl` | Per-LLM-call cost, tokens, latency |
| `reports/agent/trace_quality.jsonl` | Per-round LLM-as-judge scores on clarity / novelty / falsifiability / evidence-citing |
| `reports/agent/self_critique.jsonl` | Agent's verdict on its own past findings against current evidence |
| `reports/agent/generated_methods/` | Sandboxed Python code authored by the agent at runtime |
| `replay/agent_steps.jsonl` | Recorded LLM responses (written when `--record-replay` is set; consumed in replay mode) |

### Information flow between layers

Each layer reads from prior layers exclusively through typed parquet/JSONL artifacts (no in-memory shared state). The diagram is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#data-flow); contracts are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#layer-interfaces-artifacts).

Summary:

- **Data layer** writes `data/cache/{ohlcv,macro}.parquet`. **All other layers read from these.**
- **Forecast layer** writes `reports/ablations/<method>.parquet`. **Eval, agent, cockpit read these.**
- **Regime layer** writes `reports/regimes/{kmeans,hmm,embeddings}.parquet`. **Signal layer joins forecast_origin → regime_id; agent reads through `tools.context_snapshot()`; cockpit reads.**
- **Signal layer** writes `reports/signals/signal_ranking.parquet`. **Agent reads through `tools.get_top_features(regime_id)`; cockpit reads.**
- **Graph layer** writes `reports/graph/{edges,centrality}.parquet` (global) plus `reports/graph/per_regime/regime_<id>/...` + `regime_sensitivity.parquet` (regime-conditioned). **Agent reads through `tools.get_centrality_summary()`; cockpit reads.**
- **Agent layer** reads from all of the above; writes `reports/agent/*` artifacts. The `agent harden` step writes `reports/agent/survival.jsonl` after re-evaluating every promoted finding under FDR + adversarial replication. **Cockpit reads everything.**

## Live demo

Two zero-install ways to see AutoSignal-X without cloning:

- **[Static snapshot](https://aleksander2a.github.io/ai-research/)** — multi-page HTML rendering of every cockpit panel from the latest committed artifacts. No runtime; rebuilt automatically on every push to `main` via GitHub Actions. The "always-works" option.
- **[Live cockpit](https://autosignal-x.streamlit.app)** — full Streamlit app on Streamlit Community Cloud, defaulting to replay mode (no DeepInfra key required). Every panel is interactive against the bundled artifacts.

Both deployments run from this same repo on the same branch.

## Quick start

```bash
git clone https://github.com/Aleksander2a/ai-research.git
cd ai-research
uv sync --all-extras
make demo                      # or: uv run streamlit run app/streamlit_app.py
```

The cockpit opens at `http://localhost:8501`. Every cockpit panel renders out-of-the-box because all artifacts are committed; a fresh clone shows real results without running anything. To regenerate any layer's artifacts, run the relevant CLI / Make target (table below).
Always launch the cockpit from the repo-local uv environment. Launching Streamlit from a stale global or Conda install can import an older `autosignalx` package and break study-aware features.

## Architecture

Five model layers plus an agent loop:

| Layer | Purpose | Implementation |
|---|---|---|
| **L1 Forecasting** | Probabilistic point + interval forecasts | Frozen Chronos-2 (multivariate, with `past_covariates`) and three classical baselines (naive, seasonal-naive, ARIMA(1,1,1) on log-prices) |
| **L2 Representation** | Per-timestep regime labels | Contrastive 1D-CNN encoder (16-dim embeddings, 60-day windows, triplet loss) + KMeans on embeddings; Gaussian HMM on raw features as a parallel detector |
| **L3 Reasoning** | Per-regime feature importance | `HistGradientBoostingClassifier` per regime + custom permutation importance; **walk-forward stability** layer (rolling-window refits with rank-stability metrics) |
| **L4 Relational** | Cross-asset dependency structure | GLASSO partial correlations (`GraphicalLassoCV`) + Granger causality (statsmodels) + NetworkX centrality (degree / eigenvector / betweenness); **regime-conditioned variant** rebuilds the graph within each regime's data subset and emits per-asset cross-regime centrality dispersion |
| **L5 Agentic** | Hypothesis generation, experimentation, statistical promotion | LangGraph state machine (debate mode: Theorist / Skeptic / Adjudicator with three different DeepInfra LLMs); experiment surface includes slicing cached forecasts, authoring methods via a constrained DSL, and executing sandboxed Python forecast functions |

The agent has three escalating ways to author experiments:

1. `slice_forecasts(method, asset, regime_id)` — measure on cached data.
2. `spawn_method(spec)` — author a new method via a JSON DSL (compose primitives: base method + covariate subset + naive ensembling + asset/window filters).
3. `spawn_method_code(spec)` — execute sandboxed Python `forecast_fn`, AST-validated, run in restricted globals.

Auto-promotion: every experiment naming a non-naive method automatically runs the DM + bootstrap gate against naive on the same slice; if it passes (`p < 0.05`, skill > 0, bootstrap CI strictly above zero), the finding is appended to `reports/agent/findings.jsonl` with full provenance.

For the data flow diagram, contract schemas, and per-layer wiring, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). For methodology and results, see [REPORT.md](REPORT.md).

## Cockpit panels

The Streamlit cockpit has 19 panels in the sidebar. Walk left-to-right for the full reviewer journey:

1. **Overview** — system status, headline finding, layer grid.
2. **Data** — cache inventory; ETF and macro time series.
3. **Forecast Arena** — per-method overall metrics; per-(method, regime) stratified metrics; per-asset trajectory chart with 80% interval bands.
4. **Regime Explorer** — KMeans + HMM regime timelines; PCA-2D scatter of contrastive embeddings colored by regime.
5. **Signal Discovery Lab** — per-regime feature ranking; cross-regime importance heatmap.
6. **Cross-Asset Graph** — global partial-correlation matrix; Granger edge table; centrality table.
7. **Regime-Conditioned Graph** — same machinery rebuilt per regime; surfaces hubs and bridges that flip role across regimes plus a per-asset regime-sensitivity table (betweenness range across regimes).
8. **Signal Stability** — walk-forward feature-importance rankings; per-(regime, feature) stability metrics (mean rank, rank std, top-K share, composite stability); rank-trajectory chart across windows.
9. **Backtest Arena** — simulated trading on the test window driven by discovered structure (Phase 1). Equity curves, drawdown areas, per-strategy metric table, paired block-bootstrap CI on Sharpe-difference vs benchmark, per-regime metric breakdown. Strict no-look-ahead (backtest start > discovery end). Reads `reports/backtest/runs/<run_id>/`.
10. **Custom Study** — run AutoSignal-X on your own asset universe and date range (Phase 2). Form-based study creation, pre-flight validation (date ordering, walk-forward window count, optional yfinance availability probe), pipeline buttons (data fetch / baseline eval / backtest) with synchronous spinners, and copyable CLI commands for heavy steps. Each study has its own `data/studies/<name>/cache/` and `reports/studies/<name>/` tree.
11. **Agent Console** — chat-style ledger timeline; per-round trace-quality chart.
12. **Auto-Play Replay** — playback controls (play/pause/reset, 0.5x / 1x / 2x / 4x speed) over the ledger.
13. **Findings** — promoted findings sorted by skill-vs-naive; expandable cards with full DM/bootstrap evidence.
14. **Lineage** — Plotly DAG of hypothesis evolution across rounds, colored by status.
15. **Self-Critique** — agent's verdicts on its own past findings against current evidence.
16. **Survival Analysis** — every promoted finding re-evaluated under BH-FDR + adversarial replication (full-test, placebo regime-shuffle, 50/50 block-holdout). Pass/fail grid with full per-attack evidence in expandable cards. The methodology-defence panel: a green check on every column means the finding is robust; a red X is a research insight, not a failure.
17. **Lessons & Memory** — accumulating Markdown of consolidated session notes (long-horizon memory).
18. **Telemetry** — cost / tokens / latency per LLM call; per-model and per-step breakdown; cumulative cost chart.
19. **Sessions** — per-session productivity (rounds, findings, cost-per-finding); cumulative trend across sessions.
20. **Ask the Memory** — grounded RAG chat over the run corpus (ledger, findings, lessons, trace quality, self-critique, telemetry, backtests, survival, regime-graph, signal-stability). Every claim carries a citation back to the source artifact; off-corpus questions are refused. Live mode uses DeepInfra `bge-large-en-v1.5` embeddings; replay/no-key mode uses deterministic hashed-bag embeddings.

A **Study scope** selector in the sidebar switches study-aware panels (Forecast Arena, Backtest Arena) to read from a chosen study's tree; the default scope reads the project's canonical artifacts.

## CLI and Make targets

```
autosignalx version
autosignalx status                  Layer status, data cache, ablation files

autosignalx study create            Create a new study (custom universe / dates)
autosignalx study list / show       Inspect existing studies
autosignalx study validate <name>   Pre-flight checks (dates, windows, optional ticker probe)
autosignalx study delete <name>     Remove a study and its artifacts

autosignalx data fetch [--study X]  Pull ETF + macro from yfinance (per study or default)
autosignalx data status [--study X]

autosignalx eval baseline [--study X]   Run naive + seasonal_naive + arima ablation
autosignalx eval chronos  [--study X]   Run chronos2_univariate + chronos2_multivariate
autosignalx eval status   [--study X]

autosignalx regime fit              Train contrastive encoder + KMeans + HMM
autosignalx regime status

autosignalx signal rank             Per-regime feature importance via HistGradBoost
autosignalx signal stability        Walk-forward stability across rolling windows
autosignalx signal status

autosignalx graph build             Global GLASSO + Granger + centrality
autosignalx graph build-per-regime  Same machinery within each regime + sensitivity ranking
autosignalx graph status

autosignalx agent run [--mode single|debate] [--max-rounds N] [--fresh] [--record-replay]
autosignalx agent score-traces      LLM-as-judge per-round quality scores
autosignalx agent consolidate       Compress session into lessons.md
autosignalx agent self-critique     Re-evaluate every promoted finding
autosignalx agent harden            Re-evaluate findings under BH-FDR + adversarial (full-test/placebo/block-holdout)
autosignalx agent status

autosignalx backtest run [--study X] [--strategies "..."] [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--cost-bps N]
autosignalx backtest status [--study X]

autosignalx chat index [--force-hashed]   Build the RAG index over current artifacts
autosignalx chat status                   Inventory of indexed chunks by kind
autosignalx chat ask "<question>"         One-shot grounded Q&A
autosignalx chat eval                     Run the bundled grounding eval set

autosignalx snapshot build                Render static HTML cockpit to reports/cockpit_snapshot/
```

Make targets wrap each command (`make data`, `make baseline`, `make forecast`, `make regime`, `make signal`, `make graph`, `make agent`, `make scheduled-session`), plus `make sync`, `make test`, `make lint`, `make demo`, `make clean`.

## LLM provider

The agent layer uses [DeepInfra](https://deepinfra.com/) (OpenAI-compatible endpoint at `https://api.deepinfra.com/v1/openai`) via `langchain_openai.ChatOpenAI`. Three roles map to three env-configurable models:

```bash
DEEPINFRA_API_KEY=<your key>
DEEPINFRA_MODEL_THEORIST=moonshotai/Kimi-K2.6
DEEPINFRA_MODEL_SKEPTIC=zai-org/GLM-5.1
DEEPINFRA_MODEL_ADJUDICATOR=deepseek-ai/DeepSeek-V4-Pro
```

Per-call responses are content-hash cached on disk (`reports/agent/llm_cache/`); re-runs of the same prompt are free and deterministic. Live calls record `(model, role, step, round, prompt_tokens, completion_tokens, latency_ms, cost_usd, session_id)` to `reports/agent/telemetry.jsonl`. Cost defaults are in `agent/telemetry.py`; override per-model via `DEEPINFRA_PRICE_<MODEL>_IN` / `_OUT` env vars (USD per 1M tokens).

**Without a DeepInfra key** (or with `AUTOSIGNALX_REPLAY=true`), the agent runs in deterministic replay mode against `replay/agent_steps.jsonl`; the recorded session is committed to the repo so reviewers see the same trace without provisioning anything.

## Custom studies (Phase 2)

Run AutoSignal-X on your own assets and date range without editing config files. A `Study` is a named, isolated workspace: its own data cache (`data/studies/<name>/cache/`), its own ablations (`reports/studies/<name>/ablations/`), its own backtest runs (`reports/studies/<name>/backtest/runs/`).

```
autosignalx study create --name tech_megacap \
  --assets "AAPL,MSFT,NVDA,GOOG,META,AMZN" \
  --start 2018-01-01 --end 2024-12-31 \
  --train-end 2022-06-30 --val-end 2023-12-31 --test-end 2024-12-31

autosignalx study validate tech_megacap          # date ordering, window count, universe size
autosignalx study validate tech_megacap --check-tickers   # also probe yfinance availability

autosignalx data fetch    --study tech_megacap   # pull from yfinance into the study cache
autosignalx eval baseline --study tech_megacap   # naive + seasonal_naive + arima
autosignalx eval chronos  --study tech_megacap   # Chronos-2 forecasts (heavy)
autosignalx backtest run  --study tech_megacap   # simulated trading on the test window
```

The cockpit's **Custom Study** panel exposes the same flow form-based (create / validate / fetch / baseline / backtest) for users who prefer not to touch the terminal. The sidebar **Study scope** selector switches Forecast Arena and Backtest Arena to read from a chosen study's tree; defaults to the project's canonical artifacts.
For the cockpit path, use the repo-local launch commands above (`uv sync --all-extras`, then `uv run streamlit run app/streamlit_app.py`) so the app and `autosignalx` package stay in sync.

Default behaviour (no `--study`) is unchanged across every CLI subcommand, so studies are strictly additive.

## Scheduled execution

`scripts/run_session.sh` (bash, cron) and `scripts/run_session.ps1` (PowerShell, Windows Task Scheduler) wrap a full session: `agent run --mode debate --record-replay` → `agent score-traces` → `agent consolidate`. Configurable via `AUTOSIGNALX_ROUNDS` and `AUTOSIGNALX_MODE` env vars. Cron example baked into the script's docstring.

Cross-session aggregation in `agent/sessions.py` produces per-session and cumulative productivity views in the Sessions panel.

## Repository layout

```
src/autosignalx/         Library (one module per concern)
  data/                  yfinance pulls, parquet cache, walk-forward splits
  eval/                  Forecast contract, harness, metrics, DM/bootstrap significance
  forecast/              Baselines and Chronos-2
  regime/                Contrastive encoder, KMeans, HMM, market features
  signal/                Feature engineering, per-regime ranking
  graph/                 GLASSO, Granger, centrality, build orchestration
  agent/                 State machine, debate, tools, ledger, findings, lineage,
                         memory, telemetry, sessions, trace_eval, self_critique,
                         specs, codegen
  cli.py                 Typer entrypoint (registers every layer's sub-app)
  config.py              Pydantic settings, .env loading, YAML config reader
app/                     Streamlit cockpit (15 panels)
configs/                 YAML experiment configs
tests/                   146 pytest tests
docs/ARCHITECTURE.md     Implementation reference
data/                    (gitignored cache, reproducible from `make data`)
reports/                 Persisted artifacts (forecasts, regimes, signals, graph, agent)
replay/                  Recorded LLM responses for no-key reviewer mode
scripts/                 run_session.sh / .ps1 for cron / Task Scheduler
REPORT.md                Research questions, methodology, results
```

## Documentation

- **README.md** (this file) — system overview, inputs/outputs, panels, CLI, repository layout.
- **[REPORT.md](REPORT.md)** — research questions, methodology, results, limitations.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — implementation: data flow, contracts, per-layer wiring, agent loop, sandbox model.

## License

MIT
