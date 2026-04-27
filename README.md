# AutoSignal-X

[![Live cockpit](https://img.shields.io/badge/cockpit-live-ff4b4b?logo=streamlit&logoColor=white)](https://ai-research-aleksander2a.streamlit.app)
[![Static snapshot](https://img.shields.io/badge/snapshot-github_pages-222?logo=github&logoColor=white)](https://aleksander2a.github.io/ai-research/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

AutoSignal-X is an **AI research system for discovering predictive structure in liquid daily ETF prices**, paired with an autonomous research loop that grades its own discoveries through a multi-stage statistical methodology before any finding is published. Five model layers (forecasting · regime representation · per-regime reasoning · cross-asset relations · agentic discovery) feed a hardening pipeline that runs every promoted finding through Diebold-Mariano + block-bootstrap → BH-FDR → adversarial replication → CPCV → PBO → Deflated Sharpe → Romano-Wolf → hierarchical Bayesian shrinkage → a strict bar that is the conjunction of every gate. The contribution is the methodology and the agent that operates it; any single trade is incidental.

Three capabilities differentiate the system:

- **Long-horizon memory across sessions.** Three complementary memory cells are written and re-read at session boundaries: `reports/agent/lessons.md` (narrative summary), `reports/agent/kg/{nodes,edges}.jsonl` (structured knowledge graph of findings / methods / regimes / assets / mechanisms with `refines` / `refutes` / `generalizes` edges), and a grounded retrieval index over the entire run corpus. Each new session reads from all three before it proposes its first hypothesis.
- **Memory cell with a human chat interface.** The cockpit's *Ask the Memory* panel runs cite-or-refuse retrieval-augmented chat against the same corpus; every claim carries a citation back to its source artifact, and off-corpus questions trigger a canonical refusal instead of a hallucination.
- **Autonomy with observability and steerability.** Every agent step is ledgered with `(round, step, content, session_id)`; every hypothesis is hash-committed to a pre-registration ledger *before* its experiment runs; the Theorist's predicted confidence is calibrated against finding-survival outcomes (Brier + Expected Calibration Error); the strict survival bar is the conjunction of every gate. The full session is reproducible without an API key via the deterministic replay mode.

The system also contributes a **synthetic-known-answer benchmark** (`autosignalx eval synthetic`) where deliberately planted causal structure is injected into a synthetic universe so the apparatus' recall and false-discovery rate are themselves audited numbers per gate, and a **capability-preserving ablation** (`autosignalx eval ablate-capability`) that reports which model layers carry marginal predictive skill versus how many bytes their precomputed forecasts cost — a concrete Pareto frontier for compression decisions.

The headline scientific question the system itself answers: **for which (regime, asset, method) combinations does a layered forecasting stack produce a lift that survives every gate?** Bundled answer at this commit: 1 of 1 promoted, 0 of 1 survives the strict bar — the apparatus correctly graded its own discovery as fragile, which is the design goal.

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
| `reports/agent/survival.jsonl` | Every promoted finding re-evaluated under BH-FDR, adversarial (full-test / placebo / block-holdout), Romano-Wolf joint stepdown (Phase 8), Combinatorial Purged CV (Phase 8), Deflated Sharpe (Phase 8), and hierarchical Bayesian (Phase 12). Strict bar `survives_all_strict` is the conjunction of every gate. |
| `reports/agent/ledger.jsonl` | Append-only record of every agent step (propose / theorist / skeptic / experiment / critique / adjudicator / decide; Phase 14 adds `verifier` / `principal_investigator` / `specialist:<role>` / `kg_writer`) |
| `reports/agent/findings.jsonl` | Promoted findings (passed the DM + bootstrap gate); idempotent on hypothesis content with replication tracking |
| `reports/agent/preregistrations.jsonl` + `preregistration_resolutions.jsonl` | Phase 8: hash-committed hypothesis ledger + outcomes (append-only) |
| `reports/agent/kg/{nodes,edges}.jsonl` | Phase 14: persistent knowledge graph (findings / methods / regimes / assets / mechanisms; relations: refines / refutes / generalizes / implements / applies_to / ...) |
| `reports/agent/red_team.jsonl` | Phase 15: per-finding asset-shuffle + time-shift adversarial attacks |
| `reports/agent/calibration.jsonl` | Phase 15: agent-confidence calibration (Brier, ECE, reliability bins) |
| `reports/agent/coherence.jsonl` | Phase 15: per-session coherence (lessons-uptake, lineage branching, theme entropy) |
| `reports/agent/prompts/<role>.jsonl` | Phase 15: per-role prompt version history (idempotent on content hash) |
| `reports/agent/eval_summary.json`, `pbo.json`, `reproducibility_badge.json` | Phase 15 / Phase 16 rollups |
| `reports/agent/holdout_vault/{vault,results}.json` | Phase 8: never-touched final test slice (lock metadata + one-time eval results) |
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
- **Agent layer** reads from all of the above; writes `reports/agent/*` artifacts. The `agent harden` step writes `reports/agent/survival.jsonl` after re-evaluating every promoted finding under FDR + adversarial replication + Phase-8 selection-bias gates (CPCV, Romano-Wolf, Deflated Sharpe) + Phase-12 Bayesian posterior. The `agent eval-suite` step (Phase 15) layers calibration, RedTeam attacks, coherence scoring, and prompt-version aggregation on top. The `agent run --mode lab` mode (Phase 14) drives a planner-routed specialist research lab (Theorist → Verifier → PrincipalInvestigator → Specialist consult → Skeptic → experiment → Adjudicator → KG-writer). **Cockpit reads everything.**

## Live demo

Two zero-install ways to see AutoSignal-X without cloning:

- **[Static snapshot](https://aleksander2a.github.io/ai-research/)** — curated 20-page HTML rendering of the cockpit's most-informative pages from the latest committed artifacts (a subset of the live cockpit's 34 panels, covering every methodology-grade artifact and the agent activity feed). No runtime, no API key required; rebuilt automatically on every push to `main` via GitHub Actions. The "always-works" option.
- **[Live cockpit](https://ai-research-aleksander2a.streamlit.app)** — full Streamlit app on Streamlit Community Cloud, defaulting to replay mode (no DeepInfra key required). Every panel is interactive against the bundled artifacts.

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

The Streamlit cockpit has **34 panels** grouped into seven sidebar sections. Every panel is a read-only viewer over a typed parquet/JSONL artifact under `reports/`; expensive computation lives in CLI commands.

### Headline (1 panel)

1. **Headline** — default-landing system overview: what AutoSignal-X is, the methodology stack, the bundled-artifact metric row (promoted findings, strict-bar survivors, synthetic-benchmark recall + FDR), the strict-bar verdict on each finding, the layer-by-layer marginal-contribution table, an ASCII pipeline diagram, and a cockpit map.

### Data & Forecasts (3 panels)

2. **Overview** — system status, layer grid, statistical-methodology summary, panel index, inputs/outputs.
3. **Data** — cache inventory; ETF and macro time series. Reads `data/cache/*.parquet`.
4. **Forecast Arena** — per-method overall metrics; per-(method, regime) stratified metrics; per-asset trajectory chart with 80% interval bands. Reads `reports/ablations/*.parquet`.

### Discovery (L2-L4) (5 panels)

5. **Regime Explorer** — KMeans + HMM regime timelines; PCA-2D scatter of contrastive embeddings coloured by regime.
6. **Signal Discovery Lab** — per-regime feature ranking from HistGradientBoosting + permutation importance; cross-regime importance heatmap.
7. **Cross-Asset Graph** — global GLASSO partial-correlation matrix; Granger edge table; NetworkX centrality table.
8. **Regime-Conditioned Graph** — Phase 6: GLASSO + Granger + centrality recomputed per regime; per-asset cross-regime sensitivity (betweenness range across regimes).
9. **Signal Stability** — Phase 6: walk-forward feature-importance rankings; per-(regime, feature) stability metrics (mean rank, rank std, top-K share); rank-trajectory chart.

### Strategy & Studies (2 panels)

10. **Backtest Arena** — Phase 1: simulated trading on the test window. Equity curves, drawdown areas, per-strategy metric table, paired block-bootstrap CI on Sharpe-difference, per-regime breakdown. Strict no-look-ahead.
11. **Custom Study** — Phase 2: form-based per-study workspace (universe / dates / splits). Pre-flight validation + pipeline buttons.

### Methodology (12 panels)

12. **Survival Analysis** — Phase 5/8/12 hardening grid for every promoted finding: BH-FDR + adversarial (full-test / placebo / block-holdout) + Romano-Wolf + Deflated Sharpe + CPCV + hierarchical Bayesian. The strict bar `survives_all_strict` is the conjunction.
13. **Bayesian Evidence** — Phase 12: hierarchical Normal-Normal posterior; per-finding posterior mean / sd, P(θ>0), Bayes factor BF₁₀, posterior predictive intervals.
14. **Synthetic Benchmark** — per-gate recall + FDR on a controlled synthetic universe with deliberately planted causal structure; the apparatus' own audited discriminative power.
15. **Capability Ablation** — Phase 16: layer-by-layer marginal contribution. Each variant adds one model layer's worth of methods to the pool the promotion pipeline can draw from; reports Mean MAE, marginal MAE-drop, and a cost proxy in bytes.
16. **Coverage Map** — Phase 14/16: hypothesis search-space heatmap of (method × asset × regime) coloured by Expected Information Gain.
17. **Statistical Power** — Phase 16: per-cell Cohen's d, power at α=0.05, sample-size required for 80% power. Distinguishes under-powered failures from genuine nulls.
18. **Counterfactual Cards** — Phase 16: per-finding factor residualization (against 5-day-diff macro factors), what-if perturbations by prediction-magnitude quartile, outlier-removal stability.
19. **Pre-Registration** — Phase 8: hash-committed hypothesis ledger. Open / resolved / promoted / refuted counts.
20. **Holdout Vault** — Phase 8: never-touched final test slice. Locked / opened state, leakage assertions, one-time evaluation results.
21. **RedTeam Attacks** — Phase 15: per-finding asset-shuffle (re-test on every other asset in the same regime) + time-shift attacks (shift `forecast_origin` by 5 days).
22. **Agent Calibration** — Phase 15: reliability diagram (predicted confidence vs observed survival rate); Brier score; Expected Calibration Error.
23. **Agent Coherence** — Phase 15: per-session lessons-uptake, lineage branching factor, theme-persistence entropy, composite coherence score.

### Agent activity (9 panels)

24. **Agent Console** — chat-style ledger timeline; per-round trace-quality chart.
25. **Specialist Council** — Phase 14: lab-mode multi-role consultation feed (Statistician / Quant / RiskOfficer / Economist / Implementer / RedTeam / Historian) + PrincipalInvestigator routing log + persistent KG explorer.
26. **Auto-Play Replay** — playback controls (play / pause / reset, 0.5× / 1× / 2× / 4× speed) over the ledger.
27. **Findings** — promoted findings (passed DM + bootstrap gate) sorted by skill-vs-naive; expandable cards with full evidence.
28. **Lineage** — Plotly DAG of hypothesis evolution across rounds, coloured by status (promoted / refuted / open).
29. **Self-Critique** — agent's verdicts on its own past findings against current evidence.
30. **Lessons & Memory** — accumulating Markdown of consolidated session notes (long-horizon memory).
31. **Telemetry** — cost / tokens / latency per LLM call; per-model and per-step breakdown; cumulative cost chart.
32. **Sessions** — per-session productivity (rounds, findings, cost-per-finding); cumulative trend across sessions.

### Reproducibility & memory (2 panels)

33. **Reproducibility** — Phase 16: git commit + dirty flag + Python env + library versions + per-artifact SHA-256 + single bundle hash for the current cockpit state.
34. **Ask the Memory** — Phase 3: grounded RAG chat over the run corpus. Cite-or-refuse system prompt; off-corpus questions trigger refusal.

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
autosignalx eval returns  [--study X]   Phase 7: returns-target ablation (zero/mean/momentum)
autosignalx eval pbo                    Phase 8: Probability of Backtest Overfitting
autosignalx eval vault-init <s> <e>     Phase 8: lock the holdout vault
autosignalx eval vault-open             Phase 8: one-time vault evaluation
autosignalx eval status   [--study X]

autosignalx regime fit              Train contrastive encoder + KMeans + HMM
autosignalx regime status

autosignalx signal rank             Per-regime feature importance via HistGradBoost
autosignalx signal stability        Walk-forward stability across rolling windows
autosignalx signal status

autosignalx graph build             Global GLASSO + Granger + centrality
autosignalx graph build-per-regime  Same machinery within each regime + sensitivity ranking
autosignalx graph status

autosignalx agent run [--mode single|debate|lab] [--max-rounds N] [--fresh] [--record-replay]
autosignalx agent score-traces      LLM-as-judge per-round quality scores
autosignalx agent consolidate       Compress session into lessons.md
autosignalx agent self-critique     Re-evaluate every promoted finding
autosignalx agent harden            Re-evaluate findings under FDR + adversarial + Phase-8 (CPCV+RW+DSR) + Phase-12 (Bayes)
autosignalx agent eval-suite        Phase 15: calibration + RedTeam + coherence + prompt scoring
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
  eval/                  Forecast contract + Phase 7 target-type adapters; harness; metrics;
                         DM/bootstrap significance; FDR + adversarial replication;
                         Phase 8 CPCV / PBO / Deflated Sharpe / Romano-Wolf /
                         pre-registration / holdout vault; Phase 12 hierarchical Bayesian;
                         Phase 16 counterfactual lenses + statistical-power dashboard;
                         survival aggregator
  forecast/              Baselines + Chronos-2 + Phase 7 returns-target baselines
  regime/                Contrastive encoder, KMeans, HMM, market features
  signal/                Feature engineering, per-regime ranking, walk-forward stability
  graph/                 GLASSO, Granger, centrality, global + regime-conditioned build
  backtest/              Phase 1 vectorised engine, strategies, paired-bootstrap significance
  study/                 Phase 2 user-defined studies (universe / dates / splits)
  chat/                  Phase 3 grounded RAG over the run corpus
  snapshot/              Phase 4 static-HTML snapshot generator
  agent/                 State machine + debate + tools + ledger + findings + lineage +
                         memory + telemetry + sessions + trace_eval + self_critique +
                         specs + codegen +
                         Phase 14: specialists / lab / verifier / knowledge_graph / eig +
                         Phase 15: calibration / red_team / coherence /
                         prompt_optimizer / eval_suite
  reproducibility.py     Phase 16: git+env+artifact-hash bundle
  cli.py                 Typer entrypoint (registers every layer's sub-app)
  config.py              Pydantic settings, .env loading, YAML config reader
app/                     Streamlit cockpit (34 panels in 7 sections)
configs/                 YAML experiment configs
tests/                   342 pytest tests (74 net new across Phases 7/8/12/14/15/16 plus the apparatus-capability suite)
docs/ARCHITECTURE.md     Implementation reference
docs/roadmap/            ROADMAP.md (Phase-by-phase plan and status; Phases 1-8, 12, 14-16 shipped)
data/                    (gitignored cache; data/studies/<name>/ for per-study workspaces)
reports/                 Persisted artifacts (every layer writes here; cockpit reads here)
replay/                  Recorded LLM responses; lets the cockpit run end-to-end without an API key
scripts/                 run_session.sh / .ps1 for cron / Task Scheduler
REPORT.md                Research questions, methodology, results, limitations, future work
```

## Documentation

- **README.md** (this file) — system overview, inputs / outputs, the panel index, the CLI surface, and the repository layout.
- **[TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md)** — one-page overview describing what the system is, why it's designed this way, how the layers fit together, the methodology stack at a glance, and a five-minute path through the cockpit. A good starting point if you want a single page.
- **[REPORT.md](REPORT.md)** — research questions, methodology in detail, per-phase results, the apparatus-capability evaluation (synthetic benchmark + capability ablation), limitations, and future work.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — implementation reference: data flow, layer artifact contracts, per-layer module index, agent loop, sandbox model.
- **[docs/roadmap/ROADMAP.md](docs/roadmap/ROADMAP.md)** — phase-by-phase plan and shipping status (Phases 1-8, 12, 14-16 shipped).

## License

MIT
