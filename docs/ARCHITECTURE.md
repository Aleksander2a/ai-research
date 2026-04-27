# AutoSignal-X — Architecture

Implementation reference: data flow, contracts, per-layer wiring, agent loop, sandbox model. For research questions and results see [REPORT.md](../REPORT.md); for project framing see [README.md](../README.md).

## Design principle

Each layer is independent, persists its outputs as typed parquet/JSONL artifacts under `reports/`, and reads from prior layers exclusively through those artifacts. There is no in-memory shared state between layers, and no implicit global. The cockpit and the agent are read-only consumers of the same artifacts.

Three consequences:

1. Any layer can be re-run, skipped, or replaced without touching the others.
2. The cockpit and the agent show only what is reproducible from `reports/` on disk.
3. The forecast DataFrame schema (`autosignalx.eval.contracts.FORECAST_COLUMNS_REQUIRED`) is the **load-bearing contract**. Every forecasting method satisfies it, including agent-authored methods.

## Inputs and outputs

### System inputs

| Input | Origin | Required | Persisted to |
|---|---|---|---|
| ETF OHLCV (8 tickers) | yfinance API | Yes | `data/cache/ohlcv.parquet` |
| Macro signals (4 signals) | yfinance API | Yes | `data/cache/macro.parquet` |
| Experiment hyperparameters | `configs/default.yaml` | Yes (read by every CLI) | n/a |
| DeepInfra API key | `.env` (`DEEPINFRA_API_KEY`) | No (replay mode works without) | n/a |
| Recorded LLM responses | `replay/agent_steps.jsonl` | No (auto-used in replay mode) | n/a |

### System outputs (everything persisted under `reports/`)

| Output | Path | Producer | Schema (key columns) |
|---|---|---|---|
| Per-method forecasts | `reports/ablations/<method>.parquet` | `eval/cli.py` (baseline / chronos), `agent.specs.execute`, `agent.codegen.execute_code_spec` | `(timestamp, asset, forecast_origin, horizon, method, prediction, origin_value, target, lower?, upper?)` |
| KMeans regime labels | `reports/regimes/kmeans.parquet` | `regime/cli.py` | `(timestamp, regime_id int, method "kmeans_contrastive")` |
| HMM regime labels | `reports/regimes/hmm.parquet` | `regime/cli.py` | `(timestamp, regime_id int, method "hmm_gaussian")` |
| Contrastive embeddings | `reports/regimes/embeddings.parquet` | `regime/cli.py` | `(timestamp, c0..c15)` |
| Per-regime feature ranking | `reports/signals/signal_ranking.parquet` | `signal/cli.py` | `(regime_id, feature, importance, importance_std, n_samples, rank)` |
| Cross-asset graph edges | `reports/graph/edges.parquet` | `graph/cli.py` | `(source, target, edge_type, weight, p_value?, best_lag?)` |
| Centrality | `reports/graph/centrality.parquet` | `graph/cli.py` | `(node, degree_centrality, eigenvector_centrality, betweenness_centrality)` |
| Agent ledger | `reports/agent/ledger.jsonl` | `agent.graph.run`, `agent.debate.run_debate` | `(round, step, content, ts, session_id)` |
| Promoted findings | `reports/agent/findings.jsonl` | `agent/graph.py:experiment_node` (auto-promotion) | `(id, hypothesis, method, filters, evidence, agent_confidence, round, session_id, promoted_at, parent_hypothesis_ids, replication_count, replications)` |
| Lessons (long-horizon memory) | `reports/agent/lessons.md` | `agent/cli.py:consolidate_cmd` | Markdown sections per session |
| Telemetry | `reports/agent/telemetry.jsonl` | `agent/llm.py:LiveProvider.chat` | `(ts, model, role, step, round, prompt_tokens, completion_tokens, latency_ms, cost_usd, session_id)` |
| Trace quality | `reports/agent/trace_quality.jsonl` | `agent/cli.py:score_traces_cmd` | `(round, clarity, novelty, falsifiability, evidence_citing, rationale, ts, session_id)` |
| Self-critique | `reports/agent/self_critique.jsonl` | `agent/cli.py:self_critique_cmd` | `(finding_id, current_state, rationale, ts)` |
| Generated code | `reports/agent/generated_methods/<name>.{py,json}` | `agent/codegen.py:execute_code_spec` | Python source + metadata JSON |
| Recorded LLM trace | `replay/agent_steps.jsonl` | `agent/llm.py:LiveProvider.chat` (when `record_replay=True`) | `(round, step, content)` per call |

### Information passed between layers

```mermaid
flowchart TD
  YF["yfinance API"] --> DC["data/cache/{ohlcv,macro}.parquet"]
  DC --> EH["eval harness (walk-forward)"]
  DC --> RFP["regime feature pipeline"]
  DC --> SFP["signal feature engine"]
  DC --> GFP["graph build"]

  EH --> BL["forecast/baselines.py"]
  EH --> CH["forecast/chronos2.py"]
  BL --> AB["reports/ablations/baseline.parquet"]
  CH --> AB2["reports/ablations/chronos2.parquet"]

  RFP --> RE["regime/encoder.py + cluster.py"]
  RE --> RL["reports/regimes/{kmeans,hmm,embeddings}.parquet"]

  SFP --> SR["signal/ranking.py"]
  RL -.regime labels.-> SR
  SR --> SS["reports/signals/signal_ranking.parquet"]

  GFP --> GLA["graph/correlation.py + causality.py + centrality.py"]
  GLA --> GE["reports/graph/{edges,centrality}.parquet"]

  AB --> AG["agent/tools.py<br/>(slice / test / spawn / snapshot)"]
  AB2 --> AG
  RL --> AG
  SS --> AG
  GE --> AG

  AG --> SM["agent/graph.py + debate.py<br/>(LangGraph state machine)"]
  LLM["DeepInfra LLM<br/>(or replay/agent_steps.jsonl)"] --> SM
  SM --> AGW["reports/agent/{ledger,findings,lessons,<br/>telemetry,trace_quality,self_critique}.jsonl"]
  SM --> AGENB["reports/ablations/<agent-authored>.parquet<br/>(spawn_method / spawn_method_code)"]

  AB --> CK["app/streamlit_app.py<br/>(34 cockpit panels in 7 sections)"]
  AB2 --> CK
  RL --> CK
  SS --> CK
  GE --> CK
  AGW --> CK
  AGENB --> CK
```

**Per-edge contract:** every arrow above represents a typed parquet/JSONL read; no direct in-process passing of Python objects between layers.

## Layer interfaces (artifact schemas)

### `data/cache/ohlcv.parquet`

Long format. Required columns asserted by `data/schema.py:assert_ohlcv_schema`:

```
timestamp           datetime64[ns]   trading day (no tz)
asset               string           ticker (e.g., SPY)
open / high / low / close   float64
adj_close           float64          forecast target
volume              float64
returns             float64          adj_close.pct_change()
```

Per-asset timestamps strictly monotonic increasing.

### `data/cache/macro.parquet`

```
timestamp     datetime64[ns]
signal        string           ^TNX | ^VIX | DX-Y.NYB | CL=F
value         float64
```

Per-signal timestamps strictly monotonic increasing.

### `reports/ablations/<method>.parquet` (forecast contract)

Required columns enforced by `eval/contracts.py:assert_forecast_schema`:

```
timestamp           datetime64[ns]   target trading day
asset               string
forecast_origin     datetime64[ns]   < timestamp (no leakage)
horizon             int              days from origin to target
method              string           e.g., naive, chronos2_multivariate
prediction          float64          adj_close-units
origin_value        float64          adj_close at forecast_origin
target              float64          realized adj_close
lower (optional)    float64          10% quantile
upper (optional)    float64          90% quantile
regime_id (optional) int             populated by regime.add_regime_to_forecasts
target_type (optional) string        Phase 7: price | log_return | excess_return | vol | rank
                                     -- defaults to price for backward compatibility;
                                     read by eval.contracts.get_target_type
```

### `reports/regimes/`

```
kmeans.parquet      (timestamp, regime_id int, method="kmeans_contrastive")
hmm.parquet         (timestamp, regime_id int, method="hmm_gaussian")
embeddings.parquet  (timestamp, c0..c15)   contrastive embedding per window
```

Window-aligned: each `kmeans.parquet` row's `regime_id` labels the timestamp at the end of a 60-day window.

### `reports/signals/`

```
signal_ranking.parquet              Single-fit per-regime ranking
  regime_id        int
  feature          string
  importance       float    base_acc - mean shuffled_acc, n_repeats=2
  importance_std   float
  n_samples        int
  rank             int      1 = most important within the regime

walk_forward_ranking.parquet        Per-window per-regime ranking (Phase 6B)
  window_idx       int
  window_start     timestamp
  window_end       timestamp
  regime_id        int
  feature          string
  importance       float
  importance_std   float
  rank             int
  n_samples        int

signal_stability.parquet            Per-(regime, feature) summary
  regime_id        int
  feature          string
  mean_importance  float
  std_importance   float
  mean_rank        float
  rank_std         float
  n_windows        int
  topK_share       float    fraction of windows feature was rank<=K (default K=5)
  stability        float    1 - (rank_std / max_rank), clipped to [0, 1]
```

### `reports/graph/`

```
edges.parquet                       Global cross-asset graph (over full history)
  source        string   ticker
  target        string   ticker
  edge_type     string   "partial_corr" | "granger"
  weight        float64  partial corr in [-1, 1] | -log10(p) for Granger
  p_value       float64  (Granger only)
  best_lag      int      (Granger only)

centrality.parquet                  Global centrality
  node                       string
  degree_centrality          float64
  eigenvector_centrality     float64
  betweenness_centrality     float64

per_regime/regime_<id>/edges.parquet         Same machinery within regime <id>
per_regime/regime_<id>/centrality.parquet    Adds regime_id, n_samples columns

per_regime/centrality_by_regime.parquet      Long-format union of all per-regime centrality
per_regime/regime_sensitivity.parquet        Cross-regime dispersion per asset
  node                                         string
  degree_centrality_{mean,std,min,max}         float64
  eigenvector_centrality_{mean,std,min,max}    float64
  betweenness_centrality_{mean,std,min,max}    float64
  betweenness_centrality_range                 float64   max - min  (= "regime sensitivity")
```

### `reports/agent/`

```
ledger.jsonl            One JSON per step
                        (round, step, content, ts, session_id)
                        step ∈ {propose, theorist, skeptic, experiment,
                                critique, adjudicator, decide}

findings.jsonl          Promoted hypotheses (passed DM + bootstrap gate)
                        (id, hypothesis, method, filters, evidence,
                         agent_confidence, round, session_id, promoted_at,
                         parent_hypothesis_ids, replication_count, replications)

lessons.md              Markdown sections per session, --- separated
                        Used as long-horizon memory by next session

telemetry.jsonl         (ts, model, role, step, round,
                         prompt_tokens, completion_tokens, total_tokens,
                         latency_ms, cost_usd, session_id)

trace_quality.jsonl     (round, clarity, novelty, falsifiability,
                         evidence_citing, rationale, ts, session_id)

self_critique.jsonl     (finding_id, current_state, rationale, ts)
                        current_state ∈ {reinforced, unchanged,
                                          weakened, refuted}

survival.jsonl          Phase-5/8/12 hardening output, one row per finding
                        (finding_id, hypothesis, method, filters,
                         original_p, original_skill, fdr_alpha, fdr_q,
                         survives_fdr,
                         rw_q, survives_rw,                    # Phase 8
                         cpcv: {n_paths, skill_mean, skill_std,
                                skill_min, skill_max, skill_median, skills}, # Phase 8
                         deflated_sharpe: {sr_observed, sr_threshold_null,
                                           dsr, n_trials, n_observations}, # Phase 8
                         survives_dsr,
                         adversarial: {full_test, placebo,
                         block_holdout, survives_adversarial},
                         survives_full_test, survives_placebo,
                         survives_block_holdout, survives_all,
                         survives_all_strict,                  # Phase 8
                         bayesian: {posterior_mean, posterior_sd,
                                    prob_positive, bayes_factor, n}, # Phase 12
                         survives_bayes,                       # Phase 12
                         evaluated_at)

preregistrations.jsonl  Phase 8: hash-committed hypotheses
                        (id, hypothesis, method, baseline, filters,
                         decision_rule, predicted_effect, falsifier,
                         registered_at, session_id, round, proposer_role)

preregistration_resolutions.jsonl   Phase 8: outcomes
                        (preregistration_id, promoted, evidence,
                         notes, resolved_at)

red_team.jsonl          Phase 15: per-finding asset-shuffle + time-shift attacks
                        (finding_id, asset_shuffle, time_shift,
                         survives_red_team, evaluated_at)

calibration.jsonl       Phase 15: confidence-vs-survival reliability
                        (role, n, brier, ece, bins, evaluated_at)

coherence.jsonl         Phase 15: per-session coherence record
                        (session_id, n_rounds, lessons_uptake,
                         lineage_branching_factor,
                         theme_persistence_entropy,
                         coherence_score, evaluated_at)

prompts/<role>.jsonl    Phase 15: per-role prompt version history
                        (role, version_id, text, registered_at)

prompt_scores.json      Phase 15: aggregated quality across versions

eval_summary.json       Phase 15: rollup of all eval-suite outputs

kg/nodes.jsonl          Phase 14: persistent KG nodes
                        (id, kind, label, attrs, ts)

kg/edges.jsonl          Phase 14: persistent KG edges
                        (id, source, target, relation, weight, attrs, ts)

holdout_vault/vault.json   Phase 8: lock metadata
holdout_vault/results.json Phase 8: one-time vault-open headline numbers

pbo.json                Phase 8: PBO summary (autosignalx eval pbo)
reproducibility_badge.json  Phase 16: git+env+artifact-hashes bundle

generated_methods/      <name>.py + <name>.json
                        Sandboxed code authored at runtime + metadata

llm_cache/              content-hash-keyed plaintext (gitignored)
embed_cache/            content-hash-keyed embedding vectors (Phase 3)
```

### `replay/agent_steps.jsonl`

Pre-recorded LLM responses, one JSON per line:

```
(round int, step string, content string)
```

Used in replay mode (no API key) to walk through a recorded session deterministically.

## Layer-by-layer

### `data/`

Pulls from yfinance, normalizes to long format, persists parquet, defines walk-forward and static splits.

| File | Purpose |
|---|---|
| `schema.py` | Column contracts and `assert_*` validators |
| `cache.py` | Parquet read/write at the persistence boundary; `cache_status()` inventory |
| `fetch.py` | yfinance pulls; normalizes MultiIndex columns from yfinance |
| `splits.py` | `WalkForwardWindow` (rejects `train_end ≥ forecast_start` at construction); `StaticSplit`; `walk_forward_windows(val_end, test_end, horizon_days, step_days)` |
| `loader.py` | Wide-format pivots: `load_returns_wide()`, `load_close_wide()`, `load_macro_wide()` |
| `cli.py` | `autosignalx data fetch` / `status` |

### `eval/`

| File | Purpose |
|---|---|
| `contracts.py` | `FORECAST_COLUMNS_REQUIRED`, `assert_forecast_schema` (leakage check, non-negative horizons) |
| `metrics.py` | `mae`, `mape`, `directional_accuracy`, `skill_score`, `crps_from_quantiles` |
| `harness.py` | `run_walk_forward(method_name, forecast_fn, ohlcv, windows)`, `ablation`, `summarize`, `add_skill_score` |
| `significance.py` | `dm_test(loss_a, loss_b, horizon)` with Newey–West HAC variance; `block_bootstrap_ci`; `is_promotable` (DM + skill + bootstrap gate) |
| `fdr.py` | `benjamini_hochberg(p_values, alpha)` -- step-up FDR with monotone q-values; returns per-finding adjusted p-values + boolean survival mask |
| `adversarial.py` | `replicate_full_test`, `replicate_placebo` (shuffled regime labels), `replicate_block_holdout` (50/50 by `forecast_origin`); `adversarial_replication` bundles all three with a `survives_adversarial` rollup |
| `survival.py` | `harden_findings(findings_path, fdr_alpha)` -- joins FDR + adversarial into per-finding rows, persists `reports/agent/survival.jsonl`; `load_survival` reader |
| `targets.py` | Phase 7: target-type adapters -- `to_log_return`, `to_excess_return`, `to_realized_vol`, `to_cross_sectional_rank`, `convert_target` dispatch |
| `metrics_returns.py` | Phase 7: returns-specific metrics -- `forecast_sharpe`, `hit_rate_returns`, `ic_pearson`, `ic_spearman`, `summarise_returns` |
| `cpcv.py` | Phase 8: combinatorial purged cross-validation; `cpcv_paths(origins, n_folds, k_test, embargo)`, `cpcv_skill_distribution(...)` |
| `pbo.py` | Phase 8: Bailey-Borwein-Lopez de Prado-Zhu Probability of Backtest Overfitting; `probability_of_backtest_overfitting`, `pbo_from_forecasts` |
| `deflated_sharpe.py` | Phase 8: Deflated Sharpe Ratio; `expected_max_sharpe_under_null`, `deflated_sharpe_ratio` |
| `romano_wolf.py` | Phase 8: studentized stepdown FWER under arbitrary dependence; `romano_wolf(diff_matrix, alpha, n_bootstrap, block_size)` |
| `preregistration.py` | Phase 8: hash-committed hypothesis ledger; `PreRegistration`, `register`, `resolve`, `from_hypothesis_dict`, `trial_count` |
| `holdout_vault.py` | Phase 8: never-touched final test slice; `initialize_vault`, `assert_no_vault_leakage`, `open_vault`, `vault_status` |
| `bayesian.py` | Phase 12: Normal-Normal hierarchical model; `hierarchical_findings(findings, forecasts)` -> posterior mean/sd/prob_positive/Bayes factor; `posterior_predictive_check` |
| `counterfactual.py` | Phase 16: per-finding interrogation lenses -- `factor_residualization`, `what_if_perturbation`, `outlier_removal`, `counterfactual_card` |
| `power.py` | Phase 16: per-cell statistical power -- `cohen_d`, `power_at_alpha`, `min_n_for_power`, `power_grid` |
| `synthetic_benchmark.py` | Synthetic-known-answer benchmark; `generate_universe(...)`, `grade_apparatus(...)`, `run_benchmark(n_trials, ...)`. Plants known-true (asset, regime, method) edges + distractor cells; runs every gate; reports recall/FDR per gate. |
| `capability_ablation.py` | Layer-by-layer marginal-contribution ablation; `run_capability_ablation(reports_dir, ...)`. Constructs progressively richer pipeline variants (baseline_only → +arima → +chronos_univ → +multivariate → +regime → +graph → full_stack), reports per-variant Mean MAE, marginal MAE-drop vs the previous variant, and cost-proxy in bytes. |
| `cli.py` | `autosignalx eval baseline` / `chronos` / `returns` / `pbo` / `vault-init` / `vault-open` / `synthetic` / `ablate-capability` / `status` (hardening is exposed under `autosignalx agent harden`) |

### `forecast/`

| File | Purpose |
|---|---|
| `baselines.py` | `naive_forecast`, `seasonal_naive_forecast(season_days=252)`, `arima_forecast(order=(1,1,1))` (on log adj_close) |
| `chronos2.py` | Lazy-loaded `Chronos2Pipeline` (cached via `lru_cache`); `chronos2_univariate`; `make_chronos2_multivariate(macro)` closure factory; `batched_ablation(method_specs, ohlcv, macro, windows, horizon_days)` for fast bulk runs |

The shared **ForecastFn contract** every method satisfies:

```python
ForecastFn = Callable[
    [pd.DataFrame, pd.Timestamp, list[pd.Timestamp]],
    pd.DataFrame  # with at least: timestamp, prediction (and optional lower/upper)
]
```

### `regime/`

| File | Purpose |
|---|---|
| `encoder.py` | `RegimeEncoder` (1D-CNN, 16-dim embedding from 60-day windows); `train_encoder` (triplet-loss training); `make_windows` (sliding-window utility) |
| `cluster.py` | `kmeans_regimes(embeddings, n_regimes, seed)`; `hmm_regimes(features, n_regimes, seed, n_iter)` |
| `labels.py` | `build_market_features` (SPY+QQQ returns + 4 macros, standardized); `fit_and_save` (orchestration); `load_regime_labels(method)`; `add_regime_to_forecasts(forecasts, method)` |
| `cli.py` | `autosignalx regime fit` / `status` |

### `signal/`

| File | Purpose |
|---|---|
| `features.py` | `compute_rsi(prices, window)`; `compute_macd_signal(prices, fast, slow, signal)`; `build_features_target(asset_ohlcv, macro_wide, horizon_days)` (8 technical + 8 macro features + binary direction target); `feature_columns(df)` |
| `ranking.py` | `_permutation_importance(predict_fn, X, y, n_repeats, seed)` (one-feature-at-a-time shuffle); `rank_features_per_regime(features_df, regime_labels, feature_cols, ...)` (HistGradientBoostingClassifier per regime) |
| `stability.py` | `walk_forward_rank(features_df, regime_labels, feature_cols, n_windows)` -- slides N windows, refits per-regime ranker inside each; `summarise_stability(walk_forward_df, top_k)` -- per-(regime, feature) stability metrics; `build_and_save(...)` orchestration |
| `cli.py` | `autosignalx signal rank` / `stability` / `status` |

### `graph/`

| File | Purpose |
|---|---|
| `correlation.py` | `partial_correlation_edges(returns, threshold)` via `sklearn.covariance.GraphicalLassoCV(cv=3)` |
| `causality.py` | `granger_edges(returns, max_lag, p_threshold)` via `statsmodels.tsa.stattools.grangercausalitytests`; takes min p across lags |
| `centrality.py` | `compute_centrality(edges, node_set, directed)` via NetworkX (degree / eigenvector / betweenness) |
| `build.py` | `build_and_save(p_threshold, max_lag, pcorr_threshold)` -- global graph orchestration |
| `per_regime.py` | `build_per_regime(...)` -- runs the same machinery within each regime's data subset; persists per-regime artifacts plus `regime_sensitivity.parquet` (cross-regime centrality dispersion); `load_per_regime`, `load_regime_sensitivity` readers |
| `cli.py` | `autosignalx graph build` / `build-per-regime` / `status` |

### `agent/`

| File | Purpose |
|---|---|
| `state.py` | `AgentState` TypedDict (round, max_rounds, ledger, context, current_*, next_action, session_id) |
| `prompts.py` | System prompts (`THEORIST_SYSTEM`, `SKEPTIC_SYSTEM`, `ADJUDICATOR_SYSTEM`, plus single-mode `PROPOSER_SYSTEM`, `CRITIC_SYSTEM`, `DECIDER_SYSTEM`) and message builders |
| `llm.py` | `LiveProvider` (DeepInfra OpenAI-compatible via `langchain_openai.ChatOpenAI`); `ReplayProvider` (recorded JSONL); content-hash response cache; per-call telemetry recording; `get_provider(record_replay, role)` factory with `ROLE_TO_ENV` mapping |
| `tools.py` | `slice_forecasts`, `test_significance`, `spawn_method`, `spawn_method_code`, `get_top_features`, `get_centrality_summary`, `context_snapshot` |
| `graph.py` | `build_agent_graph` (single-LLM mode); `experiment_node` (handles all 3 experiment types + auto-promotion); `run(max_rounds, seed, record_replay, session_id)` |
| `debate.py` | `build_debate_agent_graph` (4-node-per-round multi-role mode); `run_debate(...)` |
| `specs.py` | Constrained DSL for `spawn_method`: `validate_spec`, `_build_forecast_fn`, `execute(spec, config_name)` |
| `codegen.py` | Sandboxed Python for `spawn_method_code`: `validate_code` (AST walk), `compile_forecast_fn`, `execute_code_spec(spec)` |
| `ledger.py` | `append`, `load`, `clear`, `summarize_for_prompt(entries, limit)` |
| `findings.py` | `promote(...)` (idempotent on hypothesis+method+filters; bumps `replication_count` and appends to `replications` list); `_finding_id(content)` (content-hash); `make_session_id()` (sortable `YYYYMMDD-<hex>`) |
| `lineage.py` | `hypothesis_id(content)` (content-hash); `build_lineage(ledger_entries, finding_records, parent_lookback, overlap_threshold)`; `lineage_dataframe(lineage)` |
| `memory.py` | `consolidate(session_id, ledger, findings, provider)` (LLM-driven); `append_to_lessons`; `load_lessons(max_chars)` (tail-truncated read keeping section breaks) |
| `trace_eval.py` | LLM-as-judge per-round scoring on 4 rubrics (clarity / novelty / falsifiability / evidence_citing) |
| `self_critique.py` | LLM-as-judge re-evaluation of past findings; verdict ∈ {reinforced, unchanged, weakened, refuted} |
| `telemetry.py` | `record_call(...)`, `estimate_cost_usd`, `model_prices(model_id)` (env-var override > defaults > fallback), `CallTimer` (wall-clock context manager) |
| `sessions.py` | `list_sessions`, `session_summary(sid)`, `all_summaries`, `productivity_trend` (cumulative findings + cost) |
| `specialists.py` | Phase 14: 11 specialist roles + `consult_specialist(role, payload, ...)`; system prompts in `SPECIALIST_PROMPTS` |
| `lab.py` | Phase 14: lab-mode LangGraph; nodes Theorist -> Verifier -> Planner -> Specialist -> Skeptic -> experiment -> Adjudicator -> KG-writer -> [Theorist | END]; `run_lab(max_rounds, ...)` |
| `verifier.py` | Phase 14: `verify_hypothesis(h)` checks decision_rule + falsifier presence; downgrades for missing predicted_effect |
| `knowledge_graph.py` | Phase 14: persistent KG; `Node`, `Edge`, `add_node`, `add_edge`, `neighbors`, `query`, `ingest_findings`, `kg_summary`; storage at `reports/agent/kg/{nodes,edges}.jsonl` |
| `eig.py` | Phase 14: Bayesian-experimental-design proxy; `candidate_eig(forecasts, methods, assets, regimes, findings)`, `coverage_map(...)` returns the cockpit dataframe |
| `calibration.py` | Phase 15: agent-confidence calibration; `calibration_for_role(findings, survival, role)` -> Brier, ECE, per-bin reliability |
| `red_team.py` | Phase 15: adversarial attacks; `asset_shuffle_attack`, `time_shift_attack`, `run_red_team(findings, forecasts)`; persists `reports/agent/red_team.jsonl` |
| `coherence.py` | Phase 15: per-session coherence proxies (`lessons_uptake`, `lineage_branching_factor`, `theme_persistence_entropy`, composite `coherence_score`); persists `reports/agent/coherence.jsonl` |
| `prompt_optimizer.py` | Phase 15: prompt versioning; `register_prompt(role, text)`, `score_versions(role, trace_quality)`, `best_version(role, trace_quality)` |
| `eval_suite.py` | Phase 15: orchestrator; `run_eval_suite()` runs calibration + RedTeam + coherence + prompt-version aggregation; persists `reports/agent/eval_summary.json` |
| `cli.py` | `autosignalx agent run [--mode single\|debate\|lab]` / `score-traces` / `consolidate` / `self-critique` / `harden` / `eval-suite` / `status` |

## Agent loop

### Single-LLM mode (`agent/graph.py:build_agent_graph`)

```
START → propose → experiment → critique → decide → [propose | END]
```

One LLM model handles all three reasoning steps. `decide` either continues (if not at `max_rounds`) or stops; conditional edge dispatches.

### Debate mode (`agent/debate.py:build_debate_agent_graph`)

```
START → Theorist → Skeptic → experiment → Adjudicator → [Theorist | END]
```

Three different DeepInfra models play three roles (env-configurable):

| Role | Default model | System prompt theme |
|---|---|---|
| Theorist | `moonshotai/Kimi-K2.6` | Propose specific, mechanistically motivated hypotheses |
| Skeptic | `zai-org/GLM-5.1` | Identify confounders / alternative explanations *before* the experiment runs |
| Adjudicator | `deepseek-ai/DeepSeek-V4-Pro` | Weigh proposal vs challenge against experiment result; verdict ends with `VERDICT: support \| refute \| inconclusive` |

Each LLM-touching node writes its own ledger entry (`step ∈ {theorist, skeptic, adjudicator}`).

### Experiment node (shared across modes)

`experiment_node` in `agent/graph.py` branches on `state.current_hypothesis.experiment.type`:

| Type | Action |
|---|---|
| `slice_forecasts` | Filter cached forecasts by `(method, asset, regime_id)`; compute metrics. |
| `spawn_method` | Validate spec (`agent/specs.py`), build composed `ForecastFn` from primitives, run walk-forward (capped windows), persist to `reports/ablations/<name>.parquet`. |
| `spawn_method_code` | AST-validate Python (`agent/codegen.py`), compile in restricted globals, run walk-forward (capped windows), persist code + metadata + forecasts. |

**Auto-promotion** after every non-naive method experiment: call `test_significance(method, baseline="naive", asset, regime_id)`. If `promotable: True`, append to `reports/agent/findings.jsonl` with full evidence and provenance.

## DSL spec for `spawn_method` (`agent/specs.py`)

```json
{
  "name": "<alphanumeric_with_underscores_or_dashes>",
  "base": "naive | arima | chronos2_univariate | chronos2_multivariate",
  "covariate_subset": ["DX-Y.NYB"],            // optional; only chronos2_multivariate
  "ensemble_naive_weight": 0.3,                // [0, 1]; 0 = pure base, 1 = pure naive
  "max_windows": 8,                            // cap for fast iteration
  "asset_subset": ["SPY", "EFA"]               // optional asset filter
}
```

`validate_spec` rejects bad names, unknown bases, malformed covariate subsets, out-of-range ensemble weights, bad max_windows, with specific error messages — before any code runs.

## Sandbox model (`agent/codegen.py`)

For `spawn_method_code`:

1. **AST validation** rejects:
   - Imports of any module not in `ALLOWED_IMPORTS = {numpy, pandas, math}`.
   - References to forbidden names: `exec`, `eval`, `compile`, `__import__`, `open`, `input`, `globals`, `locals`, `vars`, `getattr`, `setattr`, `delattr`, `hasattr`, `exit`, `quit`, `breakpoint`, `help`.
   - Dunder attribute access (`x.__class__`, etc.).
   - Code length > 8000 characters.

2. **Restricted globals** (`_safe_globals`): a curated `__builtins__` dict (range, len, abs, min/max/sum, sorted, ...) plus `np`, `pd`, `math` aliases. Custom `__import__` resolves only `ALLOWED_IMPORTS`.

3. **Function-shape check**: the compiled module must define a callable named `forecast_fn`.

4. **Persistence**: generated source → `reports/agent/generated_methods/<name>.py`; metadata → `<name>.json`. Forecasts go to `reports/ablations/<name>.parquet` like any other method.

This is a **soft** boundary suitable for trusted (author-controlled) prompts. It is **not** a security boundary against adversarial Python; production hardening would require OS-level isolation (firejail / gVisor / WASM) or process separation.

## Cockpit reader pattern (`app/streamlit_app.py`)

A flat module registering 34 panel render functions in a `PANELS` dict (defined at the end of `app/streamlit_app.py`), dispatched by a two-step sidebar (Section → Panel) over the `PANEL_SECTIONS` list. Every panel is a read-only reader over `reports/` (or `data/cache/` for the Data panel); none of them computes heavy work — expensive computation lives in CLI commands and is persisted to disk.

| Panel | Reads |
|---|---|
| Overview | `__version__`, `settings` |
| Data | `data/cache/*.parquet` (via `data.cache.cache_status` and `data.loader.load_*_wide`) |
| Forecast Arena | `reports/ablations/*.parquet` (concat); optionally `reports/regimes/kmeans.parquet` for stratification |
| Regime Explorer | `reports/regimes/{kmeans,hmm,embeddings}.parquet` |
| Signal Discovery Lab | `reports/signals/signal_ranking.parquet` (most recent by mtime) |
| Cross-Asset Graph | `reports/graph/{edges,centrality}.parquet` |
| Regime-Conditioned Graph | `reports/graph/per_regime/regime_<id>/{edges,centrality}.parquet` + `regime_sensitivity.parquet` (Phase 6) |
| Signal Stability | `reports/signals/walk_forward_ranking.parquet` + `signal_stability.parquet` (Phase 6) |
| Backtest Arena | `reports/backtest/runs/<run_id>/{portfolio_daily,trades,metrics,meta}` (Phase 1) |
| Custom Study | `data/studies/<name>/study.yaml` + per-study `reports/studies/<name>/` (Phase 2) |
| Coverage Map | concat ablations + `reports/agent/findings.jsonl` + ledger (via `agent.eig.coverage_map`; Phase 14/16) |
| Statistical Power | concat ablations + regimes (via `eval.power.power_grid`; Phase 16) |
| Counterfactual Cards | findings + concat ablations + `data/cache/macro.parquet` (via `eval.counterfactual.counterfactual_card`; Phase 16) |
| Bayesian Evidence | findings + concat ablations (via `eval.bayesian.hierarchical_findings`; Phase 12) |
| Specialist Council | `reports/agent/ledger.jsonl` filtered for `step` starting with `specialist:` + `reports/agent/kg/{nodes,edges}.jsonl` (Phase 14) |
| Pre-Registration | `reports/agent/preregistrations.jsonl` + `preregistration_resolutions.jsonl` (Phase 8) |
| Holdout Vault | `reports/agent/holdout_vault/{vault,results}.json` (Phase 8) |
| Agent Calibration | findings + survival (via `agent.calibration.calibration_for_role`; Phase 15) |
| RedTeam Attacks | `reports/agent/red_team.jsonl` (Phase 15) |
| Agent Coherence | `reports/agent/coherence.jsonl` (Phase 15) |
| Agent Console | `reports/agent/ledger.jsonl`, `reports/agent/trace_quality.jsonl` |
| Auto-Play Replay | `reports/agent/ledger.jsonl` (with `st.session_state` for playback) |
| Findings | `reports/agent/findings.jsonl` |
| Lineage | `reports/agent/ledger.jsonl` + `findings.jsonl` (DAG inferred via `lineage.build_lineage`) |
| Self-Critique | `reports/agent/self_critique.jsonl` |
| Survival Analysis | `reports/agent/survival.jsonl` (Phase 5/8/12) |
| Lessons & Memory | `reports/agent/lessons.md` |
| Telemetry | `reports/agent/telemetry.jsonl` |
| Sessions | All stores, aggregated by `session_id` via `agent/sessions.py` |
| Reproducibility | live computed via `autosignalx.reproducibility.reproducibility_badge`; persisted to `reports/reproducibility_badge.json` (Phase 16) |
| Synthetic Benchmark | `reports/agent/synthetic_benchmark.json` (per-gate recall/FDR on planted-structure synthetic universes) |
| Capability Ablation | `reports/agent/capability_ablation.json` (layer-drop marginal-MAE vs cost-proxy frontier) |
| Headline | aggregates the top-line metrics across `findings.jsonl` / `survival.jsonl` / `synthetic_benchmark.json` / `capability_ablation.json` / `reproducibility_badge.json` for the reviewer's first-look panel |
| Ask the Memory | corpus index + LLM (Phase 3) |

The cockpit groups all 34 panels into seven sidebar sections: **Headline** (1), **Data & Forecasts** (3), **Discovery (L2-L4)** (5), **Strategy & Studies** (2), **Methodology** (12), **Agent activity** (9), **Reproducibility & memory** (2). The two-step radio (Section → Panel) keeps the surface navigable as the panel count grew past 30.

## Adding a new layer

1. `src/autosignalx/<layer>/__init__.py` exposing the public API.
2. Define output schema as a parquet (or JSONL) under `reports/<layer>/`.
3. Per-concern modules (e.g., `model.py`, `infer.py`).
4. `<layer>/cli.py` defining a `typer.Typer`; register in `src/autosignalx/cli.py` via `app.add_typer(<layer>_app, name="<layer>")`.
5. Makefile target.
6. Streamlit panel: render function appended to `PANELS` in `app/streamlit_app.py`.
7. Tests in `tests/test_<layer>.py`.
8. If the agent should consume the new artifact: add a tool in `agent/tools.py` that loads it; bundle it into `context_snapshot()`.

## Scheduled execution

`scripts/run_session.sh` (bash, cron-compatible) and `scripts/run_session.ps1` (PowerShell, Windows Task Scheduler) wrap a full session: `agent run --mode debate --record-replay` → `agent score-traces` → `agent consolidate`. Configurable via `AUTOSIGNALX_ROUNDS` and `AUTOSIGNALX_MODE` env vars.

Cron example:

```
0 3 * * * cd /path/to/repo && bash scripts/run_session.sh >> reports/agent/cron.log 2>&1
```

Cross-session aggregation in `agent/sessions.py` (`list_sessions`, `session_summary(sid)`, `all_summaries`, `productivity_trend`) produces per-session and cumulative views in the Sessions cockpit panel.
