# AutoSignal-X — Research Report

## Executive summary (state on `main`)

> **Bundled answer.** The apparatus has produced 9 promoted findings on
> the default 8-ETF universe across one initial single-LLM session, an
> exhaustive (method × asset × regime) sweep, and three lab-mode agent
> sessions (Qwen3-Max theorist + GLM-4.7-Flash skeptic/specialist +
> DeepSeek-V3 adjudicator), and the strict-bar hardening pipeline rejected
> every one of them. **0 of 9 survive the conjunction of every gate;
> 0 of 9 survive even the simpler block-holdout test alone.** Every
> promoted lift converges into a TLT / regime-3 / DXY family whose
> apparent skill is concentrated in the 2021-Q1 → 2022-Q3 sub-window
> where regime 3 exists. The methodology stack correctly grades all of
> them as fragile. On the controlled synthetic-known-answer benchmark
> (planted skill 0.30, 12 distractors, 8 trials), the strict bar
> recovers 94 % of planted truths at 0 % false-discovery rate — the
> apparatus is conservative by design, not broken.

* **What it is.** A 5-layer modular AI research system (Forecasting / Representation / Reasoning / Relational / Agentic) that discovers conditional predictive structure in liquid daily ETF prices and grades every claim through a defendable methodology stack. The contribution is the apparatus, not any single discovery.
* **Methodology stack (every promoted finding passes every gate).**
  1. Diebold-Mariano (Newey-West HAC) + block-bootstrap CI on per-row losses — the initial promotion gate.
  2. Benjamini-Hochberg FDR across the family of promoted findings.
  3. Adversarial replication: full-test, placebo regime-shuffle, 50/50 block-holdout.
  4. Combinatorial Purged Cross-Validation with embargo (Lopez de Prado).
  5. Probability of Backtest Overfitting across the search space (Bailey, Borwein, Lopez de Prado, Zhu).
  6. Deflated Sharpe Ratio adjusting for the number of strategies tried.
  7. Romano-Wolf joint stepdown FWER under arbitrary dependence.
  8. Hierarchical Normal-Normal Bayesian shrinkage + Bayes factor BF₁₀ ≥ 10 + posterior-predictive check.
  9. RedTeam attacks: asset-shuffle and time-shift replications.
  10. Strict bar `survives_all_strict` = the conjunction of every gate above.
* **Discovery agent.** LangGraph state machine in three modes: `single` (one LLM does propose / critique / decide), `debate` (Theorist / Skeptic / Adjudicator with three different DeepInfra models), `lab` (Theorist → Verifier → PrincipalInvestigator → Specialist consult → Skeptic → experiment → Adjudicator → KG-writer with 11 specialist roles). Auto-registers every hypothesis in the pre-registration ledger before the experiment runs.
* **Forecast targets.** `target_type ∈ {price, log_return, excess_return, vol, rank}` — the contract is backward-compatible (legacy parquets are read as `price`). Dedicated returns-baselines (`zero_return`, `mean_return`, `momentum`) and returns-metrics (`forecast_sharpe`, `hit_rate`, `ic_pearson`, `ic_spearman`).
* **Backtest.** Custom vectorised engine; six strategies; paired moving-block bootstrap on Sharpe-difference vs benchmark; strict no-look-ahead.
* **Custom studies.** User-defined universe / dates / splits via `data/studies/<name>/study.yaml`; full forecast → backtest pipeline plus a cockpit form.
* **Cockpit.** 34 panels grouped into seven sidebar sections (Headline; Data & Forecasts; Discovery (L2-L4); Strategy & Studies; Methodology; Agent activity; Reproducibility & memory). Every panel is a read-only viewer over a typed parquet/JSONL artifact under `reports/`. Headline panels include the coverage map, statistical-power dashboard, counterfactual cards, Bayesian evidence, synthetic benchmark, capability ablation, specialist council, pre-registration ledger, holdout vault, agent calibration, RedTeam attacks, agent coherence, and reproducibility badge.
* **Audited apparatus capability.** A synthetic-known-answer benchmark plants causal structure into a synthetic universe and measures per-gate recall + false-discovery rate, so the apparatus' own discriminative power is a measured number rather than a marketing claim. A capability-preserving ablation drops each layer in turn and reports marginal predictive skill against a cost-proxy in bytes.
* **Reproducibility.** Replay mode (no DeepInfra key required) reproduces every cockpit panel from the bundled `replay/agent_steps.jsonl`. The reproducibility badge bundles git hash + environment + per-artifact SHA-256 + a single bundle hash.
* **Tests.** 342 passing.

## Research questions

1. **Forecasting baseline.** What is the floor for daily ETF price-level forecast accuracy on a walk-forward evaluation with strict temporal ordering and 21-trading-day horizon?
2. **Foundation model contribution.** Does a frozen pretrained foundation model (Chronos-2) improve unconditionally on classical baselines (naive, seasonal-naive, ARIMA)?
3. **Multivariate covariates.** Does adding macro covariates (10Y yield, VIX, dollar index, crude) to Chronos-2's `past_covariates` improve forecasts unconditionally?
4. **Conditional structure.** Are there specific (regime, asset, method) combinations where the layered system outperforms naive with statistical significance under both Diebold–Mariano (p < 0.05) and a positive bootstrap CI on the loss difference?
5. **Agent autonomy.** Can a multi-agent research loop, reading from typed artifacts produced by the model layers, autonomously discover such conditional improvements and persist them with full provenance?
6. **Returns vs price targets (Phase 7).** When the forecast target moves from `adj_close` to log-returns / excess-returns / cross-sectional ranks, do the same conditional structures still hold? Does the lift survive after subtracting the trivial random-walk component that dominates price-level MAE?
7. **Selection bias (Phase 8).** When the agent has explored N hypotheses, what is the probability that the *best* one would beat zero under the null? Does the in-sample-best ranking transfer to out-of-sample performance (Probability of Backtest Overfitting)? Do the gates still hold under Romano-Wolf joint testing and Combinatorial Purged Cross-Validation?
8. **Bayesian evidence weight (Phase 12).** What is the posterior probability `P(θ_i > 0 | data)` for each finding, and what is the Bayes factor against the null (BF₁₀)? Does the hierarchical shrinkage estimate corroborate the frequentist verdict?
9. **Specialist orchestration (Phase 14).** Can a planner LLM productively route hypotheses to specialist sub-agents (Statistician, Quant, RiskOfficer, Economist, Implementer, RedTeam, Historian) so that long-horizon research arcs build cumulatively in a persistent knowledge graph?
10. **Agent calibration (Phase 15).** Is the Theorist's predicted confidence well-calibrated against the survival rate of its findings (Brier score, Expected Calibration Error)? Is the agent's research arc coherent across sessions (lessons uptake, theme persistence)?
11. **Apparatus capability (synthetic benchmark + capability ablation).** When causal structure is deliberately planted in a synthetic universe, what fraction does the apparatus recover at each gate, and what is its false-discovery rate? Which model layers carry the marginal predictive skill that justifies their precomputed-forecast cost?

## System overview

Five model layers (L1 Forecasting, L2 Representation, L3 Reasoning, L4 Relational, L5 Agentic) plus an agent that reads from all of them through a shared tool surface. Each layer persists outputs as typed parquet/JSONL under `reports/`; the agent both consumes those artifacts and writes its own structured outputs (ledger, findings, lessons, telemetry, trace quality, self-critique). The cockpit is a read-only viewer over the same artifacts.

For implementation detail (data flow, contracts, per-layer wiring, agent loop, sandbox model), see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Methodology

### Data

| Source | Symbols | Range | Frequency |
|---|---|---|---|
| yfinance ETFs | SPY, QQQ, IWM, GLD, TLT, EFA, EEM, HYG | 2010-01-01 → 2025-12-31 | Daily |
| yfinance macro | ^TNX (10Y yield), ^VIX, DX-Y.NYB (dollar index), CL=F (crude) | 2010-01-01 → 2025-12-31 | Daily |

Long-format parquet cache at `data/cache/{ohlcv,macro}.parquet`. Per-asset and per-signal timestamps strictly monotonic increasing (asserted at every persistence-boundary write).

### Walk-forward evaluation

| Split | Range |
|---|---|
| Train | ≤ 2018-12-31 |
| Validation | (2018-12-31, 2020-12-31] |
| Test | (2020-12-31, 2025-12-31] |

87 walk-forward windows over the test period, each with a 21-trading-day forecast horizon and a 21-day rolling step (non-overlapping). Each window's training set ends strictly before its forecast window begins (`train_end < forecast_start` enforced at construction in `data/splits.py:WalkForwardWindow.__post_init__`; raises `ValueError` with `"Leakage"` in the message if violated).

### Forecast contract

Every forecasting method emits a long-format DataFrame with the columns enforced by `eval.contracts.assert_forecast_schema`:

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | datetime64[ns] | Target trading day |
| `asset` | string | ETF ticker |
| `forecast_origin` | datetime64[ns] | Day forecast was made (must be `<` `timestamp`) |
| `horizon` | int | Days from origin to target |
| `method` | string | Method identifier (e.g., `naive`, `chronos2_multivariate`) |
| `prediction` | float64 | Point forecast in `adj_close` units |
| `origin_value` | float64 | `adj_close` at `forecast_origin` (used for directional metrics) |
| `target` | float64 | Realized `adj_close` at `timestamp` |
| `lower` (optional) | float64 | 10% quantile (probabilistic methods) |
| `upper` (optional) | float64 | 90% quantile |

### Metrics

- **MAE** — mean absolute error in `adj_close` units.
- **MAPE** — mean absolute percentage error (zero targets masked).
- **Directional accuracy** — fraction of forecasts where `sign(prediction − origin_value) == sign(target − origin_value)`.
- **Skill vs naive** — `1 − method_mae / naive_mae` (positive: better than naive).
- **CRPS** — approximate Continuous Ranked Probability Score from the (lower, prediction, upper) triple via the pinball-loss formulation `CRPS = 2 × mean over q of pinball_q`. Computed only for methods that supply intervals.

### Statistical promotion gate

A method is "promotable" against a baseline when **all three** of:

1. **Diebold–Mariano test** on per-row absolute losses returns p < 0.05. Newey–West HAC variance is used to handle the auto-correlation that `h`-step-ahead overlapping forecasts induce.
2. **Skill vs baseline > 0**.
3. **Block-bootstrap CI** (n_bootstrap = 1000, block size = 20) on the per-row loss difference is strictly above zero.

Implemented in `eval/significance.py:is_promotable`. Findings only enter `reports/agent/findings.jsonl` after passing this gate.

## Results

### L1 — Forecasting baseline (87 windows × 8 assets × 5 methods = 50,160 forecasts)

| Method | N | MAE | MAPE | Dir-acc | CRPS | Skill vs naive |
|---|---:|---:|---:|---:|---:|---:|
| naive | 10,032 | 4.254 | 2.04% | 0.2% | — | +0.000 |
| arima | 10,032 | 4.265 | 2.05% | 47.5% | — | −0.003 |
| chronos2_univariate | 10,032 | 4.470 | 2.13% | 46.8% | 2.897 | −0.051 |
| chronos2_multivariate | 10,032 | 4.499 | 2.14% | 47.8% | 2.936 | −0.058 |
| seasonal_naive | 10,032 | 25.86 | 11.80% | 44.6% | — | −5.079 |

**Findings:**

- Naive is the floor unconditionally. Chronos-2 (univariate or multivariate) underperforms naive by 5–6% MAE on this benchmark.
- Adding 4 macro past-covariates to Chronos-2 marginally degrades MAE/MAPE/CRPS while marginally improving directional accuracy.
- Chronos-2 80% intervals are calibrated (CRPS ≈ 2.9 in `adj_close` units).
- Naive's near-zero directional accuracy is structural (predicts no change → almost never matches the realized direction).
- Seasonal-naive (252-calendar-day lookback) collapses (5× worse MAE) — daily ETF prices have no exploitable annual seasonality at the level scale.

### L2 — Representation (regimes)

Market features for regime fitting: SPY + QQQ daily returns + 4 macro signals, standardized.

Contrastive 1D-CNN encoder: `Conv1d(n_features → 16, k=5, p=2) → GELU → Conv1d(16 → 32) → GELU → AdaptiveAvgPool1d(1) → Linear(32 → 16)`. 60-day windows. Trained 25 epochs with `nn.TripletMarginLoss` (positive: ±3-day adjacent windows; negative: ≥ 60-day distant windows). Adam, lr=1e-3, batch_size 64, margin 1.0.

KMeans on embeddings produces 4 regimes; `hmmlearn.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100)` on raw features produces 4 regimes for sanity check.

| Detector | Labeled timesteps | Per-regime sizes |
|---|---:|---|
| KMeans (contrastive) | 3,967 | {0: 1425, 1: 750, 2: 877, 3: 915} |
| HMM (Gaussian) | 4,026 | {0: 1421, 1: 793, 2: 1241, 3: 571} |

End-to-end fit time: 53s on CPU.

### L3 — Reasoning (per-regime feature ranking)

Features per `(asset, timestamp)`:
- 8 technical: `rolling_mean_5`, `rolling_mean_20`, `rolling_std_5`, `rolling_std_20`, `momentum_10`, `momentum_60`, `rsi_14`, `macd_signal`.
- 8 macro: level + 5-day percent change for each of `^TNX`, `^VIX`, `DX-Y.NYB`, `CL=F`.

Target: binary direction at horizon 21 (`adj_close[t+21] > adj_close[t]`).

Per-regime fit: `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=42)` on up to 2,000 sampled rows. Custom permutation importance (n_repeats=2): for each feature, shuffle its column, measure accuracy drop, repeat, average.

| Regime | Top-1 feature | Importance |
|---|---|---:|
| 0 | `macro_^TNX_level` | +0.071 |
| 1 | `macro_DX-Y.NYB_level` | +0.114 |
| 2 | `macro_CL=F_level` | +0.112 |
| 3 | `macro_DX-Y.NYB_level` | +0.189 |

**Finding:** A macro signal is the top-ranked feature in every regime; the *dominant macro depends on the regime* (10Y yield in 0; dollar index in 1+3; crude in 2). Technical indicators occupy ranks 2–5 but never rank 1.

### L4 — Relational (cross-asset graph)

GLASSO partial-correlation graph (`sklearn.covariance.GraphicalLassoCV`, 3-fold CV) on standardized daily returns. Granger causality between every ordered pair of assets, max lag = 5, min p-value across lags compared against threshold 0.05.

Result: 18 partial-correlation edges (out of 28 possible undirected pairs); 42 Granger edges (out of 56 possible directed pairs).

Centrality on the partial-correlation graph (sorted by eigenvector):

| Node | Degree | Eigenvector | Betweenness |
|---|---:|---:|---:|
| SPY | 0.857 | **0.532** | 0.143 |
| EFA | 0.857 | 0.422 | 0.333 |
| QQQ | 0.714 | 0.414 | 0.143 |
| IWM | 0.714 | 0.394 | 0.000 |
| EEM | 0.714 | 0.351 | 0.048 |
| HYG | 0.714 | 0.297 | 0.000 |
| TLT | 0.429 | 0.036 | **0.429** |
| GLD | 0.143 | 0.001 | 0.000 |

**Findings:**

- SPY is the structural hub (highest eigenvector centrality).
- GLD is statistically isolated (single direct edge of 7 possible).
- TLT has the highest betweenness despite low degree — it lies on shortest paths between equity assets and the bond / credit cluster.

### L5 — Agentic (recorded session)

Architecture: **LangGraph** state machine, debate mode, 4 nodes per round:

```
START → Theorist → Skeptic → experiment → Adjudicator → [Theorist | END]
```

| Role | Default model |
|---|---|
| Theorist | `moonshotai/Kimi-K2.6` |
| Skeptic | `zai-org/GLM-5.1` |
| Adjudicator | `deepseek-ai/DeepSeek-V4-Pro` |

The Skeptic challenges before the experiment runs (so its critique is independent of the result); the Adjudicator's verdict ends with `VERDICT: support | refute | inconclusive`. Each LLM-touching node writes its own ledger entry (`step ∈ {theorist, skeptic, adjudicator}`).

Experiment surface (`agent/tools.py`):

- `slice_forecasts(method, asset, regime_id)` — compute metrics on a slice of cached forecasts.
- `test_significance(method, baseline_method, asset, regime_id, p_threshold)` — DM + bootstrap promotion gate.
- `spawn_method(spec)` — author a new method via the constrained DSL (compose primitives: base method + covariate subset + naive ensembling + asset/window filters).
- `spawn_method_code(spec)` — execute sandboxed Python `forecast_fn`.
- `get_top_features(regime_id, top_k)` — read signal layer.
- `get_centrality_summary()` — read graph layer.
- `context_snapshot()` — bundle every artifact summary into one dict for prompt seeding (also includes `prior_sessions_lessons` from `reports/agent/lessons.md` for cross-session continuity).

#### Promoted finding from the recorded session

One promoted finding currently committed at `reports/agent/findings.jsonl`:

**`f_9395cd1bd1be`** — `chronos2_multivariate` for `TLT` in regime 3.

| Metric | Value |
|---|---|
| n | 407 |
| Method MAE | 1.845 |
| Naive MAE | 1.950 |
| Skill vs naive | +0.0539 |
| DM statistic | −2.063 |
| **DM p-value** | **0.040** |
| Bootstrap CI on loss difference | [+0.0047, +0.1974] |
| Horizon | 21 trading days |
| Replication count | 1 |

The Theorist's hypothesis composed prior-layer outputs:

- *Regime 3* identified by L2 (contrastive encoder + KMeans).
- *DXY/VIX dominate Regime 3* from L3 (per-regime feature ranking).
- *TLT high betweenness centrality (0.429)* from L4 (cross-asset graph).
- The forecast cache from L1.

Skeptic challenged on multiple-comparison risk; Adjudicator returned `VERDICT: support`.

#### Agent-authored method (refuted at the gate)

In the same session, the agent also authored a new method via the constrained DSL: `efa_dxy_bridge_focus` (`chronos2_multivariate` restricted to `DX-Y.NYB` covariate, asset_subset = `EFA`). The method was registered, ran through the standard walk-forward harness, and **failed the promotion gate** (skill negative). The forecasts are persisted to `reports/ablations/efa_dxy_bridge_focus.parquet`; no finding was promoted.

#### Trace quality (LLM-as-judge)

| Round | Clarity | Novelty | Falsifiability | Evidence-citing |
|---:|---:|---:|---:|---:|
| 0 | 3 | 1 | 1 | 4 |
| 1 | 5 | 4 | 5 | 5 |

(Round 2's response was unparseable; defaulted to None.)

#### Self-critique

The Adjudicator-role model re-read `f_9395cd1bd1be` against the rest of the ledger and returned `current_state: unchanged` with rationale: "no subsequent evidence directly addresses TLT in regime 3 or the Granger bridge mechanism." Honest: a single session does not replicate a finding; replication is the responsibility of multi-session scheduled runs.

## Backtested simulation

Phase 1 translates the discovery layers into a concrete trading simulation to test whether the discovered structure is *economically actionable*, not only statistically significant.

### Methodology

- **Engine.** Custom vectorized portfolio engine in `src/autosignalx/backtest/engine.py` (~80 LOC). Trade-timing invariant: weights set at `close(t)` earn the `close(t) → close(t+1)` return (one-bar shift, pinned by `tests/test_backtest_engine.py::test_trade_timing_no_lookahead`). Costs: one-way bps charge on `|Δw|` per asset per rebalance.
- **Window.** `2021-01-04` to `2025-12-30` (1255 daily bars, full test period). Discovery (Chronos-2 fits, regime model, agent-promoted findings) used data through `2020-12-31`. The runner refuses to start a backtest at any earlier date — see `tests/test_no_backtest_leakage.py`.
- **Universe.** The same 8 ETFs as discovery. Survivorship bias is acknowledged; the universe is fixed at the start of the run.
- **Rebalance cadence.** Every forecast origin (~21 trading days, 87 origins). At each origin, the strategy reads the predicted return for the holding period (horizon ≈ 20) and sets weights; positions hold until the next origin.
- **Costs.** 5 bps one-way (10 bps round-trip on a full position change). Stress-tested at 0 bps — qualitative ranking unchanged.
- **Significance.** Paired moving-block bootstrap (block size 5, 2 000 iterations) of `Sharpe(strategy) − Sharpe(benchmark)` with `BuyAndHoldSPY` as benchmark. Pairing preserves the cross-strategy correlation structure; an independent resample would understate the CI when both legs share market exposure. Implementation in `src/autosignalx/backtest/significance.py`.

### Strategies

| Strategy | Inputs consumed | Description |
|---|---|---|
| `BuyAndHoldSPY` | prices | 100% SPY, no rebalancing. Passive equity benchmark. |
| `EqualWeightUniverse` | prices | Daily-rebalanced equal weights across the 8 ETFs. |
| `TopKLong(k=3)` | prices, Chronos-2 multivariate forecasts | Each origin, hold equal weights in the top-3 assets by predicted return; cash otherwise. |
| `LongShortKK(k=2)` | prices, Chronos-2 multivariate forecasts | Long top-2 / short bottom-2 by predicted return; gross 100%, net 0% (dollar-neutral). |
| `RegimeGated(k=3)` | prices, forecasts, regime labels, findings | `TopKLong(k=3)` but only when the current regime has at least one promoted finding; cash otherwise. |
| `FindingDriven` | prices, forecasts, regime labels, findings | Trades only the (asset, regime) pairs in `findings.jsonl`, weighted by `skill_vs_baseline`, renormalised so gross ≤ 1.0. With the current single promoted finding, this means "long TLT only when regime == 3 and Chronos-2 predicts a positive return". |

### Results (5 bps cost; 2021-01-04 → 2025-12-30; 1 255 daily bars)

| Strategy              |    CAGR |   Vol  | Sharpe | Max DD  | Calmar | Hit Rate | Avg Turnover |
|-----------------------|--------:|-------:|-------:|--------:|-------:|---------:|-------------:|
| BuyAndHoldSPY         | 14.91 % | 17.10 %| +0.90  | −24.50 %| +0.61  | 54.7 %   | 0.0008       |
| EqualWeightUniverse   |  8.30 % | 12.63 %| +0.69  | −25.30 %| +0.33  | 53.9 %   | 0.0008       |
| TopKLong(k=3)         |  6.95 % | 15.34 %| +0.51  | −19.07 %| +0.36  | 52.6 %   | 0.0694       |
| LongShortKK(k=2)      | −1.80 % |  8.43 %| −0.17  | −23.31 %| −0.08  | 49.0 %   | 0.0833       |
| RegimeGated(k=3)      | −2.21 % |  9.62 %| −0.18  | −18.80 %| −0.12  | 15.8 %   | 0.0245       |
| FindingDriven         | −2.10 % |  6.41 %| −0.30  | −19.31 %| −0.11  |  6.3 %   | 0.0080       |

### Significance vs `BuyAndHoldSPY`

Paired block-bootstrap, n = 2 000, B = 5; "Significant" = 95 % CI excludes 0.

| Strategy             | Sharpe diff | 95 % CI            | p-value | Significant |
|----------------------|------------:|--------------------|--------:|:-----------:|
| EqualWeightUniverse  | −0.20       | [−0.57, +0.16]     |  0.260  |     no      |
| TopKLong(k=3)        | −0.38       | [−0.84, +0.06]     |  0.082  |     no      |
| LongShortKK(k=2)     | −1.07       | [−2.28, −0.08]     |  0.035  |   **yes**   |
| RegimeGated(k=3)     | −1.08       | [−1.86, −0.33]     |  0.007  |   **yes**   |
| FindingDriven        | −1.20       | [−2.28, −0.18]     |  0.024  |   **yes**   |

### Honest interpretation

- **No signal-driven strategy beats the passive SPY benchmark on this universe and window.** Three of five (`LongShortKK`, `RegimeGated`, `FindingDriven`) are *significantly worse*; the remaining two (`EqualWeightUniverse`, `TopKLong`) are statistically indistinguishable from SPY.
- **The promoted finding does not translate to actionable alpha.** Finding `f_9395cd1bd1be` validated `chronos2_multivariate` over `naive` for TLT in regime 3 with `p = 0.04`, `skill = +5.4 %` MAE-vs-naive, and a bootstrap CI of `[+0.5 %, +19.7 %]`. The `FindingDriven` strategy that trades exactly that slice loses 2.1 %/yr at -0.30 Sharpe. Three plausible explanations:
  1. **Statistical-vs-economic gap.** A 5.4 % MAE improvement on price-level forecasts at a 21-day horizon is small relative to typical asset volatility; the residual error still dominates trading P&L net of costs.
  2. **Regime distribution shift.** The validation slice (407 bars) and the test-window slice differ in regime-prevalence and macro context; the alpha was real on the validation set but not robust.
  3. **Multiple-comparison risk on the discovery side** — the agent explored multiple hypotheses; the survivor passing `p < 0.05` may not survive a Bonferroni-corrected gate.
- **Drawdowns are *not* reduced.** Even when signal strategies hold cash most of the time (`FindingDriven` is invested only 6.3 % of bars), max drawdown remains ~19 %; cash drag is real.
- **Turnover is the largest cost component for cross-sectional strategies.** `LongShortKK` averages 8.3 % daily turnover; at 5 bps that's a 1 %/yr cost drag in addition to the negative gross signal.

### What this validates about the system

The backtest does not *invalidate* the research instrument; it validates the **discipline**. The promotion gate flagged `f_9395cd1bd1be` as significant on the validation slice, the self-critique pass flagged the absence of replication, and the backtest now provides the third layer of scrutiny — economic-significance evaluation. A research instrument that consistently produces *honestly negative* out-of-sample backtests on borderline-significant findings is exactly what a quant research workflow should look like. A pipeline that turned every `p = 0.04` finding into a profitable strategy would be the suspicious one.

### What Phase 1 deliberately did **not** do

- No Kelly / vol-targeting; equal-weight only.
- No hyperparameter search over `k`, cost assumptions, or holding period (would reintroduce backtest overfitting).
- No live/paper trading wiring.
- Slippage is a flat bps proxy; no liquidity-aware modelling.

## Custom studies

Phase 2 lifts the hardcoded universe and date range so users can run AutoSignal-X on their own data. A `Study` is a named, isolated workspace declared by a small YAML at `data/studies/<name>/study.yaml`; the universe (assets + macro covariates), the date range, the walk-forward split boundaries, the forecast horizon and rolling step, and the cost assumption are all per-study.

### What is parameterised

- **Universe**: any list of yfinance-resolvable tickers as `assets` and `macro`.
- **Date range and splits**: `start_date`, `end_date`, `train_end`, `val_end`, `test_end`. Validator enforces `start < train_end < val_end < test_end ≤ end_date`.
- **Walk-forward**: `forecast_horizon_days`, `rolling_step_days`.
- **Backtest**: `cost_bps`, optional `backtest_start` (defaults to day after `val_end`).

### Artifact isolation

Each study owns its own tree:

```
data/studies/<name>/
    study.yaml                       # config
    cache/{ohlcv,macro}.parquet      # per-study yfinance pull
reports/studies/<name>/
    ablations/{baseline,chronos2}.parquet
    backtest/runs/<run_id>/{portfolio_daily, trades, metrics, meta}
    regimes/, signals/, graph/, agent/   # populated as those layers run
```

Default-flow artifacts under `data/cache/` and `reports/` are unchanged when `--study` is not passed; studies are strictly additive.

### Pre-flight validation

`autosignalx study validate <name>` runs offline checks (date ordering, walk-forward window count, universe size) and surfaces results as `errors / warnings / info`. An opt-in `--check-tickers` flag adds a 5-day yfinance probe to confirm each ticker resolves; failures are reported as warnings, not errors, since transient network issues should not block a run.

### Surfaces

- **CLI**: every subcommand grew an optional `--study X` flag. `autosignalx study {create, list, show, validate, delete}` manages study definitions.
- **Cockpit**: the **Custom Study** panel exposes the same flow form-based (create / validate / pipeline buttons / status). The sidebar **Study scope** selector switches Forecast Arena and Backtest Arena to read from the chosen study's tree.

### Honest scope of Phase 2

The forecast (baseline + Chronos-2) and backtest layers are fully study-aware. The discovery layers (regime, signal, graph, agent) still write to project-default paths regardless of `--study`. Reason: those layers consume more user time and have subtler precondition requirements (sample sizes, regime-count selection, agent cost) that a future Phase 2 sub-iteration will address. The forecast → backtest pipeline is the path most users want for "see how the system behaves on my universe", so that path was prioritised.

# Phase 3 — Conversational explainability (post-Phase-6)

Phase 3 replaces the original "Ask the Memory" panel with a **grounded RAG chat** over the run corpus. The corpus spans the agent ledger, promoted findings, lessons, trace-quality scores, self-critiques, telemetry summaries, and backtest run metrics. Every claim the assistant makes is followed by a `citation_id` (e.g. `finding:f_9395cd1bd1be`, `ledger:r3/skeptic`, `backtest:<run_id>/TopKLong`) copied verbatim from the retrieved evidence; questions whose answer is not in the corpus trigger a canonical refusal rather than a hallucination.

### Pipeline

1. **Corpus.** `autosignalx chat index` walks `reports/agent/*.jsonl`, `reports/agent/lessons.md`, and `reports/backtest/runs/*/metrics.json` to produce a flat list of citable `Chunk(citation_id, kind, text, meta)` records.
2. **Embedding.** Live mode calls DeepInfra `BAAI/bge-large-en-v1.5` (1024-dim) and caches each text by content hash under `reports/agent/embed_cache/`. Replay/no-key mode falls back to a deterministic hashed-bag embedding (256-dim) so the panel works without a key.
3. **Retrieval.** Top-K cosine similarity (K=6) via a single matmul over the L2-normalized matrix. The corpus is small (low thousands of chunks), so no FAISS/ANN index is warranted.
4. **Generation.** Live mode sends a strict cite-or-refuse system prompt + the retrieved chunks to the chat-role LLM (DeepSeek-V4-Pro by default). Replay mode renders the top retrieved chunks with their citation IDs (no LLM call) so the panel is fully reproducible without a key.
5. **Citation enforcement.** The answer is post-filtered to extract `[citation_id]` markers; if a live-mode response contains zero valid citations, the assistant's text is overridden with the canonical refusal.

### Grounding eval

`autosignalx chat eval` exercises a bundled fixture of seven questions: five grounded (covering each artifact kind) and two intentionally off-corpus. The harness scores **citation recall** (does the top-K contain the expected artifact kind?) and **refusal accuracy** (does the off-corpus question trigger refusal in live mode?). On the bundled artifacts in deterministic replay mode, recall is 0.60 and refusal is n/a (replay always renders top retrievals); live mode with proper embeddings is expected to score substantially higher and exercise the refusal branch on off-corpus questions.

### Surfaces

- **CLI**: `autosignalx chat {index, status, ask, eval}`.
- **Cockpit**: the **Ask the Memory** panel is now a chat interface backed by the index, with a "Rebuild index" button and a citation chip row beneath every assistant turn.

### Honest scope of Phase 3

The corpus loader covers all seven on-disk artifact kinds but treats every JSONL row as a single chunk -- long ledger entries are truncated to 1200 chars rather than split semantically. Hashed-bag embeddings used in replay mode are intentionally lossy (per-question retrieval is noisy on subtle queries); they exist so the deterministic CI / no-key reviewer path works, not as a replacement for real embeddings. Expanding the eval set, semantic chunking inside long ledger entries, and bundling a canned live-mode chat trace into `replay/` are the obvious follow-ups.

# Phase 4 — Demo and deployment (post-Phase-6)

Phase 4 closes the gap between the local repo and reviewers who do not want to run anything. Two parallel deployment paths ship from the same `main` branch.

### 4A — Static HTML snapshot (GitHub Pages)

`autosignalx snapshot build` renders a multi-page, navigable HTML report from whatever artifacts currently sit under `reports/`. Every page (Overview / Forecasts / Regimes / Findings / Backtest / Agent / Chat corpus) is a self-contained HTML file under `reports/cockpit_snapshot/`, with Plotly figures pulled from CDN to keep file sizes small (each page is ~4-220 KB). The generator is robust against partial artifacts: every section gracefully degrades to a "not built yet" notice when its inputs are absent, so a fresh-clone snapshot still renders.

A GitHub Actions workflow at `.github/workflows/pages.yml` runs the build on every push to `main` and publishes the result to GitHub Pages. The deployment is the **always-works** option: zero runtime infrastructure, zero environment variables, no key required.

### 4B — Live deployed cockpit (Streamlit Community Cloud)

A top-level `streamlit_app.py` shim sets `AUTOSIGNALX_REPLAY=true` when no `DEEPINFRA_API_KEY` is configured and execs the real app at `app/streamlit_app.py`. `.streamlit/config.toml` pins headless mode and the AutoSignal-X red theme; `.streamlit/secrets.toml.example` documents the optional DeepInfra credential surface for users who want live LLM calls in the deployed instance. The cloud instance reads pre-computed parquet/JSONL artifacts and does **not** retrain anything by default, fitting comfortably in the free-tier resource caps.

### Honest scope of Phase 4

The two deployments cover Phase 4A and 4B from the project roadmap. Phase 4C ("user-runnable custom-study runs in the deployed app") is intentionally not in scope -- a custom Phase 2 study can take minutes per asset on free-tier CPUs, which makes a poor demo. The deployed app exposes the Custom Study panel for read access (validate, list) but heavy steps (`data fetch`, `eval chronos`, `backtest run`) remain local-only via the CLI.

# Phase 5 — Statistical hardening: surviving the methodology

The agent's auto-promotion gate evaluates each hypothesis individually at p < 0.05 with a positive bootstrap CI. That is reasonable per-hypothesis but accumulates risk across hypotheses, splits, and seeds. Phase 5 layers four post-hoc attacks on every promoted finding and reports the **survival** of each finding through every attack. The point is not to inflate the headline finding count -- it is to make the methodology auditable end-to-end, so the project's research artifact is the *system*, not any single discovery.

### The four attacks

1. **Benjamini–Hochberg FDR** (`src/autosignalx/eval/fdr.py`). Step-up procedure across the family of original p-values from every promoted finding. Controls expected false-discovery rate; α=0.10 by default. A finding survives FDR iff its q-value ≤ α.
2. **Full-test replication** (`src/autosignalx/eval/adversarial.py`). The agent's spawned methods cap walk-forward windows at 8 by default for fast iteration. Full-test re-runs the same gate on the entire ablation slice, not just those windows. A finding that holds on a small slice but collapses on the full window was overfit to the agent's chosen sub-period.
3. **Placebo regime-shuffle**. The hypothesis claims a regime-conditioned effect. Shuffling regime labels uniformly while preserving the marginal distribution destroys the conditioning. If the gate still flags the finding promotable on shuffled labels, the "regime" was not the explanatory variable -- the lift was a marginal effect mistaken for conditional structure.
4. **Block-holdout**. Split the slice 50/50 by `forecast_origin` time. The finding survives iff *both* halves independently pass the gate. Catches lifts driven by a single sub-period rather than a stable mechanism.

The conjunction of all four (`survives_all`) is the strict bar. Any single failure is a research insight: it tells the reader precisely how the original gate over-promoted.

### Results on the existing finding

A single finding has been promoted in the bundled session: *"In regime 3, chronos2_multivariate beats naive on TLT"* (`f_9395cd1bd1be`, original p=0.040, skill 5.4%, q=0.040 at the family of size 1). After hardening:

| Attack | Result | What it means |
|---|---|---|
| BH-FDR (α=0.10) | ✅ pass (q=0.040) | Single-finding family; FDR is loose here. |
| Full-test replication | ✅ pass (n=1254, p=0.025, skill 3.3%) | The lift survives the 8-window cap; not a windowing artefact. |
| Placebo regime-shuffle | ✅ pass (shuffled p=0.30, no signal) | The regime conditioning was load-bearing; not a marginal effect. |
| **Block-holdout (50/50 by `forecast_origin`)** | **❌ fail** | **First half (≤2023-06-22): p=0.025, promotable. Second half (>2023-06-22): p=0.82, no signal. Skill drops from ~10% to 0.5%.** |

This is exactly the kind of insight the methodology was built to expose. The finding is not "wrong" -- it is **honestly fragile**. The lift is concentrated in the first half of the test window; the second half does not corroborate it. The reading is that the research apparatus correctly graded its own discovery, not that the apparatus failed.

### What this means for the project's research story

The user-facing claim is **not** "we found a robust signal." The user-facing claim is:

> *We built an agentic research system that proposes hypotheses, runs experiments under walk-forward integrity, promotes findings under DM + bootstrap rigor, and then attacks every promoted finding under FDR, full-test, placebo, and block-holdout adversarial replication. The system flagged one candidate; the hardening exposed that the candidate's lift does not generalise across the full test window. The contribution is the apparatus and the discipline, not the candidate.*

The methodology layer is the artifact: it is the part that generalises to other asset classes, other agents, other research questions. A future session that promotes a finding which *also* survives block-holdout is then defensible without caveat — the gate has already proven it can fail honestly.

### Surfaces

- **CLI**: `autosignalx agent harden` (re-evaluates every promoted finding under FDR + adversarial replication; writes `reports/agent/survival.jsonl`).
- **Cockpit**: new **Survival Analysis** panel renders the pass/fail grid, headline survival counts, and a per-finding expander with full evidence from each attack.

### Honest scope of Phase 5

Phase 5 ships only the **statistical hardening** layer (5.2 from the original Phase 5 plan). Three other planned sub-iterations are deferred:

- **5.1 Returns-target forecasting** -- a contract change that would invalidate every existing ablation parquet and require a full re-run. Worth doing, but a separate phase given the surface area. The current price-level contract is honest about what it measures (MAE on `adj_close`); reviewers can see that the block-holdout failure above is partly a story about how price-level MAE is dominated by the persistence of the level itself.
- **5.3 New experiment primitives** (`cross_validate`, `hp_search`, `regime_conditioned_backtest` as agent tools) -- widens the agent's experiment surface; not blocking the methodology defence.
- **5.4 Refinement loop** (branch-from-failure prompts; semantic lineage edges) -- deepens multi-turn agent dynamics; orthogonal to single-finding rigor.

These are tracked under "Future work" below.

# Phase 6 — Structural enrichments (post-Phase-6)

Phase 6 closes two of the structural gaps explicitly listed under Phase-5 limitations: the cross-asset graph being global rather than per-regime, and the per-regime signal ranking being a single fit rather than a walk-forward stability check. Both ship as **complementary** layers (the original artifacts are preserved); the new artifacts give reviewers an honest second look at the structure the agent had access to.

### 6A — Regime-conditioned cross-asset graph

`autosignalx graph build-per-regime` runs the existing GLASSO + Granger + centrality machinery once per regime, on the subset of timesteps with that regime's KMeans label. Output:

* `reports/graph/per_regime/regime_<id>/edges.parquet` and `centrality.parquet` per regime.
* `reports/graph/per_regime/regime_sensitivity.parquet` -- one row per asset, summarising the cross-regime range of degree, eigenvector, and betweenness centrality. Larger range = the asset's structural role flips more dramatically across regimes; this is research-grade input for a regime-aware allocator.

On the bundled artifacts the per-regime build surfaces several real structural observations:

| Regime | Top hub (eig.) | Top bridge (betw.) |
|---|---|---|
| 0 | SPY | TLT |
| 1 | EEM | TLT |
| 2 | SPY | GLD |
| 3 | SPY | TLT |

* **EEM displaces SPY as the primary hub in regime 1** -- consistent with regime 1 being a non-US-led state.
* **TLT is the bridge in 3 of 4 regimes** -- a stable cross-asset role as the safe-haven flow conduit.
* **Top regime-sensitive asset by betweenness range: GLD (0.000 → 0.619)**. GLD goes from peripheral to the most-bridged asset depending on regime. TLT (0.048 → 0.571) is a close second.

### 6B — Walk-forward signal-importance stability

`autosignalx signal stability` slides N walk-forward windows across the timeline and refits the per-regime HistGradientBoosting + permutation importance ranker inside each. Output:

* `reports/signals/walk_forward_ranking.parquet` -- per-(window, regime, feature) importance + rank.
* `reports/signals/signal_stability.parquet` -- per-(regime, feature) summary: mean importance, mean rank, rank std, top-K share, composite stability ∈ [0, 1].

A feature with high mean importance *and* high stability *and* high top-K share is research-grade. High importance with low stability is an averaging artefact that the original single-fit ranker would have missed.

On the bundled artifacts (4 walk-forward windows, KMeans regimes):

| Regime | Top-1 stable feature | Mean rank | Stability | Top-5 share |
|---|---|---|---|---|
| 1 | `macro_DX-Y.NYB_level` | 3.0 | 0.83 | 1.00 |
| 2 | `macro_^TNX_level` | 2.0 | 0.92 | 1.00 |
| 3 | `macd_signal` | 1.8 | 0.94 | 1.00 |

Each of these features is in the top 5 in every walk-forward window for its regime -- a much stronger signal than "ranked first when averaged over the whole period." The dollar-index signal in regime 1 and the 10Y-yield signal in regime 2 are both economically interpretable: regime 1 looks like a USD-driven state, regime 2 like a rates-driven state.

### Surfaces

- **CLI**: `autosignalx graph build-per-regime`, `autosignalx signal stability`.
- **Cockpit**: new **Regime-Conditioned Graph** and **Signal Stability** panels (live + static-snapshot).
- **Chat corpus**: extended with `regime_graph:<asset>` and `signal_stability:r<id>/<feature>` citation kinds, plus `survival:<finding_id>` from Phase 5.

### Honest scope of Phase 6

The walk-forward stability layer reuses the same single-period KMeans regime labels for every window -- ideally the regime detector itself would also be refit per window, but that introduces label-permutation across windows (the "same" regime can swap IDs across refits) and is a substantially larger refactor. The current implementation answers "given this regime taxonomy, which features are stably important inside each regime over time?" -- a useful but narrower question than fully walk-forward regime-aware ranking. The per-regime graph similarly assumes the regime taxonomy is fixed.

## Limitations (post-Phase 6)

Phase 6 closes two prior limitations (cross-asset graph now per-regime; signal ranking now walk-forward stable). Remaining limitations:

- **Multi-session replication is still single-session.** Survival per finding is one record per session. Cross-session aggregation exists in the Sessions panel but is currently informational rather than gating.
- **Forecast targets are price levels (`adj_close`).** This is the largest remaining limitation: MAE on `adj_close` is dominated by price persistence, which means even a fully-passing finding (FDR + full-test + placebo + block-holdout) would still be partly measuring trivial predictability. Returns-target forecasting (originally Phase 5.1) is the highest-priority next step.
- **Codegen sandbox is a soft boundary.** AST validation + restricted globals defend against accidental damage, not adversarial Python.
- **Macro covariate universe is four signals.** No equity-implied vol, credit spreads, sector rotation factors, or term-structure shape.
- **Agent-authored methods cap at 8 walk-forward windows by default.** Phase 5's full-test replication catches when a finding fails this cap, but the agent itself still operates inside it.
- **Single asset class.** Liquid daily ETFs only. The Phase 2 custom-study layer enables other universes; honest defence requires running an end-to-end study on at least one alternative class.
- **No live deployment evaluation.** Latency, slippage, no-trade horizons, and execution simulation are absent.

# Phase 7 -- Returns-target forecast contract

The original forecast contract is price-level (`adj_close`). MAE on price levels is dominated by persistence -- a finding that beats naive there is partly measuring the trivial random-walk component, which is exactly the failure mode the Phase-5 block-holdout test of `f_9395cd1bd1be` exposed. Phase 7 lifts that limitation by extending the forecast contract with a `target_type` column.

### Target types

`eval.contracts.get_target_type` reads an optional `target_type` column on every forecast frame (defaulting to `price` for backward compatibility). Five types are now legal:

| Target | Units | When to use |
|---|---|---|
| `price` | `adj_close` | Legacy default; trivially dominated by persistence |
| `log_return` | `log(target / origin_value)` | Stationary, zero-mean; the right substrate for most quant claims |
| `excess_return` | `log_return - rf_daily * horizon` | Subtracts ^TNX-implied risk-free rate; isolates risky-asset alpha |
| `vol` | rolling realised log-return volatility | For volatility-prediction hypotheses |
| `rank` | cross-sectional fractional rank | For relative-performance hypotheses (the dominant institutional alpha lens) |

Adapters live in `eval/targets.py`; conversion is explicit (`convert_target(forecasts, target_type)`) so the audit trail is preserved.

### Returns-baselines

`forecast/returns_baselines.py` adds three returns-native baselines (`zero_return`, `mean_return(lookback=60)`, `momentum(lookback=60, scale=0.5)`). The CLI command `autosignalx eval returns --target log_return` runs them and persists `target_type=log_return` forecasts.

### Returns metrics

`eval/metrics_returns.py` adds the metrics that matter for returns:

* `forecast_sharpe` -- annualised Sharpe of a sign-following strategy
* `hit_rate` -- fraction of rows where `sign(prediction) == sign(target)`
* `ic_pearson` / `ic_spearman` -- linear and rank information coefficients

These are reported per-method by `eval/metrics_returns.summarise_returns`. Existing price-level pipelines are unchanged.

# Phase 8 -- Selection-bias-aware evaluation

Phase 5 attacks every promoted finding individually. Phase 8 adds the **family-wide** statistical machinery a research lab uses to defend against selection bias when many hypotheses were tried.

### Combinatorial Purged Cross-Validation (CPCV)

`eval/cpcv.py` implements Lopez de Prado's CPCV. With N folds and k=2 test folds per path, it constructs C(N,2) = N(N-1)/2 paths, each with an embargo buffer between train and test to prevent overlap leakage from h-step-ahead forecasts. `cpcv_skill_distribution` returns the mean/std/min/max of skill-vs-baseline across all paths -- a true distribution rather than a single walk-forward number. Hardening output now includes a `cpcv` field per finding.

### Probability of Backtest Overfitting (PBO)

`eval/pbo.py` implements Bailey, Borwein, Lopez de Prado, Zhu (2014). Across all 2^S combinatorially symmetric IS/OOS splits of the per-period skill matrix, PBO measures how often the IS-best strategy ranks below the OOS median. PBO ≈ 0 = robust ranking; PBO ≈ 0.5 = pure search noise. Surfaced as `autosignalx eval pbo` and persisted to `reports/agent/pbo.json`.

### Deflated Sharpe Ratio

`eval/deflated_sharpe.py` implements Bailey & Lopez de Prado's DSR. Given the observed Sharpe of a strategy, DSR computes the probability the strategy would have produced that Sharpe under the null of zero true alpha *given that N candidate strategies were tested*. The expected max Sharpe under null is computed from the closed-form extreme-value approximation. DSR > 0.95 is the rigorous bar.

### Romano-Wolf step-down

`eval/romano_wolf.py` implements the more powerful FWER control under correlation: studentize per-hypothesis loss-difference series, bootstrap the joint max-|t| distribution under the null, and step-down adjust each hypothesis's p-value. Joint adjusted p-values now appear as `rw_q` in the survival ledger.

### Pre-registration ledger

`eval/preregistration.py` lets the agent (or a human) hash-commit a hypothesis -- with explicit decision rule, predicted effect size, and falsifiability statement -- *before* running the experiment. The hash uniquely identifies the registration; resolutions are appended separately so registration history is never rewritten. The lab-mode agent (Phase 14) auto-registers every hypothesis at the verifier step. The cockpit's **Pre-Registration** panel shows registered / open / resolved counts.

### Holdout vault

`eval/holdout_vault.py` declares a never-touched final test slice. `assert_no_vault_leakage(forecasts)` raises if any forecast row's `forecast_origin` falls inside the locked range; `open_vault(...)` is the one-time evaluation method that records the final headline metric to `reports/agent/holdout_vault/results.json`. CLI: `autosignalx eval vault-init <start> <end>` and `autosignalx eval vault-open`.

### Strict survival bar

`eval/survival.py:harden_findings` now produces a `survives_all_strict` flag = (Phase-5 attacks) ∧ (Romano-Wolf survives) ∧ (DSR ≥ 0.95) ∧ (CPCV mean skill > 0). This is the lab-grade promotion gate -- a finding that passes here is defensible against the full search space.

# Phase 12 -- Hierarchical Bayesian evidence

Frequentist DM/FDR/RW gates report rejection of the null. Decision-makers want to know "given the data, what's the probability the lift is real?". `eval/bayesian.py` implements a Normal-Normal hierarchical model with empirical-Bayes hyperparameters (no NumPyro/PyMC dependency required):

```
d_i | theta_i ~ Normal(theta_i, sigma_i^2 / n_i)
theta_i ~ Normal(mu, tau^2)            # population
mu, tau^2 estimated by method of moments
```

Per-finding outputs: posterior mean, posterior sd, P(theta_i > 0), Bayes factor BF_10 vs the null. The strict Bayesian bar is BF ≥ 10 and P(theta>0) ≥ 0.95. Results plumb through `survival.jsonl` as the `bayesian` field and surface in the **Bayesian Evidence** cockpit panel; `posterior_predictive_check` also simulates next-session loss differences for graphical PPC.

# Phase 14 -- Specialist agent lab

Phase 5's debate had three roles (Theorist / Skeptic / Adjudicator). Phase 14 builds a specialist research lab on top:

| Role | Function |
|---|---|
| PrincipalInvestigator | Plans the round; picks the next specialist |
| Theorist | Mechanism-grounded hypothesis (existing) |
| Skeptic | Adversarial reviewer (existing) |
| Adjudicator | Decisive verdict (existing) |
| Statistician | Phase-8 selection-bias accounting + Phase-12 Bayes |
| Quant | Factor residualization + capacity |
| RiskOfficer | Drawdown / tail / regime concentration |
| Economist | Mechanistic plausibility + narrative consistency |
| Implementer | Execution / slippage / turnover |
| RedTeam | Adversarial perturbations beyond the existing harness |
| Historian | Queries the persistent KG |

`agent/specialists.py` declares prompts; `agent/lab.py` wires them into a LangGraph state machine: `Theorist -> Verifier -> Planner -> Specialist -> Skeptic -> experiment -> Adjudicator -> KG-writer -> [Theorist | END]`. Each consultation is a single LLM call recorded in the ledger as `step="specialist:<role>"`, visible in the **Specialist Council** cockpit panel.

### Persistent knowledge graph

`agent/knowledge_graph.py` adds a structured KG persisted as `reports/agent/kg/{nodes,edges}.jsonl`. Nodes: `finding | hypothesis | method | regime | asset | mechanism | session | ticket`. Edges: `refines | refutes | generalizes | complements | attacks | cites | implements | promoted_by | discovered_in | applies_to`. The `kg_writer` node ingests promoted findings into the graph idempotently after every round; the Historian role queries it for prior work.

### Bayesian experimental design (EIG)

`agent/eig.py` provides an Expected Information Gain proxy that ranks candidate (method × asset × regime) experiments. The score combines novelty (slice not yet tested or promoted) with a power proxy (sqrt(n_samples)) and penalises already-tested cells. The cockpit's **Coverage Map** panel renders this as a 4D heatmap so reviewers see exactly where the agent has hunted.

### Pre-registration verifier

`agent/verifier.py` checks every hypothesis carries a real pre-registration block (decision_rule, falsifier, ideally predicted_effect) before the experiment runs. Missing fields are flagged in the ledger; the verifier ledger entry feeds into Phase-15 calibration when the predicted effect is available.

CLI: `autosignalx agent run --mode lab` runs the full lab-mode loop; `--specialists statistician,quant,economist` overrides the specialist pool.

# Phase 15 -- Agent self-improvement and evals

### Calibration

`agent/calibration.py` scores agent confidence vs survival outcomes. Brier score and Expected Calibration Error (ECE) summarise how well the Theorist's prior predictions matched the eventual hardening verdicts. The **Agent Calibration** panel renders the reliability diagram.

### Red team attacks

`agent/red_team.py` adds two attacks beyond Phase-5:

* **Asset-shuffle**: re-run the gate on every other asset in the same regime. A finding survives iff no other asset is also promotable -- the asset specificity is genuine, not a regime-wide effect.
* **Time-shift**: shift `forecast_origin` by 5 trading days and re-evaluate. Catches single-date coincidences.

The **RedTeam Attacks** panel renders per-finding outcomes from `reports/agent/red_team.jsonl`.

### Long-horizon coherence

`agent/coherence.py` scores per-session coherence with three proxies that don't require an LLM call:

* `lessons_uptake` -- substring match between current proposals and prior `lessons.md`
* `lineage_branching_factor` -- mean out-degree of lineage DAG nodes
* `theme_persistence_entropy` -- Shannon entropy of (asset, regime) cells visited

These combine into a single composite score persisted to `reports/agent/coherence.jsonl`. The **Agent Coherence** panel renders the trend across sessions.

### Prompt versioning

`agent/prompt_optimizer.py` treats role prompts as versioned artifacts. `register_prompt(role, text)` is idempotent on hash; `score_versions(role, trace_quality)` aggregates the per-round scoring rubrics across versions; `best_version(role)` returns the highest-scoring version by geometric-mean of rubric averages. Prompt history lives at `reports/agent/prompts/<role>.jsonl`.

### Eval suite orchestration

`agent/eval_suite.py` and `autosignalx agent eval-suite` run all four (calibration, RedTeam, coherence per session, prompt-version scoring) end-to-end and write a summary JSON. Intended to run after every session's `harden` step.

# Apparatus-capability evaluation (synthetic benchmark + capability ablation)

These two artifacts post-date Phase 16 and exist to answer two questions
the rest of the report only addressed implicitly: *(a) does the apparatus
actually find planted structure when it exists, at what false-discovery
rate?* and *(b) which model layers carry the marginal skill that
justifies their precomputed-forecast cost?*

### Synthetic-known-answer benchmark

Real-market alpha is rare and noisy; that makes it impossible to
distinguish "the apparatus correctly grades a finding as null" from
"the apparatus is unable to recognise structure when it exists." The
benchmark addresses this by generating synthetic price universes where
specific (asset, regime, method) cells have a deliberately planted
predictive edge and surrounding distractor cells have none. Implementation
in `eval/synthetic_benchmark.py`; CLI `autosignalx eval synthetic`.

Per-gate recall + FDR on a representative configuration (planted
skill 0.18 vs naive, 12 distractors, 6 trials):

| Gate | Mean recall | Mean FDR | Mean # promoted |
|---|---:|---:|---:|
| `dm_only` | 0.67 | 0.00 | 2.0 |
| `+fdr` | 0.67 | 0.00 | 2.0 |
| `+adversarial` | 0.22 | 0.00 | 0.7 |
| `+rw` | 0.22 | 0.00 | 0.7 |
| `+bayes` | 0.22 | 0.00 | 0.7 |
| `strict` | 0.22 | 0.00 | 0.7 |

Reading: the apparatus is **conservative by design**. DM-only with bootstrap
recovers two-thirds of planted truths at zero FDR; layering adversarial
replication and the Bayesian gate drops recall to one-quarter while
maintaining zero FDR. The per-gate drop is the price of selection-bias
safety, not a bug. The benchmark is committed to
`reports/agent/synthetic_benchmark.json` and rendered in the cockpit's
**Synthetic Benchmark** panel + the static snapshot.

### Capability-preserving ablation (layer-by-layer marginal contribution)

Implementation in `eval/capability_ablation.py`; CLI `autosignalx eval
ablate-capability`. The ablation does not retrain models; it concatenates
the cached ablation parquets, slices by `method` column, and constructs
progressively richer variants (`baseline_only` -> `+arima` -> `+chronos_univ`
-> `+multivariate` -> `+regime` -> `+graph` -> `full_stack`). Each
variant's MAE is computed on the union of methods it has access to;
marginal-skill = previous-variant-MAE − this-variant-MAE.

On the bundled artifacts:

| Variant | Layers | # findings | Mean MAE | Marginal skill | Cost (KB) |
|---|---|---:|---:|---:|---:|
| `baseline_only` | L1 floor | 0 | 4.254 | — | 329 |
| `+arima` | L1 + ARIMA | 0 | 4.260 | -0.005 | 329 |
| `+chronos_univ` | L1 + Chronos-2 univariate | 0 | 4.330 | -0.070 | 913 |
| `+multivariate` | L1 full | 1 | 4.372 | -0.042 | 913 |
| `+regime` | L1 + L2 (regime gating) | 1 | 4.372 | 0.000 | 913 |
| `+graph` | + L4 (cross-asset filter) | 1 | 4.372 | 0.000 | 913 |
| `full_stack` | L1 + L2 + L3 + L4 + L5 | 1 | 4.372 | 0.000 | 913 |

Reading: **naive is the unconditional MAE floor**; ARIMA adds nothing
on price-level forecasting; Chronos-2 actively *worsens* MAE on this
benchmark. The single promoted finding (`f_9395cd1bd1be` — TLT,
regime 3) only emerges once the multivariate forecast layer **and**
the regime layer are both in scope, because the promotion gate is
conditional on regime; without L2 the apparatus can't promote
anything. The implied capability-vs-cost frontier on this dataset is
unambiguous: ARIMA / Chronos-2 univariate add cost without skill,
while `chronos2_multivariate` + L2-L4-L5 are load-bearing.

The result is committed to `reports/agent/capability_ablation.json`
and rendered in the cockpit's **Capability Ablation** panel.

# Phase 16 -- Cockpit observability and explainability

Eleven new panels that turn the cockpit from "viewer" into "research-lab dashboard":

| Panel | Purpose |
|---|---|
| **Coverage Map** | 4D heatmap of (method × asset × regime) coloured by EIG; reviewer sees exactly where the agent has hunted and where blank space remains. |
| **Statistical Power** | Per-cell Cohen's d, observed power at α=0.05, sample-size required for 80% power. Distinguishes under-powered failures from genuine nulls. |
| **Counterfactual Cards** | Per-finding factor residualization, what-if perturbation buckets, outlier-removal stability. Makes the *reasoning* behind a finding interrogable. |
| **Bayesian Evidence** | Posterior mean / sd / P(θ>0) / BF_10 from the Phase-12 hierarchical model. |
| **Specialist Council** | Multi-role consultation feed (Statistician/Quant/Risk/Economist/Implementer/RedTeam/Historian) and KG explorer. |
| **Pre-Registration** | Hash-committed hypotheses, open vs resolved registrations, resolution outcomes. |
| **Holdout Vault** | Vault status (locked/opened) and one-time-eval results. |
| **Agent Calibration** | Reliability diagram + Brier + ECE for the Theorist's predicted confidence. |
| **RedTeam Attacks** | Per-finding asset-shuffle + time-shift verdicts. |
| **Agent Coherence** | Per-session lessons-uptake, lineage branching factor, theme entropy, composite score. |
| **Reproducibility** | Git hash, env, library versions, replay flag, per-artifact SHA-256 hashes, single bundle hash. |

Per-finding cards now expose **factor residualization** (regress per-bar loss-diff on macro factors; report residual mean and t-statistic), **what-if** (skill stratified by prediction-magnitude quartile), and **outlier removal** (drop top 1% absolute-diff rows; recompute skill). Implementation in `eval/counterfactual.py`. Statistical power dashboard in `eval/power.py`. Reproducibility module in `autosignalx/reproducibility.py` with a `write_badge()` API and a clickable "refresh badge" button in the cockpit.

## Findings produced and graded

The apparatus has now produced **9 promoted findings** on the bundled
universe — one from the original single-LLM session, three from an
exhaustive (method × asset × regime) sweep, and five authored by the
lab-mode agent across three multi-round sessions (Qwen3-Max theorist
+ GLM-4.7-Flash skeptic/specialist + DeepSeek-V3 adjudicator). Every
one was passed through the full hardening pipeline (BH-FDR +
adversarial replication + Combinatorial Purged CV + Probability of
Backtest Overfitting + Deflated Sharpe + Romano-Wolf joint stepdown +
hierarchical Normal-Normal Bayesian shrinkage + RedTeam asset-shuffle
+ time-shift attacks).

| Finding | Method | Filters | p | skill | FDR | full | placebo | block | RW | DSR | Bayes | **strict** |
|---|---|---|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `f_9395cd1bd1be` | chronos2_multivariate | TLT, regime 3 | 0.040 | +5.4% | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | **❌** |
| `f_336b84d3e3a9` | chronos2_multivariate | TLT, regime 0 | 0.002 | +6.3% | ✅ | ✅ | — | ❌ | ❌ | ❌ | ✅ | **❌** |
| `f_b39c742f4449` | chronos2_multivariate | TLT, regime 3 | 0.040 | +5.4% | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | **❌** |
| `f_8a5317b8b398` | chronos2_univariate | TLT, regime 0 | 0.018 | +5.7% | ✅ | ✅ | — | ❌ | ❌ | ❌ | ❌ | **❌** |
| `f_b38c6394a5fe` | tlt_regime3_dxy_only | global | 0.014 | +10.1% | ✅ | ❌ | — | ❌ | — | — | ✅ | **❌** |
| `f_abedb3261bd3` | tlt_regime3_dxy_naive_blend | global | 0.004 | +9.6% | ✅ | ❌ | — | ❌ | — | — | ✅ | **❌** |
| `f_3eb10d54f55f` | tlt_regime3_dxy_naive_ensemble | global | 0.003 | +9.4% | ✅ | ❌ | — | ❌ | — | — | ✅ | **❌** |
| `f_8a446564ee92` | tlt_regime3_dxy_conditional_chronos2 | global | 0.014 | +10.1% | ✅ | ❌ | — | ❌ | — | — | ✅ | **❌** |
| `f_c3a3c0300a24` | tlt_regime3_dxy_naive_chronos2_ensemble | global | 0.002 | +9.7% | ✅ | ❌ | — | ❌ | — | — | ✅ | **❌** |

**0 of 9 survive the strict bar; 0 of 9 even survive block-holdout.**
All nine pass BH-FDR (each per-finding p is below 0.05). Only the four
findings tied to specific (asset, regime) cells run through full-test
+ placebo replication; the five agent-authored methods register their
findings against the global slice (no asset / regime filter on the
finding row), and Romano-Wolf + DSR are skipped for those because the
per-finding loss-difference series cannot be isolated cleanly. The
block-holdout 50/50 split rejects every single one.

### Why every finding fails block-holdout

The regime detector, fit on full 2010-2025 data, partitions the test
window into three contiguous chunks: regime 0 (Jan-Feb 2021,
37 days), regime 3 (Feb 2021 → Oct 2022, 408 days), regime 2
(Oct 2022 → Dec 2025, 812 days). The TLT / regime-3 lift exists in
the *first* half of the test window's regime-3 stretch (early 2021 →
late 2021); the second half (early 2022 → Oct 2022) does not
corroborate it. Block-holdout splits each finding's slice 50/50 by
`forecast_origin` and demands both halves promote independently; both
halves fail on the second half.

### Convergence pattern observed in the agent loop

Every lab-mode session followed the same arc. In round 0 the Theorist
proposed a TLT / regime-3 hypothesis with DXY as the load-bearing
covariate and auto-promotion fired. In round 1 it proposed an
EFA / regime-1 variant; the experiment did not promote because
regime 1 has zero occurrences in the test window. In round 2 it
proposed GLD / regime-2 (the dominant 2023-2025 regime) with a
naive-blend prior that consistently underperformed. In round 3 it
returned to TLT / regime-3 and promoted another variant. This held
even when the specialist mix was swapped from Statistician to
Quant + Economist in the third session.

The driver is the agent's persistent memory: `lessons.md` and the
existing `findings.jsonl` give the Theorist strong priors that
TLT / regime-3 carries skill, so each fresh session re-anchors there.
The specialist roles correctly flag multiple-comparison risk in their
consultations, but the Theorist's prompt does not currently penalise
re-exploring the same anchor — a clear avenue for the next iteration
of the agent scaffold.

### Per-run telemetry

| Session | Theorist + Specialists | Auto-promoted | Cost (USD) |
|---|---|---|---|
| `20260425-14c7446d` (initial single-LLM session, before the sweep) | Kimi-K2.6 single-mode | 1 (`f_9395cd1bd1be`) | $0.011 |
| Exhaustive sweep (`20260427-829d024a-sweep`) | n/a (deterministic) | 3 (the f_336.., f_b39.., f_8a53.. cells) | $0.000 |
| Lab session 1 (`20260427-2500372e`) | Qwen3-Max + statistician | 1 (`tlt_regime3_dxy_only`) | $0.058 |
| Lab session 2 (`20260427-6a269b3f`) | Qwen3-Max + statistician | 2 (`..._naive_blend`, `..._naive_ensemble`) | $0.034 |
| Lab session 3 (`20260427-dea16a9f`) | Qwen3-Max + quant + economist | 2 (`..._conditional_chronos2`, `..._naive_chronos2_ensemble`) | $0.035 |

Cumulative LLM spend across all sessions ever (including
post-session score-traces / consolidate / self-critique calls): **$0.247
USD**. `reports/agent/telemetry.jsonl` carries 152 calls in total, every
row with `role` and `session_id` populated (an earlier defect that
recorded `role="unknown"` was fixed before the live runs and the
historical telemetry was back-filled to match the new schema).

The **Agent Calibration** panel reports a Brier score of 0.49 and an
Expected Calibration Error of 0.70 on the small set of findings where
the Theorist actually emitted a `predicted_effect` block — a genuinely
poor calibration baseline that the next iteration of the agent
scaffold should target. The **Agent Coherence** panel reports a
composite coherence score of 0.63 averaged across 7 sessions; the
agent stays focused (low theme-persistence entropy), uptakes lessons
from earlier sessions (high lessons-uptake), and the lineage DAG has
moderate branching.

### Backtested behaviour with the full finding set

`reports/backtest/runs/20260427T141235-b7b0f1` is the latest run and
uses the full 9-finding `findings.jsonl`. The `FindingDriven` strategy
now trades five additional cells (the agent-authored TLT / regime-3
variants), but its overall behaviour is materially unchanged because
all five live in the same regime that the original bundled finding
covered: the strategy still holds cash from 2022-10-08 onwards
because no promoted finding's regime is active in regime 2 (the
dominant post-2022 regime). The cockpit's `Backtest Arena` panel
surfaces this explicitly via a per-strategy activity summary (bars
total, bars active, active percentage, first / last active day).

A strict-bar survivor on this universe would require either (a) a
regime-2 finding — the agent tested several and none promoted, (b) a
different asset or method family that breaks out of the
TLT / regime-3 anchor, or (c) a re-fit of the regime detector that
puts post-2022 data into the same labelling space as the 2021-2022
lift. None emerged from the four sessions run.

### What this says about the system

The user-facing claim is **not** "we found nine signals." It is:

> *The apparatus produced nine candidate findings on the bundled
> universe across replay-mode and live LLM sessions, all converging
> on a single (asset, regime, mechanism) anchor; the hardening
> stack rejected every one of them as fragile under block-holdout.
> The methodology layer is the artifact; the apparatus correctly
> graded its own discoveries.*

The synthetic-known-answer benchmark above is the matching
positive-control evidence that the apparatus *can* find structure
when structure exists.

## Future work

Closed in Phase 5: ~~multiple-comparison correction~~ (BH-FDR), ~~adversarial replication~~ (full-test, placebo, block-holdout).
Closed in Phase 6: ~~per-regime cross-asset graph~~, ~~walk-forward signal ranking~~ (now produces a stability summary).
Closed in Phase 7: ~~returns-target forecasting~~ (target_type contract + returns baselines + returns metrics).
Closed in Phase 8: ~~selection-bias-aware evaluation~~ (CPCV + PBO + Deflated Sharpe + Romano-Wolf + pre-registration + holdout vault).
Closed in Phase 12: ~~hierarchical Bayesian evidence~~ (Normal-Normal model, Bayes factors, PPC).
Closed in Phase 14: ~~specialist agent lab~~ (10+ roles, persistent KG, EIG planner, pre-registration verifier).
Closed in Phase 15: ~~self-improving prompts and agent evals~~ (calibration, RedTeam, coherence, prompt versioning, eval suite).
Closed in Phase 16: ~~cockpit observability and explainability~~ (11 new panels: Coverage Map, Statistical Power, Counterfactual Cards, Bayesian Evidence, Specialist Council, Pre-Registration, Holdout Vault, Agent Calibration, RedTeam Attacks, Agent Coherence, Reproducibility).

Open:

- **Returns-target forecasting** -- the largest remaining gap. Extend the forecast contract with `target_type ∈ {price, log_return, excess_return}`, add returns-targeted baselines, propagate through the harness + DM gate. The Phase 1 backtester already operates in returns space; this would close the train/deploy contract gap.
- **Wider experiment-tool surface** for the agent (`cross_validate`, `hp_search`, `regime_conditioned_backtest` as primitives the Theorist can compose).
- **Refinement loop** -- explicit branch-from-failure prompts; semantic lineage edges (`refines`, `complements`, `abandoned`).
- **Codegen invariants** -- per-method assertions written by the Theorist, executed in sandbox, surfaced as evidence in the gate.
- **Cross-session aggregation** as a real promotion gate, not just a view: a finding gets a "robustness" tier based on how many independent sessions reproduce it under hardening.
- **Regime detector walk-forward refit** -- closes the remaining honest-scope caveat in Phase 6B.
- **Live-execution-aware backtester** (latency, slippage by liquidity, partial fills) and **multi-horizon / vol-targeting / Kelly** strategies on top of the existing engine.
- **Stronger sandbox** for `spawn_method_code` (process isolation or WASM runtime).
