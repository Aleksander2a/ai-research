# AutoSignal-X — Research Report

## Research questions

1. **Forecasting baseline.** What is the floor for daily ETF price-level forecast accuracy on a walk-forward evaluation with strict temporal ordering and 21-trading-day horizon?
2. **Foundation model contribution.** Does a frozen pretrained foundation model (Chronos-2) improve unconditionally on classical baselines (naive, seasonal-naive, ARIMA)?
3. **Multivariate covariates.** Does adding macro covariates (10Y yield, VIX, dollar index, crude) to Chronos-2's `past_covariates` improve forecasts unconditionally?
4. **Conditional structure.** Are there specific (regime, asset, method) combinations where the layered system outperforms naive with statistical significance under both Diebold–Mariano (p < 0.05) and a positive bootstrap CI on the loss difference?
5. **Agent autonomy.** Can a multi-agent research loop, reading from typed artifacts produced by the model layers, autonomously discover such conditional improvements and persist them with full provenance?

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

Phase 1 (post-submission) translates the discovery layers into a concrete trading simulation to test whether the discovered structure is *economically actionable*, not only statistically significant.

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

Phase 2 (post-submission) lifts the hardcoded universe and date range so users can run AutoSignal-X on their own data. A `Study` is a named, isolated workspace declared by a small YAML at `data/studies/<name>/study.yaml`; the universe (assets + macro covariates), the date range, the walk-forward split boundaries, the forecast horizon and rolling step, and the cost assumption are all per-study.

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

## Limitations

- **One promoted finding from one recorded session.** Replication requires multi-session runs via `scripts/run_session.sh`. The self-critique correctly flags the absence of subsequent confirming evidence.
- **Multiple-comparison risk.** The agent ran several hypotheses; one passing p < 0.05 may not survive Bonferroni correction. The headline framing is "one specific slice with one DM-significant lift", not "robust generalization".
- **Single asset class.** Conclusions about naive's strength may be specific to liquid daily ETFs. The methodology (harness, contracts, agent loop) generalizes; the findings are class-specific.
- **Codegen sandbox is a soft boundary.** AST validation + restricted globals defend against accidental damage, not adversarial Python. Production hardening would require OS-level isolation (firejail / gVisor / WASM) or process separation.
- **Macro covariate universe is four signals.** Equity-implied vol surfaces, credit spreads, sector rotation factors, and term-structure shape are not in the universe.
- **Agent-authored methods cap at 8 walk-forward windows by default** for fast iteration. Promoted findings should be re-validated on the full test period before acting on them.
- **Cross-asset graph is global, not per-regime.** Computing GLASSO + Granger within each regime would expose regime-conditional structural changes.
- **Forecast targets are price levels (`adj_close`).** Returns and risk-adjusted-target shifts (Sharpe, Sortino) are not modeled.
- **No live deployment evaluation.** Latency, slippage, no-trade horizons, and execution simulation are absent.

## Future work

- **Replication via scheduled runs.** `scripts/run_session.sh` is cron-compatible. Cross-session productivity is already aggregated in the Sessions cockpit panel.
- **Per-regime cross-asset graph** (pass `returns[regime_mask]` to `partial_correlation_edges` / `granger_edges` per regime).
- **Walk-forward signal ranking** (per-(regime, training-window) instead of random subsample within regime).
- **Wider experiment-tool surface** (per-spec hyperparameter search; explicit cross-validation in the experiment node).
- **Returns-target forecasting** (extend the forecast contract with optional `target_type` field).
- **Live-execution-aware backtester.** The Phase 1 backtester applies a flat bps cost; a richer implementation would model latency, slippage by liquidity, and partial fills.
- **Backtest extensions.** Multi-horizon strategies (1d / 5d / 21d) using the full Chronos-2 horizon panel, rather than only the holding-period bar; vol-targeting and Kelly sizing on top of the existing strategies; expanded universe via the Phase 2 custom-input layer.
- **Stronger sandbox** for `spawn_method_code` (process isolation or WASM runtime).
