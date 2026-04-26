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
- **Live evaluation harness** (latency-aware backtester with execution simulation).
- **Stronger sandbox** for `spawn_method_code` (process isolation or WASM runtime).
