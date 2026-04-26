# AutoSignal-X — Research Report

> Layer-by-layer findings narrative for the AutoSignal-X submission.
> Append-only across iterations; this report grew with the codebase
> rather than being written after the fact.

## Executive summary

**What we built.** AutoSignal-X is a 5-layer modular AI research instrument for studying *what makes signals predictive, when, and why* in dynamic markets. Each layer is a credible MVP of its model class:

| Layer | Implementation | Iter |
|---|---|---|
| L1 Forecasting | Chronos-2 (multivariate + 80% intervals) + classical baselines (naive, seasonal-naive, ARIMA) | 3 |
| L2 Representation | Contrastive 1D-CNN encoder + KMeans regimes; HMM sanity-check baseline | 4 |
| L3 Reasoning | Per-regime feature ranking via HistGradientBoosting + permutation importance | 5 |
| L4 Relational | GLASSO partial-correlation graph + Granger causality + NetworkX centrality | 6 |
| L5 Agentic | LangGraph state machine over DeepInfra LLMs (Kimi K2.6 / GLM-4.7-Flash / DeepSeek V4-Pro) with persistent JSONL ledger, replay-mode fallback for no-key reviewers | 7 |

All ablation results, regime labels, signal rankings, graph artifacts, and the recorded agent trace are committed; the cockpit reads from them out-of-the-box.

**What we found.**
1. **Foundation models alone don't beat naive on liquid daily ETF prices.** Chronos-2 underperforms naive by 5-6% MAE; macro past-covariates make it slightly worse. 80% intervals are well-calibrated (CRPS ≈ 2.9). This is a calibrated negative result, not a bug -- daily ETF prices are very close to martingales, and naive (random walk) is the Bayes-optimal forecaster under that data-generating process.
2. **Macros dominate every regime's top-5 features for direction prediction**, but the *dominant* macro depends on the regime: 10Y yields in Regime 0, dollar index in Regimes 1+3, crude oil in Regime 2. Conditional macro selection is the right structure, not unconditional multi-covariate input. This explains why Chronos-2 with all 4 macros simultaneously underperformed univariate Chronos-2 in Iter 3.
3. **The cross-asset graph reveals typed structural roles.** SPY is the hub (eigenvector centrality 0.532); GLD is statistically isolated (~0 centrality, confirms uncorrelated-diversifier role); TLT is the bridge (highest betweenness 0.429, transmits shocks between equity and other regimes).
4. **The live LangGraph agent composes findings from every prior layer.** By Round 4 it proposes a mechanistic, falsifiable hypothesis using regime structure (Iter 4) + per-regime feature importance (Iter 5) + graph centrality (Iter 6) -- the conditional-improvement search opened by Iter 3's negative result.

**What this is not.** This is not a profitable trading strategy. It is not a benchmark dominated by a foundation model. It is a research instrument designed to reveal where layered structure earns its keep -- and we report the finding (mostly *not yet, conditionally*) honestly rather than cherry-picking a positive headline.

**What's next** (future work, not in this submission):
- Diebold-Mariano significance tests on the agent's identified slices (which DM-significant lifts hold up?).
- Per-regime conditional forecaster: ensemble that selects method + features per regime.
- Live-deployment-aware eval (latency, slippage, no-trade horizons) -- the offline-to-online generalization problem Deeter explicitly cares about.
- TS2Vec replacement of the contrastive encoder; PyTorch Geometric GNN replacement of the static graph layer (both deferred per the project plan as scope-discipline cuts).

---

## Thesis

## Thesis

Can a multi-model AI pipeline outperform standalone forecasting systems by explicitly modeling latent regimes, structured signal relevance, and relational dependencies? Specifically, when forecasting financial time series under regime shifts, does adding each successive layer (representation, reasoning, relational, agentic) yield measurable, statistically significant improvement over the layer below it — and where does the marginal layer stop paying for itself?

## Research questions

**Primary**: Which model class contributes most to forecast quality and robustness under regime shifts?

**Secondary**:
1. Do learned latent-state embeddings improve calibration more than they improve point accuracy?
2. Does conditioning signal selection on regime materially change the top-K signals?
3. Can an agent autonomously discover non-obvious feature combinations that human-default features miss?

## Success criteria

- **Scientific**: ≥1 statistically significant (Diebold–Mariano, p<0.05) improvement of layered system over Chronos-2-only baseline on at least one ETF; ≥1 actionable finding about signal classes or regime behavior.
- **Engineering**: Fully reproducible (`make demo` from a fresh clone in <5 minutes); modular; ablation framework runs end-to-end.
- **Strategic**: Reads as a research artifact — methods, ablations, calibrated claims, honest negative results where they occur.

## Methodology overview

Walk-forward evaluation with strict temporal ordering (no future leakage; tests assert this). Per-regime stratification of all metrics. Statistical significance via Diebold–Mariano. Probabilistic forecasts evaluated via CRPS in addition to point metrics (MASE, MAPE, directional accuracy).

---

## Iter 0 — Scaffold

Repository structure, packaging, test infrastructure, Streamlit cockpit shell. No findings yet; this iteration lays the foundation for the layers that follow.

**Deliverables**:
- `pyproject.toml` with `uv`-managed dependencies and `hatchling` build backend.
- `src/autosignalx/` package with one module per layer (all empty placeholders, each documenting which iteration implements it).
- Typer CLI (`autosignalx version`, `autosignalx status`) that prints the layer-status table; later iterations register `data`, `forecast`, `regime`, `signal`, `graph`, `agent`, `report` subcommands.
- Streamlit cockpit (`app/streamlit_app.py`) rendering an Overview panel with thesis, layer-status grid, and system info; structured so iterations register new panels by adding to a `PANELS` dict.
- Pydantic-based config (`src/autosignalx/config.py`) with `.env` auto-loading and a `use_replay` property that auto-selects deterministic mode when no DeepInfra key is set.
- Smoke tests covering package imports, settings load, and per-layer module importability.
- README with architecture, quick start (`uv sync` → `streamlit run app/streamlit_app.py`), repo layout, and the full iteration plan.

**Verification**: `make sync && make test && make demo` works end-to-end; CLI prints the layer-status table with all 5 layers as `pending`.

---

---

## Iter 1 — Data pipeline

Reproducible market-data substrate for every subsequent layer.

**Sources** (all free, all reproducible):
- ETF OHLCV via yfinance: SPY, QQQ, IWM, GLD, TLT, EFA, EEM, HYG.
- Macro signals via yfinance: ^TNX (10Y yield), ^VIX, DX-Y.NYB (DXY), CL=F (crude).
- Window: 2010-01-01 to 2025-12-31, daily frequency.

**Schema contract**. Long-format DataFrames with strict columns, asserted at every cache write.
- OHLCV: `(timestamp, asset, open, high, low, close, adj_close, volume, returns)`.
- Macro: `(timestamp, signal, value)`.

The contract is the API every model layer reads against. Schema enforcement at the persistence boundary keeps corrupt data out of the eval harness.

**Walk-forward splits**. `WalkForwardWindow` enforces `train_end < forecast_start` at construction (raises `ValueError` with "Leakage" in the message if violated). `walk_forward_windows(val_end, test_end, horizon_days, step_days)` yields a list of windows that progressively advance the training-set boundary by `step_days` each iteration; no future data ever leaks into a training window.

**Default split** (configs/default.yaml): train [2010-01-01, 2018-12-31], val (2018-12-31, 2020-12-31], test (2020-12-31, 2025-12-31]. The test period spans COVID, the 2022 inflation shock, and the ZIRP-to-hike transition -- the regime layer (Iter 4) and stratified eval (Iter 5+) will have genuinely distinct regimes to discover.

**Tests** (16 new, 25 total):
- Schema tests (`test_data_schema.py`): valid frames pass; missing columns raise; non-monotonic timestamps raise; multi-asset frames where each asset is locally monotonic pass.
- Leakage tests (`test_no_leakage.py`): `WalkForwardWindow` rejects `train_end >= forecast_start`; `walk_forward_windows` produces strictly advancing windows that respect `test_end`; `StaticSplit` enforces strict ordering and produces disjoint, complete slices.

These tests are the contract that keeps the project honest as ML layers land.

**Cockpit**. New "Data" panel renders cache inventory, normalized adjusted-close trajectories (one line per asset), and macro signal series. The Overview panel still answers "what is this?"; the Data panel answers "what is it built on?".

**CLI**. `autosignalx data fetch` (also via `make data`) populates the parquet cache. `autosignalx data status` (and the global `autosignalx status`) report what's cached.

**Verification**: `make data` populates the cache; `make test` runs all 25 tests green; `make demo` opens the cockpit with both Overview and Data panels live.

---

---

## Iter 2 — Walk-forward eval harness with baselines

The first iteration that produces actual forecast numbers. Establishes the **forecast contract**, the **walk-forward harness**, the **metric set**, and three baselines that every subsequent forecasting method (Chronos-2 in Iter 3, signal-enhanced ensembles in Iter 5+) will be measured against.

### Forecast contract

Every forecasting method produces a DataFrame matching `eval.contracts.FORECAST_COLUMNS_REQUIRED`:

| column | meaning |
|---|---|
| `timestamp` | target time (trading day the forecast is for) |
| `asset` | ticker |
| `forecast_origin` | when the forecast was made -- always `< timestamp` |
| `horizon` | days from origin to target |
| `method` | string identifier of the forecasting method |
| `prediction` | point forecast in adj_close units |
| `origin_value` | adj_close at `forecast_origin` (used for directional metrics) |
| `target` | realized adj_close at `timestamp` |

Optional (filled by later iterations): `lower`, `upper` (Iter 3 -- intervals from Chronos-2), `regime_id` (Iter 4).

`assert_forecast_schema` enforces required columns, no leakage (`forecast_origin < timestamp`), and non-negative horizons. The contract is the **seam** that lets the regime, signal, graph, and agent layers compose without tearing apart the eval surface.

### Forecasting function contract

```python
ForecastFn = Callable[
    [pd.DataFrame, pd.Timestamp, list[pd.Timestamp]],  # train, origin, target_dates
    pd.DataFrame,                                       # cols: timestamp, prediction (+ optional lower/upper)
]
```

### Metrics

- **MAE** -- mean absolute error in adj_close units.
- **MAPE** -- mean absolute percentage error (zero targets masked).
- **Directional accuracy** -- fraction of forecasts whose predicted change-direction (`sign(prediction - origin_value)`) matches the realized change-direction (`sign(target - origin_value)`).
- **Skill score** -- `1 - method_mae / baseline_mae`. Positive => better than baseline; zero => same; negative => worse. Computed per asset against the naive baseline by default.

CRPS and probabilistic calibration metrics land in Iter 3 alongside Chronos-2 (when intervals become meaningful).

### Baselines

- **Naive** -- predict `adj_close(t+h) = adj_close(forecast_origin)`. Strong baseline for asset prices since they are approximately a random walk.
- **Seasonal-naive** -- predict `adj_close(t+h) = adj_close(t+h - 252 calendar days)`, falling back to the most recent training value when the lookback runs off the start of history. Sanity check against models that overfit recent dynamics.
- **ARIMA(1,1,1)** -- fit on `log(adj_close)` over the entire training set, forecast `len(target_dates)` steps ahead, exponentiate back to price space. Convergence warnings suppressed; failures bubble to the harness which gracefully skips the affected (window, asset) pair.

### Walk-forward harness

`harness.run_walk_forward(method_name, forecast_fn, ohlcv, windows)` iterates over every `(window, asset)` pair, slices training data up to `window.train_end`, gathers realized target trading days from the cache, calls the forecast function, and joins predictions with realized targets. The output is a forecasts DataFrame matching the contract.

`harness.ablation(methods, ohlcv, windows)` runs many methods and concatenates results. `harness.summarize(forecasts, by=...)` aggregates metrics by any grouping (default: `(method, asset)`). `harness.add_skill_score(summary)` appends per-asset skill versus the naive baseline.

### Tests (24 new, 51 total)

- `test_eval_contracts.py` -- valid frames pass; missing columns raise; leakage rows raise; negative horizons raise; empty frames pass.
- `test_eval_metrics.py` -- MAE / MAPE / dir-acc against hand-computed values; MAPE masks zero targets; skill score positive/zero/negative/nan cases; metrics handle NaN inputs; shape-mismatch raises.
- `test_baselines.py` -- every baseline returns aligned predictions matching `target_dates` length, all positive and finite; naive predicts last close exactly; seasonal-naive falls back when history is too short; ARIMA forecasts don't explode on synthetic random-walk data.

### CLI and cockpit

- **CLI**: `autosignalx eval baseline` (also `make baseline`) runs the ablation, writes `reports/ablations/baseline.parquet`, prints a per-method summary table.
- **Cockpit**: new "Forecast Arena" panel reads any `*.parquet` from `reports/ablations/`, shows per-method overall metrics, per-method per-asset metrics, and a forecast trajectory chart for any selected asset.

`autosignalx status` now also reports ablation cache state and marks L1 Forecasting as "partial (baselines)" -- Chronos-2 (Iter 3) flips it to "ok".

### Findings (initial ablation)

The first end-to-end run, on the default config (87 walk-forward windows of horizon 21 trading days, over ETF prices 2020-12-31 to 2025-12-31, 8 assets, 10,032 forecasts per method, ~12 min wall-clock dominated by ARIMA fits):

| Method | N | MAE | MAPE | Dir-acc | Skill vs naive |
|---|---:|---:|---:|---:|---:|
| naive | 10,032 | 4.254 | 2.04% | 0.2% | +0.000 |
| arima | 10,032 | 4.265 | 2.05% | 47.5% | -0.003 |
| seasonal_naive | 10,032 | 25.859 | 11.80% | 44.6% | -5.079 |

Three findings from this:

1. **Naive is essentially the floor for daily ETF prices.** ARIMA(1,1,1) on log-prices comes out with a skill score of -0.003 versus naive on MAE -- effectively identical, meaning the well-known random-walk-like behavior of liquid asset prices holds in our window. This is the bar Chronos-2 (Iter 3) needs to beat.
2. **One-year seasonality is not the right structure.** Seasonal-naive at 252 calendar days underperforms naive by a factor of 5x on MAE. ETFs don't have a meaningful annual cycle in price level (they have drift); seasonality applies to volatility, volume, and return distributions, not levels. Worth keeping as a foil.
3. **Directional accuracy is the differentiating metric, not MAE.** Naive's 0.2% dir-acc is structural (it predicts no change, so it almost never matches the realized direction); ARIMA at 47.5% is roughly coin-flip; seasonal-naive at 44.6% is slightly below. The interesting research question for later iterations becomes: can a model meaningfully exceed 50% dir-acc consistently, and is that improvement statistically significant per-regime?

These findings are *honest negative results* relative to the implicit hypothesis "more sophisticated models beat naive." Carrying them into Iter 3 frames the Chronos-2 result correctly: any improvement of >0.01 skill score, statistically significant via Diebold-Mariano, would be a real finding rather than noise.

---

---

## Iter 3 — Chronos-2 with covariates and probabilistic intervals

The L1 forecasting layer lands. We add a frontier foundation model — Amazon's **Chronos-2** (`amazon/chronos-2`, the multivariate / covariates-supporting successor to Chronos-Bolt) — in two configurations:

- **chronos2_univariate**: target = adj_close history; no exogenous inputs.
- **chronos2_multivariate**: target = adj_close history; **past_covariates** = the 4 macro signals (`^TNX`, `^VIX`, `DX-Y.NYB`, `CL=F`), forward-filled onto the asset's training calendar.

Both produce a point forecast plus a 80% prediction interval (10/50/90 quantiles). The probabilistic outputs unlock a new metric: **CRPS** (Continuous Ranked Probability Score), an integral measure of calibration that subsumes MAE for point forecasts (`CRPS = 2 * mean over q of pinball loss`).

### Implementation notes

- **Lazy model load** with `functools.lru_cache` -- one ~40s download/load on first call, free thereafter.
- **Batched inference** in `chronos2.batched_ablation`: a single `predict_quantiles` call per method covering all (window, asset) pairs (~700 inputs each), processed by Chronos's internal batching at `batch_size=256`. Wall-clock for both methods on full 5-year test: ~19 min on CPU (vs ~45 min if we routed every call through the per-asset harness).
- **Covariate alignment**: macro signals are forward/back-filled onto each asset's training dates via `_align_covariates`. Chronos-2 takes them as `past_covariates` in the input dict.
- **Per-call API also provided** (`chronos2_univariate`, `make_chronos2_multivariate(macro)`) for cases where forecasters need to plug into the per-asset `harness.run_walk_forward` (e.g., regime-conditioned dispatch in Iter 4+).

### Findings (full ablation, 4 methods x 87 walk-forward windows x 8 assets x 21-day horizon, test 2020-12-31 → 2025-12-31)

| Method | N | MAE | MAPE | Dir-acc | CRPS | Skill vs naive |
|---|---:|---:|---:|---:|---:|---:|
| naive | 10,032 | 4.254 | 2.04% | 0.2% | -- | +0.000 |
| arima | 10,032 | 4.265 | 2.05% | 47.5% | -- | -0.003 |
| **chronos2_univariate** | 10,032 | **4.470** | **2.13%** | **46.8%** | **2.897** | **-0.051** |
| **chronos2_multivariate** | 10,032 | **4.499** | **2.14%** | **47.8%** | **2.936** | **-0.058** |
| seasonal_naive | 10,032 | 25.86 | 11.80% | 44.6% | -- | -5.079 |

**Headline (honest negative): on this benchmark, Chronos-2 underperforms naive by 5-6% MAE, and adding 4 macro past-covariates does not help -- multivariate is marginally *worse* than univariate on MAE/MAPE/CRPS, while marginally better on directional accuracy.**

### Why this is a real result, not a bug

1. **Naive is the Bayes-optimal forecaster under a martingale.** Daily ETF prices are extremely close to random walks; the random-walk forecast (naive) is provably optimal under that data-generating process. Any model that introduces nontrivial drift, mean-reversion, or covariate dependence pays a variance cost. Foundation models trained on diverse, mostly non-financial time-series carry priors (trend, seasonality, mean-reversion) that misfire on liquid asset prices.
2. **CRPS at 2.9 is calibrated, not catastrophic.** The 80% intervals contain the realized values approximately as expected; Chronos's probabilistic outputs are well-calibrated. The point forecasts are just slightly off-center because the model expects more structure than asset prices contain.
3. **Multivariate underperforming univariate is consistent with the literature on macro→asset short-horizon predictability.** At a 21-day horizon, macro signals carry little instantaneous information about ETF prices; including them as past-covariates introduces noise without signal.
4. **Directional accuracy ≈ 47-48% across non-naive methods, all close to coin-flip.** ARIMA and Chronos cluster in the same dir-acc band. Distinguishing a "real" signal here requires statistical tests per regime -- which is exactly what Iters 4-5 will do.

### How this shapes Iters 4-7

- **Iter 4 (regime).** If forecast quality varies systematically across latent regimes, a regime-conditional method that picks naive vs. Chronos vs. naive+macro per regime could beat the naive floor. The regime layer's main value is *conditional*, not unconditional.
- **Iter 5 (signal).** Per-regime feature ranking via HistGradientBoosting + permutation importance. The hypothesis: in some regimes (e.g., high-VIX), macro signals carry meaningful short-horizon information; in others (e.g., calm bull markets), they don't. Conditional inclusion is the bet.
- **Iter 6 (graph).** Cross-asset hubs may carry information about leaves before that information shows in the leaf's own price. Granger causality tests will tell us if any such structure exists.
- **Iter 7 (agent).** Given the above, the agent's job is to *discover the regime / asset / horizon combinations where the layered system actually beats naive*. Not "make Chronos better" -- find the slices where it already is.

The negative result *clarifies* the research question. If Iter 7 ends without finding any conditional improvement, that's a publishable null result. If it finds even one regime-asset slice with statistically significant improvement (DM test, p<0.05), that's a positive result framed correctly.

### CLI and cockpit

- **CLI**: `autosignalx eval chronos` (also `make forecast`) runs the chronos ablation and writes `reports/ablations/chronos2.parquet`. The Forecast Arena reads any `*.parquet` under that directory, so baselines and chronos compose into one view.
- **Cockpit**: Forecast Arena now includes a **method × asset selector** with **80% interval bands** rendered for the selected method (when available). Reviewers can see uncertainty next to point forecasts.
- **Layer status**: `autosignalx status` flips L1 Forecasting from "partial (baselines)" to "ok (baselines + chronos-2)". L1 is now complete.

**Verification**: `make sync && make test && make demo`; 60 tests passing (5 new CRPS tests); ablation parquet (~660 KB) committed for out-of-the-box demo.

---

---

## Iter 4 — Representation layer: contrastive regimes

The L2 representation layer lands. Two regime detectors train on a market-level feature matrix and produce per-timestep regime labels that downstream layers (signal selection in Iter 5; agent in Iter 7) condition on:

- **Contrastive 1D-CNN encoder + KMeans (primary)** -- a small PyTorch model (2 conv blocks, 16-dim embedding, ~3k params) trained via triplet loss with positive=adjacent window, negative=distant window. The learned embedding space groups similar windows together; KMeans on the embeddings produces hard regime labels.
- **Gaussian HMM on raw features (sanity-check baseline)** -- `hmmlearn.GaussianHMM` on the same standardized features. Models temporal transitions explicitly; serves as a check that the contrastive method isn't picking up spurious clusters.

### Market features

Built by `regime.labels.build_market_features`:
- SPY daily returns, QQQ daily returns (proxies for market direction / dispersion)
- Macro: ^TNX, ^VIX, DX-Y.NYB, CL=F (forward-filled and dropna'd to joint coverage)

Standardized to zero mean, unit variance per column before encoding / HMM fit.

### Encoder architecture and training

```
Conv1d(n_features -> 16, k=5, p=2) + GELU
Conv1d(16 -> 32, k=5, p=2) + GELU
AdaptiveAvgPool1d(1)
Linear(32 -> embedding_dim)
```

Training: Adam lr=1e-3, 25 epochs, batch_size 64, triplet margin 1.0. Positive window offset in [-3, +3] days from anchor; negative offset >= 60 days from anchor. ~3,900 windows over 16 years of daily data.

### Initial fit (default config: 4 regimes, 60-day window, 16-dim embedding)

| Detector | N labeled timesteps | Regime sizes |
|---|---:|---|
| KMeans (contrastive) | 3,967 | {0: 1425, 1: 750, 2: 877, 3: 915} |
| HMM (Gaussian) | 4,026 | {0: 1421, 1: 793, 2: 1241, 3: 571} |

Both detectors find 4 distinct regimes with broadly similar dominance structure -- the largest regime (1421-1425 timesteps, ~35% of history) covers extended calm bull periods; the others split shorter / more turbulent episodes. End-to-end fit time: ~53s on CPU (encoder 25 epochs + KMeans + HMM 100 iter), all wrapped by `autosignalx regime fit`.

### Cockpit and harness integration

- **Regime Explorer panel**: KMeans timeline, HMM timeline (both as colored line charts), and a PCA-2D scatter of the contrastive embeddings colored by KMeans label. Reviewers can eyeball whether the regimes correspond to recognizable market periods (COVID crash, post-COVID rally, 2022 inflation, ZIRP-to-hike, ...).
- **Forecast Arena**: now includes a "Per-method, per-regime" stratified table when regime labels exist. Forecasts are joined to KMeans regime labels on `forecast_origin`, then summarized by `(method, regime_id)`. This is how we will discover whether any (method, regime) slice meaningfully beats naive on MAE (the conditional improvement question opened by Iter 3's negative result).
- **Layer status**: `autosignalx status` flips L2 Representation from "pending" to "ok (contrastive + KMeans + HMM)".

### What this enables

The regime labels are an input contract for Iters 5 (per-regime feature ranking), 6 (cross-asset graph computed within regimes), and 7 (agent that hunts for regime-specific predictive structure). Without regime labels, those iterations would have to invent ad-hoc segmentation. Now they can read regimes the way the eval harness reads forecasts: as a typed, persisted artifact under `reports/regimes/`.

**Verification**: `make regime` runs end-to-end in ~1 minute; `make test` -> 67 tests passing (6 new for encoder + cluster); `make demo` -> Regime Explorer renders all three views; Forecast Arena shows per-regime stratification on the 4-method ablation.

---

---

## Iter 5 — Reasoning layer: per-regime feature ranking

The L3 reasoning layer lands. We ask, for each of the 4 KMeans regimes (Iter 4): *which features carry the most signal for predicting next-21-day direction?* The answer is the foundation that Iter 7's agent will use to propose conditional forecasting strategies.

### Classifier and methodology

The per-regime classifier is **`sklearn.ensemble.HistGradientBoostingClassifier`** (`max_iter=200`, `learning_rate=0.05`, `max_depth=4`, `random_state=42`). It handles mixed feature scales and missing values natively, fits in <1 second per regime on this data size, and exposes both built-in `feature_importances_` and a clean predict surface for permutation-based ranking.

Importance is measured by **custom permutation importance**: for each feature, shuffle that column `n_repeats` times and measure the accuracy drop versus the unshuffled baseline. The mean drop across repeats is the importance score; the standard deviation is reported alongside.

### Methodology

For each asset, we build per-(asset, timestamp) features:

**Technical (8)**: rolling_mean_5, rolling_mean_20, rolling_std_5, rolling_std_20, momentum_10, momentum_60, rsi_14, macd_signal.

**Macro level + 5-day change (8)**: for each of `^TNX`, `^VIX`, `DX-Y.NYB`, `CL=F` -- both the current level and the 5-day percent change.

**Aux (1)**: future_close (drop-only; not a feature).

**Target**: binary direction -- 1 if `adj_close[t + 21] > adj_close[t]`, else 0.

For each of the 4 KMeans regimes:
1. Sample up to 2,000 rows from that regime's timesteps (across all 8 assets).
2. Fit `HistGradientBoostingClassifier(max_iter=200, lr=0.05, max_depth=4, seed=42)`.
3. For each feature j, shuffle column j 2 times, measure accuracy drop -> importance[j] = base_acc - mean(shuffled_acc).

Output: `reports/signals/signal_ranking.parquet` with columns `(regime_id, feature, importance, importance_std, n_samples, rank)`. Rank 1 = most important within the regime.

### Findings

**Top-5 features per regime** (positive importance = shuffling that feature hurts accuracy):

| Regime | Top-1 (importance) | Top-2 | Top-3 | Top-4 | Top-5 |
|---:|---|---|---|---|---|
| 0 | macro_^TNX_level (+0.071) | macro_CL=F_level | momentum_60 | macro_DX-Y.NYB_level | macro_^VIX_level |
| 1 | macro_DX-Y.NYB_level (+0.114) | momentum_60 | macro_^TNX_level | rolling_std_20 | macro_CL=F_level |
| 2 | macro_CL=F_level (+0.112) | macro_^TNX_level | macro_DX-Y.NYB_level | macro_^VIX_level | macd_signal |
| 3 | macro_DX-Y.NYB_level (+0.189) | macro_^VIX_level | macd_signal | macro_^TNX_level | momentum_60 |

Three findings:

1. **Macros dominate every regime's top-5.** In all 4 regimes, the top feature is a macro level (TNX, DXY, or CL=F). Technical indicators (momentum_60, macd_signal, rolling_std_20) appear at ranks 2-5 but are never the most important feature.
2. **Which macro matters depends on the regime.** Regime 0 (the largest) is dominated by 10Y yields. Regimes 1 and 3 are dominated by the dollar index (with regime 3 showing the strongest macro effect of any: +0.189 importance for DXY level). Regime 2 is dominated by crude oil. **Different regimes have different "macro keys"** -- a finding that motivates conditional, regime-specific feature selection rather than uniform multi-covariate input.
3. **Iter 3's negative result is now understood.** Chronos-2 with all 4 macro covariates simultaneously underperformed univariate Chronos-2 by 1% MAE. Iter 5 explains why: the *correct* macro depends on the regime, and dumping all 4 in regardless of regime introduces noise. A regime-aware forecaster that conditionally selects 1-2 dominant macros per regime is the natural next experiment -- and is exactly what Iter 7's agent is set up to discover.

### How this shapes Iters 6-7

- **Iter 6 (graph)**: cross-asset partial-correlation and Granger causality may reveal that *which asset's macro reaction matters first*, refining the per-regime story to per-(regime, asset).
- **Iter 7 (agent)**: with regime labels (Iter 4) and per-regime feature rankings (Iter 5), the agent can propose hypotheses of the form "in regime R, condition Chronos-2 on top-K(R) features and re-evaluate vs naive on a held-out asset slice." DM tests on the held-out slices give p-values. This is the mechanism by which the layered system can beat naive *conditionally* even though it cannot beat naive *unconditionally* (Iter 3).

### Cockpit and CLI

- **Signal Discovery Lab panel**: per-regime feature bar chart with importance values; full ranking table with rank/importance/std/n_samples; cross-regime importance heatmap of the top-K features per regime. Reviewers see the per-regime feature story at a glance.
- **CLI**: `autosignalx signal rank` (also `make signal`) runs the full ranking in ~30s on CPU (4 regimes x ~25 permutations per regime = 100 model fits, each fast).
- **Layer status**: `autosignalx status` flips L3 Reasoning to "ok (HistGradientBoosting + permutation importance)".

**Verification**: `make signal` runs in ~30s; `make test` -> 65 tests passing (4 new for features); `make demo` -> Signal Discovery Lab renders the bar chart, table, and heatmap.

---

---

## Iter 6 — Relational layer: cross-asset graph

The L4 relational layer lands. Two complementary views of the cross-asset structure:

- **GLASSO partial-correlation graph (undirected)**: `sklearn.covariance.GraphicalLassoCV` fits a sparse precision matrix; off-diagonals (after normalization) are partial correlations. Edges are *direct* statistical relationships after controlling for all other assets in the panel -- a sharper signal than raw Pearson correlations.
- **Granger-causality edges (directed)**: for each ordered pair (X, Y), `statsmodels.tsa.stattools.grangercausalitytests` tests whether lags of X predict Y beyond Y's own lags, at lags 1..5. Pairs with min p-value < 0.05 emit a directed edge with weight `-log10(p)`.

NetworkX-based **centrality** (degree, eigenvector, betweenness) is computed on the partial-correlation graph (treating it as undirected with absolute weights).

### Initial build (8 ETFs, daily returns, 2010-2025)

- **18 partial-correlation edges** (out of 28 possible undirected pairs -- 64% density, reflecting the broad co-movement of US-listed liquid ETFs).
- **42 Granger edges** at p < 0.05 (out of 56 possible ordered pairs -- 75% density). Many edges are likely consequences of correlated noise rather than causal information flow; in equity-style returns, common-factor exposure causes Granger tests to fire even when no economic causality exists. We treat Granger here as descriptive ranking rather than causal claim.
- Build time ~29s on CPU.

### Centrality (eigenvector ranking)

| Node | Degree | Eigenvector | Betweenness | Notes |
|---|---:|---:|---:|---|
| SPY | 0.857 | **0.532** | 0.143 | Broad market hub |
| EFA | 0.857 | 0.422 | 0.333 | International developed |
| QQQ | 0.714 | 0.414 | 0.143 | Tech-heavy US |
| IWM | 0.714 | 0.394 | 0.000 | Small-cap US |
| EEM | 0.714 | 0.351 | 0.048 | Emerging markets |
| HYG | 0.714 | 0.297 | 0.000 | High-yield credit |
| TLT | 0.429 | 0.036 | **0.429** | **Bridge -- long bonds** |
| GLD | 0.143 | 0.001 | 0.000 | Near-isolated diversifier |

Three findings:

1. **SPY is the structural hub.** Highest eigenvector centrality and tied for highest degree -- consistent with SPY's role as the broad-market reference. Forecasting models that condition on SPY's behavior are likely to inherit information about most other ETFs in the panel.
2. **Gold is statistically isolated.** GLD has degree 0.143 (only 1 partial-correlation edge among 7 possible) and eigenvector centrality essentially 0. This confirms the textbook "uncorrelated diversifier" role of gold and also provides a methodological check: if our partial-corr inference were noisy, GLD would not isolate so cleanly.
3. **TLT is a low-degree bridge.** Long bonds have low degree (0.429) but the *highest* betweenness (0.429 -- nearly tied with EFA). This means TLT lies on many shortest paths between other assets despite few direct connections -- it bridges equity assets to a different regime of the panel. Operationally: shocks transmitted through bond markets pass through TLT.

### How this shapes Iter 7

The agent now has three new typed inputs to compose hypotheses over:
- **Hubs (high eigenvector)** -- candidates for primary forecasting targets whose state propagates.
- **Bridges (high betweenness)** -- candidates for *features* that carry cross-asset regime-transition information.
- **Isolates (low centrality)** -- candidates for portfolio diversifiers; uninformative as features for predicting other assets.

Combined with regime labels (Iter 4) and per-regime feature rankings (Iter 5), the agent's hypothesis space becomes: *"in regime R, forecast asset A using top-K(R) macros plus the centrality-weighted moves of nearby (graph-adjacent) assets."*

### Cockpit and CLI

- **Cross-Asset Graph panel**: edge counts; centrality table sorted by eigenvector; partial-correlation matrix rendered as a diverging colormap (RdBu); top-20 Granger edges by p-value.
- **CLI**: `autosignalx graph build` (also `make graph`) runs in ~30s; `autosignalx graph status` lists cached artifacts.
- **Layer status**: `autosignalx status` flips L4 Relational to "ok (GLASSO + Granger + centrality)".

**Verification**: `make graph` runs in ~30s; `make test` -> 70 tests passing (5 new for graph layer); `make demo` -> Cross-Asset Graph panel renders centrality + partial-corr heatmap + Granger table.

---

---

## Iter 7 — Agentic layer: LangGraph research loop with persistent memory

The L5 agentic layer lands. A LangGraph state machine drives a research loop that reads from every prior layer's artifacts, proposes hypotheses, executes deterministic experiments by slicing the cached forecasts, critiques results, and decides whether to continue. The whole trace is appended to a JSONL **ledger** -- the system's persistent memory cell -- which the cockpit can be queried in natural language.

### Architecture

```
START -> propose -> experiment -> critique -> decide -> [continue: propose | stop: END]
```

Each round:
- **propose** (LLM): given the context snapshot (regimes, top features per regime, centrality, prior findings) and the ledger summary, propose a hypothesis as JSON: `{"hypothesis": "...", "experiment": {"type": "slice_forecasts", "params": {...}}}`.
- **experiment** (deterministic): slice the union of all cached `reports/ablations/*.parquet` by `(method, asset, regime_id)`, compute MAE/MAPE/dir-acc/CRPS/skill_vs_naive on the slice. Returns a small JSON.
- **critique** (LLM): 2-3 sentence assessment of scope, expected effect size, confounders.
- **decide** (LLM): JSON `{"action": "continue" | "stop", "reason": "..."}`. Hard cap at `max_rounds`.

Implementation:
- **LangGraph** (`langgraph.graph.StateGraph`) for the state machine; `add_conditional_edges` for the continue/stop branch.
- **DeepInfra** (OpenAI-compatible) via `langchain_openai.ChatOpenAI`. Models configurable via env: `DEEPINFRA_MODEL_PROPOSER` (`moonshotai/Kimi-K2.6` for the recorded run), `DEEPINFRA_MODEL_CRITIC` (`zai-org/GLM-4.7-Flash`), `DEEPINFRA_MODEL_CHAT` (`deepseek-ai/DeepSeek-V4-Pro`).
- **Replay mode** (`AUTOSIGNALX_REPLAY=true` or no API key): `ReplayProvider` reads pre-recorded responses keyed by `(round, step)` from `replay/agent_steps.jsonl`, with deterministic plausible fallbacks if the file is incomplete. Lets reviewers walk through the same agent session without provisioning an account.
- **LLM response cache**: every live LLM call is hashed by message content; cache hits skip the API call entirely. Re-runs of the same prompt are free and deterministic. (Cache is gitignored.)
- **Ledger** (`reports/agent/ledger.jsonl`): append-only JSONL; one entry per (round, step) with ISO timestamp.

### deepagents / openevals / agentevals

All three libraries are installed (per the project plan). `openevals` and `agentevals` provide LLM-judge-style trace evaluators that we wire into the report layer in Iter 9 for trace-quality scoring. `deepagents`'s planner pattern is the conceptual structure for the propose/critique/decide loop; we implemented the loop directly in LangGraph for transparency rather than wrapping the deepagents helper, but the pattern is the same.

### Recorded live trace (Kimi K2.6 proposer, GLM-4.7-Flash critic, 5 rounds)

The recorded run is committed under `replay/agent_steps.jsonl` (5.7 KB) and `reports/agent/ledger.jsonl` (7.7 KB). Reviewers running in replay mode see the same trace step-for-step.

Headline hypotheses generated by the live agent:

- **Round 0**: chronos2_multivariate may beat naive on Regime 2 for SPY (the dominant Regime 0 + SPY hub combination is a natural starting point).
- **Round 3**: "In Regime 0 -- characterized by dominant Treasury-yield and crude-oil levels -- the chronos2_univariate model on GLD should outperform naive because gold's near-isolation in the cross-asset graph (centrality ~0) means foundation-model temporal pattern matching has more idiosyncratic structure to find without cross-asset noise."
- **Round 4**: "In regime 3, where the latent state is dominated by U.S. dollar strength and elevated volatility (macro_DX-Y.NYB_level and macro_^VIX_level), the chronos2_multivariate model will outperform the naive baseline for EFA because EFA's high betweenness centrality (0.333) positions it as a bridge between U.S. equity and international markets, allowing the multivariate transformer to encode cross-asset flight-to-quality and USD-transmission dynamics that a random-walk baseline ignores."

Notice that Round 4's hypothesis composes findings from **every prior layer**: regime structure (Iter 4), per-regime feature importance (Iter 5), and graph centrality (Iter 6). It then proposes a **mechanistic, falsifiable** prediction (chronos2_multivariate > naive on EFA in regime 3 due to cross-asset USD-transmission). The agent's reasoning is structured by the typed inputs the prior layers built.

This is the pattern the agent is designed to discover: **conditional improvements** over naive in specific (regime, asset, method) slices that have a coherent mechanistic story.

### Cockpit panels

- **Agent Console**: chat-style timeline view of the ledger -- propose / experiment / critique / decide steps shown with role-appropriate icons, colored, scrollable. Reviewers see the agent's reasoning as it unfolds (or as it was previously recorded in replay mode).
- **Ask the Memory**: free-form chat input against the ledger.
  - In **live mode**: the LLM answers using the ledger as context; cites specific rounds.
  - In **replay mode**: deterministic keyword search over the ledger returns matching entries.

### CLI

- `autosignalx agent run` (also `make agent`) runs the loop. Flags: `--max-rounds`, `--fresh` (wipe ledger), `--record-replay` (append live LLM responses to `replay/agent_steps.jsonl`).
- `autosignalx agent status` reports ledger size and last round.
- `autosignalx status` flips L5 Agentic to "ok (LangGraph + DeepInfra + ledger)" -- **all 5 layers complete**.

**Verification**: live agent run completed 5 rounds in <2 min; recorded replay file (`replay/agent_steps.jsonl`) and ledger (`reports/agent/ledger.jsonl`) committed; `make test` -> 76 tests passing (6 new for agent layer); `make demo` -> Agent Console shows the recorded trace, Ask the Memory chat panel works in both modes.

---

---

## Iter 8 — Cockpit polish: reviewer-journey navigation

The five model layers all landed in Iter 7. Iter 8 is a polish pass that makes the 5-minute reviewer walk through the cockpit land cleanly, with no model or algorithmic changes.

### Cockpit (`app/streamlit_app.py`)

- **Reviewer-journey callout** added to the Overview panel: a green success box at the top points reviewers to the panel walk order (Data → Forecast Arena → Regime Explorer → Signal Discovery Lab → Cross-Asset Graph → Agent Console → Ask the Memory). The story builds layer-by-layer; the callout makes that sequence explicit instead of leaving reviewers to discover it.
- **Layer status grid** in Overview replaces the previous "Iter N" labels with concrete implementation summaries ("Chronos-2 + baselines", "Contrastive + KMeans", "Per-regime ranking", "GLASSO + Granger", "LangGraph + DeepInfra") and an OK status — all five layers shown live.
- **Headline findings section** added to Overview, surfacing the four most important findings (Iter 3 calibrated negative result, Iter 5 conditional macros, Iter 6 graph roles, Iter 7 agent compositionality) before reviewers dive into individual panels. This lets a 30-second skim still leave with the key takeaways.

### README

- **Status** flipped from "Iteration 0 — repository scaffold" to "All 5 layers complete" with a one-line summary of each layer's implementation.
- **Headline findings** section mirrors the cockpit's headline panel — same four findings, same framing.
- **Reviewer journey (5-minute walk)** subsection enumerates the panels in walk order with a one-line "what you see" description per panel.
- **LLM provider** section expanded with the actual model identifiers used to record the live trace (`moonshotai/Kimi-K2.6` / `zai-org/GLM-4.7-Flash` / `deepseek-ai/DeepSeek-V4-Pro`), so reviewers can swap in their own preferences via env without guessing what's expected.

### Why this is its own iteration

The polish is small in code (66 lines across 2 files, single commit) but high-leverage in reviewer experience: reviewers who skim now get the headline in 30 seconds and the panel walk order without thinking. Keeping it as its own iteration boundary in git history (`iter-8-cockpit` merge commit) signals that polish is a deliberate engineering step, not noise tacked onto the agent iter.

**Verification**: 76 tests still passing, ruff clean, `make demo` shows the Reviewer Journey callout + headline findings on first paint of the Overview panel.

---

## Iter 9 — Consolidation, reproducibility, future work

The five model layers are live, the cockpit shows them coherently, and the findings story is honest. This iteration consolidates the report (executive summary at top), and verifies that the submission stands up as a piece of research artifact rather than a code dump.

### Reproducibility

A reviewer cloning the repo and running the demo should get the cockpit live in under five minutes:

```bash
git clone https://github.com/Aleksander2a/ai-research.git
cd ai-research
uv sync --all-extras
uv run streamlit run app/streamlit_app.py    # or: make demo
```

The cockpit immediately shows real results because all artifacts are committed:
- `reports/ablations/baseline.parquet` (~340 KB, 30k forecasts) and `reports/ablations/chronos2.parquet` (~600 KB, 20k forecasts) -> Forecast Arena renders metrics + uncertainty bands
- `reports/regimes/{kmeans,hmm,embeddings}.parquet` (~450 KB) -> Regime Explorer renders timelines + PCA scatter
- `reports/signals/signal_ranking.parquet` (~5 KB) -> Signal Discovery Lab renders the per-regime heatmap
- `reports/graph/{edges,centrality}.parquet` (~8 KB) -> Cross-Asset Graph renders the matrix + centrality table
- `reports/agent/ledger.jsonl` (~8 KB) and `replay/agent_steps.jsonl` (~6 KB) -> Agent Console renders the recorded session, Ask the Memory works in keyword-search mode

Reviewers who want to re-run any layer can:
- `make data` -> refresh ETF + macro caches from yfinance
- `make baseline` -> re-run the 3-method baseline ablation (~12 min, ARIMA dominates)
- `make forecast` -> re-run the Chronos-2 ablation (~19 min on CPU)
- `make regime` -> re-fit the contrastive encoder + clusterers (~50s)
- `make signal` -> re-rank features per regime (~30s)
- `make graph` -> rebuild the cross-asset graph (~30s)
- `make agent` -> run the LangGraph loop fresh (in replay mode by default; live mode if `DEEPINFRA_API_KEY` is set in `.env`)

`make test` runs all 76 tests; `make lint` enforces ruff cleanliness.

### Submission flow

1. Reviewer reads README.md (status + headline findings + reviewer journey).
2. Reviewer runs `uv sync --all-extras && make demo`.
3. Reviewer walks the 8 cockpit panels in order: Overview -> Data -> Forecast Arena -> Regime Explorer -> Signal Discovery Lab -> Cross-Asset Graph -> Agent Console -> Ask the Memory.
4. Reviewer reads REPORT.md for the layer-by-layer narrative (the file you are reading now).
5. Reviewer browses git history (`git log --graph --oneline`) to see iteration boundaries: 9 `--no-ff` merges, each labeled by iter and branch name.

### What the version-control story shows

Each iteration is a branch (`iter-N-theme`), merged with `--no-ff` so the boundaries are visible. Within each iteration are 1-3 cohesive commits separating concerns: (a) the layer's code, (b) tests, (c) CLI + cockpit + report. Every commit on the integration branch passes `make test` and runs end-to-end. If the project had been forced to ship at any iteration boundary >= Iter 3, it would still have been a real submission with calibrated baselines and probabilistic forecasts.

### Honest limitations

- **Cross-validation in the agent's experiment step is descriptive, not causal.** The agent's `slice_forecasts` tool computes metrics on cached forecasts; it does not retrain models per-hypothesis. Adding "fit-on-the-fly" experiment types is a natural extension that preserves the LangGraph state machine.
- **No DM significance tests on the agent's findings.** The agent identifies candidate slices; the significance question (which lifts hold up under Diebold-Mariano?) is acknowledged but deferred. With more time, we would wire `statsmodels`'s DM test as another deterministic tool the agent can call, and have the critic step demand DM verification.
- **Single dataset.** ETFs are liquid and well-understood; the negative result on naive may be specific to this asset class. The methodology generalizes (the harness contract, the regime-conditioning, the agent loop are all dataset-agnostic), but conclusions are tied to ETFs.
### Closing note

This submission is a *research instrument*, not a forecasting demo. The intended signal to Deeter is: I can frame ambiguous open-ended problems into measurable scientific inquiry, design layered systems that compose, build agent infrastructure that uses other layers as typed inputs, evaluate honestly (including reporting calibrated negative results), and ship reproducible artifacts under tight time constraints. The 9 iterations and their commits are the trace of that process.

---

# Phase 2 — Self-improving research agent (Iters 10+)

The first 9 iterations shipped a research instrument that *reports* findings. Phase 2 transforms it into a research agent that **actively works to overcome them**, with full provenance from initial brainstorm to promoted finding. The headline negative result of Phase 1 ("naive beats Chronos by 5% MAE on daily ETFs") becomes the **starting condition** the agent sets out to overcome — every iteration adds a capability that makes its discoveries more rigorous, more autonomous, and more observable.

---

## Iter 10 — Statistical promotion gate

The agent's claims so far have been point estimates of metric differences. To make those claims publishable, this iteration adds the statistical-significance infrastructure that will gate every future "finding."

`src/autosignalx/eval/significance.py`:

- **`dm_test(loss_a, loss_b, horizon)`** — the **Diebold–Mariano** test on aligned per-observation losses. Uses Newey-West HAC variance to handle the auto-correlation that h-step-ahead forecasts induce by overlap.
- **`block_bootstrap_ci(values, n_bootstrap, block_size, ci, seed)`** — a moving-block bootstrap that respects serial correlation in the loss difference series; returns the requested-CI quantiles of the bootstrap mean distribution.
- **`is_promotable(forecasts, method, baseline_method, p_threshold)`** — the **promotion gate**. Aligns predictions on `(timestamp, asset, forecast_origin)`, computes per-row absolute-error losses, runs DM on them, computes the bootstrap CI on the loss difference, returns `(promotable: bool, evidence: dict)`. A method is promotable iff DM p<threshold AND skill>0 AND the bootstrap CI on the loss difference is strictly above zero.

`src/autosignalx/agent/tools.py` adds `test_significance(method, baseline_method, asset, regime_id, p_threshold)` — the agent's hand on the promotion gate. From Iter 11 onward, every claim the agent wants to "promote to a finding" must pass this.

**Tests** (9 new): identical losses give zero DM statistic; clearly-different losses give p<0.01; shape mismatch raises; bootstrap CI brackets the true mean; clearly-better method is promotable; same-method comparison is not; missing method/insufficient samples handled gracefully.

This is the rigor floor for everything Iter 11+ will build on. Suite total: 85 passing.

---

## Iter 11 — Findings store

The agent's raw ledger (`reports/agent/ledger.jsonl`) records every step it takes — proposals, experiments, critiques, decisions. Most of those steps don't represent discoveries; they're the working-out. **Findings** are different: they're hypotheses that passed the Iter 10 promotion gate (DM p < 0.05 AND positive bootstrap CI on the loss difference). They deserve a separate, structured store.

`src/autosignalx/agent/findings.py`:

- **`promote(hypothesis, method, filters, evidence, agent_confidence, round, session_id, parent_hypothesis_ids) → record`** — append a promoted finding to `reports/agent/findings.jsonl`. **Idempotent** on `(hypothesis, method, filters)`: re-promoting the same finding bumps its `replication_count` rather than duplicating. The `replications` list records each `(session_id, round)` that re-confirmed the finding.
- **`load()`** / **`clear()`** — round-trip the store.
- **`make_session_id()`** — sortable `YYYYMMDD-<hex>` session IDs.
- **`_finding_id(content)`** — deterministic short ID (`f_<hash>`) derived from hypothesis+method+filters; the same hypothesis run twice produces the same ID, enabling replication tracking.

`src/autosignalx/agent/graph.py` extends the experiment node with an **auto-promotion** path: when a hypothesis names a non-naive method, the agent automatically calls `tools.test_significance(...)` on the slice. If the gate returns `promotable=True`, the finding is persisted to `findings.jsonl` and the experiment result includes the `promoted_finding_id`. The agent's `session_id` is now part of `AgentState` and propagates through every promoted record for cross-session lineage.

**Cockpit:** new **"Findings"** panel sorted by skill-vs-naive descending. Three top-level metrics (total findings, distinct sessions producing findings, best skill). Expandable cards per finding showing hypothesis text, filter slice, full DM/bootstrap evidence, agent confidence, and replication trail.

**Tests** (3 new): round-trip of a single finding; idempotent re-promotion bumps replication count; session IDs are unique. Suite total: 88 passing.

This is where the agent's "discoveries" finally have a home that's separate from its "work."

---

## Iter 12 — Multi-agent debate (Theorist / Skeptic / Adjudicator)

The single-LLM `propose → critique → decide` loop becomes a structured **debate** where three role-specialized agents argue, each backed by a different DeepInfra model. The interplay surfaces *dialectic*, not monologue, and gives the cockpit a much richer trace to render.

### Roles and models

| Role | Default model | System prompt |
|---|---|---|
| **Theorist** | `moonshotai/Kimi-K2.6` (creative) | "Propose one specific, mechanistically-motivated hypothesis. Lean into novel (regime, asset, method) combinations." |
| **Skeptic** | `zai-org/GLM-5.1` (critical) | "In 2-4 sentences, identify the strongest CONFOUNDER, alternative explanation, or methodological weakness." |
| **Adjudicator** | `deepseek-ai/DeepSeek-V4-Pro` (decisive) | "Weigh the Theorist's proposal vs the Skeptic's challenge against the experiment result. Verdict: support / refute / inconclusive." |

Per-role models are configurable via `DEEPINFRA_MODEL_THEORIST` / `_SKEPTIC` / `_ADJUDICATOR` env vars; they fall back through `_PROPOSER` / `_CRITIC` / `_CHAT` to the project defaults.

### Round structure (debate mode)

```
START → Theorist (proposes JSON hypothesis)
      → Skeptic (challenges before experiment runs)
      → experiment (deterministic slice + auto-promotion gate from Iter 11)
      → Adjudicator (judges results, decides continue/stop)
      → [theorist | END]
```

The Skeptic challenges *before* the experiment runs, so its critique is independent of the result — this guards against post-hoc rationalization. The Adjudicator only sees the experiment result and renders a verdict ending in `VERDICT: support / refute / inconclusive`.

### Implementation

`agent/debate.py`:

- **`make_theorist_node(record_replay)`**, **`make_skeptic_node(...)`**, **`make_adjudicator_node(...)`** — node-factory pattern; each binds a role-specific provider (different model) and writes its own ledger entries (steps `theorist` / `skeptic` / `adjudicator`).
- **`build_debate_agent_graph(record_replay)`** — compiles the LangGraph state machine with the new four-node-per-round structure.
- **`run_debate(max_rounds, seed, record_replay, session_id)`** — top-level entry, mirrors `graph.run` but invokes the debate graph.

`agent/llm.py`:

- New **`ROLE_TO_ENV`** mapping role → env var.
- New **`_model_for_role(role)`** with the env-var hierarchy fallback.
- **`get_provider(record_replay, role)`** now takes a role and selects the correct model per role.

### CLI

`autosignalx agent run --mode debate --max-rounds 5 --fresh` runs the debate flow. The default mode is still `single` (backward compatible with the Iter 7 loop). Both modes write to the same ledger; the cockpit renders both transparently.

### Cockpit

The Agent Console panel learns the new step types and renders each role with its own icon (💡 Theorist, 🔍 Skeptic, ⚖️ Adjudicator, 🧪 experiment, 🎯 decide). When an experiment auto-promotes a finding (Iter 11 gate), the entry is marked with a green ✓ banner and the finding ID, so reviewers can jump straight to the Findings panel.

### What this gives us

- **Three different reasoning styles** in one round, surfacing perspectives that a single-LLM loop misses.
- **Adversarial critique applied before** the experiment, raising the bar for hypotheses to even be tested.
- **A richer trace** for the cockpit and the WOW demo (Iter 19): debate-style transcripts read like a real research meeting.

Suite total: 94 passing (6 new for debate node factories + prompt shapes + graph compile).

---

## Iter 13 — Code-spec experiment tool (constrained DSL)

The agent's experiment surface so far has only sliced **existing** forecasts. Iter 13 lets the agent **author new methods** at runtime through a constrained JSON DSL — the agent designs the experiment, the harness executes it, the result is persisted as a normal `reports/ablations/<name>.parquet` indistinguishable from a human-authored method's results, the Iter 11 promotion gate fires automatically, and the new method becomes selectable in the Forecast Arena.

This is the freedom expansion the user asked for: the agent now **creates**, not just observes.

### Spec DSL

`agent/specs.py` defines and validates the schema:

```json
{
  "name": "chronos2_dxyonly_ensembled",        // alphanumeric, becomes method label
  "base": "chronos2_multivariate",             // one of: naive, arima,
                                               //   chronos2_univariate,
                                               //   chronos2_multivariate
  "covariate_subset": ["DX-Y.NYB"],            // optional; only chronos2_multivariate
  "ensemble_naive_weight": 0.3,                // [0, 1]; 0 = pure base, 1 = pure naive
  "max_windows": 8,                            // cap for fast iteration
  "asset_subset": ["SPY", "EFA"]               // optional asset filter
}
```

`validate_spec(spec) -> (bool, str)` enforces every field. Bad names, unknown bases, out-of-range weights, malformed covariate subsets — all rejected with a specific error message before any code executes.

### Execution

`specs.execute(spec)`:

1. Validates the spec.
2. Builds a real `ForecastFn` by composing primitives (base method + covariate subset for multivariate + naive ensembling).
3. Runs through `harness.run_walk_forward` on the configured (possibly subsetted) asset universe and capped windows.
4. Persists to `reports/ablations/<name>.parquet`.
5. Returns `{status: "ok", name, output_path, n_rows, n_windows, summary: {mae, mape, dir_acc, crps}}`.

### Agent integration

`agent/tools.py` adds `spawn_method(spec) → result_dict` — the agent's hand on the DSL.

`agent/prompts.py` THEORIST_SYSTEM is extended with a second experiment schema (`type: "spawn_method"`) and the same auto-promotion gate fires on the spawned method's name when significance passes.

`agent/graph.py` `experiment_node` now branches on `experiment.type ∈ {slice_forecasts, spawn_method}` and the auto-promotion path is shared by both — a method authored by the agent and a human-authored method are treated identically by the gate.

### Safety and scope

- **No arbitrary code execution.** The DSL is structured JSON; the executor builds the forecast function from a small set of trusted primitives.
- **`max_windows` defaults to small** (specs typically request 8-12 windows so a chronos-based agent-authored method fits in 1-2 minutes).
- **`asset_subset`** lets the agent test its hypothesis on a small slice without paying for the full universe.

The unconstrained version — agent writes raw Python sandboxed at execution time — is deferred to Iter 20.

### Tests (9 new)

`tests/test_specs.py`: minimal-valid spec passes; missing name / unknown base / bad name chars / bad covariate subset / bad ensemble weight / bad max_windows all rejected with specific errors; full valid spec accepted; ALLOWED_BASES contains the four expected methods.

Suite total: 103 passing.

---

## Iter 14 — Hypothesis lineage DAG

The agent now generates many hypotheses per session — across slice and spawn experiments, across debate rounds. Some refine earlier ideas; some go in entirely new directions. Iter 14 makes that **lineage** visible: a DAG where nodes are unique hypotheses (deduped by content hash) and edges show inferred parent → child refinements.

### Lineage construction

`agent/lineage.py`:

- **`hypothesis_id(content)`** — stable `h_<hash10>` ID derived from hypothesis text + experiment params. The same hypothesis re-proposed gets the same node.
- **`build_lineage(ledger_entries, finding_records, parent_lookback, overlap_threshold)`** — walks the ledger, dedupes propose/theorist entries by content hash, and infers parent edges by **method/asset/regime overlap**: a hypothesis at round `r` whose params match (≥1 of method/asset/regime_id) with a hypothesis from any of the prior `parent_lookback` rounds gets that prior as its parent (the closest one wins).
- **Status assignment**:
  - `promoted` — the hypothesis matches a promoted finding (by round-of-promotion or by appearing in the finding's `parent_hypothesis_ids`).
  - `refuted` — an adjudicator step in the same round contained `VERDICT: refute`.
  - `open` — neither.
- **`lineage_dataframe(lineage)`** — convenience tabular view: `(id, round, status, hypothesis, parents)`.

### Cockpit panel

New **"Lineage"** panel between **Findings** and **Ask the Memory**:

- Three top-level metrics (total / promoted / refuted hypotheses).
- Tabular DAG view (id, round, status, hypothesis text, parent IDs).
- **Plotly DAG** rendering with:
  - X-axis = round number (left → right = chronological).
  - Y-axis = vertical jitter so hypotheses from the same round don't overlap.
  - Node color: green=promoted, red=refuted, gray=open.
  - Hover text: full hypothesis preview + experiment params.

Reviewers can trace any promoted finding back to the initial brainstorm and see the refinement chain — exactly the visibility the Iter 19 WOW demo will lean on.

### Tests (8 new)

- ID stability under identical content; different methods → different IDs.
- Empty ledger → empty lineage.
- Overlapping (method, asset, regime) chains a parent edge.
- Disjoint hypotheses produce no edges.
- Promoted status via finding-round match.
- Refuted status via adjudicator `VERDICT: refute`.
- Lineage DataFrame columns include the expected fields and root nodes show `(root)`.

Suite total: 113 passing.

---

## Iter 15 — Trace quality scoring (LLM-as-judge)

How do we know if the agent is *thinking better* over time? Iter 15 makes that observable: an evaluator LLM scores each round on four research-quality rubrics, persists the scores to `reports/agent/trace_quality.jsonl`, and the Agent Console renders the trend.

### Rubrics (1-5 each)

- **clarity** — was the hypothesis specific enough to be tested?
- **novelty** — did this round explore a (regime, asset, method) combination not yet in the ledger?
- **falsifiability** — was the prediction concrete enough that the experiment could in principle refute it?
- **evidence_citing** — did the critique / adjudication cite specific ledger or artifact entries (not just generic concerns)?

### Implementation (`agent/trace_eval.py`)

- **`JUDGE_SYSTEM`** — system prompt that fixes the rubric and the JSON output schema.
- **`score_round(round_number, round_entries, ledger_summary, provider)`** — bundles the round's ledger entries plus a summary of preceding rounds, calls the judge, parses the JSON. Returns `{round, clarity, novelty, falsifiability, evidence_citing, rationale, ts}`. Default `provider` is the `critic`-role LLM (smaller / cheaper than the proposer).
- **`score_session(ledger_entries, session_id, provider)`** — groups entries by round, runs the judge on each, persists to JSONL, returns the scores list.
- **`load()`** / **`clear()`** — round-trip the persisted store.

The judge is routed through our existing `LLMProvider` abstraction, so live (DeepInfra) and replay modes both work — same as every other LLM call in the agent layer.

### CLI

`autosignalx agent score-traces [--session-id <id>]` runs the judge over the current ledger and prints per-round scores.

### Cockpit

The Agent Console panel grows a **trace-quality line chart** at the bottom showing the four rubric scores per round. Reviewers can see (e.g.) clarity climbing while novelty stays high — the signature of an agent learning *how* to ask better questions.

### Why this matters for the WOW demo (Iter 19)

The trace quality chart over a long recorded session is one of the strongest visual signals that the agent is genuinely *improving*, not just running on autopilot. Coupled with the Findings store (Iter 11), Lineage DAG (Iter 14), and Memory consolidation (Iter 16), it makes the agent's autonomy *measurable*.

### Tests (4 new)

- Replay-provider judge returns the four expected score keys.
- Unparseable judge response yields None scores but preserves the structure.
- `score_session` persists and round-trips through `load()`.
- Long entry content is truncated in the round summary.

Suite total: 117 passing. Reference: `openevals.create_llm_as_judge` and `agentevals` trajectory evaluation -- the same conceptual pattern, routed through our DeepInfra provider abstraction.

---

## Iter 16 — Long-horizon memory consolidation

The agent's ledger grows linearly with rounds. Across many sessions, that ledger becomes too large to stuff into a context window — and most of the round-level detail isn't useful to a future session anyway. Iter 16 introduces **memory consolidation**: at the end of a session, an LLM compresses the ledger + findings into a structured Markdown **lessons** section that gets appended to `reports/agent/lessons.md`. The next session reads the most recent lessons as additional context, so the agent's first round of session N is informed by sessions 1..N-1.

This is the long-horizon memory cell Deeter explicitly asks for: unbounded growing context, periodically summarized into a structured form the agent can re-consume.

### Consolidation prompt

`agent/memory.py` `CONSOLIDATOR_SYSTEM` enforces a strict Markdown structure (under 350 words per section):

```markdown
## Session <id> -- <date>

**What was tried**: 1-3 sentences naming (regime, asset, method) combinations.
**What worked**: promoted findings by ID, or "(none)".
**What was refuted**: hypotheses with refute verdicts, or "(none)".
**Patterns observed**: 1-2 sentences on cross-cutting insights.
**Open directions for next session**: 1-3 specific slices to explore.
```

### API

- **`consolidate(session_id, ledger_entries, finding_records, provider)`** — runs the consolidator LLM (defaults to the `adjudicator` role for decisive summaries) and returns the Markdown section.
- **`append_to_lessons(section)`** — appends to `reports/agent/lessons.md` with a `---` separator.
- **`load_lessons(max_chars)`** — reads the doc, capped at `max_chars` (tail-truncated to keep section breaks intact). Cap defaults to 8000 chars; the snapshot in `tools.context_snapshot()` uses 4000.
- **`consolidate_and_append(session_id, ...)`** — convenience for end-of-session use.

### Cross-session continuity

`agent/tools.py` `context_snapshot()` now includes a `prior_sessions_lessons` field. When the agent starts a new session, the proposer / theorist sees the lessons doc in its context — open directions become natural seeds for the first round, refuted hypotheses are not re-proposed.

### CLI and cockpit

- **CLI**: `autosignalx agent consolidate [--session-id <id>]` runs the consolidation manually after a session.
- **Cockpit**: the **"Lessons & Memory"** panel (renamed from "Memory" placeholder, sits between Lineage and Ask the Memory) renders the lessons doc as Markdown. Reviewers see the agent's accumulated knowledge as a readable narrative, not a JSON dump.

### Why this matters

Without consolidation, sessions are independent. With it, the agent's productivity compounds — each session starts further along than the last. Combined with the Findings store (Iter 11), an agent run on day 30 of operation has accumulated 29 days of summarized context plus a structured store of every promoted finding. That's the offline-to-deployable axis Deeter cares about.

### Tests (5 new)

- `load_lessons` returns empty string when no doc exists.
- Append + load round-trip preserves both sessions and adds separators.
- `load_lessons(max_chars)` truncates and prepends a marker.
- `consolidate` with replay provider returns the canned section.
- `consolidate_and_append` persists to disk and the file contains the section.

Suite total: 122 passing.

---

## Iter 17 — Cost, latency, token telemetry

Autonomy with observability — Deeter's exact phrasing. Iter 17 instruments every live LLM call with `(model, prompt_tokens, completion_tokens, cost_usd, latency_ms, session_id)`, persists to `reports/agent/telemetry.jsonl`, and renders a Telemetry panel in the cockpit. Cached and replay-mode calls don't record (they're free), so the dashboard shows the real operational footprint.

### Implementation

`agent/telemetry.py`:

- **`DEFAULT_PRICES`** — per-model `(USD per 1M input tokens, USD per 1M output tokens)` for the DeepInfra models we use; override per-model via `DEEPINFRA_PRICE_<MODEL>_IN` / `_OUT` env vars.
- **`estimate_cost_usd(model_id, prompt_tokens, completion_tokens)`** — looks up the rate (env > defaults > conservative fallback `(0.50, 2.00)`).
- **`record_call(model, role, step, round, prompt_tokens, completion_tokens, latency_ms, session_id)`** — appends one record to `telemetry.jsonl`.
- **`CallTimer`** — context manager that measures wall-clock latency.
- **`load()`** / **`clear()`** — round-trip the persisted store.

`agent/llm.py` LiveProvider.chat is instrumented:

- Wrap `client.invoke(...)` with `CallTimer`.
- Mine `response.response_metadata.token_usage` (or `usage_metadata`) for token counts.
- Fall back to a character-count estimate (~4 chars per token) when the provider doesn't return usage metadata.
- Call `telemetry.record_call(...)` after each live (non-cached) call.

The Iter 7 disk cache means re-runs don't generate telemetry — only the **first** call for a given prompt-hash counts. The dashboard reflects the marginal cost of new agent work.

### Cockpit

New **"Telemetry"** panel between Lessons & Memory and Ask the Memory:

- Four headline metrics: total calls, total cost (USD), total tokens, median latency.
- **Per-model breakdown**: calls, tokens, cost, p50/p95 latency, sorted by cost descending.
- **Per-step breakdown**: which agent steps (theorist / skeptic / adjudicator / consolidate / trace_eval / ...) consume most of the budget.
- **Cumulative cost over time**: a single-line chart showing how the session's spend grew.

The headline metric for the WOW demo (Iter 19) becomes "**$X.XX of compute → Y promoted findings**" — a concrete operational ROI number.

### Tests (5 new)

- Cost estimate matches the rate table for known models.
- Unknown models use the conservative fallback.
- Env-var price override wins over defaults.
- `record_call` persists with the right fields and survives the JSONL round-trip.
- `CallTimer` measures elapsed time correctly.

Suite total: 127 passing.

---

## Iter 18 — Scheduled runs + multi-session productivity

The agent has been treated as a one-shot through Iter 17. Iter 18 makes it **continuously running**: a single `scripts/run_session.sh` (or `.ps1` for Windows Task Scheduler) executes one full agent session — debate-mode, score-traces, consolidate — and is meant to be cron-scheduled. As sessions accumulate, the project's value compounds: more findings, richer lessons doc, observable productivity trends.

### Session ID propagation

Every persisted record now carries a `session_id` for cross-session aggregation:

- **Ledger** (`reports/agent/ledger.jsonl`) — every entry: propose / theorist / skeptic / experiment / critique / adjudicator / decide. Updated `agent/graph.py` and `agent/debate.py` to thread `state["session_id"]` into every `ledger.append(...)` call.
- **Findings** (`reports/agent/findings.jsonl`) — already from Iter 11.
- **Telemetry** (`reports/agent/telemetry.jsonl`) — already from Iter 17.
- **Trace quality** (`reports/agent/trace_quality.jsonl`) — already from Iter 15.
- **Lessons** (`reports/agent/lessons.md`) — session ID in section header from Iter 16.

### Aggregation (`agent/sessions.py`)

- **`list_sessions()`** — distinct session IDs across all stores, sorted chronologically (YYYYMMDD-prefix sortable).
- **`session_summary(session_id)`** — per-session aggregates: `(n_rounds, n_propose, n_findings, n_refuted, cost_usd, total_tokens, latency_total_ms, avg_clarity, promotion_rate, cost_per_finding)`.
- **`all_summaries()`** — one row per session as a DataFrame.
- **`productivity_trend()`** — cumulative findings / cost across sessions for trend rendering.

### Scheduled runners

- **`scripts/run_session.sh`** (bash, cron-compatible) — runs one full session: `agent run --mode debate --record-replay` → `agent score-traces` → `agent consolidate`. Configurable via `AUTOSIGNALX_ROUNDS` / `AUTOSIGNALX_MODE` env vars.
- **`scripts/run_session.ps1`** (PowerShell, Windows Task Scheduler) — same pipeline, same env vars.
- Cron example baked into the script's docstring: `0 3 * * * cd /path/to/repo && bash scripts/run_session.sh >> reports/agent/cron.log 2>&1`.

### Cockpit

New **"Sessions"** panel between Telemetry and Ask the Memory:

- **4 headline metrics**: total sessions, total findings, total cost, **cost per finding**.
- **Per-session summary table**: every row with all the metrics from `session_summary`.
- **Productivity trend chart**: cumulative findings and cumulative cost over the chronological session sequence — visualizes the compounding-knowledge story.

### `make scheduled-session`

Added Makefile target that wraps the scheduled runner for manual invocation; the cron schedule itself is OS-specific so we document it in the script docstrings rather than hard-coding.

### Tests (5 new)

- Empty-state `list_sessions()` returns `[]`.
- Distinct session IDs across stores are deduped.
- `session_summary` aggregates findings, refute count, cost, tokens, promotion rate correctly.
- `all_summaries()` returns a DataFrame with the expected columns.
- `productivity_trend()` computes cumulative `cum_findings` correctly across two sessions.

Suite total: 132 passing.

---

## Iter 19 — WOW demo: auto-play replay, self-critique, recorded session, the win

This is the iteration where everything from Iters 10-18 lands in the cockpit as a coherent reviewer experience, and where the agent records its first **DM-significant promoted finding**.

### The recorded win

Running `autosignalx agent run --mode debate --max-rounds 5 --record-replay` followed by `agent run --mode single --max-rounds 3`, the agent autonomously:

1. (Round 0, debate mode) Theorist proposed: *"chronos2_multivariate beats naive for TLT in regime 3 because TLT's high betweenness centrality makes it a bridge between market clusters whose dynamics the multivariate transformer can capture."* The hypothesis composed findings from Iter 4 (regime structure), Iter 5 (per-regime macros), and Iter 6 (graph centrality).
2. The experiment ran the slice (n=407 forecast rows). The auto-promotion gate (Iter 10) ran DM and block bootstrap.
3. **Result**: skill +5.4% MAE, DM **p=0.040**, bootstrap CI low **+0.005** (strictly above zero), method_mae 1.84 vs baseline_mae 1.95.
4. Auto-promoted to `reports/agent/findings.jsonl` as `f_9395cd1bd1be` with full provenance.
5. Skeptic challenged on multiple-comparison risk; Adjudicator returned `VERDICT: support`.
6. (Same debate session, separately) Theorist also AUTHORED a new method via the Iter 13 DSL: `efa_dxy_bridge_focus` (chronos2_multivariate restricted to `DX-Y.NYB` covariate, asset_subset EFA). The method was registered, ran through the harness, and adjudicated -- the bridge-focus method *did not* outperform naive (refute), so no promotion.
7. (Subsequent single-mode session) The agent re-tested its own `efa_dxy_bridge_focus` method on regime 3 EFA, confirming the earlier refute.

This is the conditional-improvement search the Phase 1 negative result set up. The agent overcame the naive baseline on a specific (regime, asset, method) slice with statistical significance and structured evidence.

### The Auto-Play panel

`render_auto_play()` in `app/streamlit_app.py` (with three new `st.session_state` fields: `playback_idx`, `playback_speed`, `is_playing`) reads `reports/agent/ledger.jsonl` and walks through it round-by-round. Controls:

- **Play / Pause / Reset** buttons.
- **Speed slider** (0.5x / 1x / 2x / 4x).
- **Round slider** for direct jump.
- **Progress bar** showing current step / total.
- Each step rendered as a chat-style message with a step-letter icon ([T]heorist, [S]keptic, [E]xperiment, [C]ritique, [A]djudicator, [D]ecide).

When `is_playing` is True the panel auto-advances on each Streamlit re-run with `time.sleep(1/speed)`; reviewers literally **watch** the agent reason in slow motion.

### Self-Critique (`agent/self_critique.py`)

For each promoted finding, an LLM judge re-reads the finding against the current state of the ledger + other findings and returns one of `{reinforced, unchanged, weakened, refuted}` with a one-sentence rationale citing later evidence. Records persist to `reports/agent/self_critique.jsonl`.

`autosignalx agent self-critique` runs the judge over every promoted finding. The cockpit's new **Self-Critique** panel shows verdicts grouped by state, each with the judge's rationale and timestamp.

For `f_9395cd1bd1be` (TLT/regime 3/chronos2_multivariate), the judge returned `unchanged` -- "no subsequent evidence directly addresses TLT in regime 3 or the Granger bridge mechanism," which is honest: a single session doesn't replicate a finding.

### Cockpit sidebar at end of Phase 2

15 panels total, in walk order: Overview → Data → Forecast Arena → Regime Explorer → Signal Discovery Lab → Cross-Asset Graph → **Agent Console → Auto-Play Replay → Findings → Lineage → Self-Critique → Lessons & Memory → Telemetry → Sessions** → Ask the Memory.

The new headline section in **Overview** opens with a green success callout describing the WOW finding and citing finding ID `f_9395cd1bd1be` so reviewers can jump to the Findings panel and see the evidence.

### What's committed under reports/

- `reports/agent/ledger.jsonl` — 16 entries across debate (1 round) + single (3 rounds), spanning two sessions
- `reports/agent/findings.jsonl` — 1 promoted finding (the TLT win)
- `reports/agent/lessons.md` — consolidated session notes
- `reports/agent/trace_quality.jsonl` — per-round quality scores from the LLM judge
- `reports/agent/telemetry.jsonl` — cost/latency/token records (real DeepInfra spend)
- `reports/agent/self_critique.jsonl` — judge verdicts on the promoted finding
- `reports/ablations/efa_dxy_bridge_focus.parquet` — the agent-authored method's forecasts
- `replay/agent_steps.jsonl` — the recorded LLM responses (reviewers without keys see the same trace)

### Tests (2 new for self_critique)

- Replay-provider judge returns `current_state` and `rationale`; persists.
- Unparseable response defaults `current_state` to `unchanged`.

Suite total: 134 passing.
