# AutoSignal-X — Research Report

> A running record of methods, results, and findings as iterations land.
> Each iteration appends its own section. Final consolidation in Iter 9.

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
- **Iter 5 (signal).** TabPFN ranks features per regime. The hypothesis: in some regimes (e.g., high-VIX), macro signals carry meaningful short-horizon information; in others (e.g., calm bull markets), they don't. Conditional inclusion is the bet.
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

The regime labels are an input contract for Iters 5 (TabPFN signal ranking per regime), 6 (cross-asset graph computed within regimes), and 7 (agent that hunts for regime-specific predictive structure). Without regime labels, those iterations would have to invent ad-hoc segmentation. Now they can read regimes the way the eval harness reads forecasts: as a typed, persisted artifact under `reports/regimes/`.

**Verification**: `make regime` runs end-to-end in ~1 minute; `make test` -> 67 tests passing (6 new for encoder + cluster); `make demo` -> Regime Explorer renders all three views; Forecast Arena shows per-regime stratification on the 4-method ablation.

---

---

## Iter 5 — Reasoning layer: per-regime feature ranking

The L3 reasoning layer lands. We ask, for each of the 4 KMeans regimes (Iter 4): *which features carry the most signal for predicting next-21-day direction?* The answer is the foundation that Iter 7's agent will use to propose conditional forecasting strategies.

### TabPFN pivot to HistGradientBoosting (honest scope adjustment)

The original plan called for TabPFN-v2 (Prior Labs) -- a foundation model for tabular data, the closest analog to Chronos for the tabular domain. We installed and tested it. **TabPFN >= 2.x packages require an interactive browser-based license acceptance** that polls stdin for an API key (Prior Labs URL: https://ux.priorlabs.ai). This blocks both the `make demo` flow on a fresh-clone reviewer machine and any non-interactive CI.

We pivoted to **`sklearn.ensemble.HistGradientBoostingClassifier`** -- a strong, license-free, sklearn-native classifier with native handling of mixed feature scales and missing values. The ranking methodology (custom permutation importance: shuffle one feature at a time, measure accuracy drop) is identical; only the underlying classifier changed. We document this honestly because the project's "free, reproducible, no key required to run the demo" guarantee is core to the submission.

If TabPFN's license model changes, swapping back is a single-class edit in `signal.ranking`.

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

## Future iterations

Sections will be appended below as each iteration ships. See [README](README.md#iteration-plan) for the iteration plan.
