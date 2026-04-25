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

## Future iterations

Sections will be appended below as each iteration ships. See [README](README.md#iteration-plan) for the iteration plan.
