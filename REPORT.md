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

## Future iterations

Sections will be appended below as each iteration ships. See [README](README.md#iteration-plan) for the iteration plan.
