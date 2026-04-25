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

## Future iterations

Sections will be appended below as each iteration ships. See [README](README.md#iteration-plan) for the iteration plan.
