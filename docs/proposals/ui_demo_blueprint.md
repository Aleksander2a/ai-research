# AutoSignal-X UI & Demo Blueprint

## Purpose of the UI Layer

The UI for AutoSignal-X is not intended to be a consumer-facing product. Its purpose is to function as a **research cockpit**: a transparent, inspectable, and visually compelling interface that exposes the scientific workflow, system reasoning, and empirical results.

The objective is to help reviewers understand in under a minute:
- what the system is doing,
- what it has discovered,
- how it evaluates itself,
- and why its outputs are credible.

This document defines the recommended UI structure, demo flow, tooling, and implementation priorities for a high-impact yet time-efficient presentation layer.

---

# Strategic Design Principles

## 1. Insight Density Over Visual Complexity
Favor dashboards that surface meaningful findings over decorative polish.

## 2. Research Transparency
Expose experiments, metrics, and model reasoning.

## 3. Fast Iteration
Use frameworks that maximize output within limited time.

## 4. Narrative Flow
The UI should guide the reviewer through discovery → evaluation → results.

---

# Recommended Tech Stack

## Primary UI Framework
**Streamlit**

Why:
- Python-native
- minimal frontend overhead
- excellent support for charts, tables, and ML workflows
- deployable locally or to Streamlit Cloud

---

## Visualization Libraries
- Plotly (interactive charts)
- Altair (clean analytical visuals)
- AgGrid (sortable/filterable tables)

---

## Experiment Tracking
**Weights & Biases** or **MLflow**

Purpose:
- metric logging
- run comparison
- artifact storage
- reproducibility

---

## Agent Observability
Optional but valuable:
- LangSmith traces
- custom markdown logs

---

# UI Modules to Build

## 1. Research Overview Dashboard

### Purpose
Present project objective and current best performance.

### Show
- project thesis
- current best model configuration
- key metrics (Sharpe, MAE, calibration score)
- number of experiments completed
- latest discovered signals

### Value
Immediate orientation and credibility.

---

## 2. Signal Discovery Lab

### Purpose
Display exogenous features and ranked importance.

### Show
- feature leaderboard
- source attribution
- impact on forecast quality
- accepted vs rejected signals

### Visuals
- horizontal bar charts
- sortable tables

### Value
Demonstrates actual research output.

---

## 3. Forecast Arena

### Purpose
Compare baseline vs enhanced forecasting pipelines.

### Show
- time-series prediction overlays
- uncertainty intervals
- residual analysis
- benchmark comparison

### Visuals
- interactive line charts

### Value
Empirical validation.

---

## 4. Regime Explorer

### Purpose
Surface latent market states.

### Show
- clustered embeddings
- regime labels
- transition frequencies
- performance by regime

### Visuals
- PCA / UMAP scatterplots
- heatmaps

### Value
Research sophistication.

---

## 5. Agent Debate Console

### Purpose
Expose hypothesis generation and experiment planning.

### Show
- proposed signals
- critiques
- experiment rationale
- next-step recommendations

### Format
chat-style cards / timeline view

### Value
Makes the system feel autonomous and inspectable.

---

# Demo Strategy

## Reviewer Journey (Ideal Sequence)

1. Open Research Overview
2. Inspect discovered signals
3. Compare forecasts
4. Explore latent regimes
5. Review agent reasoning

This creates a coherent story in under 5 minutes.

---

# Implementation Priorities

## Must Have
- Overview Dashboard
- Signal Discovery Lab
- Forecast Arena

## Nice to Have
- Regime Explorer
- Agent Debate Console

## Optional Stretch
- live experiment trigger button
- downloadable reports

---

# Time Management Guidance

Allocate:
- 70% backend / experiments
- 20% dashboard
- 10% polish

Avoid custom frontend work.

---

# Deliverable Packaging

Include in repository:
- /app for Streamlit UI
- /reports for experiment artifacts
- screenshots in README
- one-command startup instructions

Example:
```bash
make demo
```

---

# Why This Matters

A visible, inspectable UI increases perceived completeness and professionalism.

It shows the ability to translate research into usable systems.

That combination—technical depth + interface clarity—is highly differentiated for an AI Researcher candidate.

---

# Final Principle

Build a research cockpit, not a product.

The UI exists to make discovery legible.
That is what will impress reviewers.

