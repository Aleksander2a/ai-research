# Technical summary — AutoSignal-X

A one-page overview. Read this before opening anything else; it explains
*what* the system is, *why* it's built the way it is, *how* the pieces
fit together, and *how to evaluate it in five minutes*.

---

## What it is

AutoSignal-X is an **AI research system for discovering predictive structure
in liquid daily ETF prices**, paired with an autonomous research loop that
grades its own discoveries through a multi-stage statistical methodology
before any finding is published. Five model layers + an agentic loop
write typed parquet/JSONL artifacts to `reports/`; the cockpit and the
agent are both read-only consumers of those artifacts. The contribution
is the methodology and the agent that operates it; any single trade is
incidental.

## Why this design

Most "AI for trading" pipelines fail in one of two ways: they either
over-promote (every promising number ships as a finding) or they
generate so many candidate signals that selection bias overwhelms any
real lift. AutoSignal-X is built around the inverse failure mode: a
finding is published only after passing every gate the published
quant-research literature uses to defend its claims (Diebold-Mariano,
block-bootstrap, BH-FDR, Combinatorial Purged CV, Probability of
Backtest Overfitting, Deflated Sharpe, Romano-Wolf step-down,
hierarchical Bayesian shrinkage with a Bayes-factor floor). The
apparatus is engineered to **fail honestly when there is no signal,
and to recover known-true signal when one is planted.**

## How it's structured

| Layer | Purpose | Implementation |
|---|---|---|
| **L1 — Forecasting** | Probabilistic point + interval forecasts | Frozen Chronos-2 (multivariate, with macro covariates) + classical baselines (naive, seasonal-naive, ARIMA(1,1,1) on log-prices). Phase 7 adds returns / excess-return / vol / cross-sectional-rank target types. |
| **L2 — Representation** | Latent regime labels | Contrastive 1D-CNN encoder (16-dim, 60-day windows, triplet loss) + KMeans on embeddings; Gaussian HMM on raw features as a parallel detector. |
| **L3 — Reasoning** | Per-regime feature importance | HistGradientBoostingClassifier per regime + custom permutation importance + walk-forward stability layer. |
| **L4 — Relational** | Cross-asset structure | GLASSO partial correlations + Granger causality + NetworkX centrality; per-regime variant rebuilds the graph inside each regime. |
| **L5 — Agentic** | Hypothesis generation, experimentation, statistical promotion | LangGraph state machine in three modes: `single` (one LLM does propose / critique / decide), `debate` (Theorist / Skeptic / Adjudicator with three different DeepInfra models), `lab` (full specialist team — PrincipalInvestigator + Verifier + Statistician / Quant / RiskOfficer / Economist / Implementer / RedTeam / Historian + KG writer). |

## The methodology stack (every gate)

Every promoted finding passes the same nine gates, in order:

1. **Diebold-Mariano** with Newey-West HAC variance, on per-row absolute losses.
2. **Block-bootstrap CI** on the loss-difference must be strictly above zero.
3. **BH-FDR** at α=0.10 across the family of all promoted findings.
4. **Adversarial replication**: full-test (drop the agent's window cap) + placebo regime-shuffle + 50/50 block-holdout.
5. **CPCV** — combinatorial purged cross-validation with embargo (Lopez de Prado).
6. **Probability of Backtest Overfitting** — Bailey/Borwein/Lopez de Prado/Zhu (2014).
7. **Deflated Sharpe Ratio** — adjusts for the number of strategies tried.
8. **Romano-Wolf** — studentized stepdown FWER under arbitrary dependence.
9. **Hierarchical Bayesian** — Normal-Normal model with empirical-Bayes; Bayes factor BF₁₀ ≥ 10 and posterior P(θ > 0) ≥ 0.95.

The strict bar `survives_all_strict` is the conjunction of every gate.
Per-gate recall + FDR are themselves audited on a controlled benchmark
(`autosignalx eval synthetic`) where causal structure is deliberately
planted so the apparatus' own discriminative power is a measured number,
not a marketing claim.

## What's novel

1. **Methodology stack as the deliverable, not a single trade.** Every
   panel of the cockpit is wired to one or more of the gates above; the
   strict bar is the conjunction.
2. **Specialist agent lab + persistent knowledge-graph memory + pre-registration.**
   The `lab` mode chains Theorist → Verifier → PrincipalInvestigator →
   Specialist consult → Skeptic → experiment → Adjudicator → KG-writer.
   Every hypothesis is hash-committed to
   `reports/agent/preregistrations.jsonl` *before* its experiment runs,
   making selection bias auditable end-to-end.
3. **Calibration & coherence as agent metrics.** The Theorist's predicted
   confidence is scored against finding-survival outcomes (Brier + ECE);
   each session's coherence is scored on lessons-uptake, lineage
   branching factor, and theme-persistence entropy.
4. **Capability-preserving ablation.** Drop each layer in turn; report
   marginal MAE-skill vs cost-proxy bytes. On the bundled artifacts,
   naive is the unconditional MAE floor and Chronos-2 actually *worsens*
   raw MAE — the regime layer is what makes the surviving finding
   possible. A concrete Pareto frontier for compression decisions,
   committed at `reports/agent/capability_ablation.json`.
5. **Synthetic-known-answer benchmark.** Plant causal structure into a
   synthetic universe; measure per-gate recall + FDR. On the bundled
   configuration (planted skill 0.30, 12 distractors, 8 trials):
   DM-only recall 100 % / FDR 0 %, strict-bar recall 94 % / FDR 0 % —
   the apparatus recovers planted structure when it exists.

## What's honest

- The apparatus has produced **9 promoted findings** on the bundled
  universe across one initial single-LLM session, an exhaustive
  (method × asset × regime) sweep, and three lab-mode agent sessions
  (Qwen3-Max theorist + GLM-4.7-Flash specialists + DeepSeek-V3
  adjudicator). Every one is a TLT / regime-3 / DXY variation; the
  agent converged hard on this anchor cell. **0 of 9 survive the
  strict bar; 0 of 9 even survive the simpler block-holdout test.**
  The lift on each is concentrated in 2021-Q1 → 2022-Q3 and does not
  corroborate after that. The hardening surfaced exactly the
  fragility the gates were built to catch — this is the design goal,
  not a failure mode.
- No backtested signal-driven strategy beats passive SPY on
  risk-adjusted returns over 2021-2025 (`Backtest Arena` panel),
  reported with full paired-bootstrap CI. The `FindingDriven` and
  `RegimeGated` strategies hold cash from 2022-10 onwards because no
  promoted finding's regime is active in regime 2 (the dominant
  2023-2025 regime). The cockpit panel surfaces this with a
  per-strategy activity summary so the flat segment reads as
  intentional cash-holding, not missing data.
- Cumulative LLM spend across all live sessions: **\$0.247** out of
  the \$20 budget. Replay mode reproduces every cockpit panel
  deterministically without an API key, so the published numbers are
  auditable.
- The agent's predicted-confidence calibration is currently poor
  (Brier 0.49, ECE 0.70 on the findings where a `predicted_effect`
  block was emitted) — a clear target for the next agent-scaffold
  iteration.

## What's intentionally not yet there

* Live deployment / paper-trading shadow mode — the cron pipeline
  ships a snapshot, but does not execute trades.
* Foundation-model fine-tuning — Chronos-2 is frozen; LoRA / probing /
  distillation experiments are obvious next steps.
* Multimodal data (options surfaces, sentiment, intraday) — single
  modality (daily prices + 4 macro signals) only.
* Kelly / volatility-targeting / liquidity-aware backtest extensions.

## How to evaluate in 5 minutes

1. **Open the live cockpit:** [ai-research-aleksander2a.streamlit.app](https://ai-research-aleksander2a.streamlit.app). It opens on the **Headline** panel.
2. **Read the four headline metrics** (top of the Headline panel): how
   many findings were promoted, how many survive the strict bar, what
   the synthetic benchmark's recall + FDR are at the strict gate.
3. **Click these four panels in order** (sidebar groups: *Methodology*
   then *Agent activity*):
   - **Survival Analysis** — per-finding pass/fail across every gate.
   - **Synthetic Benchmark** — same gates on a controlled universe.
   - **Capability Ablation** — layer-drop MAE-vs-cost frontier.
   - **Specialist Council** — `lab`-mode multi-role consultation feed +
     persistent knowledge-graph explorer.
4. **Read these two sections of REPORT.md:**
   - "Executive summary" (state of the apparatus at this commit).
   - "What this validates about the system" (in the Backtest section).
5. *(Optional)* Open the **Reproducibility** panel — the bundle hash
   uniquely identifies the cockpit state; two badges with the same hash
   render identical panels.

## Files worth reading first

| Path | Why |
|---|---|
| `src/autosignalx/eval/significance.py` | The original promotion gate (DM + bootstrap). |
| `src/autosignalx/eval/survival.py` | The full hardening pipeline with all nine gates. |
| `src/autosignalx/eval/synthetic_benchmark.py` | The known-answer benchmark generator. |
| `src/autosignalx/eval/capability_ablation.py` | Layer-drop Pareto for the compression frontier. |
| `src/autosignalx/agent/lab.py` | The specialist research-lab orchestration. |
| `src/autosignalx/agent/knowledge_graph.py` | Persistent KG memory across sessions. |
| `src/autosignalx/agent/calibration.py` | Brier + ECE for the Theorist's predicted confidence. |
| `app/streamlit_app.py` | 34 cockpit panels in 7 sidebar sections; every one is a read-only viewer over `reports/`. |

## Reproducibility

* `make demo` (or `uv run streamlit run app/streamlit_app.py`) spins up
  the full cockpit from bundled artifacts; no API key required.
* `AUTOSIGNALX_REPLAY=true uv run pytest -q` runs the full 342-test
  suite in deterministic replay mode.
* Every cockpit panel carries a citation back to the artifact it reads;
  the **Reproducibility** panel exposes the SHA-256 bundle hash of the
  current state.

---

*MIT-licensed · github.com/Aleksander2a/ai-research*
