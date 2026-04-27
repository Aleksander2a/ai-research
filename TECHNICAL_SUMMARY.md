# Technical summary — AutoSignal-X

**One-page reviewer brief.** Read this before opening anything else.

---

## What it is

An **autonomous research apparatus** that discovers conditional predictive
structure in liquid daily ETF prices and grades every claim through a
methodology stack designed to fail honestly. The contribution is the
apparatus, not any single trade. 5 model layers + an agentic research loop
+ a 9-stage statistical hardening gate, all reading and writing typed
parquet/JSONL artifacts under `reports/`.

## What's novel

1. **Methodology stack as the deliverable.** Every promoted finding is graded
   through DM + bootstrap → BH-FDR → adversarial replication (full-test +
   placebo + block-holdout) → CPCV → Probability of Backtest Overfitting →
   Deflated Sharpe → Romano-Wolf joint stepdown → hierarchical Bayesian
   posterior with Bayes factor BF₁₀ + posterior-predictive check → strict
   bar `survives_all_strict`. **Per-gate recall + FDR are measured on a
   synthetic-known-answer benchmark** with planted causal structure
   (`autosignalx eval synthetic`) so the apparatus' own discriminative power
   is itself an audited number.
2. **Specialist agent lab + persistent KG memory + pre-registration.** The
   `lab` mode chains Theorist → Verifier → PrincipalInvestigator →
   Specialist consult (Statistician / Quant / RiskOfficer / Economist /
   Implementer / RedTeam / Historian) → Skeptic → experiment →
   Adjudicator → KG-writer. Every hypothesis is hash-committed before its
   experiment runs (`reports/agent/preregistrations.jsonl`). This directly
   answers Deeter Q1 (long-horizon memory) + Q4 (autonomy with
   observability).
3. **Calibration & coherence as agent metrics.** The Theorist's predicted
   confidence is scored against finding-survival outcomes (Brier + ECE);
   each session's coherence is scored on lessons-uptake, lineage branching
   factor, and theme-persistence entropy. Phase 15.
4. **Capability-preserving ablation (Deeter Q2).** Drop each layer in
   sequence; report marginal MAE-skill vs cost-proxy bytes. On the bundled
   artifacts, naive is the unconditional MAE floor and Chronos-2 *worsens*
   raw MAE — the regime layer is what makes the surviving finding
   possible. See `reports/agent/capability_ablation.json`.

## What's honest

* The single promoted finding (`f_9395cd1bd1be`: TLT, regime 3,
  chronos2_multivariate, p=0.040, skill +5.4%) **fails block-holdout**.
  The hardening surfaced exactly the fragility the gate was built to
  catch. This is a feature, not a bug — the apparatus correctly graded
  its own discovery.
* No backtested signal-driven strategy beats passive SPY on
  risk-adjusted returns over 2021–2025 (`Backtest Arena` panel).
  Reported with full paired-bootstrap CI; honest negative result.
* On the synthetic benchmark (planted skill 0.18, 12 distractors,
  6 trials), the apparatus achieves **67% recall at FDR 0% under DM-only
  and 22% recall at FDR 0% under the strict bar** — the gates are
  conservative, exactly as designed.

## What's intentionally not yet there

* Live deployment / shadow trading — explicit scope; the cron pipeline
  ships a snapshot but does not paper-trade.
* Foundation-model fine-tuning — Chronos-2 is frozen; LoRA / probing /
  distillation experiments are next.
* Multimodal data (options, sentiment, intraday) — single modality only.
* Kelly / vol-targeting / liquidity-aware backtest.

## How to evaluate in 5 minutes

1. **Open the live cockpit:** [ai-research-aleksander2a.streamlit.app](https://ai-research-aleksander2a.streamlit.app).
2. **Click these four panels in order:**
   - **Survival Analysis** — see the strict bar, the per-gate pass/fail
     grid, and the honest "0 of N findings survive every attack" headline.
   - **Synthetic Benchmark** — see per-gate recall + FDR on a controlled
     planted-structure test; confirms the apparatus actually finds signal
     when there is one.
   - **Capability Ablation** — see the per-layer marginal-MAE / cost-proxy
     Pareto; directly answers "where should compression happen?"
   - **Specialist Council** — see the lab-mode multi-role consultation
     feed + persistent knowledge graph (long-horizon memory).
3. **Read these two paragraphs of REPORT.md:**
   - "Executive summary" (state of the art at this commit).
   - "What this validates about the system" (in the Backtest section)
     — the meta-claim the apparatus makes about itself.
4. (Optional) **Reproducibility panel** carries the full bundle hash of
   the cockpit state you're seeing.

## Files worth reading

| Path | Why |
|---|---|
| `src/autosignalx/eval/significance.py` | The original promotion gate (DM + bootstrap). |
| `src/autosignalx/eval/survival.py` | The full hardening pipeline with all 9 gates. |
| `src/autosignalx/eval/synthetic_benchmark.py` | The known-answer benchmark generator. |
| `src/autosignalx/eval/capability_ablation.py` | Layer-drop Pareto. |
| `src/autosignalx/agent/lab.py` | The specialist research-lab orchestration. |
| `src/autosignalx/agent/knowledge_graph.py` | Persistent KG (Deeter Q1). |
| `src/autosignalx/agent/calibration.py` | Brier + ECE for the Theorist's predicted confidence. |
| `app/streamlit_app.py` | 31 cockpit panels, every one a read-only viewer over `reports/`. |

## Reproducibility

* `make demo` (or `uv run streamlit run app/streamlit_app.py`) spins up the
  full cockpit from bundled artifacts; no API key required.
* `AUTOSIGNALX_REPLAY=true uv run pytest -q` runs the full 337-test suite.
* Every panel carries a citation back to the artifact it reads; the
  Reproducibility panel exposes the SHA-256 bundle hash of the current
  state.

---

*Submission for Deeter Analytics AI Researcher role · 72-hour technical
submission · MIT licensed · github.com/Aleksander2a/ai-research*
