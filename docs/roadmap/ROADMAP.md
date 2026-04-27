# AutoSignal-X — phase roadmap and status

**Audience.** Anyone returning to this repository to understand which
phases have shipped, which are open, and what each phase contributes.
Working reference; not a marketing document.

**Status.**

| Phase | Theme | Status |
|---|---|---|
| 1 | Backtested simulation (engine, strategies, paired-bootstrap CI) | shipped |
| 2 | User-defined custom studies (universe, dates, splits) | shipped |
| 3 | Conversational explainability (grounded RAG over the run corpus) | shipped |
| 4 | Demo and deployment (static snapshot + Streamlit Cloud) | shipped |
| 5 | Statistical hardening (BH-FDR + adversarial replication) | shipped |
| 6 | Structural enrichments (per-regime graph + walk-forward signal stability) | shipped |
| 7 | Returns-target forecast contract (price / log_return / excess_return / vol / rank) | shipped |
| 8 | Selection-bias-aware evaluation (CPCV + PBO + Deflated Sharpe + Romano-Wolf + pre-registration + holdout vault) | shipped |
| 12 | Hierarchical Bayesian evidence (Normal-Normal posterior + Bayes factor + PPC) | shipped |
| 14 | Specialist agent lab (11 roles, lab-mode LangGraph, persistent KG memory, EIG planner, verifier) | shipped |
| 15 | Self-improving prompts + agent evals (calibration, RedTeam, coherence, prompt versioning, eval suite) | shipped |
| 16 | Cockpit observability + capability evaluation (coverage map, statistical-power dashboard, counterfactual cards, reproducibility badge, synthetic benchmark, capability ablation) | shipped |

**Hard rules that hold across every phase:**

1. **Walk-forward integrity.** Every parameter a model or strategy uses
   must be frozen at or before the data window it acts on. Test
   boundaries: `train ≤ 2018-12-31 < val ≤ 2020-12-31 < test ≤
   2025-12-31`. New code that touches the eval pipeline must add or
   extend a leakage test.
2. **Reproducibility.** `make demo` from a fresh `uv sync` continues to
   work; replay mode (`AUTOSIGNALX_REPLAY=true`) continues to work
   without a DeepInfra key.
3. **Artifact contracts.** Layers communicate exclusively through typed
   parquet / JSONL on disk; no in-memory shared state. New layers add
   new artifacts; they do not retrofit existing ones.
4. **One commit ships a working MVP.** Per-concern branches, merged
   `--no-ff` into `main`.

---

## Phases 7, 8, 12, 14, 15, 16 — research-grade methodology and observability

These phases lift the project from "rigorous research instrument" to
"autonomous discovery engine that audits its own discriminative power".

### Phase 7 — Returns-target forecast contract

* `eval/contracts.py` extended with optional `target_type` column
  (`price | log_return | excess_return | vol | rank`); legacy parquets
  default to `price`.
* `eval/targets.py` adapters: `to_log_return`, `to_excess_return` (^TNX
  proxy for `rf_daily`), `to_realized_vol`, `to_cross_sectional_rank`,
  `convert_target` dispatch.
* `forecast/returns_baselines.py`: `zero_return`, `mean_return`,
  `momentum` baselines (returns-native).
* `eval/metrics_returns.py`: `forecast_sharpe`, `hit_rate_returns`,
  `ic_pearson`, `ic_spearman`, `summarise_returns`.
* CLI: `autosignalx eval returns --target log_return`.
* Backward-compatible: existing pipelines untouched.

### Phase 8 — Selection-bias-aware evaluation

* `eval/cpcv.py`: combinatorial purged cross-validation with embargo
  (Lopez de Prado).
* `eval/pbo.py`: Probability of Backtest Overfitting (Bailey, Borwein,
  Lopez de Prado, Zhu, 2014).
* `eval/deflated_sharpe.py`: DSR with closed-form expected-max-Sharpe
  null.
* `eval/romano_wolf.py`: studentized stepdown FWER under arbitrary
  dependence.
* `eval/preregistration.py`: hash-committed hypothesis ledger with
  resolutions append-only.
* `eval/holdout_vault.py`: never-touched final test slice;
  `assert_no_vault_leakage` raises; `open_vault` is one-time.
* CLI: `autosignalx eval pbo`, `eval vault-init`, `eval vault-open`.
* `survival.jsonl` augmented with `cpcv`, `rw_q`, `survives_rw`,
  `deflated_sharpe`, `survives_dsr`, `survives_all_strict`.

### Phase 12 — Hierarchical Bayesian evidence

* `eval/bayesian.py`: Normal-Normal hierarchical model with
  empirical-Bayes hyperparameters; per-finding posterior mean / sd /
  P(theta>0) / Bayes factor BF_10. No NumPyro/PyMC dependency required.
* Plumbed through `survival.jsonl` as `bayesian` field +
  `survives_bayes` flag (BF >= 10 and P(theta>0) >= 0.95).
* `posterior_predictive_check` simulates next-session loss differences.

### Phase 14 — Specialist agent lab

* `agent/specialists.py`: 11 roles
  (PrincipalInvestigator, Theorist, Skeptic, Adjudicator, Statistician,
  Quant, RiskOfficer, Economist, Implementer, RedTeam, Historian).
* `agent/lab.py`: lab-mode LangGraph
  (Theorist -> Verifier -> Planner -> Specialist -> Skeptic ->
  experiment -> Adjudicator -> KG-writer).
* `agent/verifier.py`: pre-registration check on every hypothesis.
* `agent/knowledge_graph.py`: persistent KG (nodes + edges as JSONL);
  idempotent `ingest_findings`.
* `agent/eig.py`: Bayesian-experimental-design proxy + `coverage_map`
  for the cockpit.
* CLI: `autosignalx agent run --mode lab --specialists ...`.

### Phase 15 — Agent self-improvement and evals

* `agent/calibration.py`: Brier + ECE + reliability bins for confidence
  vs survival.
* `agent/red_team.py`: asset-shuffle + time-shift attacks beyond Phase
  5; persists `red_team.jsonl`.
* `agent/coherence.py`: lessons_uptake + lineage_branching_factor +
  theme_persistence_entropy + composite coherence_score; persists
  `coherence.jsonl`.
* `agent/prompt_optimizer.py`: per-role prompt versioning + scoring
  against trace_quality rubrics.
* `agent/eval_suite.py`: orchestrator; CLI `autosignalx agent eval-suite`.

### Phase 16 — Cockpit observability and explainability

Eleven new panels:

* **Coverage Map** -- 4D heatmap of (method × asset × regime) coloured
  by EIG.
* **Statistical Power** -- per-cell Cohen's d / power / required-n via
  `eval/power.py`.
* **Counterfactual Cards** -- factor residualization + what-if +
  outlier-removal via `eval/counterfactual.py`.
* **Bayesian Evidence** -- posterior + Bayes factors view of Phase 12.
* **Specialist Council** -- multi-role consultation feed + KG explorer.
* **Pre-Registration** -- registered / open / resolved hypotheses.
* **Holdout Vault** -- lock status + one-time-eval results.
* **Agent Calibration** -- reliability diagram for the Theorist.
* **RedTeam Attacks** -- asset-shuffle + time-shift verdicts.
* **Agent Coherence** -- per-session coherence trends.
* **Reproducibility** -- git + env + per-artifact SHA-256 + bundle hash
  via `autosignalx/reproducibility.py`.

All panels read existing artifacts; none re-train. Cockpit registration
via `PANELS` dict in `app/streamlit_app.py`.

---

## Phase 1 — Backtested simulation (SHIPPED)

> Status: shipped. Full implementation in `src/autosignalx/backtest/`; cockpit
> panel **Backtest Arena** is live; results documented in REPORT.md
> ("Backtested simulation" section). The notes below preserve the original
> design rationale for future-Claude.


### Goal

Use the discovered predictive structure (forecasts, regimes, promoted
findings) to drive a **simulated trading strategy** on historical data and
measure whether the discoveries translate into ex-ante profitable behaviour.
This is the "cherry on top" — it makes the abstract research output concrete
and visible.

### Scientific question

> Do AutoSignal-X's discovered structures (Chronos-2 + regime + signal +
> agent-promoted findings) produce **out-of-sample** trading performance
> superior to passive benchmarks, on a strict walk-forward simulation that
> never sees its own training data?

### Success criteria

- **Engineering:** end-to-end run via `autosignalx backtest` produces a
  parquet of daily portfolio state, a metrics JSON, and a cockpit panel
  visualising equity curve + drawdown + per-strategy comparison.
- **Scientific:** at least one signal-driven strategy beats both
  buy-and-hold SPY and equal-weight-universe on **risk-adjusted** metrics
  (Sharpe, Calmar) net of realistic costs (5–10 bps per turnover unit).
  Negative result is acceptable and must be reported honestly.
- **Integrity:** dedicated leakage test passes; backtest window is
  strictly disjoint from any window used to fit/select strategy parameters.

### Architectural decision: custom engine vs. framework

| Option | Verdict | Reasoning |
|---|---|---|
| `backtrader` | Reject | Heavyweight, OO bar-feed abstraction does not match our parquet contracts. No real release since 2017. |
| `vectorbt` (OSS) | Optional cross-check | Apache-2.0, fast, pandas-native. Strong fit but pulls a heavy dep tree (numba). Worth a single comparison run, not the primary engine. |
| `bt` | Reject | Less maintained, would still need adapter code. |
| **Custom vectorized engine** | **Primary** | ≤300 LOC, every line auditable for look-ahead, zero new heavy dependencies, fits our artifact contracts directly. |

Custom-first is the chosen path. Optional `vectorbt` parity check is a
stretch goal at the end of Phase 1.

### Look-ahead defenses (mandatory)

1. **Temporal disjointness.** Backtest only consumes forecasts where
   `forecast.timestamp >= 2021-01-01`. The data layer's `test` split is the
   only legal source for backtest input.
2. **Parameter freeze.** All learned parameters the strategy uses (regime
   centroids, signal weights, agent-promoted finding metadata) must have
   been fit on data with `timestamp ≤ 2020-12-31`. Strategy code asserts
   this on load.
3. **Trade timing.** A signal generated using bars up to `t` (close-of-day
   `t`) trades at `open` of `t+1`. Returns are then computed from `t+1`'s
   open to `t+2`'s open (or close-to-close as a configurable convention,
   but consistently applied). No same-bar fills.
4. **Survivorship bias.** Universe is fixed at the start (the eight ETFs
   are large and durable, so this is acceptable for an MVP; flagged in
   limitations).
5. **Test:** `tests/test_no_backtest_leakage.py` asserts (1) and (3) on a
   tiny synthetic fixture and on the real produced artifacts.

### Strategies in scope (MVP set)

| Strategy | Description | Why it's interesting |
|---|---|---|
| `BuyAndHoldSPY` | Hold 100% SPY across the backtest window. | Passive benchmark every reviewer recognises. |
| `EqualWeightUniverse` | Daily-rebalanced equal weights across the 8 ETFs. | Tests whether *any* signal helps over naïve diversification. |
| `TopKLong` (k=3) | Each day, go long the K assets with highest predicted next-period return; equal-weight within. | The simplest signal-driven strategy. |
| `LongShortKK` (k=2) | Long top-K and short bottom-K by predicted return; market-neutral notional. | Tests the cross-sectional edge of the forecast. |
| `RegimeGated` | Only deploy `TopKLong` in regimes where the per-regime significance gate flagged the forecast as promotable; otherwise hold cash. | Tests whether the regime layer's value-add survives backtest. |
| `FindingDriven` | Read promoted findings (`reports/agent/findings.jsonl`); if the finding's `(method, asset, regime)` slice is active today, take a position scaled by the finding's effect size. | The strongest test that the agent's discoveries are actionable. |

### Metrics

Per strategy, computed over the test window:

- **Return:** total, annualised (CAGR).
- **Risk:** annualised volatility, max drawdown, downside deviation.
- **Risk-adjusted:** Sharpe, Sortino, Calmar.
- **Trading:** hit rate (% of trading days with positive return), turnover
  (avg |Δw|), realised cost drag.
- **Per-regime:** above metrics conditioned on regime ID.
- **Significance:** block-bootstrap CI on Sharpe vs. each benchmark
  (reuses the same machinery as Iter 10's promotion gate).

### Architecture (file-level)

```
src/autosignalx/backtest/
    __init__.py              # public API: run_backtest(...)
    engine.py                # vectorized portfolio engine; trade timing rules
    strategies.py            # strategy classes, all subclass BaseStrategy
    metrics.py               # Sharpe/Sortino/Calmar/turnover/etc.
    runner.py                # CLI entry point: orchestrate full ablation
    schemas.py               # pydantic models for backtest config + outputs

tests/
    test_backtest_engine.py
    test_backtest_strategies.py
    test_backtest_metrics.py
    test_no_backtest_leakage.py

reports/backtest/
    runs/<run_id>/
        config.yaml
        portfolio_daily.parquet  # (timestamp, strategy, weights..., return, equity)
        trades.parquet           # (timestamp, strategy, asset, dweight, cost)
        metrics.json             # nested: per-strategy, per-regime
        meta.json                # universe, dates, costs, seed, git hash

app/streamlit_app.py
    + render_backtest_arena()    # equity curves, metric table, per-regime
```

Public API contract:

```python
from autosignalx.backtest import run_backtest

result = run_backtest(
    strategies=["BuyAndHoldSPY", "EqualWeightUniverse",
                "TopKLong", "LongShortKK", "RegimeGated", "FindingDriven"],
    start="2021-01-01",
    end="2025-12-31",
    cost_bps=5.0,
    seed=42,
)
# result is a BacktestResult dataclass; artifacts written under reports/backtest/runs/<id>/
```

### Phase 1 sub-iterations (each ships a working MVP)

| # | Branch | Theme | Hours | Pre-merge gate |
|---|---|---|---|---|
| 1.1 | `phase1-backtest` | Engine skeleton + `BuyAndHoldSPY` + `EqualWeightUniverse` + leakage test + cockpit panel stub | 3 | `make test`, manual cockpit smoke |
| 1.2 | `phase1-signal-strategies` | `TopKLong` + `LongShortKK` consuming Chronos-2 forecasts | 3 | full ablation runs end-to-end |
| 1.3 | `phase1-regime-gating` | `RegimeGated` + per-regime metric breakdown | 2 | regime-conditioned metrics in cockpit |
| 1.4 | `phase1-finding-driven` | `FindingDriven` strategy reading `findings.jsonl` | 2 | strategy fires only when finding's slice is active |
| 1.5 | `phase1-bootstrap-significance` | Block-bootstrap CI on Sharpe-diff vs. benchmarks | 2 | significance table in cockpit + REPORT |
| 1.6 | `phase1-report` | REPORT.md backtest section: methodology, ablation, honest negative results, limitations | 1 | merged to integration branch |
| 1.7 (stretch) | `phase1-vectorbt-parity` | Run a single strategy through vectorbt and assert equity curves match within tolerance | 2 | parity asserted in test |

**Total estimate: ~15 hours**, split across multiple sessions if needed.

### What Phase 1 deliberately does **not** include

- Live/paper trading wiring. The output is purely a historical simulation.
- Slippage modelling beyond a flat bps cost. (Flagged in limitations.)
- Optimisation/hyperparameter search over strategy parameters — that would
  introduce backtest overfitting. K and bps are config-frozen, not tuned
  on test.
- Position sizing via Kelly / vol-targeting. Equal-weight only in MVP.

---

## Phase 2 — User-provided inputs (SHIPPED)

### Goal

Lift the hardcoded universe (8 ETFs, 2010–2025) so a user can run the full
pipeline on their own assets and date ranges.

### Scope

- **Inputs the user controls:** ticker list, macro covariates list, start
  date, end date, walk-forward split boundaries, regime count, signal
  top-K, agent rounds, cost assumptions.
- **Inputs that stay frozen:** the layer architecture, the artifact
  contracts, the leakage rules.

### Approach

Two surfaces, same underlying mechanism:

1. **CLI flags + named configs.** Every existing CLI subcommand grows
   `--config <path>` and per-flag overrides (`--assets`, `--start`,
   `--end`, …). Configs live under `configs/user_*.yaml`. Default behaviour
   unchanged when no overrides given.
2. **Cockpit "Custom Study" panel.** A form: paste tickers, pick dates,
   pick walk-forward boundaries, hit "Run." The panel shells out to the
   CLI under the hood and streams progress. Each run is keyed by a
   content-hash of the config so reruns are cached.

### Non-trivial concerns to resolve in detail when Phase 2 starts

- **yfinance failures.** Tickers without enough history; survivorship bias
  in user-picked baskets; macro covariates that don't exist on yfinance.
  The data layer needs to detect these and surface clear errors before
  downstream layers run.
- **Minimum-data thresholds.** Chronos-2 needs a minimum context window;
  regimes need enough samples per cluster; TabPFN-style models cap rows
  per regime. Each layer must declare its preconditions and the runner
  must check them before the layer fires.
- **Walk-forward defaults.** When the user picks a date range, what are
  the default split boundaries? Plan: 60/15/25 train/val/test split by
  business days, configurable per run.
- **Regime-count selection.** User-picked or BIC-selected on val? Plan:
  default to BIC selection over `k ∈ {2..6}`, with manual override.
- **Caching key.** Config-hash + git-hash + universe-hash. Stored under
  `data/cache/user_studies/<key>/`. Invalidates automatically when any
  input changes.
- **Cost scaling.** Pipeline runtime grows linearly with universe size and
  date range; the cockpit must show an estimate before kicking off.

### Phase 2 sub-iterations (sketch)

- 2.1 Parameterise the data layer; CLI flags + config loading
- 2.2 Propagate config through all layer entry points
- 2.3 Cache layer keyed by config hash
- 2.4 Cockpit "Custom Study" form
- 2.5 Pre-flight validation (data availability, sample-size thresholds)
- 2.6 REPORT update with worked example on a non-default universe

---

## Phase 3 — Conversational explainability (SHIPPED)

### Goal

Replace the keyword "Ask the Memory" panel with a chat interface that
answers natural-language questions about the system's discoveries,
**grounded in real run artifacts** with explicit citations.

### Scope

- Corpus: ledger entries, findings, lessons, sessions, telemetry,
  forecasts (aggregated per asset/regime), backtest results (Phase 1
  output).
- Output: streamed answer + a list of cited artifact IDs (e.g.,
  `finding:f_9395cd1bd1be`, `ledger:s_abc/r3/critique`,
  `backtest:run_xyz/strategy=TopKLong`).
- Failure mode: when the corpus has no support for a claim, the assistant
  refuses or admits uncertainty rather than hallucinating.

### Approach

Standard RAG with project-specific guardrails:

1. **Indexing.** Build embeddings over a chunked view of the corpus.
   Embeddings provider: DeepInfra (`/v1/openai/embeddings`,
   `BAAI/bge-large-en-v1.5` or similar) with a deterministic on-disk
   cache. Replay mode supplies pre-computed embeddings from the bundled
   recording.
2. **Retrieval.** Top-K cosine over the cached vectors. Index is small
   (low thousands of chunks), so a single numpy matmul per query is fine.
   No FAISS needed at this scale.
3. **Generation.** A single LLM call (DeepInfra, the Adjudicator model is
   a sensible default) prompted with retrieved chunks and a strict
   "cite-or-refuse" system message.
4. **UI.** Replace the existing Ask-the-Memory panel with a chat-style
   `st.chat_message`/`st.chat_input` flow. Cited artifact IDs render as
   links to the corresponding cockpit panels.

### Non-trivial concerns to resolve when Phase 3 starts

- **Citation format.** Stable IDs that survive ledger growth; fail-loud
  when an ID can't be resolved.
- **Grounding test.** A small eval set of question-answer pairs whose
  ground truth lives in the artifacts; track answer-vs-truth + citation
  recall.
- **Replay mode parity.** Conversation must be reproducible without a
  DeepInfra key — bundle a small canned chat trace.
- **Refusal behaviour.** Easy to grade in the eval set: questions
  intentionally outside the corpus must trigger a refusal, not a
  hallucinated answer.

### Phase 3 sub-iterations (sketch)

- 3.1 Embeddings + index over existing corpus
- 3.2 Retrieval + grounded-generation prompt
- 3.3 Chat UI replaces Ask-the-Memory
- 3.4 Grounding eval set + scoring harness
- 3.5 Replay-mode bundling

---

## Phase 4 — Demo and deployment (SHIPPED)

### Goal

Reviewers (and the user) can experience AutoSignal-X via a clickable URL
without a local toolchain.

### Two sub-goals (both useful, ordered by reliability)

**4A — Static snapshot of the cockpit.** Generate a self-contained
multi-page HTML report from the existing artifacts: jinja2 templates +
embedded interactive Plotly figures + verbatim metric tables. Output goes
to `reports/cockpit_snapshot/` and is publishable via GitHub Pages with
zero runtime infrastructure. This is the **always-works** option and
should ship first.

**4B — Live deployed cockpit.** Streamlit Community Cloud is the chosen
target.

| Concern | Resolution |
|---|---|
| Hosting | Streamlit Community Cloud (free, GitHub-linked, auto-redeploy on push) |
| API keys | Replay mode default; agent panels read from bundled
  `replay/` artifacts. No DeepInfra key needed. |
| Model weights | Chronos-2 weights are already cached by the
  `chronos-forecasting` package on first import; cloud cold-starts will
  download once. Chronos-bolt-base is ~80MB, within free-tier limits. |
| Resource limits | Free tier has CPU/RAM caps. The deployed app reads
  pre-computed parquets and does NOT retrain by default. |
| Custom-study runs (Phase 2 dependency) | Optional. May exceed free
  tier; gate behind a "local-only" flag if it does. |
| Secrets | Streamlit secrets manager (TOML); used only if the user
  later enables non-replay modes. |

Hugging Face Spaces is a fallback if Streamlit Cloud has issues.

### Sub-goal C — Reviewer-runnable interactivity

Stretch: allow reviewers to actually run a small custom study from the
deployed app (Phase 2 must ship first). May require throttling and a
narrower default universe. Decide whether to keep this in scope after
Phases 1–3 land.

### Non-trivial concerns to resolve when Phase 4 starts

- Repo size: parquets and recorded LLM traces should stay <100MB total
  (GitHub Pages and Streamlit Cloud both have soft limits in this range).
  Audit `reports/` and `replay/` before deployment.
- Plotly figure size in the static HTML snapshot: large equity curves can
  bloat HTML. Pre-resample if needed.
- A "live demo" badge in the README that links to both the static report
  and the live app.

### Phase 4 sub-iterations (sketch)

- 4.1 Static HTML snapshot generator + GitHub Pages workflow
- 4.2 README badges + linked demo URLs
- 4.3 Streamlit Cloud deployment of the existing replay-mode app
- 4.4 (stretch) Custom-study runs in the deployed app

---

## Cross-phase notes

**Memory plan.** Each phase's start should write a project memory pinning
the active phase + sub-iteration. Each phase's end consolidates lessons
into the long-horizon `lessons.md` (existing infrastructure from Iter 16).

**REPORT.md discipline.** As phases land, append new sections rather than
rewriting the existing ones; the existing content is the historical
baseline new work compares against.

**Branching convention.** `phaseN-<theme>` per sub-iteration, merged
`--no-ff` into the integration branch (currently `main`). Once a phase is
fully merged, write a phase-summary commit on `main`.

**When to stop.** Each phase is independently shippable. If time runs out
mid-phase, the partial work still adds value (one new strategy, one new
panel) and should be merged with a clear status note in REPORT.
