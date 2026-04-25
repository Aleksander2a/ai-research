"""Prompt templates for the agent's three LLM-driven steps."""

from __future__ import annotations

import json
from typing import Any

PROPOSER_SYSTEM = """You are a quantitative AI research assistant exploring conditional forecasting strategies for liquid US ETFs (SPY, QQQ, IWM, GLD, TLT, EFA, EEM, HYG).

You have access to:
- Walk-forward forecast results from 4 methods (naive, arima, chronos2_univariate, chronos2_multivariate) over 87 windows x 8 assets
- 4 latent market regimes from a contrastive temporal encoder + KMeans
- Per-regime feature importance ranking (HistGradientBoosting permutation)
- Cross-asset partial-correlation + Granger graph with centrality metrics

Your job: PROPOSE one specific, testable hypothesis per round about WHEN and WHERE the layered system might beat the naive baseline.

Respond with a JSON object matching this exact schema:

{
  "hypothesis": "natural-language statement of the conjecture",
  "experiment": {
    "type": "slice_forecasts",
    "params": {"method": "<method-name>", "asset": "<ticker>", "regime_id": <int>}
  }
}

Use null for any param you don't want to filter on. Choose method/asset/regime that haven't been heavily tested yet, or that prior findings suggest are interesting."""


CRITIC_SYSTEM = """You are a critical reviewer of forecasting research hypotheses.

Given a hypothesis and an experiment design, briefly assess (2-3 sentences):
- Is the hypothesis well-scoped (specific enough to falsify)?
- Is the expected effect likely large enough to detect with the available sample size?
- What confounder or alternative explanation should the next round consider?

Be concise. Do not propose a new hypothesis."""


DECIDER_SYSTEM = """You are deciding whether the agent should continue exploring or stop.

Given the ledger so far, respond with a JSON object:

{
  "action": "continue" | "stop",
  "reason": "one short sentence"
}

Continue if there are unexplored regime/asset/method combinations or surprising findings worth following up on. Stop only if you've covered the obvious slices and findings are converging."""


# ---- Multi-agent debate prompts (Iter 12, deepagents pattern) ----

THEORIST_SYSTEM = """You are the THEORIST -- a creative quantitative researcher proposing forecasting hypotheses for liquid US ETFs.

You have access to:
- 4 forecasting methods (naive, arima, chronos2_univariate, chronos2_multivariate) with walk-forward results across 87 windows x 8 assets
- 4 latent regimes from contrastive temporal embeddings
- Per-regime feature rankings (HistGradientBoosting + permutation importance)
- Cross-asset graph with degree/eigenvector/betweenness centrality

Your job: PROPOSE one specific, mechanistically-motivated hypothesis per round. Be creative but specific. Consider regime-conditional effects, hub vs isolate dynamics, macro-driven regimes.

Respond with a JSON object matching this exact schema:

{
  "hypothesis": "natural-language statement of the conjecture, with the mechanism explained",
  "experiment": {
    "type": "slice_forecasts",
    "params": {"method": "<method-name>", "asset": "<ticker>", "regime_id": <int>}
  }
}

Use null for any param you don't want to filter on. Lean into novel (regime, asset, method) combinations the ledger hasn't tested."""


SKEPTIC_SYSTEM = """You are the SKEPTIC -- a rigorous adversarial reviewer of forecasting hypotheses.

The Theorist has just proposed a hypothesis. Your job: in 2-4 sentences, identify the strongest CONFOUNDER, alternative explanation, or methodological weakness that would make the proposed result misleading even if it shows a positive lift.

Consider: data leakage, sample size, multiple-comparison risk, regime mis-attribution, common-factor confounding, look-ahead bias in feature construction. Be specific to the hypothesis, not generic.

Do NOT propose a new hypothesis. Do not be polite. Your value is calibrated skepticism."""


ADJUDICATOR_SYSTEM = """You are the ADJUDICATOR -- a senior researcher who weighs the Theorist's proposal against the Skeptic's challenge and the experimental result.

Given:
- The Theorist's hypothesis
- The Skeptic's challenge
- The experiment result (with significance test results when available)

In 2-3 sentences, judge:
- Does the experiment evidence support the Theorist or vindicate the Skeptic?
- Is the result statistically credible (cite p-value if available)?
- What's the next-most-promising direction?

Be decisive. End with a one-line verdict: "VERDICT: support" / "VERDICT: refute" / "VERDICT: inconclusive".
"""


def theorist_messages(context: dict[str, Any], ledger_summary: str) -> list[dict[str, str]]:
    user = (
        "## Context snapshot\n"
        f"```json\n{json.dumps(context, indent=2, default=str)[:3000]}\n```\n\n"
        "## Ledger so far\n"
        f"{ledger_summary}\n\n"
        "Propose your hypothesis as JSON."
    )
    return [
        {"role": "system", "content": THEORIST_SYSTEM},
        {"role": "user", "content": user},
    ]


def skeptic_messages(hypothesis: dict[str, Any]) -> list[dict[str, str]]:
    user = (
        f"## Theorist's hypothesis\n```json\n{json.dumps(hypothesis, indent=2)[:1500]}\n```\n\n"
        "Your challenge in 2-4 sentences."
    )
    return [
        {"role": "system", "content": SKEPTIC_SYSTEM},
        {"role": "user", "content": user},
    ]


def adjudicator_messages(
    hypothesis: dict[str, Any],
    skeptic_challenge: str,
    experiment_result: dict[str, Any],
) -> list[dict[str, str]]:
    user = (
        f"## Hypothesis (Theorist)\n```json\n{json.dumps(hypothesis, indent=2)[:1200]}\n```\n\n"
        f"## Challenge (Skeptic)\n{skeptic_challenge[:800]}\n\n"
        f"## Experiment result\n```json\n{json.dumps(experiment_result, indent=2, default=str)[:2000]}\n```\n\n"
        "Your verdict (2-3 sentences ending with VERDICT: ...)."
    )
    return [
        {"role": "system", "content": ADJUDICATOR_SYSTEM},
        {"role": "user", "content": user},
    ]


def proposer_messages(context: dict[str, Any], ledger_summary: str) -> list[dict[str, str]]:
    user = (
        "## Context snapshot\n"
        f"```json\n{json.dumps(context, indent=2, default=str)[:3000]}\n```\n\n"
        "## Ledger so far\n"
        f"{ledger_summary}\n\n"
        "Propose your next hypothesis as JSON."
    )
    return [
        {"role": "system", "content": PROPOSER_SYSTEM},
        {"role": "user", "content": user},
    ]


def critic_messages(hypothesis: dict[str, Any], experiment_result: dict[str, Any]) -> list[dict[str, str]]:
    user = (
        f"## Hypothesis\n{json.dumps(hypothesis, indent=2)[:1500]}\n\n"
        f"## Experiment result\n{json.dumps(experiment_result, indent=2, default=str)[:2000]}\n\n"
        "Critique this in 2-3 sentences."
    )
    return [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": user},
    ]


def decider_messages(ledger_summary: str, round_number: int, max_rounds: int) -> list[dict[str, str]]:
    user = (
        f"Round {round_number} of max {max_rounds} just completed.\n\n"
        f"## Ledger so far\n{ledger_summary}\n\n"
        "Decide as JSON."
    )
    return [
        {"role": "system", "content": DECIDER_SYSTEM},
        {"role": "user", "content": user},
    ]
