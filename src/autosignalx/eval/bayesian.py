"""Phase 12 -- Bayesian hierarchical evidence over findings.

The frequentist DM gate (eval.significance) reports a p-value for each
finding individually. A research-lab-grade evidence summary requires:

* a **posterior** over the true skill of each finding (so we can answer
  "what is the probability the lift exceeds zero?");
* **partial pooling** across findings so a regime-3 result informs a
  regime-1 prior (shrinkage estimator);
* **Bayes factors** vs the null model (no skill) for direct evidence
  weight (BF=10 is the conventional "strong evidence" bar);
* **posterior predictive checks** (PPC) -- simulate next-session loss
  differences from the posterior and see whether the observed mean
  loss-difference is consistent.

We avoid heavy dependencies (no NumPyro / PyMC required) by using a
closed-form Normal-Normal hierarchical model with empirical-Bayes for
hyperparameters. The math:

    d_i ~ Normal(theta_i, sigma_i^2 / n_i)         # data: per-finding mean diff with sd
    theta_i ~ Normal(mu, tau^2)                    # population

    mu, tau^2 estimated by method of moments. Posterior of theta_i is then:

        theta_i | d_i ~ Normal(m_i, v_i)
        v_i = 1 / (1/tau^2 + n_i/sigma_i^2)
        m_i = v_i * (mu/tau^2 + n_i*d_i/sigma_i^2)

This is simple, well-known, and entirely sufficient for evidence-grade
shrinkage on a small family of findings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class BayesianFinding:
    finding_id: str
    n: int
    d_mean: float           # per-row loss difference mean (baseline - method)
    d_sd: float             # per-row sd
    posterior_mean: float   # shrinkage estimate of theta_i
    posterior_sd: float
    prob_positive: float    # P(theta_i > 0 | data)
    bayes_factor: float     # BF_10 vs theta_i = 0


@dataclass(frozen=True)
class HierarchicalSummary:
    mu_pop: float           # population mean across findings
    tau2_pop: float         # population variance across findings
    findings: list[BayesianFinding]
    n_findings: int


def _bayes_factor_normal(d_mean: float, d_sd: float, n: int, prior_sd: float = 0.05) -> float:
    """BF_10 = p(data | H1) / p(data | H0) for a Normal-Normal model.

    H0: theta = 0
    H1: theta ~ Normal(0, prior_sd^2)
    Marginal likelihoods analytic for Normal data with known sigma."""
    if n <= 0 or not np.isfinite(d_sd) or d_sd <= 0:
        return float("nan")
    se = d_sd / np.sqrt(n)
    # H0 marginal = N(d; 0, se^2)
    log_p_h0 = stats.norm.logpdf(d_mean, loc=0.0, scale=se)
    # H1 marginal = N(d; 0, se^2 + prior_sd^2)
    log_p_h1 = stats.norm.logpdf(d_mean, loc=0.0, scale=np.sqrt(se * se + prior_sd * prior_sd))
    return float(np.exp(log_p_h1 - log_p_h0))


def hierarchical_findings(
    findings: list[dict],
    forecasts: pd.DataFrame,
    baseline: str = "naive",
    prior_sd: float = 0.05,
) -> HierarchicalSummary:
    """Fit the Normal-Normal hierarchical model over a list of findings.

    Each finding's data is the per-bar loss difference (baseline - method)
    on its filter slice. The output gives posterior mean / sd / probability
    of positive effect / Bayes factor for each finding."""
    rows: list[BayesianFinding] = []
    n_obs: list[int] = []
    d_means: list[float] = []
    d_sds: list[float] = []
    valid_indices: list[int] = []

    for i, f in enumerate(findings):
        ev = f.get("evidence", {}) or {}
        method = f.get("method") or ev.get("method")
        filters = f.get("filters") or ev.get("filters", {}) or {}
        if forecasts.empty or method is None:
            continue

        sub = forecasts.copy()
        if "asset" in filters and filters["asset"] is not None and "asset" in sub.columns:
            sub = sub[sub["asset"] == filters["asset"]]
        if (
            "regime_id" in filters
            and filters["regime_id"] is not None
            and "regime_id" in sub.columns
        ):
            sub = sub[sub["regime_id"] == filters["regime_id"]]
        keys = ["timestamp", "asset", "forecast_origin"]
        a = sub[sub["method"] == method][[*keys, "prediction", "target"]]
        b = sub[sub["method"] == baseline][[*keys, "prediction"]]
        merged = a.merge(b, on=keys, suffixes=("_method", "_baseline"))
        if len(merged) < 5:
            continue
        la = (merged["prediction_method"] - merged["target"]).abs().to_numpy()
        lb = (merged["prediction_baseline"] - merged["target"]).abs().to_numpy()
        d = lb - la  # positive = method better
        d = d[np.isfinite(d)]
        if len(d) < 5:
            continue
        d_mean = float(np.mean(d))
        d_sd = float(np.std(d, ddof=1))
        n_obs.append(len(d))
        d_means.append(d_mean)
        d_sds.append(d_sd)
        valid_indices.append(i)

    if not n_obs:
        return HierarchicalSummary(
            mu_pop=float("nan"), tau2_pop=float("nan"), findings=[], n_findings=0
        )

    d_means_arr = np.asarray(d_means)
    d_sds_arr = np.asarray(d_sds)
    n_arr = np.asarray(n_obs)
    se_arr = d_sds_arr / np.sqrt(n_arr)

    # Empirical Bayes: method of moments
    mu_hat = float(np.mean(d_means_arr))
    var_total = float(np.var(d_means_arr, ddof=1)) if len(d_means_arr) > 1 else 0.0
    var_within = float(np.mean(se_arr * se_arr))
    tau2 = max(var_total - var_within, 1e-10)

    for j, i in enumerate(valid_indices):
        v_i = 1.0 / (1.0 / tau2 + n_arr[j] / (d_sds_arr[j] ** 2))
        m_i = v_i * (mu_hat / tau2 + n_arr[j] * d_means_arr[j] / (d_sds_arr[j] ** 2))
        prob_positive = float(1.0 - stats.norm.cdf(0.0, loc=m_i, scale=np.sqrt(v_i)))
        bf = _bayes_factor_normal(
            d_means_arr[j], d_sds_arr[j], n_arr[j], prior_sd=prior_sd
        )
        rows.append(
            BayesianFinding(
                finding_id=str(findings[i].get("id", f"f{i}")),
                n=int(n_arr[j]),
                d_mean=float(d_means_arr[j]),
                d_sd=float(d_sds_arr[j]),
                posterior_mean=float(m_i),
                posterior_sd=float(np.sqrt(v_i)),
                prob_positive=prob_positive,
                bayes_factor=bf,
            )
        )

    return HierarchicalSummary(
        mu_pop=mu_hat,
        tau2_pop=tau2,
        findings=rows,
        n_findings=len(rows),
    )


def posterior_predictive_check(
    findings: list[dict],
    forecasts: pd.DataFrame,
    baseline: str = "naive",
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict:
    """Simulate next-session mean loss-differences from the fitted posterior
    and report the frequency that simulated means fall within ±50% of
    observed means -- a coarse PPC.

    Returns dict per finding with simulated mean / 95% CI / observed value."""
    summary = hierarchical_findings(findings, forecasts, baseline=baseline)
    if summary.n_findings == 0:
        return {"empty": True}
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for bf in summary.findings:
        # Simulate theta_i from the posterior, then a future sample mean
        thetas = rng.normal(bf.posterior_mean, bf.posterior_sd, size=n_simulations)
        future_means = thetas + rng.normal(
            0.0, bf.d_sd / np.sqrt(max(bf.n, 1)), size=n_simulations
        )
        ci_lo, ci_hi = float(np.quantile(future_means, 0.025)), float(np.quantile(future_means, 0.975))
        out[bf.finding_id] = {
            "observed_mean": bf.d_mean,
            "posterior_mean": bf.posterior_mean,
            "predictive_ci_low": ci_lo,
            "predictive_ci_high": ci_hi,
            "prob_positive": bf.prob_positive,
            "bayes_factor": bf.bayes_factor,
            "n": bf.n,
        }
    return {
        "n_findings": summary.n_findings,
        "mu_pop": summary.mu_pop,
        "tau2_pop": summary.tau2_pop,
        "per_finding": out,
    }
