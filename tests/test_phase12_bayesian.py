"""Phase 12 tests: hierarchical Bayesian evidence and PPC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autosignalx.eval import bayesian


def _findings_and_forecasts(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    findings = []
    for fi, regime in enumerate([0, 1]):
        for k in range(80):
            origin = pd.Timestamp("2024-01-01") + pd.Timedelta(days=k)
            for h in range(1, 4):
                ts = origin + pd.Timedelta(days=h)
                target = float(100 + rng.normal(0, 1))
                rows.append({
                    "asset": "SPY", "timestamp": ts, "forecast_origin": origin,
                    "horizon": h, "method": "naive", "regime_id": regime,
                    "prediction": float(100 + rng.normal(0, 1.5)),
                    "origin_value": 100.0, "target": target,
                })
                rows.append({
                    "asset": "SPY", "timestamp": ts, "forecast_origin": origin,
                    "horizon": h, "method": "good", "regime_id": regime,
                    "prediction": (
                        target + float(rng.normal(0, 0.4))  # genuinely better in regime 0
                        if regime == 0
                        else float(100 + rng.normal(0, 1.5))  # noise in regime 1
                    ),
                    "origin_value": 100.0, "target": target,
                })
        findings.append({
            "id": f"f_{fi}",
            "method": "good",
            "filters": {"asset": "SPY", "regime_id": regime},
            "evidence": {"baseline_method": "naive"},
        })
    return findings, pd.DataFrame(rows)


def test_hierarchical_findings_returns_one_per_input():
    findings, fc = _findings_and_forecasts(seed=42)
    h = bayesian.hierarchical_findings(findings, fc)
    assert h.n_findings == 2
    ids = {bf.finding_id for bf in h.findings}
    assert ids == {"f_0", "f_1"}


def test_hierarchical_strong_signal_high_prob_positive():
    findings, fc = _findings_and_forecasts(seed=42)
    h = bayesian.hierarchical_findings(findings, fc)
    by_id = {bf.finding_id: bf for bf in h.findings}
    # f_0 is the genuinely-better regime
    assert by_id["f_0"].prob_positive > 0.8
    assert by_id["f_0"].bayes_factor > 1.0
    # f_1 (noise regime) should have lower prob_positive
    assert by_id["f_1"].prob_positive < by_id["f_0"].prob_positive


def test_posterior_predictive_check_runs():
    findings, fc = _findings_and_forecasts(seed=42)
    ppc = bayesian.posterior_predictive_check(findings, fc, n_simulations=200)
    assert ppc["n_findings"] == 2
    assert "f_0" in ppc["per_finding"]
    f0 = ppc["per_finding"]["f_0"]
    assert "predictive_ci_low" in f0
    assert "predictive_ci_high" in f0
    assert f0["predictive_ci_low"] <= f0["predictive_ci_high"]


def test_hierarchical_handles_no_findings():
    h = bayesian.hierarchical_findings([], pd.DataFrame())
    assert h.n_findings == 0


def test_bayes_factor_handles_zero_n():
    bf = bayesian._bayes_factor_normal(0.0, 0.0, 0)
    assert np.isnan(bf)
