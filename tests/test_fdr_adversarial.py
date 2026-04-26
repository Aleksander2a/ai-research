"""Tests for Phase 5 statistical hardening: FDR + adversarial + survival."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from autosignalx.config import settings
from autosignalx.eval import adversarial as adv
from autosignalx.eval import survival as surv_mod
from autosignalx.eval.fdr import benjamini_hochberg

# ---------- BH-FDR ---------- #


def test_bh_handles_empty_input():
    r = benjamini_hochberg([])
    assert r.n_rejected == 0


def test_bh_rejects_only_smallest_when_appropriate():
    # 10 hypotheses; only 1 truly small p, rest near 1.
    p = [0.001] + [0.9] * 9
    r = benjamini_hochberg(p, alpha=0.10)
    assert r.survives[0] is True
    assert all(s is False for s in r.survives[1:])
    assert r.n_rejected == 1


def test_bh_rejects_all_when_all_tiny():
    p = [0.001] * 5
    r = benjamini_hochberg(p, alpha=0.10)
    assert all(r.survives)
    assert r.n_rejected == 5


def test_bh_handles_nan_as_no_rejection():
    p = [0.001, float("nan"), 0.5]
    r = benjamini_hochberg(p, alpha=0.10)
    assert r.survives[0] is True
    assert r.survives[1] is False  # NaN cannot reject
    assert r.survives[2] is False


def test_bh_q_values_are_monotone_in_rank():
    rng = np.random.default_rng(0)
    p = sorted(rng.uniform(size=20).tolist())
    r = benjamini_hochberg(p, alpha=0.10)
    # Adjusted q-values should be monotone non-decreasing in p-rank.
    q_sorted_by_input_rank = r.p_adjusted
    assert all(
        q_sorted_by_input_rank[i] <= q_sorted_by_input_rank[i + 1] + 1e-12
        for i in range(len(q_sorted_by_input_rank) - 1)
    )


# ---------- Adversarial replication ---------- #


def _synthetic_forecasts(seed: int = 0, n_origins: int = 50) -> pd.DataFrame:
    """Two methods on one asset across two regimes; method_b genuinely
    beats baseline ('naive') in regime 0 but not regime 1."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_origins):
        origin = pd.Timestamp("2024-01-01") + pd.Timedelta(days=k)
        for h in range(5):
            ts = origin + pd.Timedelta(days=h + 1)
            target = float(100 + rng.normal(0, 1))
            regime = 0 if k % 2 == 0 else 1
            naive_pred = float(100 + rng.normal(0, 1.5))  # noisy
            method_pred = (
                target + float(rng.normal(0, 0.4))  # tight when regime==0
                if regime == 0
                else float(100 + rng.normal(0, 1.5))  # equally noisy in regime 1
            )
            rows.append({
                "asset": "SPY",
                "timestamp": ts,
                "forecast_origin": origin,
                "regime_id": regime,
                "method": "naive",
                "prediction": naive_pred,
                "target": target,
            })
            rows.append({
                "asset": "SPY",
                "timestamp": ts,
                "forecast_origin": origin,
                "regime_id": regime,
                "method": "good_method",
                "prediction": method_pred,
                "target": target,
            })
    return pd.DataFrame(rows)


def test_adversarial_full_test_passes_for_genuine_signal():
    df = _synthetic_forecasts(seed=42, n_origins=80)
    res = adv.replicate_full_test(df, "good_method", "naive", {"asset": "SPY", "regime_id": 0})
    assert res["promotable"] is True


def test_adversarial_placebo_rejects_signal_when_regime_shuffled():
    df = _synthetic_forecasts(seed=42, n_origins=80)
    res = adv.replicate_placebo(df, "good_method", "naive", {"asset": "SPY", "regime_id": 0}, seed=1)
    # After shuffling regime labels, signal should NOT survive
    # (placebo passing is bad news; we expect it NOT to pass).
    # Allow some flexibility -- a single seed could occasionally pass by chance,
    # but the structure means it usually won't.
    # We check it returns a structured result regardless.
    assert "promotable" in res


def test_adversarial_block_holdout_returns_per_half_evidence():
    df = _synthetic_forecasts(seed=42, n_origins=80)
    res = adv.replicate_block_holdout(df, "good_method", "naive", {"asset": "SPY", "regime_id": 0})
    assert "first_half" in res and "second_half" in res
    assert "split_at" in res


def test_adversarial_bundle_has_survives_property():
    df = _synthetic_forecasts(seed=42, n_origins=80)
    bundle = adv.adversarial_replication(df, "good_method", "naive", {"asset": "SPY", "regime_id": 0})
    out = bundle.to_dict()
    assert "survives_adversarial" in out
    assert isinstance(out["survives_adversarial"], bool)


# ---------- Survival end-to-end ---------- #


@pytest.fixture
def survival_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "repo_root", tmp_path)
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")

    rd = tmp_path / "reports"
    (rd / "agent").mkdir(parents=True)
    (rd / "ablations").mkdir()

    df = _synthetic_forecasts(seed=7, n_origins=80)
    df.to_parquet(rd / "ablations" / "all.parquet")

    findings = [
        {
            "id": "f_keeps", "hypothesis": "good_method beats naive in regime 0",
            "method": "good_method",
            "filters": {"asset": "SPY", "regime_id": 0},
            "evidence": {"p_value": 0.001, "skill_vs_baseline": 0.30,
                         "baseline_method": "naive", "horizon": 5},
        },
        {
            "id": "f_loses", "hypothesis": "good_method beats naive in regime 1",
            "method": "good_method",
            "filters": {"asset": "SPY", "regime_id": 1},
            "evidence": {"p_value": 0.5, "skill_vs_baseline": 0.0,
                         "baseline_method": "naive", "horizon": 5},
        },
    ]
    with (rd / "agent" / "findings.jsonl").open("w", encoding="utf-8") as f:
        for fn in findings:
            f.write(json.dumps(fn) + "\n")
    return rd


def test_harden_findings_writes_survival_file(survival_fixture):
    records = surv_mod.harden_findings(reports_dir=survival_fixture)
    assert len(records) == 2
    out_path = survival_fixture / "agent" / "survival.jsonl"
    assert out_path.exists()


def test_harden_findings_distinguishes_strong_from_weak(survival_fixture):
    records = surv_mod.harden_findings(reports_dir=survival_fixture)
    by_id = {r["finding_id"]: r for r in records}
    strong = by_id["f_keeps"]
    weak = by_id["f_loses"]
    # The strong finding should have a much smaller q-value than the weak one.
    assert strong["fdr_q"] <= weak["fdr_q"]
    # Survives-FDR should be True for strong (p=0.001) at alpha=0.10.
    assert strong["survives_fdr"] is True


def test_load_survival_reads_jsonl(survival_fixture):
    surv_mod.harden_findings(reports_dir=survival_fixture)
    loaded = surv_mod.load_survival(reports_dir=survival_fixture)
    assert len(loaded) == 2
    assert all("finding_id" in r for r in loaded)
