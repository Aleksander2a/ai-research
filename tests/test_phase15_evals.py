"""Phase 15 tests: calibration, RedTeam attacks, coherence, prompt versioning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from autosignalx.agent import (
    calibration as calibration_mod,
)
from autosignalx.agent import (
    coherence as coherence_mod,
)
from autosignalx.agent import (
    prompt_optimizer as prompt_mod,
)
from autosignalx.agent import (
    red_team as redteam_mod,
)

# ---------- Calibration ---------- #


def test_coerce_confidence_handles_text():
    assert calibration_mod._coerce_confidence("high") == 0.8
    assert calibration_mod._coerce_confidence(0.65) == 0.65
    assert calibration_mod._coerce_confidence("85") == 0.85
    assert calibration_mod._coerce_confidence(None) is None


def test_calibration_with_known_outcomes():
    findings = [
        {"id": "f1", "predicted_effect": {"expected_skill": 0.8}},
        {"id": "f2", "predicted_effect": {"expected_skill": 0.2}},
        {"id": "f3", "predicted_effect": {"expected_skill": 0.9}},
    ]
    survival = [
        {"finding_id": "f1", "survives_all_strict": True},
        {"finding_id": "f2", "survives_all_strict": False},
        {"finding_id": "f3", "survives_all_strict": True},
    ]
    rec = calibration_mod.calibration_for_role(findings, survival, role="theorist")
    assert rec.n == 3
    assert rec.brier < 0.1  # well-calibrated example


def test_calibration_zero_observations():
    rec = calibration_mod.calibration_for_role([], [], role="theorist")
    assert rec.n == 0
    assert np.isnan(rec.brier)


# ---------- RedTeam ---------- #


def _two_asset_two_regime_forecasts(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for asset in ["TLT", "SPY"]:
        for regime in [0, 1]:
            for k in range(40):
                origin = pd.Timestamp("2024-01-01") + pd.Timedelta(days=k)
                for h in range(1, 4):
                    ts = origin + pd.Timedelta(days=h)
                    target = float(100 + rng.normal(0, 1))
                    rows.append({
                        "asset": asset, "timestamp": ts, "forecast_origin": origin,
                        "horizon": h, "method": "naive", "regime_id": regime,
                        "prediction": float(100 + rng.normal(0, 1.5)),
                        "origin_value": 100.0, "target": target,
                    })
                    rows.append({
                        "asset": asset, "timestamp": ts, "forecast_origin": origin,
                        "horizon": h, "method": "good", "regime_id": regime,
                        "prediction": (
                            target + float(rng.normal(0, 0.4))
                            if (asset == "TLT" and regime == 0)
                            else float(100 + rng.normal(0, 1.5))
                        ),
                        "origin_value": 100.0, "target": target,
                    })
    return pd.DataFrame(rows)


def test_asset_shuffle_attack_runs():
    df = _two_asset_two_regime_forecasts(seed=42)
    res = redteam_mod.asset_shuffle_attack(
        df, method="good", baseline="naive", asset="TLT", regime_id=0,
    )
    assert "n_other_assets" in res
    assert "promotable_elsewhere" in res
    assert "survives" in res


def test_time_shift_attack_runs():
    df = _two_asset_two_regime_forecasts(seed=42)
    res = redteam_mod.time_shift_attack(
        df, method="good", baseline="naive", asset="TLT", regime_id=0,
    )
    assert "shift_days" in res
    assert "promotable_after_shift" in res


def test_run_red_team_writes_jsonl(tmp_path: Path, monkeypatch):
    df = _two_asset_two_regime_forecasts(seed=42)
    findings = [{
        "id": "f_test",
        "method": "good",
        "filters": {"asset": "TLT", "regime_id": 0},
        "evidence": {"baseline_method": "naive", "horizon": 21},
    }]
    out_path = tmp_path / "red_team.jsonl"
    monkeypatch.setattr(redteam_mod, "RED_TEAM_PATH", out_path)
    records = redteam_mod.run_red_team(findings, df, out_path=out_path)
    assert len(records) == 1
    assert out_path.exists()
    loaded = redteam_mod.load_red_team(path=out_path)
    assert len(loaded) == 1


# ---------- Coherence ---------- #


def test_lessons_uptake_matches_phrase():
    proposals = [
        {"hypothesis": "Test in regime three for TLT betweenness"},
        {"hypothesis": "Different topic entirely"},
    ]
    lessons = "regime three is the dollar-driven state; TLT high betweenness centrality"
    score = coherence_mod.lessons_uptake(proposals, lessons)
    assert 0 < score <= 1.0


def test_theme_persistence_entropy_zero_when_one_cell():
    proposals = [
        {"experiment": {"params": {"asset": "SPY", "regime_id": 0}}},
        {"experiment": {"params": {"asset": "SPY", "regime_id": 0}}},
    ]
    e = coherence_mod.theme_persistence_entropy(proposals)
    assert e == 0.0


def test_theme_persistence_entropy_nonzero_when_diverse():
    proposals = [
        {"experiment": {"params": {"asset": "SPY", "regime_id": 0}}},
        {"experiment": {"params": {"asset": "QQQ", "regime_id": 1}}},
        {"experiment": {"params": {"asset": "TLT", "regime_id": 2}}},
    ]
    e = coherence_mod.theme_persistence_entropy(proposals)
    assert e > 0.5


def test_lineage_branching_factor_zero_when_no_edges():
    bf = coherence_mod.lineage_branching_factor({"nodes": [{"id": "a"}], "edges": []})
    assert bf == 0.0


# ---------- Prompt versioning ---------- #


def test_register_prompt_idempotent(tmp_path: Path):
    p = tmp_path / "theorist.jsonl"
    v1 = prompt_mod.register_prompt("theorist", "Be a quant.", path=p)
    v2 = prompt_mod.register_prompt("theorist", "Be a quant.", path=p)
    assert v1.version_id == v2.version_id
    assert len(prompt_mod.load_versions("theorist", path=p)) == 1


def test_register_prompt_new_version_appends(tmp_path: Path):
    p = tmp_path / "theorist.jsonl"
    prompt_mod.register_prompt("theorist", "v1 prompt", path=p)
    prompt_mod.register_prompt("theorist", "v2 prompt", path=p)
    assert len(prompt_mod.load_versions("theorist", path=p)) == 2


def test_score_versions_returns_aggregates(tmp_path: Path):
    p = tmp_path / "theorist.jsonl"
    prompt_mod.register_prompt("theorist", "v1", path=p)
    # Patch _path_for_role to use our temp path
    import autosignalx.agent.prompt_optimizer as pm

    orig = pm._path_for_role
    pm._path_for_role = lambda role: p
    try:
        traces = [{"session_id": "s1", "clarity": 4, "novelty": 3, "falsifiability": 4, "evidence_citing": 5}]
        out = pm.score_versions("theorist", traces)
        assert len(out) == 1
        assert out[0]["avg_clarity"] == 4.0
    finally:
        pm._path_for_role = orig
