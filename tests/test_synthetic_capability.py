"""Tests for the synthetic-known-answer benchmark + capability ablation."""

from __future__ import annotations

from pathlib import Path

from autosignalx.eval import capability_ablation
from autosignalx.eval import synthetic_benchmark as sb


def test_generate_universe_has_truths_and_distractors():
    u = sb.generate_universe(seed=0, planted_cells=2, distractor_cells=4)
    assert len(u.truths) == 2
    assert len(u.distractors) == 4
    assert not u.forecasts.empty
    # Required columns
    for col in ("asset", "timestamp", "forecast_origin", "method", "regime_id",
                "prediction", "target", "origin_value"):
        assert col in u.forecasts.columns
    # naive method always present
    assert "naive" in u.forecasts["method"].unique()


def test_grade_apparatus_recovers_some_planted_truth():
    u = sb.generate_universe(
        seed=42, planted_cells=2, distractor_cells=4,
        n_origins=120, planted_skill=0.45, noise_std=1.0,
    )
    res = sb.grade_apparatus(u)
    assert res["n_truths"] == 2
    # DM-only should pick at least one of the planted truths in this regime
    dm = next(g for g in res["gates"] if g["gate"] == "dm_only")
    assert dm["n_promoted"] >= 1
    # Strict gate's recall must be <= DM-only's recall (monotonic gate stack)
    strict = next(g for g in res["gates"] if g["gate"] == "strict")
    assert strict["recall"] <= dm["recall"] + 1e-9


def test_run_benchmark_writes_json(tmp_path: Path):
    out = tmp_path / "synthetic_benchmark.json"
    summary = sb.run_benchmark(
        n_trials=2, seed=7, planted_cells=2, distractor_cells=4,
        n_origins=80, planted_skill=0.40, out_path=out,
    )
    assert out.exists()
    assert summary["n_trials"] == 2
    assert "ablations" in summary
    gate_names = {row["gate"] for row in summary["ablations"]}
    assert {"dm_only", "+fdr", "+adversarial", "+rw", "+bayes", "strict"} <= gate_names


def test_capability_ablation_handles_no_ablations(tmp_path: Path):
    summary = capability_ablation.run_capability_ablation(
        reports_dir=tmp_path, out_path=tmp_path / "out.json",
    )
    assert summary["rows"] == []


def test_capability_ablation_runs_on_synthetic(tmp_path: Path):
    abl_dir = tmp_path / "ablations"
    abl_dir.mkdir(parents=True)
    # Drop a tiny synthetic ablation per method
    u = sb.generate_universe(seed=0, planted_cells=1, distractor_cells=1, n_origins=40)
    for method, sub in u.forecasts.groupby("method"):
        # only keep methods the ablation variants enumerate
        if method in ("naive", "good_M0"):
            sub = sub.copy()
            # Rename the planted method to the canonical "arima" so the variant
            # listing recognises it; keeps the test independent of the planted name.
            if method == "good_M0":
                sub["method"] = "arima"
            sub.to_parquet(abl_dir / f"{method if method == 'naive' else 'arima'}.parquet", index=False)
    out = tmp_path / "out.json"
    summary = capability_ablation.run_capability_ablation(
        reports_dir=tmp_path, out_path=out,
    )
    assert out.exists()
    rows = summary["rows"]
    assert rows
    variants = {r["variant"] for r in rows}
    assert {"baseline_only", "+arima"} <= variants
    # baseline_only's cost <= +arima's cost
    cost_by_variant = {r["variant"]: r.get("cost_proxy", 0) for r in rows}
    assert cost_by_variant["baseline_only"] <= cost_by_variant["+arima"]
