"""Synthetic-known-answer benchmark for the discovery apparatus.

Real-market alpha is rare and noisy; that makes it impossible to
distinguish "the apparatus correctly grades a finding as null" from
"the apparatus is unable to recognise structure when it exists."

This module addresses that by generating synthetic price universes
where we **plant** specific (regime, asset, method) cells with a real
predictive edge and surround them with **distractor** cells that look
similar but carry no signal. The apparatus is then graded on:

* **Recall** -- fraction of planted truths the apparatus promotes.
* **False-discovery rate (FDR)** -- fraction of promoted findings that
  are distractors rather than planted truths.
* **Selection-bias-aware** survival counts at each gate (DM-only ->
  + FDR -> + adversarial -> + Romano-Wolf -> + Deflated Sharpe -> strict).

The generator is deterministic on a single ``seed``; running with
``n_trials`` re-seeds and averages, so the resulting recall/FDR are
sample means with bootstrap CIs. Output goes to
``reports/agent/synthetic_benchmark.json`` so the cockpit + static
snapshot can render it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autosignalx.config import settings
from autosignalx.eval.adversarial import adversarial_replication
from autosignalx.eval.bayesian import hierarchical_findings
from autosignalx.eval.fdr import benjamini_hochberg
from autosignalx.eval.romano_wolf import romano_wolf
from autosignalx.eval.significance import is_promotable

OUT_PATH = settings.reports_dir / "agent" / "synthetic_benchmark.json"


@dataclass(frozen=True)
class TruthRow:
    asset: str
    regime_id: int
    method: str
    skill: float


@dataclass
class SyntheticUniverse:
    forecasts: pd.DataFrame
    truths: list[TruthRow] = field(default_factory=list)
    distractors: list[tuple[str, int, str]] = field(default_factory=list)


def generate_universe(
    seed: int = 0,
    n_assets: int = 4,
    n_origins: int = 80,
    horizon: int = 5,
    n_regimes: int = 2,
    planted_cells: int = 3,
    distractor_cells: int = 9,
    planted_skill: float = 0.20,
    noise_std: float = 1.0,
) -> SyntheticUniverse:
    """Build a synthetic forecast frame with planted edges and distractor cells.

    Each row carries the standard forecast contract (timestamp, asset,
    forecast_origin, horizon, method, prediction, origin_value, target,
    regime_id) so it can be fed directly to the existing harness +
    hardening pipeline.

    Planted cell: method ``good_M{i}`` predicts the target with low noise
    on asset/regime cell ``(asset, regime)``, achieving roughly
    ``planted_skill`` MAE-vs-naive lift. Outside that cell, ``good_M{i}``
    is no better than naive.

    Distractor cell: method ``noise_M{j}`` predicts uniformly noisy values
    in *every* cell -- it has no real edge, but the agent might
    hallucinate one if its multiple-comparison correction is weak.
    """
    rng = np.random.default_rng(seed)
    assets = [f"A{i}" for i in range(n_assets)]
    methods_planted = [f"good_M{i}" for i in range(planted_cells)]
    methods_distractor = [f"noise_M{j}" for j in range(distractor_cells)]

    truths: list[TruthRow] = []
    rng_choice = np.random.default_rng(seed + 1)
    pool = [(a, r) for a in assets for r in range(n_regimes)]
    rng_choice.shuffle(pool)
    truth_cells = pool[:planted_cells]
    for k, (a, r) in enumerate(truth_cells):
        truths.append(TruthRow(asset=a, regime_id=r, method=methods_planted[k], skill=planted_skill))
    truth_set = {(t.asset, t.regime_id, t.method) for t in truths}

    distractor_cells_resolved: list[tuple[str, int, str]] = []
    for j, m in enumerate(methods_distractor):
        a = assets[j % n_assets]
        r = j % n_regimes
        if (a, r, m) not in truth_set:
            distractor_cells_resolved.append((a, r, m))

    rows: list[dict[str, Any]] = []
    base = pd.Timestamp("2024-01-01")
    for k in range(n_origins):
        origin = base + pd.Timedelta(days=k)
        regime = k % n_regimes
        for asset in assets:
            origin_value = float(100 + rng.normal(0, 0.5))
            for h in range(1, horizon + 1):
                ts = origin + pd.Timedelta(days=h)
                # Realised target: a random walk in price space.
                target = origin_value + float(rng.normal(0, noise_std))
                # naive: predicts last value
                rows.append({
                    "asset": asset, "timestamp": ts, "forecast_origin": origin,
                    "horizon": h, "regime_id": regime, "method": "naive",
                    "prediction": origin_value,
                    "origin_value": origin_value, "target": target,
                })
                # Planted methods: low-noise inside their cell, naive-like elsewhere.
                for t in truths:
                    if asset == t.asset and regime == t.regime_id:
                        # Target with planted_skill fraction less noise than naive
                        signal = float(rng.normal(0, max(noise_std * (1 - t.skill), 1e-3)))
                        pred = target - signal  # near target
                    else:
                        pred = origin_value + float(rng.normal(0, noise_std * 1.0))
                    rows.append({
                        "asset": asset, "timestamp": ts, "forecast_origin": origin,
                        "horizon": h, "regime_id": regime, "method": t.method,
                        "prediction": pred,
                        "origin_value": origin_value, "target": target,
                    })
                # Distractor methods: uniformly noisy everywhere
                for m in methods_distractor:
                    pred = origin_value + float(rng.normal(0, noise_std * 0.95))
                    rows.append({
                        "asset": asset, "timestamp": ts, "forecast_origin": origin,
                        "horizon": h, "regime_id": regime, "method": m,
                        "prediction": pred,
                        "origin_value": origin_value, "target": target,
                    })
    df = pd.DataFrame(rows)
    return SyntheticUniverse(
        forecasts=df, truths=truths, distractors=distractor_cells_resolved,
    )


@dataclass(frozen=True)
class GateResult:
    gate: str
    n_promoted: int
    recall: float
    fdr: float
    n_true_positives: int
    n_false_positives: int


def _evaluate_cell(
    forecasts: pd.DataFrame,
    method: str,
    asset: str,
    regime_id: int,
    horizon: int = 5,
) -> dict[str, Any]:
    sub = forecasts[(forecasts["asset"] == asset) & (forecasts["regime_id"] == regime_id)]
    promotable, evidence = is_promotable(
        sub, method=method, baseline_method="naive", horizon=horizon, min_samples=20,
    )
    return {"promotable": bool(promotable), **evidence, "asset": asset, "regime_id": regime_id, "method": method}


def _gate_metrics(
    promoted_keys: set[tuple[str, int, str]],
    truth_set: set[tuple[str, int, str]],
) -> tuple[int, int, float, float]:
    tp = len(promoted_keys & truth_set)
    fp = len(promoted_keys - truth_set)
    n = len(promoted_keys)
    recall = (tp / max(len(truth_set), 1)) if truth_set else float("nan")
    fdr = (fp / max(n, 1)) if n else 0.0
    return tp, fp, recall, fdr


def grade_apparatus(universe: SyntheticUniverse) -> dict[str, Any]:
    """Run the full hardening pipeline against a single synthetic universe.

    Reports per-gate recall + FDR. The gates are:

    * ``dm_only`` -- DM + bootstrap (the original promotion gate).
    * ``+fdr`` -- BH-FDR over the family of dm_only-promoted findings.
    * ``+adversarial`` -- + full-test + placebo + block-holdout.
    * ``+rw`` -- + Romano-Wolf joint stepdown.
    * ``+bayes`` -- + hierarchical Bayesian (BF >= 10 and P(theta>0) >= 0.95).
    * ``strict`` -- conjunction of every gate above.
    """
    fc = universe.forecasts
    truth_set = {(t.asset, t.regime_id, t.method) for t in universe.truths}
    candidate_methods = sorted(m for m in fc["method"].unique() if m != "naive")
    candidate_keys: list[tuple[str, int, str]] = []
    for m in candidate_methods:
        for a in sorted(fc["asset"].unique()):
            for r in sorted(int(x) for x in fc["regime_id"].unique()):
                candidate_keys.append((a, r, m))

    # Stage 1: DM-only
    p_values: list[float] = []
    diffs: list[np.ndarray] = []
    dm_promoted: list[tuple[str, int, str]] = []
    cell_records: list[dict[str, Any]] = []
    for a, r, m in candidate_keys:
        ev = _evaluate_cell(fc, method=m, asset=a, regime_id=r)
        cell_records.append(ev)
        p_values.append(ev.get("p_value") or 1.0)
        # Build the per-bar loss-difference series for RW + Bayes
        sub = fc[(fc["asset"] == a) & (fc["regime_id"] == r)]
        keys = ["timestamp", "asset", "forecast_origin"]
        a_df = sub[sub["method"] == m][[*keys, "prediction", "target"]]
        b_df = sub[sub["method"] == "naive"][[*keys, "prediction"]]
        merged = a_df.merge(b_df, on=keys, suffixes=("_method", "_baseline"))
        if not merged.empty:
            la = (merged["prediction_method"] - merged["target"]).abs().to_numpy()
            lb = (merged["prediction_baseline"] - merged["target"]).abs().to_numpy()
            diffs.append(lb - la)
        else:
            diffs.append(np.array([]))
        if ev["promotable"]:
            dm_promoted.append((a, r, m))

    dm_set = set(dm_promoted)
    tp, fp, rec, fdr = _gate_metrics(dm_set, truth_set)
    out_gates = {"dm_only": GateResult("dm_only", len(dm_set), rec, fdr, tp, fp)}

    # Stage 2: + FDR
    fdr_res = benjamini_hochberg(p_values, alpha=0.10)
    fdr_promoted = {
        candidate_keys[i] for i, s in enumerate(fdr_res.survives) if s
    } & dm_set
    tp, fp, rec, f = _gate_metrics(fdr_promoted, truth_set)
    out_gates["+fdr"] = GateResult("+fdr", len(fdr_promoted), rec, f, tp, fp)

    # Stage 3: + adversarial
    adv_pass: set[tuple[str, int, str]] = set()
    for a, r, m in fdr_promoted:
        adv = adversarial_replication(
            forecasts=fc, method=m, baseline_method="naive",
            filters={"asset": a, "regime_id": r}, horizon=5,
        )
        if adv.survives:
            adv_pass.add((a, r, m))
    tp, fp, rec, f = _gate_metrics(adv_pass, truth_set)
    out_gates["+adversarial"] = GateResult("+adversarial", len(adv_pass), rec, f, tp, fp)

    # Stage 4: + Romano-Wolf
    rw_pass: set[tuple[str, int, str]] = set()
    usable = [(i, candidate_keys[i]) for i, d in enumerate(diffs) if len(d) > 30]
    if len(usable) >= 2:
        min_len = min(len(diffs[i]) for i, _ in usable)
        stacked = np.column_stack([diffs[i][-min_len:] for i, _ in usable])
        rw = romano_wolf(stacked, alpha=0.05, n_bootstrap=200, block_size=10)
        for j, (_, key) in enumerate(usable):
            if rw.survives[j] and key in adv_pass:
                rw_pass.add(key)
    else:
        rw_pass = set(adv_pass)
    tp, fp, rec, f = _gate_metrics(rw_pass, truth_set)
    out_gates["+rw"] = GateResult("+rw", len(rw_pass), rec, f, tp, fp)

    # Stage 5: + Bayesian (treat each promoted key as a synthetic finding)
    synth_findings = [
        {"id": f"k_{a}_{r}_{m}", "method": m, "filters": {"asset": a, "regime_id": r},
         "evidence": {"baseline_method": "naive"}}
        for (a, r, m) in rw_pass
    ]
    bayes_pass: set[tuple[str, int, str]] = set()
    if synth_findings:
        hsum = hierarchical_findings(synth_findings, fc, baseline="naive")
        for bf in hsum.findings:
            if bf.prob_positive >= 0.95 and bf.bayes_factor >= 10.0:
                tag = bf.finding_id
                # Tag format: k_<asset>_<regime>_<method>. Method names can
                # contain underscores, so we scan rw_pass for the matching
                # tuple instead of string-splitting.
                for (a, r, m) in rw_pass:
                    if f"k_{a}_{r}_{m}" == tag:
                        bayes_pass.add((a, r, m))
                        break
    tp, fp, rec, f = _gate_metrics(bayes_pass, truth_set)
    out_gates["+bayes"] = GateResult("+bayes", len(bayes_pass), rec, f, tp, fp)

    # Strict bar = the conjunction (== bayes_pass since each gate is monotone)
    out_gates["strict"] = GateResult("strict", len(bayes_pass), rec, f, tp, fp)

    return {
        "n_truths": len(truth_set),
        "n_distractors": len(universe.distractors),
        "n_candidates": len(candidate_keys),
        "gates": [g.__dict__ for g in out_gates.values()],
    }


def run_benchmark(
    n_trials: int = 5,
    seed: int = 0,
    n_assets: int = 4,
    n_origins: int = 80,
    n_regimes: int = 2,
    planted_cells: int = 3,
    distractor_cells: int = 9,
    planted_skill: float = 0.20,
    noise_std: float = 1.0,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Run ``n_trials`` independent universes; aggregate per-gate recall/FDR."""
    out_path = out_path or OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_trial: list[dict[str, Any]] = []
    for t in range(n_trials):
        u = generate_universe(
            seed=seed + 1000 * t,
            n_assets=n_assets, n_origins=n_origins, n_regimes=n_regimes,
            planted_cells=planted_cells, distractor_cells=distractor_cells,
            planted_skill=planted_skill, noise_std=noise_std,
        )
        per_trial.append(grade_apparatus(u))

    # Aggregate per-gate
    gate_names = [g["gate"] for g in per_trial[0]["gates"]] if per_trial else []
    rows: list[dict[str, Any]] = []
    for gname in gate_names:
        recalls = [
            g["recall"] for trial in per_trial for g in trial["gates"]
            if g["gate"] == gname and g["recall"] == g["recall"]
        ]
        fdrs = [
            g["fdr"] for trial in per_trial for g in trial["gates"]
            if g["gate"] == gname
        ]
        n_promoted = [
            g["n_promoted"] for trial in per_trial for g in trial["gates"] if g["gate"] == gname
        ]
        rows.append({
            "gate": gname,
            "mean_recall": float(np.mean(recalls)) if recalls else float("nan"),
            "mean_fdr": float(np.mean(fdrs)) if fdrs else float("nan"),
            "mean_n_promoted": float(np.mean(n_promoted)) if n_promoted else 0.0,
        })

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_trials": n_trials,
        "planted_truths": planted_cells,
        "distractors": distractor_cells,
        "ablations": rows,
        "per_trial": per_trial,
        "config": {
            "n_assets": n_assets, "n_origins": n_origins, "n_regimes": n_regimes,
            "planted_skill": planted_skill, "noise_std": noise_std,
        },
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
