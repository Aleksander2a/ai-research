"""Survival analysis: aggregate every finding through every gate.

For each promoted finding, recompute:
* the original DM + bootstrap evidence (already in ``findings.jsonl``),
* the BH-FDR-adjusted q-value across all findings (corrects for
  multiple comparisons),
* the three adversarial replications (full-test / placebo / holdout).

Persists to ``reports/agent/survival.jsonl`` so the cockpit can render
a pass/fail grid without recomputing on every panel load.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autosignalx.config import settings
from autosignalx.eval import adversarial as adv
from autosignalx.eval.fdr import benjamini_hochberg


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_forecast_frame(reports_dir: Path) -> pd.DataFrame:
    """Concatenate every ablation parquet so adversarial replication can
    operate on the union of methods (the original gate aligns method vs
    baseline within a single concatenated frame)."""
    abl_dir = reports_dir / "ablations"
    if not abl_dir.exists():
        return pd.DataFrame()
    frames = []
    for fp in sorted(abl_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(fp)
        except Exception:  # noqa: BLE001
            continue
        if "method" not in df.columns:
            df = df.copy()
            df["method"] = fp.stem
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    forecasts = pd.concat(frames, ignore_index=True)

    # Join regime_id on forecast_origin if a regime layer artifact exists.
    # The agent's promotion gate does this lookup at slice time; we
    # replicate it here so adversarial/placebo replications can read
    # regime labels off a single dataframe.
    regimes_path = reports_dir / "regimes" / "kmeans.parquet"
    if regimes_path.exists() and "forecast_origin" in forecasts.columns:
        try:
            rl = pd.read_parquet(regimes_path)
            if {"timestamp", "regime_id"} <= set(rl.columns):
                rl_join = rl[["timestamp", "regime_id"]].rename(columns={"timestamp": "forecast_origin"})
                rl_join["forecast_origin"] = pd.to_datetime(rl_join["forecast_origin"])
                forecasts["forecast_origin"] = pd.to_datetime(forecasts["forecast_origin"])
                forecasts = forecasts.merge(rl_join, on="forecast_origin", how="left")
        except Exception:  # noqa: BLE001
            pass
    return forecasts


def harden_findings(
    findings_path: Path | None = None,
    out_path: Path | None = None,
    fdr_alpha: float = 0.10,
    reports_dir: Path | None = None,
    cpcv_n_folds: int = 6,
    cpcv_k_test: int = 2,
    rw_alpha: float = 0.05,
    rw_n_bootstrap: int = 500,
    n_trials_override: int | None = None,
) -> list[dict[str, Any]]:
    """Apply FDR + adversarial replication + Phase-8 selection-bias gates.

    Phase-8 augments survival.jsonl with three additional fields per finding:

    * ``cpcv`` -- distribution of skill-vs-baseline across combinatorial purged
      cross-validation paths (mean / std / min / max / median).
    * ``romano_wolf`` -- adjusted p-value under Romano-Wolf step-down across the
      family of promoted findings (more powerful than BH-FDR under correlation).
    * ``deflated_sharpe`` -- DSR of the per-bar loss-difference treated as a
      return series, using ``preregistration.trial_count()`` as ``n_trials``
      (or ``n_trials_override`` when supplied).
    """
    rd = reports_dir or settings.reports_dir
    findings_path = findings_path or (rd / "agent" / "findings.jsonl")
    out_path = out_path or (rd / "agent" / "survival.jsonl")

    findings = _load_jsonl(findings_path)
    if not findings:
        return []

    forecasts = _load_forecast_frame(rd)
    have_forecasts = not forecasts.empty

    # Stage 1: BH-FDR over the original p-values.
    p_values = [f.get("evidence", {}).get("p_value", float("nan")) for f in findings]
    fdr = benjamini_hochberg(p_values, alpha=fdr_alpha)

    # Phase 8 -- precompute per-finding loss-difference vectors so we can run
    # Romano-Wolf jointly across the family and Deflated Sharpe per finding.
    diff_vectors: list[np.ndarray] = []
    if have_forecasts:
        from autosignalx.eval import adversarial as _adv_pre

        for f in findings:
            ev = f.get("evidence", {}) or {}
            method = f.get("method") or ev.get("method")
            baseline = ev.get("baseline_method", "naive")
            filters = f.get("filters") or ev.get("filters", {}) or {}
            sub = _adv_pre._filter_for_finding(forecasts, filters)
            if sub.empty or method is None:
                diff_vectors.append(np.array([]))
                continue
            keys = ["timestamp", "asset", "forecast_origin"]
            a = sub[sub["method"] == method][[*keys, "prediction", "target"]]
            b = sub[sub["method"] == baseline][[*keys, "prediction"]]
            merged = a.merge(b, on=keys, suffixes=("_method", "_baseline"))
            if merged.empty:
                diff_vectors.append(np.array([]))
                continue
            la = (merged["prediction_method"] - merged["target"]).abs().to_numpy()
            lb = (merged["prediction_baseline"] - merged["target"]).abs().to_numpy()
            diff_vectors.append(lb - la)

    # Romano-Wolf joint adjustment (truncate to common length so we can stack)
    rw_q: list[float | None] = [None] * len(findings)
    rw_survives: list[bool | None] = [None] * len(findings)
    if diff_vectors:
        usable_idx = [i for i, v in enumerate(diff_vectors) if len(v) > 30]
        if len(usable_idx) >= 2:
            min_len = min(len(diff_vectors[i]) for i in usable_idx)
            stacked = np.column_stack(
                [diff_vectors[i][-min_len:] for i in usable_idx]
            )
            from autosignalx.eval.romano_wolf import romano_wolf

            rw = romano_wolf(stacked, alpha=rw_alpha, n_bootstrap=rw_n_bootstrap)
            for j, i in enumerate(usable_idx):
                rw_q[i] = rw.p_adjusted[j]
                rw_survives[i] = rw.survives[j]

    # Determine effective n_trials for DSR
    if n_trials_override is not None:
        n_trials = int(n_trials_override)
    else:
        try:
            from autosignalx.eval.preregistration import trial_count as _trial_count

            n_trials = max(_trial_count(), len(findings))
        except Exception:  # noqa: BLE001
            n_trials = len(findings)

    # Phase 12 -- hierarchical Bayesian summary across findings
    bayes_per_finding: dict[str, dict[str, Any]] = {}
    if have_forecasts:
        try:
            from autosignalx.eval.bayesian import hierarchical_findings

            hsum = hierarchical_findings(findings, forecasts, baseline="naive")
            for bf in hsum.findings:
                bayes_per_finding[bf.finding_id] = {
                    "posterior_mean": bf.posterior_mean,
                    "posterior_sd": bf.posterior_sd,
                    "prob_positive": bf.prob_positive,
                    "bayes_factor": bf.bayes_factor,
                    "n": bf.n,
                }
        except Exception as exc:  # noqa: BLE001
            bayes_per_finding["__error__"] = {"reason": str(exc)}

    records: list[dict[str, Any]] = []
    for i, f in enumerate(findings):
        ev = f.get("evidence", {}) or {}
        method = f.get("method") or ev.get("method")
        baseline = ev.get("baseline_method", "naive")
        filters = f.get("filters") or ev.get("filters", {}) or {}
        horizon = ev.get("horizon", 21)

        rec: dict[str, Any] = {
            "finding_id": f.get("id"),
            "hypothesis": f.get("hypothesis"),
            "method": method,
            "filters": filters,
            "original_p": ev.get("p_value"),
            "original_skill": ev.get("skill_vs_baseline"),
            "fdr_alpha": fdr_alpha,
            "fdr_q": fdr.p_adjusted[i],
            "survives_fdr": fdr.survives[i],
            "rw_q": rw_q[i] if i < len(rw_q) else None,
            "survives_rw": rw_survives[i] if i < len(rw_survives) else None,
        }

        # CPCV: distribution of skill across combinatorial purged paths
        if have_forecasts and method:
            try:
                from autosignalx.eval import adversarial as _adv_cp
                from autosignalx.eval.cpcv import cpcv_skill_distribution

                sub = _adv_cp._filter_for_finding(forecasts, filters)
                cpcv_summary = cpcv_skill_distribution(
                    sub,
                    method=method,
                    baseline_method=baseline,
                    n_folds=cpcv_n_folds,
                    k_test=cpcv_k_test,
                )
            except Exception as exc:  # noqa: BLE001
                cpcv_summary = {"error": str(exc)}
            rec["cpcv"] = cpcv_summary
        else:
            rec["cpcv"] = {"error": "no_forecasts_available"}

        # Deflated Sharpe of the loss-difference series (treated as a daily return)
        if i < len(diff_vectors) and len(diff_vectors[i]) > 4:
            from autosignalx.eval.deflated_sharpe import deflated_sharpe_ratio

            dsr = deflated_sharpe_ratio(diff_vectors[i], n_trials=n_trials)
            rec["deflated_sharpe"] = {
                "sr_observed": dsr.sharpe_observed,
                "sr_threshold_null": dsr.sharpe_threshold_null,
                "dsr": dsr.deflated_sharpe,
                "n_trials": dsr.n_trials,
                "n_observations": dsr.n_observations,
            }
            rec["survives_dsr"] = bool(dsr.deflated_sharpe >= 0.95)
        else:
            rec["deflated_sharpe"] = None
            rec["survives_dsr"] = None

        if have_forecasts and method:
            adv_result = adv.adversarial_replication(
                forecasts=forecasts,
                method=method,
                baseline_method=baseline,
                filters=filters,
                horizon=horizon,
            )
            rec["adversarial"] = adv_result.to_dict()
            rec["survives_full_test"] = adv_result.full_test.get("promotable", False)
            # Placebo "survives" only if it actually ran AND failed to find a signal.
            # If the placebo couldn't run (e.g. no regime column), record as None
            # rather than silently passing.
            placebo_d = adv_result.placebo
            if placebo_d.get("reason"):
                rec["survives_placebo"] = None
            else:
                rec["survives_placebo"] = not placebo_d.get("promotable", True)
            rec["survives_block_holdout"] = adv_result.block_holdout.get("promotable", False)
        else:
            rec["adversarial"] = {"reason": "no_forecasts_available"}
            rec["survives_full_test"] = None
            rec["survives_placebo"] = None
            rec["survives_block_holdout"] = None

        rec["survives_all"] = bool(
            rec["survives_fdr"]
            and rec.get("survives_full_test")
            and rec.get("survives_placebo")
            and rec.get("survives_block_holdout")
        )
        # Strict bar: original Phase-5 gates + Phase-8 (RW + DSR + positive CPCV mean)
        cpcv_mean = (rec.get("cpcv") or {}).get("skill_mean")
        rec["survives_all_strict"] = bool(
            rec["survives_all"]
            and (rec.get("survives_rw") in (True, None))
            and (rec.get("survives_dsr") in (True, None))
            and (cpcv_mean is None or cpcv_mean > 0)
        )

        # Phase 12 -- attach Bayesian shrinkage estimate
        fid = rec["finding_id"]
        if fid in bayes_per_finding:
            rec["bayesian"] = bayes_per_finding[fid]
            rec["survives_bayes"] = bool(
                rec["bayesian"].get("prob_positive", 0.0) >= 0.95
                and rec["bayesian"].get("bayes_factor", 0.0) >= 10.0
            )
        else:
            rec["bayesian"] = None
            rec["survives_bayes"] = None

        rec["evaluated_at"] = datetime.now(UTC).isoformat()
        records.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    return records


def load_survival(reports_dir: Path | None = None) -> list[dict[str, Any]]:
    rd = reports_dir or settings.reports_dir
    return _load_jsonl(rd / "agent" / "survival.jsonl")
