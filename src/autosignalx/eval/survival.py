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
) -> list[dict[str, Any]]:
    """Apply FDR + adversarial replication to every promoted finding.

    Returns the per-finding survival records and writes them to
    ``reports/agent/survival.jsonl``.
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
        }

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
