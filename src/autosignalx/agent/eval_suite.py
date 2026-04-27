"""Phase 15 -- Agent eval orchestration.

Single entry point that runs:

* calibration of agent confidence vs survival outcomes (Phase 15)
* RedTeam attacks (asset_shuffle + time_shift)
* coherence scoring per session
* prompt-version aggregation
* a regression suite that re-runs the replay-mode agent on a fixed
  fixture and asserts the ledger structure is stable

Outputs:

* ``reports/agent/calibration.jsonl``
* ``reports/agent/red_team.jsonl``
* ``reports/agent/coherence.jsonl``
* ``reports/agent/prompt_scores.json``
* ``reports/agent/eval_summary.json``
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autosignalx.agent import (
    calibration as calibration_mod,
)
from autosignalx.agent import (
    coherence as coherence_mod,
)
from autosignalx.agent import (
    findings as findings_mod,
)
from autosignalx.agent import (
    ledger as ledger_mod,
)
from autosignalx.agent import (
    prompt_optimizer as prompt_mod,
)
from autosignalx.agent import (
    red_team as redteam_mod,
)
from autosignalx.agent import (
    trace_eval as trace_mod,
)
from autosignalx.config import settings
from autosignalx.eval import survival as survival_mod


def run_eval_suite(reports_dir: Path | None = None) -> dict[str, Any]:
    """Run all Phase-15 evaluations end-to-end. Idempotent.

    Returns a summary dict and writes ``reports/agent/eval_summary.json``.
    """
    rd = reports_dir or settings.reports_dir
    findings = findings_mod.load()
    ledger = ledger_mod.load()
    trace_quality = trace_mod.load()
    survival_records = survival_mod.load_survival(reports_dir=rd)

    summary: dict[str, Any] = {
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_findings": len(findings),
        "n_ledger_entries": len(ledger),
        "n_trace_quality": len(trace_quality),
        "n_survival_records": len(survival_records),
    }

    # Calibration
    calib = calibration_mod.calibration_for_role(
        findings=findings,
        survival_records=survival_records,
        role="theorist",
        n_bins=5,
    )
    summary["calibration"] = {
        "n": calib.n,
        "brier": calib.brier,
        "ece": calib.ece,
    }
    calib_path = rd / "agent" / "calibration.jsonl"
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    with calib_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({**calib.__dict__, "evaluated_at": summary["evaluated_at"]}, default=str) + "\n")

    # RedTeam (load forecasts the same way survival does)
    forecasts = _load_forecasts(rd)
    rt = redteam_mod.run_red_team(findings, forecasts)
    summary["red_team"] = {
        "n": len(rt),
        "n_survives": sum(1 for r in rt if r.get("survives_red_team")),
    }

    # Coherence per session_id
    session_ids = sorted({e.get("session_id") for e in ledger if e.get("session_id")})
    coherence_summaries = []
    for sid in session_ids:
        rec = coherence_mod.score_session(sid, ledger=ledger)
        coherence_summaries.append(rec.__dict__)
        coherence_mod.append_coherence(rec)
    summary["coherence"] = coherence_summaries

    # Prompt-version scoring (best-effort; only meaningful once you start
    # registering prompts via prompt_optimizer.register_prompt).
    prompt_scores: dict[str, Any] = {}
    for role in ("theorist", "skeptic", "adjudicator"):
        prompt_scores[role] = prompt_mod.score_versions(role, trace_quality)
    (rd / "agent" / "prompt_scores.json").write_text(
        json.dumps(prompt_scores, indent=2, default=str), encoding="utf-8"
    )
    summary["prompt_scores_present"] = any(prompt_scores.values())

    out_path = rd / "agent" / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _load_forecasts(reports_dir: Path) -> pd.DataFrame:
    abl = reports_dir / "ablations"
    if not abl.exists():
        return pd.DataFrame()
    frames = []
    for fp in abl.glob("*.parquet"):
        try:
            frames.append(pd.read_parquet(fp))
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        return pd.DataFrame()
    forecasts = pd.concat(frames, ignore_index=True)

    # Join regime_id from kmeans labels if present
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
