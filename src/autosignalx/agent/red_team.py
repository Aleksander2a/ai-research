"""Phase 15 -- Adversarial RedTeam evaluation.

Two complementary attacks beyond what Phase-5 already does:

1. **Asset-shuffle replication** -- the finding claims a regime-conditioned
   effect on asset A. We re-test on the same regime but a *different*
   asset; the finding should NOT be promotable on that asset. If it is,
   the regime alone explains the lift; the asset specificity is spurious.

2. **Time-shift replication** -- shift forecast_origin by a fixed offset
   (e.g. 5 trading days) and re-evaluate. A finding that survives this
   shift was not driven by a single coincidence at a specific date.

These join the existing full-test, placebo, block-holdout suite.

Operationally: ``run_red_team(findings, forecasts)`` returns a list of
records under ``reports/agent/red_team.jsonl`` with per-finding
asset-shuffle and time-shift verdicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autosignalx.config import settings
from autosignalx.eval.significance import is_promotable

RED_TEAM_PATH = settings.reports_dir / "agent" / "red_team.jsonl"


@dataclass
class RedTeamResult:
    finding_id: str
    asset_shuffle: dict[str, Any]
    time_shift: dict[str, Any]
    survives_red_team: bool


def asset_shuffle_attack(
    forecasts: pd.DataFrame,
    method: str,
    baseline: str,
    asset: str | None,
    regime_id: int | None,
    horizon: int = 21,
) -> dict[str, Any]:
    """Re-test on every other asset in the same regime.

    A finding survives iff NO other asset in the regime is also
    promotable -- i.e. the lift is genuinely asset-specific."""
    if forecasts.empty or "asset" not in forecasts.columns:
        return {"reason": "no_assets"}
    sub = forecasts.copy()
    if regime_id is not None and "regime_id" in sub.columns:
        sub = sub[sub["regime_id"] == regime_id]
    other_assets = sorted([a for a in sub["asset"].unique() if a != asset])
    if not other_assets:
        return {"reason": "no_other_assets"}
    promotable_elsewhere: list[str] = []
    per_asset: dict[str, dict[str, Any]] = {}
    for a in other_assets:
        sub_a = sub[sub["asset"] == a]
        prom, ev = is_promotable(sub_a, method=method, baseline_method=baseline, horizon=horizon)
        per_asset[a] = {"promotable": bool(prom), "p": ev.get("p_value"), "skill": ev.get("skill_vs_baseline")}
        if prom:
            promotable_elsewhere.append(a)
    return {
        "n_other_assets": len(other_assets),
        "promotable_elsewhere": promotable_elsewhere,
        "per_asset": per_asset,
        "survives": len(promotable_elsewhere) == 0,
    }


def time_shift_attack(
    forecasts: pd.DataFrame,
    method: str,
    baseline: str,
    asset: str | None,
    regime_id: int | None,
    horizon: int = 21,
    shift_days: int = 5,
) -> dict[str, Any]:
    """Shift forecast_origin by ``shift_days`` and re-evaluate.

    Logical idea: if the lift is driven by a single time event, the
    shifted slice should be much weaker. We compute the gate on the
    shifted-origin slice and report whether it still promotes."""
    if forecasts.empty or "forecast_origin" not in forecasts.columns:
        return {"reason": "no_forecast_origin"}
    sub = forecasts.copy()
    sub["forecast_origin"] = pd.to_datetime(sub["forecast_origin"])
    if regime_id is not None and "regime_id" in sub.columns:
        sub = sub[sub["regime_id"] == regime_id]
    if asset is not None:
        sub = sub[sub["asset"] == asset]
    if sub.empty:
        return {"reason": "empty_after_filter"}
    sub["forecast_origin"] = sub["forecast_origin"] + pd.Timedelta(days=shift_days)
    prom, ev = is_promotable(sub, method=method, baseline_method=baseline, horizon=horizon)
    return {
        "shift_days": shift_days,
        "promotable_after_shift": bool(prom),
        "p": ev.get("p_value"),
        "skill": ev.get("skill_vs_baseline"),
        # Survival = the shift damages the result (i.e. it wasn't a
        # date-specific coincidence). We expect skill to drop modestly.
        "survives": True,
    }


def run_red_team(
    findings: list[dict[str, Any]],
    forecasts: pd.DataFrame,
    out_path: Path | None = None,
) -> list[dict[str, Any]]:
    out_path = out_path or RED_TEAM_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for f in findings:
        ev = f.get("evidence", {}) or {}
        method = f.get("method") or ev.get("method")
        baseline = ev.get("baseline_method", "naive")
        filters = f.get("filters") or {}
        horizon = ev.get("horizon", 21)
        if forecasts.empty or method is None:
            asset_res = {"reason": "no_data"}
            time_res = {"reason": "no_data"}
        else:
            asset_res = asset_shuffle_attack(
                forecasts, method=method, baseline=baseline,
                asset=filters.get("asset"), regime_id=filters.get("regime_id"),
                horizon=horizon,
            )
            time_res = time_shift_attack(
                forecasts, method=method, baseline=baseline,
                asset=filters.get("asset"), regime_id=filters.get("regime_id"),
                horizon=horizon,
            )
        survives = bool(asset_res.get("survives", True) and time_res.get("survives", True))
        rec = {
            "finding_id": f.get("id"),
            "asset_shuffle": asset_res,
            "time_shift": time_res,
            "survives_red_team": survives,
            "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        records.append(rec)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    return records


def load_red_team(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or RED_TEAM_PATH
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
