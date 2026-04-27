"""Exhaustive hypothesis sweep across the cached forecast cells.

For every (method, asset, regime) cell where we have data, runs the
DM + bootstrap promotion gate and -- when it passes -- registers the
hypothesis in the pre-registration ledger and promotes it to
``reports/agent/findings.jsonl``. This is the deterministic systematic
counterpart to the agent's narrative search loop; running both lets us
compare the search-space coverage the agent achieved against what an
exhaustive enumeration would surface.

The script is idempotent (pre-registrations and findings are de-duped on
content hash) and writes a JSON summary to
``reports/agent/sweep_summary.json`` for the cockpit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autosignalx.agent import findings as findings_mod
from autosignalx.agent import knowledge_graph as kg_mod
from autosignalx.agent.findings import make_session_id
from autosignalx.agent.tools import _load_all_forecasts, _load_regime_labels
from autosignalx.config import settings
from autosignalx.eval import preregistration as prereg_mod
from autosignalx.eval.significance import is_promotable

OUT_PATH = settings.reports_dir / "agent" / "sweep_summary.json"


def main() -> int:
    forecasts = _load_all_forecasts()
    regimes = _load_regime_labels()
    if forecasts.empty:
        print("[sweep] no forecasts cached; nothing to do.")
        return 0

    # Join regime labels onto every forecast row by forecast_origin.
    if not regimes.empty and "forecast_origin" in forecasts.columns:
        rl = regimes[["timestamp", "regime_id"]].rename(
            columns={"timestamp": "forecast_origin"}
        )
        rl["forecast_origin"] = pd.to_datetime(rl["forecast_origin"])
        forecasts["forecast_origin"] = pd.to_datetime(forecasts["forecast_origin"])
        forecasts = forecasts.merge(rl, on="forecast_origin", how="left")

    methods = sorted(m for m in forecasts["method"].unique() if m != "naive")
    assets = sorted(forecasts["asset"].unique())
    regime_ids: list[int | None] = (
        sorted(int(x) for x in forecasts["regime_id"].dropna().unique())
        if "regime_id" in forecasts.columns
        else []
    )
    if not regime_ids:
        regime_ids = [None]

    session_id = make_session_id() + "-sweep"
    print(
        f"[sweep] enumerating {len(methods)} methods x {len(assets)} assets "
        f"x {len(regime_ids)} regimes = "
        f"{len(methods) * len(assets) * len(regime_ids)} cells "
        f"under session_id={session_id}"
    )

    n_tested = 0
    n_skipped = 0
    n_promoted = 0
    n_registered = 0
    promoted_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    for m in methods:
        for a in assets:
            for r in regime_ids:
                sub = forecasts[(forecasts["method"].isin([m, "naive"]))
                                & (forecasts["asset"] == a)]
                if r is not None and "regime_id" in sub.columns:
                    sub = sub[sub["regime_id"] == r]
                if sub.empty:
                    n_skipped += 1
                    continue
                promotable, evidence = is_promotable(
                    sub, method=m, baseline_method="naive", horizon=21,
                )
                n_tested += 1
                row = {
                    "method": m,
                    "asset": a,
                    "regime_id": r,
                    "n": evidence.get("n"),
                    "promotable": promotable,
                    "p_value": evidence.get("p_value"),
                    "skill_vs_baseline": evidence.get("skill_vs_baseline"),
                    "ci_low": evidence.get("bootstrap_ci_low"),
                    "ci_high": evidence.get("bootstrap_ci_high"),
                }
                rows.append(row)
                if not promotable:
                    continue

                hypothesis_text = (
                    f"On asset {a}"
                    + (f" inside regime {r}" if r is not None else "")
                    + f", method `{m}` beats naive on the cached forecast slice "
                    f"under the DM + block-bootstrap gate "
                    f"(p={evidence.get('p_value', 0):.4f}, "
                    f"skill={evidence.get('skill_vs_baseline', 0):.4f})."
                )
                preg = prereg_mod.PreRegistration(
                    hypothesis=hypothesis_text,
                    method=m, baseline="naive",
                    filters={"asset": a, "regime_id": r},
                    decision_rule={
                        "p_threshold": 0.05,
                        "skill_threshold": 0.0,
                        "ci_must_be_positive": True,
                    },
                    predicted_effect={
                        "expected_skill": float(evidence.get("skill_vs_baseline", 0.0)),
                        "expected_p": float(evidence.get("p_value", 0.0) or 0.0),
                    },
                    falsifier=(
                        "Refuted iff DM p>=0.05 OR skill<=0 OR bootstrap CI "
                        "includes 0 on full-test or block-holdout replication."
                    ),
                    session_id=session_id,
                    proposer_role="exhaustive_sweep",
                )
                rec = prereg_mod.register(preg)
                n_registered += 1

                f = findings_mod.promote(
                    hypothesis=hypothesis_text,
                    method=m,
                    filters={"asset": a, "regime_id": r},
                    evidence=evidence,
                    agent_confidence="exhaustive sweep auto-promoted",
                    round=0,
                    session_id=session_id,
                    parent_hypothesis_ids=[rec["id"]],
                )
                fid = f.get("id")
                if fid:
                    n_promoted += 1
                    promoted_ids.append(fid)

    # Ingest the new findings into the persistent knowledge graph so the
    # Specialist Council panel + the cockpit's KG explorer pick them up.
    kg_result = kg_mod.ingest_findings(findings_mod.load())

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "session_id": session_id,
        "n_methods": len(methods),
        "n_assets": len(assets),
        "n_regimes": len(regime_ids),
        "n_tested": n_tested,
        "n_skipped": n_skipped,
        "n_pre_registered": n_registered,
        "n_promoted": n_promoted,
        "promoted_ids": promoted_ids,
        "kg_ingest": kg_result,
        "rows": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(
        f"[sweep] tested {n_tested} cells, skipped {n_skipped} (no data); "
        f"pre-registered {n_registered}, promoted {n_promoted} new findings; "
        f"KG: +{kg_result.get('nodes_added', 0)} nodes, "
        f"+{kg_result.get('edges_added', 0)} edges. "
        f"Summary -> {OUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
