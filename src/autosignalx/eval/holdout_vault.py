"""Phase 8 -- Holdout vault: never-touched final test slice.

A research lab freezes a slice of data the model/agent never sees.
After all research is "done," that slice is opened *once* and the
final headline metric is reported on it. Reviewers see the difference
between the in-sample finding and the holdout-final result; if they
diverge, the research apparatus over-promoted.

The vault is a pickle of (start, end) timestamps and a hash of the
slice contents. Operations:

* ``initialize_vault(start, end)`` -- declare and lock; subsequent
  agent runs cannot include forecasts whose forecast_origin is in
  [start, end].
* ``open_vault()`` -- one-time evaluation; records the headline metric
  to ``reports/agent/holdout_eval.json`` and disables further opens.
* ``vault_status()`` -- inspect (locked / opened / never-initialized).

The lock semantics are advisory (we cannot prevent a determined
researcher from inspecting a parquet); their value is making accidental
look-ahead a deliberate, audit-trailed action.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autosignalx.config import settings

VAULT_DIR = settings.reports_dir / "agent" / "holdout_vault"
VAULT_META = VAULT_DIR / "vault.json"
VAULT_RESULTS = VAULT_DIR / "results.json"


def _ensure_dir() -> Path:
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    return VAULT_DIR


def initialize_vault(
    start: str,
    end: str,
    description: str = "",
) -> dict[str, Any]:
    """Declare and lock a holdout vault. Idempotent on (start, end)."""
    _ensure_dir()
    if VAULT_META.exists():
        existing = json.loads(VAULT_META.read_text(encoding="utf-8"))
        if existing.get("start") == start and existing.get("end") == end:
            return existing
        raise RuntimeError(
            f"Vault already initialized with different bounds: {existing}"
        )
    rec = {
        "start": start,
        "end": end,
        "description": description,
        "locked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "opened": False,
    }
    rec["lock_hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]
    VAULT_META.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def vault_status() -> dict[str, Any]:
    if not VAULT_META.exists():
        return {"initialized": False}
    return {**json.loads(VAULT_META.read_text(encoding="utf-8")), "initialized": True}


def is_in_vault(timestamps: pd.Series) -> pd.Series:
    """Boolean mask of timestamps that fall inside the locked vault."""
    status = vault_status()
    if not status.get("initialized"):
        return pd.Series([False] * len(timestamps), index=timestamps.index)
    start = pd.Timestamp(status["start"])
    end = pd.Timestamp(status["end"])
    ts = pd.to_datetime(timestamps)
    return (ts >= start) & (ts <= end)


def assert_no_vault_leakage(forecasts: pd.DataFrame) -> None:
    """Raise if any forecast row's ``forecast_origin`` is inside the vault.

    Called by every agent-authored experiment that produces forecasts;
    a fail here means the research pipeline accidentally consulted the
    holdout window."""
    status = vault_status()
    if not status.get("initialized"):
        return
    if "forecast_origin" not in forecasts.columns or forecasts.empty:
        return
    in_vault = is_in_vault(forecasts["forecast_origin"])
    if in_vault.any():
        n = int(in_vault.sum())
        raise RuntimeError(
            f"Holdout-vault leakage: {n} forecast rows have forecast_origin in "
            f"[{status['start']}, {status['end']}]. Discovery / research is "
            f"forbidden inside the vault. Open the vault explicitly via "
            f"``autosignalx eval vault-open`` to evaluate."
        )


def open_vault(
    forecasts: pd.DataFrame,
    methods: list[str],
    baseline: str = "naive",
) -> dict[str, Any]:
    """One-time evaluation on the vault slice.

    Computes per-method MAE on the locked slice, the skill vs baseline,
    and writes the result to ``reports/agent/holdout_vault/results.json``.
    Marks the vault as opened so subsequent calls warn rather than
    silently re-evaluating."""
    status = vault_status()
    if not status.get("initialized"):
        raise RuntimeError("Vault not initialized.")
    if status.get("opened"):
        existing = (
            json.loads(VAULT_RESULTS.read_text(encoding="utf-8"))
            if VAULT_RESULTS.exists()
            else {}
        )
        return {"already_opened": True, "previous_results": existing}

    start = pd.Timestamp(status["start"])
    end = pd.Timestamp(status["end"])
    f = forecasts.copy()
    f["forecast_origin"] = pd.to_datetime(f["forecast_origin"])
    sub = f[(f["forecast_origin"] >= start) & (f["forecast_origin"] <= end)]
    if sub.empty:
        return {"empty": True, "n_rows": 0}

    sub = sub.copy()
    sub["abs_err"] = (sub["prediction"] - sub["target"]).abs()
    per_method_mae = sub.groupby("method", observed=True)["abs_err"].mean().to_dict()
    baseline_mae = float(per_method_mae.get(baseline, float("nan")))
    results = {
        "vault": status,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_rows": int(len(sub)),
        "per_method_mae": {str(k): float(v) for k, v in per_method_mae.items()},
        "skill_vs_baseline": {
            str(m): (1.0 - per_method_mae[m] / baseline_mae) if baseline_mae > 0 else None
            for m in methods
        },
    }
    VAULT_RESULTS.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    status["opened"] = True
    status["opened_at"] = results["evaluated_at"]
    VAULT_META.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return results
