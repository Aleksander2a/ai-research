"""Phase 14 -- Bayesian experimental design (Expected Information Gain).

The agent should not pick experiments uniformly at random. Given the
current posterior over hypothesis truth, the optimal next experiment
is the one with maximum *expected information gain* about the open
hypotheses.

For our setting we use a simple proxy:

    EIG(experiment) ≈ -log(p_existing) + alpha * novelty + beta * power

where:

* p_existing -- 1 if the (method, asset, regime, horizon) combination
  has already been tested; 0 otherwise. (We avoid re-testing settled
  cases.)
* novelty -- 1 if no prior finding lives in this slice; 0 if a strong
  finding already does.
* power -- crude estimate of statistical power = sqrt(n_samples).

We weight these heuristically (alpha=1.0, beta=0.1 by default) and rank
candidate experiments. The cockpit can render a coverage map of the
explored (method × asset × regime) cube colored by EIG.

This is not the full Bayes-optimal design, but it captures the *spirit*:
the agent avoids settled questions and prioritises power-adequate slices.
A future iteration can swap this for a posterior-uncertainty estimate
from the Phase-12 Bayesian model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExperimentCandidate:
    method: str
    asset: str
    regime_id: int | None
    eig_score: float
    n_samples: int
    novelty: float
    already_tested: bool


def _slice_key(method: str, asset: str | None, regime_id: int | None) -> tuple:
    return (str(method), str(asset) if asset else None, int(regime_id) if regime_id is not None else None)


def candidate_eig(
    forecasts: pd.DataFrame,
    methods: list[str],
    assets: list[str],
    regimes: list[int],
    findings: list[dict[str, Any]],
    tested_keys: set[tuple] | None = None,
    alpha_novelty: float = 1.0,
    beta_power: float = 0.1,
) -> list[ExperimentCandidate]:
    """Rank candidate experiments by an EIG proxy.

    A high score means: under-explored, sample-adequate, and not yet
    promoted. The agent's planner should consult this list before
    asking the Theorist to propose a new experiment."""
    if tested_keys is None:
        tested_keys = _build_tested_keys_from_ledger()

    promoted_keys = set()
    for f in findings:
        filters = f.get("filters") or {}
        promoted_keys.add(
            _slice_key(f.get("method", ""), filters.get("asset"), filters.get("regime_id"))
        )

    candidates: list[ExperimentCandidate] = []
    for m in methods:
        for a in assets:
            for r in regimes:
                key = _slice_key(m, a, r)
                already = key in tested_keys
                novelty = 0.0 if already or key in promoted_keys else 1.0
                # power proxy: how many rows would the slice produce?
                if forecasts.empty:
                    n = 0
                else:
                    sub = forecasts[
                        (forecasts["method"] == m) & (forecasts["asset"] == a)
                    ]
                    if "regime_id" in forecasts.columns:
                        sub = sub[sub["regime_id"] == r]
                    n = int(len(sub))
                power = math.sqrt(max(n, 0))
                # EIG proxy: novelty weight + power; subtract a penalty
                # for already-tested combinations (so they sink to the
                # bottom but remain visible)
                eig = alpha_novelty * novelty + beta_power * power
                if already:
                    eig -= 0.5
                candidates.append(
                    ExperimentCandidate(
                        method=m,
                        asset=a,
                        regime_id=r,
                        eig_score=float(eig),
                        n_samples=n,
                        novelty=novelty,
                        already_tested=already,
                    )
                )
    candidates.sort(key=lambda c: -c.eig_score)
    return candidates


def _build_tested_keys_from_ledger() -> set[tuple]:
    """Walk the ledger for prior slice_forecasts experiments."""
    try:
        from autosignalx.agent import ledger as ledger_mod
    except ImportError:
        return set()
    out: set[tuple] = set()
    for entry in ledger_mod.load():
        if entry.get("step") not in ("propose", "theorist"):
            continue
        content = entry.get("content")
        if not isinstance(content, dict):
            continue
        params = (content.get("experiment") or {}).get("params") or {}
        method = params.get("method")
        if not method:
            continue
        out.add(_slice_key(method, params.get("asset"), params.get("regime_id")))
    return out


def coverage_map(
    forecasts: pd.DataFrame,
    methods: list[str],
    assets: list[str],
    regimes: list[int],
    findings: list[dict[str, Any]],
    tested_keys: set[tuple] | None = None,
) -> pd.DataFrame:
    """Long-form (method, asset, regime, n_samples, status, eig) frame for the cockpit."""
    cands = candidate_eig(
        forecasts=forecasts,
        methods=methods,
        assets=assets,
        regimes=regimes,
        findings=findings,
        tested_keys=tested_keys,
    )
    rows = []
    promoted = set()
    for f in findings:
        filt = f.get("filters") or {}
        promoted.add(_slice_key(f.get("method", ""), filt.get("asset"), filt.get("regime_id")))
    for c in cands:
        key = _slice_key(c.method, c.asset, c.regime_id)
        if key in promoted:
            status = "promoted"
        elif c.already_tested:
            status = "tested"
        elif c.n_samples == 0:
            status = "no_data"
        else:
            status = "open"
        rows.append({
            "method": c.method,
            "asset": c.asset,
            "regime_id": c.regime_id,
            "n_samples": c.n_samples,
            "novelty": c.novelty,
            "eig_score": c.eig_score,
            "status": status,
        })
    return pd.DataFrame(rows)
