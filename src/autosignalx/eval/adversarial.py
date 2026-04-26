"""Adversarial replication: try hard to break a promoted finding.

A real research scientist's first move after a positive result is to
attack it. This module implements three attacks layered on top of the
existing promotion gate:

1. **full_test_replication** -- re-run DM + bootstrap on the full test
   window, ignoring any per-spec window cap (e.g. the agent's default
   ``max_windows=8``). A finding that holds on a small slice but
   collapses on the full window is overfit to the slice.

2. **placebo_replication** -- shuffle regime labels (preserving the
   marginal distribution) and re-run the gate. A finding that survives
   a placebo regime label is structurally suspect: the "regime" was
   not the explanatory variable.

3. **block_holdout_replication** -- split the test window 50/50 by
   forecast_origin, run the gate on each half independently, and
   require both halves to be promotable. Catches findings driven by a
   single sub-period.

A finding's ``survives_adversarial`` flag is the conjunction of all
three. The cockpit's Survival panel reports each independently so a
reviewer can see *which* attack a finding fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from autosignalx.eval.significance import is_promotable


@dataclass
class AdversarialResult:
    full_test: dict[str, Any]
    placebo: dict[str, Any]
    block_holdout: dict[str, Any]

    @property
    def survives(self) -> bool:
        return bool(
            self.full_test.get("promotable", False)
            and not self.placebo.get("promotable", True)
            and self.block_holdout.get("promotable", False)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_test": self.full_test,
            "placebo": self.placebo,
            "block_holdout": self.block_holdout,
            "survives_adversarial": self.survives,
        }


def _filter_for_finding(
    forecasts: pd.DataFrame, filters: dict[str, Any]
) -> pd.DataFrame:
    df = forecasts
    if "asset" in filters:
        df = df[df["asset"] == filters["asset"]]
    if "regime_id" in filters and "regime_id" in df.columns:
        df = df[df["regime_id"] == filters["regime_id"]]
    return df


def replicate_full_test(
    forecasts: pd.DataFrame,
    method: str,
    baseline_method: str,
    filters: dict[str, Any],
    horizon: int = 21,
) -> dict[str, Any]:
    """Run the gate on the full forecast frame for the finding's slice."""
    sliced = _filter_for_finding(forecasts, filters)
    promotable, evidence = is_promotable(
        sliced, method=method, baseline_method=baseline_method, horizon=horizon
    )
    return {"promotable": bool(promotable), **evidence}


def replicate_placebo(
    forecasts: pd.DataFrame,
    method: str,
    baseline_method: str,
    filters: dict[str, Any],
    horizon: int = 21,
    seed: int = 42,
) -> dict[str, Any]:
    """Shuffle regime labels and re-run the gate.

    If the finding's mechanism really depends on the regime, the
    shuffled-label slice should NOT be promotable. ``promotable=True``
    here is *bad news* for the finding."""
    if "regime_id" not in forecasts.columns or "regime_id" not in filters:
        return {"promotable": False, "reason": "no_regime_column"}

    rng = np.random.default_rng(seed)
    shuffled = forecasts.copy()
    # Preserve the marginal distribution of regime labels by permutation.
    shuffled["regime_id"] = rng.permutation(shuffled["regime_id"].to_numpy())
    sliced = _filter_for_finding(shuffled, filters)
    promotable, evidence = is_promotable(
        sliced, method=method, baseline_method=baseline_method, horizon=horizon
    )
    return {"promotable": bool(promotable), **evidence}


def replicate_block_holdout(
    forecasts: pd.DataFrame,
    method: str,
    baseline_method: str,
    filters: dict[str, Any],
    horizon: int = 21,
) -> dict[str, Any]:
    """Split the slice 50/50 by forecast_origin time and require both halves
    to pass the gate independently."""
    sliced = _filter_for_finding(forecasts, filters)
    if "forecast_origin" not in sliced.columns or sliced.empty:
        return {"promotable": False, "reason": "no_forecast_origin"}
    origins = sorted(pd.to_datetime(sliced["forecast_origin"]).unique())
    if len(origins) < 4:
        return {"promotable": False, "reason": "insufficient_origins", "n_origins": len(origins)}
    midpoint = origins[len(origins) // 2]
    first = sliced[pd.to_datetime(sliced["forecast_origin"]) < midpoint]
    second = sliced[pd.to_datetime(sliced["forecast_origin"]) >= midpoint]
    p1, e1 = is_promotable(first, method, baseline_method, horizon=horizon)
    p2, e2 = is_promotable(second, method, baseline_method, horizon=horizon)
    return {
        "promotable": bool(p1 and p2),
        "first_half": {"promotable": bool(p1), **e1},
        "second_half": {"promotable": bool(p2), **e2},
        "split_at": str(midpoint),
    }


def adversarial_replication(
    forecasts: pd.DataFrame,
    method: str,
    baseline_method: str,
    filters: dict[str, Any],
    horizon: int = 21,
    placebo_seed: int = 42,
) -> AdversarialResult:
    """Run all three adversarial replications and bundle the result."""
    return AdversarialResult(
        full_test=replicate_full_test(forecasts, method, baseline_method, filters, horizon),
        placebo=replicate_placebo(forecasts, method, baseline_method, filters, horizon, placebo_seed),
        block_holdout=replicate_block_holdout(forecasts, method, baseline_method, filters, horizon),
    )
