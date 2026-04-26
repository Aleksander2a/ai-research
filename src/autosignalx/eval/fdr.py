"""Multiple-comparison correction via Benjamini-Hochberg FDR.

The agent's auto-promotion gate evaluates each hypothesis individually
at p < 0.05. Across many hypotheses, the family-wise false-discovery
rate balloons. BH-FDR controls the expected proportion of false
discoveries among rejected nulls; we apply it across all promoted
findings as a post-hoc rigor check.

Returns per-finding adjusted p-values + a boolean ``survives_fdr``
flag, exposed by the cockpit's Survival panel and the
``autosignalx agent harden`` CLI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FDRResult:
    p_values: list[float]
    p_adjusted: list[float]
    survives: list[bool]
    alpha: float
    n_rejected: int


def benjamini_hochberg(p_values: list[float], alpha: float = 0.10) -> FDRResult:
    """Two-stage BH-FDR: rank p-values ascending, reject H_(i) iff
    p_(i) <= (i / m) * alpha. Returns adjusted p-values (q-values) via
    the standard step-up procedure.

    NaN inputs are treated as "did not reject" (q=1.0). Ties are handled
    by the natural ordering; ``alpha`` defaults to 0.10 (a research
    convention slightly more permissive than 0.05 for FDR vs FWER).
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return FDRResult(p_values=[], p_adjusted=[], survives=[], alpha=alpha, n_rejected=0)

    finite_mask = np.isfinite(p)
    p_clean = np.where(finite_mask, p, 1.0)
    order = np.argsort(p_clean)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)

    # Step-up adjusted p-values: q_(i) = min_{j>=i} ( m/j * p_(j) ), enforced monotone.
    sorted_p = p_clean[order]
    raw_q = sorted_p * m / np.arange(1, m + 1)
    monotone_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    monotone_q = np.minimum(monotone_q, 1.0)

    q_unsorted = np.empty(m, dtype=float)
    q_unsorted[order] = monotone_q

    survives = (q_unsorted <= alpha) & finite_mask
    return FDRResult(
        p_values=[float(x) for x in p],
        p_adjusted=[float(x) for x in q_unsorted],
        survives=[bool(x) for x in survives],
        alpha=alpha,
        n_rejected=int(survives.sum()),
    )
