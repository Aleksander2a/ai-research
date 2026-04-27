"""Phase 8 -- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

The observed maximum Sharpe across N strategies is biased upward.
Deflated Sharpe asks: given that you computed Sharpes for N candidate
strategies, what is the probability the *best one* would beat zero
under the null of zero true alpha?

DSR = Phi( ((SR_obs - SR_max_under_null) * sqrt(T-1)) / sqrt(1 - g_3*SR_obs + (g_4-1)/4 * SR_obs^2) )

where:
* T = number of observations
* g_3, g_4 = sample skewness and kurtosis of returns
* SR_max_under_null = expected max Sharpe under null = sqrt(2*ln(N)) for large N
  with a small Euler-correction term

A strategy passes the rigorous bar iff DSR > 0.95 (i.e. observed
Sharpe is in the top 5% of what the null would produce after running N
back-tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DeflatedSharpeResult:
    sharpe_observed: float
    sharpe_threshold_null: float  # expected max Sharpe under null
    deflated_sharpe: float
    n_trials: int
    n_observations: int


def expected_max_sharpe_under_null(n_trials: int) -> float:
    """E[max_i SR_i] for N i.i.d. standard-normal Sharpe estimates.

    Closed-form approximation via the extreme-value distribution:
    E[max] ≈ (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e))
    where gamma is Euler-Mascheroni."""
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649  # Euler-Mascheroni
    e = np.e
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float((1 - gamma) * z1 + gamma * z2)


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
) -> DeflatedSharpeResult:
    """Compute DSR for a single strategy's per-period returns.

    Args:
        returns: 1D array of per-period returns (e.g. daily portfolio
            log returns, or the daily realized return of a sign-following
            forecast strategy).
        n_trials: number of strategies the search examined (the agent's
            distinct hypotheses, not just the survivors).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = int(len(r))
    if T < 4 or n_trials < 1:
        return DeflatedSharpeResult(
            sharpe_observed=float("nan"),
            sharpe_threshold_null=float("nan"),
            deflated_sharpe=float("nan"),
            n_trials=n_trials,
            n_observations=T,
        )
    sigma = float(np.std(r, ddof=1))
    if sigma <= 0:
        return DeflatedSharpeResult(
            sharpe_observed=float("nan"),
            sharpe_threshold_null=float("nan"),
            deflated_sharpe=float("nan"),
            n_trials=n_trials,
            n_observations=T,
        )
    sr = float(np.mean(r) / sigma)
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=True, bias=False)) + 3.0  # convert to non-fisher
    sr_null = expected_max_sharpe_under_null(n_trials)
    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr, 1e-12))
    z = (sr - sr_null) * np.sqrt(T - 1.0) / denom
    dsr = float(stats.norm.cdf(z))
    return DeflatedSharpeResult(
        sharpe_observed=sr,
        sharpe_threshold_null=sr_null,
        deflated_sharpe=dsr,
        n_trials=n_trials,
        n_observations=T,
    )
