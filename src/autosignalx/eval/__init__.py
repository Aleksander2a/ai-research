"""Evaluation layer (Iter 2) -- walk-forward harness, metrics, ablations.

Public API:
- ``contracts.assert_forecast_schema(df)``
- ``metrics.mae`` / ``mape`` / ``directional_accuracy`` / ``skill_score``
- ``harness.run_walk_forward(method_name, forecast_fn, ohlcv, windows)``
- ``harness.ablation(methods, ohlcv, windows)``
- ``harness.summarize(forecasts)`` / ``add_skill_score(summary)``

The harness defines the contract every forecasting method satisfies. The
contract is the seam that lets the regime, signal, graph, and agent layers
plug in without tearing the eval surface apart."""

from autosignalx.eval import contracts, harness, metrics  # noqa: F401
from autosignalx.eval.contracts import assert_forecast_schema  # noqa: F401
from autosignalx.eval.harness import (  # noqa: F401
    ablation,
    add_skill_score,
    run_walk_forward,
    summarize,
)
