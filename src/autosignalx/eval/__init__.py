"""Evaluation layer — walk-forward harness, metrics, ablations, significance tests.

Implementation lands in **Iter 2** (baselines + harness) and grows in
subsequent iterations as each model layer plugs into it. The harness
defines the contract every model layer satisfies (regime-stratified
DataFrame schema). See README for the iteration plan."""
