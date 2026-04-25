"""Representation layer (L2) — contrastive temporal encoder + KMeans regimes,
with HMM as a sanity-check baseline.

Implementation lands in **Iter 4**. Regime labels feed back into the eval
harness for stratified metrics and into the signal layer for per-regime
feature ranking. See README for the iteration plan."""
