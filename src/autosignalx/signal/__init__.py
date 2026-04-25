"""Reasoning layer (L3) — feature engineering + TabPFN signal relevance ranking.

Implementation lands in **Iter 5**. Ranks candidate exogenous signals
per regime; top-K signals are injected into the forecast as residual
correction. See README for the iteration plan."""
