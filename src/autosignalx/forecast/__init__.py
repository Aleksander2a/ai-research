"""Forecasting layer (L1) — Chronos-2 with multivariate covariates,
plus classical baselines for ablation.

Implementation lands in **Iter 3**. The forecasting layer produces
probabilistic point + interval forecasts; outputs follow the eval
harness DataFrame contract. See README for the iteration plan."""
