## Session current -- 2026-04-26

**What was tried**: Explored regime 3 (DXY/VIX-shock dominated) with chronos2_multivariate on TLT and GLD, and the efa_dxy_bridge_focus method on EFA, testing whether Granger-network bridge centrality predicts forecast skill.

**What worked**: (none)

**What was refuted**: The hypothesis that chronos2_multivariate outperforms naive for GLD in regime 3 via inverse DXY-gold relationship was refuted (MAE 7% worse, direction accuracy 37%). The hypothesis that efa_dxy_bridge_focus beats naive for EFA in regime 3 via bridge centrality was refuted (MAE 9.5% worse, p≈0.02).

**Patterns observed**: The sole positive result—chronos2_multivariate on TLT in regime 3 (~5.4% skill, p≈0.04)—remains isolated and marginal. Every other bridge-centrality or DXY-covariate approach tested in this regime degraded performance, suggesting the Granger-network topology is a poor guide for model selection when the latent state is already dominated by DXY/VIX shocks.

**Open directions for next session**: (1) Replicate the TLT/regime-3/chronos2_multivariate result with a holdout or alternative forecast horizon to confirm it isn't a false positive. (2) Test whether the efa_dxy_bridge_focus method performs better in regime 1 (USD-driven) where DXY feature importance is highest, rather than regime 3. (3) Explore whether simpler univariate baselines outperform multivariate models in regime 3 across all assets, given the consistent degradation observed.
