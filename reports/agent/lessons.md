## Session current -- 2026-04-26

**What was tried**: Explored regime 3 (DXY/VIX-shock dominated) with chronos2_multivariate on TLT and GLD, and the efa_dxy_bridge_focus method on EFA, testing whether Granger-network bridge centrality predicts forecast skill.

**What worked**: (none)

**What was refuted**: The hypothesis that chronos2_multivariate outperforms naive for GLD in regime 3 via inverse DXY-gold relationship was refuted (MAE 7% worse, direction accuracy 37%). The hypothesis that efa_dxy_bridge_focus beats naive for EFA in regime 3 via bridge centrality was refuted (MAE 9.5% worse, p≈0.02).

**Patterns observed**: The sole positive result—chronos2_multivariate on TLT in regime 3 (~5.4% skill, p≈0.04)—remains isolated and marginal. Every other bridge-centrality or DXY-covariate approach tested in this regime degraded performance, suggesting the Granger-network topology is a poor guide for model selection when the latent state is already dominated by DXY/VIX shocks.

**Open directions for next session**: (1) Replicate the TLT/regime-3/chronos2_multivariate result with a holdout or alternative forecast horizon to confirm it isn't a false positive. (2) Test whether the efa_dxy_bridge_focus method performs better in regime 1 (USD-driven) where DXY feature importance is highest, rather than regime 3. (3) Explore whether simpler univariate baselines outperform multivariate models in regime 3 across all assets, given the consistent degradation observed.


---

## Session 20260427-2500372e -- 2026-04-27

**What was tried**: Explored HYG (Regime 0), EFA (Regime 1), GLD (Regime 2), and TLT (Regime 3) using various covariate restrictions and naive ensemble blending methods.

**What worked**:  
- `f_b38c6394a5fe`: TLT in Regime 3 showed statistically significant improvement (p=0.014) when modeled with only DX-Y.NYB as a covariate.

**What was refuted**:  
- HYG's returns cannot be adequately explained by SPY and TLT alone.  
- Blending naive forecasts with chronos2_multivariate predictions does not improve performance in Regime 3.  
- EFA's sensitivity to dollar strength in Regime 1 is not strong enough to justify restricting covariates to DX-Y.NYB.  
- GLD's predictability in Regime 2 is not enhanced by downweighting macro signals.

**Patterns observed**: Assets with low eigenvector centrality (e.g., TLT) can still exhibit regime-specific predictability when conditioned on dominant macro drivers (e.g., DX-Y.NYB in Regime 3).

**Open directions for next session**:  
1. Explore TLT's sensitivity to DX-Y.NYB in other regimes.  
2. Investigate whether other low-centrality assets exhibit similar regime-specific predictability.  
3. Test alternative ensemble methods for blending naive and multivariate forecasts in Regime 3.


---

## Session 20260427-6a269b3f -- 2026-04-27

**What was tried**: Explored TLT predictability in regime 3 (USD-dominated) using hybrid signals blending naive momentum with DXY sensitivity, and naive ensembles combining TLT lagged returns with DXY movements.

**What worked**:  
- `f_abedb3261bd3`: Hybrid signal blending naive momentum with DXY sensitivity improved TLT predictability in regime 3 (p=0.004).  
- `f_3eb10d54f55f`: Naive ensemble of TLT lagged returns with DXY movements enhanced predictability in regime 3 (p=0.003).  

**What was refuted**:  
- Hypothesis that EFA predictability in regime 1 (USD-dominated) improves with DXY-conditioned Chronos2 model (p=0.25).  
- Hypothesis that GLD acts as a latent inflation hedge in regime 2 (oil/yield-dominated) with naive-univariate blend (p=0.346).  

**Patterns observed**: TLT, despite its low centrality, consistently benefits from USD sensitivity in regime 3, suggesting its role as a safe-haven asset is amplified during USD dominance.  

**Open directions for next session**:  
- Explore TLT predictability in regime 3 using alternative USD proxies (e.g., EUR/USD).  
- Investigate whether TLT's USD sensitivity extends to other regimes with elevated USD importance.  
- Test if blending TLT with other safe-haven assets (e.g., GLD) improves predictability in regime 3.


---

## Session 20260427-dea16a9f -- 2026-04-27

**What was tried**: Explored TLT predictability in Regime 3 (DXY-dominated) using conditional Chronos2 models and naive-Chronos2 ensembles.

**What worked**:  
- `f_8a446564ee92`: Conditional Chronos2 model for TLT using DXY improved MAE (p=0.014).  
- `f_c3a3c0300a24`: Naive-Chronos2 ensemble for TLT outperformed baseline (p=0.002).

**What was refuted**:  
- Hypothesis that naive forecasts for GLD outperform multivariate models in Regime 2 (CL=F/^TNX-dominated).  
- Hypothesis that DXY-only Chronos2 improves EFA predictability in Regime 1.

**Patterns observed**: In DXY-dominated regimes, TLT shows consistent inverse sensitivity to DXY movements, making it amenable to DXY-conditioned models.

**Open directions for next session**:  
1. Validate TLT-DXY relationship in Regime 3 using out-of-sample regime definitions to address look-ahead bias concerns.  
2. Explore alternative ensemble weights for TLT-DXY models beyond 50-50 splits.  
3. Investigate whether TLT's low centrality in Regime 3 persists across other DXY-dominated regimes.
