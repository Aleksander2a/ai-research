"""Representation layer (L2) -- contrastive temporal encoder + KMeans + HMM
sanity-check baseline.

Public API:
- ``encoder.RegimeEncoder`` / ``train_encoder(features, ...)``
- ``cluster.kmeans_regimes(embeddings, ...)`` / ``hmm_regimes(features, ...)``
- ``labels.build_market_features()`` -- market-level feature matrix (SPY/QQQ
  returns + macro)
- ``labels.fit_and_save(...)`` -- end-to-end orchestration; writes
  reports/regimes/{kmeans,hmm,embeddings}.parquet
- ``labels.load_regime_labels(method)`` / ``add_regime_to_forecasts(...)``

Regime labels are joined to forecasts on ``forecast_origin`` to produce
regime-stratified metrics in the eval layer."""

from autosignalx.regime import cluster, encoder, labels  # noqa: F401
from autosignalx.regime.labels import (  # noqa: F401
    add_regime_to_forecasts,
    build_market_features,
    fit_and_save,
    load_regime_labels,
)
