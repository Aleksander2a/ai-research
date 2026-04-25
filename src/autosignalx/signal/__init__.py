"""Reasoning layer (L3) -- feature engineering + TabPFN per-regime ranking.

Public API:
- ``features.build_features_target(asset_ohlcv, macro_wide, horizon_days)``
- ``features.compute_rsi(prices, window)`` / ``compute_macd_signal(prices)``
- ``features.feature_columns(df)``
- ``ranking.rank_features_per_regime(features_df, regime_labels, feature_cols, ...)``

Output schema (per ranking row):
``(regime_id, feature, importance, importance_std, n_samples, rank)``
Rank 1 = most important within the regime."""

from autosignalx.signal import features, ranking  # noqa: F401
from autosignalx.signal.features import (  # noqa: F401
    build_features_target,
    compute_macd_signal,
    compute_rsi,
    feature_columns,
)
from autosignalx.signal.ranking import rank_features_per_regime  # noqa: F401
