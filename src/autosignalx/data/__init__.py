"""Data layer (Iter 1) — yfinance + macro pulls, parquet caching, walk-forward splits.

Public API:
- ``fetch.fetch_all(assets, macro, start, end)`` — pull from yfinance.
- ``cache.read_ohlcv()`` / ``read_macro()`` — load cached parquet.
- ``loader.load_returns_wide()`` / ``load_close_wide()`` / ``load_macro_wide()``.
- ``splits.walk_forward_windows(val_end, test_end, horizon_days, step_days)``.
- ``splits.StaticSplit(train_end, val_end, test_end).slice(df)``.

The data layer guarantees temporal ordering (asserted at write time via
``schema.assert_*``) and provides leakage-free splits for the eval harness."""

from autosignalx.data import cache, fetch, loader, schema, splits  # noqa: F401
from autosignalx.data.splits import (  # noqa: F401
    StaticSplit,
    WalkForwardWindow,
    walk_forward_windows,
)
