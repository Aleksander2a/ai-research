"""Data layer — yfinance + macro pulls, parquet caching, walk-forward splits.

Implementation lands in **Iter 1**. The data layer guarantees temporal
ordering and provides leakage-tested train/val/test splits for the eval
harness. See README for the iteration plan."""
