"""Convenience loader API. Re-exports common read functions and provides
wide-format pivots used by the cockpit, the regime layer (Iter 4), and
the graph layer (Iter 6)."""

from __future__ import annotations

import pandas as pd

from autosignalx.data.cache import read_macro, read_ohlcv  # noqa: F401  (public re-export)


def load_returns_wide() -> pd.DataFrame:
    """Returns matrix indexed by timestamp, one column per asset."""
    df = read_ohlcv()
    return df.pivot(index="timestamp", columns="asset", values="returns").sort_index()


def load_close_wide() -> pd.DataFrame:
    """Adjusted close matrix indexed by timestamp, one column per asset."""
    df = read_ohlcv()
    return df.pivot(index="timestamp", columns="asset", values="adj_close").sort_index()


def load_macro_wide() -> pd.DataFrame:
    """Macro signal value matrix indexed by timestamp."""
    df = read_macro()
    return df.pivot(index="timestamp", columns="signal", values="value").sort_index()
