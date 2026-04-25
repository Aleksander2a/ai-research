"""Contracts for data DataFrames.

The eval harness and all model layers depend on these schemas; tests assert
them and ``cache.write_*`` enforces them at the persistence boundary."""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "returns",
)

MACRO_COLUMNS: tuple[str, ...] = ("timestamp", "signal", "value")


def assert_ohlcv_schema(df: pd.DataFrame) -> None:
    """Validate an OHLCV long-format DataFrame.

    Required columns are present; for each asset, timestamps are
    strictly monotonically increasing (i.e., temporally ordered)."""
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {sorted(missing)}")
    for asset, sub in df.groupby("asset", observed=True):
        if not sub["timestamp"].is_monotonic_increasing:
            raise ValueError(
                f"OHLCV timestamps for asset {asset!r} are not monotonically increasing"
            )


def assert_macro_schema(df: pd.DataFrame) -> None:
    """Validate a macro long-format DataFrame."""
    missing = set(MACRO_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Macro DataFrame missing columns: {sorted(missing)}")
    for signal, sub in df.groupby("signal", observed=True):
        if not sub["timestamp"].is_monotonic_increasing:
            raise ValueError(
                f"Macro timestamps for signal {signal!r} are not monotonically increasing"
            )
