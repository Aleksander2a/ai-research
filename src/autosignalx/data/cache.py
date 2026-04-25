"""Parquet read/write for cached data. Files live under ``data/cache/``.

The cache is the persistent contract between fetch (writer) and loader
(reader). Schema enforcement happens at write time via ``assert_*_schema``
so corrupt data never reaches the eval harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from autosignalx.config import settings
from autosignalx.data.schema import assert_macro_schema, assert_ohlcv_schema


def _cache_root() -> Path:
    root = settings.data_dir / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _path(name: str) -> Path:
    return _cache_root() / f"{name}.parquet"


def write_ohlcv(df: pd.DataFrame) -> Path:
    """Persist an OHLCV frame after validating its schema."""
    assert_ohlcv_schema(df)
    path = _path("ohlcv")
    df.to_parquet(path, index=False)
    return path


def write_macro(df: pd.DataFrame) -> Path:
    """Persist a macro frame after validating its schema."""
    assert_macro_schema(df)
    path = _path("macro")
    df.to_parquet(path, index=False)
    return path


def read_ohlcv() -> pd.DataFrame:
    """Load the cached OHLCV frame and validate it.

    Raises ``FileNotFoundError`` with a helpful hint if the cache is empty."""
    path = _path("ohlcv")
    if not path.exists():
        raise FileNotFoundError(
            f"No cached OHLCV at {path}. Run `autosignalx data fetch` first."
        )
    df = pd.read_parquet(path)
    assert_ohlcv_schema(df)
    return df


def read_macro() -> pd.DataFrame:
    """Load the cached macro frame and validate it."""
    path = _path("macro")
    if not path.exists():
        raise FileNotFoundError(
            f"No cached macro at {path}. Run `autosignalx data fetch` first."
        )
    df = pd.read_parquet(path)
    assert_macro_schema(df)
    return df


def cache_status() -> dict[str, dict[str, Any]]:
    """Inventory of what's currently cached. Surfaced by the cockpit Data panel
    and the ``autosignalx status`` CLI."""
    info: dict[str, dict[str, Any]] = {}
    for name in ("ohlcv", "macro"):
        path = _path(name)
        if path.exists():
            df = pd.read_parquet(path)
            info[name] = {
                "exists": True,
                "path": str(path),
                "rows": len(df),
                "columns": list(df.columns),
                "earliest": str(df["timestamp"].min()) if "timestamp" in df else None,
                "latest": str(df["timestamp"].max()) if "timestamp" in df else None,
            }
        else:
            info[name] = {"exists": False, "path": str(path)}
    return info
