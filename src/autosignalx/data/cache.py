"""Parquet read/write for cached data.

Default location: ``data/cache/``. Per-study runs override the cache
root via the optional ``cache_root`` argument (used by the Phase 2
``Study`` layer); when omitted, the default project-wide cache is used.

The cache is the persistent contract between fetch (writer) and loader
(reader). Schema enforcement happens at write time via ``assert_*_schema``
so corrupt data never reaches the eval harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from autosignalx.config import settings
from autosignalx.data.schema import assert_macro_schema, assert_ohlcv_schema


def _default_cache_root() -> Path:
    return settings.data_dir / "cache"


def _resolve_root(cache_root: Path | None) -> Path:
    root = cache_root or _default_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _path(name: str, cache_root: Path | None = None) -> Path:
    return _resolve_root(cache_root) / f"{name}.parquet"


def write_ohlcv(df: pd.DataFrame, cache_root: Path | None = None) -> Path:
    """Persist an OHLCV frame after validating its schema."""
    assert_ohlcv_schema(df)
    path = _path("ohlcv", cache_root)
    df.to_parquet(path, index=False)
    return path


def write_macro(df: pd.DataFrame, cache_root: Path | None = None) -> Path:
    """Persist a macro frame after validating its schema."""
    assert_macro_schema(df)
    path = _path("macro", cache_root)
    df.to_parquet(path, index=False)
    return path


def read_ohlcv(cache_root: Path | None = None) -> pd.DataFrame:
    """Load the cached OHLCV frame and validate it.

    Raises ``FileNotFoundError`` with a helpful hint if the cache is empty."""
    path = _path("ohlcv", cache_root)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached OHLCV at {path}. Run `autosignalx data fetch` first."
        )
    df = pd.read_parquet(path)
    assert_ohlcv_schema(df)
    return df


def read_macro(cache_root: Path | None = None) -> pd.DataFrame:
    """Load the cached macro frame and validate it."""
    path = _path("macro", cache_root)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached macro at {path}. Run `autosignalx data fetch` first."
        )
    df = pd.read_parquet(path)
    assert_macro_schema(df)
    return df


def cache_status(cache_root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Inventory of what's currently cached. Surfaced by the cockpit Data panel
    and the ``autosignalx status`` CLI."""
    info: dict[str, dict[str, Any]] = {}
    for name in ("ohlcv", "macro"):
        path = _path(name, cache_root)
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
