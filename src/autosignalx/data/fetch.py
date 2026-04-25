"""yfinance pulls for ETFs and macro signals.

Wide-format yfinance output is normalized into long-format DataFrames
that match the contract in ``data.schema``. Defensive against single-vs-
multi-ticker MultiIndex column quirks across yfinance versions."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import yfinance as yf


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns a MultiIndex columns even for a single
    ticker; normalize to flat single-level columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.droplevel(1)
    return df


def _to_long_ohlcv(raw: pd.DataFrame, asset: str) -> pd.DataFrame:
    df = _flatten_columns(raw).reset_index()
    ts_col = "Date" if "Date" in df.columns else "Datetime"
    df = df.rename(
        columns={
            ts_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["asset"] = asset
    df["returns"] = df["adj_close"].pct_change()
    df = df[
        [
            "timestamp",
            "asset",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "returns",
        ]
    ]
    return df.dropna(subset=["close"])


def fetch_ohlcv(assets: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Pull OHLCV for each asset and return a long-format DataFrame
    matching the OHLCV schema. Raises if any requested asset is empty."""
    frames: list[pd.DataFrame] = []
    for asset in assets:
        raw = yf.download(
            asset,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            actions=False,
        )
        if raw.empty:
            raise RuntimeError(
                f"yfinance returned no data for {asset!r} ({start}..{end})"
            )
        frames.append(_to_long_ohlcv(raw, asset))

    out = pd.concat(frames, ignore_index=True)
    out["asset"] = out["asset"].astype("string")
    return out.sort_values(["asset", "timestamp"]).reset_index(drop=True)


def _to_long_macro(raw: pd.DataFrame, signal: str) -> pd.DataFrame:
    df = _flatten_columns(raw).reset_index()
    ts_col = "Date" if "Date" in df.columns else "Datetime"
    value_col = "Close" if "Close" in df.columns else df.columns[-1]
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col]).dt.tz_localize(None),
            "signal": signal,
            "value": df[value_col].astype(float),
        }
    )
    return out.dropna(subset=["value"])


def fetch_macro(signals: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Pull macro signals (e.g., ``^TNX``, ``^VIX``). Tolerates individual
    empty results -- a single missing signal is logged into the warning,
    but the batch only fails if *every* request returned empty."""
    frames: list[pd.DataFrame] = []
    for signal in signals:
        raw = yf.download(
            signal,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            actions=False,
        )
        if raw.empty:
            continue
        frames.append(_to_long_macro(raw, signal))
    if not frames:
        raise RuntimeError("yfinance returned no macro data for any requested signal")

    out = pd.concat(frames, ignore_index=True)
    out["signal"] = out["signal"].astype("string")
    return out.sort_values(["signal", "timestamp"]).reset_index(drop=True)


def fetch_all(
    assets: Iterable[str],
    macro: Iterable[str],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: pull both OHLCV and macro in one call."""
    return fetch_ohlcv(assets, start, end), fetch_macro(macro, start, end)
