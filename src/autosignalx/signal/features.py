"""Feature engineering for the signal layer.

Builds per-(asset, timestamp) features and a binary direction target
for `target_direction` ahead-of-horizon prediction. Features include
classical technical indicators and lagged macro signals."""

from __future__ import annotations

import pandas as pd


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI on a price series. Returns values in [0, 100]."""
    delta = prices.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_macd_signal(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """MACD signal line (EMA of MACD line)."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return macd_line.ewm(span=signal, adjust=False).mean()


def build_features_target(
    asset_ohlcv: pd.DataFrame,
    macro_wide: pd.DataFrame,
    horizon_days: int = 21,
) -> pd.DataFrame:
    """Build features and binary direction target for one asset.

    Returns a DataFrame with ``[timestamp, feature_*, target_direction]``
    rows, dropping any rows with NaN in either features or target.

    Target: ``1`` if ``adj_close[t + horizon_days] > adj_close[t]`` else ``0``.

    Macro signals are joined on timestamp (forward-filled) and lagged by
    1 / 5 days; current macro level and 5-day macro change are included
    as features."""
    df = asset_ohlcv.sort_values("timestamp").copy()
    df["returns_1d"] = df["adj_close"].pct_change()

    df["rolling_mean_5"] = df["returns_1d"].rolling(5).mean()
    df["rolling_mean_20"] = df["returns_1d"].rolling(20).mean()
    df["rolling_std_5"] = df["returns_1d"].rolling(5).std()
    df["rolling_std_20"] = df["returns_1d"].rolling(20).std()
    df["momentum_10"] = df["adj_close"].pct_change(10)
    df["momentum_60"] = df["adj_close"].pct_change(60)
    df["rsi_14"] = compute_rsi(df["adj_close"], window=14)
    df["macd_signal"] = compute_macd_signal(df["adj_close"])

    if not macro_wide.empty:
        macro_aligned = macro_wide.reindex(df["timestamp"].values).ffill().bfill()
        for col in macro_wide.columns:
            df[f"macro_{col}_level"] = macro_aligned[col].to_numpy()
            df[f"macro_{col}_chg5"] = (
                macro_aligned[col].pct_change(5).to_numpy()
            )

    df["future_close"] = df["adj_close"].shift(-horizon_days)
    df["target_direction"] = (df["future_close"] > df["adj_close"]).astype(int)
    return df.dropna(
        subset=["target_direction", "rolling_std_20", "momentum_60", "rsi_14", "macd_signal"]
    )


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of feature columns (excluding ids/target/aux)."""
    drop = {"timestamp", "asset", "target_direction", "future_close", "returns_1d", "open",
            "high", "low", "close", "adj_close", "volume"}
    return [c for c in df.columns if c not in drop]
