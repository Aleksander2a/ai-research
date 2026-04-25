"""Build market features, train encoder + clusterers, save regime labels.

The orchestration sits between the data layer (cached parquet) and the
eval layer (consumes regime labels for stratified metrics)."""

from __future__ import annotations

import pandas as pd

from autosignalx.config import settings
from autosignalx.data import loader
from autosignalx.regime import cluster, encoder

REGIMES_DIR = settings.reports_dir / "regimes"


def build_market_features() -> pd.DataFrame:
    """Compose a market-level feature matrix used by the encoder and HMM.

    SPY and QQQ daily returns (proxy for market direction / dispersion)
    plus the four macro signals. Forward-filled and dropna for joint
    coverage; columns standardized to zero mean, unit variance later."""
    returns = loader.load_returns_wide()
    macro = loader.load_macro_wide()

    market_cols = []
    for sym in ("SPY", "QQQ"):
        if sym in returns.columns:
            market_cols.append(returns[sym].rename(f"{sym.lower()}_returns"))

    return pd.concat(market_cols + [macro], axis=1).ffill().dropna()


def fit_and_save(
    n_regimes: int = 4,
    embedding_dim: int = 16,
    window_days: int = 60,
    epochs: int = 25,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Train encoder, cluster (KMeans + HMM), and save labels to ``reports/regimes/``."""
    REGIMES_DIR.mkdir(parents=True, exist_ok=True)
    features_df = build_market_features()
    raw = features_df.to_numpy(dtype="float64")
    standardized = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-8)

    enc, embeddings = encoder.train_encoder(
        standardized.astype("float32"),
        embedding_dim=embedding_dim,
        window_days=window_days,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    km_labels, _ = cluster.kmeans_regimes(embeddings, n_regimes=n_regimes, seed=seed)
    aligned_dates = features_df.index[window_days - 1 :]
    km_df = pd.DataFrame(
        {
            "timestamp": aligned_dates,
            "regime_id": km_labels.astype(int),
            "method": "kmeans_contrastive",
        }
    )
    km_df.to_parquet(REGIMES_DIR / "kmeans.parquet", index=False)

    hmm_labels = cluster.hmm_regimes(standardized, n_regimes=n_regimes, seed=seed)
    hmm_df = pd.DataFrame(
        {
            "timestamp": features_df.index,
            "regime_id": hmm_labels.astype(int),
            "method": "hmm_gaussian",
        }
    )
    hmm_df.to_parquet(REGIMES_DIR / "hmm.parquet", index=False)

    embed_df = (
        pd.DataFrame(embeddings, index=aligned_dates)
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )
    embed_df.to_parquet(REGIMES_DIR / "embeddings.parquet", index=False)

    return {"kmeans": km_df, "hmm": hmm_df, "embeddings": embed_df}


def load_regime_labels(method: str = "kmeans_contrastive") -> pd.DataFrame:
    """Load regime labels for a method.

    Returns DataFrame with ``(timestamp, regime_id, method)``. Raises
    ``FileNotFoundError`` if the labels haven't been fit yet."""
    if method.startswith("kmeans"):
        path = REGIMES_DIR / "kmeans.parquet"
    elif method.startswith("hmm"):
        path = REGIMES_DIR / "hmm.parquet"
    else:
        raise ValueError(f"Unknown regime method: {method!r}")
    if not path.exists():
        raise FileNotFoundError(
            f"Regime labels not found: {path}. Run `autosignalx regime fit` first."
        )
    return pd.read_parquet(path)


def add_regime_to_forecasts(
    forecasts: pd.DataFrame,
    method: str = "kmeans_contrastive",
) -> pd.DataFrame:
    """Join regime labels to a forecasts DataFrame on ``forecast_origin``."""
    rl = load_regime_labels(method)
    rl = rl[["timestamp", "regime_id"]].rename(columns={"timestamp": "forecast_origin"})
    return forecasts.merge(rl, on="forecast_origin", how="left")
