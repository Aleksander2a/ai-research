"""Strategy classes consumed by the backtest runner.

Each strategy implements ``weights(prices, context) -> DataFrame`` where
the returned DataFrame is indexed by timestamp and has one column per
asset in the universe. Strategies that need forecasts/regimes/findings
pull them via the ``context`` mapping the runner threads in.

Strategy specs accepted by ``parse_strategy_spec`` follow the form
``ClassName`` for default kwargs or ``ClassName:k=3,method=foo`` for
overrides; the latter is what the CLI's comma-separated list parses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """ABC for backtest strategies."""

    @property
    def name(self) -> str:
        """Display name. Subclasses override for parameterised strategies."""
        return self.__class__.__name__

    @abstractmethod
    def weights(
        self, prices: pd.DataFrame, context: dict | None = None
    ) -> pd.DataFrame:
        """Return a (timestamp x asset) weight matrix over the prices index."""


# ---------------------------------------------------------------------------
# Passive baselines (Phase 1.1)
# ---------------------------------------------------------------------------


class BuyAndHoldSPY(BaseStrategy):
    def weights(
        self, prices: pd.DataFrame, context: dict | None = None  # noqa: ARG002
    ) -> pd.DataFrame:
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if "SPY" not in w.columns:
            raise ValueError("BuyAndHoldSPY requires SPY in the universe")
        w["SPY"] = 1.0
        return w


class EqualWeightUniverse(BaseStrategy):
    def weights(
        self, prices: pd.DataFrame, context: dict | None = None  # noqa: ARG002
    ) -> pd.DataFrame:
        n = len(prices.columns)
        if n == 0:
            raise ValueError("EqualWeightUniverse requires a non-empty universe")
        return pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)


# ---------------------------------------------------------------------------
# Signal-driven strategies (Phase 1.2)
# ---------------------------------------------------------------------------


def _build_rebalance_weights(
    signals_wide: pd.DataFrame,
    prices: pd.DataFrame,
    weight_at_origin: callable,
) -> pd.DataFrame:
    """Shared scaffold for origin-based rebalancing strategies.

    Args:
        signals_wide: DataFrame indexed by ``forecast_origin``, columns
            are assets, values are predicted returns over the horizon.
        prices: prices panel; weights are emitted on the prices calendar.
        weight_at_origin: callable taking a row Series (asset -> signal)
            and returning a Series of weights (asset -> weight) for that
            origin. May return a Series with only the active assets.

    The output frame is initialised to NaN, set on each rebalance date
    (the first prices bar at or after the origin), then forward-filled
    so positions persist between rebalances. Final NaNs are replaced
    with 0.0 (cash) before emit.
    """
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for origin, row in signals_wide.iterrows():
        ranked = row.dropna()
        if ranked.empty:
            continue
        future = prices.index[prices.index >= origin]
        if len(future) == 0:
            continue
        target_date = future[0]
        new_w = pd.Series(0.0, index=prices.columns)
        chosen = weight_at_origin(ranked)
        chosen = chosen.reindex(prices.columns).fillna(0.0)
        new_w = new_w.add(chosen, fill_value=0.0)
        weights.loc[target_date] = new_w.values
    return weights.ffill().fillna(0.0)


class TopKLong(BaseStrategy):
    """Cross-sectional long-only: each rebalance, hold equal weights in
    the top-K assets by predicted return; cash otherwise."""

    def __init__(self, k: int = 3, method: str = "chronos2_multivariate"):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.method = method

    @property
    def name(self) -> str:
        return f"TopKLong(k={self.k})"

    def weights(
        self, prices: pd.DataFrame, context: dict | None = None
    ) -> pd.DataFrame:
        sigs = (context or {}).get("forecast_signals")
        if sigs is None or sigs.empty:
            raise ValueError(f"{self.name} requires 'forecast_signals' in context")
        wide = (
            sigs.pivot(index="forecast_origin", columns="asset", values="predicted_return")
            .reindex(columns=prices.columns)
            .sort_index()
        )

        def pick(ranked: pd.Series) -> pd.Series:
            top = ranked.nlargest(min(self.k, len(ranked))).index
            return pd.Series(1.0 / self.k, index=top)

        return _build_rebalance_weights(wide, prices, pick)


class LongShortKK(BaseStrategy):
    """Dollar-neutral cross-section: long top-K (equal weight) and
    short bottom-K (equal weight). Gross exposure ~100%, net 0."""

    def __init__(self, k: int = 2, method: str = "chronos2_multivariate"):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.method = method

    @property
    def name(self) -> str:
        return f"LongShortKK(k={self.k})"

    def weights(
        self, prices: pd.DataFrame, context: dict | None = None
    ) -> pd.DataFrame:
        sigs = (context or {}).get("forecast_signals")
        if sigs is None or sigs.empty:
            raise ValueError(f"{self.name} requires 'forecast_signals' in context")
        wide = (
            sigs.pivot(index="forecast_origin", columns="asset", values="predicted_return")
            .reindex(columns=prices.columns)
            .sort_index()
        )

        def pick(ranked: pd.Series) -> pd.Series:
            n = len(ranked)
            if n < 2 * self.k:
                return pd.Series(dtype=float)
            top = ranked.nlargest(self.k).index
            bot = ranked.nsmallest(self.k).index
            longs = pd.Series(0.5 / self.k, index=top)
            shorts = pd.Series(-0.5 / self.k, index=bot)
            return pd.concat([longs, shorts])

        return _build_rebalance_weights(wide, prices, pick)


# ---------------------------------------------------------------------------
# Registry + spec parsing
# ---------------------------------------------------------------------------


STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "BuyAndHoldSPY": BuyAndHoldSPY,
    "EqualWeightUniverse": EqualWeightUniverse,
    "TopKLong": TopKLong,
    "LongShortKK": LongShortKK,
}


def parse_strategy_spec(spec: str) -> BaseStrategy:
    """Parse a CLI-friendly strategy spec into an instance.

    Accepts ``Name`` for default kwargs or ``Name:k=3,method=foo`` for
    overrides. Values are interpreted as int when possible, then float,
    then left as str.
    """
    head, _, tail = spec.partition(":")
    cls = STRATEGY_REGISTRY.get(head)
    if cls is None:
        raise KeyError(f"Unknown strategy {head!r}. Known: {sorted(STRATEGY_REGISTRY)}")
    kwargs: dict = {}
    if tail:
        for part in tail.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(f"bad strategy spec fragment {part!r}")
            k, v = part.split("=", 1)
            kwargs[k.strip()] = _coerce(v.strip())
    return cls(**kwargs)


def _coerce(v: str) -> int | float | str:
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def build_strategy(name: str) -> BaseStrategy:
    """Backwards-compatible builder used by the runner; delegates to
    ``parse_strategy_spec``."""
    return parse_strategy_spec(name)
