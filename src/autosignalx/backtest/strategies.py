"""Strategy classes consumed by the backtest runner.

Each strategy implements ``weights(prices, **context) -> DataFrame`` where
the returned DataFrame is indexed by timestamp and has one column per
asset in the universe. The runner aligns the result to the prices
calendar before passing it to the engine.

Phase 1.1 ships the two passive strategies (``BuyAndHoldSPY``,
``EqualWeightUniverse``); signal-driven strategies arrive in Phase 1.2+.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """ABC for backtest strategies.

    Subclasses set a class-level ``name`` and implement ``weights``.
    Strategies that need forecasts/regimes/findings pull them via the
    ``context`` mapping the runner threads in.
    """

    name: str = "base"

    @abstractmethod
    def weights(
        self, prices: pd.DataFrame, context: dict | None = None
    ) -> pd.DataFrame:
        """Return a (timestamp x asset) weight matrix over the prices index."""


class BuyAndHoldSPY(BaseStrategy):
    name = "BuyAndHoldSPY"

    def weights(
        self, prices: pd.DataFrame, context: dict | None = None  # noqa: ARG002
    ) -> pd.DataFrame:
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if "SPY" not in w.columns:
            raise ValueError("BuyAndHoldSPY requires SPY in the universe")
        w["SPY"] = 1.0
        return w


class EqualWeightUniverse(BaseStrategy):
    name = "EqualWeightUniverse"

    def weights(
        self, prices: pd.DataFrame, context: dict | None = None  # noqa: ARG002
    ) -> pd.DataFrame:
        n = len(prices.columns)
        if n == 0:
            raise ValueError("EqualWeightUniverse requires a non-empty universe")
        return pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)


STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    BuyAndHoldSPY.name: BuyAndHoldSPY,
    EqualWeightUniverse.name: EqualWeightUniverse,
}


def build_strategy(name: str) -> BaseStrategy:
    """Look up a strategy by registered name and instantiate it."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown strategy {name!r}. Known: {sorted(STRATEGY_REGISTRY)}"
        )
    return cls()
