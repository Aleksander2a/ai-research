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
    weight_at_origin,
) -> pd.DataFrame:
    """Shared scaffold for origin-based rebalancing strategies.

    Args:
        signals_wide: DataFrame indexed by ``forecast_origin``, columns
            are assets, values are predicted returns over the horizon.
        prices: prices panel; weights are emitted on the prices calendar.
        weight_at_origin: callable ``(ranked, origin) -> Series`` that
            returns weights (asset -> weight) for the given origin. May
            return an empty Series to signal "go to cash."

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
        chosen = weight_at_origin(ranked, origin)
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

        def pick(ranked: pd.Series, _origin) -> pd.Series:
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

        def pick(ranked: pd.Series, _origin) -> pd.Series:
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


class RegimeGated(BaseStrategy):
    """TopKLong, but only when the current regime has at least one
    promoted finding. Outside those regimes, hold cash. The set of
    "good" regimes is read from ``reports/agent/findings.jsonl`` and is
    therefore frozen at agent-promotion time -- a parameter of the
    discovery pipeline, not of the backtest."""

    def __init__(self, k: int = 3, method: str = "chronos2_multivariate"):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.method = method

    @property
    def name(self) -> str:
        return f"RegimeGated(k={self.k})"

    def weights(
        self, prices: pd.DataFrame, context: dict | None = None
    ) -> pd.DataFrame:
        ctx = context or {}
        sigs = ctx.get("forecast_signals")
        regimes = ctx.get("regimes")
        findings = ctx.get("findings", [])
        if sigs is None or sigs.empty:
            raise ValueError(f"{self.name} requires 'forecast_signals' in context")
        if regimes is None:
            raise ValueError(f"{self.name} requires 'regimes' in context")

        good_regimes = {
            f.get("filters", {}).get("regime_id")
            for f in findings
            if f.get("filters", {}).get("regime_id") is not None
        }

        wide = (
            sigs.pivot(index="forecast_origin", columns="asset", values="predicted_return")
            .reindex(columns=prices.columns)
            .sort_index()
        )
        regime_at = regimes.sort_index()

        def pick(ranked: pd.Series, origin) -> pd.Series:
            if not good_regimes:
                return pd.Series(dtype=float)
            past = regime_at.loc[regime_at.index <= origin]
            if past.empty:
                return pd.Series(dtype=float)
            current = past.iloc[-1]
            if current not in good_regimes:
                return pd.Series(dtype=float)
            top = ranked.nlargest(min(self.k, len(ranked))).index
            return pd.Series(1.0 / self.k, index=top)

        return _build_rebalance_weights(wide, prices, pick)


class FindingDriven(BaseStrategy):
    """Trade only the (asset, regime) combinations promoted in
    ``findings.jsonl``. At each rebalance, for each asset whose current
    regime matches a finding's regime_id and whose predicted return is
    positive, take a long position weighted by the finding's
    ``skill_vs_baseline``. Weights are renormalised so gross <= 1.0."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "FindingDriven"

    def weights(
        self, prices: pd.DataFrame, context: dict | None = None
    ) -> pd.DataFrame:
        ctx = context or {}
        sigs = ctx.get("forecast_signals")
        regimes = ctx.get("regimes")
        findings = ctx.get("findings", [])
        if sigs is None or sigs.empty:
            raise ValueError(f"{self.name} requires 'forecast_signals' in context")
        if regimes is None:
            raise ValueError(f"{self.name} requires 'regimes' in context")
        if not findings:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        finding_table: dict[tuple[str, int], float] = {}
        for f in findings:
            filt = f.get("filters", {}) or {}
            asset = filt.get("asset")
            regime = filt.get("regime_id")
            if asset is None or regime is None:
                continue
            skill = float(f.get("evidence", {}).get("skill_vs_baseline", 0.0))
            key = (asset, int(regime))
            finding_table[key] = max(finding_table.get(key, 0.0), skill)
        if not finding_table:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        wide = (
            sigs.pivot(index="forecast_origin", columns="asset", values="predicted_return")
            .reindex(columns=prices.columns)
            .sort_index()
        )
        regime_at = regimes.sort_index()

        def pick(ranked: pd.Series, origin) -> pd.Series:
            past = regime_at.loc[regime_at.index <= origin]
            if past.empty:
                return pd.Series(dtype=float)
            current = int(past.iloc[-1])
            raw: dict[str, float] = {}
            for asset, predicted in ranked.items():
                skill = finding_table.get((asset, current), 0.0)
                if skill > 0.0 and predicted > 0.0:
                    raw[asset] = skill
            if not raw:
                return pd.Series(dtype=float)
            total = sum(raw.values())
            return pd.Series({a: w / total for a, w in raw.items()})

        return _build_rebalance_weights(wide, prices, pick)


STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "BuyAndHoldSPY": BuyAndHoldSPY,
    "EqualWeightUniverse": EqualWeightUniverse,
    "TopKLong": TopKLong,
    "LongShortKK": LongShortKK,
    "RegimeGated": RegimeGated,
    "FindingDriven": FindingDriven,
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
