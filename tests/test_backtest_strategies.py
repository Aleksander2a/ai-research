"""Strategy-class tests."""

from __future__ import annotations

import pandas as pd
import pytest

from autosignalx.backtest import strategies


def _prices(assets: tuple[str, ...]) -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-04", periods=5)
    return pd.DataFrame(100.0, index=idx, columns=list(assets))


def test_buy_and_hold_spy_full_weight_in_spy():
    px = _prices(("SPY", "QQQ", "TLT"))
    s = strategies.build_strategy("BuyAndHoldSPY")
    w = s.weights(px)
    assert (w["SPY"] == 1.0).all()
    assert (w["QQQ"] == 0.0).all()
    assert (w["TLT"] == 0.0).all()


def test_buy_and_hold_spy_requires_spy_in_universe():
    px = _prices(("QQQ", "TLT"))
    s = strategies.build_strategy("BuyAndHoldSPY")
    with pytest.raises(ValueError, match="requires SPY"):
        s.weights(px)


def test_equal_weight_universe_sums_to_one():
    px = _prices(("SPY", "QQQ", "TLT", "GLD"))
    s = strategies.build_strategy("EqualWeightUniverse")
    w = s.weights(px)
    row_sums = w.sum(axis=1)
    assert (row_sums.round(9) == 1.0).all()
    assert (w == 0.25).all().all()


def test_unknown_strategy_raises():
    with pytest.raises(KeyError, match="Unknown strategy"):
        strategies.build_strategy("DoesNotExist")


def test_registry_contains_phase1_1_strategies():
    assert "BuyAndHoldSPY" in strategies.STRATEGY_REGISTRY
    assert "EqualWeightUniverse" in strategies.STRATEGY_REGISTRY
