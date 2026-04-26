"""Engine-level tests for the backtest layer.

Each test pins a specific invariant of the engine: trade-timing shift,
cost handling, all-cash flat-equity baseline, and weight-alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autosignalx.backtest import engine


def _synthetic_prices(n: int = 21, assets: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=(n, len(assets)))
    px = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(px, index=idx, columns=list(assets))


def test_zero_weights_produces_flat_equity():
    px = _synthetic_prices()
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    out = engine.run_engine(w, px, cost_bps=5.0)
    assert (out["equity"] == 1.0).all()
    assert out["returns"].sum() == 0.0


def test_full_long_single_asset_matches_price_ratio():
    """100% A from t0 onwards => equity at end = price_A[end] / price_A[start]."""
    px = _synthetic_prices()
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w["A"] = 1.0
    out = engine.run_engine(w, px, cost_bps=0.0)
    expected_final = px["A"].iloc[-1] / px["A"].iloc[0]
    # First-bar return is 0 because shift(1) makes held_weights 0 on bar 0.
    # So compounded return matches price ratio exactly under no costs.
    assert out["equity"].iloc[-1] == pytest.approx(expected_final, rel=1e-9)


def test_trade_timing_no_lookahead():
    """Weight set at close(t) must not earn the bar(t) return.

    Construct a price series where bar 0 has a +10% jump and bar 1 is flat.
    Set w=1.0 on bar 0 only. If the engine respected look-ahead, the bar-0
    return would be earned. Under the correct shift, bar 0 return is 0.
    """
    idx = pd.bdate_range("2021-01-04", periods=3)
    px = pd.DataFrame({"A": [100.0, 110.0, 110.0]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 0.0, 0.0]}, index=idx)
    out = engine.run_engine(w, px, cost_bps=0.0)
    # Bar 0: held_weight = 0 (shifted from prior absent state) -> return 0
    # Bar 1: held_weight = 1 (last close's weight) -> earn (110-100)/100 = 0
    #        wait: ret bar 1 = px[1]/px[0]-1 = 0.10
    # Bar 2: held_weight = 0 (we set w[1]=0) -> return 0
    assert out["returns"].iloc[0] == pytest.approx(0.0)
    assert out["returns"].iloc[1] == pytest.approx(0.10, rel=1e-9)
    assert out["returns"].iloc[2] == pytest.approx(0.0)


def test_cost_charged_on_turnover():
    """Going from 0% to 100% A on bar 0 charges 100 bps cost at 100 bps/unit."""
    idx = pd.bdate_range("2021-01-04", periods=2)
    px = pd.DataFrame({"A": [100.0, 100.0]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1.0]}, index=idx)
    out = engine.run_engine(w, px, cost_bps=100.0)  # 100 bps = 1%
    assert out["turnover"].iloc[0] == pytest.approx(1.0)
    assert out["cost"].iloc[0] == pytest.approx(0.01)
    # No further trades on bar 1 -> no additional cost
    assert out["turnover"].iloc[1] == pytest.approx(0.0)


def test_engine_rejects_non_overlapping_assets():
    px = _synthetic_prices(assets=("A",))
    w = pd.DataFrame(1.0, index=px.index, columns=["B"])
    with pytest.raises(ValueError, match="no overlapping assets"):
        engine.run_engine(w, px)


def test_assemble_trades_is_sparse():
    idx = pd.bdate_range("2021-01-04", periods=4)
    w = pd.DataFrame(
        {"A": [1.0, 1.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0, 1.0]}, index=idx
    )
    trades = engine.assemble_trades(w, cost_bps=10.0)
    # Trades: bar 0 (A: 0->1), bar 2 (A: 1->0), bar 3 (B: 0->1) = 3 rows
    assert len(trades) == 3
    assert {"timestamp", "asset", "dweight", "cost"} <= set(trades.columns)
    assert (trades["cost"] >= 0).all()
