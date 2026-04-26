"""Independent reference-implementation parity check for the engine.

The pandas-based ``engine.run_engine`` is the production code. This test
re-implements the same trade-timing and cost logic in pure NumPy with a
different control flow (explicit per-bar loop instead of vectorised
shift) and asserts equality on a non-trivial random fixture. If the two
disagree, either the production engine or the reference has a bug; both
must be inspected before changing the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autosignalx.backtest import engine


def _reference_run(
    weights: np.ndarray, prices: np.ndarray, cost_bps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar loop, no pandas, no vectorised shift.

    Returns (net_returns, equity, turnover) of length T.
    """
    n_bars, n_assets = weights.shape
    net_ret = np.zeros(n_bars)
    turn = np.zeros(n_bars)
    eq = np.zeros(n_bars)
    cum = 1.0
    cost_rate = cost_bps / 10_000.0
    prev_w = np.zeros(n_assets)
    for t in range(n_bars):
        w_t = weights[t].copy()
        # Asset return at t: pct_change of prices.
        asset_ret = (
            np.zeros(n_assets) if t == 0 else prices[t] / prices[t - 1] - 1.0
        )
        # Held weight = previous bar's chosen weight (one-bar shift).
        held = prev_w
        gross = float((held * asset_ret).sum())
        delta = w_t - prev_w
        turnover = float(np.abs(delta).sum())
        cost = turnover * cost_rate
        net = gross - cost
        cum *= 1.0 + net
        net_ret[t] = net
        turn[t] = turnover
        eq[t] = cum
        prev_w = w_t
    return net_ret, eq, turn


def test_engine_matches_numpy_reference_on_random_fixture():
    rng = np.random.default_rng(42)
    n_bars, assets = 60, ("A", "B", "C", "D")
    idx = pd.bdate_range("2021-01-04", periods=n_bars)

    px = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, size=(n_bars, len(assets))),
                                   axis=0))
    px_df = pd.DataFrame(px, index=idx, columns=list(assets))

    # Random tradable weights: some bars rebalance, others hold.
    raw = rng.uniform(-1.0, 1.0, size=(n_bars, len(assets)))
    raw = raw / np.maximum(np.abs(raw).sum(axis=1, keepdims=True), 1.0)
    # Force most bars to hold previous weights -- only rebalance on a sparse mask.
    rebalance = rng.random(n_bars) < 0.2
    rebalance[0] = True
    weights = np.zeros_like(raw)
    cur = np.zeros(len(assets))
    for t in range(n_bars):
        if rebalance[t]:
            cur = raw[t]
        weights[t] = cur
    w_df = pd.DataFrame(weights, index=idx, columns=list(assets))

    cost_bps = 7.5
    out = engine.run_engine(w_df, px_df, cost_bps=cost_bps)
    ref_net, ref_eq, ref_turn = _reference_run(weights, px, cost_bps=cost_bps)

    np.testing.assert_allclose(out["returns"].values, ref_net, atol=1e-12)
    np.testing.assert_allclose(out["equity"].values, ref_eq, atol=1e-12)
    np.testing.assert_allclose(out["turnover"].values, ref_turn, atol=1e-12)


def test_parity_holds_with_zero_costs():
    rng = np.random.default_rng(7)
    n_bars, assets = 30, ("A", "B")
    idx = pd.bdate_range("2021-01-04", periods=n_bars)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n_bars, 2)), axis=0))
    weights = rng.uniform(0, 1, size=(n_bars, 2))
    weights /= weights.sum(axis=1, keepdims=True)

    px_df = pd.DataFrame(px, index=idx, columns=list(assets))
    w_df = pd.DataFrame(weights, index=idx, columns=list(assets))

    out = engine.run_engine(w_df, px_df, cost_bps=0.0)
    ref_net, _, _ = _reference_run(weights, px, cost_bps=0.0)
    np.testing.assert_allclose(out["returns"].values, ref_net, atol=1e-12)


def test_parity_zero_costs_and_buy_and_hold_matches_price_path():
    """Sanity: full long single asset with no costs => equity == price/price[0]."""
    rng = np.random.default_rng(0)
    n_bars = 40
    idx = pd.bdate_range("2021-01-04", periods=n_bars)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n_bars)))
    px_df = pd.DataFrame({"A": px}, index=idx)
    w_df = pd.DataFrame({"A": np.ones(n_bars)}, index=idx)

    out = engine.run_engine(w_df, px_df, cost_bps=0.0)
    expected = px / px[0]
    # First bar = 1.0 (no return earned on bar 0 due to one-bar shift).
    assert out["equity"].iloc[0] == pytest.approx(1.0, abs=1e-12)
    assert out["equity"].iloc[-1] == pytest.approx(expected[-1], abs=1e-9)
