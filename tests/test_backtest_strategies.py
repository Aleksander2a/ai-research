"""Strategy-class tests."""

from __future__ import annotations

import pandas as pd
import pytest

from autosignalx.backtest import strategies


def _prices(assets: tuple[str, ...], n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(100.0, index=idx, columns=list(assets))


def _signals_at(origins: list[str], values: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for i, origin in enumerate(origins):
        for asset, vals in values.items():
            rows.append({"forecast_origin": pd.Timestamp(origin),
                         "asset": asset, "predicted_return": vals[i]})
    return pd.DataFrame(rows)


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


def test_registry_contains_all_strategies():
    for name in ("BuyAndHoldSPY", "EqualWeightUniverse", "TopKLong", "LongShortKK"):
        assert name in strategies.STRATEGY_REGISTRY


def test_parse_spec_handles_kwargs():
    s = strategies.parse_strategy_spec("TopKLong:k=5")
    assert isinstance(s, strategies.TopKLong)
    assert s.k == 5
    assert s.name == "TopKLong(k=5)"


def test_parse_spec_default_kwargs():
    s = strategies.parse_strategy_spec("LongShortKK")
    assert s.k == 2  # default


def test_top_k_long_picks_top_k():
    px = _prices(("A", "B", "C", "D"), n=15)
    sigs = _signals_at(
        ["2021-01-04", "2021-01-15"],
        {"A": [0.10, -0.05], "B": [0.05, 0.20], "C": [-0.02, 0.10], "D": [-0.10, -0.20]},
    )
    s = strategies.parse_strategy_spec("TopKLong:k=2")
    w = s.weights(px, context={"forecast_signals": sigs})
    # On 2021-01-04: top 2 are A (0.10) and B (0.05) -> 0.5 each
    first = w.iloc[0]
    assert first["A"] == pytest.approx(0.5)
    assert first["B"] == pytest.approx(0.5)
    assert first["C"] == 0.0
    assert first["D"] == 0.0
    # After 2021-01-15: top 2 are B (0.20) and C (0.10)
    last = w.iloc[-1]
    assert last["B"] == pytest.approx(0.5)
    assert last["C"] == pytest.approx(0.5)
    assert last["A"] == 0.0
    assert last["D"] == 0.0


def test_top_k_long_holds_between_origins():
    px = _prices(("A", "B"), n=10)
    sigs = _signals_at(["2021-01-04"], {"A": [0.1], "B": [-0.1]})
    s = strategies.parse_strategy_spec("TopKLong:k=1")
    w = s.weights(px, context={"forecast_signals": sigs})
    assert (w["A"] == 1.0).all()
    assert (w["B"] == 0.0).all()


def test_long_short_is_dollar_neutral():
    px = _prices(("A", "B", "C", "D"), n=10)
    sigs = _signals_at(
        ["2021-01-04"], {"A": [0.20], "B": [0.10], "C": [-0.10], "D": [-0.20]}
    )
    s = strategies.parse_strategy_spec("LongShortKK:k=2")
    w = s.weights(px, context={"forecast_signals": sigs})
    row = w.iloc[0]
    assert row.sum() == pytest.approx(0.0)  # dollar neutral
    assert row.abs().sum() == pytest.approx(1.0)  # gross 100%
    assert row["A"] > 0 and row["B"] > 0
    assert row["C"] < 0 and row["D"] < 0


def test_signal_strategy_requires_context():
    px = _prices(("A", "B"))
    s = strategies.parse_strategy_spec("TopKLong:k=1")
    with pytest.raises(ValueError, match="forecast_signals"):
        s.weights(px, context=None)


def _regimes(idx: pd.DatetimeIndex, regime_id: int) -> pd.Series:
    return pd.Series(regime_id, index=idx, name="regime_id")


def test_regime_gated_holds_cash_when_no_findings():
    px = _prices(("A", "B"), n=10)
    sigs = _signals_at(["2021-01-04"], {"A": [0.1], "B": [-0.1]})
    regimes = _regimes(px.index, regime_id=2)
    s = strategies.parse_strategy_spec("RegimeGated:k=1")
    w = s.weights(px, context={
        "forecast_signals": sigs, "regimes": regimes, "findings": []
    })
    assert (w == 0.0).all().all()


def test_regime_gated_trades_only_in_good_regime():
    px = _prices(("A", "B"), n=15)
    # Regime 3 (good) for the first 4 bars; regime 0 (bad) thereafter.
    regimes = pd.Series([3] * 4 + [0] * 11, index=px.index, name="regime_id")
    # Two rebalances: origin 0 lands inside regime 3, origin at index 5
    # lands inside regime 0.
    sigs = _signals_at(
        [str(px.index[0].date()), str(px.index[5].date())],
        {"A": [0.1, 0.1], "B": [-0.1, -0.1]},
    )
    findings = [{"filters": {"asset": "A", "regime_id": 3},
                 "evidence": {"skill_vs_baseline": 0.05}}]
    s = strategies.parse_strategy_spec("RegimeGated:k=1")
    w = s.weights(px, context={
        "forecast_signals": sigs, "regimes": regimes, "findings": findings
    })
    # First rebalance (regime 3): A long.
    assert w.iloc[0]["A"] == pytest.approx(1.0)
    # Second rebalance (regime 0, not in good_regimes): cash.
    assert w.iloc[-1]["A"] == 0.0
    assert w.iloc[-1]["B"] == 0.0


def test_finding_driven_returns_cash_when_no_findings():
    px = _prices(("A", "B"), n=10)
    sigs = _signals_at(["2021-01-04"], {"A": [0.1], "B": [0.05]})
    regimes = _regimes(px.index, regime_id=3)
    s = strategies.parse_strategy_spec("FindingDriven")
    w = s.weights(px, context={
        "forecast_signals": sigs, "regimes": regimes, "findings": []
    })
    assert (w == 0.0).all().all()


def test_finding_driven_only_trades_promoted_pairs():
    px = _prices(("A", "B", "C"), n=10)
    # All assets predict positive returns.
    sigs = _signals_at(
        ["2021-01-04"], {"A": [0.05], "B": [0.10], "C": [0.20]}
    )
    regimes = _regimes(px.index, regime_id=3)
    # Only B is promoted in regime 3.
    findings = [{"filters": {"asset": "B", "regime_id": 3},
                 "evidence": {"skill_vs_baseline": 0.05}}]
    s = strategies.parse_strategy_spec("FindingDriven")
    w = s.weights(px, context={
        "forecast_signals": sigs, "regimes": regimes, "findings": findings
    })
    assert w.iloc[0]["B"] == pytest.approx(1.0)
    assert w.iloc[0]["A"] == 0.0
    assert w.iloc[0]["C"] == 0.0


def test_finding_driven_skips_negative_predicted_return():
    """Finding promoted but predicted return negative -> stay flat."""
    px = _prices(("A",), n=10)
    sigs = _signals_at(["2021-01-04"], {"A": [-0.05]})
    regimes = _regimes(px.index, regime_id=3)
    findings = [{"filters": {"asset": "A", "regime_id": 3},
                 "evidence": {"skill_vs_baseline": 0.05}}]
    s = strategies.parse_strategy_spec("FindingDriven")
    w = s.weights(px, context={
        "forecast_signals": sigs, "regimes": regimes, "findings": findings
    })
    assert (w == 0.0).all().all()
