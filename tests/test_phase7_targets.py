"""Phase 7 tests: returns-target forecast contract + metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autosignalx.eval import contracts, metrics_returns
from autosignalx.eval import targets as targets_mod
from autosignalx.forecast import returns_baselines


def _price_forecasts() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for k in range(40):
        origin = pd.Timestamp("2024-01-01") + pd.Timedelta(days=k)
        ov = float(100 + rng.normal(0, 1))
        for h in range(1, 6):
            ts = origin + pd.Timedelta(days=h)
            target = ov * float(np.exp(rng.normal(0, 0.01)))
            pred = ov * float(np.exp(rng.normal(0, 0.01)))
            rows.append({
                "asset": "SPY",
                "timestamp": ts,
                "forecast_origin": origin,
                "horizon": h,
                "method": "naive",
                "prediction": pred,
                "origin_value": ov,
                "target": target,
            })
    return pd.DataFrame(rows)


def test_get_target_type_defaults_to_price():
    df = _price_forecasts()
    assert contracts.get_target_type(df) == "price"


def test_get_target_type_reads_column():
    df = _price_forecasts()
    df["target_type"] = "log_return"
    assert contracts.get_target_type(df) == "log_return"


def test_get_target_type_rejects_heterogeneous():
    df = _price_forecasts()
    df["target_type"] = ["price"] * (len(df) // 2) + ["log_return"] * (len(df) - len(df) // 2)
    with pytest.raises(ValueError, match="heterogeneous"):
        contracts.get_target_type(df)


def test_to_log_return_produces_log_units():
    df = _price_forecasts()
    lr = targets_mod.to_log_return(df)
    assert (lr["target_type"] == "log_return").all()
    # log returns should be tiny (~1% std)
    assert lr["target"].abs().mean() < 0.1


def test_to_log_return_rejects_zero_origin():
    df = _price_forecasts()
    df.loc[0, "origin_value"] = 0.0
    lr = targets_mod.to_log_return(df)
    # row with bad origin dropped
    assert len(lr) < len(df)


def test_to_excess_return_handles_missing_rf():
    df = _price_forecasts()
    er = targets_mod.to_excess_return(df, rf_daily=pd.DataFrame())
    assert (er["target_type"] == "excess_return").all()


def test_cross_sectional_rank_produces_unit_interval():
    rng = np.random.default_rng(0)
    rows = []
    for k in range(15):
        origin = pd.Timestamp("2024-01-01") + pd.Timedelta(days=k)
        for asset in ["SPY", "QQQ", "TLT", "IWM"]:
            ov = 100.0
            ts = origin + pd.Timedelta(days=1)
            target = ov * float(np.exp(rng.normal(0, 0.01)))
            pred = ov * float(np.exp(rng.normal(0, 0.01)))
            rows.append({
                "asset": asset,
                "timestamp": ts,
                "forecast_origin": origin,
                "horizon": 1,
                "method": "naive",
                "prediction": pred,
                "origin_value": ov,
                "target": target,
            })
    df = pd.DataFrame(rows)
    ranks = targets_mod.to_cross_sectional_rank(df)
    assert (ranks["target_type"] == "rank").all()
    assert ((ranks["prediction"] >= 0) & (ranks["prediction"] <= 1)).all()
    assert ((ranks["target"] >= 0) & (ranks["target"] <= 1)).all()


def test_zero_return_baseline_predicts_last_price():
    asset_train = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=30), "adj_close": np.arange(100, 130, dtype=float)}
    )
    targets = pd.date_range("2024-02-01", periods=5).tolist()
    out = returns_baselines.zero_return_forecast(asset_train, asset_train["timestamp"].iloc[-1], targets)
    assert (out["prediction"] == 129.0).all()


def test_mean_return_baseline_extrapolates_drift():
    asset_train = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=120), "adj_close": np.exp(np.linspace(np.log(100), np.log(110), 120))}
    )
    targets = pd.date_range("2024-05-01", periods=5).tolist()
    out = returns_baselines.mean_return_forecast(asset_train, asset_train["timestamp"].iloc[-1], targets)
    # Drift is positive so each subsequent prediction is larger
    assert out["prediction"].is_monotonic_increasing


def test_momentum_baseline_runs_without_error():
    asset_train = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=120), "adj_close": np.exp(np.linspace(np.log(100), np.log(95), 120))}
    )
    targets = pd.date_range("2024-05-01", periods=5).tolist()
    out = returns_baselines.momentum_forecast(asset_train, asset_train["timestamp"].iloc[-1], targets)
    assert len(out) == 5
    assert out["prediction"].notna().all()


def test_returns_metrics_run_on_log_return_frame():
    df = _price_forecasts()
    lr = targets_mod.to_log_return(df)
    s = metrics_returns.summarise_returns(lr, by=["method"])
    assert "forecast_sharpe" in s.columns
    assert "hit_rate" in s.columns
    assert "ic_pearson" in s.columns


def test_forecast_sharpe_handles_perfect_signal():
    pred = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    target = np.array([0.01, -0.01, 0.01, -0.01, 0.01])
    sh = metrics_returns.forecast_sharpe(pred, target)
    # With perfect positive correlation, Sharpe is undefined (zero stdev),
    # so we accept NaN or +inf-like; just check it doesn't error.
    assert sh != float("-inf")


def test_forecast_sharpe_negative_signal():
    rng = np.random.default_rng(0)
    target = rng.normal(0, 0.01, size=200)
    # Predict the opposite sign of every realisation -> negative Sharpe
    pred = -np.sign(target)
    sh = metrics_returns.forecast_sharpe(pred, target)
    assert sh < 0


def test_hit_rate_perfect_signal_is_one():
    pred = np.array([1.0, -1.0, 1.0])
    target = np.array([0.01, -0.01, 0.005])
    assert metrics_returns.hit_rate_returns(pred, target) == 1.0


def test_convert_target_dispatches_correctly():
    df = _price_forecasts()
    p = targets_mod.convert_target(df, "price")
    assert (p["target_type"] == "price").all()
    lr = targets_mod.convert_target(df, "log_return")
    assert (lr["target_type"] == "log_return").all()


def test_convert_target_rejects_unknown():
    df = _price_forecasts()
    with pytest.raises(ValueError, match="Unknown target_type"):
        targets_mod.convert_target(df, "garbage")
