"""Tests for the Study pre-flight validator."""

from __future__ import annotations

import pytest

from autosignalx.config import settings
from autosignalx.study import Study, validation


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "repo_root", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "reports").mkdir()
    yield tmp_path


def _good_study(**overrides) -> Study:
    base = dict(
        name="ok",
        assets=["SPY", "QQQ", "TLT"],
        macro=["^VIX"],
        start_date="2010-01-01",
        end_date="2025-12-31",
        train_end="2018-12-31",
        val_end="2020-12-31",
        test_end="2025-12-31",
        forecast_horizon_days=21,
        rolling_step_days=21,
    )
    base.update(overrides)
    return Study(**base)


def test_default_dates_pass(temp_root):
    r = validation.validate(_good_study())
    assert r.ok, f"unexpected errors: {r.errors}"


def test_misordered_splits_error(temp_root):
    s = _good_study(
        train_end="2020-12-31", val_end="2018-12-31", test_end="2025-12-31"
    )
    r = validation.validate(s)
    assert not r.ok
    assert any("date ordering" in e for e in r.errors)


def test_test_end_after_end_date_error(temp_root):
    s = _good_study(end_date="2022-12-31", test_end="2024-12-31")
    r = validation.validate(s)
    assert not r.ok
    assert any("end_date" in e for e in r.errors)


def test_walk_forward_construction_failure_recorded_as_error(temp_root):
    """Underlying splits builder raises on val_end == test_end; validator
    surfaces this as an error rather than letting it propagate."""
    s = _good_study(
        train_end="2020-12-30", val_end="2020-12-31", test_end="2020-12-31",
        end_date="2020-12-31",
    )
    r = validation.validate(s)
    assert not r.ok
    # date-ordering check fires first; window-construction also fails.
    assert any("date ordering" in e or "window construction" in e for e in r.errors)


def test_short_total_span_warning(temp_root):
    s = _good_study(
        start_date="2022-01-01", train_end="2022-06-30",
        val_end="2022-09-30", test_end="2022-12-31", end_date="2022-12-31",
    )
    r = validation.validate(s)
    assert any("short total span" in w for w in r.warnings)


def test_few_windows_warning(temp_root):
    """Test span just long enough to produce a small handful of windows."""
    s = _good_study(
        start_date="2018-01-01", train_end="2020-06-30",
        val_end="2021-01-31", test_end="2021-03-15", end_date="2021-03-15",
        forecast_horizon_days=21, rolling_step_days=21,
    )
    r = validation.validate(s)
    assert r.ok
    assert any("only" in w and "windows" in w for w in r.warnings)


def test_single_asset_warning(temp_root):
    r = validation.validate(_good_study(assets=["SPY"]))
    assert any("single-asset" in w for w in r.warnings)


def test_large_universe_warning(temp_root):
    big = [f"T{i}" for i in range(60)]
    r = validation.validate(_good_study(assets=big))
    assert any("large universe" in w for w in r.warnings)
