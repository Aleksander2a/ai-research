"""Leakage tests for the backtest layer.

Hard invariant: the backtest window must start strictly after the
discovery window end (2020-12-31). Any artifact produced by
``run_backtest`` must obey this. The runner asserts at entry; this test
both unit-tests the assertion and validates that the public default
config respects the floor.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from autosignalx.backtest import runner
from autosignalx.backtest.schemas import DISCOVERY_END, BacktestConfig


def test_discovery_end_constant_matches_project_split():
    """If anyone moves DISCOVERY_END earlier, this test breaks loudly.

    The project's val_end (configs/default.yaml) is 2020-12-31; the
    backtest floor must equal it, not be earlier.
    """
    assert datetime(2020, 12, 31) == DISCOVERY_END


def test_default_config_starts_after_discovery_end():
    cfg = BacktestConfig()
    start = pd.Timestamp(cfg.start_date)
    assert start > pd.Timestamp(DISCOVERY_END), (
        "default backtest start must be strictly after discovery end"
    )


def test_runner_rejects_pre_discovery_start():
    cfg = BacktestConfig(start_date="2020-12-31", end_date="2021-12-31")
    with pytest.raises(ValueError, match="strictly after"):
        runner.run_backtest(cfg)


def test_runner_rejects_inside_discovery_window():
    cfg = BacktestConfig(start_date="2018-01-01", end_date="2019-12-31")
    with pytest.raises(ValueError, match="strictly after"):
        runner.run_backtest(cfg)


def test_runner_accepts_first_legal_day():
    """2021-01-01 is the first legal start. Even if no trading happens
    until 2021-01-04, the assertion itself must accept the date."""
    cfg = BacktestConfig(
        strategies=[],  # empty so we don't exercise data loading
        start_date="2021-01-01",
        end_date="2021-01-31",
    )
    start = pd.Timestamp(cfg.start_date)
    # We only test the assertion in isolation, not full execution
    runner._assert_no_leakage(start)
