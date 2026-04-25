"""Leakage tests: walk-forward windows must respect temporal ordering;
StaticSplit slices must be non-overlapping. Non-negotiable prerequisites
for the eval harness in Iter 2."""

from __future__ import annotations

import pandas as pd
import pytest

from autosignalx.data.splits import StaticSplit, WalkForwardWindow, walk_forward_windows


class TestWalkForwardWindow:
    def test_train_end_strictly_before_forecast_start_passes(self) -> None:
        WalkForwardWindow(
            train_end=pd.Timestamp("2020-01-01"),
            forecast_start=pd.Timestamp("2020-01-02"),
            forecast_end=pd.Timestamp("2020-01-22"),
        )

    def test_train_end_equal_to_forecast_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="Leakage"):
            WalkForwardWindow(
                train_end=pd.Timestamp("2020-01-01"),
                forecast_start=pd.Timestamp("2020-01-01"),
                forecast_end=pd.Timestamp("2020-01-21"),
            )

    def test_train_end_after_forecast_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="Leakage"):
            WalkForwardWindow(
                train_end=pd.Timestamp("2020-01-02"),
                forecast_start=pd.Timestamp("2020-01-01"),
                forecast_end=pd.Timestamp("2020-01-21"),
            )

    def test_empty_forecast_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            WalkForwardWindow(
                train_end=pd.Timestamp("2020-01-01"),
                forecast_start=pd.Timestamp("2020-01-22"),
                forecast_end=pd.Timestamp("2020-01-15"),
            )


class TestWalkForwardWindows:
    def test_first_window_starts_after_val_end(self) -> None:
        windows = walk_forward_windows(
            val_end="2020-12-31",
            test_end="2021-12-31",
            horizon_days=21,
            step_days=21,
        )
        assert windows
        assert windows[0].train_end == pd.Timestamp("2020-12-31")
        assert windows[0].forecast_start > pd.Timestamp("2020-12-31")

    def test_windows_strictly_advance(self) -> None:
        windows = walk_forward_windows(
            val_end="2020-12-31",
            test_end="2021-12-31",
            horizon_days=21,
            step_days=21,
        )
        for prev, curr in zip(windows, windows[1:], strict=False):
            assert curr.train_end > prev.train_end, (
                f"Walk-forward must monotonically advance: "
                f"{prev.train_end} -> {curr.train_end}"
            )
            assert curr.forecast_start > prev.forecast_start

    def test_last_window_does_not_exceed_test_end(self) -> None:
        windows = walk_forward_windows(
            val_end="2020-12-31",
            test_end="2021-12-31",
            horizon_days=21,
            step_days=21,
        )
        assert windows[-1].forecast_end <= pd.Timestamp("2021-12-31")

    def test_no_window_train_overlaps_forecast(self) -> None:
        """The bedrock leakage test: for every window, train data ends
        strictly before forecast data begins."""
        windows = walk_forward_windows(
            val_end="2020-12-31",
            test_end="2021-12-31",
            horizon_days=21,
            step_days=21,
        )
        for w in windows:
            assert w.train_end < w.forecast_start

    def test_val_end_after_test_end_rejected(self) -> None:
        with pytest.raises(ValueError):
            walk_forward_windows(
                val_end="2021-01-01",
                test_end="2020-12-31",
                horizon_days=21,
                step_days=21,
            )


class TestStaticSplit:
    def test_split_ordering_enforced(self) -> None:
        with pytest.raises(ValueError, match="strictly ordered"):
            StaticSplit(
                train_end=pd.Timestamp("2020-12-31"),
                val_end=pd.Timestamp("2018-12-31"),  # out of order
                test_end=pd.Timestamp("2025-12-31"),
            )

    def test_slices_are_disjoint_and_complete(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2010-01-01", "2025-12-31", freq="D"),
                "value": 1,
            }
        )
        split = StaticSplit(
            train_end=pd.Timestamp("2018-12-31"),
            val_end=pd.Timestamp("2020-12-31"),
            test_end=pd.Timestamp("2025-12-31"),
        )
        parts = split.slice(df)
        assert parts["train"]["timestamp"].max() <= split.train_end
        assert parts["val"]["timestamp"].min() > split.train_end
        assert parts["val"]["timestamp"].max() <= split.val_end
        assert parts["test"]["timestamp"].min() > split.val_end
        assert parts["test"]["timestamp"].max() <= split.test_end
        total_rows = len(df[df["timestamp"] <= split.test_end])
        union_rows = sum(len(p) for p in parts.values())
        assert union_rows == total_rows

    def test_slices_dont_overlap_in_timestamps(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2010-01-01", "2025-12-31", freq="D"),
                "value": 1,
            }
        )
        split = StaticSplit(
            train_end=pd.Timestamp("2018-12-31"),
            val_end=pd.Timestamp("2020-12-31"),
            test_end=pd.Timestamp("2025-12-31"),
        )
        parts = split.slice(df)
        for a, b in [("train", "val"), ("val", "test"), ("train", "test")]:
            overlap = set(parts[a]["timestamp"]) & set(parts[b]["timestamp"])
            assert not overlap, f"{a} and {b} timestamps must not overlap"
