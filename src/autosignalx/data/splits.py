"""Walk-forward and static train/val/test split definitions.

The standard split partitions the timeline into train (initial training),
validation (hyperparameter search), and test (out-of-sample, walk-forward
re-training over a rolling window). Walk-forward windows yield successive
forecast horizons whose training sets monotonically widen as time advances --
no future data ever leaks into a training window."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    """One walk-forward step: train on (-inf, train_end], forecast (forecast_start, forecast_end].

    Constraints (asserted at construction):
    - ``train_end < forecast_start`` (no leakage)
    - ``forecast_start <= forecast_end`` (non-empty forecast window)"""

    train_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.train_end < self.forecast_start:
            raise ValueError(
                f"Leakage: train_end {self.train_end} >= forecast_start {self.forecast_start}"
            )
        if not self.forecast_start <= self.forecast_end:
            raise ValueError(
                f"Empty forecast window: {self.forecast_start} > {self.forecast_end}"
            )


def walk_forward_windows(
    val_end: pd.Timestamp | str,
    test_end: pd.Timestamp | str,
    horizon_days: int,
    step_days: int,
) -> list[WalkForwardWindow]:
    """Generate walk-forward windows over the test period.

    Starting at ``val_end``, each window's training set ends right before
    its forecast starts (no future leakage). Successive windows advance
    by ``step_days``; each forecast covers ``horizon_days`` (clamped at
    ``test_end``)."""
    val_end_ts = pd.Timestamp(val_end)
    test_end_ts = pd.Timestamp(test_end)
    if not val_end_ts < test_end_ts:
        raise ValueError(f"val_end {val_end_ts} must precede test_end {test_end_ts}")

    windows: list[WalkForwardWindow] = []
    train_end = val_end_ts
    while train_end < test_end_ts:
        forecast_start = train_end + pd.Timedelta(days=1)
        forecast_end = min(
            train_end + pd.Timedelta(days=horizon_days),
            test_end_ts,
        )
        windows.append(
            WalkForwardWindow(
                train_end=train_end,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
            )
        )
        train_end = train_end + pd.Timedelta(days=step_days)

    return windows


@dataclass(frozen=True)
class StaticSplit:
    """Single train/val/test partition by date boundary.

    Slice semantics: ``train`` = (-inf, train_end]; ``val`` = (train_end, val_end];
    ``test`` = (val_end, test_end]. Boundaries are inclusive on the right of
    each split, exclusive on the left, so the three splits are disjoint."""

    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        if not (self.train_end < self.val_end < self.test_end):
            raise ValueError(
                "Splits must be strictly ordered: "
                f"train_end={self.train_end}, val_end={self.val_end}, test_end={self.test_end}"
            )

    def slice(self, df: pd.DataFrame, ts_col: str = "timestamp") -> dict[str, pd.DataFrame]:
        """Return ``{'train', 'val', 'test'}`` sub-frames respecting the boundaries."""
        return {
            "train": df[df[ts_col] <= self.train_end].copy(),
            "val": df[(df[ts_col] > self.train_end) & (df[ts_col] <= self.val_end)].copy(),
            "test": df[(df[ts_col] > self.val_end) & (df[ts_col] <= self.test_end)].copy(),
        }
