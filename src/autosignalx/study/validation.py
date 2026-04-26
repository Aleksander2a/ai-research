"""Pre-flight validation for a Study config.

Catches problems that would otherwise surface deep into a pipeline
(missing tickers, splits in the wrong order, too few walk-forward
windows, ...). Two tiers:

  - **errors**:   block downstream work (returned by validate(...) as
                  the ``errors`` list; a non-empty list means do not
                  proceed).
  - **warnings**: surfaced to the user but do not block.

Network-touching checks (yfinance ticker availability) are opt-in so
the cheap config-sanity validation runs offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from autosignalx.data import splits
from autosignalx.study.config import Study


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "info": list(self.info),
        }


def validate(study: Study, check_tickers: bool = False) -> ValidationReport:
    """Run all enabled checks; return a ValidationReport."""
    report = ValidationReport()
    _check_date_ordering(study, report)
    _check_window_count(study, report)
    _check_min_assets(study, report)
    if check_tickers:
        _check_ticker_availability(study, report)
    return report


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _check_date_ordering(study: Study, r: ValidationReport) -> None:
    try:
        start = _ts(study.start_date)
        end = _ts(study.end_date)
        train = _ts(study.train_end)
        val = _ts(study.val_end)
        test = _ts(study.test_end)
    except (TypeError, ValueError) as e:
        r.errors.append(f"could not parse one of the dates: {e}")
        return

    if not (start < train < val < test):
        r.errors.append(
            f"date ordering violated: need start ({study.start_date}) "
            f"< train_end ({study.train_end}) < val_end ({study.val_end}) "
            f"< test_end ({study.test_end})"
        )
    if end < test:
        r.errors.append(
            f"end_date ({study.end_date}) must be on or after test_end "
            f"({study.test_end})"
        )

    # Helpful information
    span_days = (end - start).days
    if span_days < 365 * 3:
        r.warnings.append(
            f"short total span ({span_days} days, ~{span_days / 252:.1f} trading "
            f"years); regime + signal layers prefer >=3 years"
        )
    train_days = (train - start).days
    if train_days < 252 * 2:
        r.warnings.append(
            f"short training window ({train_days} days, ~{train_days / 252:.1f} "
            f"trading years); Chronos-2 zero-shot still works but small fits "
            f"may be unstable"
        )


def _check_window_count(study: Study, r: ValidationReport) -> None:
    try:
        windows = splits.walk_forward_windows(
            val_end=study.val_end,
            test_end=study.test_end,
            horizon_days=study.forecast_horizon_days,
            step_days=study.rolling_step_days,
        )
    except Exception as e:  # noqa: BLE001
        r.errors.append(f"walk-forward window construction failed: {e}")
        return

    n = len(windows)
    if n == 0:
        r.errors.append(
            f"zero walk-forward windows: test span "
            f"({study.val_end} -> {study.test_end}) is too short for "
            f"horizon={study.forecast_horizon_days}d step={study.rolling_step_days}d"
        )
    elif n < 5:
        r.warnings.append(
            f"only {n} walk-forward windows; statistical significance tests "
            f"will be wide. Consider a longer test span or smaller step."
        )
    else:
        r.info.append(f"walk-forward windows: {n}")


def _check_min_assets(study: Study, r: ValidationReport) -> None:
    if len(study.assets) < 2:
        r.warnings.append(
            "single-asset universe; cross-sectional strategies "
            "(LongShortKK, TopKLong) need at least 2 assets to function"
        )
    if len(study.assets) > 50:
        r.warnings.append(
            f"large universe ({len(study.assets)} assets); pipeline runtime "
            f"scales linearly. Chronos-2 inference dominates -- expect long runs."
        )


def _check_ticker_availability(study: Study, r: ValidationReport) -> None:
    """Light yfinance probe: try a 5-day fetch on each ticker.

    Network-bound; opt-in only. Failures appear as warnings (not errors)
    because intermittent failures should not block a run that would
    otherwise succeed once the network recovers.
    """
    try:
        import yfinance as yf
    except ImportError:
        r.warnings.append("yfinance not installed; skipping availability check")
        return

    bad = []
    for ticker in list(study.assets) + list(study.macro):
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if hist is None or hist.empty:
                bad.append(ticker)
        except Exception:  # noqa: BLE001
            bad.append(ticker)
    if bad:
        r.warnings.append(
            f"yfinance returned no data for: {', '.join(bad)} (typo or delisted?)"
        )
    else:
        r.info.append(
            f"yfinance availability OK for "
            f"{len(study.assets) + len(study.macro)} tickers"
        )
