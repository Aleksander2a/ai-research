"""Study-aware backtest strategy selection tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from autosignalx.backtest.strategy_selection import (
    default_study_strategies,
    ensure_strategy_prerequisites,
)
from autosignalx.config import settings
from autosignalx.study import Study


@pytest.fixture
def temp_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "repo_root", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "reports").mkdir()
    yield tmp_path


def _make_study(name: str, assets: list[str]) -> Study:
    s = Study(
        name=name,
        assets=assets,
        macro=["^VIX"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        train_end="2022-06-30",
        val_end="2023-06-30",
        test_end="2024-12-31",
    )
    s.save()
    return s


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_default_study_strategies_without_spy_or_forecasts(temp_repo_root):
    study = _make_study("plain", ["AAPL", "MSFT"])
    assert default_study_strategies(study) == ["EqualWeightUniverse"]


def test_default_study_strategies_with_spy_without_forecasts(temp_repo_root):
    study = _make_study("spy_only", ["SPY", "QQQ"])
    assert default_study_strategies(study) == ["BuyAndHoldSPY", "EqualWeightUniverse"]


def test_default_study_strategies_include_forecast_bundle_when_chronos_exists(temp_repo_root):
    study = _make_study("chronos", ["AAPL", "MSFT"])
    _touch(study.ablations_dir / "chronos2.parquet")

    assert default_study_strategies(study) == [
        "EqualWeightUniverse",
        "TopKLong:k=3",
        "LongShortKK:k=2",
    ]


def test_default_study_strategies_include_regime_bundle_when_all_artifacts_exist(temp_repo_root):
    study = _make_study("full", ["AAPL", "MSFT"])
    _touch(study.ablations_dir / "chronos2.parquet")
    _touch(study.regimes_dir / "kmeans.parquet")
    _touch(study.agent_dir / "findings.jsonl", text='{"ok": true}\n')

    assert default_study_strategies(study) == [
        "EqualWeightUniverse",
        "TopKLong:k=3",
        "LongShortKK:k=2",
        "RegimeGated:k=3",
        "FindingDriven",
    ]


def test_explicit_forecast_strategy_requires_study_chronos_artifact(temp_repo_root):
    study = _make_study("needs_forecast", ["AAPL", "MSFT"])

    with pytest.raises(ValueError, match=r"autosignalx eval chronos --study needs_forecast"):
        ensure_strategy_prerequisites(["TopKLong:k=3"], study=study, universe=study.assets)
