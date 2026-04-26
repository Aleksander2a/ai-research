"""Default strategy bundles and prerequisite validation for backtests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from autosignalx.config import settings

if TYPE_CHECKING:
    from autosignalx.study import Study


DEFAULT_CLI_STRATEGIES = [
    "BuyAndHoldSPY",
    "EqualWeightUniverse",
    "TopKLong:k=3",
    "LongShortKK:k=2",
    "RegimeGated:k=3",
    "FindingDriven",
]
FORECAST_DRIVEN = {"TopKLong", "LongShortKK"}
REGIME_DRIVEN = {"RegimeGated", "FindingDriven"}


def default_cli_strategies() -> list[str]:
    """Historical default bundle for the project-wide backtest flow."""
    return list(DEFAULT_CLI_STRATEGIES)


def default_study_strategies(study: Study) -> list[str]:
    """Pick the compatible study bundle from the artifacts currently present."""
    selected: list[str] = []
    if "SPY" in study.assets:
        selected.append("BuyAndHoldSPY")
    selected.append("EqualWeightUniverse")

    artifacts = _artifact_paths(study)
    if artifacts["chronos"].exists():
        selected.extend(["TopKLong:k=3", "LongShortKK:k=2"])
    if (
        artifacts["chronos"].exists()
        and artifacts["regimes"].exists()
        and artifacts["findings"].exists()
    ):
        selected.extend(["RegimeGated:k=3", "FindingDriven"])
    return selected


def ensure_strategy_prerequisites(
    strategy_specs: list[str], *, study: Study | None = None, universe: list[str] | None = None
) -> None:
    """Fail early when a requested strategy's inputs are unavailable."""
    heads = [_strategy_head(spec) for spec in strategy_specs]
    artifacts = _artifact_paths(study)
    scope = f"study {study.name!r}" if study is not None else "the default project scope"
    asset_universe = universe or (list(study.assets) if study is not None else None)

    if "BuyAndHoldSPY" in heads and asset_universe is not None and "SPY" not in asset_universe:
        raise ValueError(
            "BuyAndHoldSPY requires SPY in the active universe. "
            f"Current universe for {scope} is {asset_universe}."
        )

    for head in heads:
        if head in FORECAST_DRIVEN and not artifacts["chronos"].exists():
            raise ValueError(_missing_chronos_message(head, artifacts["chronos"], study))
        if head in REGIME_DRIVEN:
            if not artifacts["chronos"].exists():
                raise ValueError(_missing_chronos_message(head, artifacts["chronos"], study))
            if not artifacts["regimes"].exists():
                raise ValueError(_missing_regimes_message(head, artifacts["regimes"], study))
            if not artifacts["findings"].exists():
                raise ValueError(_missing_findings_message(head, artifacts["findings"], study))


def _artifact_paths(study: Study | None) -> dict[str, Path]:
    if study is None:
        return {
            "chronos": settings.reports_dir / "ablations" / "chronos2.parquet",
            "regimes": settings.reports_dir / "regimes" / "kmeans.parquet",
            "findings": settings.reports_dir / "agent" / "findings.jsonl",
        }
    return {
        "chronos": study.ablations_dir / "chronos2.parquet",
        "regimes": study.regimes_dir / "kmeans.parquet",
        "findings": study.agent_dir / "findings.jsonl",
    }


def _strategy_head(spec: str) -> str:
    return spec.partition(":")[0]


def _missing_chronos_message(head: str, path: Path, study: Study | None) -> str:
    if study is None:
        return (
            f"{head} requires forecast signals from {path}. "
            "Run `autosignalx eval chronos` first."
        )
    return (
        f"{head} requires study-local forecast signals from {path}. "
        f"Run `autosignalx eval chronos --study {study.name}` first."
    )


def _missing_regimes_message(head: str, path: Path, study: Study | None) -> str:
    if study is None:
        return (
            f"{head} requires regime labels at {path}. "
            "Run `autosignalx regime fit` first."
        )
    return (
        f"{head} requires study-local regime labels at {path}. "
        "Custom studies currently auto-generate data, forecast, and backtest artifacts only; "
        "the project-wide regime command is `autosignalx regime fit`, which does not write "
        "into this study tree."
    )


def _missing_findings_message(head: str, path: Path, study: Study | None) -> str:
    if study is None:
        return (
            f"{head} requires promoted findings at {path}. "
            "Run `autosignalx agent run --max-rounds 5` first."
        )
    return (
        f"{head} requires study-local promoted findings at {path}. "
        "Custom studies currently auto-generate data, forecast, and backtest artifacts only; "
        "the project-wide agent command is `autosignalx agent run --max-rounds 5`, which does "
        "not write into this study tree."
    )
