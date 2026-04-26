"""Pydantic ``Study`` model + on-disk persistence.

A study config is a small YAML at ``data/studies/<name>/study.yaml`` that
declares the universe, dates, and per-layer hyperparameters. The
artifact subdirectories under that root are layer-owned (cache,
ablations, regimes, signals, graph, backtest, agent), and each layer's
read/write functions accept the study's resolved paths as overrides.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from autosignalx.config import settings

DEFAULT_ASSETS: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "GLD", "TLT", "EFA", "EEM", "HYG",
)

DEFAULT_MACRO: tuple[str, ...] = ("^TNX", "^VIX", "DX-Y.NYB", "CL=F")

NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


class StudyNotFoundError(KeyError):
    """No study by that name on disk."""


class StudyExistsError(FileExistsError):
    """A study with this name already exists on disk."""


class Study(BaseModel):
    """User-defined run configuration for a complete pipeline pass."""

    name: str = Field(..., description="Unique alphanumeric identifier.")
    description: str = Field(default="", description="Free-form notes.")
    assets: list[str] = Field(default_factory=lambda: list(DEFAULT_ASSETS))
    macro: list[str] = Field(default_factory=lambda: list(DEFAULT_MACRO))
    start_date: str = Field(default="2010-01-01")
    end_date: str = Field(default="2025-12-31")
    train_end: str = Field(default="2018-12-31")
    val_end: str = Field(default="2020-12-31")
    test_end: str = Field(default="2025-12-31")
    forecast_horizon_days: int = Field(default=21, ge=1)
    rolling_step_days: int = Field(default=21, ge=1)
    n_regimes: int = Field(default=4, ge=2, le=10)
    signal_top_k: int = Field(default=8, ge=1)
    cost_bps: float = Field(default=5.0, ge=0.0)
    backtest_start: str | None = Field(
        default=None,
        description="Defaults to the day after val_end if unset.",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ValueError(
                f"Study name {v!r} must match {NAME_RE.pattern} "
                f"(alphanumeric, underscore, hyphen)."
            )
        if v == "default":
            raise ValueError("'default' is reserved; pick a different name.")
        return v

    @field_validator("assets", "macro")
    @classmethod
    def _validate_tickers(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v if t and t.strip()]
        if not cleaned:
            raise ValueError("ticker list must be non-empty")
        return cleaned

    # --- path resolution ---------------------------------------------------

    @property
    def root(self) -> Path:
        return settings.repo_root / "data" / "studies" / self.name

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def reports_root(self) -> Path:
        return settings.repo_root / "reports" / "studies" / self.name

    @property
    def ablations_dir(self) -> Path:
        return self.reports_root / "ablations"

    @property
    def regimes_dir(self) -> Path:
        return self.reports_root / "regimes"

    @property
    def signals_dir(self) -> Path:
        return self.reports_root / "signals"

    @property
    def graph_dir(self) -> Path:
        return self.reports_root / "graph"

    @property
    def backtest_runs_dir(self) -> Path:
        return self.reports_root / "backtest" / "runs"

    @property
    def agent_dir(self) -> Path:
        return self.reports_root / "agent"

    @property
    def config_path(self) -> Path:
        return self.root / "study.yaml"

    @property
    def effective_backtest_start(self) -> str:
        if self.backtest_start:
            return self.backtest_start
        # Day after val_end (string-add 1 day). Use pandas for safety.
        import pandas as pd

        return (pd.Timestamp(self.val_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # --- persistence -------------------------------------------------------

    def save(self, *, overwrite: bool = False) -> Path:
        """Write the study YAML to disk; create the root + standard subdirs."""
        if self.config_path.exists() and not overwrite:
            raise StudyExistsError(
                f"Study {self.name!r} already exists at {self.root}; "
                f"use overwrite=True or pick a different name."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))
        return self.config_path

    @classmethod
    def load(cls, name: str) -> Study:
        path = settings.repo_root / "data" / "studies" / name / "study.yaml"
        if not path.exists():
            raise StudyNotFoundError(
                f"No study {name!r} at {path}. "
                f"Create one with `autosignalx study create --name {name} ...`"
            )
        return cls(**yaml.safe_load(path.read_text()))

    def delete(self) -> None:
        """Remove the study directory tree; safe-no-op if missing."""
        import shutil

        if self.root.exists():
            shutil.rmtree(self.root)
        if self.reports_root.exists():
            shutil.rmtree(self.reports_root)


def list_studies() -> list[str]:
    """Names of all studies currently on disk."""
    studies_root = settings.repo_root / "data" / "studies"
    if not studies_root.exists():
        return []
    out = []
    for child in sorted(studies_root.iterdir()):
        if child.is_dir() and (child / "study.yaml").exists():
            out.append(child.name)
    return out
