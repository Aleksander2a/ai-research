"""Smoke tests: package imports cleanly, config loads, every layer module
is importable. Catches packaging and basic configuration regressions."""

from __future__ import annotations

import importlib

import pytest

LAYER_MODULES = [
    "autosignalx.data",
    "autosignalx.eval",
    "autosignalx.forecast",
    "autosignalx.regime",
    "autosignalx.signal",
    "autosignalx.graph",
    "autosignalx.agent",
]


def test_package_imports_and_has_version() -> None:
    import autosignalx

    assert autosignalx.__version__
    assert isinstance(autosignalx.__version__, str)


def test_settings_load_with_defaults() -> None:
    """Without a DeepInfra key, replay mode must default on so the cockpit
    demo works for reviewers without a DeepInfra account."""
    from autosignalx.config import settings

    assert settings.repo_root.exists()
    assert isinstance(settings.use_replay, bool)
    assert settings.use_replay is True or settings.deepinfra_api_key != ""


@pytest.mark.parametrize("module_name", LAYER_MODULES)
def test_layer_module_importable(module_name: str) -> None:
    """Every layer namespace exists and imports cleanly, even before its
    iteration lands. Catches packaging errors early and ensures the CLI's
    layer-status table doesn't lie."""
    importlib.import_module(module_name)
