"""Lightweight runtime validation for local app / CLI launches.

The primary failure mode this guards against is launching the Streamlit
cockpit from a stale global environment (for example a Conda install)
while the repo source has moved on. In that scenario the app file comes
from the current checkout but ``autosignalx`` resolves from somewhere
else, so study-aware calls fail with confusing signature mismatches.
"""

from __future__ import annotations

import inspect
from pathlib import Path


def validate_repo_runtime(expected_repo_root: Path) -> dict[str, object]:
    """Return runtime diagnostics for the current ``autosignalx`` import.

    The result is a plain dict so callers can load this module by path
    without depending on package-local class identity.
    """
    expected_repo_root = expected_repo_root.resolve()
    try:
        import autosignalx
        from autosignalx.config import settings
        from autosignalx.data import cache as data_cache
        from autosignalx.study import pipeline as study_pipeline
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "errors": [
                "Could not import the expected AutoSignal-X runtime modules. "
                f"The current environment is likely stale or incomplete: {exc}"
            ],
            "details": [],
        }

    errors: list[str] = []
    details: list[str] = []
    module_locations = {
        "autosignalx": Path(inspect.getfile(autosignalx)).resolve(),
        "autosignalx.data.cache": Path(inspect.getfile(data_cache)).resolve(),
        "autosignalx.study.pipeline": Path(inspect.getfile(study_pipeline)).resolve(),
    }
    for name, path in module_locations.items():
        details.append(f"{name}: {path}")
        if not path.is_relative_to(expected_repo_root):
            errors.append(
                f"{name} resolved outside the repo checkout: {path}"
            )

    details.append(f"settings.repo_root: {settings.repo_root}")
    if settings.repo_root.resolve() != expected_repo_root:
        errors.append(
            "autosignalx.config.settings.repo_root does not match the current repo "
            f"checkout ({settings.repo_root} != {expected_repo_root})"
        )

    for fn_name in ("write_ohlcv", "write_macro"):
        sig = inspect.signature(getattr(data_cache, fn_name))
        details.append(f"{fn_name}{sig}")
        if "cache_root" not in sig.parameters:
            errors.append(
                f"{fn_name}{sig} is missing the expected `cache_root` parameter"
            )

    return {
        "ok": not errors,
        "errors": errors,
        "details": details,
    }
