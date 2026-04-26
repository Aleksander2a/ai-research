"""Per-study configuration + isolated artifact directories.

A ``Study`` is the unit of "run AutoSignal-X on my own assets and dates".
It owns its own data cache (``data/studies/<name>/cache/``) and reports
tree (``reports/studies/<name>/``) so multiple studies can coexist
without clobbering each other or the default project artifacts.

Default behaviour (no ``--study`` flag) is unchanged: layers continue
to read/write the canonical paths under ``data/cache/`` and
``reports/``. Studies are strictly opt-in.
"""

from autosignalx.study.config import (
    DEFAULT_ASSETS,
    DEFAULT_MACRO,
    Study,
    StudyExistsError,
    StudyNotFoundError,
    list_studies,
)

__all__ = [
    "DEFAULT_ASSETS",
    "DEFAULT_MACRO",
    "Study",
    "StudyExistsError",
    "StudyNotFoundError",
    "list_studies",
]
