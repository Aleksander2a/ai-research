"""Phase 16 -- Reproducibility badge: git hash + env hash + RNG / data hashes.

A research result is only as credible as the artifacts it claims to
have come from. The reproducibility badge bundles:

* git commit hash + dirty flag
* Python version + key library versions
* environment variables that affect runtime (replay flag, model names)
* hash of every parquet under reports/ (so tampering is detectable)

Output: a deterministic JSON dict suitable for rendering in the cockpit
and committing alongside any published finding. The badge is *informational*,
not enforceable -- but a reviewer can compare two badges to detect drift.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_hash(repo_root: Path) -> dict[str, Any]:
    try:
        rev = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        return {
            "commit": rev,
            "branch": branch,
            "dirty": bool(status),
            "porcelain_lines": len(status.splitlines()),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _python_env() -> dict[str, Any]:
    libs = {}
    for lib in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "langgraph", "statsmodels"):
        try:
            mod = __import__(lib if lib != "scikit-learn" else "sklearn")
            libs[lib] = getattr(mod, "__version__", "?")
        except ImportError:
            libs[lib] = "missing"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": libs,
    }


def _file_hashes(reports_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not reports_dir.exists():
        return hashes
    for fp in sorted(reports_dir.rglob("*.parquet")):
        try:
            h = hashlib.sha256()
            with fp.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            rel = str(fp.relative_to(reports_dir))
            hashes[rel] = h.hexdigest()[:16]
        except Exception:  # noqa: BLE001
            continue
    for fp in sorted(reports_dir.rglob("*.jsonl")):
        try:
            h = hashlib.sha256()
            with fp.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            rel = str(fp.relative_to(reports_dir))
            hashes[rel] = h.hexdigest()[:16]
        except Exception:  # noqa: BLE001
            continue
    return hashes


def reproducibility_badge(
    repo_root: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute the reproducibility badge for the current state."""
    from autosignalx.config import settings

    rr = repo_root or settings.repo_root
    rd = reports_dir or settings.reports_dir
    hashes = _file_hashes(rd)
    bundle_hash = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": _git_hash(rr),
        "env": _python_env(),
        "replay_mode": bool(settings.use_replay),
        "n_artifacts": len(hashes),
        "artifacts_bundle_hash": bundle_hash,
        "artifact_hashes": hashes,
    }


def write_badge(path: Path | None = None) -> Path:
    from autosignalx.config import settings

    p = path or (settings.reports_dir / "reproducibility_badge.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(reproducibility_badge(), indent=2, default=str),
        encoding="utf-8",
    )
    return p
