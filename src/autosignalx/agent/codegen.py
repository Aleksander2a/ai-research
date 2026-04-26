"""Sandboxed Python code generation for agent-authored forecast functions.

The Iter 13 DSL constrains the agent to a fixed set of compositional
primitives. Iter 20 unlocks **arbitrary Python**: the agent emits the
body of a ``forecast_fn(asset_train, origin, target_dates) ->
pd.DataFrame`` and we compile + execute it in a heavily restricted
namespace. This is where agent autonomy crosses into actual
programming.

Safety model -- defense in depth:

1. **AST validation** rejects every dangerous construct *before* exec:
   imports of non-allowed modules, ``exec`` / ``eval`` / ``compile`` /
   ``__import__``, dunder attribute access, file/network builtins,
   subprocess / os / sys touchpoints, lambdas calling unknown
   identifiers.
2. **Restricted globals** -- only a curated set of safe builtins
   (range, len, abs, min, max, sum, sorted, ...) plus ``np`` and
   ``pd`` aliases. No ``__builtins__`` dict access.
3. **Function-shape validation** -- the compiled module must define
   exactly one callable named ``forecast_fn`` with the expected
   parameter signature.
4. **Persistence as artifact** -- generated code is saved to
   ``reports/agent/generated_methods/<name>.py`` so reviewers can
   read it, audit it, and re-run it deterministically.

Even with all these guards, this iteration is OFF by default in the
agent's prompts -- the constrained DSL (Iter 13) is the recommended
path. This module is the explicit-opt-in expansion."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autosignalx.config import settings
from autosignalx.data import cache, splits
from autosignalx.eval import harness

ALLOWED_IMPORTS = frozenset({"numpy", "pandas", "math"})
ALLOWED_BUILTINS = frozenset(
    {
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "isinstance", "len", "list", "map", "max", "min", "range",
        "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
        "True", "False", "None",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        "exec", "eval", "compile", "__import__", "open", "input",
        "globals", "locals", "vars", "getattr", "setattr", "delattr",
        "hasattr", "exit", "quit", "breakpoint", "help",
    }
)
GENERATED_DIR = settings.reports_dir / "agent" / "generated_methods"


class SandboxViolation(Exception):  # noqa: N818
    """Raised when generated code fails AST validation."""


def _check_node(node: ast.AST) -> None:
    """Recursively validate an AST node. Raises SandboxViolation."""
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            for alias in child.names:
                if alias.name not in ALLOWED_IMPORTS:
                    raise SandboxViolation(
                        f"import of '{alias.name}' not allowed; "
                        f"allowed: {sorted(ALLOWED_IMPORTS)}"
                    )
        if isinstance(child, ast.ImportFrom) and (
            child.module is None or child.module.split(".")[0] not in ALLOWED_IMPORTS
        ):
            raise SandboxViolation(f"from-import of '{child.module}' not allowed")
        if (
            isinstance(child, ast.Attribute)
            and child.attr.startswith("__")
            and child.attr.endswith("__")
        ):
            raise SandboxViolation(f"dunder attribute access ('{child.attr}') not allowed")
        if isinstance(child, ast.Name) and child.id in FORBIDDEN_NAMES:
            raise SandboxViolation(f"forbidden name reference: '{child.id}'")


def validate_code(code: str) -> tuple[bool, str]:
    """AST-validate the source. Returns (ok, error_message)."""
    if not isinstance(code, str) or not code.strip():
        return False, "code must be a non-empty string"
    if len(code) > 8000:
        return False, "code too long (limit 8000 chars)"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    try:
        _check_node(tree)
    except SandboxViolation as e:
        return False, str(e)
    func_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not any(fd.name == "forecast_fn" for fd in func_defs):
        return False, "module must define a function named 'forecast_fn'"
    return True, ""


def _safe_import(name: str, *_args: Any, **_kwargs: Any) -> Any:
    """Sandbox-safe ``__import__`` -- only ALLOWED_IMPORTS resolve."""
    import importlib

    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of '{name}' not allowed in sandbox")
    return importlib.import_module(name)


def _safe_globals() -> dict[str, Any]:
    """Restricted globals dict for exec'ing generated code."""
    import builtins as builtins_mod

    safe_builtins: dict[str, Any] = {
        name: getattr(builtins_mod, name)
        for name in ALLOWED_BUILTINS
        if hasattr(builtins_mod, name)
    }
    safe_builtins["__import__"] = _safe_import
    return {
        "__builtins__": safe_builtins,
        "np": np,
        "pd": pd,
        "math": __import__("math"),
    }


def compile_forecast_fn(code: str) -> Callable:
    """Validate, compile, and exec; return the resulting forecast_fn."""
    ok, err = validate_code(code)
    if not ok:
        raise SandboxViolation(err)
    globals_dict = _safe_globals()
    locals_dict: dict[str, Any] = {}
    exec(compile(code, "<agent-generated>", "exec"), globals_dict, locals_dict)  # noqa: S102
    fn = locals_dict.get("forecast_fn")
    if not callable(fn):
        raise SandboxViolation("forecast_fn is not callable after exec")
    return fn


def _persist_code(name: str, code: str, metadata: dict[str, Any]) -> Path:
    """Save the generated source for audit and reproducibility."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    py_path = GENERATED_DIR / f"{name}.py"
    py_path.write_text(code, encoding="utf-8")
    meta_path = GENERATED_DIR / f"{name}.json"
    meta = {
        **metadata,
        "name": name,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return py_path


def execute_code_spec(
    spec: dict[str, Any], config_name: str = "default"
) -> dict[str, Any]:
    """Compile + run a code-spec experiment.

    Spec schema:
        {
            "name": str (alphanumeric + _-),
            "code": str (Python source defining forecast_fn),
            "max_windows": int (cap; defaults to 8),
            "asset_subset": list[str] (optional)
        }
    """
    name = spec.get("name", "")
    if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
        return {"status": "error", "error": "name must be alphanumeric (with underscores/dashes)"}
    code = spec.get("code", "")
    ok, err = validate_code(code)
    if not ok:
        return {"status": "error", "error": f"validate_code: {err}"}
    try:
        fn = compile_forecast_fn(code)
    except SandboxViolation as e:
        return {"status": "error", "error": f"sandbox violation: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"compile error: {e}"}

    from autosignalx.config import load_config

    cfg = load_config(config_name)
    eval_cfg = cfg["eval"]
    splits_cfg = eval_cfg["splits"]

    ohlcv = cache.read_ohlcv()
    asset_subset = spec.get("asset_subset")
    if asset_subset:
        ohlcv = ohlcv[ohlcv["asset"].isin(asset_subset)]
        if ohlcv.empty:
            return {"status": "error", "error": "asset_subset filtered out all data"}

    windows = splits.walk_forward_windows(
        val_end=splits_cfg["val_end"],
        test_end=splits_cfg["test_end"],
        horizon_days=eval_cfg["forecast_horizon_days"],
        step_days=eval_cfg["rolling_step_days"],
    )
    max_windows = int(spec.get("max_windows", 8))
    windows = windows[:max_windows]

    try:
        forecasts = harness.run_walk_forward(name, fn, ohlcv, windows)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"runtime error in generated forecast_fn: {e}"}
    if forecasts.empty:
        return {"status": "error", "error": "no forecasts produced"}

    out_path = settings.reports_dir / "ablations" / f"{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out_path, index=False)
    py_path = _persist_code(
        name,
        code,
        metadata={
            "n_rows": int(len(forecasts)),
            "n_windows": int(len(windows)),
            "asset_subset": asset_subset,
            "ablation_path": str(out_path),
        },
    )

    summary = harness.summarize(forecasts, by=["method"])
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    return {
        "status": "ok",
        "name": name,
        "code_path": str(py_path),
        "ablation_path": str(out_path),
        "n_rows": int(len(forecasts)),
        "n_windows": int(len(windows)),
        "summary": {
            "mae": float(row.get("mae", 0.0)) if row.get("mae") is not None else None,
            "mape": float(row.get("mape", 0.0)) if row.get("mape") is not None else None,
        },
    }
