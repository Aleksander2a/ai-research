"""Tests for the sandbox-executed code-spec (Iter 20)."""

from __future__ import annotations

import pytest

from autosignalx.agent import codegen

_GOOD_CODE = """
import numpy as np
import pandas as pd

def forecast_fn(asset_train, origin, target_dates):
    last = float(asset_train["adj_close"].iloc[-1])
    return pd.DataFrame({"timestamp": target_dates, "prediction": [last] * len(target_dates)})
"""

_NO_FUNCTION_CODE = """
x = 5
"""

_FORBIDDEN_IMPORT_CODE = """
import os

def forecast_fn(asset_train, origin, target_dates):
    return None
"""

_FORBIDDEN_BUILTIN_CODE = """
def forecast_fn(asset_train, origin, target_dates):
    eval("1+1")
    return None
"""

_DUNDER_ACCESS_CODE = """
def forecast_fn(asset_train, origin, target_dates):
    return asset_train.__class__
"""

_SYNTAX_ERROR_CODE = """
def forecast_fn(asset_train, origin, target_dates
    return None
"""


def test_validate_good_code() -> None:
    ok, err = codegen.validate_code(_GOOD_CODE)
    assert ok is True
    assert err == ""


def test_validate_empty_code() -> None:
    ok, err = codegen.validate_code("")
    assert ok is False
    assert "non-empty" in err


def test_validate_too_long_code() -> None:
    ok, err = codegen.validate_code("x = 1\n" * 2000)
    assert ok is False
    assert "too long" in err


def test_validate_no_forecast_fn() -> None:
    ok, err = codegen.validate_code(_NO_FUNCTION_CODE)
    assert ok is False
    assert "forecast_fn" in err


def test_validate_forbidden_import() -> None:
    ok, err = codegen.validate_code(_FORBIDDEN_IMPORT_CODE)
    assert ok is False
    assert "import of 'os'" in err


def test_validate_forbidden_builtin() -> None:
    ok, err = codegen.validate_code(_FORBIDDEN_BUILTIN_CODE)
    assert ok is False
    assert "eval" in err


def test_validate_dunder_access() -> None:
    ok, err = codegen.validate_code(_DUNDER_ACCESS_CODE)
    assert ok is False
    assert "dunder" in err


def test_validate_syntax_error() -> None:
    ok, err = codegen.validate_code(_SYNTAX_ERROR_CODE)
    assert ok is False
    assert "syntax error" in err


def test_compile_good_code_callable() -> None:
    fn = codegen.compile_forecast_fn(_GOOD_CODE)
    assert callable(fn)


def test_compile_forbidden_raises_sandbox_violation() -> None:
    with pytest.raises(codegen.SandboxViolation):
        codegen.compile_forecast_fn(_FORBIDDEN_IMPORT_CODE)


def test_execute_code_spec_validates_bad_name() -> None:
    """Spec without a valid name is rejected before any code runs."""
    result = codegen.execute_code_spec({"name": "bad name!", "code": _GOOD_CODE})
    assert result["status"] == "error"
    assert "alphanumeric" in result["error"]


def test_execute_code_spec_validates_bad_code() -> None:
    result = codegen.execute_code_spec(
        {"name": "x", "code": _FORBIDDEN_IMPORT_CODE}
    )
    assert result["status"] == "error"
    assert "validate_code" in result["error"]
