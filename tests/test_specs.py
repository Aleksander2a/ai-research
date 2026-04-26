"""Tests for the constrained code-spec DSL (Iter 13).

Validation tests use synthetic specs (no real fits). Execution is
exercised by the integration runs in Iter 19's recorded session."""

from __future__ import annotations

import pytest

from autosignalx.agent import specs


def test_validate_minimal_naive_spec_ok() -> None:
    ok, err = specs.validate_spec({"name": "test_method", "base": "naive"})
    assert ok is True
    assert err == ""


def test_validate_missing_name_fails() -> None:
    ok, err = specs.validate_spec({"base": "naive"})
    assert ok is False
    assert "name" in err


def test_validate_unknown_base_fails() -> None:
    ok, err = specs.validate_spec({"name": "x", "base": "ghost"})
    assert ok is False
    assert "base" in err


def test_validate_bad_name_chars_fails() -> None:
    ok, err = specs.validate_spec({"name": "bad name!", "base": "naive"})
    assert ok is False
    assert "alphanumeric" in err


def test_validate_bad_covariate_subset_fails() -> None:
    ok, err = specs.validate_spec(
        {"name": "x", "base": "chronos2_multivariate", "covariate_subset": "not_a_list"}
    )
    assert ok is False
    assert "covariate_subset" in err


@pytest.mark.parametrize("w", [-0.1, 1.5, "not_a_number"])
def test_validate_bad_ensemble_weight_fails(w) -> None:
    ok, err = specs.validate_spec(
        {"name": "x", "base": "naive", "ensemble_naive_weight": w}
    )
    assert ok is False
    assert "ensemble_naive_weight" in err


def test_validate_bad_max_windows_fails() -> None:
    ok, err = specs.validate_spec(
        {"name": "x", "base": "naive", "max_windows": -5}
    )
    assert ok is False
    assert "max_windows" in err


def test_validate_full_valid_spec() -> None:
    ok, err = specs.validate_spec(
        {
            "name": "chronos2_dxyonly_ensembled",
            "base": "chronos2_multivariate",
            "covariate_subset": ["DX-Y.NYB"],
            "ensemble_naive_weight": 0.3,
            "max_windows": 8,
            "asset_subset": ["SPY", "EFA"],
        }
    )
    assert ok is True
    assert err == ""


def test_allowed_bases_set() -> None:
    assert "naive" in specs.ALLOWED_BASES
    assert "chronos2_multivariate" in specs.ALLOWED_BASES
    assert len(specs.ALLOWED_BASES) == 4
