"""Tests for the Study model and per-study artifact isolation."""

from __future__ import annotations

import pandas as pd
import pytest

from autosignalx.config import settings
from autosignalx.data import cache as data_cache
from autosignalx.study import Study, StudyExistsError, StudyNotFoundError, list_studies


@pytest.fixture
def temp_studies_root(tmp_path, monkeypatch):
    """Redirect studies/ + reports/studies/ under tmp_path for isolation."""
    monkeypatch.setattr(settings, "repo_root", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "reports").mkdir()
    yield tmp_path


def _ohlcv_fixture() -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-04", periods=5)
    rows = []
    for asset in ("AAA", "BBB"):
        for ts in idx:
            rows.append({
                "timestamp": ts, "asset": asset,
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
                "adj_close": 100.5, "volume": 1_000_000, "returns": 0.0,
            })
    return pd.DataFrame(rows)


def test_study_validates_name(temp_studies_root):
    with pytest.raises(ValueError, match="must match"):
        Study(name="bad name with spaces", assets=["SPY"])


def test_study_rejects_default_name(temp_studies_root):
    with pytest.raises(ValueError, match="reserved"):
        Study(name="default", assets=["SPY"])


def test_study_rejects_empty_assets(temp_studies_root):
    with pytest.raises(ValueError, match="non-empty"):
        Study(name="x", assets=[])


def test_study_save_load_roundtrip(temp_studies_root):
    s = Study(
        name="my_test",
        description="hello",
        assets=["AAPL", "MSFT"],
        start_date="2018-01-01",
        end_date="2024-12-31",
    )
    path = s.save()
    assert path.exists()
    assert s.cache_dir.exists()
    loaded = Study.load("my_test")
    assert loaded.name == "my_test"
    assert loaded.description == "hello"
    assert loaded.assets == ["AAPL", "MSFT"]


def test_study_save_rejects_duplicate(temp_studies_root):
    s = Study(name="dup", assets=["SPY"])
    s.save()
    with pytest.raises(StudyExistsError):
        s.save()


def test_study_save_overwrite(temp_studies_root):
    s = Study(name="dup", assets=["SPY"], description="v1")
    s.save()
    s2 = Study(name="dup", assets=["SPY"], description="v2")
    s2.save(overwrite=True)
    assert Study.load("dup").description == "v2"


def test_load_unknown_raises(temp_studies_root):
    with pytest.raises(StudyNotFoundError):
        Study.load("nonexistent")


def test_list_studies(temp_studies_root):
    assert list_studies() == []
    Study(name="alpha", assets=["SPY"]).save()
    Study(name="beta", assets=["QQQ"]).save()
    assert list_studies() == ["alpha", "beta"]


def test_study_delete_removes_tree(temp_studies_root):
    s = Study(name="rm", assets=["SPY"])
    s.save()
    assert s.root.exists()
    s.delete()
    assert not s.root.exists()
    assert "rm" not in list_studies()


def test_effective_backtest_start_defaults_to_day_after_val_end(temp_studies_root):
    s = Study(name="x", assets=["SPY"], val_end="2020-12-31")
    assert s.effective_backtest_start == "2021-01-01"


def test_effective_backtest_start_respects_override(temp_studies_root):
    s = Study(
        name="x", assets=["SPY"], val_end="2020-12-31", backtest_start="2022-06-01"
    )
    assert s.effective_backtest_start == "2022-06-01"


def test_cache_writes_under_study_dir(temp_studies_root):
    """write_ohlcv with study cache_root lands inside the study's tree."""
    s = Study(name="iso", assets=["AAA", "BBB"])
    s.save()
    df = _ohlcv_fixture()

    out_path = data_cache.write_ohlcv(df, cache_root=s.cache_dir)
    assert (s.cache_dir / "ohlcv.parquet").exists()
    # Returned path is inside the study tree.
    assert s.cache_dir in out_path.parents

    got = data_cache.read_ohlcv(cache_root=s.cache_dir)
    assert len(got) == len(df)


def test_cache_default_and_study_paths_differ(temp_studies_root):
    """Default cache root and study cache root resolve to distinct dirs."""
    s = Study(name="distinct", assets=["AAA"])
    s.save()
    df = _ohlcv_fixture()
    data_cache.write_ohlcv(df, cache_root=s.cache_dir)
    info_study = data_cache.cache_status(cache_root=s.cache_dir)
    info_default = data_cache.cache_status()  # uses module default
    assert info_study["ohlcv"]["path"] != info_default["ohlcv"]["path"]


def test_cache_status_for_study(temp_studies_root):
    s = Study(name="iso2", assets=["AAA", "BBB"])
    s.save()
    df = _ohlcv_fixture()
    data_cache.write_ohlcv(df, cache_root=s.cache_dir)
    info = data_cache.cache_status(cache_root=s.cache_dir)
    assert info["ohlcv"]["exists"] is True
    assert info["ohlcv"]["rows"] == len(df)
    assert info["macro"]["exists"] is False
