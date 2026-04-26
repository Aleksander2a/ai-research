"""Tests for the Phase 4 static HTML snapshot generator."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from autosignalx.config import settings
from autosignalx.snapshot import builder


@pytest.fixture
def fake_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "repo_root", tmp_path)
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    rd = tmp_path / "reports"
    (rd / "agent").mkdir(parents=True)
    (rd / "ablations").mkdir()
    (rd / "regimes").mkdir()
    (rd / "chat").mkdir()
    (rd / "backtest" / "runs" / "run_01").mkdir(parents=True)

    # Findings
    (rd / "agent" / "findings.jsonl").write_text(
        json.dumps({
            "id": "f_x", "hypothesis": "H1", "method": "chronos2_multivariate",
            "filters": {"asset": "TLT", "regime_id": 3},
            "evidence": {"n": 100, "skill_vs_baseline": 0.05, "p_value": 0.03,
                         "bootstrap_ci_low": 0.01, "bootstrap_ci_high": 0.09},
            "replication_count": 1,
        }) + "\n",
        encoding="utf-8",
    )
    # Ledger + telemetry + trace
    (rd / "agent" / "ledger.jsonl").write_text(
        json.dumps({"round": 0, "step": "theorist", "content": "hyp"}) + "\n", encoding="utf-8"
    )
    (rd / "agent" / "telemetry.jsonl").write_text(
        json.dumps({"model": "M", "cost_usd": 0.01, "ts": "2025-01-01"}) + "\n", encoding="utf-8"
    )
    (rd / "agent" / "trace_quality.jsonl").write_text(
        json.dumps({"round": 0, "clarity": 4, "novelty": 3, "falsifiability": 5,
                    "evidence_citing": 4, "rationale": "ok"}) + "\n", encoding="utf-8"
    )

    # Forecasts ablation
    df = pd.DataFrame({
        "asset": ["SPY"] * 5,
        "timestamp": pd.date_range("2024-01-01", periods=5),
        "target": [100.0, 101.0, 102.0, 103.0, 104.0],
        "prediction": [100.5, 101.2, 101.9, 103.1, 104.2],
    })
    df.to_parquet(rd / "ablations" / "naive.parquet")

    # Regimes
    rdf = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5),
        "regime_id": [0, 1, 1, 2, 0],
    })
    rdf.to_parquet(rd / "regimes" / "kmeans.parquet")

    # Backtest run
    (rd / "backtest" / "runs" / "run_01" / "metrics.json").write_text(
        json.dumps({
            "BuyAndHoldSPY": {"cagr": 0.07, "sharpe": 0.9, "max_drawdown": -0.2,
                              "calmar": 0.35, "avg_turnover": 0.001},
            "TopKLong": {"cagr": 0.09, "sharpe": 1.1, "max_drawdown": -0.18,
                         "calmar": 0.5, "avg_turnover": 0.3},
        }),
        encoding="utf-8",
    )
    portfolio = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10).tolist() * 2,
        "strategy": ["BuyAndHoldSPY"] * 10 + ["TopKLong"] * 10,
        "equity": list(range(1, 11)) + list(range(1, 11)),
    })
    portfolio.to_parquet(rd / "backtest" / "runs" / "run_01" / "portfolio_daily.parquet")

    # Chat
    (rd / "chat" / "chunks.jsonl").write_text(
        json.dumps({"citation_id": "finding:f_x", "kind": "finding", "text": "Promoted finding f_x", "meta": {}}) + "\n",
        encoding="utf-8",
    )
    return rd


def test_build_snapshot_writes_all_pages(fake_reports, tmp_path):
    out = tmp_path / "snapshot_out"
    result = builder.build_snapshot(reports_dir=fake_reports, out_dir=out)
    expected = {"index.html", "forecasts.html", "regimes.html", "findings.html",
                "backtest.html", "agent.html", "chat.html"}
    assert expected <= set(result.pages_written)
    for name in expected:
        assert (out / name).exists()
        assert (out / name).read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    assert (out / "manifest.json").exists()


def test_index_page_has_metric_cards(fake_reports, tmp_path):
    out = tmp_path / "out"
    builder.build_snapshot(reports_dir=fake_reports, out_dir=out)
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "Promoted findings" in html
    assert "AutoSignal-X" in html
    assert "Architecture" in html


def test_findings_page_renders_finding(fake_reports, tmp_path):
    out = tmp_path / "out"
    builder.build_snapshot(reports_dir=fake_reports, out_dir=out)
    html = (out / "findings.html").read_text(encoding="utf-8")
    assert "f_x" in html
    assert "chronos2_multivariate" in html


def test_backtest_page_has_metrics_table(fake_reports, tmp_path):
    out = tmp_path / "out"
    result = builder.build_snapshot(reports_dir=fake_reports, out_dir=out)
    html = (out / "backtest.html").read_text(encoding="utf-8")
    assert "TopKLong" in html
    assert "BuyAndHoldSPY" in html
    # Should produce at least one Plotly figure (equity curve)
    assert result.figures >= 1


def test_chat_page_lists_chunks(fake_reports, tmp_path):
    out = tmp_path / "out"
    builder.build_snapshot(reports_dir=fake_reports, out_dir=out)
    html = (out / "chat.html").read_text(encoding="utf-8")
    assert "finding:f_x" in html


def test_graceful_degradation_with_no_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "empty_reports")
    out = tmp_path / "snap"
    result = builder.build_snapshot(out_dir=out)
    assert len(result.pages_written) == 10
    # Pages should be valid HTML even with no artifacts
    for name in result.pages_written:
        html = (out / name).read_text(encoding="utf-8")
        assert html.startswith("<!DOCTYPE html>")
