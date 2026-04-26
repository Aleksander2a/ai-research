"""Tests for Phase 3 chat RAG layer.

These run in deterministic hashed-embedding mode (no DeepInfra key
required) and verify the corpus, index, retrieval, grounded-answer
contract, and the eval harness.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from autosignalx.chat import answer as answer_mod
from autosignalx.chat import corpus as corpus_mod
from autosignalx.chat import embed as embed_mod
from autosignalx.chat import eval as eval_mod
from autosignalx.chat import index as index_mod
from autosignalx.config import settings


@pytest.fixture
def fake_reports(tmp_path, monkeypatch):
    """Build a tiny, fully-controlled artifact tree under tmp_path."""
    monkeypatch.setattr(settings, "repo_root", tmp_path)
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    # Force replay-equivalent mode (no key, hashed embeddings)
    monkeypatch.setattr(settings, "deepinfra_api_key", "")
    monkeypatch.setattr(settings, "autosignalx_replay", True)

    rd = tmp_path / "reports"
    (rd / "agent").mkdir(parents=True)
    (rd / "backtest" / "runs" / "run_test").mkdir(parents=True)

    (rd / "agent" / "ledger.jsonl").write_text(
        json.dumps({"round": 0, "step": "theorist", "content": {"hypothesis": "SPY beats naive in regime 2"}, "session_id": "s1"})
        + "\n"
        + json.dumps({"round": 0, "step": "skeptic", "content": "The hypothesis is well-scoped but small effect.", "session_id": "s1"})
        + "\n",
        encoding="utf-8",
    )
    (rd / "agent" / "findings.jsonl").write_text(
        json.dumps({
            "id": "f_test01",
            "hypothesis": "TLT chronos2_multivariate beats naive in regime 3",
            "method": "chronos2_multivariate",
            "filters": {"asset": "TLT", "regime_id": 3},
            "evidence": {
                "n": 100, "skill_vs_baseline": 0.05, "p_value": 0.03,
                "bootstrap_ci_low": 0.01, "bootstrap_ci_high": 0.09,
            },
            "session_id": "s1", "round": 0, "replication_count": 1,
        }) + "\n",
        encoding="utf-8",
    )
    (rd / "agent" / "lessons.md").write_text(
        "# Lessons\n\nNaive forecasts are surprisingly hard to beat in low-vol regimes.\n",
        encoding="utf-8",
    )
    (rd / "agent" / "telemetry.jsonl").write_text(
        json.dumps({"model": "M1", "cost_usd": 0.01, "total_tokens": 500}) + "\n",
        encoding="utf-8",
    )
    (rd / "agent" / "trace_quality.jsonl").write_text(
        json.dumps({"round": 0, "clarity": 4, "novelty": 3, "falsifiability": 5,
                    "evidence_citing": 4, "rationale": "well-cited"}) + "\n",
        encoding="utf-8",
    )
    (rd / "agent" / "self_critique.jsonl").write_text(
        json.dumps({"finding_id": "f_test01", "current_state": "unchanged",
                    "rationale": "no later evidence contradicts"}) + "\n",
        encoding="utf-8",
    )
    (rd / "backtest" / "runs" / "run_test" / "metrics.json").write_text(
        json.dumps({
            "per_strategy": {
                "TopKLong": {"cagr": 0.08, "sharpe": 1.2, "max_drawdown": -0.15, "turnover": 0.3},
                "BuyAndHoldSPY": {"cagr": 0.07, "sharpe": 0.9, "max_drawdown": -0.20, "turnover": 0.0},
            }
        }),
        encoding="utf-8",
    )
    return rd


def test_build_corpus_covers_all_kinds(fake_reports):
    chunks = corpus_mod.build_corpus(reports_dir=fake_reports)
    kinds = {c.kind for c in chunks}
    assert {"ledger", "finding", "lesson", "telemetry", "trace_quality", "self_critique", "backtest"} <= kinds


def test_chunks_have_unique_citation_ids(fake_reports):
    chunks = corpus_mod.build_corpus(reports_dir=fake_reports)
    ids = [c.citation_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_hashed_embed_is_deterministic():
    a = embed_mod.hashed_embed(["hello world", "second text"])
    b = embed_mod.hashed_embed(["hello world", "second text"])
    np.testing.assert_array_equal(a, b)
    assert a.shape == (2, embed_mod.HASHED_DIM)


def test_build_index_then_load_round_trip(fake_reports):
    idx = index_mod.build_index(reports_dir=fake_reports, force_hashed=True)
    assert len(idx.chunks) > 0
    loaded = index_mod.load_index()
    assert loaded is not None
    assert len(loaded.chunks) == len(idx.chunks)
    assert loaded.vectors.shape == idx.vectors.shape


def test_retrieval_finds_relevant_finding(fake_reports):
    idx = index_mod.build_index(reports_dir=fake_reports, force_hashed=True)
    embedder = embed_mod.EmbeddingProvider(force_hashed=True)
    qvec = embedder.embed(["TLT chronos2 multivariate regime 3"])[0]
    top = idx.search(qvec, k=3)
    cids = [c.citation_id for c, _ in top]
    assert any("f_test01" in c for c in cids)


def test_answer_replay_mode_includes_citations(fake_reports):
    idx = index_mod.build_index(reports_dir=fake_reports, force_hashed=True)
    a = answer_mod.answer_question("What finding is promoted for TLT?", index=idx)
    assert a.mode == "replay"
    assert a.retrieved
    assert any("finding:f_test01" in c.citation_id for c, _ in a.retrieved)


def test_answer_handles_empty_index(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "empty_reports")
    a = answer_mod.answer_question("anything")
    assert a.mode == "empty"


def test_eval_harness_runs_in_replay(fake_reports):
    idx = index_mod.build_index(reports_dir=fake_reports, force_hashed=True)
    summary = eval_mod.run_eval(index=idx)
    assert summary["n"] == len(eval_mod.CASES)
    # Citation recall should be > 0 -- at least the finding/backtest cases
    # land on the right kind in the tiny fixture.
    assert summary["citation_recall"] > 0.0
    # Refusal score is 1.0 in replay mode by definition.
    assert summary["refusal_correct"] == 1.0
