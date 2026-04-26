"""Tests for cost / latency / token telemetry (Iter 17)."""

from __future__ import annotations

import time
from pathlib import Path

from autosignalx.agent import telemetry


def test_estimate_cost_usd_known_model() -> None:
    # Kimi-K2.6 default: 0.55 in, 2.20 out per M tokens
    cost = telemetry.estimate_cost_usd("moonshotai/Kimi-K2.6", 1_000_000, 1_000_000)
    assert abs(cost - (0.55 + 2.20)) < 1e-6


def test_estimate_cost_usd_unknown_model_uses_fallback() -> None:
    cost = telemetry.estimate_cost_usd("unknown/model", 1_000_000, 1_000_000)
    # Default (0.50, 2.00)
    assert abs(cost - 2.50) < 1e-6


def test_model_prices_env_override(monkeypatch) -> None:
    monkeypatch.setenv("DEEPINFRA_PRICE_FOO_BAR_IN", "0.10")
    monkeypatch.setenv("DEEPINFRA_PRICE_FOO_BAR_OUT", "0.40")
    in_p, out_p = telemetry.model_prices("foo/bar")
    assert in_p == 0.10
    assert out_p == 0.40


def test_record_call_persists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "TELEMETRY_DIR", tmp_path)
    telemetry.clear()
    rec = telemetry.record_call(
        model="moonshotai/Kimi-K2.6",
        role="theorist",
        step="theorist",
        round_n=0,
        prompt_tokens=1000,
        completion_tokens=500,
        latency_ms=1234.5,
    )
    assert rec["model"] == "moonshotai/Kimi-K2.6"
    assert rec["total_tokens"] == 1500
    assert rec["cost_usd"] > 0
    rows = telemetry.load()
    assert len(rows) == 1
    assert rows[0]["latency_ms"] == 1234.5


def test_call_timer_measures_elapsed() -> None:
    with telemetry.CallTimer() as t:
        time.sleep(0.01)
    assert t.elapsed_ms >= 5  # gives some slack for noisy timers
