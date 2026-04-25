"""Cost / latency / token telemetry for live LLM calls.

Every call through the LiveProvider is timed and the OpenAI-compatible
``response.response_metadata`` is mined for token counts. Cost is
estimated from a hard-coded per-model rate table (DeepInfra publishes
prices on its model pages; the rates here are conservative defaults
that can be overridden via ``DEEPINFRA_PRICE_<...>`` env vars).

Records persist to ``reports/agent/telemetry.jsonl``; the cockpit
Telemetry panel renders them as per-model breakdowns and per-session
totals."""

from __future__ import annotations

import contextlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autosignalx.config import settings

TELEMETRY_DIR = settings.reports_dir / "agent"


def _telemetry_path() -> Path:
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    return TELEMETRY_DIR / "telemetry.jsonl"


# Conservative defaults in $ per 1,000,000 tokens (input, output).
# Override per-model via DEEPINFRA_PRICE_<KEY>_IN / _OUT env vars where
# <KEY> is the model id with non-alphanumerics replaced by underscores
# and uppercased.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "moonshotai/Kimi-K2.6": (0.55, 2.20),
    "moonshotai/Kimi-K2.5": (0.55, 2.20),
    "zai-org/GLM-5.1": (0.50, 2.00),
    "zai-org/GLM-4.7-Flash": (0.10, 0.40),
    "zai-org/GLM-4.7": (0.40, 1.60),
    "deepseek-ai/DeepSeek-V4-Pro": (0.40, 1.80),
    "deepseek-ai/DeepSeek-V3.1-Terminus": (0.30, 1.30),
    "deepseek-ai/DeepSeek-V3": (0.27, 1.10),
    "Qwen/Qwen3-Max": (0.50, 2.00),
    "Qwen/Qwen3.5-122B-A10B": (0.40, 1.60),
    "MiniMaxAI/MiniMax-M2.5": (0.40, 1.60),
}


def _env_price_key(model_id: str, suffix: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in model_id).upper()
    return f"DEEPINFRA_PRICE_{safe}_{suffix}"


def model_prices(model_id: str) -> tuple[float, float]:
    """Return (price_per_M_input_tokens, price_per_M_output_tokens) in USD.

    Env-var override wins; otherwise DEFAULT_PRICES; otherwise a
    conservative placeholder of (0.50, 2.00)."""
    in_env = os.environ.get(_env_price_key(model_id, "IN"), "").strip()
    out_env = os.environ.get(_env_price_key(model_id, "OUT"), "").strip()
    in_price, out_price = DEFAULT_PRICES.get(model_id, (0.50, 2.00))
    if in_env:
        with contextlib.suppress(ValueError):
            in_price = float(in_env)
    if out_env:
        with contextlib.suppress(ValueError):
            out_price = float(out_env)
    return in_price, out_price


def estimate_cost_usd(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost for a call given input/output token counts and the price table."""
    in_p, out_p = model_prices(model_id)
    return (prompt_tokens / 1_000_000.0) * in_p + (completion_tokens / 1_000_000.0) * out_p


def record_call(
    model: str,
    role: str,
    step: str,
    round_n: int,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Persist one telemetry record. Returns the record."""
    rec = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model,
        "role": role,
        "step": step,
        "round": round_n,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
        "latency_ms": float(latency_ms),
        "cost_usd": estimate_cost_usd(model, prompt_tokens, completion_tokens),
        "session_id": session_id,
    }
    with _telemetry_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def load() -> list[dict[str, Any]]:
    path = _telemetry_path()
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def clear() -> None:
    p = _telemetry_path()
    if p.exists():
        p.unlink()


class CallTimer:
    """Context manager that measures wall-clock latency of a block."""

    def __init__(self) -> None:
        self.start_ms: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "CallTimer":
        self.start_ms = time.perf_counter() * 1000
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = time.perf_counter() * 1000 - self.start_ms
