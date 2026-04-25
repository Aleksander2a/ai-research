"""LLM provider abstraction with live (DeepInfra) and replay modes.

Live mode wraps ``langchain_openai.ChatOpenAI`` pointed at DeepInfra's
OpenAI-compatible endpoint. Replay mode reads pre-recorded responses
from ``replay/agent_steps.jsonl`` keyed by ``(round, step)``, so
reviewers without a DeepInfra key can still walk through a complete
agent session in the cockpit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

from autosignalx.config import settings

REPLAY_PATH = settings.replay_dir / "agent_steps.jsonl"


class LLMProvider(Protocol):
    def chat(self, messages: list[dict[str, str]], step: str, round: int) -> str: ...
    @property
    def mode(self) -> str: ...


class ReplayProvider:
    """Plays back recorded LLM responses keyed by ``(round, step)``."""

    mode = "replay"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or REPLAY_PATH
        self._records: dict[tuple[int, str], str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (int(rec.get("round", 0)), str(rec.get("step", "")))
                self._records[key] = str(rec.get("content", ""))

    def chat(
        self,
        messages: list[dict[str, str]],  # noqa: ARG002 (protocol signature)
        step: str,
        round: int,
    ) -> str:
        key = (round, step)
        if key in self._records:
            return self._records[key]
        return _fallback_response(step, round)


class LiveProvider:
    """DeepInfra (OpenAI-compatible) chat via ``langchain_openai.ChatOpenAI``.

    Caches responses by content hash so re-runs are deterministic and free."""

    mode = "live"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.deepinfra.com/v1/openai",
        temperature: float = 0.0,
        cache_dir: Path | None = None,
        record_path: Path | None = None,
    ) -> None:
        from langchain_openai import ChatOpenAI

        self.client = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )
        self.cache_dir = cache_dir or (settings.reports_dir / "agent" / "llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.record_path = record_path

    def _cache_key(self, messages: list[dict[str, str]]) -> str:
        return hashlib.sha256(
            json.dumps(messages, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]

    def chat(self, messages: list[dict[str, str]], step: str, round: int) -> str:
        key = self._cache_key(messages)
        cache_file = self.cache_dir / f"{key}.txt"
        if cache_file.exists():
            content = cache_file.read_text(encoding="utf-8")
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            lc_messages = []
            for m in messages:
                role = m.get("role", "user")
                if role == "system":
                    lc_messages.append(SystemMessage(content=m["content"]))
                else:
                    lc_messages.append(HumanMessage(content=m["content"]))
            response = self.client.invoke(lc_messages)
            content = response.content if isinstance(response.content, str) else str(response.content)
            cache_file.write_text(content, encoding="utf-8")

        if self.record_path is not None:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            with self.record_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"round": round, "step": step, "content": content}, default=str
                    )
                    + "\n"
                )
        return content


def _fallback_response(step: str, round: int) -> str:
    """Deterministic plausible response for replay mode when no record exists.

    Lets the agent loop keep running smoothly even if the replay file is
    incomplete; the resulting trace is structured but obviously synthetic."""
    if step == "propose":
        slots = [
            {
                "hypothesis": "Chronos-2 multivariate may beat naive on Regime 2 for SPY.",
                "experiment": {
                    "type": "slice_forecasts",
                    "params": {"method": "chronos2_multivariate", "asset": "SPY", "regime_id": 2},
                },
            },
            {
                "hypothesis": "Naive's dominance is asset-specific; check QQQ in Regime 1.",
                "experiment": {
                    "type": "slice_forecasts",
                    "params": {"method": "chronos2_univariate", "asset": "QQQ", "regime_id": 1},
                },
            },
            {
                "hypothesis": "TLT (low centrality) may have a different forecast profile.",
                "experiment": {
                    "type": "slice_forecasts",
                    "params": {"method": "arima", "asset": "TLT", "regime_id": 0},
                },
            },
        ]
        return json.dumps(slots[round % len(slots)])
    if step == "critique":
        return (
            "The hypothesis is well-scoped (single method, asset, regime). "
            "The expected effect size is small; treat any observed lift as "
            "directional rather than conclusive without a Diebold-Mariano test."
        )
    if step == "decide":
        return json.dumps({"action": "continue", "reason": "more slices to explore"})
    return ""


ROLE_TO_ENV = {
    "proposer": "DEEPINFRA_MODEL_PROPOSER",
    "critic": "DEEPINFRA_MODEL_CRITIC",
    "chat": "DEEPINFRA_MODEL_CHAT",
    "theorist": "DEEPINFRA_MODEL_THEORIST",
    "skeptic": "DEEPINFRA_MODEL_SKEPTIC",
    "adjudicator": "DEEPINFRA_MODEL_ADJUDICATOR",
}


def _model_for_role(role: str) -> str:
    """Resolve which DeepInfra model to use for a role.

    Falls back through the env var hierarchy: role-specific env >
    proposer/critic/chat default > a sensible global default."""
    env_key = ROLE_TO_ENV.get(role, "DEEPINFRA_MODEL_PROPOSER")
    explicit = os.environ.get(env_key, "").strip()
    if explicit:
        return explicit
    if role in ("theorist", "proposer"):
        return settings.deepinfra_model_proposer or "moonshotai/Kimi-K2.6"
    if role in ("skeptic", "critic"):
        return settings.deepinfra_model_critic or "zai-org/GLM-5.1"
    if role in ("adjudicator", "chat"):
        return settings.deepinfra_model_chat or "deepseek-ai/DeepSeek-V4-Pro"
    return settings.deepinfra_model_proposer or "moonshotai/Kimi-K2.6"


def get_provider(record_replay: bool = False, role: str = "proposer") -> LLMProvider:
    """Factory: live DeepInfra if a key is set and replay isn't forced;
    otherwise the deterministic replay provider.

    The ``role`` parameter selects the model via env-var hierarchy
    (DEEPINFRA_MODEL_<ROLE>), letting different agent roles use different
    models -- e.g. a creative proposer model for the Theorist, a critical
    one for the Skeptic, a decisive one for the Adjudicator.

    When ``record_replay`` is True and we end up live, every response is
    appended to ``replay/agent_steps.jsonl`` so a future no-key reviewer
    sees the same trace."""
    if settings.use_replay:
        return ReplayProvider()
    record_path = REPLAY_PATH if record_replay else None
    return LiveProvider(
        api_key=settings.deepinfra_api_key,
        model=_model_for_role(role),
        base_url=settings.deepinfra_base_url,
        record_path=record_path,
    )
