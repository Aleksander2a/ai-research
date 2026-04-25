"""Project-wide configuration: paths, environment, runtime flags.

Loads `.env` automatically if present. All paths are absolute and resolved
relative to the repo root (computed from this file's location), so the package
behaves the same whether invoked from the repo root, a worktree, or an
installed venv."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env", override=False)


class Settings(BaseSettings):
    """Runtime configuration sourced from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    reports_dir: Path = REPO_ROOT / "reports"
    replay_dir: Path = REPO_ROOT / "replay"
    configs_dir: Path = REPO_ROOT / "configs"

    deepinfra_api_key: str = Field(
        default="",
        description="DeepInfra API key for the agentic layer.",
    )
    deepinfra_model_proposer: str = ""
    deepinfra_model_critic: str = ""
    deepinfra_model_chat: str = ""
    deepinfra_base_url: str = "https://api.deepinfra.com/v1/openai"

    autosignalx_replay: bool = Field(
        default=False,
        description="Force the agentic layer into deterministic replay mode.",
    )

    @property
    def use_replay(self) -> bool:
        """True if the agent layer should play back recorded traces instead of
        making live LLM calls. Triggers when no API key is set or when replay
        mode is explicitly forced."""
        return self.autosignalx_replay or not self.deepinfra_api_key


settings = Settings()
