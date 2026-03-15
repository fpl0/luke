from __future__ import annotations

import os
import sys
from functools import cached_property
from pathlib import Path

from pydantic import SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    assistant_name: str = "Luke"
    telegram_bot_token: SecretStr = SecretStr("")
    chat_id: str = ""
    max_concurrent: int = 5
    scheduler_interval: float = 60.0
    recall_content_limit: int = 2000
    luke_dir: Path = Path.home() / ".luke"  # user-configurable via LUKE_DIR env var

    # Timeouts
    transcription_timeout: float = 300.0  # max seconds for audio transcription
    ffmpeg_timeout: float = 30.0  # max seconds for frame extraction

    # Media processing
    max_images_per_prompt: int = 5  # max images sent to Claude per turn

    # Auto-recall injection
    auto_recall_limit: int = 5
    auto_recall_cache_ttl: float = 300.0  # 5 min cache for embedding search

    # Composite scoring weights (must sum to 1.0)
    score_weight_relevance: float = 0.4
    score_weight_importance: float = 0.25
    score_weight_recency: float = 0.2
    score_weight_access: float = 0.15

    # Scoring internals
    recency_decay_days: float = 30.0  # half-life for recency scoring
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    max_consolidation_clusters: int = 3  # max clusters per consolidation run
    utility_floor: float = 0.7  # minimum fraction of access_score at 0% utility
    utility_weight: float = 0.3  # how much utility_rate can boost above floor

    # Adaptive forgetting decay rates per memory type (hourly)
    decay_rate_entity: float = 0.9998
    decay_rate_episode: float = 0.999
    decay_rate_procedure: float = 0.9999
    decay_rate_insight: float = 0.9995
    decay_rate_goal: float = 0.9997

    # Consolidation
    consolidation_interval: float = 86400.0  # daily (seconds)
    consolidation_min_cluster: int = 2

    # FTS retention
    fts_retention_days: int = 1825  # 5 years; prune low-importance episodes older than this

    # Reflection + proactive scan + goal execution
    reflection_interval: float = 604800.0  # weekly (seconds)
    proactive_scan_interval: float = 86400.0  # daily (seconds)
    goal_execution_interval: float = 43200.0  # every 12 hours (seconds)

    # Agent processing
    agent_timeout: float = 1800.0  # max seconds per agent run (30 min)
    max_retries: int = 3  # max agent failures before skipping messages
    agent_max_turns: int = 200  # safety net: 3-5x a complex task
    agent_max_budget_usd: float = 5.0  # cost ceiling per invocation
    behavior_max_turns: int = 75  # behaviors are focused single-purpose tasks
    behavior_max_budget_usd: float = 1.0
    agent_model: str = "opus"
    agent_fallback_model: str = "sonnet"
    max_sends_per_run: int = 20  # rate-limit outbound Telegram messages per agent run

    # Telegram retry
    telegram_send_retries: int = 3
    telegram_retry_base_delay: float = 1.0  # seconds, exponential backoff

    # Sessions
    session_timeout: float = 3600.0  # stale session cleanup threshold (seconds)

    # Maintenance
    cleanup_interval: float = 3600.0  # FTS cleanup + importance decay (seconds)
    error_cooldown: float = 30.0  # min seconds between error messages per chat
    db_busy_timeout: int = 5000  # SQLite busy_timeout in ms

    # Graph traversal
    graph_max_depth: int = 2
    graph_decay_per_hop: float = 0.5

    @field_validator("telegram_bot_token")
    @classmethod
    def _check_token(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value():
            raise ValueError("missing")
        return v

    @field_validator(
        "decay_rate_entity",
        "decay_rate_episode",
        "decay_rate_procedure",
        "decay_rate_insight",
        "decay_rate_goal",
    )
    @classmethod
    def _check_decay_rate(cls, v: float) -> float:
        if not (0 < v < 1):
            raise ValueError(f"decay rate must be in (0, 1), got {v}")
        return v

    @model_validator(mode="after")
    def _check_score_weights(self) -> Settings:
        total = (
            self.score_weight_relevance
            + self.score_weight_importance
            + self.score_weight_recency
            + self.score_weight_access
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"scoring weights must sum to 1.0, got {total}")
        return self

    @cached_property
    def workspace_dir(self) -> Path:
        return self.luke_dir / "workspace"

    @cached_property
    def memory_dir(self) -> Path:
        return self.luke_dir / "memory"

    @cached_property
    def store_dir(self) -> Path:
        return self.luke_dir

    @cached_property
    def decay_rates(self) -> dict[str, float]:
        return {
            "entity": self.decay_rate_entity,
            "episode": self.decay_rate_episode,
            "procedure": self.decay_rate_procedure,
            "insight": self.decay_rate_insight,
            "goal": self.decay_rate_goal,
        }


def _load_dotenv() -> dict[str, str]:
    """Parse .env file and return key-value pairs."""
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    env_vars: dict[str, str] = {}
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip().strip("'\"")
    return env_vars


# Env vars the claude CLI subprocess needs but that aren't Settings fields
_PASSTHROUGH_VARS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_AUTH_TOKEN",
    }
)


def _export_env(env_vars: dict[str, str]) -> None:
    """Export credential vars to os.environ so the claude subprocess inherits them.

    pydantic-settings reads .env for its own fields but does NOT set os.environ.
    The Claude Agent SDK spawns a claude subprocess that needs these credentials.
    """
    for key, value in env_vars.items():
        if key in _PASSTHROUGH_VARS and key not in os.environ:
            os.environ[key] = value


def _check_env() -> None:
    """Print friendly errors for missing credentials and exit."""
    env_vars = _load_dotenv()

    missing: list[str] = []

    # Claude credentials (used by claude_agent_sdk, not in Settings)
    has_claude = (
        env_vars.get("ANTHROPIC_API_KEY")
        or env_vars.get("CLAUDE_CODE_OAUTH_TOKEN")
        or env_vars.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    if not has_claude:
        missing.append(
            "  Claude credentials not found.\n\n"
            "  Claude subscription (Pro/Max):\n"
            "    1. Run: claude setup-token\n"
            "    2. Copy the token and add to .env:\n"
            "       CLAUDE_CODE_OAUTH_TOKEN=your-token-here\n\n"
            "  Anthropic API key:\n"
            "    Add to .env: ANTHROPIC_API_KEY=sk-ant-..."
        )

    # Telegram token
    if not env_vars.get("TELEGRAM_BOT_TOKEN") and not os.environ.get("TELEGRAM_BOT_TOKEN"):
        missing.append(
            "  Telegram bot token not found.\n\n"
            "  1. Message @BotFather on Telegram and create a bot\n"
            "  2. Add to .env: TELEGRAM_BOT_TOKEN=your-token-here"
        )

    if missing:
        print("\n" + "\n\n".join(missing), file=sys.stderr)
        print("\n  Or run: claude /setup\n", file=sys.stderr)
        sys.exit(1)


_export_env(_load_dotenv())

try:
    settings = Settings()
except ValidationError:
    # Show friendly error messages before re-raising
    _check_env()
    raise
