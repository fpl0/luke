# Opinionated Minimalism

Luke is an opinionated agent implementation, not a framework.

## No Optional Dependencies

All dependencies in `pyproject.toml` are required. There are no `try/except ImportError` patterns, no "if available" checks, no graceful degradation for missing packages.

A broken install crashes at startup with a clear import error rather than silently losing features.

**Runtime errors return `None`. Missing deps crash.** A corrupt image fails to encode → `None`. A missing import → crash. Different failure categories, different handling.

## Platform Constants vs Settings

Telegram's 4096-character message limit, Anthropic's 1568px max image dimension — these are facts about the environment. They live as module-level constants in the files that use them (e.g. `_TG_MAX_MSG_LEN` in `agent.py`, `_MAX_IMAGE_DIM` in `media.py`).

`config.py` holds values the user genuinely controls: bot token, chat ID, concurrency limits, scoring weights, timeouts.

The boundary: if changing a value would break a platform contract, it's a constant. If it's a preference, it's a setting.

## No Speculative Abstractions

No plugin system, no provider abstraction, no config for hypothetical features, no abstract base classes.

Luke uses Telegram, Claude, and SQLite directly — no abstraction layers. To use a different messaging platform, rewrite the handlers.

## Customization = Code Changes

The codebase is designed to be forked. No configuration sprawl, no YAML feature flags. Claude Code skills (`/setup`, `/customize`, `/debug`) guide modifications.
