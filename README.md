# Luke

A personal agent that remembers, researches, builds, decides, and acts — powered by the Claude Agent SDK, talking to you over Telegram.

Heavily inspired by [NanoClaw](https://github.com/qwibitai/nanoclaw). Where NanoClaw is a multi-channel agent platform with container sandboxing, Luke strips the idea down to a single-user agent focused on simplicity. Same philosophy — small enough to understand, designed to be forked and customized — just scoped to one person.

## What Luke Can Do

- **Act** — does the work, doesn't describe it. "Find flights to Rome" → searches, compares, sends results
- **Execute code** — Python, shell, anything via Claude Code's sandboxed Bash
- **Browse the web** — search, fetch pages, extract data
- **Research** — spawn sub-agents that search in parallel, synthesize findings
- **Automate** — scheduled tasks via cron, interval, or one-time triggers; autonomous deep work on goals
- **Remember** — structured long-term memory with FTS5 search, relationship graphs, and temporal queries
- **Communicate** — photos, documents, voice, video, polls, inline keyboards, reactions, location sharing

## Setup

```bash
git clone <your-fork-url> && cd luke
claude
```

Then tell Claude: `/setup`

Claude handles everything: Python 3.14, uv, dependencies, BotFather walkthrough, chat ID detection, launchd service installation.

**Requirements:** macOS + [Claude Code](https://claude.ai/download). Everything else is installed automatically.

## Architecture

Single Python process: [aiogram](https://docs.aiogram.dev/) dispatches Telegram messages → Claude Agent SDK powers the agent → responds via Telegram. An in-process MCP server exposes 27 tools. Smart model routing selects haiku/sonnet/opus per message. SQLite stores everything.

```
Telegram → aiogram dispatcher → Claude Agent SDK → responds
                                       │
                         ┌─────────────┼─────────────┐
                         │             │             │
                    27 MCP tools   Built-in tools  Sub-agents
                    (Telegram,     (Bash, Web,     (parallel
                     memory,       files, etc.)    research)
                     scheduling)
```

Key design choices:
- **Cursor model** — messages accumulate, agent processes all pending as one batch
- **At-least-once delivery** — cursor advances only after successful agent run
- **Per-chat locking** — `asyncio.Lock` per chat + global `asyncio.Semaphore` (default: 5)
- **Error recovery** — agent failure sends error to user, messages retry next dispatch

## Memory System

Memories are markdown files with YAML frontmatter, indexed in SQLite FTS5:

```
luke/memory/
├── entities/      # People, projects, concepts (evolve over time)
├── episodes/      # Events, decisions, outcomes (accumulate)
├── procedures/    # How-to knowledge, reusable scripts (stable)
├── insights/      # Patterns, preferences, rules (distilled)
└── goals/         # Active objectives with deadlines and progress
```

Luke manages memory through 8 tools: `remember`, `recall`, `recall_conversation`, `forget`, `connect`, `restore`, `bulk_memory`, and `memory_history`. A weekly consolidation task reviews conversations, distills insights, and prunes stale data. Nothing is auto-deleted — old memories persist and remain searchable.

## Project Structure

```
luke/
├── pyproject.toml           # 10 dependencies
├── CLAUDE.md                # Instructions for Claude Code
├── .env                     # Telegram token + Claude auth (gitignored)
├── src/luke/
│   ├── app.py               # Telegram handlers + agent dispatch
│   ├── agent.py             # Claude SDK client + 27 MCP tools + model routing
│   ├── db.py                # SQLite: messages, sessions, memory FTS5
│   ├── config.py            # pydantic-settings from .env
│   ├── scheduler.py         # Cron/interval/once task execution
│   ├── behaviors.py         # Consolidation, reflection, proactive scan, deep work
│   ├── media.py             # Image encoding, video frames, whisper transcription
│   └── templates/
│       └── LUKE.md          # Default persona template (seeded to LUKE_DIR on first run)
└── $LUKE_DIR (~/.luke)/     # All runtime data (configurable via .env)
    ├── LUKE.md              # Agent persona and behavior guide
    ├── context.yaml         # User-specific context
    ├── luke.db              # SQLite database
    ├── memory/              # Structured long-term memory
    └── workspace/           # Media files, apps, scripts Luke builds
```

## Customization

No configuration sprawl. If you want different behavior, modify the code. The codebase is small enough that this is safe and practical. Claude Code skills guide changes:

```
/setup      — First-time installation and configuration
/upgrade    — Pull latest code, sync deps, restart service
/customize  — Add capabilities, change behavior
/debug      — Troubleshooting and diagnostics
```

## Development

```bash
uv run luke                     # Run directly
uv run ruff check src/          # Lint
uv run ruff format src/         # Format
uv run mypy --strict src/luke/  # Type check
uv run pytest                   # Test
```

## Acknowledgments

Luke is heavily inspired by [NanoClaw](https://github.com/qwibitai/nanoclaw) by [Qwibit](https://github.com/qwibitai). NanoClaw's philosophy of small, auditable, AI-native agents designed to be forked and customized is the foundation this project builds on.

## License

[MIT](LICENSE)
