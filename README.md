# Luke

Most AI tools wait for you to type something and then respond. Luke doesn't wait. He remembers what you talked about last week, researches things while you sleep, schedules your reminders, builds tools when you need them, and messages you on Telegram when something needs your attention.

Luke is a personal agent — not a chatbot, not a framework, not a platform. One person, one agent, running on a Mac. If you want your own, fork it.

Built on the [Claude Agent SDK](https://github.com/anthropics/claude-code-sdk-python) and [aiogram](https://docs.aiogram.dev/). Heavily inspired by [NanoClaw](https://github.com/qwibitai/nanoclaw) — same philosophy of small, auditable agents designed to be forked. Luke just scopes it down to one person.

## Why

I wanted something that actually *does* things. Not "here's a summary of what you could do" — actually does them. Search for flights and tell me which one to book. Draft the email and show it to me before sending. Notice I have a meeting tomorrow and prep me for it.

The gap between what AI can do and what AI actually does in your daily life is enormous. Luke is an attempt to close that gap for one person.

## What it looks like

You talk to Luke on Telegram. Text, voice messages, photos, documents — whatever. Luke responds, but also acts: runs code, browses the web, manages files, and spawns sub-agents for heavier research. Everything runs locally on your Mac.

A few real examples of things Luke does:

- Morning briefing with world news, AI updates, and a dream journal prompt
- Researched companies before discovery calls and wrote briefing docs
- Built a web dashboard and served it through a Cloudflare tunnel as a Telegram Mini App
- Scheduled reminders, tracked goals, prepped conversation strategies
- Transcribed voice messages with Whisper and responded to the content

## Setup

```bash
git clone <your-fork-url> && cd luke
claude
```

Then tell Claude: `/setup`

That's it. Claude handles Python 3.14, uv, dependencies, BotFather walkthrough, chat ID detection, and launchd service installation.

**Requires:** macOS and [Claude Code](https://claude.ai/download). Everything else is installed for you.

## How it works

Single Python process. Telegram messages come in through aiogram, get dispatched to the Claude Agent SDK, and responses go back through Telegram. An in-process MCP server exposes 27 tools. Smart model routing picks haiku, sonnet, or opus depending on the message.

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

Messages accumulate while the agent is busy and get processed as a batch — like a cursor. At-least-once delivery: the cursor only advances after a successful run. If something fails, the user gets an error and messages retry next time.

## Memory

Luke remembers things in markdown files with YAML frontmatter, indexed in SQLite FTS5. Five categories, each with a different purpose:

```
memory/
├── entities/      # People, projects, concepts — evolve over time
├── episodes/      # Events, decisions, outcomes — accumulate
├── procedures/    # How-to knowledge, reusable scripts — stable
├── insights/      # Patterns, preferences, rules — distilled
└── goals/         # Active objectives with deadlines and progress
```

Eight tools manage memory: `remember`, `recall`, `recall_conversation`, `forget`, `connect`, `restore`, `bulk_memory`, `memory_history`. A weekly consolidation reviews conversations, distills insights, and prunes stale data. Nothing is auto-deleted.

## Project structure

```
luke/
├── pyproject.toml           # 10 dependencies
├── CLAUDE.md                # Instructions for Claude Code
├── .env                     # Telegram token + Claude auth (gitignored)
├── src/luke/
│   ├── app.py               # Telegram handlers + agent dispatch
│   ├── agent.py             # Claude SDK client + 27 MCP tools + model routing
│   ├── memory.py            # Memory: FTS5 indexing, semantic search, graph, scoring
│   ├── db.py                # SQLite: messages, sessions, tasks, cost tracking
│   ├── config.py            # pydantic-settings from .env
│   ├── scheduler.py         # Cron/interval/once task execution
│   ├── behaviors.py         # Consolidation, reflection, proactive scan, deep work
│   ├── media.py             # Image encoding, video frames, whisper transcription
│   └── templates/
│       └── LUKE.md          # Default persona template
└── $LUKE_DIR (~/.luke)/     # All runtime data (configurable via .env)
    ├── LUKE.md              # Agent persona and behavior guide
    ├── context.yaml         # User-specific context
    ├── luke.db              # SQLite database
    ├── memory/              # Structured long-term memory
    └── workspace/           # Media, apps, scripts Luke builds
```

## Making it yours

No configuration files to learn. If you want different behavior, change the code. The codebase is small enough that this is practical.

Claude Code skills guide common changes:

```
/setup      — First-time installation
/upgrade    — Pull latest, sync deps, restart
/customize  — Add capabilities, change behavior
/debug      — Troubleshooting
```

## Development

```bash
uv run luke                     # Run
uv run ruff check src/          # Lint
uv run ruff format src/         # Format
uv run mypy --strict src/luke/  # Type check
uv run pytest                   # Test
```

## Acknowledgments

Luke builds on [NanoClaw](https://github.com/qwibitai/nanoclaw) by [Qwibit](https://github.com/qwibitai). The idea of small, auditable, AI-native agents designed to be forked and customized is theirs. Luke just takes that idea and points it at one person's life.

## License

[MIT](LICENSE)
