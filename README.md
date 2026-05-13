# Luke

A personal AI agent that lives on your Mac, talks to you on Telegram, and remembers everything.

One person, one agent. Luke runs as a single Python process: he keeps memory across conversations, executes real work (code, browsing, files, research), schedules his own follow-ups, and pings you when something actually needs your attention.

Fork it and make it yours.

---

## What he does

- **Talks** — text, voice, photos, documents over Telegram
- **Remembers** — persistent long-term memory across sessions
- **Acts** — runs code, browses the web, edits files, spawns sub-agents
- **Schedules** — cron, interval, and one-shot tasks; wakes himself up
- **Routes** — picks Haiku, Sonnet, or Opus per message based on cost and difficulty

## Quick start

```bash
git clone <your-fork-url> && cd luke
claude
```

Then in Claude Code, run:

```
/setup
```

That walks you through Python 3.14, `uv`, dependencies, BotFather, chat ID detection, and the launchd service.

**Requires:** macOS + [Claude Code](https://claude.ai/download).

## Architecture

```
Telegram ──► aiogram ──► Claude Agent SDK ──► response
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         27 MCP tools    Built-in tools   Sub-agents
        (Telegram,        (Bash, Web,     (parallel
         memory,           files, …)       research)
         scheduling)
```

Single Python process. Messages accumulate while the agent is busy and get processed as a batch. At-least-once delivery — the cursor only advances on success.

## Memory

Markdown files with YAML frontmatter, indexed in SQLite FTS5. Five types, each with a different purpose:

```
memory/
├── entities/    # People, projects, concepts — evolve over time
├── episodes/    # Events, decisions, outcomes — accumulate
├── procedures/  # How-to knowledge, reusable scripts — stable
├── insights/    # Patterns, preferences, rules — distilled
└── goals/       # Active objectives with deadlines
```

Eight tools manage it: `remember`, `recall`, `recall_conversation`, `forget`, `connect`, `restore`, `bulk_memory`, `memory_history`. A weekly consolidation pass distills insights and links related memories.

## Project layout

```
luke/
├── pyproject.toml          # 10 dependencies
├── CLAUDE.md               # Instructions for Claude Code
└── src/luke/
    ├── app.py              # Telegram handlers + agent dispatch
    ├── agent.py            # Claude SDK client, MCP tools, model routing
    ├── memory.py           # FTS5 + semantic search + graph + scoring
    ├── db.py               # SQLite: messages, sessions, tasks, costs
    ├── scheduler.py        # Cron / interval / one-shot tasks
    ├── behaviors.py        # Consolidation, reflection, deep work
    └── media.py            # Images, video frames, Whisper transcription

$LUKE_DIR (~/.luke)/        # All runtime data
├── LUKE.md                 # Persona and behavior guide
├── context.yaml            # User context
├── luke.db                 # SQLite database
├── memory/                 # Long-term memory
└── workspace/              # Media, apps, scripts Luke builds
```

## Skills

Claude Code skills guide common changes:

| Skill        | Purpose                          |
| ------------ | -------------------------------- |
| `/setup`     | First-time installation          |
| `/upgrade`   | Pull latest, sync deps, restart  |
| `/customize` | Add capabilities, change behavior |
| `/debug`     | Troubleshooting                  |

## Development

```bash
uv run luke                     # Run
uv run ruff check src/          # Lint
uv run ruff format src/         # Format
uv run mypy --strict src/luke/  # Type check
uv run pytest                   # Test
```

## Built on

- [Claude Agent SDK](https://github.com/anthropics/claude-code-sdk-python) — agent loop
- [aiogram](https://docs.aiogram.dev/) — Telegram bot framework
- Inspired by [NanoClaw](https://github.com/qwibitai/nanoclaw) — small, auditable, forkable agents

## License

[MIT](LICENSE)
