# Luke

> **Archived:** Luke has been superseded by [Theo](https://github.com/fpl0/theo), the new version of this project. Please use Theo for continued development and updates.

A personal AI agent that lives on your Mac, talks to you on Telegram, and remembers everything.

One person, one agent. Luke runs as a single Python process: he keeps memory across conversations,
does real work (code, browsing, files, research), schedules his own follow-ups, and pings you only
when something actually needs you.

Fork it and make it yours.

---

## What he does

- **Talks** — text, voice, photos, documents over Telegram
- **Remembers** — persistent long-term memory across sessions
- **Acts** — runs code, browses the web, edits files, spawns sub-agents
- **Schedules** — cron, interval, and one-shot tasks; wakes himself up
- **Routes** — picks Haiku, Sonnet, or Opus per message based on cost and difficulty
- **Improves himself** — analyzes his own failures, writes the lesson down, and patches his own code

## Quick start

```bash
git clone <your-fork-url> && cd luke
claude
```

Then in Claude Code, run:

```
/setup
```

That walks you through Python 3.14, `uv`, dependencies, BotFather, chat ID detection, and the
launchd service.

**Requires:** macOS + [Claude Code](https://claude.ai/download).

## Architecture

```
Telegram ──► aiogram ──► Claude Agent SDK ──► response
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         33 MCP tools    Built-in tools   Sub-agents
        (Telegram,        (Bash, Web,     (parallel
         memory,           files, …)       research)
         scheduling)
```

Single Python process. Messages that arrive while the agent is busy accumulate and get handled as
one batch. Delivery is at-least-once: the cursor only advances on success.

## Memory

Markdown files with YAML frontmatter, indexed in SQLite FTS5. Five types, each with its own job:

```
memory/
├── entities/    # People, projects, concepts — evolve over time
├── episodes/    # Events, decisions, outcomes — accumulate
├── procedures/  # How-to knowledge, reusable scripts — stable
├── insights/    # Patterns, preferences, rules — distilled
└── goals/       # Active objectives with deadlines
```

Nine tools manage it: `remember`, `recall`, `recall_conversation`, `forget`, `connect`, `restore`,
`bulk_memory`, `memory_history`, and `review_corrections`. A weekly consolidation pass distills
insights and links related memories.

## Self-improvement

When something breaks, Luke is expected to fix the cause, not log it. A failed task, a claim that
turned out to be wrong, a session that landed badly — each is an event that starts a loop, and the
loop is only finished when code changed.

| Loop                 | Fires on             | Produces                                       |
| -------------------- | -------------------- | ---------------------------------------------- |
| **Reflexion**        | A specific failure   | Root cause → an insight → usually a patch      |
| **Reflection**       | Weekly               | Patterns across feedback, distilled into rules |
| **Dream**            | Idle time            | Defects found by cross-reading memory          |
| **Skill extraction** | Repeated manual work | A reusable procedure or script                 |
| **Plan momentum**    | A stalled goal       | Unblocks it, or marks it honestly as blocked   |

Three things keep that from becoming journalling:

- **A lesson has to graduate into a mechanism.** Prose gets forgotten the next time it matters. A
  repeat failure is supposed to end up as a test, a gate, or a script that runs — not a paragraph.
- **Circuit-breakers run before the analysis.** Reflexion won't spawn on an empty payload or a
  failure mode it has already saturated, so it can't burn tokens relearning what it knows.
- **Sessions are graded 1-5 on whether the work landed**, not on how much was produced. Below 1.5
  pauses the goal: something is structurally wrong and retrying is the wrong move.

Luke also runs several sessions at once, so each one starts by being told which workspace files a
sibling touched in the last 20 minutes — and is steered onto different work rather than racing it.

## Project layout

```
luke/
├── pyproject.toml          # 12 dependencies
├── CLAUDE.md               # Instructions for Claude Code
└── src/luke/
    ├── app.py              # Telegram handlers + agent dispatch
    ├── agent.py            # Claude SDK client, MCP tools, model routing
    ├── memory.py           # FTS5 + semantic search + graph + scoring
    ├── db.py               # SQLite: messages, sessions, tasks, costs
    ├── scheduler.py        # Cron / interval / one-shot tasks
    ├── behaviors.py        # Deep work, reflexion, dreams, consolidation
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

| Skill        | Purpose                            |
| ------------ | ---------------------------------- |
| `/setup`     | First-time installation            |
| `/upgrade`   | Pull latest, sync deps, restart    |
| `/customize` | Add capabilities, change behavior  |
| `/debug`     | Troubleshooting                    |

## Development

```bash
uv run luke                     # Run
uv run ruff check src/ tests/   # Lint
uv run ruff format src/ tests/  # Format
uv run mypy --strict src/luke/  # Type check
uv run pyright src/             # Type check (second opinion)
uv run pytest                   # Test
```

All five must pass before a commit. They are gates, not suggestions — if one fails, fix the cause
rather than silencing it.

## Built on

- [Claude Agent SDK](https://github.com/anthropics/claude-code-sdk-python) — agent loop
- [aiogram](https://docs.aiogram.dev/) — Telegram bot framework
- Inspired by [NanoClaw](https://github.com/qwibitai/nanoclaw) — small, auditable, forkable agents

## License

[MIT](LICENSE)
