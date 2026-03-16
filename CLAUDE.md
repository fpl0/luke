# Luke

Personal agent on Telegram — remembers, researches, builds, decides, acts. Nine files, one process, no frameworks. See [README.md](README.md) for philosophy and setup.

## Opinionated Design

Luke is an opinionated agent implementation, not a framework. Every architectural decision is intentional and made for coherence.

**Hard rules:**
- **No optional dependencies.** All deps are required: fastembed, sqlite-vec, mlx-whisper, Pillow. No "if available" checks, no graceful degradation for missing packages. If it's in `pyproject.toml`, assume it works.
- **Platform constants are constants, not settings.** Telegram's 4096-char limit, Anthropic's 1568px max, the embedding model name — these are facts about the environment, not tuning knobs. They live as module-level constants, not in `config.py`.
- **Settings are for user-specific values only.** `config.py` holds things the user genuinely controls: tokens, chat ID, data directory (`LUKE_DIR`), timeouts, scoring weights, batch sizes. Not library versions or platform constraints.
- **Runtime errors return None; missing deps crash.** A corrupt image fails to encode → `None` (runtime error, expected). A missing import → crash at startup (broken install). Never conflate the two.
- **No speculative abstractions.** Don't add configurability for hypothetical future needs. If there's one sensible value, hardcode it. If someone needs different behavior, they change the code — the codebase is small enough.
- **Comments match behavior.** No "best effort" or "graceful fallback" language for required functionality. If something is required, say so. If an operation can fail at runtime, document why.

## Quick Context

Single Python process: aiogram dispatches Telegram messages → Claude Agent SDK powers the agent → responds via Telegram. In-process MCP server provides 27 tools (Telegram, memory, scheduling, monitoring). Smart model routing selects haiku/sonnet/opus per message based on complexity. SQLite stores everything. Python 3.14+, managed with uv.

## Key Files

**Source (in repo):**

| File | Purpose |
|------|---------|
| `src/luke/app.py` | Orchestrator: Telegram handlers, `_store` helper, `process` + `_dispatch`, main loop |
| `src/luke/agent.py` | Claude SDK client + 27 MCP tools (Telegram, memory, scheduling, monitoring) + model routing |
| `src/luke/db.py` | SQLite: messages, sessions, tasks, memory index (FTS5) |
| `src/luke/config.py` | Settings from `.env` via pydantic-settings (`SecretStr` for token) |
| `src/luke/scheduler.py` | Cron/interval/once task execution + hourly maintenance |
| `src/luke/behaviors.py` | Autonomous behaviors: consolidation, reflection, proactive scan, deep work |
| `src/luke/media.py` | Image encoding, video frames, whisper transcription, multimodal prompt building |
| `src/luke/__main__.py` | Entry point: `python -m luke` |

**Data (in `$LUKE_DIR`, default `~/.luke`):**

| File | Purpose |
|------|---------|
| `LUKE.md` | Agent persona and behavior (system prompt) |
| `luke.db` | SQLite database (messages, sessions, tasks, memory index) |
| `memory/` | Markdown memory files with YAML frontmatter |
| `workspace/` | Media files, apps, scripts Luke builds, deep work plans (`plans/`) |
| `context.yaml` | User-specific context (name, timezone) |

## Skills

| Skill | When to Use |
|-------|-------------|
| `/setup` | First-time installation, Telegram auth, register main chat, start service |
| `/upgrade` | Pull latest code, sync deps, apply migrations, update persona, restart service |
| `/customize` | Adding capabilities, changing behavior, new integrations |
| `/debug` | Troubleshooting: logs, environment, Telegram issues, errors |

## Development

Run commands directly — don't tell the user to run them.

```bash
uv run luke              # Run directly
uv run ruff check src/ tests/   # Lint
uv run ruff format src/ tests/  # Format
uv run mypy --strict src/luke/ tests/  # Type check (strict)
uv run pyright src/ tests/  # Pylance/Pyright check (strict)
uv run pytest            # Test
```

Service management:
```bash
# macOS (launchd)
launchctl load ~/Library/LaunchAgents/com.luke.plist
launchctl unload ~/Library/LaunchAgents/com.luke.plist
launchctl kickstart -k gui/$(id -u)/com.luke  # restart

# View logs (default LUKE_DIR=~/.luke)
tail -f ~/.luke/luke.log
tail -f ~/.luke/luke.err
```

## Type Safety

- **mypy strict** — all source and test files pass `mypy --strict` with zero errors
- **Pyright strict** — all source and test files pass with zero errors (Pylance in VS Code)
- **Always run both** — `uv run mypy --strict src/luke/ tests/ && uv run pyright src/ tests/`
- **ruff** — lint + format on save via VS Code settings
- **pydantic plugin** — `plugins = ["pydantic.mypy"]` in pyproject.toml
- **`from __future__ import annotations`** — all files use deferred evaluation
- **`SecretStr`** for `telegram_bot_token` — prevents accidental logging/serialization

## Architecture

- **No containers** — Claude Agent SDK with Docker Sandboxes for tool isolation
- **No polling** — aiogram event-driven dispatch via long-polling
- **No abstractions** — module-level handlers, direct bot calls
- **Concurrency** — `asyncio.Lock` per chat + `asyncio.Semaphore` global limit (default: 8); `_behavior_sem` (max: 3) caps concurrent behaviors, guaranteeing ≥5 slots for user messages; deep work runs as a background `asyncio.Task` (non-blocking)
- **Memory** — Hybrid FTS5 + semantic (fastembed) search, relationship graph with multi-hop traversal, composite scoring (relevance × importance × recency × access), adaptive forgetting, auto-injection, and periodic consolidation
- **Cursor model** — messages accumulate, agent processes all pending as one batch
- **Startup replay** — on restart, pending messages from before crash are dispatched immediately
- **At-least-once delivery** — cursor advances only after successful agent run; on error, messages stay pending for retry (up to `max_retries` attempts, then skipped with user notification)
- **Error recovery** — transient failures retry automatically with exponential backoff; session preserved across retries; retry counter resets on process restart (gives messages fresh attempts)

## Security

- **Path traversal** — Telegram filenames sanitized with `Path(...).name`; memory IDs sanitized via `sanitize_memory_id()` in db.py; memory types validated against allowlist
- **Duplicate messages** — `send_long_message()` checks SHA-256 content hash against `outbound_log` table (5-min window) before sending; hourly cleanup prunes old entries
- **FTS5 injection** — `recall()` catches `sqlite3.OperationalError` from malformed MATCH queries
- **Schedule validation** — `create_task()` validates type is `cron|interval|once`, validates cron with `croniter.is_valid()`, validates interval is integer
- **Bot token** — `SecretStr` in pydantic, `.get_secret_value()` only at `Bot()` construction
- **SQL** — all queries use parameterized `?` placeholders (the one f-string in `recall()` temporal branch only interpolates hardcoded condition strings, never user input)

## Testing

Always write tests for new functionality. Test files live in `tests/` and mirror the source structure:
- `tests/test_db.py` — message storage, sessions, tasks
- `tests/test_db_memory.py` — memory indexing, recall, FTS, embeddings
- `tests/test_media.py` — image encoding, transcription, ffmpeg
- `tests/test_behaviors.py` — autonomous behavior functions
- `tests/test_scheduler.py` — scheduler loop, task execution
- `tests/test_app.py` — Telegram handlers, process() flow
- `tests/test_agent.py` — MCP tools, send_long_message
- `tests/test_config.py` — settings validation
- `tests/conftest.py` — shared fixtures (tmp_settings, test_db)

When implementing new features, add tests that cover the happy path + edge cases.

## Documentation

Docs live in `docs/` and explain design decisions, architecture, and behavior in depth. When making changes:
- **Update existing docs** if a change affects behavior they describe (e.g., changing the memory system → update `docs/memory.md`)
- **Add new docs** when introducing significant new functionality or architectural concepts that warrant explanation beyond code comments
- Docs are for *why* and *how it works* — not API reference (the code is small enough to read directly)

## Tools (27 MCP tools in `agent.py`)

**Telegram (15):** send_message, send_photo, send_document, send_voice, send_video, send_location, send_poll, send_buttons, reply, forward, react, get_reactions, edit_message, delete_message, pin

**Memory (8):** remember, recall, recall_conversation, forget, connect, restore, bulk_memory, memory_history

**Scheduling (3):** schedule_task, list_tasks, delete_task

**Monitoring (1):** get_cost_report

**Tool scoping by model tier:**
- **Haiku** — send_message, reply, react, remember, recall, recall_conversation (+ Read, Glob, Grep builtins only)
- **Sonnet** — all except schedule_task, bulk_memory (+ all builtins)
- **Opus** — everything

**Plus all Claude Code built-in tools:** Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch, Agent teams, TodoWrite, etc.

## Memory System

Memories are markdown files with YAML frontmatter in `$LUKE_DIR/memory/`. Indexed in SQLite FTS5 for search. Five types:
- **entities/** — people, projects, concepts (evolve over time)
- **episodes/** — events, decisions, outcomes (accumulate, include reasoning chains)
- **procedures/** — how-to knowledge, reusable scripts (stable, updated)
- **insights/** — patterns, preferences, rules (distilled)
- **goals/** — active objectives with deadlines and progress tracking (updated on progress)

Key behaviors:
- **Hybrid retrieval** — FTS5 (lexical, OR semantics) + fastembed (semantic) search, merged via Reciprocal Rank Fusion
- **Composite scoring** — relevance gates context quality (importance × recency × access × utility); non-query results dampened; importance clamped to [0,1] in scoring
- **Utility tracking** — distinguishes intentional access (agent tools) from speculative (auto-injection); utility rate modulates access score to penalize frequently-surfaced-but-unused memories
- **Multi-hop graph** — BFS up to depth 2 with exponential weight decay per hop; Hebbian co-access strengthening evolves link weights over time
- **Adaptive forgetting** — hourly type-aware importance decay modulated by access count (spaced repetition)
- **Auto-injection** — `process()` recalls relevant memories + 1-hop graph neighbors concurrently with prompt building, plus conversation state injection for continuity
- **Conflict detection** — entity updates detect and report changes; history recorded in `memory_history` table
- **Consolidation** — daily scheduler task clusters related episodes (≥2 shared tags) and synthesizes insights via agent
- **Self-healing** — `sync_memory_index()` on startup detects unindexed files and indexes them (with embeddings)
- **Recall limit** — configurable via `RECALL_CONTENT_LIMIT` env var (default: 2000 chars)
- **Embeddings** — `fastembed` + `sqlite-vec` (both required); hybrid FTS + semantic search always active
- **Voice transcription** — `mlx-whisper` (required); configurable model via `WHISPER_MODEL` env var (default: `mlx-community/whisper-large-v3-turbo`). Transcripts saved as `.txt` alongside `.ogg` files

## Message Storage

Messages table stores `sender_name`, `sender_id` (Telegram user ID), `msg_id` (Telegram message ID), `content`, `timestamp`, `reply_to`, and `media_file_id`. The prompt format includes `msg:{id}` so the agent can use the `reply` tool to respond to specific messages.

## Scheduler

The scheduler loop runs every 60 seconds (configurable via `SCHEDULER_INTERVAL`). It:
1. Runs hourly maintenance: archived FTS cleanup + adaptive importance decay + stale session cleanup + outbound log pruning
2. Collects due maintenance behaviors and runs them **in parallel** via `asyncio.gather()`:
   - Daily: consolidation (episodes → insights) + proactive scan (can take lightweight goal actions)
   - Weekly: reflection + FTS pruning
3. Launches deep work as a **background task** (non-blocking, budget-limited to `daily_deep_work_budget_usd`):
   - Every 8h: autonomous multi-step goal execution with plan-before-execute pattern
   - Plans stored in `$LUKE_DIR/workspace/plans/{goal_id}.md` with status tracking
   - Rate-limited to 1 outbound message per session
4. Fetches active tasks (excludes completed `once` tasks at SQL level)
5. Checks due-ness via `_is_due()` (handles cron, interval, once with timezone-aware datetimes)
6. Deduplicates — won't fire a task that's still running from a previous tick
7. Persists behavior timestamps to DB (`behavior_state` table) — survives restarts
8. Alerts user after 3 consecutive task failures

## Data Separation

Code lives in the repo (`src/`). Data lives in `$LUKE_DIR` (default `~/.luke`), configurable via the `LUKE_DIR` env var in `.env`. The user chooses where data goes — could be local, on an external drive, etc. LUKE.md persona is seeded from the repo template if absent.

## Customization = Code Changes

No configuration sprawl. If you want different behavior, modify the code. The codebase is small enough (~3950 lines) that this is safe and practical. Skills (`/customize`) guide these changes.
