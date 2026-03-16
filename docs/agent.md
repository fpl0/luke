# Agent Design

Luke uses the Claude Agent SDK as its runtime. The SDK handles conversation turns, tool dispatch, and context window management. Luke configures it with tools, hooks, sub-agents, and a two-layer system prompt.

## MCP Tools

Tools are built per invocation in a closure that captures `chat_id` and `bot`, scoping each set to a specific chat.

### Categories

**Telegram (15)** — `send_message`, `send_photo`, `send_document`, `send_voice`, `send_video`, `send_location`, `send_poll`, `send_buttons`, `reply`, `forward`, `react`, `get_reactions`, `edit_message`, `delete_message`, `pin`

**Memory (8)** — `remember`, `recall`, `recall_conversation`, `forget`, `connect`, `restore`, `bulk_memory`, `memory_history`

**Scheduling (3)** — `schedule_task`, `list_tasks`, `delete_task`

**Monitoring (1)** — `get_cost_report` (usage stats by period)

### Patterns

- All tools return a standardized `_ok(text)` response
- File-sending tools validate paths against allowed roots via `_safe_path()`
- Tool annotations (`openWorldHint`, `readOnlyHint`, `destructiveHint`) communicate intent to the SDK

## Hooks

Four hooks customize agent behavior at lifecycle points:

**Stop** — When the session ends, prompts the agent to persist what it learned: new entities, episodes, insights, goals, scheduled follow-ups, and a `conversation-state-latest` episode for continuity (topic, mood, unresolved items). Without this hook, learned information is lost when the session ends.

**PreToolUse** — Counts outbound Telegram sends per run and blocks after a threshold (default: 20, overridable per invocation via `max_sends`). Counter is per-invocation. Deep work sessions use `max_sends=1`.

**PreCompact** — When the SDK trims context, instructs the agent to preserve memory IDs, pending actions, active goals, and relationship links in the compaction summary.

**Notification** — Logs SDK notifications to structlog.

## Smart Model Routing

`_classify_effort()` categorizes messages and selects the model tier:

| Effort | Model | Thinking | Use Case |
|--------|-------|----------|----------|
| Low | `settings.model_low` (haiku) | disabled | <15 words, no questions, no media |
| Medium | `settings.model_medium` (sonnet) | adaptive | casual conversation, simple questions |
| High | `settings.model_high` (opus) | enabled (16K budget) | deep reasoning, research, multi-step |

Runs *before* memory injection so injected context doesn't skew the heuristic.

**Post-classification boosts:**
- **Memory-aware boost** — if recalled memories include goals or procedures, model is bumped to at least sonnet
- **Session ratchet** — once a conversation escalates to a higher model, it stays there for the session (resets on session timeout or error)

**Tool scoping** — each tier sees a different tool set. Haiku gets only basic send/recall tools. Sonnet gets most tools (excluding `schedule_task`, `bulk_memory`). Opus gets full access.

## Sub-Agents

Three specialized agents for parallel work:

| Agent | Model | Tools | Purpose |
|-------|-------|-------|---------|
| **researcher** | opus | WebSearch, WebFetch, Read, Grep | Web research and synthesis |
| **coder** | opus | Bash, Read, Write, Edit, Glob, Grep | Code and file operations |
| **memory_curator** | haiku | All memory MCP tools | Consolidation, retagging, linking |

Each sub-agent has a restricted tool set.

## Sessions

Session IDs are stored in the database and passed as `resume` to the SDK. Multi-turn conversations persist across `process()` calls. Stale sessions are cleaned hourly (default: 1 hour timeout). On retry exhaustion, the session is cleared.

## System Prompt

Two layers:

1. **`claude_code` preset** — tool competence (Bash, files, web, sub-agents)
2. **LUKE.md persona** — personality, behavioral guidelines, memory instructions

See [persona](persona.md).

## Conversation Continuity

Each conversation starts with auto-injected context for continuity:

1. **Conversation state** — the Stop hook saves a `conversation-state-latest` episode. On the next message, `_get_conversation_state()` loads it (with staleness marking if >24h old).
2. **Fallback** — if no saved state exists, synthesizes context from the last 10 messages (no LLM call, just formatting).
3. **Concurrent loading** — conversation state, memory recall, and prompt building all run concurrently via `asyncio.gather()`.

## Self-Monitoring

- **Duplicate detection** — `send_long_message()` checks SHA-256 hash against `outbound_log` table before sending (5-min window)
- **Task failure alerting** — scheduler alerts user after 3 consecutive task failures
- **Cost anomaly detection** — logs warning when a single run exceeds 3x the 7-day per-run average (and >$2)
- **Session loss detection** — logs warning when session ID changes unexpectedly

## Key Configuration

The `ClaudeAgentOptions` passed to the SDK include: model (selected per-message via routing), fallback model, tier-scoped `allowed_tools` list, `bypassPermissions` mode, project + user setting sources, dynamic thinking config, turn and budget limits, file checkpointing, and Docker sandbox.

## Message Chunking

`send_long_message()` splits responses at Telegram's 4096-character limit, preferring newline boundaries. Each chunk retries with exponential backoff and falls back to plaintext on HTML parse failure. Duplicate messages are blocked via content hash.
