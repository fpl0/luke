# Agent Design

Luke uses the Claude Agent SDK as its runtime. The SDK handles conversation turns, tool dispatch, and context window management. Luke configures it with tools, hooks, sub-agents, and a two-layer system prompt.

## MCP Tools

Tools are built per invocation in a closure that captures `chat_id` and `bot`, scoping each set to a specific chat.

### Categories

**Telegram (14)** — `send_message`, `send_photo`, `send_document`, `send_voice`, `send_video`, `send_location`, `send_poll`, `send_buttons`, `reply`, `forward`, `react`, `edit_message`, `delete_message`, `pin`

**Memory (8)** — `remember`, `recall`, `recall_conversation`, `forget`, `connect`, `restore`, `bulk_memory`, `memory_history`

**Scheduling (1)** — `schedule_task` (cron, interval, once)

**Monitoring (1)** — `get_cost_report` (usage stats by period)

### Patterns

- All tools return a standardized `_ok(text)` response
- File-sending tools validate paths against allowed roots via `_safe_path()`
- Tool annotations (`openWorldHint`, `readOnlyHint`, `destructiveHint`) communicate intent to the SDK

## Hooks

Four hooks customize agent behavior at lifecycle points:

**Stop** — When the session ends, prompts the agent to persist what it learned: new entities, episodes, insights, goals, and scheduled follow-ups. Without this hook, learned information is lost when the session ends.

**PreToolUse** — Counts outbound Telegram sends per run and blocks after a threshold (default: 20). Counter is per-invocation.

**PreCompact** — When the SDK trims context, instructs the agent to preserve memory IDs, pending actions, active goals, and relationship links in the compaction summary.

**Notification** — Logs SDK notifications to structlog.

## Effort Classification

`_classify_effort()` categorizes messages before the agent runs:

- **Low** (<15 words, no questions, no media) → thinking disabled
- **High** (>150 words, multiple questions, media, code blocks, complex keywords) → extended thinking (16K token budget)
- **Medium** (everything else) → adaptive thinking

Runs *before* memory injection so injected context doesn't skew the heuristic.

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

## Key Configuration

The `ClaudeAgentOptions` passed to the SDK include: model with fallback, explicit allowed tool list (Claude Code built-ins + `mcp__luke__*` wildcard), `bypassPermissions` mode, project + user setting sources, dynamic thinking config, turn and budget limits, file checkpointing, and Docker sandbox.

## Message Chunking

`send_long_message()` splits responses at Telegram's 4096-character limit, preferring newline boundaries. Each chunk retries with exponential backoff and falls back to plaintext on HTML parse failure.
