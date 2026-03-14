# Autonomous Behaviors

Four agent invocations run on timers, independent of user messages. Each has a specific prompt, budget limit, and purpose.

## Scheduler

`start_scheduler_loop()` ticks every 60 seconds:

```
tick
  ├── Hourly maintenance (no agent invocation)
  │     ├── FTS cleanup — remove archived entries from search index
  │     ├── Importance decay — apply type-specific forgetting rates
  │     └── Session cleanup — remove stale sessions
  │
  ├── Collect due behaviors → run in parallel (asyncio.gather)
  │     ├── Consolidation (daily)
  │     ├── Proactive scan (daily)
  │     ├── Goal execution (every 12h)
  │     └── Reflection (weekly) + FTS pruning
  │
  └── Check user-created tasks (cron / interval / once)
```

### Persistence

Behavior timestamps are stored in a `behavior_state` table. On startup, `_load_offset()` converts stored wall-clock timestamps to monotonic offsets so behaviors resume their schedule correctly after restarts.

## Consolidation

**Interval:** daily

Clusters related episodes (≥3 shared tags or ≥2 shared links) and asks the agent to synthesize insights, connect them to entities, and archive redundant episodes. Up to 3 clusters per run.

## Reflection

**Interval:** weekly

Reviews the past 7 days of memories (up to 20) and last 50 messages. Prompts the agent to analyze which responses got positive reactions, which were corrected or ignored, patterns in satisfaction, recurring requests that should become procedures, and goal progress. Outputs actionable insights.

## Proactive Scan

**Interval:** daily

Reviews active goals, recent insights, and 14 days of conversation summaries. The agent decides whether to message the user about approaching deadlines, stale goals, unresolved items, or undelivered follow-ups.

Key constraint: the prompt explicitly says *"If nothing warrants a message, do nothing. Don't message just to check in."*

## Goal Execution

**Interval:** every 12 hours

Picks the highest-priority active goal and executes one concrete step — research, code, a question to the user, or a scheduled follow-up. After acting, the agent documents what it did (episode), updates the goal memory with progress, and schedules follow-up if needed.

The "one step" constraint caps budget usage per cycle and keeps progress incremental.

## Budget

All behaviors run through `_run_behavior()`, which acquires the same global semaphore as message processing and enforces lower limits:

| | User Messages | Behaviors |
|---|---|---|
| Max turns | 200 | 75 |
| Max budget | $5.00 | $1.00 |
| Timeout | 30 min | 30 min |

When multiple behaviors are due, they run via `asyncio.gather()`. Only successful behaviors have their timestamps updated — failures retry on the next tick.

## User Tasks

The scheduler also handles user-created tasks via `schedule_task`:

- **cron** — standard expressions (`0 9 * * 1`)
- **interval** — milliseconds between runs
- **once** — ISO timestamp, marked completed after firing

Task deduplication prevents the same task from firing twice if its previous run is still in-flight.
