# Autonomous Behaviors

Four agent invocations run on timers, independent of user messages. Each has a specific prompt, budget limit, and purpose. Models are routed per behavior via config settings.

## Scheduler

`start_scheduler_loop()` ticks every 60 seconds:

```
tick
  ├── Hourly maintenance (no agent invocation)
  │     ├── FTS cleanup — remove archived entries from search index
  │     ├── Importance decay — apply type-specific forgetting rates
  │     ├── Session cleanup — remove stale sessions + model ratchet
  │     └── Outbound log pruning — remove dedup entries older than 24h
  │
  ├── Collect due maintenance behaviors → run in parallel (asyncio.gather)
  │     ├── Consolidation (daily)
  │     ├── Proactive scan (daily) — can also take lightweight goal actions
  │     └── Reflection (weekly) + FTS pruning
  │
  ├── Launch deep work as background task (asyncio.create_task, non-blocking)
  │     └── Deep work (every 8h, budget-limited) — autonomous goal execution
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

**Interval:** daily | **Model:** `settings.proactive_scan_model` (sonnet)

Reviews active goals, recent insights, and 14 days of conversation summaries (capped at 500 messages). The agent decides whether to message the user or take autonomous action.

Can now take lightweight goal actions (research, schedule, write) in addition to messaging. Key constraint: *"Reserve messaging the user for things that genuinely need their input. If nothing warrants action or a message, do nothing."*

## Deep Work

**Interval:** every 8 hours | **Model:** `settings.deep_work_model` (opus) | **Replaces:** goal execution

Autonomous multi-step goal execution with plan-before-execute pattern:

1. **Plan** — picks highest-priority goal, creates/resumes a work plan at `workspace/plans/{goal_id}.md`
2. **Execute** — works through steps with self-reflection ("Am I making progress?"), updates plan after each step
3. **Wrap up** — saves summary episode, marks plan completed/blocked

Key differences from the old goal execution:
- **Multi-step** — executes as many steps as budget allows, not just one
- **Plan persistence** — plans survive crashes; next session resumes from saved plan
- **Budget-limited** — per-session cap (`deep_work_max_budget_usd`) + daily cap (`daily_deep_work_budget_usd`)
- **Background task** — runs via `asyncio.create_task()`, doesn't block the scheduler loop
- **Rate-limited** — max 1 outbound message per session (prefer updating plan's Blockers section)

## Budget

All behaviors run through `_run_behavior()`, which acquires the behavior semaphore (max 2 concurrent) and the global semaphore, enforcing limits:

| | User Messages | Maintenance Behaviors | Deep Work |
|---|---|---|---|
| Max turns | 200 | 75 | 300 |
| Max budget | $5.00 | $1.00 | $3.00/session, $60/day |
| Max sends | 20 | 20 | 1 |
| Timeout | 30 min | 30 min | 30 min |

Maintenance behaviors run via `asyncio.gather()` (awaited). Deep work runs as a background task (non-blocking). Only successful behaviors have their timestamps updated — failures retry on the next tick.

## User Tasks

The scheduler also handles user-created tasks via `schedule_task`:

- **cron** — standard expressions (`0 9 * * 1`)
- **interval** — milliseconds between runs
- **once** — ISO timestamp, marked completed after firing

Task deduplication prevents the same task from firing twice if its previous run is still in-flight.
