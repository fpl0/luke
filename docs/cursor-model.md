# Cursor Model

The cursor model handles message batching, failure recovery, restart resilience, and chat isolation through one mechanism.

## Message Flow

```
Telegram message
  → aiogram handler
    → _store()       insert into messages table (dedup check)
    → _dispatch()    fire-and-forget background task
      → process()    fetch pending, run agent, advance cursor
```

`_store()` and `_dispatch()` are fast — insert a row, schedule a task. The heavy work happens asynchronously in `process()`.

## Cursor Semantics

Two tables: `messages` stores every message with an auto-incrementing ID, `cursors` tracks the last processed ID per chat.

`process()` fetches all messages with `id > cursor`. After a successful agent run, cost logging and cursor advancement are wrapped in a single `batch()` commit — both succeed or neither does.

The cursor advances only on success.

## At-Least-Once Delivery

On failure, the cursor stays put. Messages remain pending and are fetched again on the next attempt. Messages may be processed more than once on failure, but are never silently skipped.

## Batch Processing

Messages that arrive while the agent is busy accumulate. The next `process()` call fetches them all as one prompt with sender names, timestamps, and message IDs. The agent sees the full conversational context rather than isolated messages.

## Startup Replay

On start, `main()` checks for pending messages and dispatches immediately. Messages sent during downtime are processed on restart.

## Retry with Exponential Backoff

Failed agent runs schedule retries via `call_later()` with increasing delays (30s, 60s, 120s). After `max_retries` (default: 3) failures, the messages are skipped — cursor advances, session clears, user gets notified with a timeout-specific or generic error message.

Retry counts are in-memory (not persisted). A process restart resets all counts, giving messages fresh attempts.

## Error Throttling

`_should_send_error()` rate-limits error notifications to one per 30 seconds per chat.

## Full Lifecycle

1. **Arrive** — aiogram handler fires
2. **Store** — `_store()` inserts into SQLite (dedup)
3. **Queue** — `_dispatch()` creates background task
4. **Lock** — `process()` acquires per-chat lock
5. **Fetch** — all messages with `id > cursor`
6. **Recall** — auto-inject relevant memories
7. **Classify** — determine effort level
8. **Execute** — `run_agent()` via Claude SDK
9. **Advance** — cursor moves (atomic with cost log)
10. **Respond** — send output via Telegram

Failure at step 8 → step 9 doesn't happen → messages stay pending for retry.
