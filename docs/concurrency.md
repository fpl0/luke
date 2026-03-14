# Concurrency & Reliability

Single Python process handling messages, agent invocations, scheduled tasks, and autonomous behaviors simultaneously.

## Locking

Three layers:

**Per-chat lock** — `asyncio.Lock` per chat serializes processing within a single chat. Multiple dispatches for the same chat queue up; only the first finds pending messages.

**Global semaphore** — `asyncio.Semaphore` (default: 5) caps total concurrent agent invocations across all chats and behaviors. Both message processing and autonomous behaviors acquire the same semaphore.

**Process lock** — `fcntl.flock` on a lock file ensures one Luke instance at a time. Non-blocking — exits immediately if another instance holds the lock.

## SQLite

**Thread-local connections** — each thread gets its own `sqlite3.Connection` via `threading.local()`. Required because `asyncio.to_thread()` is used for CPU-bound operations like embedding inference.

**Batch context manager** — `batch()` groups multiple operations into a single atomic commit. Nestable. Rolls back on exception at the outermost level. Used to make cursor advancement and cost logging atomic.

**PRAGMAs** — WAL mode (concurrent reads during writes), `synchronous=NORMAL`, `foreign_keys=ON`, 5s busy timeout, 64 MB mmap, 8 MB page cache, temp tables in memory.

## Retry

Failed agent runs schedule retries via `call_later()` with exponential backoff (30s, 60s, 120s). After `max_retries` (default: 3): skip messages, clear session, notify user.

Retry counts are in-memory, not persisted. A process restart resets all counts — correct behavior for transient errors.

## Error Throttling

One error notification per 30 seconds per chat via `_should_send_error()`.

## Task Deduplication

The scheduler tracks running tasks in a dict and skips any task whose previous run is still in-flight.

## Shutdown

Signal handlers (`SIGTERM`, `SIGINT`) set a shutdown event. Then:

1. Stop dispatcher — no new messages accepted
2. Drain in-flight background tasks via `asyncio.gather()`
3. Notify user ("Going offline.")
4. Scheduler drains running tasks

In-flight agent runs complete, cursors advance, results are sent.

## Background Tasks

`_dispatch()` creates fire-and-forget `asyncio.Task`s tracked in a set. Tasks self-remove on completion. Exceptions are logged. All remaining tasks are awaited during shutdown.
