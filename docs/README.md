# Architecture

Luke is a personal Telegram agent. Single Python process, no frameworks.

```
Telegram message
  → aiogram handler → store in SQLite → dispatch background task
    → process(): fetch pending → inject memories → classify effort
      → run_agent(): Claude SDK + 27 MCP tools + model routing + sub-agents
        → advance cursor → send responses
```

A scheduler loop runs alongside, ticking every 60 seconds — hourly maintenance, daily consolidation and proactive scans, 8-hour deep work sessions (autonomous goal execution), weekly reflection.

## Design Decisions

- **[Opinionated Minimalism](opinionated-minimalism.md)** — No optional dependencies. No speculative abstractions. Platform constants are constants, not config. Customization means changing the code.

- **[Cursor Model](cursor-model.md)** — Messages accumulate in SQLite. A cursor tracks what's processed. Advances only on success → at-least-once delivery, batch processing, startup replay.

- **[Memory](memory.md)** — Five types as markdown files. Indexed with FTS5 + semantic embeddings. Hybrid retrieval via Reciprocal Rank Fusion. Composite scoring. Adaptive forgetting with spaced repetition.

- **[Letta Backend](letta.md)** — Optional server-side memory backend behind two reversible seams: semantic-candidate sourcing for recall and self-editing core-memory context for turns. Fail-safe by construction — sqlite remains the source of truth and every failure falls back to it.

- **[Agent](agent.md)** — Claude Agent SDK with MCP tools, four hooks, smart model routing (haiku/sonnet/opus), three sub-agents, conversation continuity, self-monitoring.

- **[Autonomous Behaviors](autonomous-behaviors.md)** — Four agent invocations on timers: consolidation, reflection, proactive scan, deep work (autonomous goal execution).

- **[Concurrency](concurrency.md)** — Per-chat locks, global semaphore, process lock. Thread-local SQLite with atomic batches. Retry with backoff. Graceful shutdown.

- **[Security](security.md)** — Path traversal prevention, FTS injection protection, rate limiting, budget caps, Docker sandbox.

- **[Persona](persona.md)** — Two-layer system prompt separating tool competence from personality. Autonomy boundaries. Memory hygiene. Goal-driven action.
