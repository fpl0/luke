# Architecture

Luke is a personal Telegram agent. Single Python process, no frameworks.

```
Telegram message
  → aiogram handler → store in SQLite → dispatch background task
    → process(): fetch pending → inject memories → classify effort
      → run_agent(): Claude SDK + 24 MCP tools + sub-agents
        → advance cursor → send responses
```

A scheduler loop runs alongside, ticking every 60 seconds — hourly maintenance, daily consolidation and proactive scans, 12-hour goal execution, weekly reflection.

## Design Decisions

- **[Opinionated Minimalism](opinionated-minimalism.md)** — No optional dependencies. No speculative abstractions. Platform constants are constants, not config. Customization means changing the code.

- **[Cursor Model](cursor-model.md)** — Messages accumulate in SQLite. A cursor tracks what's processed. Advances only on success → at-least-once delivery, batch processing, startup replay.

- **[Memory](memory.md)** — Five types as markdown files. Indexed with FTS5 + semantic embeddings. Hybrid retrieval via Reciprocal Rank Fusion. Composite scoring. Adaptive forgetting with spaced repetition.

- **[Agent](agent.md)** — Claude Agent SDK with MCP tools, four hooks, dynamic effort classification, three sub-agents, two-layer system prompt.

- **[Autonomous Behaviors](autonomous-behaviors.md)** — Four agent invocations on timers: consolidation, reflection, proactive scan, goal execution.

- **[Concurrency](concurrency.md)** — Per-chat locks, global semaphore, process lock. Thread-local SQLite with atomic batches. Retry with backoff. Graceful shutdown.

- **[Security](security.md)** — Path traversal prevention, FTS injection protection, rate limiting, budget caps, Docker sandbox.

- **[Persona](persona.md)** — Two-layer system prompt separating tool competence from personality. Autonomy boundaries. Memory hygiene. Goal-driven action.
