"""Event bus: typed events, in-process pub/sub, SQLite persistence.

The event bus is Luke's nervous system. Every perception, action, and internal
state change flows through it as a typed event. Subscribers register patterns
and receive matching events asynchronously.

Usage::

    from luke.bus import bus, Event

    # Subscribe to events (async handler)
    async def on_tool_use(event: Event) -> None:
        print(f"Tool used: {event.payload}")

    bus.on("tool_use", on_tool_use)          # exact match
    bus.on("deep_work_*", on_deep_work)      # glob pattern
    bus.on("*", on_any_event)                # catch-all

    # Emit events (sync — safe from hooks and async code)
    bus.emit("tool_use", {"tool": "remember", "duration_ms": 42})

    # Unsubscribe
    bus.off("tool_use", on_tool_use)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Type alias for event handlers
SyncHandler = Callable[["Event"], None]
AsyncHandler = Callable[["Event"], Coroutine[Any, Any, None]]
Handler = SyncHandler | AsyncHandler


@dataclass(frozen=True, slots=True)
class Event:
    """An immutable event flowing through the bus.

    Attributes:
        id: Database row ID (0 if not persisted).
        kind: Event type string (e.g. "tool_use", "new_episode").
        payload: Structured data, schema varies per kind.
        timestamp: ISO 8601 creation time (set by DB default).
    """

    id: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


class EventBus:
    """In-process event routing with glob pattern matching and SQLite persistence.

    Events are persisted to the ``events`` table synchronously (safe from any
    thread/context), then fanned out to matching async and sync handlers.
    """

    def __init__(self) -> None:
        self._sync_handlers: dict[str, list[SyncHandler]] = defaultdict(list)
        self._async_handlers: dict[str, list[AsyncHandler]] = defaultdict(list)
        self._stats: dict[str, int] = defaultdict(int)  # kind → emit count

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def on(self, pattern: str, handler: Handler) -> None:
        """Register a handler for events matching *pattern* (glob syntax).

        Async callables go into the async dispatch path; plain functions are
        called synchronously during ``emit()``.

        Examples::

            bus.on("tool_use", sync_handler)
            bus.on("deep_work_*", async_handler)
            bus.on("*", catch_all)
        """
        if inspect.iscoroutinefunction(handler):
            self._async_handlers[pattern].append(handler)  # type: ignore[arg-type]
        else:
            self._sync_handlers[pattern].append(handler)  # type: ignore[arg-type]

    def off(self, pattern: str, handler: Handler) -> None:
        """Remove a previously registered handler."""
        for registry in (self._sync_handlers, self._async_handlers):
            handlers = registry.get(pattern)
            if handlers:
                try:
                    handlers.remove(handler)  # type: ignore[arg-type]
                except ValueError:
                    pass
                if not handlers:
                    del registry[pattern]

    def has_listeners(self, kind: str) -> bool:
        """Check if any handler matches the given event kind."""
        for pattern in list(self._sync_handlers) + list(self._async_handlers):
            if fnmatch(kind, pattern):
                return True
        return False

    def listener_count(self, pattern: str | None = None) -> int:
        """Count registered handlers, optionally filtered by pattern."""
        if pattern is not None:
            sync = len(self._sync_handlers.get(pattern, []))
            async_ = len(self._async_handlers.get(pattern, []))
            return sync + async_
        total = sum(len(h) for h in self._sync_handlers.values())
        total += sum(len(h) for h in self._async_handlers.values())
        return total

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, kind: str, payload: dict[str, Any] | None = None) -> Event:
        """Persist an event and fan out to matching handlers.

        This method is **sync-safe** — it can be called from SDK hooks,
        synchronous code, or async code. Async handlers are scheduled onto
        the running event loop (if any); if no loop is running, only sync
        handlers fire.

        Returns the :class:`Event` with its database ID populated.
        """
        from . import db  # deferred to avoid circular imports

        payload = payload or {}
        payload_json = json.dumps(payload, default=str)

        # Persist
        event_id = db.emit_event(kind, payload_json)

        event = Event(
            id=event_id,
            kind=kind,
            payload=payload,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        self._stats[kind] += 1

        # Fan-out: sync handlers (immediate)
        self._dispatch_sync(event)

        # Fan-out: async handlers (scheduled)
        self._schedule_async(event)

        return event

    def emit_local(self, kind: str, payload: dict[str, Any] | None = None) -> Event:
        """Emit an event *without* persisting to the database.

        Useful for ephemeral signals (e.g. internal coordination) that don't
        need to survive restarts or appear in the dashboard.
        """
        payload = payload or {}
        event = Event(id=0, kind=kind, payload=payload)
        self._stats[kind] += 1
        self._dispatch_sync(event)
        self._schedule_async(event)
        return event

    # ------------------------------------------------------------------
    # Dispatch internals
    # ------------------------------------------------------------------

    def _dispatch_sync(self, event: Event) -> None:
        """Call all matching sync handlers."""
        for pattern, handlers in list(self._sync_handlers.items()):
            if fnmatch(event.kind, pattern):
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception:
                        logger.exception(
                            "Sync handler %s failed for event %s",
                            getattr(handler, "__name__", handler),
                            event.kind,
                        )

    def _schedule_async(self, event: Event) -> None:
        """Schedule matching async handlers on the running event loop."""
        matching: list[AsyncHandler] = []
        for pattern, handlers in list(self._async_handlers.items()):
            if fnmatch(event.kind, pattern):
                matching.extend(handlers)

        if not matching:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop — can't dispatch async handlers
            return

        for handler in matching:
            loop.create_task(self._safe_async_call(handler, event))

    @staticmethod
    async def _safe_async_call(handler: AsyncHandler, event: Event) -> None:
        """Call an async handler with exception logging."""
        try:
            await handler(event)
        except Exception:
            logger.exception(
                "Async handler %s failed for event %s",
                getattr(handler, "__name__", handler),
                event.kind,
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        """Emit counts per event kind since process start."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Clear emission statistics."""
        self._stats.clear()

    def clear(self) -> None:
        """Remove all handlers and reset stats. Mainly for testing."""
        self._sync_handlers.clear()
        self._async_handlers.clear()
        self._stats.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

bus = EventBus()
"""The global event bus instance. Import and use directly::

    from luke.bus import bus
    bus.emit("tool_use", {"tool": "remember"})
"""
