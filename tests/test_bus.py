"""Tests for the event bus: emission, subscription, pattern matching, dispatch."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from luke.bus import Event, EventBus


@pytest.fixture()
def bus(test_db: Any) -> EventBus:
    """Fresh event bus per test, backed by the test DB."""
    b = EventBus()
    yield b
    b.clear()


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


class TestEvent:
    def test_event_is_frozen(self) -> None:
        e = Event(id=1, kind="test", payload={"k": "v"})
        with pytest.raises(AttributeError):
            e.kind = "other"  # type: ignore[misc]

    def test_event_defaults(self) -> None:
        e = Event(id=0, kind="x")
        assert e.payload == {}
        assert e.timestamp == ""

    def test_event_equality(self) -> None:
        a = Event(id=1, kind="x", payload={"a": 1})
        b = Event(id=1, kind="x", payload={"a": 1})
        assert a == b

    def test_event_hashable(self) -> None:
        """Events with dict payloads are not hashable (dict is unhashable)."""
        e = Event(id=1, kind="x", payload={})
        # frozen dataclass with slots — hash depends on field hashability
        # dict is unhashable, so Event with non-empty payload isn't hashable
        # but empty dict is fine for __eq__


# ---------------------------------------------------------------------------
# Emit + persistence
# ---------------------------------------------------------------------------


class TestEmit:
    def test_emit_returns_event_with_id(self, bus: EventBus) -> None:
        event = bus.emit("test_event", {"key": "value"})
        assert event.id > 0
        assert event.kind == "test_event"
        assert event.payload == {"key": "value"}

    def test_emit_default_payload(self, bus: EventBus) -> None:
        event = bus.emit("bare_event")
        assert event.payload == {}
        assert event.id > 0

    def test_emit_persists_to_db(self, bus: EventBus, test_db: Any) -> None:
        bus.emit("persisted_event", {"x": 42})
        assert test_db.count_unconsumed_events("persisted_event") == 1

    def test_emit_multiple_increments_ids(self, bus: EventBus) -> None:
        e1 = bus.emit("a")
        e2 = bus.emit("b")
        assert e2.id > e1.id

    def test_emit_tracks_stats(self, bus: EventBus) -> None:
        bus.emit("tool_use")
        bus.emit("tool_use")
        bus.emit("new_episode")
        assert bus.stats == {"tool_use": 2, "new_episode": 1}


class TestEmitLocal:
    def test_emit_local_no_persistence(self, bus: EventBus, test_db: Any) -> None:
        event = bus.emit_local("ephemeral")
        assert event.id == 0
        assert test_db.count_unconsumed_events("ephemeral") == 0

    def test_emit_local_tracks_stats(self, bus: EventBus) -> None:
        bus.emit_local("ping")
        bus.emit_local("ping")
        assert bus.stats["ping"] == 2


# ---------------------------------------------------------------------------
# Sync handlers
# ---------------------------------------------------------------------------


class TestSyncHandlers:
    def test_exact_match(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.on("tool_use", received.append)
        bus.emit("tool_use", {"tool": "remember"})
        assert len(received) == 1
        assert received[0].kind == "tool_use"

    def test_no_match(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.on("tool_use", received.append)
        bus.emit("new_episode")
        assert len(received) == 0

    def test_glob_star(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.on("deep_work_*", received.append)
        bus.emit("deep_work_oriented")
        bus.emit("deep_work_skipped")
        bus.emit("tool_use")
        assert len(received) == 2

    def test_catch_all(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.on("*", received.append)
        bus.emit("a")
        bus.emit("b")
        bus.emit("c")
        assert len(received) == 3

    def test_multiple_handlers_same_pattern(self, bus: EventBus) -> None:
        r1: list[Event] = []
        r2: list[Event] = []
        bus.on("x", r1.append)
        bus.on("x", r2.append)
        bus.emit("x")
        assert len(r1) == 1
        assert len(r2) == 1

    def test_handler_exception_isolated(self, bus: EventBus) -> None:
        """A failing handler doesn't prevent other handlers from running."""
        received: list[Event] = []

        def bad_handler(e: Event) -> None:
            raise ValueError("boom")

        bus.on("x", bad_handler)
        bus.on("x", received.append)
        bus.emit("x")
        assert len(received) == 1  # second handler still ran

    def test_handler_receives_payload(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.on("test", received.append)
        bus.emit("test", {"answer": 42})
        assert received[0].payload["answer"] == 42


# ---------------------------------------------------------------------------
# Async handlers
# ---------------------------------------------------------------------------


class TestAsyncHandlers:
    @pytest.mark.asyncio
    async def test_async_handler_called(self, bus: EventBus) -> None:
        received: list[Event] = []

        async def handler(e: Event) -> None:
            received.append(e)

        bus.on("async_test", handler)
        bus.emit("async_test", {"val": 1})
        # Allow the scheduled task to run
        await asyncio.sleep(0.05)
        assert len(received) == 1
        assert received[0].payload["val"] == 1

    @pytest.mark.asyncio
    async def test_async_glob_pattern(self, bus: EventBus) -> None:
        received: list[Event] = []

        async def handler(e: Event) -> None:
            received.append(e)

        bus.on("memory_*", handler)
        bus.emit("memory_created")
        bus.emit("memory_deleted")
        bus.emit("tool_use")
        await asyncio.sleep(0.05)
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_async_exception_isolated(self, bus: EventBus) -> None:
        received: list[Event] = []

        async def bad_handler(e: Event) -> None:
            raise RuntimeError("async boom")

        async def good_handler(e: Event) -> None:
            received.append(e)

        bus.on("x", bad_handler)
        bus.on("x", good_handler)
        bus.emit("x")
        await asyncio.sleep(0.05)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_mixed_sync_and_async(self, bus: EventBus) -> None:
        sync_received: list[Event] = []
        async_received: list[Event] = []

        def sync_h(e: Event) -> None:
            sync_received.append(e)

        async def async_h(e: Event) -> None:
            async_received.append(e)

        bus.on("mixed", sync_h)
        bus.on("mixed", async_h)
        bus.emit("mixed", {"data": True})
        await asyncio.sleep(0.05)
        assert len(sync_received) == 1
        assert len(async_received) == 1


# ---------------------------------------------------------------------------
# Unsubscribe
# ---------------------------------------------------------------------------


class TestOff:
    def test_off_removes_handler(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.on("x", received.append)
        bus.off("x", received.append)
        bus.emit("x")
        assert len(received) == 0

    def test_off_nonexistent_handler_noop(self, bus: EventBus) -> None:
        bus.off("x", lambda e: None)  # no error

    def test_off_nonexistent_pattern_noop(self, bus: EventBus) -> None:
        bus.off("never_registered", lambda e: None)  # no error

    def test_off_only_removes_specific_handler(self, bus: EventBus) -> None:
        r1: list[Event] = []
        r2: list[Event] = []
        bus.on("x", r1.append)
        bus.on("x", r2.append)
        bus.off("x", r1.append)
        bus.emit("x")
        assert len(r1) == 0
        assert len(r2) == 1


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_has_listeners_true(self, bus: EventBus) -> None:
        bus.on("tool_use", lambda e: None)
        assert bus.has_listeners("tool_use") is True

    def test_has_listeners_false(self, bus: EventBus) -> None:
        assert bus.has_listeners("tool_use") is False

    def test_has_listeners_glob(self, bus: EventBus) -> None:
        bus.on("tool_*", lambda e: None)
        assert bus.has_listeners("tool_use") is True
        assert bus.has_listeners("memory_x") is False

    def test_listener_count_specific(self, bus: EventBus) -> None:
        bus.on("x", lambda e: None)
        bus.on("x", lambda e: None)
        assert bus.listener_count("x") == 2

    def test_listener_count_total(self, bus: EventBus) -> None:
        bus.on("a", lambda e: None)
        bus.on("b", lambda e: None)
        assert bus.listener_count() == 2

    def test_reset_stats(self, bus: EventBus) -> None:
        bus.emit("a")
        bus.reset_stats()
        assert bus.stats == {}

    def test_clear(self, bus: EventBus) -> None:
        bus.on("x", lambda e: None)
        bus.emit("x")
        bus.clear()
        assert bus.listener_count() == 0
        assert bus.stats == {}


# ---------------------------------------------------------------------------
# Pattern matching edge cases
# ---------------------------------------------------------------------------


class TestPatternMatching:
    def test_question_mark_glob(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.on("event_?", received.append)
        bus.emit("event_a")
        bus.emit("event_b")
        bus.emit("event_ab")  # no match
        assert len(received) == 2

    def test_bracket_glob(self, bus: EventBus) -> None:
        received: list[Event] = []
        bus.on("level_[123]", received.append)
        bus.emit("level_1")
        bus.emit("level_2")
        bus.emit("level_4")  # no match
        assert len(received) == 2

    def test_overlapping_patterns(self, bus: EventBus) -> None:
        """An event matching multiple patterns triggers all matching handlers."""
        r1: list[Event] = []
        r2: list[Event] = []
        bus.on("tool_use", r1.append)
        bus.on("tool_*", r2.append)
        bus.emit("tool_use")
        assert len(r1) == 1
        assert len(r2) == 1
