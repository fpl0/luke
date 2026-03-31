"""Tests for the event-driven behavior system (db events + behavior no-ops)."""

from __future__ import annotations

import json
from typing import Any

import pytest


class TestEmitEvent:
    def test_emit_returns_id(self, test_db: Any) -> None:
        event_id = test_db.emit_event("new_episode", '{"id": "ep-1"}')
        assert event_id > 0

    def test_emit_multiple_events(self, test_db: Any) -> None:
        id1 = test_db.emit_event("new_episode")
        id2 = test_db.emit_event("new_episode")
        assert id2 > id1

    def test_emit_default_payload(self, test_db: Any) -> None:
        event_id = test_db.emit_event("user_message")
        assert event_id > 0

    def test_emit_nested_json_payload(self, test_db: Any) -> None:
        payload = json.dumps({"user": "filipe", "tags": ["urgent", "bug"], "count": 3})
        event_id = test_db.emit_event("complex_event", payload)
        assert event_id > 0

    def test_emit_empty_string_payload(self, test_db: Any) -> None:
        event_id = test_db.emit_event("minimal_event", "")
        assert event_id > 0

    def test_emit_plain_text_payload(self, test_db: Any) -> None:
        event_id = test_db.emit_event("text_event", "just a plain string")
        assert event_id > 0


class TestCountUnconsumedEvents:
    def test_empty(self, test_db: Any) -> None:
        assert test_db.count_unconsumed_events("new_episode") == 0

    def test_counts_matching_type(self, test_db: Any) -> None:
        test_db.emit_event("new_episode")
        test_db.emit_event("new_episode")
        test_db.emit_event("new_insight")
        assert test_db.count_unconsumed_events("new_episode") == 2

    def test_counts_multiple_types(self, test_db: Any) -> None:
        test_db.emit_event("new_episode")
        test_db.emit_event("new_insight")
        test_db.emit_event("feedback_negative")
        assert test_db.count_unconsumed_events("new_episode", "new_insight") == 2

    def test_excludes_consumed(self, test_db: Any) -> None:
        test_db.emit_event("new_episode")
        test_db.emit_event("new_episode")
        test_db.consume_events("new_episode")
        assert test_db.count_unconsumed_events("new_episode") == 0

    def test_no_types_returns_zero(self, test_db: Any) -> None:
        test_db.emit_event("new_episode")
        assert test_db.count_unconsumed_events() == 0

    def test_since_filters_old_events(self, test_db: Any) -> None:
        """Events created before `since` are excluded from the count."""
        test_db.emit_event("new_episode")
        test_db.emit_event("new_episode")
        # Backdate both events to 2 days ago
        conn = test_db._db()
        conn.execute("UPDATE events SET created = datetime('now', '-2 days')")
        conn.commit()
        # Emit a fresh event
        test_db.emit_event("new_episode")
        # Count since 1 day ago should only see the fresh one
        since = conn.execute("SELECT datetime('now', '-1 day') AS ts").fetchone()["ts"]
        assert test_db.count_unconsumed_events("new_episode", since=since) == 1

    def test_since_none_counts_all(self, test_db: Any) -> None:
        """When since=None, all unconsumed events are counted (default behavior)."""
        test_db.emit_event("new_episode")
        test_db.emit_event("new_episode")
        # Backdate one event
        conn = test_db._db()
        conn.execute(
            "UPDATE events SET created = datetime('now', '-30 days') WHERE id = 1"
        )
        conn.commit()
        assert test_db.count_unconsumed_events("new_episode", since=None) == 2

    def test_since_with_multiple_types(self, test_db: Any) -> None:
        """The since parameter works correctly across multiple event types."""
        test_db.emit_event("new_episode")
        test_db.emit_event("new_insight")
        # Backdate both
        conn = test_db._db()
        conn.execute("UPDATE events SET created = datetime('now', '-5 days')")
        conn.commit()
        # Add fresh events
        test_db.emit_event("new_episode")
        test_db.emit_event("new_insight")
        since = conn.execute("SELECT datetime('now', '-1 day') AS ts").fetchone()["ts"]
        assert test_db.count_unconsumed_events("new_episode", "new_insight", since=since) == 2


class TestConsumeEvents:
    def test_consumes_matching(self, test_db: Any) -> None:
        test_db.emit_event("new_episode")
        test_db.emit_event("new_episode")
        consumed = test_db.consume_events("new_episode")
        assert consumed == 2

    def test_does_not_consume_other_types(self, test_db: Any) -> None:
        test_db.emit_event("new_episode")
        test_db.emit_event("new_insight")
        consumed = test_db.consume_events("new_episode")
        assert consumed == 1
        assert test_db.count_unconsumed_events("new_insight") == 1

    def test_consume_multiple_types(self, test_db: Any) -> None:
        test_db.emit_event("new_episode")
        test_db.emit_event("feedback_negative")
        consumed = test_db.consume_events("new_episode", "feedback_negative")
        assert consumed == 2

    def test_consume_idempotent(self, test_db: Any) -> None:
        test_db.emit_event("new_episode")
        test_db.consume_events("new_episode")
        consumed = test_db.consume_events("new_episode")
        assert consumed == 0

    def test_no_types_returns_zero(self, test_db: Any) -> None:
        test_db.emit_event("new_episode")
        assert test_db.consume_events() == 0

    def test_since_only_consumes_recent(self, test_db: Any) -> None:
        """consume_events with since only marks events after that timestamp."""
        test_db.emit_event("new_episode")
        test_db.emit_event("new_episode")
        # Backdate both to 3 days ago
        conn = test_db._db()
        conn.execute("UPDATE events SET created = datetime('now', '-3 days')")
        conn.commit()
        # Add a fresh event
        test_db.emit_event("new_episode")
        since = conn.execute("SELECT datetime('now', '-1 day') AS ts").fetchone()["ts"]
        consumed = test_db.consume_events("new_episode", since=since)
        assert consumed == 1
        # The two old events remain unconsumed
        assert test_db.count_unconsumed_events("new_episode") == 2

    def test_since_none_consumes_all(self, test_db: Any) -> None:
        """When since=None, all unconsumed events of the type are consumed."""
        test_db.emit_event("new_episode")
        # Backdate
        conn = test_db._db()
        conn.execute("UPDATE events SET created = datetime('now', '-10 days')")
        conn.commit()
        test_db.emit_event("new_episode")
        consumed = test_db.consume_events("new_episode", since=None)
        assert consumed == 2
        assert test_db.count_unconsumed_events("new_episode") == 0

    def test_since_with_multiple_types(self, test_db: Any) -> None:
        """consume_events with since works across multiple event types."""
        test_db.emit_event("new_episode")
        test_db.emit_event("feedback_negative")
        # Backdate
        conn = test_db._db()
        conn.execute("UPDATE events SET created = datetime('now', '-5 days')")
        conn.commit()
        # Fresh events
        test_db.emit_event("new_episode")
        test_db.emit_event("feedback_negative")
        since = conn.execute("SELECT datetime('now', '-1 day') AS ts").fetchone()["ts"]
        consumed = test_db.consume_events("new_episode", "feedback_negative", since=since)
        assert consumed == 2
        # Old ones remain
        assert test_db.count_unconsumed_events("new_episode", "feedback_negative") == 2


class TestCleanupEvents:
    def test_removes_old_consumed(self, test_db: Any) -> None:
        test_db.emit_event("new_episode")
        test_db.consume_events("new_episode")
        # Backdate the event
        conn = test_db._db()
        conn.execute("UPDATE events SET created = datetime('now', '-10 days')")
        conn.commit()
        removed = test_db.cleanup_events(retention_days=7)
        assert removed == 1

    def test_keeps_unconsumed(self, test_db: Any) -> None:
        test_db.emit_event("new_episode")
        conn = test_db._db()
        conn.execute("UPDATE events SET created = datetime('now', '-10 days')")
        conn.commit()
        removed = test_db.cleanup_events(retention_days=7)
        assert removed == 0

    def test_mixed_removes_consumed_keeps_unconsumed(self, test_db: Any) -> None:
        """With both consumed and unconsumed old events, only consumed are removed."""
        # Emit two events, consume one
        test_db.emit_event("new_episode")
        test_db.emit_event("new_insight")
        test_db.consume_events("new_episode")
        # Backdate both to beyond retention
        conn = test_db._db()
        conn.execute("UPDATE events SET created = datetime('now', '-10 days')")
        conn.commit()
        removed = test_db.cleanup_events(retention_days=7)
        assert removed == 1
        # The unconsumed new_insight event survives
        assert test_db.count_unconsumed_events("new_insight") == 1

    def test_keeps_recent_consumed(self, test_db: Any) -> None:
        """Consumed events within the retention window are not removed."""
        test_db.emit_event("new_episode")
        test_db.consume_events("new_episode")
        # Event was created just now (within retention), so nothing should be removed
        removed = test_db.cleanup_events(retention_days=7)
        assert removed == 0

    def test_cleanup_multiple_old_consumed(self, test_db: Any) -> None:
        """Multiple old consumed events of different types are all cleaned up."""
        for etype in ("new_episode", "new_insight", "feedback_negative"):
            test_db.emit_event(etype)
        test_db.consume_events("new_episode", "new_insight", "feedback_negative")
        conn = test_db._db()
        conn.execute("UPDATE events SET created = datetime('now', '-15 days')")
        conn.commit()
        removed = test_db.cleanup_events(retention_days=7)
        assert removed == 3


class TestBehaviorNoOps:
    def test_get_no_ops_default(self, test_db: Any) -> None:
        # Need to insert a behavior_state row first
        test_db.set_behavior_last_run("consolidation", "2026-01-01T00:00:00")
        assert test_db.get_behavior_no_ops("consolidation") == 0

    def test_increment_no_ops(self, test_db: Any) -> None:
        test_db.set_behavior_last_run("consolidation", "2026-01-01T00:00:00")
        result = test_db.increment_behavior_no_ops("consolidation")
        assert result == 1

    def test_increment_multiple(self, test_db: Any) -> None:
        test_db.set_behavior_last_run("consolidation", "2026-01-01T00:00:00")
        test_db.increment_behavior_no_ops("consolidation")
        test_db.increment_behavior_no_ops("consolidation")
        result = test_db.increment_behavior_no_ops("consolidation")
        assert result == 3

    def test_reset_no_ops(self, test_db: Any) -> None:
        test_db.set_behavior_last_run("consolidation", "2026-01-01T00:00:00")
        test_db.increment_behavior_no_ops("consolidation")
        test_db.increment_behavior_no_ops("consolidation")
        test_db.reset_behavior_no_ops("consolidation")
        assert test_db.get_behavior_no_ops("consolidation") == 0

    def test_get_nonexistent_behavior(self, test_db: Any) -> None:
        assert test_db.get_behavior_no_ops("nonexistent") == 0


class TestFeedbackNegativeEventEmission:
    def test_negative_reaction_emits_event(self, test_db: Any) -> None:
        test_db.store_reaction_feedback(
            chat_id="12345",
            msg_id=1,
            sender_id="user1",
            emoji="👎",
            timestamp="2026-03-31T21:00:00",
        )
        assert test_db.count_unconsumed_events("feedback_negative") == 1

    def test_positive_reaction_no_event(self, test_db: Any) -> None:
        test_db.store_reaction_feedback(
            chat_id="12345",
            msg_id=1,
            sender_id="user1",
            emoji="👍",
            timestamp="2026-03-31T21:00:00",
        )
        assert test_db.count_unconsumed_events("feedback_negative") == 0

    def test_neutral_reaction_no_event(self, test_db: Any) -> None:
        test_db.store_reaction_feedback(
            chat_id="12345",
            msg_id=1,
            sender_id="user1",
            emoji="🤔",
            timestamp="2026-03-31T21:00:00",
        )
        assert test_db.count_unconsumed_events("feedback_negative") == 0
