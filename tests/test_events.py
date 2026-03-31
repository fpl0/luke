"""Tests for the event-driven behavior system (db events + behavior no-ops)."""

from __future__ import annotations

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
