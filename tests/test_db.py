"""Tests for luke.db — helpers, messages, sessions, and tasks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from luke.db import (
    batch,
    ensure_utc,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestEnsureUtc:
    def test_naive_datetime(self) -> None:
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = ensure_utc(dt)
        assert result.tzinfo is UTC

    def test_aware_datetime_preserved(self) -> None:
        tz = timezone(timedelta(hours=5))
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=tz)
        result = ensure_utc(dt)
        assert result.tzinfo is tz


# ---------------------------------------------------------------------------
# Messages & Cursors (require test_db fixture)
# ---------------------------------------------------------------------------


class TestMessages:
    def test_store_and_retrieve(self, test_db: Any) -> None:
        test_db.store_message(
            chat_id="100",
            sender_name="Alice",
            sender_id="1",
            message_id=42,
            content="Hello",
            timestamp="2024-01-01T00:00:00",
        )
        pending = test_db.get_pending_messages("100")
        assert len(pending) == 1
        assert pending[0].content == "Hello"
        assert pending[0].sender_name == "Alice"
        assert pending[0].message_id == 42

    def test_pending_empty_initially(self, test_db: Any) -> None:
        assert test_db.get_pending_messages("999") == []

    def test_advance_cursor_skips_old(self, test_db: Any) -> None:
        test_db.store_message(
            chat_id="100", sender_name="A", content="msg1", timestamp="2024-01-01T00:00:00"
        )
        test_db.store_message(
            chat_id="100", sender_name="A", content="msg2", timestamp="2024-01-01T00:01:00"
        )
        msgs = test_db.get_pending_messages("100")
        assert len(msgs) == 2

        test_db.advance_cursor("100", msgs[0].id)
        remaining = test_db.get_pending_messages("100")
        assert len(remaining) == 1
        assert remaining[0].content == "msg2"

    def test_advance_cursor_idempotent(self, test_db: Any) -> None:
        test_db.store_message(
            chat_id="100", sender_name="A", content="msg1", timestamp="2024-01-01T00:00:00"
        )
        msgs = test_db.get_pending_messages("100")
        test_db.advance_cursor("100", msgs[0].id)
        test_db.advance_cursor("100", msgs[0].id)  # second call should be safe
        assert test_db.get_pending_messages("100") == []

    def test_recent_messages_ordering(self, test_db: Any) -> None:
        for i in range(5):
            test_db.store_message(
                chat_id="100",
                sender_name="A",
                content=f"msg{i}",
                timestamp=f"2024-01-01T00:0{i}:00",
            )
        recent = test_db.get_recent_messages("100", limit=3)
        assert len(recent) == 3
        # Should be chronological (oldest first)
        assert recent[0]["content"] == "msg2"
        assert recent[2]["content"] == "msg4"

    def test_recent_messages_limit(self, test_db: Any) -> None:
        for i in range(10):
            test_db.store_message(
                chat_id="100",
                sender_name="A",
                content=f"msg{i}",
                timestamp=f"2024-01-01T00:{i:02}:00",
            )
        recent = test_db.get_recent_messages("100", limit=3)
        assert len(recent) == 3


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessions:
    def test_session_none_initially(self, test_db: Any) -> None:
        assert test_db.get_session("unknown_chat") is None

    def test_set_and_get_session(self, test_db: Any) -> None:
        test_db.set_session("100", "session-abc")
        assert test_db.get_session("100") == "session-abc"

    def test_session_overwrite(self, test_db: Any) -> None:
        test_db.set_session("100", "old-session")
        test_db.set_session("100", "new-session")
        assert test_db.get_session("100") == "new-session"


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class TestTasks:
    def test_create_task_cron_valid(self, test_db: Any) -> None:
        task_id = test_db.create_task("100", "do stuff", "cron", "*/5 * * * *")
        assert len(task_id) == 8

    def test_create_task_cron_invalid(self, test_db: Any) -> None:
        with pytest.raises(ValueError, match="Invalid cron"):
            test_db.create_task("100", "do stuff", "cron", "not-a-cron")

    def test_create_task_interval_valid(self, test_db: Any) -> None:
        task_id = test_db.create_task("100", "do stuff", "interval", "60000")
        assert len(task_id) == 8

    def test_create_task_interval_invalid_string(self, test_db: Any) -> None:
        with pytest.raises(ValueError, match="milliseconds"):
            test_db.create_task("100", "do stuff", "interval", "abc")

    def test_create_task_interval_negative(self, test_db: Any) -> None:
        with pytest.raises(ValueError, match="positive"):
            test_db.create_task("100", "do stuff", "interval", "-100")

    def test_create_task_once_valid(self, test_db: Any) -> None:
        ts = datetime.now(UTC).isoformat()
        task_id = test_db.create_task("100", "do stuff", "once", ts)
        assert len(task_id) == 8

    def test_create_task_once_invalid(self, test_db: Any) -> None:
        with pytest.raises(ValueError, match="ISO timestamp"):
            test_db.create_task("100", "do stuff", "once", "not-a-date")

    def test_create_task_invalid_type(self, test_db: Any) -> None:
        with pytest.raises(ValueError, match="Invalid schedule_type"):
            test_db.create_task("100", "do stuff", "weekly", "monday")

    def test_get_due_tasks_excludes_completed_once(self, test_db: Any) -> None:
        ts = datetime.now(UTC).isoformat()
        task_id = test_db.create_task("100", "once task", "once", ts)
        test_db.update_task_last_run(task_id, ts)
        # Once tasks with last_run are filtered at SQL level
        tasks = test_db.get_due_tasks()
        assert all(t["id"] != task_id for t in tasks)

    def test_update_task_status(self, test_db: Any) -> None:
        task_id = test_db.create_task("100", "task", "cron", "*/5 * * * *")
        test_db.update_task_status(task_id, "paused")
        tasks = test_db.get_due_tasks()
        assert all(t["id"] != task_id for t in tasks)

    def test_log_task_run(self, test_db: Any) -> None:
        task_id = test_db.create_task("100", "task", "cron", "*/5 * * * *")
        started = datetime.now(UTC).isoformat()
        test_db.log_task_run(task_id, started, started, "ok")
        row = (
            test_db._db()
            .execute("SELECT * FROM task_logs WHERE task_id = ?", (task_id,))
            .fetchone()
        )
        assert row is not None
        assert row["result"] == "ok"


# ---------------------------------------------------------------------------
# Duplicate message detection
# ---------------------------------------------------------------------------


class TestDuplicateMessages:
    def test_unique_message_returns_true(self, test_db: Any) -> None:
        result = test_db.store_message(
            chat_id="100",
            sender_name="Alice",
            sender_id="1",
            message_id=42,
            content="Hello",
            timestamp="2024-01-01T00:00:00",
        )
        assert result is True

    def test_duplicate_returns_false(self, test_db: Any) -> None:
        test_db.store_message(
            chat_id="100",
            sender_name="Alice",
            sender_id="1",
            message_id=42,
            content="Hello",
            timestamp="2024-01-01T00:00:00",
        )
        result = test_db.store_message(
            chat_id="100",
            sender_name="Alice",
            sender_id="1",
            message_id=42,
            content="Hello",
            timestamp="2024-01-01T00:00:01",
        )
        assert result is False

    def test_same_msg_id_different_content_not_duplicate(self, test_db: Any) -> None:
        """Reactions/edits reuse msg_id but have different content."""
        test_db.store_message(
            chat_id="100",
            sender_name="Alice",
            sender_id="1",
            message_id=42,
            content="Hello",
            timestamp="2024-01-01T00:00:00",
        )
        result = test_db.store_message(
            chat_id="100",
            sender_name="Alice",
            sender_id="1",
            message_id=42,
            content="[Edited message 42]: Hello World",
            timestamp="2024-01-01T00:01:00",
        )
        assert result is True

    def test_no_sender_id_skips_dedup(self, test_db: Any) -> None:
        """Messages without sender_id should not be deduped."""
        test_db.store_message(
            chat_id="100",
            sender_name="System",
            sender_id="",
            message_id=0,
            content="Ping",
            timestamp="2024-01-01T00:00:00",
        )
        result = test_db.store_message(
            chat_id="100",
            sender_name="System",
            sender_id="",
            message_id=0,
            content="Ping",
            timestamp="2024-01-01T00:00:01",
        )
        assert result is True


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------


class TestSessionCleanup:
    def test_cleanup_stale_sessions(self, test_db: Any) -> None:
        # Set a session with old timestamp
        test_db._db().execute(
            "INSERT INTO sessions (chat_id, session_id, updated_at) VALUES (?, ?, ?)",
            ("999", "old-session", "2020-01-01T00:00:00+00:00"),
        )
        test_db._db().commit()
        cleaned = test_db.cleanup_stale_sessions(3600.0)
        assert cleaned == ["999"]
        assert test_db.get_session("999") is None

    def test_cleanup_keeps_fresh_sessions(self, test_db: Any) -> None:
        test_db.set_session("100", "fresh-session")
        cleaned = test_db.cleanup_stale_sessions(3600.0)
        assert cleaned == []
        assert test_db.get_session("100") == "fresh-session"

    def test_set_session_records_timestamp(self, test_db: Any) -> None:
        test_db.set_session("100", "sess-123")
        row = (
            test_db._db()
            .execute("SELECT updated_at FROM sessions WHERE chat_id = ?", ("100",))
            .fetchone()
        )
        assert row["updated_at"] != ""


# ---------------------------------------------------------------------------
# Reaction feedback
# ---------------------------------------------------------------------------


class TestReactionFeedback:
    def test_classify_positive(self) -> None:
        from luke.db import classify_reaction

        assert classify_reaction("\U0001f44d") == "positive"
        assert classify_reaction("\u2764\ufe0f") == "positive"
        assert classify_reaction("\U0001f525") == "positive"

    def test_classify_negative(self) -> None:
        from luke.db import classify_reaction

        assert classify_reaction("\U0001f44e") == "negative"
        assert classify_reaction("\U0001f621") == "negative"

    def test_classify_neutral(self) -> None:
        from luke.db import classify_reaction

        assert classify_reaction("\U0001f914") == "neutral"
        assert classify_reaction("\U0001f60a") == "neutral"

    def test_store_reaction_feedback(self, test_db: Any) -> None:
        test_db.store_reaction_feedback(
            chat_id="100",
            msg_id=42,
            sender_id="1",
            emoji="\U0001f44d",
            timestamp="2024-01-01T00:00:00",
        )
        row = test_db._db().execute("SELECT * FROM reaction_feedback WHERE msg_id = 42").fetchone()
        assert row is not None
        assert row["sentiment"] == "positive"
        assert row["emoji"] == "\U0001f44d"

    def test_reaction_feedback_idempotent(self, test_db: Any) -> None:
        """Same reaction from same user on same message should not duplicate."""
        for _ in range(3):
            test_db.store_reaction_feedback(
                chat_id="100",
                msg_id=42,
                sender_id="1",
                emoji="\U0001f44d",
                timestamp="2024-01-01T00:00:00",
            )
        count = (
            test_db._db()
            .execute("SELECT COUNT(*) as cnt FROM reaction_feedback WHERE msg_id = 42")
            .fetchone()["cnt"]
        )
        assert count == 1

    def test_get_reactions_basic(self, test_db: Any) -> None:
        test_db.store_reaction_feedback(
            chat_id="100",
            msg_id=42,
            sender_id="1",
            emoji="\U0001f44d",
            timestamp="2024-01-01T00:00:00",
        )
        results = test_db.get_reactions("100")
        assert len(results) == 1
        assert results[0]["emoji"] == "\U0001f44d"
        assert results[0]["sentiment"] == "positive"

    def test_get_reactions_with_message_join(self, test_db: Any) -> None:
        test_db.store_message(
            chat_id="100",
            sender_name="Alice",
            sender_id="1",
            message_id=42,
            content="Hello world",
            timestamp="2024-01-01T00:00:00",
        )
        test_db.store_reaction_feedback(
            chat_id="100",
            msg_id=42,
            sender_id="1",
            emoji="\U0001f44d",
            timestamp="2024-01-01T00:01:00",
        )
        results = test_db.get_reactions("100")
        assert results[0]["msg_sender"] == "Alice"
        assert "Hello world" in results[0]["msg_preview"]

    def test_get_reactions_filter_sentiment(self, test_db: Any) -> None:
        test_db.store_reaction_feedback(
            chat_id="100",
            msg_id=1,
            sender_id="1",
            emoji="\U0001f44d",
            timestamp="2024-01-01T00:00:00",
        )
        test_db.store_reaction_feedback(
            chat_id="100",
            msg_id=2,
            sender_id="1",
            emoji="\U0001f44e",
            timestamp="2024-01-01T00:01:00",
        )
        pos = test_db.get_reactions("100", sentiment="positive")
        assert len(pos) == 1
        neg = test_db.get_reactions("100", sentiment="negative")
        assert len(neg) == 1

    def test_get_reactions_filter_msg_id(self, test_db: Any) -> None:
        test_db.store_reaction_feedback(
            chat_id="100",
            msg_id=42,
            sender_id="1",
            emoji="\U0001f44d",
            timestamp="2024-01-01T00:00:00",
        )
        test_db.store_reaction_feedback(
            chat_id="100",
            msg_id=99,
            sender_id="1",
            emoji="\U0001f525",
            timestamp="2024-01-01T00:01:00",
        )
        results = test_db.get_reactions("100", msg_id=42)
        assert len(results) == 1
        assert results[0]["msg_id"] == 42

    def test_get_reactions_empty(self, test_db: Any) -> None:
        results = test_db.get_reactions("100")
        assert results == []

    def test_get_reactions_limit(self, test_db: Any) -> None:
        for i in range(5):
            test_db.store_reaction_feedback(
                chat_id="100",
                msg_id=i,
                sender_id="1",
                emoji="\U0001f44d",
                timestamp=f"2024-01-0{i + 1}T00:00:00",
            )
        results = test_db.get_reactions("100", limit=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Batch commit context manager
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_defers_commit(self, test_db: Any) -> None:
        """Inside batch(), individual writes should not auto-commit."""
        with batch():
            test_db.store_message(
                chat_id="100", sender_name="A", content="msg1", timestamp="2024-01-01T00:00:00"
            )
            test_db.store_message(
                chat_id="100", sender_name="A", content="msg2", timestamp="2024-01-01T00:01:00"
            )
        # After batch exits, data should be committed
        msgs = test_db.get_pending_messages("100")
        assert len(msgs) == 2

    def test_batch_nested(self, test_db: Any) -> None:
        """Nested batch() should only commit at outermost exit."""
        with batch():
            test_db.store_message(
                chat_id="100", sender_name="A", content="outer", timestamp="2024-01-01T00:00:00"
            )
            with batch():
                test_db.store_message(
                    chat_id="100",
                    sender_name="A",
                    content="inner",
                    timestamp="2024-01-01T00:01:00",
                )
        msgs = test_db.get_pending_messages("100")
        assert len(msgs) == 2


# ---------------------------------------------------------------------------
# Behavior state
# ---------------------------------------------------------------------------


class TestBehaviorState:
    def test_get_none_initially(self, test_db: Any) -> None:
        assert test_db.get_behavior_last_run("consolidation") is None

    def test_set_and_get(self, test_db: Any) -> None:
        test_db.set_behavior_last_run("consolidation", "2024-01-01T00:00:00+00:00")
        assert test_db.get_behavior_last_run("consolidation") == "2024-01-01T00:00:00+00:00"

    def test_overwrite(self, test_db: Any) -> None:
        test_db.set_behavior_last_run("consolidation", "2024-01-01T00:00:00+00:00")
        test_db.set_behavior_last_run("consolidation", "2024-06-01T00:00:00+00:00")
        assert test_db.get_behavior_last_run("consolidation") == "2024-06-01T00:00:00+00:00"

    def test_multiple_behaviors(self, test_db: Any) -> None:
        test_db.set_behavior_last_run("consolidation", "2024-01-01T00:00:00+00:00")
        test_db.set_behavior_last_run("reflection", "2024-02-01T00:00:00+00:00")
        assert test_db.get_behavior_last_run("consolidation") == "2024-01-01T00:00:00+00:00"
        assert test_db.get_behavior_last_run("reflection") == "2024-02-01T00:00:00+00:00"
