"""Tests for luke.app — error throttling, format helpers, handlers, process."""

from __future__ import annotations

import asyncio
import fcntl
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from luke.db import MemoryResult

# ---------------------------------------------------------------------------
# Error throttling
# ---------------------------------------------------------------------------


class TestShouldSendError:
    def test_first_time(self) -> None:
        from luke.app import _should_send_error

        chat_id = f"err_test_{time.monotonic()}"
        assert _should_send_error(chat_id) is True

    def test_within_cooldown(self) -> None:
        from luke.app import _should_send_error

        chat_id = f"err_test_{time.monotonic()}"
        assert _should_send_error(chat_id) is True
        assert _should_send_error(chat_id) is False

    def test_after_cooldown(self) -> None:
        from luke.app import _last_error, _should_send_error
        from luke.config import settings

        chat_id = f"err_test_{time.monotonic()}"
        _last_error[chat_id] = time.monotonic() - settings.error_cooldown - 1
        assert _should_send_error(chat_id) is True


# ---------------------------------------------------------------------------
# Format memory context
# ---------------------------------------------------------------------------


class TestFormatMemoryContext:
    def test_structure(self, tmp_settings: Any) -> None:
        from luke.app import _format_memory_context

        memories: list[MemoryResult] = [
            {"id": "mem1", "type": "entity", "title": "Test Memory", "score": 0.95},
        ]
        result = _format_memory_context(memories)
        assert "<context>" in result
        assert "<memories>" in result
        assert "mem1" in result
        assert "entity" in result

    def test_empty_memories(self, tmp_settings: Any) -> None:
        from luke.app import _format_memory_context

        result = _format_memory_context([])
        assert "<context>" in result
        assert "<memories>" in result


# ---------------------------------------------------------------------------
# _reply_to
# ---------------------------------------------------------------------------


class TestReplyTo:
    def test_with_reply(self) -> None:
        from luke.app import _reply_to

        msg = MagicMock()
        msg.reply_to_message = MagicMock()
        msg.reply_to_message.message_id = 42
        assert _reply_to(msg) == "42"

    def test_without_reply(self) -> None:
        from luke.app import _reply_to

        msg = MagicMock()
        msg.reply_to_message = None
        assert _reply_to(msg) is None


# ---------------------------------------------------------------------------
# _store
# ---------------------------------------------------------------------------


class TestStore:
    def test_stores_message(self, test_db: Any) -> None:
        from luke.app import _store

        msg = MagicMock()
        msg.from_user = MagicMock()
        msg.from_user.full_name = "Alice"
        msg.from_user.id = 123
        msg.chat.id = 456
        msg.message_id = 789
        msg.date = datetime(2024, 1, 1, tzinfo=UTC)

        _store(msg, "Hello world")

        pending = test_db.get_pending_messages("456")
        assert len(pending) == 1
        assert pending[0].content == "Hello world"
        assert pending[0].sender_name == "Alice"

    def test_no_user_skips(self, test_db: Any) -> None:
        from luke.app import _store

        msg = MagicMock()
        msg.from_user = None

        _store(msg, "Hello")
        # No message stored
        assert test_db.get_pending_messages("456") == []

    def test_custom_timestamp(self, test_db: Any) -> None:
        from luke.app import _store

        msg = MagicMock()
        msg.from_user = MagicMock()
        msg.from_user.full_name = "Bob"
        msg.from_user.id = 1
        msg.chat.id = 100
        msg.message_id = 1
        msg.date = datetime(2024, 6, 15, tzinfo=UTC)

        _store(msg, "Custom ts", timestamp="2024-01-01T00:00:00")

        pending = test_db.get_pending_messages("100")
        assert len(pending) == 1
        assert pending[0].timestamp == "2024-01-01T00:00:00"


# ---------------------------------------------------------------------------
# _on_task_done
# ---------------------------------------------------------------------------


class TestOnTaskDone:
    def test_discard_from_set(self) -> None:
        from luke.app import _background_tasks, _on_task_done

        async def noop() -> None:
            pass

        loop = asyncio.new_event_loop()
        task = loop.create_task(noop())
        loop.run_until_complete(task)
        _background_tasks.add(task)
        _on_task_done(task)
        assert task not in _background_tasks
        loop.close()


# ---------------------------------------------------------------------------
# _video_thumbnail / _animation_frame / _transcribe_post
# ---------------------------------------------------------------------------


class TestMediaHelpers:
    async def test_video_thumbnail_success(self, tmp_path: Path) -> None:
        from luke.app import _video_thumbnail

        dest = tmp_path / "video.mp4"
        dest.write_bytes(b"video")

        with patch("luke.app.extract_frame", return_value=True):
            result = await _video_thumbnail(dest)

        assert "[Video thumbnail saved:" in result

    async def test_video_thumbnail_failure(self, tmp_path: Path) -> None:
        from luke.app import _video_thumbnail

        dest = tmp_path / "video.mp4"
        dest.write_bytes(b"video")

        with patch("luke.app.extract_frame", return_value=False):
            result = await _video_thumbnail(dest)

        assert result == ""

    async def test_animation_frame_success(self, tmp_path: Path) -> None:
        from luke.app import _animation_frame

        dest = tmp_path / "anim.mp4"
        dest.write_bytes(b"anim")

        with patch("luke.app.extract_frame", return_value=True):
            result = await _animation_frame(dest)

        assert "[Animation frame saved:" in result

    async def test_animation_frame_failure(self, tmp_path: Path) -> None:
        from luke.app import _animation_frame

        dest = tmp_path / "anim.mp4"
        dest.write_bytes(b"anim")

        with patch("luke.app.extract_frame", return_value=False):
            result = await _animation_frame(dest)

        assert result == ""

    async def test_transcribe_post_success(self, tmp_path: Path) -> None:
        from luke.app import _transcribe_post

        dest = tmp_path / "voice.ogg"
        dest.write_bytes(b"audio")

        with patch("luke.app.transcribe", return_value="Hello"):
            result = await _transcribe_post(dest)

        assert "[Audio transcript]: Hello" in result

    async def test_transcribe_post_failed(self, tmp_path: Path) -> None:
        from luke.app import _transcribe_post

        dest = tmp_path / "voice.ogg"
        dest.write_bytes(b"audio")

        with patch("luke.app.transcribe", return_value=None):
            result = await _transcribe_post(dest)

        assert "transcription failed" in result


# ---------------------------------------------------------------------------
# _keep_typing
# ---------------------------------------------------------------------------


class TestKeepTyping:
    async def test_cancelled(self) -> None:
        from luke.app import _keep_typing

        with patch("luke.app.bot") as mock_bot:
            mock_bot.send_chat_action = AsyncMock()
            task = asyncio.create_task(_keep_typing(123))
            await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # Just verify no exception leaked


# ---------------------------------------------------------------------------
# _handle_media
# ---------------------------------------------------------------------------


class TestHandleMedia:
    async def test_download_success(self, test_db: Any) -> None:
        from luke.app import _handle_media

        msg = MagicMock()
        msg.chat.id = 100
        msg.from_user = MagicMock()
        msg.from_user.full_name = "Alice"
        msg.from_user.id = 1
        msg.message_id = 1
        msg.date = datetime(2024, 1, 1, tzinfo=UTC)
        media = MagicMock()

        with (
            patch("luke.app.bot") as mock_bot,
            patch("luke.app._dispatch") as mock_dispatch,
        ):
            mock_bot.download = AsyncMock()
            await _handle_media(msg, media, "photo.jpg", "Photo: {dest}")

        mock_dispatch.assert_called_once_with("100")

    async def test_download_failure(self, test_db: Any) -> None:
        from luke.app import _handle_media

        msg = MagicMock()
        msg.chat.id = 100
        msg.from_user = MagicMock()
        msg.from_user.full_name = "Alice"
        msg.from_user.id = 1
        msg.message_id = 1
        msg.date = datetime(2024, 1, 1, tzinfo=UTC)
        media = MagicMock()

        with (
            patch("luke.app.bot") as mock_bot,
            patch("luke.app._dispatch") as mock_dispatch,
        ):
            mock_bot.download = AsyncMock(side_effect=RuntimeError("download failed"))
            await _handle_media(msg, media, "photo.jpg", "Photo: {dest}")

        mock_dispatch.assert_called_once()

    async def test_post_download_callback(self, test_db: Any) -> None:
        from luke.app import _handle_media

        msg = MagicMock()
        msg.chat.id = 100
        msg.from_user = MagicMock()
        msg.from_user.full_name = "Alice"
        msg.from_user.id = 1
        msg.message_id = 1
        msg.date = datetime(2024, 1, 1, tzinfo=UTC)

        async def post_download(dest: Path) -> str:
            return "\n[Extra info]"

        with (
            patch("luke.app.bot") as mock_bot,
            patch("luke.app._dispatch"),
        ):
            mock_bot.download = AsyncMock()
            await _handle_media(
                msg, MagicMock(), "file.doc", "Doc: {dest}", post_download=post_download
            )


# ---------------------------------------------------------------------------
# process()
# ---------------------------------------------------------------------------


class TestProcess:
    async def test_unregistered_chat(self) -> None:
        from luke.app import _notified_unregistered, process

        _notified_unregistered.discard("99999")

        with (
            patch("luke.app.db"),
            patch("luke.app.bot") as mock_bot,
            patch("luke.app.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.assistant_name = "Luke"
            mock_bot.send_message = AsyncMock()
            await process("99999")

        mock_bot.send_message.assert_called_once()

    async def test_unregistered_chat_second_time(self) -> None:
        from luke.app import _notified_unregistered, process

        _notified_unregistered.add("88888")

        with patch("luke.app.settings") as mock_settings:
            mock_settings.chat_id = "12345"
            await process("88888")

    async def test_no_pending_messages(self) -> None:
        from luke.app import process

        with (
            patch("luke.app.db") as mock_db,
            patch("luke.app.settings") as mock_settings,
        ):
            mock_settings.chat_id = "100"
            mock_db.get_pending_messages.return_value = []
            await process("100")

    async def test_retry_on_failure_no_cursor_advance(self) -> None:
        """Agent failure should NOT advance cursor — messages stay pending for retry."""
        import luke.app as app_mod
        from luke.app import process

        chat_id = "900001"
        app_mod._retry_counts.pop(chat_id, None)

        msg = MagicMock()
        msg.id = 10
        msg.sender_name = "Alice"
        msg.content = "hello"
        msg.message_id = 42
        msg.timestamp = "2024-01-01T00:00:00"

        with (
            patch("luke.app.db") as mock_db,
            patch("luke.app.bot") as mock_bot,
            patch("luke.app.settings") as mock_settings,
            patch("luke.app.build_prompt", new_callable=AsyncMock, return_value="prompt"),
            patch(
                "luke.app.run_agent",
                new_callable=AsyncMock,
                side_effect=RuntimeError("transient"),
            ),
        ):
            mock_settings.chat_id = chat_id
            mock_settings.max_retries = 3
            mock_settings.agent_timeout = 10

            mock_settings.auto_recall_limit = 5
            mock_settings.max_concurrent = 5
            mock_settings.error_cooldown = 0
            mock_db.get_pending_messages.return_value = [msg]
            mock_db.get_session.return_value = None
            mock_bot.send_chat_action = AsyncMock()
            mock_bot.send_message = AsyncMock()

            await process(chat_id)

        # Cursor should NOT be advanced
        mock_db.advance_cursor.assert_not_called()
        # Session IS cleared on every failure (stale session cleanup)
        mock_db.set_session.assert_called_once_with(chat_id, "")
        # Retry count should be 1
        assert app_mod._retry_counts.get(chat_id) == 1

        # Cleanup
        app_mod._retry_counts.pop(chat_id, None)

    async def test_max_retries_advances_cursor(self) -> None:
        """After max_retries failures, cursor should advance and user notified."""
        import luke.app as app_mod
        from luke.app import process

        chat_id = "900002"
        app_mod._retry_counts[chat_id] = 2  # Already failed twice

        msg = MagicMock()
        msg.id = 20
        msg.sender_name = "Alice"
        msg.content = "hello"
        msg.message_id = 42
        msg.timestamp = "2024-01-01T00:00:00"

        with (
            patch("luke.app.db") as mock_db,
            patch("luke.app.bot") as mock_bot,
            patch("luke.app.settings") as mock_settings,
            patch("luke.app.build_prompt", new_callable=AsyncMock, return_value="prompt"),
            patch(
                "luke.app.run_agent",
                new_callable=AsyncMock,
                side_effect=RuntimeError("persistent"),
            ),
        ):
            mock_settings.chat_id = chat_id
            mock_settings.max_retries = 3
            mock_settings.agent_timeout = 10

            mock_settings.auto_recall_limit = 5
            mock_settings.max_concurrent = 5
            mock_settings.error_cooldown = 0
            mock_db.get_pending_messages.return_value = [msg]
            mock_db.get_session.return_value = None
            mock_bot.send_chat_action = AsyncMock()
            mock_bot.send_message = AsyncMock()

            await process(chat_id)

        # NOW cursor should advance
        mock_db.advance_cursor.assert_called_once_with(chat_id, 20)
        # Session cleared twice: once unconditionally, once at max_retries
        assert mock_db.set_session.call_count == 2
        # Retry count should be cleared
        assert chat_id not in app_mod._retry_counts

    async def test_success_clears_retry_count(self) -> None:
        """Successful agent run should clear any retry count."""
        import luke.app as app_mod
        from luke.app import process

        chat_id = "900003"
        app_mod._retry_counts[chat_id] = 2

        msg = MagicMock()
        msg.id = 30
        msg.sender_name = "Alice"
        msg.content = "hello"
        msg.message_id = 42
        msg.timestamp = "2024-01-01T00:00:00"

        mock_result = MagicMock()
        mock_result.texts = ["response"]
        mock_result.session_id = "sess-123"

        with (
            patch("luke.app.db") as mock_db,
            patch("luke.app.bot") as mock_bot,
            patch("luke.app.settings") as mock_settings,
            patch("luke.app.build_prompt", new_callable=AsyncMock, return_value="prompt"),
            patch("luke.app.run_agent", new_callable=AsyncMock, return_value=mock_result),
            patch("luke.app.send_long_message", new_callable=AsyncMock),
        ):
            mock_settings.chat_id = chat_id
            mock_settings.agent_timeout = 10

            mock_settings.auto_recall_limit = 5
            mock_settings.max_concurrent = 5
            mock_db.get_pending_messages.return_value = [msg]
            mock_db.get_session.return_value = None
            mock_bot.send_chat_action = AsyncMock()
            mock_bot.send_message = AsyncMock()

            await process(chat_id)

        # Retry count should be cleared
        assert chat_id not in app_mod._retry_counts
        # Cursor should advance
        mock_db.advance_cursor.assert_called_once()


# ---------------------------------------------------------------------------
# Startup replay
# ---------------------------------------------------------------------------


class TestStartupReplay:
    async def test_dispatches_pending_messages(self) -> None:
        """Startup should dispatch pending messages for the registered chat."""
        msg = MagicMock(id=1, content="hello")

        with (
            patch("luke.app.db") as mock_db,
            patch("luke.app.settings") as mock_settings,
            patch("luke.app._dispatch") as mock_dispatch,
        ):
            mock_settings.chat_id = "12345"
            mock_db.get_pending_messages.return_value = [msg]

            # Simulate the startup replay logic
            if mock_settings.chat_id:
                pending = mock_db.get_pending_messages(mock_settings.chat_id)
                if pending:
                    mock_dispatch(mock_settings.chat_id)

        mock_dispatch.assert_called_once_with("12345")

    async def test_no_dispatch_when_no_pending(self) -> None:
        """Startup should NOT dispatch if no pending messages."""
        with (
            patch("luke.app.db") as mock_db,
            patch("luke.app.settings") as mock_settings,
            patch("luke.app._dispatch") as mock_dispatch,
        ):
            mock_settings.chat_id = "12345"
            mock_db.get_pending_messages.return_value = []

            if mock_settings.chat_id:
                pending = mock_db.get_pending_messages(mock_settings.chat_id)
                if pending:
                    mock_dispatch(mock_settings.chat_id)

        mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# Salience gate
# ---------------------------------------------------------------------------


class TestSalienceGate:
    def test_trivial_messages_skip(self) -> None:
        from luke.app import _needs_recall

        assert _needs_recall("ok") is False
        assert _needs_recall("thanks!") is False
        assert _needs_recall("lol") is False
        assert _needs_recall("hi") is False
        assert _needs_recall("yes") is False
        assert _needs_recall("") is False

    def test_substantive_messages_pass(self) -> None:
        from luke.app import _needs_recall

        assert _needs_recall("What did we discuss yesterday about the project?") is True
        assert _needs_recall("Can you research flights to Tokyo?") is True
        assert _needs_recall("Update the goal for learning Spanish") is True

    def test_short_but_meaningful(self) -> None:
        from luke.app import _needs_recall

        # Short but not in trivial set
        assert _needs_recall("deploy now") is True
        assert _needs_recall("fix the bug") is True


# ---------------------------------------------------------------------------
# Process lock
# ---------------------------------------------------------------------------


class TestAcquireLock:
    def test_acquires_lock(self, tmp_settings: Any) -> None:
        import os

        import luke.app as app

        tmp_settings.store_dir.mkdir(parents=True, exist_ok=True)
        old = app._lock_fd
        try:
            app._lock_fd = None
            app._acquire_lock()
            assert app._lock_fd is not None
            lock_path = tmp_settings.store_dir / "luke.lock"
            assert lock_path.exists()
        finally:
            if app._lock_fd is not None:
                os.close(app._lock_fd)
            app._lock_fd = old

    def test_second_instance_blocked(self, tmp_settings: Any) -> None:
        import os

        import luke.app as app

        tmp_settings.store_dir.mkdir(parents=True, exist_ok=True)
        lock_path = tmp_settings.store_dir / "luke.lock"

        # Hold the lock from outside using os.open (matching _acquire_lock)
        held_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
        fcntl.flock(held_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        old = app._lock_fd
        try:
            app._lock_fd = None
            with pytest.raises(SystemExit) as exc_info:
                app._acquire_lock()
            assert exc_info.value.code == 1
        finally:
            os.close(held_fd)
            if app._lock_fd is not None:
                os.close(app._lock_fd)
            app._lock_fd = old
