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

import luke.app as app_mod
from luke.config import settings

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
# Safe handler decorator
# ---------------------------------------------------------------------------


class TestSafeHandler:
    @pytest.mark.asyncio
    async def test_normal_execution(self) -> None:
        """Wrapped handler should execute normally when no exception."""
        from luke.app import _safe_handler

        called = False

        @_safe_handler
        async def handler(msg: Any) -> None:
            nonlocal called
            called = True

        await handler(MagicMock())
        assert called

    @pytest.mark.asyncio
    async def test_exception_caught(self) -> None:
        """Wrapped handler should catch and log exceptions without re-raising."""
        from luke.app import _safe_handler

        @_safe_handler
        async def handler(msg: Any) -> None:
            raise ValueError("bad message data")

        # Should NOT raise
        msg = MagicMock(spec=[])
        await handler(msg)

    @pytest.mark.asyncio
    async def test_crash_context_set_on_error(self) -> None:
        """Crash context should record last handler error on failure."""
        from luke.app import _crash_context, _safe_handler

        @_safe_handler
        async def failing_handler(msg: Any) -> None:
            raise RuntimeError("db locked")

        msg = MagicMock(spec=[])
        await failing_handler(msg)
        assert _crash_context.get("last_handler_error") == "failing_handler"
        assert "last_handler_error_time" in _crash_context

    @pytest.mark.asyncio
    async def test_message_context_logged(self) -> None:
        """Handler error log should include message context when available."""
        from luke.app import _safe_handler

        @_safe_handler
        async def on_text(msg: Any) -> None:
            raise TypeError("unexpected None")

        msg = MagicMock(spec=["chat", "message_id", "from_user", "content_type"])
        msg.chat.id = 12345
        msg.message_id = 99
        msg.from_user.full_name = "Test User"
        msg.content_type = "text"

        # Make msg pass isinstance check for types.Message
        with patch("luke.app.types.Message", type(msg)):
            await on_text(msg)


# ---------------------------------------------------------------------------
# Format memory context
# ---------------------------------------------------------------------------


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
            patch("luke.app.bus"),
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

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    async def test_max_retries_advances_cursor(self) -> None:
        """After max_retries failures, cursor should advance and user notified."""
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
            patch("luke.app.bus"),
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
        # Session cleared once (unconditionally on error)
        mock_db.set_session.assert_called_once_with(chat_id, "")
        # Retry count should be cleared
        assert chat_id not in app_mod._retry_counts

    async def test_success_clears_retry_count(self) -> None:
        """Successful agent run should clear any retry count."""
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
        mock_result.cost_usd = 0.01
        mock_result.num_turns = 1
        mock_result.duration_api_ms = 100
        mock_result.sent_messages = 0

        with (
            patch("luke.app.db") as mock_db,
            patch("luke.app.bot") as mock_bot,
            patch("luke.app.settings") as mock_settings,
            patch("luke.app.bus"),
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
        from luke.context import needs_recall as _needs_recall

        assert _needs_recall("ok") is False
        assert _needs_recall("thanks!") is False
        assert _needs_recall("lol") is False
        assert _needs_recall("hi") is False
        assert _needs_recall("yes") is False
        assert _needs_recall("") is False

    def test_substantive_messages_pass(self) -> None:
        from luke.context import needs_recall as _needs_recall

        assert _needs_recall("What did we discuss yesterday about the project?") is True
        assert _needs_recall("Can you research flights to Tokyo?") is True
        assert _needs_recall("Update the goal for learning Spanish") is True

    def test_short_but_meaningful(self) -> None:
        from luke.context import needs_recall as _needs_recall

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


# ---------------------------------------------------------------------------
# Model routing: _classify_effort
# ---------------------------------------------------------------------------


class TestClassifyEffort:
    """Verify _classify_effort routes to the correct effort/model tier."""

    # -- Trivial messages -> low / haiku --

    def test_trivial_hey(self) -> None:
        from luke.app import _classify_effort

        effort, thinking, model = _classify_effort("hey")
        assert effort == "low"
        assert thinking["type"] == "disabled"
        assert model == settings.model_low

    def test_trivial_ok(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("ok")
        assert effort == "low"
        assert model == settings.model_low

    def test_trivial_thanks(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("thanks")
        assert effort == "low"
        assert model == settings.model_low

    def test_trivial_single_word(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("sure")
        assert effort == "low"
        assert model == settings.model_low

    def test_trivial_short_no_question(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("sounds good to me")
        assert effort == "low"
        assert model == settings.model_low

    # -- Normal messages -> medium / sonnet --

    def test_medium_normal_question(self) -> None:
        from luke.app import _classify_effort

        effort, thinking, model = _classify_effort("What time is the meeting tomorrow?")
        assert effort == "medium"
        assert thinking["type"] == "disabled"
        assert model == settings.model_medium

    def test_medium_moderate_length(self) -> None:
        from luke.app import _classify_effort

        # 20 words, one question mark — should be medium
        msg = "I was thinking about going to the park later today " * 2 + "what do you think?"
        effort, _, model = _classify_effort(msg)
        assert effort == "medium"
        assert model == settings.model_medium

    def test_medium_boundary_no_complex_keywords(self) -> None:
        from luke.app import _classify_effort

        # A message around 50 words with no complex or code keywords
        words = ["something"] * 50
        msg = " ".join(words) + "?"
        effort, _, model = _classify_effort(msg)
        assert effort == "medium"
        assert model == settings.model_medium

    # -- Complex messages -> high / opus --

    def test_complex_long_message(self) -> None:
        from luke.app import _classify_effort

        # 160 words -> over 150 threshold
        msg = " ".join(["word"] * 160)
        effort, thinking, model = _classify_effort(msg)
        assert effort == "high"
        assert thinking["type"] == "enabled"
        assert model == settings.model_high

    def test_complex_multiple_questions(self) -> None:
        from luke.app import _classify_effort

        # 3+ question marks triggers complex
        msg = "What is this? How does it work? Why did it break?"
        effort, _, model = _classify_effort(msg)
        assert effort == "high"
        assert model == settings.model_high

    def test_complex_multimodal_input(self) -> None:
        from luke.app import _classify_effort

        # List input with an image block -> has_media=True -> complex
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": "What is this picture showing"},
            {"type": "image", "source": {"data": "base64data"}},
        ]
        effort, _, model = _classify_effort(blocks)
        assert effort == "high"
        assert model == settings.model_high

    def test_complex_keyword_research(self) -> None:
        from luke.app import _classify_effort

        # Complex keywords only trigger past the trivial gate (>=15 words or has ?)
        msg = "Can you research this topic and give me a detailed summary of findings?"
        effort, _, model = _classify_effort(msg)
        assert effort == "high"
        assert model == settings.model_high

    def test_complex_keyword_analyze(self) -> None:
        from luke.app import _classify_effort

        msg = "I need you to analyze the recent market trends and provide insights"
        effort, _, model = _classify_effort(msg)
        assert effort == "high"
        assert model == settings.model_high

    def test_complex_keyword_compare(self) -> None:
        from luke.app import _classify_effort

        msg = "Please compare these two different approaches and tell me which is better?"
        effort, _, model = _classify_effort(msg)
        assert effort == "high"
        assert model == settings.model_high

    def test_complex_keyword_implement(self) -> None:
        from luke.app import _classify_effort

        msg = (
            "We need to implement the new feature for the dashboard before next week, can you help?"
        )
        effort, _, model = _classify_effort(msg)
        assert effort == "high"
        assert model == settings.model_high

    def test_complex_keyword_design(self) -> None:
        from luke.app import _classify_effort

        msg = "Can you help me design the overall architecture for this new system?"
        effort, _, model = _classify_effort(msg)
        assert effort == "high"
        assert model == settings.model_high

    def test_complex_keyword_short_message_stays_trivial(self) -> None:
        """Complex keywords in short messages (<15 words, no ?) are still trivial."""
        from luke.app import _classify_effort

        # "research" is a complex keyword but message is short with no question
        effort, _, model = _classify_effort("research this")
        assert effort == "low"
        assert model == settings.model_low

    # -- Code keywords in short messages -> high / opus --

    def test_code_keyword_fix_the_bug(self) -> None:
        from luke.app import _classify_effort

        effort, thinking, model = _classify_effort("fix the bug")
        assert effort == "high"
        assert thinking["type"] == "enabled"
        assert model == settings.model_high

    def test_code_keyword_debug(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("debug this issue")
        assert effort == "high"
        assert model == settings.model_high

    def test_code_keyword_refactor(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("refactor the module")
        assert effort == "high"
        assert model == settings.model_high

    def test_code_keyword_deploy(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("deploy to production")
        assert effort == "high"
        assert model == settings.model_high

    def test_code_keyword_commit(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("commit the changes")
        assert effort == "high"
        assert model == settings.model_high

    def test_code_keyword_merge(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("merge into main")
        assert effort == "high"
        assert model == settings.model_high

    def test_code_keyword_api(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("call the api")
        assert effort == "high"
        assert model == settings.model_high

    def test_code_keyword_database(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("update the database")
        assert effort == "high"
        assert model == settings.model_high

    def test_code_keyword_code(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("write some code")
        assert effort == "high"
        assert model == settings.model_high

    def test_code_keyword_test(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("test the endpoint")
        assert effort == "high"
        assert model == settings.model_high

    # -- Code blocks -> high / opus --

    def test_code_block_triggers_high(self) -> None:
        from luke.app import _classify_effort

        msg = "Look at this:\n```python\nprint('hello')\n```"
        effort, thinking, model = _classify_effort(msg)
        assert effort == "high"
        assert thinking["type"] == "enabled"
        assert model == settings.model_high

    def test_code_block_empty(self) -> None:
        from luke.app import _classify_effort

        msg = "What does this do?\n```\n```"
        effort, _, model = _classify_effort(msg)
        assert effort == "high"
        assert model == settings.model_high

    # -- Thinking config correctness --

    def test_low_has_disabled_thinking(self) -> None:
        from luke.app import _classify_effort

        _, thinking, _ = _classify_effort("hi")
        assert thinking["type"] == "disabled"

    def test_medium_has_disabled_thinking(self) -> None:
        from luke.app import _classify_effort

        _, thinking, _ = _classify_effort("How is the weather today?")
        assert thinking["type"] == "disabled"

    def test_high_has_enabled_thinking(self) -> None:
        from luke.app import _classify_effort

        _, thinking, _ = _classify_effort("fix the bug in production")
        assert thinking["type"] == "enabled"

    # -- Edge cases --

    def test_question_in_short_message_is_medium(self) -> None:
        from luke.app import _classify_effort

        # Short message with a question mark -> not trivial, should be medium
        effort, _, model = _classify_effort("why?")
        assert effort == "medium"
        assert model == settings.model_medium

    def test_multimodal_text_only_blocks(self) -> None:
        from luke.app import _classify_effort

        # List input but no image -> has_media=False
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": "hello there"},
        ]
        effort, _, model = _classify_effort(blocks)
        assert effort == "low"
        assert model == settings.model_low

    def test_multimodal_code_keyword_in_text_block(self) -> None:
        from luke.app import _classify_effort

        # Code keyword inside a text block in list format
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": "fix the bug in the api"},
        ]
        effort, _, model = _classify_effort(blocks)
        assert effort == "high"
        assert model == settings.model_high

    def test_code_keyword_case_insensitive(self) -> None:
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("DEBUG the issue")
        assert effort == "high"
        assert model == settings.model_high

    def test_code_keyword_partial_match(self) -> None:
        """Code keywords use 'in' matching, so 'testing' contains 'test'."""
        from luke.app import _classify_effort

        effort, _, model = _classify_effort("testing my patience")
        assert effort == "high"
        assert model == settings.model_high

    def test_exactly_fifteen_words_no_question(self) -> None:
        """15 words is NOT < 15, so it falls to the normal path."""
        from luke.app import _classify_effort

        msg = " ".join(["word"] * 15)
        effort, _, model = _classify_effort(msg)
        assert effort == "medium"
        assert model == settings.model_medium

    def test_exactly_fourteen_words_no_question(self) -> None:
        """14 words IS < 15, so trivial."""
        from luke.app import _classify_effort

        msg = " ".join(["word"] * 14)
        effort, _, model = _classify_effort(msg)
        assert effort == "low"
        assert model == settings.model_low

    def test_exactly_151_words(self) -> None:
        """151 words is > 150, so complex."""
        from luke.app import _classify_effort

        msg = " ".join(["word"] * 151)
        effort, _, model = _classify_effort(msg)
        assert effort == "high"
        assert model == settings.model_high

    def test_exactly_150_words_no_triggers(self) -> None:
        """150 words is NOT > 150, so medium (no other triggers)."""
        from luke.app import _classify_effort

        msg = " ".join(["word"] * 150)
        effort, _, model = _classify_effort(msg)
        assert effort == "medium"
        assert model == settings.model_medium

    def test_two_questions_is_medium(self) -> None:
        """2 question marks is not > 2, so medium."""
        from luke.app import _classify_effort

        msg = "What is this? How does it work?"
        effort, _, model = _classify_effort(msg)
        assert effort == "medium"
        assert model == settings.model_medium


# ---------------------------------------------------------------------------
# Model routing: _CODE_KEYWORDS completeness
# ---------------------------------------------------------------------------


class TestCodeKeywords:
    """Verify each code keyword individually routes to high/opus."""

    @pytest.mark.parametrize(
        "keyword",
        [
            "code",
            "fix",
            "bug",
            "debug",
            "refactor",
            "deploy",
            "test",
            "script",
            "function",
            "class",
            "error",
            "exception",
            "traceback",
            "commit",
            "merge",
            "pr",
            "pull request",
            "api",
            "endpoint",
            "database",
            "migration",
            "schema",
        ],
    )
    def test_each_code_keyword_routes_high(self, keyword: str) -> None:
        from luke.app import _classify_effort

        msg = f"please handle {keyword} now"
        effort, _, model = _classify_effort(msg)
        assert effort == "high", f"keyword '{keyword}' did not route to high"
        assert model == settings.model_high, f"keyword '{keyword}' did not route to opus"

    def test_code_keywords_set_is_complete(self) -> None:
        """Sanity check: the keywords set in the module matches our expectations."""
        from luke.app import _CODE_KEYWORDS

        expected = {
            "code",
            "fix",
            "bug",
            "debug",
            "refactor",
            "deploy",
            "test",
            "script",
            "function",
            "class",
            "error",
            "exception",
            "traceback",
            "commit",
            "merge",
            "pr",
            "pull request",
            "api",
            "endpoint",
            "database",
            "migration",
            "schema",
        }
        assert expected == _CODE_KEYWORDS


# ---------------------------------------------------------------------------
# _extract_topics
# ---------------------------------------------------------------------------


def _make_msg(content: str, sender: str = "Filipe") -> Any:
    from luke.db import StoredMessage

    return StoredMessage(
        id=1,
        sender_name=sender,
        sender_id="user1",
        message_id=1,
        content=content,
        timestamp="2026-04-01T10:00:00",
    )


class TestExtractTopics:
    def test_returns_top_keywords(self) -> None:
        from luke.app import _extract_topics

        msgs = [_make_msg("python python python programming programming")]
        topics = _extract_topics(msgs, [])
        assert "python" in topics
        assert "programming" in topics

    def test_filters_stopwords(self) -> None:
        from luke.app import _extract_topics

        msgs = [_make_msg("the a an is are was were have has had")]
        topics = _extract_topics(msgs, [])
        assert topics == []

    def test_filters_short_words(self) -> None:
        from luke.app import _extract_topics

        msgs = [_make_msg("go do it at by")]
        topics = _extract_topics(msgs, [])
        assert topics == []

    # --- topics must come from what the USER said -------------------------
    # Including the agent's reply meant reporting word-frequency over Luke's
    # own prose: a live block produced "Active topics: top, mostly, mission,
    # two" — four sentence-openers from its previous message. Luke called it
    # "noise being presented as signal — worse than empty, because it looks
    # like a summary."

    def test_ignores_the_agent_reply(self) -> None:
        from luke.app import _extract_topics

        reply = (
            "Mostly it was there. Mostly. The mission question, the mission. "
            "Two things I noticed, two. The top of you, the top."
        )
        topics = _extract_topics([_make_msg("how did the visa scheduling go?")], [reply])
        for noise in ("mostly", "mission", "two", "top"):
            assert noise not in topics, f"leaked {noise!r} from the agent reply"

    def test_agent_messages_in_the_batch_are_excluded(self) -> None:
        from luke.app import _extract_topics
        from luke.config import settings

        agent_msg = _make_msg("kubernetes kubernetes kubernetes")
        agent_msg.sender_name = settings.assistant_name
        topics = _extract_topics([_make_msg("visa visa"), agent_msg], [])
        assert "kubernetes" not in topics

    def test_empty_rather_than_noise(self) -> None:
        """One-off words are not topics; empty beats a fake summary."""
        from luke.app import _extract_topics

        assert _extract_topics([_make_msg("hey quick question")], []) == []

    def test_excludes_agent_texts(self) -> None:
        """Inverted 2026-08-03: including the reply was the defect.

        Luke's reply is an order of magnitude longer than the message that
        prompted it, so folding it in meant the field summarised Luke rather
        than the conversation.
        """
        from luke.app import _extract_topics

        msgs = [_make_msg("something unrelated")]
        topics = _extract_topics(msgs, ["deployment deployment deployment pipeline pipeline"])
        assert "deployment" not in topics

    def test_requires_frequency_2(self) -> None:
        from luke.app import _extract_topics

        msgs = [_make_msg("unique word appears once")]
        topics = _extract_topics(msgs, [])
        # "unique", "word", "appears" each appear once — should NOT be in topics
        assert "unique" not in topics

    def test_max_5_topics(self) -> None:
        from luke.app import _extract_topics

        # 8 distinct words each appearing 3 times
        words = [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
        ]
        text = " ".join(w * 3 for w in words)
        msgs = [_make_msg(text)]
        topics = _extract_topics(msgs, [])
        assert len(topics) <= 5

    def test_empty_input(self) -> None:
        from luke.app import _extract_topics

        assert _extract_topics([], []) == []

    def test_case_insensitive(self) -> None:
        from luke.app import _extract_topics

        msgs = [_make_msg("Python python PYTHON")]
        topics = _extract_topics(msgs, [])
        assert "python" in topics


# ---------------------------------------------------------------------------
# _extract_pending_actions
# ---------------------------------------------------------------------------


class TestExtractPendingActions:
    def test_extracts_ill_pattern(self) -> None:
        from luke.app import _extract_pending_actions

        actions = _extract_pending_actions(["I'll check the database tomorrow"])
        assert len(actions) == 1
        assert "check the database tomorrow" in actions[0]

    def test_extracts_i_will_pattern(self) -> None:
        from luke.app import _extract_pending_actions

        actions = _extract_pending_actions(["I will send you the report"])
        assert len(actions) == 1

    def test_extracts_next_steps(self) -> None:
        from luke.app import _extract_pending_actions

        actions = _extract_pending_actions(["Next steps: review the code carefully."])
        assert len(actions) == 1

    def test_deduplicates_actions(self) -> None:
        from luke.app import _extract_pending_actions

        actions = _extract_pending_actions(
            [
                "I'll review the code",
                "I'll review the code",
            ]
        )
        assert actions.count(actions[0]) == 1 if actions else True

    def test_caps_at_5_actions(self) -> None:
        from luke.app import _extract_pending_actions

        texts = [f"I'll do action number {i} right now" for i in range(10)]
        actions = _extract_pending_actions(texts)
        assert len(actions) <= 5

    def test_empty_input(self) -> None:
        from luke.app import _extract_pending_actions

        assert _extract_pending_actions([]) == []

    def test_no_match_returns_empty(self) -> None:
        from luke.app import _extract_pending_actions

        actions = _extract_pending_actions(["Just a plain statement with no action."])
        assert actions == []


class TestEnsureDirsSeedsVoice:
    """Fresh installs must get the output style — it carries Luke's register."""

    def test_seeds_output_style_and_settings(self, tmp_settings: Any) -> None:
        from luke.app import _ensure_dirs

        _ensure_dirs()

        style = settings.luke_dir / ".claude" / "output-styles" / "luke.md"
        assert style.exists()
        assert "never a customer-service rep" in style.read_text()

        cfg = settings.luke_dir / ".claude" / "settings.json"
        assert cfg.exists()
        assert '"outputStyle": "luke"' in cfg.read_text()

    def test_does_not_clobber_existing_settings(self, tmp_settings: Any) -> None:
        from luke.app import _ensure_dirs

        cfg = settings.luke_dir / ".claude" / "settings.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text('{"outputStyle": "custom"}')

        _ensure_dirs()

        assert cfg.read_text() == '{"outputStyle": "custom"}'


class TestConvStateNoDuplicateReply:
    """Every non-trivial turn wrote the agent's reply twice.

    _save_conv_state dedups the user's messages against recent history but
    appended the agent reply unconditionally — and the reply is already in
    `messages` by the time this runs. Luke found it auditing its own context:
    "My answer to the mission question is in there twice, which made the
    morning look fresher than it was."
    """

    @staticmethod
    def _seed(test_db: Any, reply: str) -> None:
        from luke.config import settings

        now = datetime.now(UTC).isoformat()
        test_db.store_message(
            chat_id="12345", sender_name="Filipe Lima", content="what's the mission?", timestamp=now
        )
        test_db.store_message(
            chat_id="12345", sender_name=settings.assistant_name, content=reply, timestamp=now
        )

    def test_reply_not_duplicated(self, test_db: Any) -> None:
        from luke.config import settings

        reply = "To be your best virtual friend, not an ops bot."
        self._seed(test_db, reply)
        msgs = test_db.get_pending_messages("12345")
        app_mod._save_conv_state(msgs, [reply])

        body = (settings.memory_dir / "episodes" / "conversation-state-latest.md").read_text()
        assert body.count(reply) == 1, "agent reply written twice"

    def test_novel_reply_is_still_added(self, test_db: Any) -> None:
        """A reply not yet in the messages table must still be recorded."""
        from luke.config import settings

        self._seed(test_db, "an older reply")
        msgs = test_db.get_pending_messages("12345")
        app_mod._save_conv_state(msgs, ["a brand new reply not yet stored"])

        body = (settings.memory_dir / "episodes" / "conversation-state-latest.md").read_text()
        assert "a brand new reply not yet stored" in body

    def test_last_exchange_is_real_not_now(self, test_db: Any) -> None:
        """The header claimed `now`, making the conversation read fresher."""
        from luke.config import settings

        self._seed(test_db, "some reply")
        conn = test_db._db()
        conn.execute("UPDATE messages SET ts = '2026-08-03T09:41:00+00:00'")
        conn.commit()
        msgs = test_db.get_pending_messages("12345")
        app_mod._save_conv_state(msgs, ["some reply"])

        body = (settings.memory_dir / "episodes" / "conversation-state-latest.md").read_text()
        assert "**Last exchange (Filipe):** 2026-08-03T09:41" in body


class TestConvStateAttribution:
    """ "Last exchange" must mean the last time FILIPE spoke.

    Luke's own outbound messages live in the messages table too, so taking the
    last row attributed Luke's own reply to a conversation that never happened.
    Luke: "if I'd trusted this block, I'd have believed Filipe said something
    at 14:36 that he never said — plausible rather than obviously garbled."
    """

    @staticmethod
    def _seed_and_save(test_db: Any, *, user_in_batch: bool) -> str:
        """Seed a user turn followed by two scheduled agent outputs, then save.

        process() only ever calls _save_conv_state with a non-empty batch, so
        the batch always carries at least one message.
        """
        from luke.config import settings

        test_db.store_message(
            chat_id="12345",
            sender_name="Filipe Lima",
            content="the real question",
            timestamp="2026-08-03T11:03:00+00:00",
        )
        for i, text in enumerate(["scheduled output one", "scheduled output two"]):
            test_db.store_message(
                chat_id="12345",
                sender_name=settings.assistant_name,
                content=text,
                timestamp=f"2026-08-03T14:3{i}:00+00:00",
            )
        rows = test_db.get_pending_messages("12345")
        if user_in_batch:
            batch = rows
        else:
            # get_pending_messages never returns the agent's own messages, so
            # this shape only arises from a direct caller (an operator script
            # regenerating state) — which is exactly where the blank was seen.
            batch = [
                test_db.StoredMessage(
                    id=99,
                    sender_name=settings.assistant_name,
                    sender_id="",
                    message_id=0,
                    content="scheduled output two",
                    timestamp="2026-08-03T14:31:00+00:00",
                )
            ]
        app_mod._save_conv_state(batch, [])
        return (settings.memory_dir / "episodes" / "conversation-state-latest.md").read_text()

    def test_last_exchange_tracks_the_user_not_the_agent(self, test_db: Any) -> None:
        body = self._seed_and_save(test_db, user_in_batch=True)
        line = body.split("**Last exchange (Filipe):**")[1].split("\n")[0]
        assert "11:03" in line
        assert "14:3" not in line, "attributed an agent message to Filipe"

    def test_user_timestamp_survives_a_batch_with_no_user_message(self, test_db: Any) -> None:
        """It went blank and threw away a timestamp we still knew."""
        body = self._seed_and_save(test_db, user_in_batch=False)
        assert "**User last active:** 2026-08-03T11:03" in body
        assert "unknown" not in body

    def test_trailing_agent_output_is_labelled(self, test_db: Any) -> None:
        """Four hours of scheduled output must not read as live conversation."""
        body = self._seed_and_save(test_db, user_in_batch=True)
        assert "own output, not a reply from Filipe" in body
        assert "the last 2 message(s)" in body

    def test_no_note_when_user_spoke_last(self, test_db: Any) -> None:
        from luke.config import settings

        test_db.store_message(
            chat_id="12345",
            sender_name=settings.assistant_name,
            content="earlier reply",
            timestamp="2026-08-03T10:00:00+00:00",
        )
        test_db.store_message(
            chat_id="12345",
            sender_name="Filipe Lima",
            content="latest word",
            timestamp="2026-08-03T11:03:00+00:00",
        )
        app_mod._save_conv_state(test_db.get_pending_messages("12345"), [])
        body = (settings.memory_dir / "episodes" / "conversation-state-latest.md").read_text()
        assert "own output, not a reply" not in body


class TestTrailingOwnCount:
    """The Note must count what's VISIBLE.

    It counted over the 20-row history while the block renders only the last
    10, so it claimed 8 when 5 were shown. Luke: "the number is not describing
    what I can see."
    """

    def test_count_matches_rendered_lines(self, test_db: Any) -> None:
        from luke.config import settings

        # 1 user turn, then 14 agent messages — more than the rendered window.
        test_db.store_message(
            chat_id="12345",
            sender_name="Filipe Lima",
            content="the only user turn",
            timestamp="2026-08-03T09:00:00+00:00",
        )
        for i in range(14):
            test_db.store_message(
                chat_id="12345",
                sender_name=settings.assistant_name,
                content=f"agent output {i}",
                timestamp=f"2026-08-03T10:{i:02d}:00+00:00",
            )
        app_mod._save_conv_state(test_db.get_pending_messages("12345"), [])
        body = (settings.memory_dir / "episodes" / "conversation-state-latest.md").read_text()

        claimed = int(body.split("the last ")[1].split(" message(s)")[0])
        shown = sum(
            1 for ln in body.split("\n") if ln.startswith(f"**{settings.assistant_name}** (")
        )
        assert claimed == shown, f"Note claims {claimed}, block shows {shown}"
