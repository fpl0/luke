"""Tests for luke.scheduler — _is_due, _run_task, start_scheduler_loop."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from luke.db import TaskRecord
from luke.scheduler import _is_due, _run_task, start_scheduler_loop

_SEM = asyncio.Semaphore(5)


def _task(
    schedule_type: str = "cron",
    schedule_value: str = "*/5 * * * *",
    last_run: str | None = None,
    **overrides: object,
) -> TaskRecord:
    """Build a minimal TaskRecord for testing."""
    base: TaskRecord = {
        "id": "test-id",
        "chat_id": "100",
        "prompt": "do stuff",
        "schedule_type": schedule_type,
        "schedule_value": schedule_value,
        "status": "active",
        "last_run": last_run,
        "created_at": datetime.now(UTC).isoformat(),
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


class TestOnce:
    def test_not_due_before_time(self) -> None:
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        task = _task(schedule_type="once", schedule_value=future)
        assert _is_due(task, datetime.now(UTC)) is False

    def test_due_at_time(self) -> None:
        past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        task = _task(schedule_type="once", schedule_value=past)
        assert _is_due(task, datetime.now(UTC)) is True

    def test_already_run(self) -> None:
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        task = _task(schedule_type="once", schedule_value=past, last_run=past)
        assert _is_due(task, datetime.now(UTC)) is False


class TestCron:
    def test_not_due_immediately_after_creation(self) -> None:
        task = _task(schedule_type="cron", schedule_value="*/5 * * * *", last_run=None)
        assert _is_due(task, datetime.now(UTC)) is False

    def test_due_first_run_after_window(self) -> None:
        created = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        task = _task(
            schedule_type="cron", schedule_value="*/5 * * * *", last_run=None, created_at=created
        )
        assert _is_due(task, datetime.now(UTC)) is True

    def test_due_after_next_fire(self) -> None:
        last = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        task = _task(schedule_type="cron", schedule_value="*/5 * * * *", last_run=last)
        assert _is_due(task, datetime.now(UTC)) is True

    def test_not_due_before_next(self) -> None:
        # Use hourly cron so 30s ago is never near the next fire
        last = (datetime.now(UTC) - timedelta(seconds=30)).isoformat()
        task = _task(schedule_type="cron", schedule_value="0 * * * *", last_run=last)
        assert _is_due(task, datetime.now(UTC)) is False


class TestInterval:
    def test_not_due_immediately_after_creation(self) -> None:
        task = _task(schedule_type="interval", schedule_value="60000", last_run=None)
        assert _is_due(task, datetime.now(UTC)) is False

    def test_due_first_run_after_interval(self) -> None:
        created = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
        task = _task(
            schedule_type="interval", schedule_value="60000", last_run=None, created_at=created
        )
        assert _is_due(task, datetime.now(UTC)) is True

    def test_due_after_elapsed(self) -> None:
        last = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
        task = _task(schedule_type="interval", schedule_value="60000", last_run=last)
        assert _is_due(task, datetime.now(UTC)) is True

    def test_not_due_before_elapsed(self) -> None:
        last = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
        task = _task(schedule_type="interval", schedule_value="60000", last_run=last)
        assert _is_due(task, datetime.now(UTC)) is False


class TestUnknownType:
    def test_unknown_returns_false(self) -> None:
        task = _task(schedule_type="weekly", schedule_value="monday")
        assert _is_due(task, datetime.now(UTC)) is False


# ---------------------------------------------------------------------------
# _run_task
# ---------------------------------------------------------------------------


class TestRunTask:
    async def test_successful_run(self) -> None:
        mock_bot = AsyncMock()
        mock_result = MagicMock()
        mock_result.texts = ["Task output"]
        mock_result.sent_messages = 0

        task = _task(schedule_type="cron")

        with (
            patch("luke.scheduler.run_agent", return_value=mock_result) as mock_agent,
            patch("luke.scheduler.send_long_message", new_callable=AsyncMock) as mock_send,
            patch("luke.scheduler.db") as mock_db,
        ):
            await _run_task(task, mock_bot)

        mock_agent.assert_called_once()
        mock_db.log_task_run.assert_called_once()
        mock_db.update_task_last_run.assert_called_once()
        # Should send the output via send_long_message
        mock_send.assert_called()

    async def test_once_task_marked_completed(self) -> None:
        mock_bot = AsyncMock()
        mock_result = MagicMock()
        mock_result.texts = []

        task = _task(schedule_type="once", schedule_value=datetime.now(UTC).isoformat())

        with (
            patch("luke.scheduler.run_agent", return_value=mock_result),
            patch("luke.scheduler.db") as mock_db,
        ):
            await _run_task(task, mock_bot)

        mock_db.update_task_status.assert_called_once_with("test-id", "completed")

    async def test_failed_run_logs_error(self) -> None:
        mock_bot = AsyncMock()

        task = _task(schedule_type="cron")

        with (
            patch("luke.scheduler.run_agent", side_effect=RuntimeError("agent failed")),
            patch("luke.scheduler.db") as mock_db,
        ):
            await _run_task(task, mock_bot)

        mock_db.log_task_run.assert_called_once()
        # Check result is "error"
        assert mock_db.log_task_run.call_args[0][3] == "error"
        mock_db.update_task_last_run.assert_called_once()

    async def test_failed_once_task_marked_completed(self) -> None:
        mock_bot = AsyncMock()

        task = _task(schedule_type="once", schedule_value=datetime.now(UTC).isoformat())

        with (
            patch("luke.scheduler.run_agent", side_effect=RuntimeError("boom")),
            patch("luke.scheduler.db") as mock_db,
        ):
            await _run_task(task, mock_bot)

        mock_db.update_task_status.assert_called_once_with("test-id", "completed")


# ---------------------------------------------------------------------------
# start_scheduler_loop
# ---------------------------------------------------------------------------


class TestSchedulerLoop:
    async def test_immediate_shutdown(self) -> None:
        """Loop exits immediately when shutdown event is already set."""
        mock_bot = AsyncMock()
        shutdown = asyncio.Event()
        shutdown.set()

        with patch("luke.scheduler.db") as mock_db:
            mock_db.get_behavior_last_run.return_value = None
            # Should return quickly without hanging
            await asyncio.wait_for(
                start_scheduler_loop(mock_bot, _SEM, shutdown=shutdown),
                timeout=5.0,
            )

    async def test_shutdown_after_tick(self) -> None:
        """Loop runs one tick then exits on shutdown."""
        mock_bot = AsyncMock()
        shutdown = asyncio.Event()

        with (
            patch("luke.scheduler.settings") as mock_settings,
            patch("luke.scheduler.db") as mock_db,
        ):
            mock_settings.scheduler_interval = 0.01  # Very fast ticks
            mock_settings.cleanup_interval = 999999
            mock_settings.consolidation_interval = 999999
            mock_settings.reflection_interval = 999999
            mock_settings.proactive_scan_interval = 999999
            mock_settings.goal_execution_interval = 999999
            mock_db.get_due_tasks.return_value = []
            mock_db.get_behavior_last_run.return_value = None

            async def set_shutdown() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(start_scheduler_loop(mock_bot, _SEM, shutdown=shutdown))
                tg.create_task(set_shutdown())
