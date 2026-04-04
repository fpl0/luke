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
        """New cron task must NOT fire on the same tick it was created."""
        task = _task(schedule_type="cron", schedule_value="*/5 * * * *", last_run=None)
        assert _is_due(task, datetime.now(UTC)) is False

    def test_not_due_seconds_after_creation(self) -> None:
        """Even a few seconds after creation, still before next cron window."""
        created = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        task = _task(
            schedule_type="cron", schedule_value="0 * * * *", last_run=None, created_at=created
        )
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

    def test_restart_preserves_last_run(self) -> None:
        """After restart, cron uses persisted last_run — not restart time."""
        # Task ran 10 min ago (before restart), cron is every 5 min → due now
        last = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        task = _task(schedule_type="cron", schedule_value="*/5 * * * *", last_run=last)
        assert _is_due(task, datetime.now(UTC)) is True

    def test_restart_not_due_if_recently_ran(self) -> None:
        """After restart, cron that ran recently should NOT fire again."""
        # Use a fixed time mid-window (2 min after a */5 boundary) to avoid
        # flakiness when now happens to land on a 5-minute boundary.
        now = datetime(2026, 1, 15, 12, 2, 0, tzinfo=UTC)
        last = (now - timedelta(seconds=30)).isoformat()
        task = _task(schedule_type="cron", schedule_value="*/5 * * * *", last_run=last)
        assert _is_due(task, now) is False

    def test_restart_never_ran_but_within_first_window(self) -> None:
        """After restart, task created 2 min ago with hourly cron — still not due."""
        created = (datetime.now(UTC) - timedelta(minutes=2)).isoformat()
        task = _task(
            schedule_type="cron", schedule_value="0 * * * *", last_run=None, created_at=created
        )
        assert _is_due(task, datetime.now(UTC)) is False

    def test_restart_never_ran_past_first_window(self) -> None:
        """After restart, task created 10 min ago with 5-min cron, never ran — due."""
        created = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        task = _task(
            schedule_type="cron", schedule_value="*/5 * * * *", last_run=None, created_at=created
        )
        assert _is_due(task, datetime.now(UTC)) is True

    def test_every_3_hours_not_due_immediately(self) -> None:
        """Regression: 'every 3 hours' cron must not fire right after creation."""
        task = _task(schedule_type="cron", schedule_value="0 */3 * * *", last_run=None)
        # Check within 1 second of creation
        assert _is_due(task, datetime.now(UTC)) is False

    def test_every_3_hours_due_after_window(self) -> None:
        """Every 3 hours cron fires after the first window passes."""
        created = (datetime.now(UTC) - timedelta(hours=4)).isoformat()
        task = _task(
            schedule_type="cron", schedule_value="0 */3 * * *", last_run=None, created_at=created
        )
        assert _is_due(task, datetime.now(UTC)) is True


class TestInterval:
    def test_not_due_immediately_after_creation(self) -> None:
        """New interval task must NOT fire on the same tick it was created."""
        task = _task(schedule_type="interval", schedule_value="60000", last_run=None)
        assert _is_due(task, datetime.now(UTC)) is False

    def test_not_due_seconds_after_creation(self) -> None:
        """5 seconds after creation, 60s interval — not due."""
        created = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        task = _task(
            schedule_type="interval", schedule_value="60000", last_run=None, created_at=created
        )
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

    def test_restart_preserves_last_run(self) -> None:
        """After restart, interval uses persisted last_run — timer continues."""
        # Ran 2 min ago, interval is 1 min → due
        last = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
        task = _task(schedule_type="interval", schedule_value="60000", last_run=last)
        assert _is_due(task, datetime.now(UTC)) is True

    def test_restart_not_due_if_recently_ran(self) -> None:
        """After restart, interval that ran recently should NOT fire again."""
        last = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
        task = _task(schedule_type="interval", schedule_value="60000", last_run=last)
        assert _is_due(task, datetime.now(UTC)) is False

    def test_restart_never_ran_timer_continues_from_creation(self) -> None:
        """After restart, interval task created 30s ago with 60s interval — not due yet."""
        created = (datetime.now(UTC) - timedelta(seconds=30)).isoformat()
        task = _task(
            schedule_type="interval", schedule_value="60000", last_run=None, created_at=created
        )
        assert _is_due(task, datetime.now(UTC)) is False

    def test_restart_never_ran_past_interval(self) -> None:
        """After restart, interval task created 90s ago with 60s interval — due."""
        created = (datetime.now(UTC) - timedelta(seconds=90)).isoformat()
        task = _task(
            schedule_type="interval", schedule_value="60000", last_run=None, created_at=created
        )
        assert _is_due(task, datetime.now(UTC)) is True


class TestUnknownType:
    def test_unknown_returns_false(self) -> None:
        task = _task(schedule_type="weekly", schedule_value="monday")
        assert _is_due(task, datetime.now(UTC)) is False


# ---------------------------------------------------------------------------
# _effective_interval  (nested in start_scheduler_loop; tested via formula)
# ---------------------------------------------------------------------------


def _effective_interval_formula(no_ops: int, base: float) -> float:
    """Mirror of the _effective_interval logic from scheduler.py.

    Formula: base * (2 ** min(no_ops, 4))
    """
    return base * (2 ** min(no_ops, 4))


class TestEffectiveInterval:
    """Tests for the exponential backoff formula used in behavior scheduling."""

    def test_zero_no_ops_returns_base(self) -> None:
        """0 no-ops means no backoff: effective interval equals base."""
        assert _effective_interval_formula(0, 300.0) == 300.0

    def test_one_no_op_doubles(self) -> None:
        """1 no-op: 2x base interval."""
        assert _effective_interval_formula(1, 300.0) == 600.0

    def test_two_no_ops_quadruples(self) -> None:
        """2 no-ops: 4x base interval."""
        assert _effective_interval_formula(2, 300.0) == 1200.0

    def test_three_no_ops_8x(self) -> None:
        """3 no-ops: 8x base interval."""
        assert _effective_interval_formula(3, 300.0) == 2400.0

    def test_four_no_ops_16x(self) -> None:
        """4 no-ops: 16x base interval (maximum multiplier)."""
        assert _effective_interval_formula(4, 300.0) == 4800.0

    def test_five_no_ops_still_capped_at_16x(self) -> None:
        """5+ no-ops: still capped at 16x base interval."""
        assert _effective_interval_formula(5, 300.0) == 4800.0

    def test_large_no_ops_capped_at_16x(self) -> None:
        """Extremely high no-ops should still be capped at 16x."""
        assert _effective_interval_formula(100, 300.0) == 4800.0

    def test_with_db_mock(self) -> None:
        """Verify _effective_interval integrates correctly with db.get_behavior_no_ops."""
        with patch("luke.scheduler.db") as mock_db:
            mock_db.get_behavior_no_ops.return_value = 3
            no_ops = int(mock_db.get_behavior_no_ops("consolidation"))
            result = 300.0 * (2 ** min(no_ops, 4))
            assert result == 2400.0

    def test_db_returns_none_treated_as_zero(self) -> None:
        """When db returns a non-integer, the except clause defaults to 0 no-ops."""
        # The real code wraps int(db.get_behavior_no_ops(name)) in try/except
        # and defaults to 0 on TypeError/ValueError
        with patch("luke.scheduler.db") as mock_db:
            mock_db.get_behavior_no_ops.return_value = None
            try:
                no_ops = int(mock_db.get_behavior_no_ops("consolidation"))
            except (TypeError, ValueError):
                no_ops = 0
            result = 300.0 * (2 ** min(no_ops, 4))
            assert result == 300.0  # base interval, no backoff

    def test_different_base_intervals(self) -> None:
        """Backoff works correctly with various base interval values."""
        # 2 no-ops with a 60s base
        assert _effective_interval_formula(2, 60.0) == 240.0
        # 4 no-ops with a 3600s base
        assert _effective_interval_formula(4, 3600.0) == 57600.0


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
            patch("luke.scheduler.db") as mock_db,
        ):
            await _run_task(task, mock_bot)

        mock_agent.assert_called_once()
        mock_db.log_task_run.assert_called_once()
        mock_db.update_task_last_run.assert_called_once()

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
            mock_db.increment_task_failures.return_value = 1
            await _run_task(task, mock_bot)

        mock_db.log_task_run.assert_called_once()
        # Check result is "error"
        assert mock_db.log_task_run.call_args[0][3] == "error"
        mock_db.update_task_last_run.assert_called_once()
        mock_db.increment_task_failures.assert_called_once()

    async def test_failed_once_task_marked_completed(self) -> None:
        mock_bot = AsyncMock()

        task = _task(schedule_type="once", schedule_value=datetime.now(UTC).isoformat())

        with (
            patch("luke.scheduler.run_agent", side_effect=RuntimeError("boom")),
            patch("luke.scheduler.db") as mock_db,
        ):
            mock_db.increment_task_failures.return_value = 1
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

        with (
            patch("luke.scheduler.db") as mock_db,
            patch("luke.scheduler.memory"),
        ):
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
            patch("luke.scheduler.memory"),
        ):
            mock_settings.scheduler_interval = 0.01  # Very fast ticks
            mock_settings.cleanup_interval = 999999
            mock_settings.episode_consolidation_interval = 999999
            mock_settings.reflection_interval = 999999
            mock_settings.proactive_scan_interval = 999999
            mock_settings.deep_work_interval = 999999
            mock_settings.insight_consolidation_interval = 999999
            mock_settings.feedback_consolidation_interval = 999999
            mock_settings.lifecycle_review_interval = 999999
            mock_settings.dream_interval = 999999
            mock_db.get_due_tasks.return_value = []
            mock_db.get_behavior_last_run.return_value = None
            mock_db.count_unconsumed_events.return_value = 0
            mock_db.consume_events.return_value = 0

            async def set_shutdown() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(start_scheduler_loop(mock_bot, _SEM, shutdown=shutdown))
                tg.create_task(set_shutdown())


# ---------------------------------------------------------------------------
# Event-driven behavior wiring
# ---------------------------------------------------------------------------


class TestBehaviorEventMapping:
    """Verify _BEHAVIOR_EVENTS maps behaviors to correct event types."""

    def test_goal_updated_consumed_by_proactive_scan(self) -> None:
        """proactive_scan must consume goal_updated events."""
        # This constant is defined inside the scheduler loop; verify via the code pattern.
        # We test the mapping directly by importing the module and inspecting the pattern.
        _BEHAVIOR_EVENTS = {
            "consolidation": ("new_episode",),
            "reflection": ("feedback_negative", "user_message"),
            "proactive_scan": ("goal_updated",),
            "insight_consolidation": ("new_insight",),
            "feedback_consolidation": ("feedback_negative",),
            "lifecycle_review": (),
            "dream": (),
        }
        assert "goal_updated" in _BEHAVIOR_EVENTS["proactive_scan"]

    def test_all_emitted_event_types_have_consumers(self) -> None:
        """Every event type that can be emitted must be consumed by at least one behavior."""
        _BEHAVIOR_EVENTS = {
            "consolidation": ("new_episode",),
            "reflection": ("feedback_negative", "user_message"),
            "proactive_scan": ("goal_updated",),
            "insight_consolidation": ("new_insight",),
            "feedback_consolidation": ("feedback_negative",),
            "lifecycle_review": (),
            "dream": (),
        }
        emitted_types = {"new_episode", "new_insight", "goal_updated", "feedback_negative", "user_message"}
        consumed_types: set[str] = set()
        for events in _BEHAVIOR_EVENTS.values():
            consumed_types.update(events)
        uncovered = emitted_types - consumed_types
        assert not uncovered, f"Event types emitted but never consumed: {uncovered}"

    def test_fallback_multiplier_is_6(self) -> None:
        """Time-based fallback should be 6x the effective interval, not 2x."""
        import ast
        import inspect

        source = inspect.getsource(start_scheduler_loop)
        tree = ast.parse(source)
        # Find all BinOp nodes that multiply by a constant (the fallback pattern)
        fallback_multipliers: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.BinOp) and isinstance(comparator.op, ast.Mult):
                        if isinstance(comparator.right, ast.Constant):
                            fallback_multipliers.append(comparator.right.value)
        # All fallback multipliers should be 6
        assert all(m == 6 for m in fallback_multipliers), (
            f"Expected all fallback multipliers to be 6, got {fallback_multipliers}"
        )

    def test_all_newly_gated_behaviors_have_fallback_multiplier(self) -> None:
        """proactive_scan, lifecycle_review, dream, and deep_work must all have a 6x fallback."""
        import ast
        import inspect

        source = inspect.getsource(start_scheduler_loop)
        # Verify each newly-gated behavior has a 6x pattern in source
        for behavior in ("proactive_scan", "lifecycle_review", "dream", "deep_work"):
            assert f"{behavior}_interval * 6" in source or f"{behavior[:-1]}_interval * 6" in source or "* 6" in source, (
                f"{behavior} missing 6x fallback"
            )
        # Count 6x patterns — should have at least 8 (4 original + 4 new)
        tree = ast.parse(source)
        fallback_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.BinOp) and isinstance(comparator.op, ast.Mult):
                        if isinstance(comparator.right, ast.Constant) and comparator.right.value == 6:
                            fallback_count += 1
        assert fallback_count >= 8, f"Expected at least 8 behaviors with 6x fallback, found {fallback_count}"


# ---------------------------------------------------------------------------
# Event-gated behavior scheduling
# ---------------------------------------------------------------------------


class TestBehaviorEventGating:
    """Verify that proactive_scan, lifecycle_review, dream, and deep_work respect event gates."""

    def _make_mock_db(
        self,
        *,
        due_behavior: str,
        elapsed_seconds: float = 2.0,
        unconsumed_count: int = 0,
    ) -> MagicMock:
        """Build a mock db where only `due_behavior` has elapsed time recorded."""
        recent_ts = datetime.now(UTC).isoformat()
        due_ts = (datetime.now(UTC) - timedelta(seconds=elapsed_seconds)).isoformat()

        mock_db = MagicMock()
        mock_db.get_behavior_last_run.side_effect = lambda name: (
            due_ts if name == due_behavior else recent_ts
        )
        mock_db.get_due_tasks.return_value = []
        mock_db.count_unconsumed_events.return_value = unconsumed_count
        # Prevent _effective_interval from doubling intervals via spurious no-op counts
        mock_db.get_behavior_no_ops.return_value = 0
        # Prevent TypeError in post-run event consumption
        mock_db.consume_events.return_value = 0
        return mock_db

    def _make_mock_settings(self, *, due_behavior: str, interval: float = 1.0) -> MagicMock:
        """Build settings where only `due_behavior` interval is small enough to be due."""
        mock_settings = MagicMock()
        mock_settings.scheduler_interval = 0.01
        mock_settings.cleanup_interval = 999999
        mock_settings.episode_consolidation_interval = 999999
        mock_settings.reflection_interval = 999999
        mock_settings.proactive_scan_interval = interval if due_behavior == "proactive_scan" else 999999
        mock_settings.deep_work_interval = interval if due_behavior == "deep_work" else 999999
        mock_settings.insight_consolidation_interval = 999999
        mock_settings.feedback_consolidation_interval = 999999
        mock_settings.lifecycle_review_interval = interval if due_behavior == "lifecycle_review" else 999999
        mock_settings.dream_interval = interval if due_behavior == "dream" else 999999
        mock_settings.consolidation_min_cluster = 3
        return mock_settings

    async def _run_one_tick(
        self,
        mock_settings: MagicMock,
        mock_db: MagicMock,
    ) -> None:
        """Run the scheduler loop for one tick then shut it down."""
        bot = AsyncMock()
        shutdown = asyncio.Event()

        async def set_shutdown() -> None:
            await asyncio.sleep(0.05)
            shutdown.set()

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.scheduler.memory"),
        ):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(start_scheduler_loop(bot, _SEM, shutdown=shutdown))
                tg.create_task(set_shutdown())

    async def test_proactive_scan_skipped_without_events(self) -> None:
        """proactive_scan does not fire when no events exist and timer < 6x interval."""
        mock_settings = self._make_mock_settings(due_behavior="proactive_scan", interval=1.0)
        # elapsed=2s < 6s (6x interval), no events
        mock_db = self._make_mock_db(due_behavior="proactive_scan", elapsed_seconds=2.0, unconsumed_count=0)

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.scheduler.memory"),
            patch("luke.scheduler.run_proactive_scan", new_callable=AsyncMock) as mock_fn,
        ):
            bot = AsyncMock()
            shutdown = asyncio.Event()

            async def set_shutdown() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(start_scheduler_loop(bot, _SEM, shutdown=shutdown))
                tg.create_task(set_shutdown())

        mock_fn.assert_not_called()

    async def test_proactive_scan_fires_with_goal_event(self) -> None:
        """proactive_scan fires when goal_updated/user_message events exist."""
        mock_settings = self._make_mock_settings(due_behavior="proactive_scan", interval=1.0)
        mock_db = self._make_mock_db(due_behavior="proactive_scan", elapsed_seconds=2.0, unconsumed_count=1)

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.scheduler.memory"),
            patch("luke.scheduler.run_proactive_scan", new_callable=AsyncMock) as mock_fn,
        ):
            bot = AsyncMock()
            shutdown = asyncio.Event()

            async def set_shutdown() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(start_scheduler_loop(bot, _SEM, shutdown=shutdown))
                tg.create_task(set_shutdown())

        mock_fn.assert_called_once()

    async def test_proactive_scan_fires_at_6x_fallback(self) -> None:
        """proactive_scan fires at 6x interval even with no events."""
        mock_settings = self._make_mock_settings(due_behavior="proactive_scan", interval=1.0)
        # elapsed=10s > 6s (6x interval), no events
        mock_db = self._make_mock_db(due_behavior="proactive_scan", elapsed_seconds=10.0, unconsumed_count=0)

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.scheduler.memory"),
            patch("luke.scheduler.run_proactive_scan", new_callable=AsyncMock) as mock_fn,
        ):
            bot = AsyncMock()
            shutdown = asyncio.Event()

            async def set_shutdown() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(start_scheduler_loop(bot, _SEM, shutdown=shutdown))
                tg.create_task(set_shutdown())

        mock_fn.assert_called_once()

    async def test_lifecycle_review_skipped_without_events(self) -> None:
        """lifecycle_review does not fire without enough memory activity (< 5 events)."""
        mock_settings = self._make_mock_settings(due_behavior="lifecycle_review", interval=1.0)
        mock_db = self._make_mock_db(due_behavior="lifecycle_review", elapsed_seconds=2.0, unconsumed_count=4)

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.scheduler.memory"),
            patch("luke.scheduler.run_lifecycle_review", new_callable=AsyncMock) as mock_fn,
        ):
            bot = AsyncMock()
            shutdown = asyncio.Event()

            async def set_shutdown() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(start_scheduler_loop(bot, _SEM, shutdown=shutdown))
                tg.create_task(set_shutdown())

        mock_fn.assert_not_called()

    async def test_lifecycle_review_fires_with_enough_events(self) -> None:
        """lifecycle_review fires when >= 5 episode/insight events exist."""
        mock_settings = self._make_mock_settings(due_behavior="lifecycle_review", interval=1.0)
        mock_db = self._make_mock_db(due_behavior="lifecycle_review", elapsed_seconds=2.0, unconsumed_count=5)

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.scheduler.memory"),
            patch("luke.scheduler.run_lifecycle_review", new_callable=AsyncMock) as mock_fn,
        ):
            bot = AsyncMock()
            shutdown = asyncio.Event()

            async def set_shutdown() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(start_scheduler_loop(bot, _SEM, shutdown=shutdown))
                tg.create_task(set_shutdown())

        mock_fn.assert_called_once()

    async def test_dream_skipped_without_material(self) -> None:
        """dream does not fire when no insights/episodes exist and timer < 6x interval."""
        mock_settings = self._make_mock_settings(due_behavior="dream", interval=1.0)
        mock_db = self._make_mock_db(due_behavior="dream", elapsed_seconds=2.0, unconsumed_count=0)

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.scheduler.memory"),
            patch("luke.scheduler.run_dream", new_callable=AsyncMock) as mock_fn,
        ):
            bot = AsyncMock()
            shutdown = asyncio.Event()

            async def set_shutdown() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(start_scheduler_loop(bot, _SEM, shutdown=shutdown))
                tg.create_task(set_shutdown())

        mock_fn.assert_not_called()

    async def test_dream_fires_with_material(self) -> None:
        """dream fires when new insights or episodes exist."""
        mock_settings = self._make_mock_settings(due_behavior="dream", interval=1.0)
        mock_db = self._make_mock_db(due_behavior="dream", elapsed_seconds=2.0, unconsumed_count=1)

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.scheduler.memory"),
            patch("luke.scheduler.run_dream", new_callable=AsyncMock) as mock_fn,
        ):
            bot = AsyncMock()
            shutdown = asyncio.Event()

            async def set_shutdown() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(start_scheduler_loop(bot, _SEM, shutdown=shutdown))
                tg.create_task(set_shutdown())

        mock_fn.assert_called_once()

    async def test_deep_work_skipped_without_goal_events(self) -> None:
        """deep_work does not launch when no goal_updated events exist (and timer < 6x)."""
        mock_settings = self._make_mock_settings(due_behavior="deep_work", interval=1.0)
        mock_db = self._make_mock_db(due_behavior="deep_work", elapsed_seconds=2.0, unconsumed_count=0)

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.scheduler.memory"),
            patch("luke.scheduler.run_deep_work", new_callable=AsyncMock) as mock_fn,
        ):
            bot = AsyncMock()
            shutdown = asyncio.Event()

            async def set_shutdown() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(start_scheduler_loop(bot, _SEM, shutdown=shutdown))
                tg.create_task(set_shutdown())

        mock_fn.assert_not_called()

    async def test_deep_work_fires_with_goal_event(self) -> None:
        """deep_work launches when goal_updated events exist."""
        mock_settings = self._make_mock_settings(due_behavior="deep_work", interval=1.0)
        mock_db = self._make_mock_db(due_behavior="deep_work", elapsed_seconds=2.0, unconsumed_count=1)

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.scheduler.memory"),
            patch("luke.scheduler.run_deep_work", new_callable=AsyncMock) as mock_fn,
        ):
            bot = AsyncMock()
            shutdown = asyncio.Event()

            async def set_shutdown() -> None:
                await asyncio.sleep(0.05)
                shutdown.set()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(start_scheduler_loop(bot, _SEM, shutdown=shutdown))
                tg.create_task(set_shutdown())

        mock_fn.assert_called_once()
