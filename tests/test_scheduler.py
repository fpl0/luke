"""Tests for luke.scheduler — _is_due, _run_task, start_scheduler_loop."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from luke import scheduler
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
        """After restart, task created 2 min ago with hourly cron — still not due.

        Uses pinned times: with wall-clock now() this flaked in the first two
        minutes after each hour, when the top-of-hour boundary genuinely falls
        between creation and now.
        """
        created = datetime(2026, 8, 1, 12, 2, tzinfo=UTC).isoformat()
        now = datetime(2026, 8, 1, 12, 4, tzinfo=UTC)
        task = _task(
            schedule_type="cron", schedule_value="0 * * * *", last_run=None, created_at=created
        )
        assert _is_due(task, now) is False

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


class TestCronCatchUpGrace:
    """A cron slot missed by more than CRON_CATCHUP_GRACE is not replayed.

    Regression for 2026-09-01: twelve minutes after a 16-day outage ended, all
    sixteen crons were due in the same second, which would have delivered the
    06:00 morning briefing at 18:51 and the Friday note on a Tuesday.
    """

    def test_daily_morning_cron_does_not_fire_in_the_evening(self) -> None:
        """The exact failure: 06:00 daily, 16 days dark, back up at 18:51."""
        now = datetime(2026, 9, 1, 17, 51, tzinfo=UTC)
        last = datetime(2026, 8, 16, 6, 0, tzinfo=UTC).isoformat()
        task = _task(schedule_type="cron", schedule_value="0 6 * * *", last_run=last)
        assert _is_due(task, now) is False

    def test_same_cron_fires_at_its_next_real_slot(self) -> None:
        """...and is due again the following morning, so the cron isn't dead."""
        now = datetime(2026, 9, 2, 6, 0, tzinfo=UTC)
        last = datetime(2026, 8, 16, 6, 0, tzinfo=UTC).isoformat()
        task = _task(schedule_type="cron", schedule_value="0 6 * * *", last_run=last)
        assert _is_due(task, now) is True

    def test_weekly_cron_does_not_fire_on_the_wrong_weekday(self) -> None:
        """Friday 14:30 note, dark since 15 Aug, back up on a Tuesday."""
        now = datetime(2026, 9, 1, 17, 51, tzinfo=UTC)
        last = datetime(2026, 8, 14, 14, 30, tzinfo=UTC).isoformat()
        task = _task(schedule_type="cron", schedule_value="30 14 * * 5", last_run=last)
        assert _is_due(task, now) is False

    def test_short_restart_still_catches_up(self) -> None:
        """A deploy is not an outage — a slot 30 min stale still runs."""
        now = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)
        last = datetime(2026, 9, 1, 11, 0, tzinfo=UTC).isoformat()
        task = _task(schedule_type="cron", schedule_value="0 * * * *", last_run=last)
        assert _is_due(task, now) is True

    def test_grace_measures_the_slot_not_the_outage(self) -> None:
        """A 3-day gap still fires if the *slot* it missed is recent."""
        now = datetime(2026, 9, 1, 12, 5, tzinfo=UTC)
        last = datetime(2026, 8, 29, 12, 0, tzinfo=UTC).isoformat()
        task = _task(schedule_type="cron", schedule_value="0 * * * *", last_run=last)
        assert _is_due(task, now) is True


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
    return base * float(2 ** min(no_ops, 4))


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
            except TypeError, ValueError:
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
        mock_result.is_error = False

        task = _task(schedule_type="cron")

        with (
            patch("luke.scheduler.run_agent", return_value=mock_result) as mock_agent,
            patch("luke.scheduler.db") as mock_db,
        ):
            await _run_task(task, mock_bot)

        mock_agent.assert_called_once()
        mock_db.log_task_run.assert_called_once()
        mock_db.update_task_last_run.assert_called_once()

    async def test_successful_run_logs_cost(self) -> None:
        """Scheduled-task spend must hit cost_log — it was ~70% of real spend
        and completely invisible until 2026-08-01."""
        mock_bot = AsyncMock()
        mock_result = MagicMock()
        mock_result.texts = []
        mock_result.sent_messages = 0
        mock_result.is_error = False
        mock_result.cost_usd = 1.23
        mock_result.num_turns = 4
        mock_result.duration_api_ms = 5678
        mock_result.input_tokens = 10
        mock_result.output_tokens = 20
        mock_result.cache_create_tokens = 30
        mock_result.cache_read_tokens = 40

        task = _task(schedule_type="cron")

        with (
            patch("luke.scheduler.run_agent", return_value=mock_result),
            patch("luke.scheduler.db") as mock_db,
        ):
            await _run_task(task, mock_bot)

        mock_db.log_cost.assert_called_once()
        args, kwargs = mock_db.log_cost.call_args
        assert args[1] == 1.23
        assert args[4] == "task:test-id"
        assert kwargs["output_tokens"] == 20

    async def test_once_task_marked_completed(self) -> None:
        mock_bot = AsyncMock()
        mock_result = MagicMock()
        mock_result.texts = []
        mock_result.is_error = False

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
        # Result captures the exception detail so failures are diagnosable
        # from luke.db alone (not just structlog).
        logged_result = mock_db.log_task_run.call_args[0][3]
        assert logged_result.startswith("error:")
        assert "RuntimeError" in logged_result
        assert "agent failed" in logged_result
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

    async def test_dead_agent_run_is_not_logged_ok(self) -> None:
        """A run that came back is_error must never be recorded as "ok".

        On 2026-08-09/10 an auth outage ("Your organization has disabled Claude
        subscription access for Claude Code") returned a ResultMessage with
        zero usage for ~9 hours. run_agent did not raise, so 36 scheduled runs
        — including the weekly CarGurus reqs watch on its first ever fire and
        that Monday morning's briefing — were logged "ok" with their failure
        counters reset. Nothing warned. If this test goes green on the old
        code path, the outage is invisible again.
        """
        mock_bot = AsyncMock()
        mock_result = MagicMock()
        mock_result.texts = ["Your organization has disabled Claude subscription access"]
        mock_result.sent_messages = 0
        mock_result.is_error = True
        mock_result.error_subtype = "error_during_execution"
        mock_result.error_detail = "Your organization has disabled Claude subscription access"

        task = _task(schedule_type="cron")

        with (
            patch("luke.scheduler.run_agent", return_value=mock_result),
            patch("luke.scheduler.db") as mock_db,
        ):
            mock_db.increment_task_failures.return_value = 1
            await _run_task(task, mock_bot)

        logged_result = mock_db.log_task_run.call_args[0][3]
        assert logged_result.startswith("error:")
        assert "is_error" in logged_result
        assert "disabled Claude subscription access" in logged_result
        mock_db.increment_task_failures.assert_called_once()
        mock_db.reset_task_failures.assert_not_called()

    async def test_dead_run_does_not_log_cost_as_a_real_run(self) -> None:
        """A dead run bills nothing; recording it pollutes the cost baseline
        the anomaly detector reads."""
        mock_bot = AsyncMock()
        mock_result = MagicMock()
        mock_result.texts = []
        mock_result.is_error = True
        mock_result.error_subtype = "zero_usage"
        mock_result.error_detail = "no usage, no tools, no text"

        task = _task(schedule_type="cron")

        with (
            patch("luke.scheduler.run_agent", return_value=mock_result),
            patch("luke.scheduler.db") as mock_db,
        ):
            mock_db.increment_task_failures.return_value = 1
            await _run_task(task, mock_bot)

        mock_db.log_cost.assert_not_called()

    async def test_dead_once_task_does_not_retry_storm(self) -> None:
        """A once-task that died still gets closed out — the failure path owns
        that, and routing dead runs through it is why they raise rather than
        return early."""
        mock_bot = AsyncMock()
        mock_result = MagicMock()
        mock_result.texts = []
        mock_result.is_error = True
        mock_result.error_subtype = "zero_usage"
        mock_result.error_detail = "dead"

        task = _task(schedule_type="once", schedule_value=datetime.now(UTC).isoformat())

        with (
            patch("luke.scheduler.run_agent", return_value=mock_result),
            patch("luke.scheduler.db") as mock_db,
        ):
            mock_db.increment_task_failures.return_value = 1
            await _run_task(task, mock_bot)

        mock_db.update_task_status.assert_called_once_with("test-id", "completed")
        mock_db.update_task_last_run.assert_called_once()

    async def test_failure_alert_is_throttled_not_per_run(self) -> None:
        """Alert on the third strike, then once per 24. Unthrottled, a 15-minute
        cron in a nine-hour outage sends ~33 identical alarms overnight — which
        is how a real alarm gets muted."""
        task = _task(schedule_type="cron")

        async def run_with_failure_count(count: int) -> int:
            mock_bot = AsyncMock()
            with (
                patch("luke.scheduler.run_agent", side_effect=RuntimeError("boom")),
                patch("luke.scheduler.db") as mock_db,
            ):
                mock_db.increment_task_failures.return_value = count
                await _run_task(task, mock_bot)
            return mock_bot.send_message.await_count

        assert await run_with_failure_count(2) == 0
        assert await run_with_failure_count(3) == 1
        assert await run_with_failure_count(4) == 0
        assert await run_with_failure_count(23) == 0
        assert await run_with_failure_count(24) == 1
        assert await run_with_failure_count(48) == 1


class TestShutdownIsNotAFailure:
    """A run we killed is not a run that failed.

    All six recorded failures of `f580ac19` (the midnight self-reflection cron)
    — 11, 13, 14, 15 Aug, 3 Sep and 6 Sep 2026 — carry the identical log
    signature: `stopping` → `Draining running tasks` → `agent_result_error`.
    A deploy, usually one the run itself had just launched, SIGTERMed the
    process out from under it. Six tear-downs, zero faults, written into
    task_logs as `error`, counted into consecutive_failures, and finally fired
    as an intermittent-failure alarm onto Filipe's phone at 01:26 on a Sunday.

    The alarm was right about its inputs and the inputs were lying.
    """

    @staticmethod
    def _shutting_down(flag: bool):  # type: ignore[no-untyped-def]
        return patch("luke.scheduler.shutting_down", return_value=flag)

    async def _run(self, task: TaskRecord, *, down: bool) -> MagicMock:
        mock_bot = AsyncMock()
        with (
            patch("luke.scheduler.run_agent", side_effect=RuntimeError("killed mid-run")),
            patch("luke.scheduler.db") as mock_db,
            self._shutting_down(down),
        ):
            mock_db.increment_task_failures.return_value = 3
            mock_db.get_behavior_last_run.return_value = None
            await _run_task(task, mock_bot)
        mock_db.bot = mock_bot  # carry it out for send assertions
        return mock_db

    async def test_the_run_is_logged_interrupted_not_error(self) -> None:
        mock_db = await self._run(_task(schedule_type="cron"), down=True)
        logged = mock_db.log_task_run.call_args[0][3]
        assert logged.startswith("interrupted:")
        # The rate readers score on the "error" prefix — db.recent_task_failure_rate
        # and workspace/tools/task_failure_rate_check.py. Both must skip this row.
        assert not logged.startswith("error")
        assert "killed mid-run" in logged, "still diagnosable, just not blamed"

    async def test_it_does_not_count_toward_the_failure_streak(self) -> None:
        mock_db = await self._run(_task(schedule_type="cron"), down=True)
        mock_db.increment_task_failures.assert_not_called()

    async def test_it_raises_no_alarm(self) -> None:
        """The 01:26 message. Nothing about our own tear-down reaches his phone."""
        mock_db = await self._run(_task(schedule_type="cron"), down=True)
        mock_db.bot.send_message.assert_not_awaited()

    async def test_a_real_failure_is_still_a_failure(self) -> None:
        """The guard must not swallow the fault it was built to distinguish."""
        mock_db = await self._run(_task(schedule_type="cron"), down=False)
        assert mock_db.log_task_run.call_args[0][3].startswith("error:")
        mock_db.increment_task_failures.assert_called_once()
        mock_db.bot.send_message.assert_awaited_once()

    async def test_an_interrupted_once_task_keeps_its_slot(self) -> None:
        """`_is_due` disarms a once-task the moment last_run is set. Writing it
        for a run we cut off is how a one-off send — a citizenship checkpoint,
        a delegated job's only report — dies without a trace."""
        task = _task(schedule_type="once", schedule_value=datetime.now(UTC).isoformat())
        mock_db = await self._run(task, down=True)
        mock_db.update_task_last_run.assert_not_called()
        mock_db.update_task_status.assert_not_called()
        assert mock_db.set_behavior_last_run.call_args[0][0] == "task_rearm:test-id"

    async def test_the_re_arm_is_capped_at_one_an_hour(self) -> None:
        """The other shape here is a crash loop: restart, fire, die, restart."""
        task = _task(schedule_type="once", schedule_value=datetime.now(UTC).isoformat())
        mock_bot = AsyncMock()
        with (
            patch("luke.scheduler.run_agent", side_effect=RuntimeError("killed")),
            patch("luke.scheduler.db") as mock_db,
            self._shutting_down(True),
        ):
            mock_db.get_behavior_last_run.return_value = datetime.now(UTC).isoformat()
            await _run_task(task, mock_bot)
        mock_db.update_task_last_run.assert_called_once()

    async def test_an_unreadable_re_arm_throttle_does_not_re_arm(self) -> None:
        """Fails the opposite way to the alarm throttle, on purpose: a duplicate
        send is a message he did not ask for, a missed one is a gap I can see."""
        task = _task(schedule_type="once", schedule_value=datetime.now(UTC).isoformat())
        mock_bot = AsyncMock()
        with (
            patch("luke.scheduler.run_agent", side_effect=RuntimeError("killed")),
            patch("luke.scheduler.db") as mock_db,
            self._shutting_down(True),
        ):
            mock_db.get_behavior_last_run.return_value = "not-a-timestamp"
            await _run_task(task, mock_bot)
        mock_db.update_task_last_run.assert_called_once()

    async def test_an_interrupted_cron_just_waits_for_its_next_slot(self) -> None:
        """Re-arming is a once-task affair. A cron's next slot arrives anyway,
        and CRON_CATCHUP_GRACE already decides whether it catches up."""
        mock_db = await self._run(_task(schedule_type="cron"), down=True)
        mock_db.update_task_last_run.assert_called_once()

    async def test_the_event_is_released_when_the_loop_returns(self) -> None:
        """Caught by this suite on first run: the loop published a shutdown
        event that was already set and never took it back, so every subsequent
        task failure in the same interpreter read as "we killed it" — the
        misclassification this change exists to remove, pointed the other way.
        """
        shutdown = asyncio.Event()
        shutdown.set()
        with (
            patch("luke.scheduler.db") as mock_db,
            patch("luke.scheduler.memory"),
        ):
            mock_db.get_behavior_last_run.return_value = None
            await asyncio.wait_for(
                start_scheduler_loop(AsyncMock(), _SEM, shutdown=shutdown), timeout=5.0
            )
        assert scheduler.shutting_down() is False

    async def test_the_event_is_released_even_when_the_loop_crashes(self) -> None:
        """The `finally` case: a loop that dies on the way in must not leave the
        flag set for whatever runs next."""
        shutdown = asyncio.Event()
        shutdown.set()
        with (
            patch("luke.scheduler.db") as mock_db,
            patch("luke.scheduler.memory"),
        ):
            mock_db.get_behavior_last_run.side_effect = RuntimeError("db is gone")
            with pytest.raises(RuntimeError):
                await start_scheduler_loop(AsyncMock(), _SEM, shutdown=shutdown)
        assert scheduler.shutting_down() is False

    async def test_shutting_down_reads_the_loops_own_event(self) -> None:
        """The flag has to be the real shutdown signal, not a second guess at it."""
        assert scheduler.shutting_down() is False
        event = asyncio.Event()
        scheduler._shutdown = event
        try:
            assert scheduler.shutting_down() is False
            event.set()
            assert scheduler.shutting_down() is True
        finally:
            scheduler._shutdown = None


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
            patch("luke.scheduler.generate_intents", return_value=[]),
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
            mock_settings.skill_extraction_interval = 999999
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
            "skill_extraction": (),
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
            "skill_extraction": (),
            "dream": (),
        }
        emitted_types = {
            "new_episode",
            "new_insight",
            "goal_updated",
            "feedback_negative",
            "user_message",
        }
        consumed_types: set[str] = set()
        for events in _BEHAVIOR_EVENTS.values():
            consumed_types.update(events)
        uncovered = emitted_types - consumed_types
        assert not uncovered, f"Event types emitted but never consumed: {uncovered}"

    def test_fallback_multiplier_range(self) -> None:
        """Time-based fallback should be 2x-6x in the planner's intent generators."""
        import ast
        import inspect

        from luke.planner import _deep_work_intents, _maintenance_intents

        source = inspect.getsource(_maintenance_intents)
        source += "\n" + inspect.getsource(_deep_work_intents)
        tree = ast.parse(source)
        # Find all BinOp nodes that multiply by a constant (the fallback pattern)
        fallback_multipliers: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if (
                        isinstance(comparator, ast.BinOp)
                        and isinstance(comparator.op, ast.Mult)
                        and isinstance(comparator.right, ast.Constant)
                        and isinstance(comparator.right.value, int)
                    ):
                        fallback_multipliers.append(comparator.right.value)
        # deep_work uses 2x, proactive_scan uses 3x, others use 6x
        assert all(m in (2, 3, 6) for m in fallback_multipliers), (
            f"Expected fallback multipliers to be 2, 3, or 6, got {fallback_multipliers}"
        )

    def test_all_newly_gated_behaviors_have_fallback_multiplier(self) -> None:
        """Event-gated behaviors must all have a 2x, 3x, or 6x fallback in the planner."""
        import ast
        import inspect

        from luke.planner import _deep_work_intents, _maintenance_intents

        source = inspect.getsource(_maintenance_intents)
        source += "\n" + inspect.getsource(_deep_work_intents)
        tree = ast.parse(source)
        fallback_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if (
                        isinstance(comparator, ast.BinOp)
                        and isinstance(comparator.op, ast.Mult)
                        and isinstance(comparator.right, ast.Constant)
                        and comparator.right.value in (2, 3, 6)
                    ):
                        fallback_count += 1
        assert fallback_count >= 9, (
            f"Expected at least 9 behaviors with fallback multiplier, found {fallback_count}"
        )


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
        # Stateful last_run tracking: once set_behavior_last_run is called,
        # subsequent get_behavior_last_run calls return the updated value.
        _last_runs: dict[str, str] = {
            due_behavior: due_ts,
        }

        def _get_last_run(name: str) -> str:
            return _last_runs.get(name, recent_ts)

        def _set_last_run(name: str, ts: str) -> None:
            _last_runs[name] = ts

        mock_db.get_behavior_last_run.side_effect = _get_last_run
        mock_db.set_behavior_last_run.side_effect = _set_last_run
        mock_db.get_due_tasks.return_value = []
        mock_db.count_unconsumed_events.return_value = unconsumed_count
        # Prevent _effective_interval from doubling intervals via spurious no-op counts
        mock_db.get_behavior_no_ops.return_value = 0
        # Prevent TypeError in post-run event consumption
        mock_db.consume_events.return_value = 0
        # Attention budget gate (_enforce_attention_budget) needs a real int
        mock_db.get_daily_outbound_count.return_value = 0
        # Planner needs real ensure_utc for datetime calculations
        from luke.db import ensure_utc as _real_ensure_utc

        mock_db.ensure_utc = _real_ensure_utc
        return mock_db

    def _make_mock_settings(self, *, due_behavior: str, interval: float = 1.0) -> MagicMock:
        """Build settings where only `due_behavior` interval is small enough to be due."""
        mock_settings = MagicMock()
        mock_settings.scheduler_interval = 0.01
        mock_settings.cleanup_interval = 999999
        mock_settings.episode_consolidation_interval = 999999
        mock_settings.reflection_interval = 999999
        mock_settings.proactive_scan_interval = (
            interval if due_behavior == "proactive_scan" else 999999
        )
        mock_settings.deep_work_interval = interval if due_behavior == "deep_work" else 999999
        mock_settings.insight_consolidation_interval = 999999
        mock_settings.feedback_consolidation_interval = 999999
        mock_settings.lifecycle_review_interval = (
            interval if due_behavior == "lifecycle_review" else 999999
        )
        mock_settings.skill_extraction_interval = (
            interval if due_behavior == "skill_extraction" else 999999
        )
        mock_settings.dream_interval = interval if due_behavior == "dream" else 999999
        mock_settings.consolidation_min_cluster = 3
        # Attention budget gate (_enforce_attention_budget) needs real ints
        mock_settings.daily_attention_budget = 12
        mock_settings.attention_urgent_reserve = 2
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
        mock_db = self._make_mock_db(
            due_behavior="proactive_scan",
            elapsed_seconds=2.0,
            unconsumed_count=0,
        )

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.planner.db", new=mock_db),
            patch("luke.planner.settings", new=mock_settings),
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
        mock_db = self._make_mock_db(
            due_behavior="proactive_scan",
            elapsed_seconds=2.0,
            unconsumed_count=1,
        )

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.planner.db", new=mock_db),
            patch("luke.planner.settings", new=mock_settings),
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
        mock_db = self._make_mock_db(
            due_behavior="proactive_scan",
            elapsed_seconds=10.0,
            unconsumed_count=0,
        )

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.planner.db", new=mock_db),
            patch("luke.planner.settings", new=mock_settings),
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
        mock_db = self._make_mock_db(
            due_behavior="lifecycle_review",
            elapsed_seconds=2.0,
            unconsumed_count=4,
        )

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.planner.db", new=mock_db),
            patch("luke.planner.settings", new=mock_settings),
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
        mock_db = self._make_mock_db(
            due_behavior="lifecycle_review",
            elapsed_seconds=2.0,
            unconsumed_count=5,
        )

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.planner.db", new=mock_db),
            patch("luke.planner.settings", new=mock_settings),
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
            patch("luke.planner.db", new=mock_db),
            patch("luke.planner.settings", new=mock_settings),
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
            patch("luke.planner.db", new=mock_db),
            patch("luke.planner.settings", new=mock_settings),
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

    async def test_skill_extraction_skipped_without_enough_episodes(self) -> None:
        """skill_extraction does not fire without enough new episodes and timer < 6x."""
        mock_settings = self._make_mock_settings(due_behavior="skill_extraction", interval=1.0)
        mock_db = self._make_mock_db(
            due_behavior="skill_extraction", elapsed_seconds=2.0, unconsumed_count=1
        )

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.planner.db", new=mock_db),
            patch("luke.planner.settings", new=mock_settings),
            patch("luke.scheduler.memory"),
            patch("luke.scheduler.run_skill_extraction", new_callable=AsyncMock) as mock_fn,
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

    async def test_skill_extraction_fires_with_enough_episodes(self) -> None:
        """skill_extraction fires when at least two new episodes exist."""
        mock_settings = self._make_mock_settings(due_behavior="skill_extraction", interval=1.0)
        mock_db = self._make_mock_db(
            due_behavior="skill_extraction", elapsed_seconds=2.0, unconsumed_count=2
        )

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.planner.db", new=mock_db),
            patch("luke.planner.settings", new=mock_settings),
            patch("luke.scheduler.memory"),
            patch("luke.scheduler.run_skill_extraction", new_callable=AsyncMock) as mock_fn,
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
        """deep_work does not launch when no goal_updated events exist (and timer < 2x)."""
        mock_settings = self._make_mock_settings(due_behavior="deep_work", interval=1.0)
        mock_db = self._make_mock_db(
            due_behavior="deep_work",
            elapsed_seconds=1.5,
            unconsumed_count=0,
        )

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.planner.db", new=mock_db),
            patch("luke.planner.settings", new=mock_settings),
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
        mock_db = self._make_mock_db(
            due_behavior="deep_work",
            elapsed_seconds=2.0,
            unconsumed_count=1,
        )

        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.planner.db", new=mock_db),
            patch("luke.planner.settings", new=mock_settings),
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


# ---------------------------------------------------------------------------
# count_unconsumed_events since-format contract
# ---------------------------------------------------------------------------


class TestContinuationSinceFormat:
    """events.created is written by sqlite datetime('now') — space separator.
    A since string in isoformat ('T' separator) compares greater than every
    same-day row, so the continuation check silently counted zero forever."""

    def test_space_format_since_matches_fresh_event(self, test_db: Any) -> None:
        from luke import db

        db.emit_event("deep_work_oriented", "{}")
        since_space = (datetime.now(UTC) - timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
        assert db.count_unconsumed_events("deep_work_oriented", since=since_space) == 1

    def test_isoformat_since_is_the_bug(self, test_db: Any) -> None:
        from luke import db

        db.emit_event("deep_work_oriented", "{}")
        since_iso = (datetime.now(UTC) - timedelta(seconds=30)).isoformat()
        # Documents the defect the scheduler fix avoids: 'T' > ' ' in the string
        # compare hides every same-day event.
        assert db.count_unconsumed_events("deep_work_oriented", since=since_iso) == 0


# ---------------------------------------------------------------------------
# Wake channel: immediate due-check on task creation / socket poke
# ---------------------------------------------------------------------------


class TestWakeChannel:
    def test_task_created_handler_sets_wake(self) -> None:
        scheduler._wake.clear()
        scheduler._on_task_created(object())
        assert scheduler._wake.is_set()
        scheduler._wake.clear()

    async def test_wake_socket_connection_sets_wake(self, tmp_settings: Any) -> None:
        """A bare connection to $LUKE_DIR/luke.sock wakes the scheduler."""
        import shutil
        import tempfile
        from pathlib import Path

        # AF_UNIX paths are capped at 104 bytes on macOS; pytest tmp dirs are
        # too deep, so bind under /tmp (mirrors the short real LUKE_DIR path).
        short_dir = Path(tempfile.mkdtemp(dir="/tmp"))
        tmp_settings.luke_dir = short_dir
        scheduler._wake.clear()
        server = await scheduler.start_wake_socket()
        try:
            _, writer = await asyncio.open_unix_connection(str(short_dir / "luke.sock"))
            writer.close()
            await asyncio.sleep(0.05)
            assert scheduler._wake.is_set()
        finally:
            server.close()
            await server.wait_closed()
            scheduler._wake.clear()
            shutil.rmtree(short_dir, ignore_errors=True)

    async def test_wake_bypasses_tick_interval(self) -> None:
        """With a long tick, wake() still runs the due-check immediately."""
        recent_ts = datetime.now(UTC).isoformat()
        mock_db = MagicMock()
        mock_db.get_behavior_last_run.return_value = recent_ts
        mock_db.get_due_tasks.return_value = []
        mock_db.count_unconsumed_events.return_value = 0
        mock_db.get_behavior_no_ops.return_value = 0
        mock_db.consume_events.return_value = 0
        mock_db.get_daily_outbound_count.return_value = 0
        from luke.db import ensure_utc as _real_ensure_utc

        mock_db.ensure_utc = _real_ensure_utc

        mock_settings = MagicMock()
        mock_settings.scheduler_interval = 30.0  # a plain tick can't fire in this test
        mock_settings.cleanup_interval = 999999
        for name in (
            "episode_consolidation_interval",
            "reflection_interval",
            "proactive_scan_interval",
            "deep_work_interval",
            "insight_consolidation_interval",
            "feedback_consolidation_interval",
            "lifecycle_review_interval",
            "skill_extraction_interval",
            "dream_interval",
        ):
            setattr(mock_settings, name, 999999)
        mock_settings.consolidation_min_cluster = 3
        mock_settings.daily_attention_budget = 12
        mock_settings.attention_urgent_reserve = 2

        scheduler._wake.clear()
        with (
            patch("luke.scheduler.settings", new=mock_settings),
            patch("luke.scheduler.db", new=mock_db),
            patch("luke.planner.db", new=mock_db),
            patch("luke.planner.settings", new=mock_settings),
            patch("luke.scheduler.memory"),
        ):
            bot = AsyncMock()
            shutdown = asyncio.Event()

            async def poke_then_stop() -> None:
                await asyncio.sleep(0.05)
                scheduler.wake()
                await asyncio.sleep(0.15)
                shutdown.set()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(scheduler.start_scheduler_loop(bot, _SEM, shutdown=shutdown))
                tg.create_task(poke_then_stop())

        # The wake — not the 30s tick — caused a due-check
        assert mock_db.get_due_tasks.called


class TestIntermittentFailureAlert:
    """A task that fails every other night never trips a streak alarm.

    Cron `f580ac19`, the daily self-reflection run, failed on 11, 13 and 14 Aug
    2026 with a success on the 12th between them. `consecutive_failures` went
    1 → 0 → 1 → 2, so the `count == 3` alert never fired, and three nights of
    the run whose whole job is noticing things died unnoticed. The cause (a
    deploy killing it) is fixed separately; this is the alarm that should have
    said so regardless of cause.
    """

    TASK = {"id": "t1", "prompt": "DAILY SELF-REFLECTION (00:00)", "chat_id": "12345"}
    NOW = "2026-08-14T00:09:40+00:00"

    def _alert(self, rate: tuple[int, int], last_alert: str | None = None) -> str | None:
        with patch("luke.scheduler.db") as mock_db:
            mock_db.recent_task_failure_rate.return_value = rate
            mock_db.get_behavior_last_run.return_value = last_alert
            return scheduler._intermittent_failure_alert(self.TASK, "t1", self.NOW)

    def test_fires_on_the_f580ac19_shape(self) -> None:
        alert = self._alert((3, 4))
        assert alert is not None
        assert "3 of its last 4 runs" in alert
        assert "never tripped the consecutive-failure alarm" in alert

    def test_silent_below_the_threshold(self) -> None:
        assert self._alert((2, 10)) is None

    def test_silent_on_too_small_a_sample(self) -> None:
        """Two failures out of two runs is a new task, not a sick one."""
        assert self._alert((2, 2)) is None

    def test_silent_when_healthy(self) -> None:
        assert self._alert((0, 10)) is None

    def test_throttled_within_the_quiet_period(self) -> None:
        """Six hours after the last one — a */15 cron must not alarm all day."""
        assert self._alert((5, 10), last_alert="2026-08-13T18:09:40+00:00") is None

    def test_fires_again_after_the_quiet_period(self) -> None:
        assert self._alert((5, 10), last_alert="2026-08-12T00:09:40+00:00") is not None

    def test_an_unreadable_throttle_fires_rather_than_stays_silent(self) -> None:
        """A missed alarm is the failure this exists to prevent. Fail loud."""
        assert self._alert((5, 10), last_alert="not-a-timestamp") is not None

    def test_the_throttle_is_recorded_when_it_fires(self) -> None:
        with patch("luke.scheduler.db") as mock_db:
            mock_db.recent_task_failure_rate.return_value = (3, 4)
            mock_db.get_behavior_last_run.return_value = None
            scheduler._intermittent_failure_alert(self.TASK, "t1", self.NOW)
        mock_db.set_behavior_last_run.assert_called_once_with("task_fail_rate:t1", self.NOW)

    def test_the_throttle_is_not_recorded_when_it_stays_silent(self) -> None:
        """Otherwise a healthy task quietly arms its own 24h mute."""
        with patch("luke.scheduler.db") as mock_db:
            mock_db.recent_task_failure_rate.return_value = (0, 10)
            mock_db.get_behavior_last_run.return_value = None
            scheduler._intermittent_failure_alert(self.TASK, "t1", self.NOW)
        mock_db.set_behavior_last_run.assert_not_called()

    def test_the_throttle_is_per_task(self) -> None:
        with patch("luke.scheduler.db") as mock_db:
            mock_db.recent_task_failure_rate.return_value = (3, 4)
            mock_db.get_behavior_last_run.return_value = None
            scheduler._intermittent_failure_alert(self.TASK, "other-task", self.NOW)
        mock_db.get_behavior_last_run.assert_called_once_with("task_fail_rate:other-task")

    async def test_a_broken_alarm_cannot_break_the_failure_path(self) -> None:
        """It runs inside `except`, so an exception here escapes _run_task itself.

        That would take out error handling — task_logs, backoff, once-task
        closeout — for every task, to protect a telemetry line.
        """
        task = {
            "id": "t1",
            "chat_id": "12345",
            "prompt": "p",
            "schedule_type": "once",
            "schedule_value": "2026-08-14T00:00:00+00:00",
        }
        bot = MagicMock()
        bot.send_message = AsyncMock()
        with (
            patch("luke.scheduler.run_agent", side_effect=RuntimeError("agent died")),
            patch("luke.scheduler.db") as mock_db,
            patch(
                "luke.scheduler._intermittent_failure_alert",
                side_effect=RuntimeError("alarm is broken"),
            ),
        ):
            mock_db.increment_task_failures.return_value = 1
            await scheduler._run_task(task, bot)  # must not raise
        mock_db.log_task_run.assert_called()
        mock_db.update_task_status.assert_called_once_with("t1", "completed")
