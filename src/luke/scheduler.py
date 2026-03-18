"""Task scheduler: cron, interval, and one-time tasks."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from datetime import UTC, datetime

import structlog
from aiogram import Bot
from croniter import croniter
from structlog.stdlib import BoundLogger

from . import db, memory
from .agent import run_agent
from .behaviors import run_consolidation, run_deep_work, run_proactive_scan, run_reflection
from .config import settings
from .db import TaskRecord, ensure_utc

log: BoundLogger = structlog.get_logger()

# Track long-running deep work task across scheduler ticks
_deep_work_task: asyncio.Task[None] | None = None

# Cap concurrent behaviors to avoid starving message processing.
# With max_concurrent=8 for the main semaphore, limiting behaviors to 3
# guarantees at least 5 slots remain available for user messages.
_behavior_sem = asyncio.Semaphore(3)


async def _limit_behavior(cap: asyncio.Semaphore, coro: Coroutine[object, object, None]) -> None:
    """Run a behavior coroutine under a concurrency limit."""
    async with cap:
        await coro


def _is_due(task: TaskRecord, now: datetime) -> bool:
    """Check if a task should run now."""
    stype = task["schedule_type"]
    sval = task["schedule_value"]
    last_run = task["last_run"]

    if stype == "once":
        if last_run:
            return False
        return now >= ensure_utc(datetime.fromisoformat(sval))

    if stype == "cron":
        if not last_run:
            # Use task creation time as anchor so we wait for the next window
            last = ensure_utc(datetime.fromisoformat(task["created_at"]))
        else:
            last = ensure_utc(datetime.fromisoformat(last_run))
        next_run: datetime = ensure_utc(croniter(sval, last).get_next(datetime))
        return now >= next_run

    if stype == "interval":
        interval_ms = int(sval)
        if not last_run:
            # Use task creation time as anchor so we wait for the first interval
            last = ensure_utc(datetime.fromisoformat(task["created_at"]))
            elapsed_ms = (now - last).total_seconds() * 1000
            return elapsed_ms >= interval_ms
        last = ensure_utc(datetime.fromisoformat(last_run))
        elapsed_ms = (now - last).total_seconds() * 1000
        return elapsed_ms >= interval_ms

    return False


async def _run_task(task: TaskRecord, bot: Bot) -> None:
    """Execute a single scheduled task."""
    task_id = task["id"]
    started = datetime.now(UTC).isoformat()
    log.info(
        "task_start",
        task_id=task_id,
        chat_id=task["chat_id"],
        type=task["schedule_type"],
    )

    try:
        # Scheduled tasks use the same pattern as behaviors: text output is
        # never auto-forwarded to Telegram.  The agent must call send_message
        # (or another send tool) explicitly to reach the user.  A short
        # preamble tells the agent this fact so it doesn't rely on text output.
        prompt = (
            "[Scheduled task — text output is not delivered to the user. "
            "Use send_message/reply to communicate.]\n\n" + task["prompt"]
        )
        result = await asyncio.wait_for(
            run_agent(
                chat_id=task["chat_id"],
                prompt=prompt,
                session_id=None,
                bot=bot,
            ),
            timeout=settings.agent_timeout,
        )

        finished = datetime.now(UTC).isoformat()
        db.log_task_run(task_id, started, finished, "ok")
        db.reset_task_failures(task_id)
        log.info(
            "task_done",
            task_id=task_id,
            sent=result.sent_messages,
            dropped_texts=len(result.texts),
        )

        # One-time tasks complete after running
        if task["schedule_type"] == "once":
            db.update_task_status(task_id, "completed")

    except Exception:
        finished = datetime.now(UTC).isoformat()
        db.log_task_run(task_id, started, finished, "error")
        log.exception("Task failed", task_id=task_id)
        count = db.increment_task_failures(task_id)
        if count >= 3:
            try:
                await bot.send_message(
                    chat_id=int(settings.chat_id),
                    text=f"⚠️ Task '{task['prompt'][:50]}' has failed {count} times in a row.",
                )
            except Exception:
                log.exception("Failed to send task failure alert")
        # Mark failed once-tasks as completed to prevent retry storms
        if task["schedule_type"] == "once":
            db.update_task_status(task_id, "completed")
    finally:
        # Always update last_run to prevent immediate re-firing on next tick
        db.update_task_last_run(task_id, started)


_running_tasks: dict[str, asyncio.Task[None]] = {}


async def start_scheduler_loop(
    bot: Bot, sem: asyncio.Semaphore, *, shutdown: asyncio.Event | None = None
) -> None:
    """Main scheduler loop — checks for due tasks every interval.

    If *shutdown* is provided, the loop exits when the event is set.
    """
    global _deep_work_task
    log.info("Scheduler started", interval=settings.scheduler_interval)
    now_mono = time.monotonic()
    now_wall = datetime.now(UTC)

    def _load_offset(name: str, interval: float) -> float:
        """Load last run from DB; return monotonic timestamp for scheduler comparison."""
        iso = db.get_behavior_last_run(name)
        if iso is None:
            return now_mono - interval  # fire on first tick
        last_wall = ensure_utc(datetime.fromisoformat(iso))
        elapsed = (now_wall - last_wall).total_seconds()
        return now_mono - elapsed

    last_cleanup = _load_offset("cleanup", settings.cleanup_interval)
    last_consolidation = _load_offset("consolidation", settings.consolidation_interval)
    last_reflection = _load_offset("reflection", settings.reflection_interval)
    last_proactive = _load_offset("proactive_scan", settings.proactive_scan_interval)
    last_deep_work = _load_offset("deep_work", settings.deep_work_interval)

    while not (shutdown and shutdown.is_set()):
        # Use wait with timeout so we wake up promptly on shutdown
        if shutdown:
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=settings.scheduler_interval)
                break  # Shutdown signalled
            except TimeoutError:
                pass  # Normal tick
        else:
            await asyncio.sleep(settings.scheduler_interval)
        now_mono = time.monotonic()

        # Hourly: FTS cleanup + adaptive importance decay + session cleanup
        if now_mono - last_cleanup >= settings.cleanup_interval:
            last_cleanup = now_mono
            memory.cleanup_archived_fts()
            updated = memory.decay_importance(settings.decay_rates)
            cleaned_ids = db.cleanup_stale_sessions(settings.session_timeout)
            # Clear model ratchet only for the specific expired sessions
            if cleaned_ids:
                from .app import _session_models

                for cid in cleaned_ids:
                    _session_models.pop(cid, None)
            pruned_logs = db.cleanup_task_logs()
            pruned_outbound = db.cleanup_outbound_log()
            db.set_behavior_last_run("cleanup", datetime.now(UTC).isoformat())
            log.info(
                "hourly_maintenance",
                decayed=updated,
                sessions_cleaned=len(cleaned_ids),
                task_logs_pruned=pruned_logs,
                outbound_pruned=pruned_outbound,
            )

        # Step 1: Collect and run due maintenance behaviors (short-lived, awaited)
        maintenance_coros: list[tuple[str, Coroutine[object, object, None]]] = []

        if now_mono - last_consolidation >= settings.consolidation_interval:
            last_consolidation = now_mono
            maintenance_coros.append(("consolidation", run_consolidation(bot, sem)))

        if now_mono - last_proactive >= settings.proactive_scan_interval:
            last_proactive = now_mono
            maintenance_coros.append(("proactive_scan", run_proactive_scan(bot, sem)))

        if now_mono - last_reflection >= settings.reflection_interval:
            last_reflection = now_mono
            maintenance_coros.append(("reflection", run_reflection(bot, sem)))
            # Weekly FTS pruning (alongside reflection — not urgent)
            pruned = memory.prune_old_fts_entries(settings.fts_retention_days)
            if pruned:
                log.info("fts_pruned", count=pruned)

        if maintenance_coros:
            names = [name for name, _ in maintenance_coros]
            if len(names) > 1:
                log.warning("multiple_behaviors_due", count=len(names), behaviors=names)
            log.info("behaviors_start", behaviors=names)
            coros = [coro for _, coro in maintenance_coros]
            results: list[None | BaseException] = await asyncio.gather(
                *[_limit_behavior(_behavior_sem, c) for c in coros],
                return_exceptions=True,
            )
            now_iso = datetime.now(UTC).isoformat()
            with db.batch():
                for (name, _), result in zip(maintenance_coros, results, strict=True):
                    if isinstance(result, BaseException):
                        log.exception(f"{name}_error", exc_info=result)
                    else:
                        db.set_behavior_last_run(name, now_iso)

        # Step 2: Launch deep work as background task (long-lived, NOT awaited)
        deep_work_running = _deep_work_task is not None and not _deep_work_task.done()
        if not deep_work_running and now_mono - last_deep_work >= settings.deep_work_interval:
            last_deep_work = now_mono
            db.set_behavior_last_run("deep_work", datetime.now(UTC).isoformat())
            _deep_work_task = asyncio.create_task(
                _limit_behavior(_behavior_sem, run_deep_work(bot, sem))
            )

            def _on_deep_work_done(fut: asyncio.Task[None]) -> None:
                exc = fut.exception() if not fut.cancelled() else None
                if exc:
                    log.exception("deep_work_task_error", exc_info=exc)

            _deep_work_task.add_done_callback(_on_deep_work_done)
            log.info("deep_work_launched")

        try:
            now = datetime.now(UTC)
            tasks = db.get_due_tasks()
            launched = 0
            for task in tasks:
                task_id = task["id"]
                if task_id in _running_tasks and not _running_tasks[task_id].done():
                    continue
                if _is_due(task, now):
                    # Set last_run immediately to prevent re-firing on next tick
                    db.update_task_last_run(task_id, now.isoformat())
                    t = asyncio.create_task(_run_task(task, bot))
                    _running_tasks[task_id] = t
                    launched += 1

                    def _cleanup(fut: asyncio.Task[None], *, tid: str = task_id) -> None:
                        _running_tasks.pop(tid, None)

                    t.add_done_callback(_cleanup)
            if launched:
                log.debug(
                    "scheduler_tick",
                    total=len(tasks),
                    launched=launched,
                    running=len(_running_tasks),
                )
        except Exception:
            log.exception("Scheduler loop error")

    # Drain in-flight tasks before exiting (snapshot to avoid mutation during gather)
    pending = list(_running_tasks.values())
    if pending:
        log.info("Draining running tasks", count=len(pending))
        await asyncio.gather(*pending, return_exceptions=True)
