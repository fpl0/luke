"""Task scheduler: cron, interval, and one-time tasks."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Coroutine
from datetime import UTC, datetime

import structlog
from aiogram import Bot
from croniter import croniter
from structlog.stdlib import BoundLogger

from . import db, memory
from .agent import run_agent
from .behaviors import (
    run_consolidation,
    run_deep_work,
    run_dream,
    run_feedback_consolidation,
    run_insight_consolidation,
    run_lifecycle_review,
    run_proactive_scan,
    run_reflection,
    run_skill_extraction,
)
from .config import settings
from .db import TaskRecord, ensure_utc

log: BoundLogger = structlog.get_logger()


def write_heartbeat(status: str = "idle") -> None:
    """Write a heartbeat file so the external watchdog knows we're alive.

    File format: ``<unix_timestamp> <pid> <status>``
    Written atomically (write-to-tmp + rename) to avoid partial reads.
    """
    heartbeat_path = settings.store_dir / "heartbeat"
    tmp_path = settings.store_dir / "heartbeat.tmp"
    try:
        content = f"{int(time.time())} {os.getpid()} {status}\n"
        tmp_path.write_text(content)
        tmp_path.rename(heartbeat_path)
    except OSError:
        pass  # best effort — don't crash the scheduler over a heartbeat

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
                autonomous=True,
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
    last_consolidation = _load_offset("consolidation", settings.episode_consolidation_interval)
    last_reflection = _load_offset("reflection", settings.reflection_interval)
    last_proactive = _load_offset("proactive_scan", settings.proactive_scan_interval)
    last_deep_work = _load_offset("deep_work", settings.deep_work_interval)
    last_insight_consolidation = _load_offset(
        "insight_consolidation", settings.insight_consolidation_interval
    )
    last_feedback_consolidation = _load_offset(
        "feedback_consolidation", settings.feedback_consolidation_interval
    )
    last_lifecycle_review = _load_offset("lifecycle_review", settings.lifecycle_review_interval)
    last_skill_extraction = _load_offset("skill_extraction", settings.skill_extraction_interval)
    last_dream = _load_offset("dream", settings.dream_interval)

    write_heartbeat("startup")

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
        write_heartbeat("tick")

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
            pruned_events = db.cleanup_events()
            expired_working = memory.expire_working_memories()
            db.set_behavior_last_run("cleanup", datetime.now(UTC).isoformat())
            log.info(
                "hourly_maintenance",
                decayed=updated,
                sessions_cleaned=len(cleaned_ids),
                task_logs_pruned=pruned_logs,
                outbound_pruned=pruned_outbound,
                events_pruned=pruned_events,
                working_expired=expired_working,
            )

        # Step 1: Collect and run due maintenance behaviors (short-lived, awaited)
        # Each behavior uses a hybrid trigger: time-based interval AND event awareness.
        # Smart backoff: if a behavior produced nothing useful, its effective interval
        # doubles (up to 16x) until a relevant event resets the counter.
        maintenance_coros: list[tuple[str, Coroutine[object, object, None]]] = []

        def _effective_interval(name: str, base: float) -> float:
            """Apply exponential backoff based on consecutive no-op runs."""
            try:
                no_ops = int(db.get_behavior_no_ops(name))
            except (TypeError, ValueError):
                no_ops = 0
            return float(base * (2 ** min(no_ops, 4)))  # caps at 16x base interval

        # Consolidation: time-based + needs episode events
        consol_interval = _effective_interval(
            "consolidation", settings.episode_consolidation_interval
        )
        if now_mono - last_consolidation >= consol_interval:
            has_episodes = (
                db.count_unconsumed_events("new_episode") >= settings.consolidation_min_cluster
            )
            if has_episodes or now_mono - last_consolidation >= consol_interval * 6:
                last_consolidation = now_mono
                maintenance_coros.append(("consolidation", run_consolidation(bot, sem)))

        # Proactive scan: time-based + needs goal or user activity
        proactive_interval = _effective_interval("proactive_scan", settings.proactive_scan_interval)
        if now_mono - last_proactive >= proactive_interval:
            has_activity = db.count_unconsumed_events("goal_updated", "user_message") >= 1
            if has_activity or now_mono - last_proactive >= proactive_interval * 6:
                last_proactive = now_mono
                maintenance_coros.append(("proactive_scan", run_proactive_scan(bot, sem)))

        # Reflection: time-based + needs user interaction or feedback
        reflection_interval = _effective_interval("reflection", settings.reflection_interval)
        if now_mono - last_reflection >= reflection_interval:
            has_feedback = db.count_unconsumed_events("feedback_negative", "user_message") >= 5
            if has_feedback or now_mono - last_reflection >= reflection_interval * 6:
                last_reflection = now_mono
                maintenance_coros.append(("reflection", run_reflection(bot, sem)))
                # Weekly FTS pruning (alongside reflection — not urgent)
                pruned = memory.prune_old_fts_entries(settings.fts_retention_days)
                if pruned:
                    log.info("fts_pruned", count=pruned)

        # Insight consolidation: time-based + needs insight events
        insight_interval = _effective_interval(
            "insight_consolidation", settings.insight_consolidation_interval
        )
        if now_mono - last_insight_consolidation >= insight_interval:
            has_insights = db.count_unconsumed_events("new_insight") >= 3
            if has_insights or now_mono - last_insight_consolidation >= insight_interval * 6:
                last_insight_consolidation = now_mono
                maintenance_coros.append(
                    ("insight_consolidation", run_insight_consolidation(bot, sem))
                )

        # Feedback consolidation: time-based + needs negative feedback
        feedback_interval = _effective_interval(
            "feedback_consolidation", settings.feedback_consolidation_interval
        )
        if now_mono - last_feedback_consolidation >= feedback_interval:
            has_negatives = db.count_unconsumed_events("feedback_negative") >= 2
            if has_negatives or now_mono - last_feedback_consolidation >= feedback_interval * 6:
                last_feedback_consolidation = now_mono
                maintenance_coros.append(
                    ("feedback_consolidation", run_feedback_consolidation(bot, sem))
                )

        # Lifecycle review: time-based + needs accumulated memory activity
        if now_mono - last_lifecycle_review >= settings.lifecycle_review_interval:
            has_activity = db.count_unconsumed_events("new_episode", "new_insight") >= 5
            if (
                has_activity
                or now_mono - last_lifecycle_review >= settings.lifecycle_review_interval * 6
            ):
                last_lifecycle_review = now_mono
                maintenance_coros.append(("lifecycle_review", run_lifecycle_review(bot, sem)))

        # Skill extraction: time-based + needs multiple new episodes to infer a reusable pattern.
        skill_extraction_interval = _effective_interval(
            "skill_extraction", settings.skill_extraction_interval
        )
        if now_mono - last_skill_extraction >= skill_extraction_interval:
            has_recent_episodes = db.count_unconsumed_events("new_episode") >= 2
            if (
                has_recent_episodes
                or now_mono - last_skill_extraction >= skill_extraction_interval * 6
            ):
                last_skill_extraction = now_mono
                maintenance_coros.append(("skill_extraction", run_skill_extraction(bot, sem)))

        # Dream: autonomous thinking periods (quiet-gated internally + needs mental material)
        if now_mono - last_dream >= settings.dream_interval:
            has_material = db.count_unconsumed_events("new_insight", "new_episode") >= 1
            if has_material or now_mono - last_dream >= settings.dream_interval * 6:
                last_dream = now_mono
                maintenance_coros.append(("dream", run_dream(bot, sem)))

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
            # Map behavior names to the events they consume
            _BEHAVIOR_EVENTS: dict[str, tuple[str, ...]] = {
                "consolidation": ("new_episode",),
                "reflection": ("feedback_negative", "user_message"),
                "proactive_scan": ("goal_updated",),
                "insight_consolidation": ("new_insight",),
                "feedback_consolidation": ("feedback_negative",),
                "lifecycle_review": (),
                # Shares new_episode as an input signal with consolidation, so it
                # must not consume those events and starve the other behavior.
                "skill_extraction": (),
                "dream": (),
            }
            with db.batch():
                for (name, _), result in zip(maintenance_coros, results, strict=True):
                    if isinstance(result, BaseException):
                        log.exception(f"{name}_error", exc_info=result)
                    else:
                        db.set_behavior_last_run(name, now_iso)
                        # Consume events this behavior was triggered by
                        events = _BEHAVIOR_EVENTS.get(name, ())
                        if events:
                            consumed = db.consume_events(*events)
                            # Track no-ops for smart backoff
                            if consumed > 0:
                                db.reset_behavior_no_ops(name)
                            else:
                                db.increment_behavior_no_ops(name)
                        else:
                            # Behaviors without event gates don't backoff
                            pass

        # Step 2: Launch deep work as background task (long-lived, NOT awaited)
        deep_work_running = _deep_work_task is not None and not _deep_work_task.done()
        if not deep_work_running and now_mono - last_deep_work >= settings.deep_work_interval:
            has_goal_activity = db.count_unconsumed_events("goal_updated") >= 1
            if has_goal_activity or now_mono - last_deep_work >= settings.deep_work_interval * 6:
                last_deep_work = now_mono
                db.set_behavior_last_run("deep_work", datetime.now(UTC).isoformat())
                _deep_work_task = asyncio.create_task(
                    _limit_behavior(_behavior_sem, run_deep_work(bot, sem))
                )

                def _on_deep_work_done(fut: asyncio.Task[None]) -> None:
                    exc = fut.exception() if not fut.cancelled() else None
                    if exc:
                        log.exception("deep_work_task_error", exc_info=exc)
                    else:
                        # Consume goal events that deep work acts on
                        consumed = db.consume_events("goal_updated")
                        if consumed > 0:
                            db.reset_behavior_no_ops("deep_work")
                        log.info("deep_work_events_consumed", consumed=consumed)

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
