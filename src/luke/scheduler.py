"""Task scheduler: cron, interval, and one-time tasks."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta

import structlog
from aiogram import Bot
from croniter import croniter
from structlog.stdlib import BoundLogger

from . import db, memory
from .bus import bus
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
    run_reflexion,
    run_skill_extraction,
)
from .config import settings
from .db import TaskRecord, ensure_utc
from .planner import BEHAVIOR_EVENTS, generate_intents, plan

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

    # Only cleanup still uses monotonic tracking (planner handles all behaviors)
    last_cleanup = _load_offset("cleanup", settings.cleanup_interval)

    write_heartbeat("startup")

    # --- Reflexion event subscriptions (event-driven, no time-based schedule) ---
    async def _on_reflexion_event(event: object) -> None:
        """Handle events that trigger reflexion analysis."""
        from .bus import Event

        if not isinstance(event, Event):
            return
        # Filter deep_work_skipped to only trigger on relevant reasons
        if event.kind == "deep_work_skipped":
            reason = event.payload.get("reason", "")
            if reason not in ("all_goals_filtered", "quality_blocked"):
                return
        await run_reflexion(
            bot,
            _behavior_sem,
            event_kind=event.kind,
            event_payload=event.payload,
        )

    bus.on("low_quality_work", _on_reflexion_event)
    bus.on("deep_work_skipped", _on_reflexion_event)
    bus.on("continuation_failure", _on_reflexion_event)

    # --- Cron-memory sync: detect cron ID drift when procedures are updated ---
    import re as _re

    _CRON_ID_RE = _re.compile(r"\b[0-9a-f]{8}\b")

    async def _on_procedure_updated(event: object) -> None:
        """Check if updated procedure references cron IDs that don't match live tasks."""
        from .bus import Event

        if not isinstance(event, Event):
            return
        proc_id = event.payload.get("procedure_id", "")
        if not proc_id:
            return
        try:
            body = memory.read_memory_body("procedure", proc_id, 3000)
            if not body:
                return
            referenced_ids = set(_CRON_ID_RE.findall(body))
            if not referenced_ids:
                return
            live_tasks = db.get_due_tasks()
            live_ids = {t["id"][:8] for t in live_tasks}
            stale = referenced_ids - live_ids
            if stale:
                log.warning(
                    "cron_memory_drift",
                    procedure=proc_id,
                    stale_ids=list(stale),
                    live_ids=list(live_ids),
                )
                memory.flag_for_review(
                    proc_id,
                    f"Procedure references cron IDs {stale} that don't match any active task. "
                    f"Live task IDs: {live_ids}. Review and rebuild crons if prompts changed.",
                    confidence=0.8,
                    source="cron_memory_sync",
                )
        except Exception:
            log.warning("cron_memory_sync_failed", procedure=proc_id)

    bus.on("procedure_updated", _on_procedure_updated)

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
            try:
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
            except sqlite3.OperationalError:
                log.warning("hourly_maintenance_skipped", reason="database locked")

        # Step 1: Generate and plan intents (replaces per-behavior "am I due?" blocks)
        # The planner checks all signal sources (goals, events, time) and returns
        # a prioritized list of maintenance intents + an optional deep work intent.
        intents = generate_intents()
        maintenance_intents, deep_work_intent = plan(intents)

        if intents:
            log.debug(
                "planner_intents",
                total=len(intents),
                maintenance=len(maintenance_intents),
                deep_work=deep_work_intent is not None,
                top_intent=intents[0].kind if intents else None,
                top_priority=max(i.priority for i in intents) if intents else 0,
            )

        # Intent-to-behavior mapping
        _INTENT_BEHAVIOR = {
            "consolidation": run_consolidation,
            "reflection": run_reflection,
            "proactive_scan": run_proactive_scan,
            "insight_consolidation": run_insight_consolidation,
            "feedback_consolidation": run_feedback_consolidation,
            "lifecycle_review": run_lifecycle_review,
            "skill_extraction": run_skill_extraction,
            "dream": run_dream,
        }

        # Build maintenance coroutines from planned intents
        maintenance_coros: list[tuple[str, Coroutine[object, object, None]]] = []
        for intent in maintenance_intents:
            behavior_fn = _INTENT_BEHAVIOR.get(intent.kind)
            if behavior_fn:
                maintenance_coros.append((intent.kind, behavior_fn(bot, sem)))

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
                        # Consume events this behavior was triggered by
                        events = BEHAVIOR_EVENTS.get(name, ())
                        if events:
                            consumed = db.consume_events(*events)
                            # Track no-ops for smart backoff
                            if consumed > 0:
                                db.reset_behavior_no_ops(name)
                            else:
                                db.increment_behavior_no_ops(name)

                        # Weekly FTS pruning alongside reflection
                        if name == "reflection":
                            pruned = memory.prune_old_fts_entries(settings.fts_retention_days)
                            if pruned:
                                log.info("fts_pruned", count=pruned)

        # Step 2: Launch deep work as background task (long-lived, NOT awaited)
        deep_work_running = _deep_work_task is not None and not _deep_work_task.done()
        if not deep_work_running and deep_work_intent is not None:
            # Track continuation: deep work launching after maintenance ran
            is_continuation = bool(maintenance_coros)
            if is_continuation:
                bus.emit("post_trigger_continuation")
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

                    # --- Deep work output critic ---
                    # Check if any plan file was actually advanced (steps checked off)
                    try:
                        plans_dir = settings.workspace_dir / "plans"
                        if plans_dir.exists():
                            for plan_file in plans_dir.glob("*.md"):
                                text = plan_file.read_text(encoding="utf-8")[:3000]
                                # Count checked vs unchecked steps
                                checked = text.count("- [x]") + text.count("- [X]")
                                unchecked = text.count("- [ ]")
                                # Check for self-rating
                                import re as _re2

                                rating_m = _re2.search(
                                    r"(?:rating|quality)[:\s]*(\d(?:\.\d)?)", text, _re2.IGNORECASE
                                )
                                if rating_m and checked == 0 and unchecked > 0:
                                    claimed = float(rating_m.group(1))
                                    if claimed >= 3.0:
                                        log.warning(
                                            "deep_work_critic_override",
                                            plan=plan_file.name,
                                            claimed_rating=claimed,
                                            checked_steps=checked,
                                            unchecked_steps=unchecked,
                                        )
                                        goal_id = plan_file.stem
                                        bus.emit("low_quality_work", {
                                            "goal_id": goal_id,
                                            "reason": f"Critic: claimed {claimed}/5 but 0 steps checked off",
                                        })
                    except Exception:
                        log.warning("deep_work_critic_failed")

            _deep_work_task.add_done_callback(_on_deep_work_done)
            log.info(
                "deep_work_launched",
                priority=deep_work_intent.priority,
                source=deep_work_intent.source,
            )

            # Continuation verification: check if deep_work actually oriented
            if is_continuation:

                async def _verify_continuation() -> None:
                    """Wait briefly then check if deep_work_oriented was emitted."""
                    await asyncio.sleep(5)
                    # Check for deep_work_oriented events in the last 30 seconds
                    since = (
                        datetime.now(UTC) - timedelta(seconds=30)
                    ).isoformat()
                    oriented = db.count_unconsumed_events(
                        "deep_work_oriented", since=since
                    )
                    if oriented > 0:
                        bus.emit("continuation_success")
                        log.info("continuation_verified", result="success")
                    else:
                        bus.emit("continuation_failure")
                        log.info("continuation_verified", result="failure")

                asyncio.create_task(_verify_continuation())

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

    # Unsubscribe event handlers on shutdown
    bus.off("low_quality_work", _on_reflexion_event)
    bus.off("deep_work_skipped", _on_reflexion_event)
    bus.off("continuation_failure", _on_reflexion_event)
    bus.off("procedure_updated", _on_procedure_updated)

    # Drain in-flight tasks before exiting (snapshot to avoid mutation during gather)
    pending = list(_running_tasks.values())
    if pending:
        log.info("Draining running tasks", count=len(pending))
        await asyncio.gather(*pending, return_exceptions=True)
