"""Task scheduler: cron, interval, and one-time tasks."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from aiogram import Bot
from aiogram.types import ReplyParameters
from croniter import croniter
from structlog.stdlib import BoundLogger

from . import db, memory
from .agent import AgentResult, parse_delegation, run_agent, send_long_message
from .behaviors import (
    enforce_plan_momentum,
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
from .bus import bus
from .config import settings
from .db import TaskRecord, ensure_utc
from .planner import BEHAVIOR_EVENTS, generate_intents, plan

log: BoundLogger = structlog.get_logger()

# How stale a missed cron slot may be and still be caught up on restart.
#
# Without this, `_is_due` computes the next slot after last_run and fires the
# moment now passes it — so after a long outage every cron fires at once, at
# whatever hour the process happened to come back. Observed 2026-09-01, twelve
# minutes after the 16-day blackout ended: all sixteen crons started in the same
# second, which would have delivered the 06:00 morning briefing at 18:51 and the
# Friday note on a Tuesday.
#
# The grace is measured against the SLOT, not the outage: an hourly cron that
# missed 12:00 during a 90-minute outage still runs at 12:30, because that slot
# is only 30 minutes stale. A slot older than this has had its moment pass —
# waiting for the next real one is the only sane delivery.
CRON_CATCHUP_GRACE = timedelta(hours=1)

# Wake signal: set by task creation (bus: cron_created) or an external poke at
# the wake socket. The loop waits on it alongside the tick timeout, so newly
# queued work starts in ~0s instead of up to a full scheduler_interval later.
_wake = asyncio.Event()

# The shutdown event the loop was started with, published module-wide so the
# failure path can tell "this task died" from "we killed this task".
#
# Every one of f580ac19's six recorded failures (11/13/14/15 Aug, 3 Sep, 6 Sep
# 2026) has the same log signature: `stopping` → `Draining running tasks` →
# `agent_result_error`. The run never failed. A deploy — usually one the run
# itself had just launched — SIGTERMed the process out from under it, and the
# tear-down was written into task_logs as `error`, counted into
# consecutive_failures, and eventually fired an intermittent-failure alarm at
# 01:26 in the morning about a task that had never once failed on its own.
#
# A signal you generate yourself is not evidence about the thing you pointed it
# at. Anything killed inside the drain is recorded as `interrupted`, which the
# rate readers (db.recent_task_failure_rate, task_failure_rate_check.py) do not
# count, because both score on the `error` prefix.
_shutdown: asyncio.Event | None = None


def shutting_down() -> bool:
    """True while the process is tearing down — set by the shutdown signal."""
    return _shutdown is not None and _shutdown.is_set()


def _release_shutdown() -> None:
    """Forget the loop's shutdown event once the drain is over."""
    global _shutdown
    _shutdown = None


class AgentRunFailed(RuntimeError):
    """The agent run returned without raising, but did not actually happen.

    Raised so a dead run flows through the same failure path as a crash —
    logged, counted, backed off, and reported — instead of being recorded
    as a successful silent run.
    """


def wake() -> None:
    """Wake the scheduler loop now — the next due-check runs immediately."""
    _wake.set()


def _on_task_created(event: object) -> None:
    """Bus handler: a task was just created in-process — check due-ness now."""
    wake()


async def start_wake_socket() -> asyncio.Server:
    """Unix socket at $LUKE_DIR/luke.sock: any connection wakes the scheduler.

    This is the operator channel's latency fix. External processes (Claude
    Code operator sessions, scripts) queue work by inserting into the tasks
    table; a poke here makes the scheduler pick it up immediately instead of
    on the next tick. Same-user filesystem permissions are the auth boundary.
    """
    path = settings.luke_dir / "luke.sock"
    path.unlink(missing_ok=True)

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        wake()
        writer.close()

    return await asyncio.start_unix_server(_handle, path=str(path))


async def _sleep_until_tick(shutdown: asyncio.Event | None) -> None:
    """Wait one scheduler_interval — or less if woken or shut down."""
    waiters = [asyncio.ensure_future(_wake.wait())]
    if shutdown:
        waiters.append(asyncio.ensure_future(shutdown.wait()))
    try:
        await asyncio.wait(
            waiters,
            timeout=settings.scheduler_interval,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for w in waiters:
            w.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
    _wake.clear()


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


# Intermittent-failure alarm. `consecutive_failures` only ever sees a streak, so
# a task that fails every other night is invisible to it forever — see
# db.recent_task_failure_rate and the 2026-08-14 03:00 addendum in
# workspace/plans/perf-audit-2026-08-01.md.
_FAILURE_RATE_WINDOW = 10  # how many recent runs to look at
_FAILURE_RATE_MIN_RUNS = 4  # below this the sample says nothing
_FAILURE_RATE_THRESHOLD = 3  # failures in the window before it is worth saying
_FAILURE_RATE_QUIET_H = 24  # at most one of these a day, per task


def _rate_alert_is_due(task_id: str, now_iso: str) -> bool:
    """True when this task's rate alarm has not fired in _FAILURE_RATE_QUIET_H.

    Persisted in behavior_state rather than in memory: a restart must not reset
    the throttle, or a permanently-broken */15 cron alarms afresh every deploy.
    A throttle that cannot be read is treated as "fire" — a missed alarm is the
    failure mode this whole mechanism exists to prevent.
    """
    last = db.get_behavior_last_run(f"task_fail_rate:{task_id}")
    if not last:
        return True
    try:
        elapsed = (
            ensure_utc(datetime.fromisoformat(now_iso)) - ensure_utc(datetime.fromisoformat(last))
        ).total_seconds()
    except (TypeError, ValueError):
        return True
    return elapsed >= _FAILURE_RATE_QUIET_H * 3600


_REARM_QUIET_H = 1  # at most one re-arm an hour per once-task


def _rearm_is_due(task_id: str, now_iso: str) -> bool:
    """True when this once-task may be re-armed after a tear-down killed it.

    The mirror image of `_rate_alert_is_due`, and it fails the other way: an
    unreadable throttle blocks the re-arm. Firing a scheduled send twice is a
    message to Filipe he did not ask for; not firing it is a gap I can see in
    task_logs. Prefer the visible failure.
    """
    last = db.get_behavior_last_run(f"task_rearm:{task_id}")
    if not last:
        return True
    try:
        elapsed = (
            ensure_utc(datetime.fromisoformat(now_iso)) - ensure_utc(datetime.fromisoformat(last))
        ).total_seconds()
    except (TypeError, ValueError):
        return False
    return elapsed >= _REARM_QUIET_H * 3600


def _intermittent_failure_alert(task: TaskRecord, task_id: str, finished: str) -> str | None:
    """The alert text for a task failing intermittently, or None.

    Split out from the failure handler so it can be tested without driving a
    whole task run, and so the handler can treat it as best-effort.
    """
    fails, runs = db.recent_task_failure_rate(task_id, window=_FAILURE_RATE_WINDOW)
    if runs < _FAILURE_RATE_MIN_RUNS or fails < _FAILURE_RATE_THRESHOLD:
        return None
    # The streak counter's own throttle is useless here — the streak keeps
    # resetting, which is the whole point — so throttle on wall-clock, persisted,
    # at most one a day per task. Unthrottled on a */15 cron that is dozens.
    if not _rate_alert_is_due(task_id, finished):
        return None
    db.set_behavior_last_run(f"task_fail_rate:{task_id}", finished)
    return (
        f"⚠️ Task '{str(task['prompt'])[:50]}' has failed {fails} of its last {runs} runs "
        "— intermittently, so it never tripped the consecutive-failure alarm."
    )


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
            # Use task creation time as anchor so we wait for the next window.
            # No catch-up grace here: a task that has never run isn't replaying
            # a missed slot, it's waiting for its first one.
            anchor = ensure_utc(datetime.fromisoformat(task["created_at"]))
        else:
            # Flooring the anchor at now - CRON_CATCHUP_GRACE makes croniter
            # return the next *upcoming* slot rather than one that passed days
            # ago, so a long outage resumes the schedule instead of replaying it.
            last = ensure_utc(datetime.fromisoformat(last_run))
            anchor = max(last, now - CRON_CATCHUP_GRACE)
        next_run: datetime = ensure_utc(croniter(sval, anchor).get_next(datetime))
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

    # Set to the timestamp of a tear-down that killed this run mid-flight, so
    # the `finally` clause below can tell a run that finished from a run we cut
    # off — a once-task cut off has never had its moment and must keep it.
    interrupted_at: str | None = None

    raw_prompt = task["prompt"]
    if isinstance(raw_prompt, bytes):
        raw_prompt = raw_prompt.decode("utf-8", errors="replace")
    # Delegated jobs (created by the delegate tool) invert the delivery rule:
    # their final text output IS the report, relayed to Filipe by code below —
    # the loop closes deterministically, never by trusting the model to send.
    delegation = parse_delegation(raw_prompt)

    try:
        if delegation is not None:
            prompt = (
                "[Delegated background job — do the work now. Your final text "
                "output is relayed to Filipe automatically as the job's report; "
                "make it concise and outcome-focused.]\n\n" + delegation[0]
            )
        else:
            # Scheduled tasks use the same pattern as behaviors: text output is
            # never auto-forwarded to Telegram.  The agent must call send_message
            # (or another send tool) explicitly to reach the user.  A short
            # preamble tells the agent this fact so it doesn't rely on text output.
            prompt = (
                "[Scheduled task — text output is not delivered to the user. "
                "Use send_message/reply to communicate.]\n\n" + raw_prompt
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

        # The run can come back dead without raising — auth failure, API error,
        # or zero usage with no tools. Route it into the failure path below
        # rather than recording "ok": task_logs is the surface every later
        # audit reads, and a lying "ok" is worse than a missing row.
        # (2026-08-09/10: 36 dead runs across ~9h all logged "ok", including
        # the weekly reqs watch on its first ever fire and that Monday's
        # morning briefing. Nothing anywhere said a word.)
        if result.is_error:
            raise AgentRunFailed(
                f"agent returned is_error[{result.error_subtype}]: {result.error_detail}"
            )

        finished = datetime.now(UTC).isoformat()
        db.log_task_run(task_id, started, finished, "ok")
        db.reset_task_failures(task_id)
        # Scheduled tasks are the largest spend category — without this line they
        # were invisible to cost_log (~70% of real spend untracked, found 2026-08-01).
        db.log_cost(
            task["chat_id"],
            result.cost_usd,
            result.num_turns,
            result.duration_api_ms,
            f"task:{task_id}",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_create_tokens=result.cache_create_tokens,
            cache_read_tokens=result.cache_read_tokens,
        )
        log.info(
            "task_done",
            task_id=task_id,
            sent=result.sent_messages,
            dropped_texts=len(result.texts),
            delegated=delegation is not None,
        )

        if delegation is not None:
            await _deliver_delegation_report(task, result, delegation[1], bot)

        # One-time tasks complete after running
        if task["schedule_type"] == "once":
            db.update_task_status(task_id, "completed")

    except Exception as exc:
        finished = datetime.now(UTC).isoformat()
        # Capture the exception into task_logs so failures are diagnosable
        # from luke.db alone — otherwise the cause lives only in structlog
        # and self-reflection sees a bare "error" black box.
        detail = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]
        if shutting_down():
            # We killed it. See the note on `_shutdown` above: this is our own
            # tear-down coming back as an exception, not a fault in the task.
            interrupted_at = finished
            db.log_task_run(task_id, started, finished, f"interrupted: {detail}")
            log.warning(
                "task_interrupted_by_shutdown", task_id=task_id, detail=detail
            )
            return
        db.log_task_run(task_id, started, finished, f"error: {detail}")
        log.exception("Task failed", task_id=task_id)
        count = db.increment_task_failures(task_id)
        # Alert on the third strike, then only once per 24 further failures.
        # Unthrottled this fires every run: a 15-minute cron in a nine-hour
        # outage would have sent ~33 identical alarms overnight, which is how
        # a real alarm gets muted. Once loudly, then a heartbeat.
        alert: str | None = None
        if count == 3 or (count > 3 and count % 24 == 0):
            alert = f"⚠️ Task '{task['prompt'][:50]}' has failed {count} times in a row."
        else:
            # A streak counter is blind to an INTERMITTENT failure — one success
            # resets it. f580ac19, the daily self-reflection cron, failed 11, 13
            # and 14 Aug 2026 with a success on the 12th between them: the count
            # went 1 → 0 → 1 → 2 and nothing ever said a word. Three nights of
            # the run whose whole job is noticing things, dying unnoticed.
            # Never let the alarm break the failure path it lives in. We are
            # already inside `except`; an exception raised here escapes _run_task
            # entirely, so a broken alarm would take out the error handling for
            # every task. Best-effort, exactly like inflight.py.
            try:
                alert = _intermittent_failure_alert(task, task_id, finished)
            except Exception:
                log.exception("intermittent_failure_alert_failed", task_id=task_id)
        if alert:
            try:
                await bot.send_message(chat_id=int(settings.chat_id), text=alert)
            except Exception:
                log.exception("Failed to send task failure alert")
        # A delegated job is a promise to Filipe — its death must speak.
        # Once-tasks don't retry, so this is the only notice he'll ever get.
        if delegation is not None:
            try:
                await send_long_message(
                    bot,
                    chat_id=int(task["chat_id"]),
                    text=(
                        f"⚠️ Background job {task_id} died: {detail}\n"
                        "It won't retry on its own — re-delegate if still needed."
                    ),
                )
            except Exception:
                log.exception("delegation_death_report_failed", task_id=task_id)
        # Mark failed once-tasks as completed to prevent retry storms
        if task["schedule_type"] == "once":
            db.update_task_status(task_id, "completed")
    finally:
        # `_is_due` disarms a once-task the moment last_run is set, so writing
        # it here for a run WE cut off is how a one-off send dies silently: the
        # citizenship checkpoint, a fasting-prep note, a delegated job's only
        # report. Leaving last_run unset re-arms it for the next tick after the
        # restart, which is the behaviour a deploy should have had all along.
        #
        # Capped to one re-arm an hour per task, because the other shape here is
        # a crash loop: restart, fire, die, restart. An hour is far longer than
        # a guardian restart cycle and far shorter than any once-task's useful
        # life. Cron and interval tasks keep the normal write — their next slot
        # comes around on its own, and CRON_CATCHUP_GRACE already governs it.
        if (
            interrupted_at is not None
            and task["schedule_type"] == "once"
            and _rearm_is_due(task_id, interrupted_at)
        ):
            db.set_behavior_last_run(f"task_rearm:{task_id}", interrupted_at)
            log.warning("once_task_rearmed_after_interruption", task_id=task_id)
        else:
            # Always update last_run to prevent immediate re-firing on next tick
            db.update_task_last_run(task_id, started)


async def _deliver_delegation_report(
    task: TaskRecord,
    result: AgentResult,
    trigger_msg_id: int | None,
    bot: Bot,
) -> None:
    """Relay a delegated job's outcome to Filipe. The loop ALWAYS closes:
    real output is forwarded verbatim, a job that already reported through
    send tools is left alone, and an empty result is flagged — a background
    job can only ever end by messaging (or having messaged) the chat."""
    reply = "\n\n".join(result.texts).strip()
    if not reply:
        if result.sent_messages > 0:
            return  # the job already delivered its report via send tools
        reply = (
            f"Background job {task['id']} finished but produced no output — "
            "flagging it so it doesn't disappear silently."
        )
    kwargs: dict[str, Any] = {}
    if trigger_msg_id:
        kwargs["reply_parameters"] = ReplyParameters(message_id=trigger_msg_id)
    try:
        await send_long_message(bot, chat_id=int(task["chat_id"]), text=reply, **kwargs)
    except Exception:
        log.exception("delegation_report_failed", task_id=task["id"])


_running_tasks: dict[str, asyncio.Task[None]] = {}

# Strong refs to short-lived background tasks (prevents premature GC)
_background_tasks: set[asyncio.Task[None]] = set()


async def start_scheduler_loop(
    bot: Bot, sem: asyncio.Semaphore, *, shutdown: asyncio.Event | None = None
) -> None:
    """Main scheduler loop — checks for due tasks every interval.

    If *shutdown* is provided, the loop exits when the event is set.

    A thin wrapper so the published shutdown event is released on EVERY exit,
    including a crash on the way in. A stuck-set event makes every later task
    failure read as "we killed it" — the same misclassification as before, only
    inverted and quieter.
    """
    try:
        await _scheduler_loop(bot, sem, shutdown=shutdown)
    finally:
        _release_shutdown()


async def _scheduler_loop(
    bot: Bot, sem: asyncio.Semaphore, *, shutdown: asyncio.Event | None = None
) -> None:
    global _deep_work_task, _shutdown
    # Publish it before the first tick: a task that starts must be able to find
    # out, in its own failure handler, whether the process is going down.
    _shutdown = shutdown
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
    # Newly created tasks (schedule_task tool) start now, not next tick
    bus.on("cron_created", _on_task_created)

    while not (shutdown and shutdown.is_set()):
        # Wait one tick — or less, when a task is created or the wake socket
        # is poked. Shutdown interrupts the wait as before.
        await _sleep_until_tick(shutdown)
        if shutdown and shutdown.is_set():
            break
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
                expired_corrections = memory.prune_pending_corrections()
                embeddings_backfilled = memory.backfill_missing_embeddings()
                plans_reconciled = memory.reconcile_stale_plans()
                plans_nudged = await enforce_plan_momentum(bot)
                db.set_behavior_last_run("cleanup", datetime.now(UTC).isoformat())
                log.info(
                    "hourly_maintenance",
                    decayed=updated,
                    sessions_cleaned=len(cleaned_ids),
                    task_logs_pruned=pruned_logs,
                    outbound_pruned=pruned_outbound,
                    events_pruned=pruned_events,
                    working_expired=expired_working,
                    corrections_expired=expired_corrections,
                    embeddings_backfilled=embeddings_backfilled,
                    plans_reconciled=plans_reconciled,
                    plans_nudged=plans_nudged,
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
            results: list[BaseException | None] = await asyncio.gather(
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
                            reflections = memory.prune_stale_reflections()
                            if reflections:
                                log.info("stale_reflections_archived", count=reflections)

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
                                        bus.emit(
                                            "low_quality_work",
                                            {
                                                "goal_id": goal_id,
                                                "reason": (
                                                    f"Critic: claimed {claimed}/5 "
                                                    "but 0 steps checked off"
                                                ),
                                            },
                                        )
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
                    # Check for deep_work_oriented events in the last 30 seconds.
                    # events.created is written by sqlite datetime('now') — space
                    # separator, no offset — so the comparison string must match
                    # that format ('T' > ' ' made isoformat() always compare false,
                    # turning every check into a spurious continuation_failure).
                    since = (datetime.now(UTC) - timedelta(seconds=30)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    oriented = db.count_unconsumed_events("deep_work_oriented", since=since)
                    if oriented > 0:
                        bus.emit("continuation_success")
                        log.info("continuation_verified", result="success")
                    else:
                        bus.emit("continuation_failure")
                        log.info("continuation_verified", result="failure")

                verify_task = asyncio.create_task(_verify_continuation())
                _background_tasks.add(verify_task)
                verify_task.add_done_callback(_background_tasks.discard)

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
    # This is where the interruptions happen, so `_shutdown` must still be
    # readable HERE — it is released only after the last task has been awaited.
    pending = list(_running_tasks.values())
    if pending:
        log.info("Draining running tasks", count=len(pending))
        await asyncio.gather(*pending, return_exceptions=True)

    # `start_scheduler_loop`'s finally releases the shutdown event from here.
