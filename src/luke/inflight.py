"""In-flight agent-run marker — lets an external process know when Luke is busy.

WHY THIS EXISTS
---------------
``deploy.sh`` restarts the service with ``launchctl kickstart -k``, which SIGTERMs
immediately.  ``app.py``'s shutdown handler *does* await its background tasks, but
launchd's default ``ExitTimeOut`` is 20 seconds and an agent turn runs 5-15 minutes,
so the drain never completes: launchd SIGKILLs the process with the turn still open.
Observed on 2026-08-01 — a 17-minute answer lost during the morning storm, and a
15:00 cron killed mid-run.

The durable fix is not a longer timeout (nobody waits 15 minutes for a deploy) but
*not restarting while a turn is open*.  That needs a signal a shell script can read,
which is this file.

FORMAT
------
``$LUKE_DIR/inflight`` holds one line::

    <unix_ts> <pid> <user_runs> <auto_runs>

Written atomically (tmp + rename) on every transition, so a reader never sees a
partial line.  ``pid`` is what makes it safe: a process killed mid-run leaves a
non-zero count behind forever, so a reader MUST ignore the file unless ``pid``
matches the live process (cross-check against ``$LUKE_DIR/heartbeat``, which the
scheduler refreshes every tick).  That is why the timestamp alone is not enough —
a legitimately long run leaves the timestamp 15 minutes stale.

Everything here is best-effort.  A failure to write the marker must never take down
a turn, so every OSError is swallowed: the worst case is a deploy that restarts
during a turn, which is exactly today's behaviour.
"""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from .config import settings

_user_runs = 0
_auto_runs = 0


def _path() -> Any:
    return settings.store_dir / "inflight"


def _write() -> None:
    """Persist the current counts. Best effort — never raises."""
    try:
        target = _path()
        tmp = target.with_suffix(".tmp")
        tmp.write_text(f"{int(time.time())} {os.getpid()} {_user_runs} {_auto_runs}\n")
        tmp.rename(target)
    except Exception:
        # Deliberately broad. This wraps every user turn and every scheduled run;
        # a marker that cannot be written must degrade to "deploy might interrupt
        # a turn" — the status quo — never to a turn that dies writing telemetry.
        pass


def reset() -> None:
    """Zero the counters and the file. Called once at startup.

    A process that died mid-run left a stale non-zero file behind.  Readers are
    told to gate on the pid, but clearing it at boot means a reader that forgets
    still gets the truth rather than a permanent "busy".
    """
    global _user_runs, _auto_runs
    _user_runs = 0
    _auto_runs = 0
    _write()


def counts() -> tuple[int, int]:
    """(user_runs, auto_runs) currently open in this process."""
    return _user_runs, _auto_runs


def begin(*, autonomous: bool) -> None:
    global _user_runs, _auto_runs
    if autonomous:
        _auto_runs += 1
    else:
        _user_runs += 1
    _write()


def end(*, autonomous: bool) -> None:
    global _user_runs, _auto_runs
    if autonomous:
        _auto_runs = max(0, _auto_runs - 1)
    else:
        _user_runs = max(0, _user_runs - 1)
    _write()


def tracked[T](fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorator: count an async run as in-flight for its whole lifetime.

    Reads ``autonomous`` from the call's keyword arguments — ``run_agent`` is
    keyword-only, so this is reliable.  A user turn and a scheduled task are
    counted separately because they are worth different amounts: a deploy should
    always wait out a user turn, and may reasonably choose not to wait out a cron.
    """

    @functools.wraps(fn)
    async def _wrapper(*args: Any, **kwargs: Any) -> T:
        autonomous = bool(kwargs.get("autonomous", False))
        begin(autonomous=autonomous)
        try:
            return await fn(*args, **kwargs)
        finally:
            end(autonomous=autonomous)

    return _wrapper
