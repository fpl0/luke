"""Goal-level work claims — stop two autonomous sessions building the same thing.

The scheduler already dedups *per task id* (docs/concurrency.md: "skips any task whose
previous run is still in-flight"). That is not enough. Several DIFFERENT tasks can drive
work on the SAME goal — dedicated build-session crons (02:00 / 11:00 / 19:00), the generic
deep-work trigger, and the Sunday self-audit all pick a goal off the active list and start
editing. Different task ids, so per-task dedup never fires, and two sessions end up writing
the same files.

Observed, not hypothetical (2026-08-02): a deep-work tick at 03:03Z and a build session
that started minutes earlier were both implementing the same harness fix, editing the same
two modules concurrently. The same overlap had already
cost real work — the 01:03Z session died after 30 minutes leaving 1,245 uncommitted lines
that the next session had to find and recover before it could do anything else.

A claim is ADVISORY and deliberately weak in one direction: it can occasionally grant twice
under a race, but it must never wrongly *deny*. Two design rules follow from the Aug 10
deadline:

  * **Fail open.** Any unexpected error grants the claim and logs a warning. A bug in this
    module must never be able to stall a goal that has a deadline.
  * **Never kill the holder.** ``app.py``'s single-instance lock SIGTERMs a stale holder,
    which is right for "only one Luke daemon". It is wrong here: the peer is a sibling work
    session that may hold uncommitted edits, and killing it would cause exactly the lost
    work this module exists to prevent. A loser yields; it never evicts.

Staleness is TTL-based rather than flock-based on purpose. ``fcntl.flock`` is tied to an
open fd, so it dies with the process that took it — useless when the claim has to outlive a
short-lived CLI call and cover a 30-minute agent session. A recorded pid is checked when it
is on this machine (best effort), but the TTL is what guarantees a crashed session's claim
is eventually reclaimable.

**Only record a pid whose life matches the work.** The first cut defaulted to
``os.getpid()`` everywhere, which quietly inverted the whole module when driven from the
CLI: the recorded pid belonged to the CLI process, which had already exited, so the next
caller saw "holder is gone", reclaimed, and the claim granted to *everyone*. In-process
callers keep the automatic pid (the process really is the holder); the CLI records none by
default and leans on the TTL, and takes ``--pid`` when the caller knows the real session's
pid. Unit tests missed this because both claims ran inside one live pytest process — it
only showed up when the CLI was run twice for real.

CLI, for use from a cron/session prompt:

    TOKEN=$(python3 -m luke.work_claim acquire goal-parity-build --holder "02:00 build" | tail -1)
      exit 0  -> claimed; the release token is the LAST line of stdout
      exit 3  -> held by a live session; the caller should do something else
    python3 -m luke.work_claim release goal-parity-build "$TOKEN"
    python3 -m luke.work_claim status goal-parity-build

``tail -1`` is not incidental: structlog writes its own lines to stdout, so the token is
emitted last and callers must take the last line rather than the whole capture.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import time
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog

from .config import settings

log = structlog.get_logger()

# The TTL has to outlive the session it protects, so it is DERIVED from the session ceiling
# rather than guessed. The first cut hardcoded 2700 ("a deep-work session runs ~30 min — the
# 2026-08-01 01:03Z session was killed by a 30-minute watchdog"), and that assumption expired
# within a day: commit 6abfe45 raised `deep_work_timeout` to 5400. The constant stayed 2700,
# so every claim silently stopped covering the second half of the session it was taken for —
# at minute 45 the goal reads FREE and a peer walks straight into the running session. That is
# the exact collision this module exists to prevent, merely delayed by 45 minutes.
#
# The cost of the fix is one-sided and worth paying: a session that dies WITHOUT releasing
# sits on its goal for ~95 min instead of ~45. That failure is safe (a peer yields and does
# something else) and only reachable on the CLI path, where no pid is recorded — in-process
# callers record a real pid, so a crash is detected immediately regardless of TTL. Being
# robbed mid-session is the unsafe direction, and it already cost 1,245 uncommitted lines.
_TTL_MARGIN_SECONDS = 300


def default_ttl_seconds() -> int:
    """Claim lifetime: the deep-work session ceiling plus a margin."""
    try:
        return int(settings.deep_work_timeout) + _TTL_MARGIN_SECONDS
    except Exception:  # pragma: no cover - settings is always readable in practice
        return 5700


DEFAULT_TTL_SECONDS = default_ttl_seconds()

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]")

# Sentinel: "record this process's pid". Distinct from None, which means "record no pid at
# all — judge staleness by TTL alone", the correct choice for an ephemeral CLI caller.
_AUTO_PID = -1


@dataclass(frozen=True)
class Claim:
    """A granted claim. ``token`` is required to release it."""

    goal_id: str
    token: str
    holder: str
    expires_at: float

    @property
    def seconds_left(self) -> float:
        return max(0.0, self.expires_at - time.time())


def _claims_dir() -> str:
    return os.path.join(str(settings.store_dir), "claims")


def _claim_path(goal_id: str) -> str:
    # Goal ids are internal, but they reach here from cron prompt text, so never let one
    # escape the claims directory.
    safe = _SAFE_ID.sub("_", goal_id.strip()) or "unnamed"
    return os.path.join(_claims_dir(), f"{safe}.json")


def _pid_alive(pid: int | None) -> bool | None:
    """True/False if we can tell, None if we cannot (no pid recorded)."""
    if not pid:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except Exception:
        return None


def _read(path: str) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except Exception:
        # A truncated or corrupt claim file is treated as no claim — fail open.
        return None


def _is_stale(rec: dict[str, Any]) -> tuple[bool, str]:
    """Is an existing claim reclaimable? Returns (stale, reason)."""
    expires = rec.get("expires_at")
    if not isinstance(expires, (int, float)):
        return True, "no expiry recorded"
    if time.time() >= expires:
        return True, f"expired {int(time.time() - expires)}s ago"
    # Fresh by the clock — but if the holder is provably gone, do not make the next session
    # wait out the full TTL. Only trusted when the pid was recorded on this same host.
    if rec.get("host") == os.uname().nodename and _pid_alive(rec.get("pid")) is False:
        return True, f"holder pid {rec.get('pid')} is gone"
    return False, ""


def _write_claim(path: str, rec: dict[str, Any]) -> bool:
    """Create the claim file, failing if another process got there first."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(rec, f, indent=2)
        return True
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise


def claim(
    goal_id: str,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    holder: str = "",
    pid: int | None = _AUTO_PID,
) -> Claim | None:
    """Claim exclusive work on *goal_id*, or return None if a live session already has it.

    None is the ONLY denial signal, and it means "a peer is working this goal right now —
    go do something else". Every error path grants instead, because a false denial before a
    deadline is worse than an occasional double-grant on an advisory lock.

    ``pid`` defaults to this process, which is right when the caller IS the holder. Pass
    ``None`` when the claim will outlive the calling process (the CLI does), so staleness
    falls back to the TTL instead of a pid that is about to die.
    """
    try:
        os.makedirs(_claims_dir(), exist_ok=True)
        path = _claim_path(goal_id)
        token = secrets.token_hex(8)
        now = time.time()
        expires_at = now + max(60, int(ttl_seconds))
        rec = {
            "goal_id": goal_id,
            "token": token,
            "holder": holder,
            "pid": os.getpid() if pid == _AUTO_PID else pid,
            "host": os.uname().nodename,
            "claimed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "expires_at": expires_at,
        }

        if not _write_claim(path, rec):
            existing = _read(path)
            if existing is None:
                # Unreadable/corrupt — treat as free and take it.
                with contextlib.suppress(OSError):
                    os.unlink(path)
                if not _write_claim(path, rec):
                    return None
            else:
                stale, why = _is_stale(existing)
                if not stale:
                    log.info(
                        "work_claim_denied",
                        goal_id=goal_id,
                        held_by=existing.get("holder"),
                        pid=existing.get("pid"),
                    )
                    return None
                # Reclaim: unlink only while the file still shows the SAME stale claim we
                # inspected, so we cannot delete a claim someone else just refreshed. The
                # window between check and unlink is not closed — this is an advisory claim
                # between sessions minutes apart, not a distributed lock, and the failure it
                # guards is wasted work rather than corruption.
                current = _read(path)
                if not current or current.get("token") != existing.get("token"):
                    return None
                with contextlib.suppress(OSError):
                    os.unlink(path)
                if not _write_claim(path, rec):
                    return None
                log.info("work_claim_reclaimed", goal_id=goal_id, reason=why)

        # Confirm we are the claim of record — catches the reclaim race above.
        final = _read(path)
        if not final or final.get("token") != token:
            return None

        log.info("work_claim_granted", goal_id=goal_id, holder=holder, ttl=ttl_seconds)
        return Claim(goal_id=goal_id, token=token, holder=holder, expires_at=expires_at)
    except Exception as e:
        # Fail OPEN — never let a bug here stall a goal with a deadline.
        log.warning("work_claim_error_failing_open", goal_id=goal_id, error=str(e))
        return Claim(
            goal_id=goal_id,
            token="",
            holder=holder,
            expires_at=time.time() + DEFAULT_TTL_SECONDS,
        )


def release(goal_id: str, token: str) -> bool:
    """Release a claim. Only the holder's token can release it; True if it was removed.

    An EMPTY token is refused rather than treated as a force-unlink. The only thing that
    produces a token-less claim is the fail-open path, which never wrote a claim file — so
    "release" from such a holder would delete whatever claim happens to be there, quite
    possibly a live peer's, turning a defensive fallback into the collision this module
    exists to prevent. Two call sites already had to remember that guard themselves; the
    trap belongs here instead.
    """
    try:
        if not token:
            log.info("work_claim_release_no_token", goal_id=goal_id)
            return False
        path = _claim_path(goal_id)
        rec = _read(path)
        if not rec:
            return False
        if rec.get("token") != token:
            log.info("work_claim_release_rejected", goal_id=goal_id)
            return False
        os.unlink(path)
        log.info("work_claim_released", goal_id=goal_id)
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        log.warning("work_claim_release_failed", goal_id=goal_id, error=str(e))
        return False


def current(goal_id: str) -> dict[str, Any] | None:
    """The live claim on *goal_id*, or None when free (including a stale claim)."""
    rec = _read(_claim_path(goal_id))
    if not rec:
        return None
    stale, _why = _is_stale(rec)
    if stale:
        return None
    return rec


@contextlib.contextmanager
def claimed(goal_id: str, **kw: Any) -> Generator[Claim | None]:
    """Context manager: yields a Claim, or None when a peer holds the goal."""
    c = claim(goal_id, **kw)
    try:
        yield c
    finally:
        if c is not None and c.token:
            release(goal_id, c.token)


def _main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd, goal_id = argv[1], argv[2]
    if cmd == "acquire":
        ttl = DEFAULT_TTL_SECONDS
        holder = ""
        if "--ttl" in argv:
            ttl = int(argv[argv.index("--ttl") + 1])
        if "--holder" in argv:
            holder = argv[argv.index("--holder") + 1]
        # No pid by default: THIS process exits in milliseconds while the claim must cover
        # the session that invoked it, and a recorded-then-dead pid makes every later caller
        # reclaim instantly (see the module docstring). --pid lets a caller that knows the
        # real session pid opt back into liveness detection.
        holder_pid = int(argv[argv.index("--pid") + 1]) if "--pid" in argv else None
        c = claim(goal_id, ttl_seconds=ttl, holder=holder, pid=holder_pid)
        if c is None:
            held = current(goal_id) or {}
            print(
                f"HELD {goal_id} by {held.get('holder') or 'another session'} "
                f"(pid {held.get('pid')}, since {held.get('claimed_at')})"
            )
            return 3
        print(c.token)
        return 0
    if cmd == "release":
        token = argv[3] if len(argv) > 3 else ""
        return 0 if release(goal_id, token) else 1
    if cmd == "status":
        snapshot = current(goal_id)
        print(json.dumps(snapshot, indent=2) if snapshot else f"FREE {goal_id}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
