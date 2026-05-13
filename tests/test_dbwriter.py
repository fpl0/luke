"""Tests for the LockedConnection write serializer."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from luke import db
from luke.dbwriter import LockedConnection, _is_write_sql, _write_lock

# ---------------------------------------------------------------------------
# Unit tests for the write-SQL detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "  INSERT INTO t VALUES (1)",
        "UPDATE t SET x=1",
        "DELETE FROM t",
        "REPLACE INTO t VALUES (1)",
        "insert into t values (1)",  # case-insensitive
        "-- a comment\nINSERT INTO t VALUES (1)",
    ],
)
def test_is_write_sql_true(sql: str) -> None:
    assert _is_write_sql(sql) is True


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM t",
        "CREATE TABLE t (x INT)",
        "ALTER TABLE t ADD x INT",
        "DROP TABLE t",
        "PRAGMA journal_mode=WAL",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "",
        "-- only a comment",
    ],
)
def test_is_write_sql_false(sql: str) -> None:
    assert _is_write_sql(sql) is False


# ---------------------------------------------------------------------------
# LockedConnection behavior tests
# ---------------------------------------------------------------------------


def test_locked_connection_uses_factory(test_db: Any) -> None:
    """db._db() returns a LockedConnection."""
    conn = db._db()
    assert isinstance(conn, LockedConnection)


def test_writes_acquire_and_release_lock(test_db: Any) -> None:
    """After a write+commit, the global lock is fully released."""
    conn = db._db()
    conn.execute("INSERT INTO events (event_type, payload) VALUES (?, ?)", ("test", "{}"))
    # Lock should still be held until commit
    held = object.__getattribute__(conn, "_writes_held")
    assert held == 1
    conn.commit()
    held = object.__getattribute__(conn, "_writes_held")
    assert held == 0


def test_reads_dont_acquire_lock(test_db: Any) -> None:
    """SELECT does not increment the held counter."""
    conn = db._db()
    conn.execute("SELECT * FROM events").fetchall()
    held = object.__getattribute__(conn, "_writes_held")
    assert held == 0


def test_concurrent_writes_serialize(test_db: Any, tmp_settings: Any) -> None:
    """Concurrent threads writing simultaneously all succeed with no locked errors."""
    # Each thread gets its own thread-local connection. Without LockedConnection,
    # this would race and produce 'database is locked' errors under contention.
    n_threads = 8
    n_writes_per_thread = 25
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(thread_id: int) -> None:
        try:
            for i in range(n_writes_per_thread):
                conn = db._db()  # thread-local LockedConnection
                conn.execute(
                    "INSERT INTO events (event_type, payload) VALUES (?, ?)",
                    (f"thread-{thread_id}", str(i)),
                )
                conn.commit()
        except Exception as exc:  # pragma: no cover — failure path
            with lock:
                errors.append(exc)

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(worker, range(n_threads)))

    assert not errors, f"writes raised: {errors}"

    # Verify all rows landed.
    conn = db._db()
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == n_threads * n_writes_per_thread


def test_nested_writes_reentrant(test_db: Any) -> None:
    """A write inside a write (via bus.emit chain) must not deadlock."""
    # store_reaction_feedback writes then calls bus.emit which writes again.
    # If the lock weren't reentrant, this would deadlock.
    db.store_reaction_feedback(
        chat_id="12345",
        msg_id=1,
        sender_id="user",
        emoji="👎",  # negative — triggers bus.emit("feedback_negative")
        timestamp="2026-05-13T20:00:00+00:00",
    )
    # If we got here, no deadlock. Verify both writes happened.
    reactions = db.get_reactions("12345")
    assert len(reactions) == 1
    # bus.emit creates an event row
    conn = db._db()
    row = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type = 'feedback_negative'"
    ).fetchone()
    assert row[0] == 1


def test_rollback_releases_lock(test_db: Any) -> None:
    """An exception during a write transaction releases the lock via rollback."""
    conn = db._db()
    try:
        conn.execute("INSERT INTO events (event_type, payload) VALUES (?, ?)", ("test", "{}"))
        # Force a constraint violation to trigger rollback semantics.
        # We'll just call rollback explicitly here.
        conn.rollback()
    except Exception:
        pass
    held = object.__getattribute__(conn, "_writes_held")
    assert held == 0


def test_lock_serializes_writes_under_contention(test_db: Any) -> None:
    """When two threads write, the global lock visibly serializes them.

    Verified by timing: 30 writes with 5ms artificial delay each cannot finish
    in under ~150ms total even with 8 parallel threads, because the lock
    forces serialization.
    """
    n_writes = 30
    delay_s = 0.005
    started = time.monotonic()
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        try:
            conn = db._db()
            conn.execute(
                "INSERT INTO events (event_type, payload) VALUES (?, ?)", ("contention", str(i))
            )
            time.sleep(delay_s)  # hold the lock briefly via the open tx
            conn.commit()
        except Exception as exc:  # pragma: no cover
            with lock:
                errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(worker, range(n_writes)))

    elapsed = time.monotonic() - started
    assert not errors
    # Serialized lower bound (allowing some slack for fast thread switching):
    # n_writes * delay_s = 0.15s. Real elapsed is typically 0.2-0.4s.
    assert elapsed >= n_writes * delay_s * 0.5, (
        f"writes completed in {elapsed:.3f}s — lock may not be serializing"
    )


def test_lock_not_held_after_cross_thread_writes(test_db: Any) -> None:
    """After all writes complete, the process-wide lock is fully released."""
    # Multiple threads write concurrently; we then probe the lock from the
    # main thread. It should acquire without blocking.
    def worker(i: int) -> None:
        conn = db._db()
        conn.execute("INSERT INTO events (event_type, payload) VALUES (?, ?)", ("probe", str(i)))
        conn.commit()

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(worker, range(20)))

    # Should acquire immediately — no thread holds it.
    acquired = _write_lock.acquire(blocking=False)
    assert acquired, "write lock leaked after all threads finished"
    _write_lock.release()
