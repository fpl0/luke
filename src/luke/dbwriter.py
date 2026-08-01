"""Single-writer coordination for SQLite via a Connection subclass.

The crash on 2026-05-13 was caused by two threads racing to write to the same
SQLite file: the main event loop thread emitting an event via ``bus.emit`` while
``asyncio.to_thread`` workers (corrections, memory updates) were writing through
their own thread-local connections. WAL allows concurrent reads but file-level
locking serializes writes; ``busy_timeout=10s`` plus contention bursts exhausted
the wait and surfaced as ``sqlite3.OperationalError: database is locked``.

This module provides one mechanism — :class:`LockedConnection`, a
``sqlite3.Connection`` subclass that auto-acquires a process-wide reentrant
lock on every write statement and releases it on commit/rollback. One writer
at a time across all threads eliminates the race entirely. Reads stay
unguarded — WAL handles concurrent reader/writer correctly.

Used by ``db._db()`` as the connection factory; no other code needs to know
about it. New write paths in any module participate automatically because they
go through the same connection.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any, cast

import structlog
from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.get_logger()


# Process-wide write lock. Reentrant so a write that triggers another write
# (e.g. ``store_reaction_feedback`` → ``bus.emit`` → ``emit_event``) doesn't
# deadlock. Only LockedConnection should touch this directly.
_write_lock = threading.RLock()


_WRITE_PREFIXES: tuple[str, ...] = ("INSERT", "UPDATE", "DELETE", "REPLACE")


def _is_write_sql(sql: str) -> bool:
    """Return True if the SQL statement begins with a write keyword.

    Tolerates leading whitespace and single-line ``--`` comments. Returns False
    for SELECT, CREATE, ALTER, DROP, BEGIN, COMMIT, ROLLBACK, PRAGMA — those
    either don't need the lock (reads, pragmas) or run only during startup
    (migrations, BEGIN/COMMIT issued explicitly).
    """
    if not sql:
        return False
    text = sql.lstrip()
    while text.startswith("--"):
        newline = text.find("\n")
        if newline == -1:
            return False
        text = text[newline + 1 :].lstrip()
    if not text:
        return False
    first_token = text[:10].upper().split(None, 1)[0]
    return first_token in _WRITE_PREFIXES


class LockedConnection(sqlite3.Connection):
    """sqlite3.Connection subclass that auto-serializes writes process-wide.

    Every ``execute``/``executemany``/``executescript`` call inspects its SQL.
    If it's a write (INSERT/UPDATE/DELETE/REPLACE), the global write lock is
    acquired and held until ``commit()`` or ``rollback()`` releases it. Reads
    don't acquire the lock — SQLite's WAL allows concurrent readers.

    The lock is reentrant: nested writes (e.g. ``store_reaction_feedback``
    calling ``bus.emit`` which calls ``emit_event``) acquire it again without
    deadlocking. An instance counter tracks how many times we acquired so
    ``commit``/``rollback`` releases all of them.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # sqlite3.Connection is implemented in C; use object.__setattr__ to
        # attach instance state.
        object.__setattr__(self, "_writes_held", 0)

    def _acquire_write(self) -> None:
        _write_lock.acquire()
        held = cast(int, object.__getattribute__(self, "_writes_held"))
        object.__setattr__(self, "_writes_held", held + 1)

    def _release_writes(self) -> None:
        held = cast(int, object.__getattribute__(self, "_writes_held"))
        while held > 0:
            try:
                _write_lock.release()
            except RuntimeError:
                break
            held -= 1
        object.__setattr__(self, "_writes_held", held)

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        if _is_write_sql(sql):
            self._acquire_write()
        return super().execute(sql, parameters)

    def executemany(self, sql: str, seq_of_parameters: Any, /) -> sqlite3.Cursor:
        if _is_write_sql(sql):
            self._acquire_write()
        return super().executemany(sql, seq_of_parameters)

    def executescript(self, sql_script: str, /) -> sqlite3.Cursor:
        # Conservatively acquire if any statement in the script is a write.
        if any(_is_write_sql(stmt) for stmt in sql_script.split(";")):
            self._acquire_write()
        return super().executescript(sql_script)

    def commit(self) -> None:
        try:
            super().commit()
        finally:
            self._release_writes()

    def rollback(self) -> None:
        try:
            super().rollback()
        finally:
            self._release_writes()
