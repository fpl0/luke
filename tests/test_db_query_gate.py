"""Tests for the luke.db hand-query gate.

The BLOCKED cases below are real commands lifted from luke.log on 2026-08-07/08
(the "no such column" run that reflect.sh counted at 12 in 30h). The PASSES are
the neighbours that must stay untouched — a gate that fires on unrelated
commands gets waved away, and then it protects nothing.
"""

from __future__ import annotations

import pytest

from luke.db_query_gate import REASON, blocks_bash_command, blocks_tool_input

# --- Real commands from the log that should have been stopped ----------------
REAL_MISSES = [
    'sqlite3 luke.db ".tables" 2>&1 | head -40',
    'sqlite3 luke.db "pragma table_info(tasks);"',
    'sqlite3 luke.db ".schema messages"',
    'sqlite3 luke.db ".schema tasks" | head -20',
    'sqlite3 /Users/filipelm/Luke/luke.db "select id, type, substr(title,1,70) from memory_meta"',
    'sqlite3 luke.db "select ts, sender, substr(text,1,300) from messages order by ts desc"',
    "python3 -c \"import sqlite3; c=sqlite3.connect('/Users/filipelm/Luke/luke.db')\"",
    "cd /Users/filipelm/Luke && sqlite3 luke.db 'select created_at from events'",
]


@pytest.mark.parametrize("command", REAL_MISSES)
def test_blocks_real_hand_written_queries(command: str) -> None:
    assert blocks_bash_command(command) is True


# --- Neighbours that must keep working ---------------------------------------
MUST_PASS = [
    # The sanctioned wrapper itself — this is what the gate steers to.
    'workspace/tools/q.sh "select id from events limit 5"',
    "workspace/tools/q.sh --schema events tasks",
    "/Users/filipelm/Luke/workspace/tools/q.sh --schema",
    "bash workspace/tools/reflect.sh",
    # Naming the DB without querying it.
    'find . -maxdepth 3 -iname "luke.db" 2>/dev/null | grep -v job-search',
    "ls -la /Users/filipelm/Luke/luke.db",
    "cp /Users/filipelm/Luke/luke.db /Users/filipelm/Luke/backups/luke.db.bak",
    "grep -rn luke.db src/",
    # sqlite against some other database is none of this gate's business.
    'sqlite3 ~/Library/Mail/envelope.db "select * from messages limit 1"',
    "python3 -c \"import sqlite3; sqlite3.connect('other.db')\"",
    # Not a shell command at all.
    "",
]


@pytest.mark.parametrize("command", MUST_PASS)
def test_allows_everything_else(command: str) -> None:
    assert blocks_bash_command(command) is False


def test_non_string_input_is_ignored() -> None:
    assert blocks_bash_command(None) is False
    assert blocks_bash_command(42) is False
    assert blocks_bash_command({"command": "sqlite3 luke.db 'select 1'"}) is False


def test_blocks_tool_input_unwraps_the_payload() -> None:
    assert blocks_tool_input({"command": 'sqlite3 luke.db "select 1"'}) is True
    assert blocks_tool_input({"command": "workspace/tools/q.sh --schema"}) is False
    assert blocks_tool_input({}) is False
    assert blocks_tool_input(None) is False
    # A non-Bash payload shape must not throw.
    assert blocks_tool_input({"file_path": "luke.db"}) is False


def test_reason_names_the_replacement_and_the_real_columns() -> None:
    # The steer is the whole value of the block: it has to carry the working
    # command, not just a refusal.
    assert "workspace/tools/q.sh" in REASON
    assert "--schema" in REASON
    assert "event_type" in REASON
    assert "memory_meta" in REASON


def test_python_sqlite3_module_reference_alone_does_not_trip_it() -> None:
    # Catching an exception type is not opening a connection.
    assert blocks_bash_command("python3 -c 'except sqlite3.OperationalError: pass'") is False
