"""deep_work_quality.shipped — the fact the rating gate computes must reach the row.

Built 2026-09-06 by the midnight self-reflection. Context: `rating_gate.py`
shipped 2026-09-03 to stop a 4 in a session that delivered nothing. It worked in
the narrow sense and failed in the wide one — over the 24 sessions after it
landed, EVERY rating was a 3. The gate truncated the top of the scale without
making the bottom reachable, so the histogram went from degenerate to constant,
which carries strictly less information than the two-value version before it.

The reason that was invisible for three days is that `shipped` was computed at
the gate and thrown away, leaving `deep_work_calibration.py` to infer reach from
the messages table BY DATE — it prints the caveat itself: "nothing links a send
to the session that produced it". These tests pin the fact to the row.

2026-09-07, marker SHIPPED-NEVER-CROSSED-THE-TASK-BOUNDARY. The 6 Sep version of
this file passed on every run while the column stayed NULL on all 514 rows in
production, because every test set the value and read it back INSIDE ONE TASK —
the single configuration in which a ContextVar works. The production path puts
the PreToolUse hook and the MCP tool handler in different asyncio tasks, and a
Task copies its context at creation, so the writer never saw it. The test named
`..._does_not_leak_across_concurrent_runs` was the closest miss: it ran two
coroutines, but each one published AND wrote, so it proved isolation without
ever crossing the boundary that actually mattered.

`test_publisher_and_writer_in_different_tasks` below is the test that would have
caught it, and it is the reason the mechanism is now a module-level map.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

import pytest

from luke import db as dbmod
from luke.config import settings
from luke.rating_gate import (
    _PENDING_SHIPPED,
    TOOL_NAME,
    blocks,
    consume_shipped,
    publish_shipped,
)


def _rows() -> list[tuple[Any, ...]]:
    con = sqlite3.connect(settings.store_dir / "luke.db")
    try:
        return con.execute(
            "SELECT goal_id, rating, shipped FROM deep_work_quality ORDER BY id"
        ).fetchall()
    finally:
        con.close()


@pytest.fixture()
def clean_ctx():
    """Empty the pending map around each test — it is process-wide."""
    _PENDING_SHIPPED.clear()
    yield
    _PENDING_SHIPPED.clear()


def test_migration_adds_a_nullable_column(test_db, clean_ctx):
    con = sqlite3.connect(settings.store_dir / "luke.db")
    try:
        cols = {r[1]: r for r in con.execute("PRAGMA table_info(deep_work_quality)")}
    finally:
        con.close()
    assert "shipped" in cols, "migration 17 did not run"
    # PRAGMA reports dflt_value as the literal SQL text, so `DEFAULT NULL`
    # reads back as the STRING 'NULL' — assert the property, not the spelling.
    notnull, default = cols["shipped"][3], cols["shipped"][4]
    assert notnull == 0
    assert default in (None, "NULL")

    # The invariant that actually matters: a row written without the column
    # reads NULL, not 0. The 501 historical rows predate the fact and "unknown"
    # is the honest value — a DEFAULT 0 would invent 501 unshipped sessions and
    # every reading downstream would inherit the fiction.
    con = sqlite3.connect(settings.store_dir / "luke.db")
    try:
        con.execute("INSERT INTO deep_work_quality (goal_id, rating) VALUES ('legacy', 3)")
        con.commit()
        assert con.execute(
            "SELECT shipped FROM deep_work_quality WHERE goal_id='legacy'"
        ).fetchone() == (None,)
    finally:
        con.close()


def test_unset_context_records_unknown_not_false(test_db, clean_ctx):
    """No hook ran -> NULL. That is the whole point of the None sentinel."""
    dbmod.log_deep_work_quality("goal-x", 3)
    assert _rows() == [("goal-x", 3, None)]


def test_shipped_and_unshipped_are_persisted(test_db, clean_ctx):
    publish_shipped("goal-a", True)
    dbmod.log_deep_work_quality("goal-a", 4)
    publish_shipped("goal-b", False)
    dbmod.log_deep_work_quality("goal-b", 3)
    assert _rows() == [("goal-a", 4, 1), ("goal-b", 3, 0)]


def test_out_of_range_rating_still_writes_nothing(test_db, clean_ctx):
    publish_shipped("goal-x", True)
    dbmod.log_deep_work_quality("goal-x", 9)
    dbmod.log_deep_work_quality("goal-x", 0)
    assert _rows() == []


def test_publisher_and_writer_in_different_tasks(test_db, clean_ctx):
    """THE regression test. This is the production shape and the 6 Sep bug.

    The PreToolUse hook and the MCP tool handler are separate asyncio tasks. A
    Task copies the context at creation, so a ContextVar set in the first is
    structurally invisible to the second — which is why all 514 rows were NULL
    while every test passed. Publishing must survive the boundary.
    """

    async def main() -> None:
        # The hook's task publishes...
        await asyncio.create_task(_publish("goal-a", True))
        # ...and a DIFFERENT task does the write, exactly as in production.
        await asyncio.create_task(_write("goal-a", 4))

    async def _publish(goal: str, shipped: bool) -> None:
        publish_shipped(goal, shipped)

    async def _write(goal: str, rating: int) -> None:
        dbmod.log_deep_work_quality(goal, rating)

    asyncio.run(main())
    assert _rows() == [("goal-a", 4, 1)], "the fact did not cross the task boundary"


def test_concurrent_runs_on_different_goals_stay_separate(test_db, clean_ctx):
    """Concurrent runs are real — two Luke sessions on one trigger, 2026-09-03.

    Keying on goal_id is what buys back the isolation the ContextVar was chosen
    for. Run A's send must not make run B's rating look landed.
    """

    async def run(name: str, shipped: bool, rating: int) -> None:
        publish_shipped(name, shipped)
        await asyncio.sleep(0)  # let the other task interleave
        dbmod.log_deep_work_quality(name, rating)

    async def main() -> None:
        await asyncio.gather(run("goal-a", True, 4), run("goal-b", False, 3))

    asyncio.run(main())
    assert {g: s for g, _, s in _rows()} == {"goal-a": 1, "goal-b": 0}


def test_consume_is_read_and_clear(test_db, clean_ctx):
    """A blocked rating leaves an entry; it must not attach to a later write.

    The gate fires the hook on every attempt but only unblocked attempts reach
    the writer, so the map would otherwise hand a stale fact to the next
    hookless rating of the same goal.
    """
    publish_shipped("goal-a", True)
    assert consume_shipped("goal-a") is True
    assert consume_shipped("goal-a") is None
    dbmod.log_deep_work_quality("goal-a", 3)
    assert _rows() == [("goal-a", 3, None)]


def test_pending_map_is_bounded(test_db, clean_ctx):
    """Blocked attempts leak an entry each; the map must not grow forever."""
    for i in range(200):
        publish_shipped(f"goal-{i}", True)
    assert len(_PENDING_SHIPPED) <= 64
    # Eviction is oldest-first, so the most recent publishes survive.
    assert consume_shipped("goal-199") is True
    assert consume_shipped("goal-0") is None


def test_gate_still_blocks_only_a_four_that_shipped_nothing():
    """Guard the 2026-09-03 behaviour while adding to the module around it."""
    unshipped = dict(shipped=False, already_fired=False)
    assert blocks(TOOL_NAME, {"rating": 4}, **unshipped)
    assert blocks(TOOL_NAME, {"rating": 5}, **unshipped)
    assert not blocks(TOOL_NAME, {"rating": 3}, **unshipped)
    assert not blocks(TOOL_NAME, {"rating": 4}, shipped=True, already_fired=False)
    assert not blocks(TOOL_NAME, {"rating": 4}, shipped=False, already_fired=True)
    assert not blocks("mcp__luke__send_message", {"rating": 4}, **unshipped)
