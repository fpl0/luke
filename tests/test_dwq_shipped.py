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
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

import pytest

from luke import db as dbmod
from luke.config import settings
from luke.rating_gate import CURRENT_RUN_SHIPPED, TOOL_NAME, blocks


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
    """Reset the contextvar around each test — it is process-wide otherwise."""
    token = CURRENT_RUN_SHIPPED.set(None)
    yield
    CURRENT_RUN_SHIPPED.reset(token)


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
        con.execute(
            "INSERT INTO deep_work_quality (goal_id, rating) VALUES ('legacy', 3)"
        )
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
    CURRENT_RUN_SHIPPED.set(True)
    dbmod.log_deep_work_quality("goal-a", 4)
    CURRENT_RUN_SHIPPED.set(False)
    dbmod.log_deep_work_quality("goal-b", 3)
    assert _rows() == [("goal-a", 4, 1), ("goal-b", 3, 0)]


def test_out_of_range_rating_still_writes_nothing(test_db, clean_ctx):
    CURRENT_RUN_SHIPPED.set(True)
    dbmod.log_deep_work_quality("goal-x", 9)
    dbmod.log_deep_work_quality("goal-x", 0)
    assert _rows() == []


def test_contextvar_does_not_leak_across_concurrent_runs(test_db, clean_ctx):
    """Concurrent runs are real — two Luke sessions on one trigger, 2026-09-03.

    A module global would let run A's send make run B's rating look landed.
    """

    async def run(name: str, shipped: bool, rating: int) -> None:
        CURRENT_RUN_SHIPPED.set(shipped)
        await asyncio.sleep(0)  # let the other task interleave
        dbmod.log_deep_work_quality(name, rating)

    async def main() -> None:
        await asyncio.gather(run("goal-a", True, 4), run("goal-b", False, 3))

    asyncio.run(main())
    assert {g: s for g, _, s in _rows()} == {"goal-a": 1, "goal-b": 0}


def test_gate_still_blocks_only_a_four_that_shipped_nothing():
    """Guard the 2026-09-03 behaviour while adding to the module around it."""
    unshipped = dict(shipped=False, already_fired=False)
    assert blocks(TOOL_NAME, {"rating": 4}, **unshipped)
    assert blocks(TOOL_NAME, {"rating": 5}, **unshipped)
    assert not blocks(TOOL_NAME, {"rating": 3}, **unshipped)
    assert not blocks(TOOL_NAME, {"rating": 4}, shipped=True, already_fired=False)
    assert not blocks(TOOL_NAME, {"rating": 4}, shipped=False, already_fired=True)
    assert not blocks("mcp__luke__send_message", {"rating": 4}, **unshipped)
