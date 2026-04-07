"""Tests for luke.context — working memory injection and preservation manifests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from luke import context, db


def _insert_memory(
    conn: Any,
    mem_id: str,
    mem_type: str = "entity",
    title: str = "",
    content: str = "",
    importance: float = 1.0,
    status: str = "active",
    access_count: int = 0,
) -> None:
    """Insert a test memory into both memory_meta and memory_fts."""
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """INSERT INTO memory_meta
           (id, type, created, updated, access_count, importance, status, tags_json, links_json, last_accessed)
           VALUES (?, ?, ?, ?, ?, ?, ?, '[]', '[]', ?)""",
        (mem_id, mem_type, now, now, access_count, importance, status, now),
    )
    conn.execute(
        "INSERT INTO memory_fts (id, type, title, content, tags) VALUES (?, ?, ?, ?, '')",
        (mem_id, mem_type, title or mem_id, content or f"Content for {mem_id}"),
    )
    conn.commit()


class TestBuildWorkingContext:
    """Tests for build_working_context()."""

    def test_empty_db_returns_empty(self, test_db: Any) -> None:
        result = context.build_working_context()
        assert result == ""

    def test_single_goal_injected(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "goal-test", "goal", "Test Goal", "Status: active\nProgress: 50%", importance=1.5)
        result = context.build_working_context()
        assert "goal-test" in result
        assert "Active Goals" in result

    def test_single_entity_injected(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "person-alice", "entity", "Alice", "Alice is a developer", importance=1.3)
        result = context.build_working_context()
        assert "person-alice" in result
        assert "Key Entities" in result

    def test_insights_show_title_only(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "insight-test", "insight", "Test Insight Title", "Long content here " * 50)
        result = context.build_working_context()
        assert "insight-test" in result
        assert "Test Insight Title" in result
        assert "Active Insights" in result

    def test_archived_memories_excluded(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "goal-archived", "goal", "Old Goal", "Archived", importance=2.0, status="archived")
        result = context.build_working_context()
        assert result == ""

    def test_priority_ordering(self, test_db: Any) -> None:
        """High importance goals should appear before low importance entities."""
        conn = db._db()
        _insert_memory(conn, "goal-important", "goal", "Important Goal", "Critical", importance=2.0)
        _insert_memory(conn, "entity-low", "entity", "Low Entity", "Not important", importance=0.3)
        result = context.build_working_context()
        goal_pos = result.index("goal-important")
        entity_pos = result.index("entity-low")
        assert goal_pos < entity_pos

    def test_budget_limits_output(self, test_db: Any) -> None:
        """Very small budget should limit the number of memories."""
        conn = db._db()
        for i in range(20):
            _insert_memory(
                conn, f"entity-{i}", "entity", f"Entity {i}",
                f"Content for entity {i} " * 100, importance=1.0,
            )
        # Tiny budget — should include very few memories
        result = context.build_working_context(budget_tokens=200)
        # Count how many entity IDs appear
        count = sum(1 for i in range(20) if f"entity-{i}" in result)
        assert count < 20

    def test_stats_comment_included(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "goal-x", "goal", "Goal X", "Active goal", importance=1.5)
        result = context.build_working_context()
        assert "<!-- context:" in result
        assert "1 goals" in result

    def test_multiple_types_structured(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "goal-a", "goal", "Goal A", "Active", importance=1.5)
        _insert_memory(conn, "person-b", "entity", "Person B", "Developer", importance=1.3)
        _insert_memory(conn, "insight-c", "insight", "Insight C", "Pattern", importance=1.0)
        result = context.build_working_context()
        assert "Active Goals" in result
        assert "Key Entities" in result
        assert "Active Insights" in result


class TestBuildPreservationManifest:
    """Tests for build_preservation_manifest()."""

    def test_includes_goals(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "goal-preserve", "goal", "Must Preserve Goal", "Active", importance=1.5)
        result = context.build_preservation_manifest()
        assert "goal-preserve" in result
        assert "Must Preserve Goal" in result
        assert "ACTIVE GOALS" in result

    def test_includes_high_importance_entities(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "person-key", "entity", "Key Person", "Important", importance=1.5)
        _insert_memory(conn, "entity-low", "entity", "Low Entity", "Unimportant", importance=0.5)
        result = context.build_preservation_manifest()
        assert "person-key" in result
        assert "entity-low" not in result

    def test_includes_constitutional_invariants(self, test_db: Any) -> None:
        result = context.build_preservation_manifest()
        assert "You are Luke" in result
        assert "warm, unhurried, wry" in result

    def test_includes_preservation_rules(self, test_db: Any) -> None:
        result = context.build_preservation_manifest()
        assert "PRESERVATION RULES" in result
        assert "memory IDs" in result

    def test_empty_db_still_returns_manifest(self, test_db: Any) -> None:
        """Even with no memories, should still return constitutional invariants."""
        result = context.build_preservation_manifest()
        assert "CONSTITUTIONAL INVARIANTS" in result

    def test_recent_insights_included(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "insight-recent", "insight", "Recent Insight", "Fresh pattern")
        result = context.build_preservation_manifest()
        assert "insight-recent" in result
        assert "RECENT INSIGHTS" in result


class TestRecencyScore:
    """Tests for _recency_score helper."""

    def test_recent_scores_high(self) -> None:
        now = datetime.now(UTC).isoformat()
        score = context._recency_score(now)
        assert score > 0.9

    def test_old_scores_low(self) -> None:
        old = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        score = context._recency_score(old)
        assert score < 0.05

    def test_empty_string_returns_zero(self) -> None:
        assert context._recency_score("") == 0.0

    def test_invalid_iso_returns_zero(self) -> None:
        assert context._recency_score("not-a-date") == 0.0


class TestLoadPriorityMemories:
    """Tests for _load_priority_memories internal."""

    def test_goals_score_higher_than_episodes(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "goal-hi", "goal", "Goal", "Active goal", importance=1.0)
        _insert_memory(conn, "ep-lo", "episode", "Episode", "Something happened", importance=1.0)
        memories = context._load_priority_memories()
        goal_score = next(m["score"] for m in memories if m["id"] == "goal-hi")
        ep_score = next(m["score"] for m in memories if m["id"] == "ep-lo")
        assert goal_score > ep_score

    def test_high_importance_scores_higher(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "entity-hi", "entity", "High", "Important", importance=2.0)
        _insert_memory(conn, "entity-lo", "entity", "Low", "Unimportant", importance=0.3)
        memories = context._load_priority_memories()
        hi_score = next(m["score"] for m in memories if m["id"] == "entity-hi")
        lo_score = next(m["score"] for m in memories if m["id"] == "entity-lo")
        assert hi_score > lo_score
