"""Tests for luke.context — working memory injection and preservation manifests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Constitutional layer tests
# ---------------------------------------------------------------------------

_SAMPLE_CONSTITUTIONAL = {
    "identity": {
        "name": "Luke",
        "self_description": "A close friend who builds anything.",
        "never_self_identify_as": ["AI", "assistant"],
    },
    "values": ["authenticity over impressiveness", "depth over speed"],
    "communication": {
        "voice": "warm, unhurried, sometimes wry",
        "format": "HTML only (Telegram)",
        "address": "always 'you', never 'the user'",
        "length": "one message, not three",
        "endings": "stop when done",
        "never_say": ["Great question!", "Absolutely!"],
        "patterns": {
            "uncertainty": "'I think so, let me check'",
            "greetings": "use memory",
        },
    },
    "hard_rules": [
        "Don't say 'I'll remember that' without calling remember",
        "Don't ask 'would you like me to...' — just do it",
    ],
    "decision_heuristics": {
        "autonomy": {
            "borderline": "do the work, show the result, ask before the final action"
        }
    },
}


class TestLoadConstitutional:
    """Tests for load_constitutional()."""

    def test_loads_yaml_file(self, tmp_settings: Any) -> None:
        """Loads constitutional.yaml from luke_dir."""
        import yaml as _yaml

        yaml_path = tmp_settings.luke_dir / "constitutional.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(_yaml.dump(_SAMPLE_CONSTITUTIONAL))
        context._constitutional_cache = None  # clear cache
        data = context.load_constitutional(force_reload=True)
        assert data["identity"]["name"] == "Luke"
        assert "AI" in data["identity"]["never_self_identify_as"]

    def test_returns_empty_dict_when_missing(self, tmp_settings: Any) -> None:
        """Returns {} when file doesn't exist."""
        tmp_settings.luke_dir.mkdir(parents=True, exist_ok=True)
        context._constitutional_cache = None
        data = context.load_constitutional(force_reload=True)
        assert data == {}

    def test_caches_result(self, tmp_settings: Any) -> None:
        """Second call returns cached result without re-reading."""
        import yaml as _yaml

        yaml_path = tmp_settings.luke_dir / "constitutional.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(_yaml.dump({"identity": {"name": "TestBot"}}))
        context._constitutional_cache = None
        first = context.load_constitutional()
        # Delete the file — cached value should persist
        yaml_path.unlink()
        second = context.load_constitutional()
        assert first is second
        assert first["identity"]["name"] == "TestBot"

    def test_force_reload_bypasses_cache(self, tmp_settings: Any) -> None:
        import yaml as _yaml

        yaml_path = tmp_settings.luke_dir / "constitutional.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(_yaml.dump({"identity": {"name": "V1"}}))
        context._constitutional_cache = None
        context.load_constitutional()
        yaml_path.write_text(_yaml.dump({"identity": {"name": "V2"}}))
        data = context.load_constitutional(force_reload=True)
        assert data["identity"]["name"] == "V2"


class TestFormatConstitutionalSummary:
    """Tests for format_constitutional_summary()."""

    def test_full_summary_from_data(self) -> None:
        result = context.format_constitutional_summary(_SAMPLE_CONSTITUTIONAL)
        assert "CONSTITUTIONAL INVARIANTS" in result
        assert "You are Luke" in result
        assert "warm, unhurried, sometimes wry" in result
        assert "HTML only (Telegram)" in result
        assert "AI, assistant" in result
        assert "authenticity over impressiveness" in result
        assert "Don't say 'I'll remember that'" in result
        assert "borderline" in result.lower()

    def test_empty_data_returns_fallback(self) -> None:
        result = context.format_constitutional_summary({})
        assert "CONSTITUTIONAL INVARIANTS" in result
        assert "Luke" in result

    def test_partial_data_handles_missing_keys(self) -> None:
        """Only identity section — other sections should be absent, no crash."""
        partial = {"identity": {"name": "TestBot"}}
        result = context.format_constitutional_summary(partial)
        assert "You are TestBot" in result
        assert "warm" not in result  # no communication section

    def test_never_say_capped_at_five(self) -> None:
        data = {
            "communication": {
                "never_say": [f"phrase-{i}" for i in range(10)],
            }
        }
        result = context.format_constitutional_summary(data)
        # Should include at most 5 phrases
        assert "phrase-0" in result
        assert "phrase-4" in result
        assert "phrase-5" not in result

    def test_preservation_manifest_uses_constitutional_yaml(self, tmp_settings: Any, test_db: Any) -> None:
        """build_preservation_manifest() should include dynamically loaded invariants."""
        import yaml as _yaml

        yaml_path = tmp_settings.luke_dir / "constitutional.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(_yaml.dump(_SAMPLE_CONSTITUTIONAL))
        context._constitutional_cache = None
        result = context.build_preservation_manifest()
        # Should contain YAML-derived content, not just hardcoded strings
        assert "warm, unhurried, sometimes wry" in result
        assert "authenticity over impressiveness" in result
        assert "Don't ask 'would you like me to...' — just do it" in result


# ---------------------------------------------------------------------------
# Compression audit tests
# ---------------------------------------------------------------------------


class TestAuditCompression:
    """Tests for audit_compression()."""

    def test_perfect_retention(self, test_db: Any) -> None:
        """All expected references present → retention_score ~1.0."""
        text = "[goal-ship-v2] ship version 2. [person-alice] Alice is key. Luke says hi."
        result = context.audit_compression(
            compressed_text=text,
            goal_ids=["goal-ship-v2"],
            entity_ids=["person-alice"],
            memory_ids=["goal-ship-v2", "person-alice"],
            persist=False,
        )
        assert result["goals_preserved"] == 1
        assert result["entities_preserved"] == 1
        assert result["memory_ids_preserved"] == 2
        assert result["identity_anchor"] is True
        assert result["retention_score"] >= 0.9

    def test_partial_retention(self, test_db: Any) -> None:
        """Some references missing → lower retention score."""
        text = "[goal-ship-v2] ship version 2. Luke is here."
        result = context.audit_compression(
            compressed_text=text,
            goal_ids=["goal-ship-v2", "goal-learn-spanish"],
            entity_ids=["person-alice", "person-bob"],
            persist=False,
        )
        assert result["goals_preserved"] == 1
        assert result["goals_missing"] == ["goal-learn-spanish"]
        assert result["entities_preserved"] == 0
        assert result["entities_missing"] == ["person-alice", "person-bob"]
        assert result["retention_score"] < 0.8

    def test_missing_identity_anchor(self, test_db: Any) -> None:
        """No identity name in text → identity_anchor is False."""
        text = "Some compressed summary without the name."
        result = context.audit_compression(
            compressed_text=text,
            persist=False,
        )
        assert result["identity_anchor"] is False

    def test_empty_expectations_with_identity(self, test_db: Any) -> None:
        """No goals/entities expected, just identity check."""
        text = "Luke is doing stuff."
        result = context.audit_compression(
            compressed_text=text,
            persist=False,
        )
        assert result["retention_score"] == 1.0
        assert result["identity_anchor"] is True

    def test_persists_to_db(self, test_db: Any) -> None:
        """When persist=True, writes to compression_audit table."""
        conn = db._db()
        text = "[goal-x] goal preserved. Luke."
        context.audit_compression(
            compressed_text=text,
            goal_ids=["goal-x"],
            entity_ids=["person-missing"],
            messages_compressed=10,
            messages_kept=5,
            persist=True,
        )
        row = conn.execute("SELECT * FROM compression_audit ORDER BY id DESC LIMIT 1").fetchone()
        assert row is not None
        assert row["messages_compressed"] == 10
        assert row["messages_kept"] == 5
        assert row["goals_expected"] == 1
        assert row["goals_preserved"] == 1
        assert row["entities_expected"] == 1
        assert row["entities_preserved"] == 0
        assert row["identity_anchor"] == 1
        assert 0 < row["retention_score"] < 1.0

    def test_case_insensitive_matching(self, test_db: Any) -> None:
        """ID matching should be case-insensitive."""
        text = "The GOAL-SHIP-V2 is progressing. PERSON-ALICE helped. luke approves."
        result = context.audit_compression(
            compressed_text=text,
            goal_ids=["goal-ship-v2"],
            entity_ids=["person-alice"],
            persist=False,
        )
        assert result["goals_preserved"] == 1
        assert result["entities_preserved"] == 1
        assert result["identity_anchor"] is True

    def test_summary_tokens_counted(self, test_db: Any) -> None:
        """summary_tokens should be a positive integer for non-empty text."""
        text = "Some compressed text about Luke and goals."
        result = context.audit_compression(compressed_text=text, persist=False)
        assert result["summary_tokens"] > 0
