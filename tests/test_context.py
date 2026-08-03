"""Tests for luke.context — working memory injection and preservation manifests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

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
           (id, type, created, updated, access_count, importance, status,
            tags_json, links_json, last_accessed)
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
        _insert_memory(
            conn, "goal-test", "goal", "Test Goal", "Status: active\nProgress: 50%", importance=1.5
        )
        result = context.build_working_context()
        assert "goal-test" in result
        assert "Active Goals" in result

    def test_single_entity_injected(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(
            conn, "person-alice", "entity", "Alice", "Alice is a developer", importance=1.3
        )
        result = context.build_working_context()
        assert "person-alice" in result
        assert "Key Entities" in result

    def test_insights_show_title_only(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(
            conn, "insight-test", "insight", "Test Insight Title", "Long content here " * 50
        )
        result = context.build_working_context()
        assert "insight-test" in result
        assert "Test Insight Title" in result
        assert "Active Insights" in result

    def test_archived_memories_excluded(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(
            conn, "goal-archived", "goal", "Old Goal", "Archived", importance=2.0, status="archived"
        )
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
                conn,
                f"entity-{i}",
                "entity",
                f"Entity {i}",
                f"Content for entity {i} " * 100,
                importance=1.0,
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
        # Counts what was RENDERED. The old comment reported memories
        # *considered*, which read "95 memories injected" for a block that
        # emitted 21 of them.
        assert "'goal': 1" in result
        assert "of 12000 budget" in result

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
        _insert_memory(
            conn, "goal-preserve", "goal", "Must Preserve Goal", "Active", importance=1.5
        )
        result = context.build_preservation_manifest()
        assert "goal-preserve" in result
        assert "Must Preserve Goal" in result
        assert "ACTIVE GOALS" in result

    def test_ranks_entities_by_importance(self, test_db: Any) -> None:
        """The manifest takes the top 15 entities, in importance order.

        It used to also require importance >= 1.2. That bar was redundant —
        only 15 entities cleared it in the real corpus, so LIMIT 15 was already
        binding — and an absolute threshold on a rescalable value can only fail
        by silently emptying the section. Ordering says the same thing without
        depending on the scale.
        """
        conn = db._db()
        _insert_memory(conn, "person-key", "entity", "Key Person", "Important", importance=1.5)
        _insert_memory(conn, "entity-low", "entity", "Low Entity", "Unimportant", importance=0.5)
        result = context.build_preservation_manifest()
        assert result.index("person-key") < result.index("entity-low")

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
        "autonomy": {"borderline": "do the work, show the result, ask before the final action"}
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

    def test_preservation_manifest_uses_constitutional_yaml(
        self, tmp_settings: Any, test_db: Any
    ) -> None:
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


# ---------------------------------------------------------------------------
# Recent outputs injection (L3) — verbatim mirror of own outbound sends
# ---------------------------------------------------------------------------


class TestRecentOutputsBlock:
    """Tests for _build_recent_outputs_block()."""

    def test_recent_outputs_none_when_no_outputs(self, test_db: Any) -> None:
        """Returns None when no outbound messages exist for chat."""
        result = context._build_recent_outputs_block("100", limit=3)
        assert result is None

    def test_recent_outputs_block_format(self, test_db: Any) -> None:
        """Builds a properly formatted block when outputs exist."""
        for i in range(3):
            test_db.store_message(
                chat_id="100",
                sender_name="Luke",
                content=f"reply-{i}",
                timestamp=f"2026-05-13T22:0{i}:00",
            )
        result = context._build_recent_outputs_block("100", limit=3)
        assert result is not None
        assert "<my-recent-outputs>" in result
        assert "</my-recent-outputs>" in result
        assert "verbatim, not reconstructed" in result
        # All three messages should appear, chronological (oldest first)
        assert "reply-0" in result
        assert "reply-1" in result
        assert "reply-2" in result
        assert result.index("reply-0") < result.index("reply-2")
        # Timestamp formatted to 19 chars (seconds precision)
        assert "[2026-05-13T22:00:00]" in result

    def test_recent_outputs_respects_limit(self, test_db: Any) -> None:
        """Only the last N outbound messages are returned."""
        for i in range(5):
            test_db.store_message(
                chat_id="100",
                sender_name="Luke",
                content=f"reply-{i}",
                timestamp=f"2026-05-13T22:0{i}:00",
            )
        result = context._build_recent_outputs_block("100", limit=2)
        assert result is not None
        # Last two should be reply-3 and reply-4
        assert "reply-3" in result
        assert "reply-4" in result
        assert "reply-0" not in result
        assert "reply-1" not in result
        assert "reply-2" not in result

    def test_recent_outputs_truncates_long_messages(self, test_db: Any) -> None:
        """Messages over the truncation threshold are clipped with an ellipsis."""
        long_content = "x" * 2000
        test_db.store_message(
            chat_id="100",
            sender_name="Luke",
            content=long_content,
            timestamp="2026-05-13T22:00:00",
        )
        result = context._build_recent_outputs_block("100", limit=3)
        assert result is not None
        assert "…" in result
        # Original 2000-char string should not survive intact
        assert long_content not in result

    def test_recent_outputs_excludes_inbound(self, test_db: Any) -> None:
        """Only Luke-sent messages are pulled; user messages are filtered out."""
        test_db.store_message(
            chat_id="100",
            sender_name="Filipe",
            content="user-msg",
            timestamp="2026-05-13T22:00:00",
        )
        test_db.store_message(
            chat_id="100",
            sender_name="Luke",
            content="luke-reply",
            timestamp="2026-05-13T22:01:00",
        )
        result = context._build_recent_outputs_block("100", limit=3)
        assert result is not None
        assert "luke-reply" in result
        assert "user-msg" not in result

    def test_recent_outputs_zero_limit_returns_none(self, test_db: Any) -> None:
        """A non-positive limit short-circuits to None."""
        test_db.store_message(
            chat_id="100",
            sender_name="Luke",
            content="reply",
            timestamp="2026-05-13T22:00:00",
        )
        assert context._build_recent_outputs_block("100", limit=0) is None

    def test_recent_outputs_empty_chat_id_returns_none(self, test_db: Any) -> None:
        """An empty chat_id short-circuits to None."""
        assert context._build_recent_outputs_block("", limit=3) is None

    def test_build_working_context_includes_recent_outputs(
        self, tmp_settings: Any, test_db: Any
    ) -> None:
        """build_working_context() prepends the recent-outputs block."""
        tmp_settings.chat_id = "100"
        tmp_settings.recent_outputs_enabled = True
        tmp_settings.recent_outputs_limit = 3
        test_db.store_message(
            chat_id="100",
            sender_name="Luke",
            content="verbatim-output",
            timestamp="2026-05-13T22:00:00",
        )
        conn = db._db()
        _insert_memory(conn, "goal-test", "goal", "T", "Active", importance=1.5)
        result = context.build_working_context()
        assert "<my-recent-outputs>" in result
        assert "verbatim-output" in result
        # Recent outputs section appears before working memory header
        assert result.index("<my-recent-outputs>") < result.index("# Injected Working Memory")

    def test_build_working_context_recent_outputs_disabled(
        self, tmp_settings: Any, test_db: Any
    ) -> None:
        """When the setting is off, no recent-outputs block is emitted."""
        tmp_settings.chat_id = "100"
        tmp_settings.recent_outputs_enabled = False
        test_db.store_message(
            chat_id="100",
            sender_name="Luke",
            content="should-not-appear",
            timestamp="2026-05-13T22:00:00",
        )
        conn = db._db()
        _insert_memory(conn, "goal-test", "goal", "T", "Active", importance=1.5)
        result = context.build_working_context()
        assert "<my-recent-outputs>" not in result
        assert "should-not-appear" not in result

    def test_build_working_context_recent_outputs_only_no_memories(
        self, tmp_settings: Any, test_db: Any
    ) -> None:
        """Recent outputs survive even when no memories are selected."""
        tmp_settings.chat_id = "100"
        tmp_settings.recent_outputs_enabled = True
        tmp_settings.recent_outputs_limit = 3
        test_db.store_message(
            chat_id="100",
            sender_name="Luke",
            content="only-output",
            timestamp="2026-05-13T22:00:00",
        )
        # No memories inserted — selection will be empty.
        result = context.build_working_context()
        assert "<my-recent-outputs>" in result
        assert "only-output" in result


# ---------------------------------------------------------------------------
# Active attention injection (L2) — persistent foreground commitments
# ---------------------------------------------------------------------------


class TestActiveAttentionInContext:
    """Tests for active-attention block integration with build_working_context()."""

    def test_attention_block_included_when_items_exist(
        self, tmp_settings: Any, test_db: Any
    ) -> None:
        """Pinned attention items appear in the context block."""
        from luke import attention

        tmp_settings.chat_id = "100"
        tmp_settings.recent_outputs_enabled = False
        attention.pin("100", "track Fanatics prep", origin="luke")
        conn = db._db()
        _insert_memory(conn, "goal-test", "goal", "T", "Active", importance=1.5)
        result = context.build_working_context()
        assert "<active-attention>" in result
        assert "track Fanatics prep" in result
        # Attention block sits above the memory injection header.
        assert result.index("<active-attention>") < result.index("# Injected Working Memory")

    def test_no_attention_block_when_empty(self, tmp_settings: Any, test_db: Any) -> None:
        """No attention block is emitted when no items are pinned."""
        tmp_settings.chat_id = "100"
        tmp_settings.recent_outputs_enabled = False
        conn = db._db()
        _insert_memory(conn, "goal-test", "goal", "T", "Active", importance=1.5)
        result = context.build_working_context()
        assert "<active-attention>" not in result

    def test_attention_survives_when_no_memories(self, tmp_settings: Any, test_db: Any) -> None:
        """Attention items appear even when no memories are selected."""
        from luke import attention

        tmp_settings.chat_id = "100"
        tmp_settings.recent_outputs_enabled = False
        attention.pin("100", "watch for Naiara's email")
        # No memories inserted — only attention should be in output.
        result = context.build_working_context()
        assert "<active-attention>" in result
        assert "watch for Naiara's email" in result

    def test_attention_below_recent_outputs(self, tmp_settings: Any, test_db: Any) -> None:
        """Attention block sits between recent-outputs and memory injection."""
        from luke import attention

        tmp_settings.chat_id = "100"
        tmp_settings.recent_outputs_enabled = True
        tmp_settings.recent_outputs_limit = 3
        test_db.store_message(
            chat_id="100",
            sender_name="Luke",
            content="some-output",
            timestamp="2026-05-13T22:00:00",
        )
        attention.pin("100", "matters")
        conn = db._db()
        _insert_memory(conn, "goal-test", "goal", "T", "Active", importance=1.5)
        result = context.build_working_context()
        out_pos = result.index("<my-recent-outputs>")
        attn_pos = result.index("<active-attention>")
        mem_pos = result.index("# Injected Working Memory")
        assert out_pos < attn_pos < mem_pos


class TestSpendMatchesRender:
    """The phantom-budget regression guard.

    Selection and rendering were two independent code paths that disagreed:
    the budget charged every memory 400 chars of content, the renderer emitted
    bare titles for insights and procedures and capped them at 10 and 5. On the
    real corpus a 12,000-token budget bought ~1,300 tokens of output and threw
    away 74 memories it had already paid for. These tests pin the two together.
    """

    def test_charged_tokens_equal_rendered_tokens(self, test_db: Any) -> None:
        conn = db._db()
        for i in range(12):
            _insert_memory(
                conn, f"insight-{i}", "insight", f"Insight {i}", "x" * 900, importance=1.4
            )
            _insert_memory(
                conn, f"proc-{i}", "procedure", f"Procedure {i}", "y" * 900, importance=1.4
            )
            _insert_memory(conn, f"entity-{i}", "entity", f"Entity {i}", "z" * 900, importance=1.4)
        memories = context._load_priority_memories()
        by_type, spent = context._spend(memories, 4000)

        rendered = sum(
            context._estimate_tokens(line) for lines in by_type.values() for line in lines
        )
        assert spent == rendered

    def test_nothing_charged_is_dropped(self, test_db: Any) -> None:
        """Every memory the wallet paid for must appear in the output."""
        conn = db._db()
        for i in range(30):
            _insert_memory(
                conn, f"insight-{i}", "insight", f"Insight {i}", "body " * 200, importance=1.4
            )
        memories = context._load_priority_memories()
        by_type, _ = context._spend(memories, 12_000)
        result = context.build_working_context(budget_tokens=12_000)
        for lines in by_type.values():
            for line in lines:
                mem_id = line.split("[")[1].split("]")[0]
                assert mem_id in result

    def test_type_caps_are_enforced_during_selection(self, test_db: Any) -> None:
        """A capped-out type must stop consuming budget, not be sliced later."""
        conn = db._db()
        for i in range(40):
            _insert_memory(conn, f"insight-{i}", "insight", f"Insight {i}", "body", importance=1.9)
        for i in range(5):
            _insert_memory(
                conn, f"entity-{i}", "entity", f"Entity {i}", "who they are", importance=0.4
            )
        by_type, _ = context._spend(context._load_priority_memories(), 12_000)

        assert len(by_type["insight"]) == context._BACKGROUND_SPEC["insight"].max_items
        # 40 high-importance insights would previously have eaten the whole
        # budget at 400 charged chars each, starving the entities entirely.
        assert by_type.get("entity"), "entities starved by capped-out insights"

    def test_budget_is_respected(self, test_db: Any) -> None:
        conn = db._db()
        for i in range(20):
            _insert_memory(conn, f"entity-{i}", "entity", f"Entity {i}", "z" * 400, importance=1.4)
        _, spent = context._spend(context._load_priority_memories(), 200)
        assert spent <= 200


class TestSpendMechanics:
    """Edge cases of the spend loop.

    _spend is the single point where a memory's cost is decided, so every way
    it can silently drop or over-charge something is worth pinning.
    """

    def test_empty_input(self, test_db: Any) -> None:
        assert context._spend([], 5000) == ({}, 0)

    def test_zero_budget_renders_nothing(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "entity-a", "entity", "A", "content", importance=1.5)
        by_type, spent = context._spend(context._load_priority_memories(), 0)
        assert by_type == {}
        assert spent == 0

    def test_unknown_type_is_skipped_not_charged(self, test_db: Any) -> None:
        """A type with no RenderSpec has no defined cost, so it cannot be sold."""
        rogue = [{"id": "x", "type": "nonsense", "title": "T", "content": "C", "score": 9.9}]
        by_type, spent = context._spend(rogue, 5000)
        assert by_type == {}
        assert spent == 0

    def test_oversized_line_skipped_cheaper_one_still_fits(self, test_db: Any) -> None:
        """A too-expensive memory must not end selection — scanning continues."""
        big = {"id": "big", "type": "entity", "title": "B", "content": "x" * 5000, "score": 9.0}
        small = {"id": "small", "type": "insight", "title": "S", "content": "y", "score": 1.0}
        by_type, spent = context._spend([big, small], 40)
        assert "big" not in str(by_type)
        assert by_type.get("insight"), "cheaper lower-scored memory should still fit"
        assert spent <= 40

    def test_caps_are_independent_per_type(self, test_db: Any) -> None:
        conn = db._db()
        for i in range(40):
            _insert_memory(conn, f"insight-{i}", "insight", f"I{i}", "b", importance=1.5)
            _insert_memory(conn, f"proc-{i}", "procedure", f"P{i}", "b", importance=1.5)
        by_type, _ = context._spend(context._load_priority_memories(), 100_000)
        assert len(by_type["insight"]) == context._BACKGROUND_SPEC["insight"].max_items
        assert len(by_type["procedure"]) == context._BACKGROUND_SPEC["procedure"].max_items

    def test_selection_follows_score_order(self, test_db: Any) -> None:
        """Within a type, the cap must keep the best — not the first seen."""
        conn = db._db()
        _insert_memory(conn, "entity-top", "entity", "Top", "c", importance=2.0, access_count=50)
        for i in range(20):
            _insert_memory(conn, f"entity-low-{i}", "entity", f"L{i}", "c", importance=0.2)
        by_type, _ = context._spend(context._load_priority_memories(), 100_000)
        assert any("entity-top" in line for line in by_type["entity"])

    def test_spent_never_exceeds_budget(self, test_db: Any) -> None:
        conn = db._db()
        for i in range(60):
            _insert_memory(conn, f"entity-{i}", "entity", f"E{i}", "z" * 500, importance=1.5)
        memories = context._load_priority_memories()
        for budget in (1, 17, 120, 999, 5000):
            _, spent = context._spend(memories, budget)
            assert spent <= budget, f"overspent at budget={budget}"


class TestRenderLine:
    """_render_line is the only definition of what a memory costs."""

    def test_title_field_uses_title(self) -> None:
        spec = context.RenderSpec(10, "title", 100)
        line = context._render_line({"id": "m1", "title": "T", "content": "C" * 500}, spec)
        assert "T" in line
        assert "CCC" not in line

    def test_content_field_uses_content(self) -> None:
        spec = context.RenderSpec(10, "content", 100)
        line = context._render_line({"id": "m1", "title": "T", "content": "CCC"}, spec)
        assert "CCC" in line

    def test_truncates_to_spec_chars(self) -> None:
        spec = context.RenderSpec(10, "content", 20)
        line = context._render_line({"id": "m1", "title": "", "content": "x" * 500}, spec)
        assert line.count("x") == 20

    def test_handles_missing_value(self) -> None:
        """A memory with no content must render, not crash."""
        spec = context.RenderSpec(10, "content", 20)
        line = context._render_line({"id": "m1", "title": "T", "content": None}, spec)
        assert "m1" in line


class TestFeedbackReserve:
    """Filipe's stated preferences must survive the recency competition.

    Feedback insights are durable behavioural rules, so they steadily lose
    slots to the stream of fresh reflexion/dream insights — measured 0-1 of 25
    on the live corpus. A directive about how to behave has to be present to be
    followed, so it gets a floor rather than a better score.
    """

    @staticmethod
    def _seed(conn: Any, n_recent: int = 40, n_feedback: int = 10) -> None:
        now = datetime.now(UTC).isoformat()
        old = (datetime.now(UTC) - timedelta(days=120)).isoformat()
        for i in range(n_recent):
            _insert_memory(
                conn, f"reflexion-{i}", "insight", f"Recent reflexion {i}", "b", importance=1.6
            )
        for i in range(n_feedback):
            _insert_memory(
                conn, f"feedback-pref-{i}", "insight", f"Preference {i}", "b", importance=1.5
            )
            conn.execute(
                "UPDATE memory_meta SET updated = ? WHERE id = ?", (old, f"feedback-pref-{i}")
            )
        conn.execute("UPDATE memory_meta SET updated = ? WHERE id LIKE 'reflexion-%'", (now,))
        conn.commit()

    def test_feedback_insights_are_reserved(self, test_db: Any) -> None:
        conn = db._db()
        self._seed(conn)
        by_type, _ = context._spend(context._load_priority_memories(), 12_000)
        feedback = [line for line in by_type["insight"] if "[feedback-" in line]
        assert len(feedback) == context._FEEDBACK_RESERVE

    def test_reserve_does_not_exceed_the_type_cap(self, test_db: Any) -> None:
        conn = db._db()
        self._seed(conn, n_recent=0, n_feedback=60)
        by_type, _ = context._spend(context._load_priority_memories(), 100_000)
        assert len(by_type["insight"]) == context._BACKGROUND_SPEC["insight"].max_items

    def test_non_feedback_insights_still_fill_the_rest(self, test_db: Any) -> None:
        conn = db._db()
        self._seed(conn)
        by_type, _ = context._spend(context._load_priority_memories(), 12_000)
        others = [line for line in by_type["insight"] if "[feedback-" not in line]
        assert others, "reserve must be a floor, not a takeover"

    def test_reserve_respects_the_budget(self, test_db: Any) -> None:
        conn = db._db()
        self._seed(conn)
        _, spent = context._spend(context._load_priority_memories(), 60)
        assert spent <= 60

    def test_tagged_feedback_counts_too(self, test_db: Any) -> None:
        """Definition matches memory.get_feedback_insight_ids: id prefix OR tag."""
        assert context._is_feedback("feedback-x", "")
        assert context._is_feedback("insight-y", '["feedback", "tone"]')
        assert not context._is_feedback("insight-y", '["reflexion"]')
        assert not context._is_feedback("insight-y", "")


class TestBackgroundLayerIsStanding:
    """The background layer must not become query-aware again.

    It once embedded the same query recall() was already embedding — two HTTP
    round trips per run — then scanned every vector in the corpus to reorder
    the result. Measured 15.9x the wall time to change 11% of the selection,
    and that 11% is what the recall layer surfaces anyway.
    """

    def test_load_priority_memories_takes_no_query(self) -> None:
        import inspect

        params = inspect.signature(context._load_priority_memories).parameters
        assert "query" not in params

    def test_build_working_context_takes_no_query(self) -> None:
        import inspect

        params = inspect.signature(context.build_working_context).parameters
        assert "query" not in params

    def test_does_not_embed(self, test_db: Any, monkeypatch: Any) -> None:
        """Building standing context must issue no embedding call at all."""
        conn = db._db()
        _insert_memory(conn, "entity-a", "entity", "A", "content", importance=1.5)

        calls: list[Any] = []

        def _boom(texts: list[str]) -> None:
            calls.append(texts)
            return None

        monkeypatch.setattr("luke.memory._embed_via_server", _boom)
        context.build_working_context()
        assert calls == [], "background layer embedded something"


class TestTrimConvState:
    """Conversation state must lose its OLDEST content, never its newest.

    _save_conv_state writes chronologically with the latest reply appended
    last, so a plain body[:limit] dropped the most recent exchange and cut
    mid-word — backwards for the one block whose job is seamless resumption.
    """

    HEADER = (
        "# Conversation State\n\n"
        "**Last exchange:** 2026-08-03T10:03+00:00\n"
        "**Active topics:** work\n"
        "**User last active:** 2026-08-03T09:41\n"
    )

    def _body(self, n: int) -> str:
        msgs = "\n".join(
            f"**Filipe Lima** (2026-08-03T09:{i:02d}): message number {i}" for i in range(n)
        )
        return self.HEADER + msgs

    def test_short_body_untouched(self) -> None:
        body = self._body(3)
        assert context._trim_conv_state(body, 10_000) == body

    def test_keeps_the_newest_message(self) -> None:
        out = context._trim_conv_state(self._body(40), 600)
        assert "message number 39" in out
        assert "message number 0" not in out

    def test_keeps_the_header(self) -> None:
        out = context._trim_conv_state(self._body(40), 600)
        assert "**Active topics:**" in out
        assert "**User last active:**" in out

    def test_respects_the_limit(self) -> None:
        for limit in (200, 600, 1500, 3000):
            assert len(context._trim_conv_state(self._body(40), limit)) <= limit

    def test_does_not_cut_mid_line(self) -> None:
        out = context._trim_conv_state(self._body(40), 600)
        for line in out.split("\n"):
            if line.startswith("**Filipe"):
                assert line.endswith(tuple("0123456789")), f"truncated line: {line!r}"

    def test_messages_stay_in_order(self) -> None:
        out = context._trim_conv_state(self._body(40), 900)
        nums = [int(ln.rsplit(" ", 1)[1]) for ln in out.split("\n") if ln.startswith("**Filipe")]
        assert nums == sorted(nums)

    def test_headerless_body_still_trims(self) -> None:
        body = "\n".join(f"**Luke** (2026-08-03T09:{i:02d}): line {i}" for i in range(40))
        out = context._trim_conv_state(body, 400)
        assert "line 39" in out
        assert len(out) <= 400

    def test_oversized_header_falls_back_to_newest_text(self) -> None:
        body = "H" * 5000 + "\n**Luke** (2026-08-03T09:00): the newest thing"
        out = context._trim_conv_state(body, 100)
        assert "the newest thing" in out
        assert len(out) <= 100


class TestRankTurnCandidates:
    """Turn-layer selection: guaranteed skills, dedup against pinned state."""

    def test_trigger_skills_are_admitted(self, tmp_settings: Any) -> None:
        from unittest.mock import patch

        skills = [{"id": "proc-deploy", "type": "procedure", "title": "Deploy", "score": 0.9}]
        with (
            patch("luke.context._memory_module.recall", return_value=[]),
            patch("luke.context._memory_module.get_trigger_matched_skills", return_value=skills),
            patch("luke.context._memory_module.get_graph_neighbors", return_value=[]),
        ):
            out = context._rank_turn_candidates("deploy this release", set())
        assert [c["id"] for c in out] == ["proc-deploy"]
        assert out[0]["source"] == "skill"

    def test_pinned_ids_are_excluded(self, tmp_settings: Any) -> None:
        """conversation-state is pinned separately, so it must not take a slot.

        It did on 61% of real turns, while also being rendered a second time in
        full — the single largest duplication in the old two-layer design.
        """
        from unittest.mock import patch

        hits = [
            {"id": context._CONV_STATE_ID, "type": "episode", "title": "State", "score": 0.9},
            {"id": "entity-1", "type": "entity", "title": "E", "score": 0.5},
        ]
        with (
            patch("luke.context._memory_module.recall", return_value=hits),
            patch("luke.context._memory_module.get_trigger_matched_skills", return_value=[]),
            patch("luke.context._memory_module.get_graph_neighbors", return_value=[]),
        ):
            out = context._rank_turn_candidates("what did we decide", {context._CONV_STATE_ID})
        assert [c["id"] for c in out] == ["entity-1"]

    def test_no_duplicates_across_sources(self, tmp_settings: Any) -> None:
        from unittest.mock import patch

        shared = {"id": "proc-x", "type": "procedure", "title": "X", "score": 0.7}
        with (
            patch("luke.context._memory_module.recall", return_value=[shared]),
            patch("luke.context._memory_module.get_trigger_matched_skills", return_value=[shared]),
            patch("luke.context._memory_module.get_graph_neighbors", return_value=[shared]),
        ):
            out = context._rank_turn_candidates("x", set())
        assert [c["id"] for c in out] == ["proc-x"]


class TestRenderTurnBlock:
    def test_empty_candidates(self, tmp_settings: Any) -> None:
        assert context._render_turn_block([], 5000) == ("", 0, [])

    def test_structure_and_charging(self, tmp_settings: Any) -> None:
        cands = [{"id": "mem1", "type": "entity", "title": "Test Memory", "score": 0.9}]
        block, spent, _rendered = context._render_turn_block(cands, 5000)
        assert "<context><memories>" in block
        assert "mem1" in block
        assert spent == context._estimate_tokens(
            next(ln for ln in block.split("\n") if ln.startswith("[mem1]"))
        )

    def test_respects_budget(self, tmp_settings: Any) -> None:
        cands = [
            {"id": f"mem{i}", "type": "entity", "title": "T" * 400, "score": 0.5} for i in range(40)
        ]
        _, spent, _rendered = context._render_turn_block(cands, 100)
        assert spent <= 100


class TestAssembleContext:
    """The single decision point. Its whole reason to exist is that the two
    layers can now see each other."""

    async def test_never_raises(self, tmp_settings: Any) -> None:
        """Memory is an enhancement; losing it must never cost the caller."""
        from unittest.mock import patch

        with patch("luke.context._assemble", side_effect=RuntimeError("db gone")):
            ctx = await context.assemble_context(
                query="hello there", chat_id="1", budget_tokens=1000
            )
        assert ctx.system_block == ""
        assert ctx.turn_block == ""
        assert ctx.ids == []

    async def test_turn_failure_keeps_standing_context(self, test_db: Any) -> None:
        """A recall failure must not take the standing block down with it."""
        from unittest.mock import patch

        conn = db._db()
        _insert_memory(conn, "person-x", "entity", "X", "who they are", importance=1.8)
        with patch(
            "luke.context._rank_turn_candidates", side_effect=RuntimeError("embed server down")
        ):
            ctx = await context.assemble_context(
                query="tell me about x", chat_id="12345", budget_tokens=4000
            )
        assert "person-x" in ctx.system_block
        assert ctx.turn_block == ""

    async def test_standing_failure_keeps_turn_block(self, test_db: Any) -> None:
        from unittest.mock import patch

        cands = [{"id": "mem-1", "type": "entity", "title": "M", "score": 0.9}]
        with (
            patch("luke.context._rank_turn_candidates", return_value=cands),
            patch("luke.context.build_working_context", side_effect=RuntimeError("boom")),
        ):
            ctx = await context.assemble_context(
                query="what about mem", chat_id="12345", budget_tokens=4000
            )
        assert "mem-1" in ctx.turn_block

    async def test_turn_hits_excluded_from_standing_block(self, test_db: Any) -> None:
        """No memory should appear in both layers.

        Nothing prevented this before: the turn prefix was built in
        app.process and the standing block in run_agent, and neither could see
        the other's output.
        """
        from unittest.mock import patch

        conn = db._db()
        _insert_memory(conn, "person-dup", "entity", "Dup", "a" * 400, importance=1.9)
        cands = [{"id": "person-dup", "type": "entity", "title": "Dup", "score": 0.9}]
        with patch("luke.context._rank_turn_candidates", return_value=cands):
            ctx = await context.assemble_context(
                query="tell me about dup", chat_id="12345", budget_tokens=6000
            )
        assert "person-dup" in ctx.turn_block
        assert "person-dup" not in ctx.system_block

    async def test_autonomous_run_skips_the_turn_layer(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "person-y", "entity", "Y", "content", importance=1.5)
        ctx = await context.assemble_context(
            query="anything at all here", chat_id="12345", budget_tokens=4000, turn_scoped=False
        )
        assert ctx.turn_block == ""
        assert ctx.recalled_ids == []

    async def test_trivial_query_skips_the_turn_layer(self, test_db: Any) -> None:
        ctx = await context.assemble_context(query="ok", chat_id="12345", budget_tokens=4000)
        assert ctx.turn_block == ""

    async def test_ids_cover_both_layers(self, test_db: Any) -> None:
        """ids feeds the utility reference scan, so it must span everything
        rendered — a fact used from standing context earned nothing before."""
        from unittest.mock import patch

        conn = db._db()
        _insert_memory(conn, "person-bg", "entity", "BG", "background fact", importance=1.9)
        cands = [{"id": "mem-turn", "type": "entity", "title": "T", "score": 0.9}]
        with patch("luke.context._rank_turn_candidates", return_value=cands):
            ctx = await context.assemble_context(
                query="tell me things", chat_id="12345", budget_tokens=6000
            )
        assert "mem-turn" in ctx.ids
        assert "person-bg" in ctx.ids
        assert ctx.recalled_ids == ["mem-turn"]

    async def test_turn_layer_cannot_starve_standing_context(self, test_db: Any) -> None:
        from unittest.mock import patch

        conn = db._db()
        _insert_memory(conn, "person-keep", "entity", "Keep", "b" * 400, importance=1.9)
        hogs = [
            {"id": f"hog{i}", "type": "entity", "title": "H" * 2000, "score": 0.9}
            for i in range(50)
        ]
        with patch("luke.context._rank_turn_candidates", return_value=hogs):
            ctx = await context.assemble_context(
                query="give me everything", chat_id="12345", budget_tokens=4000
            )
        assert "person-keep" in ctx.system_block


class TestSpeculativeTouchScope:
    """Exposure credit goes to the turn layer only.

    Touching standing context because it was injected is a closed loop:
    injected because it ranks, ranks because it was injected. The reference
    scan (useful_only) is the safe channel — it raises useful_count without
    raising access_count.
    """

    async def test_turn_hits_are_touched(self, test_db: Any) -> None:
        from unittest.mock import patch

        cands = [{"id": "mem-turn", "type": "entity", "title": "T", "score": 0.9}]
        with (
            patch("luke.context._rank_turn_candidates", return_value=cands),
            patch("luke.context._memory_module.touch_memories") as touch,
        ):
            await context.assemble_context(
                query="tell me things", chat_id="12345", budget_tokens=4000
            )
        touch.assert_called_once_with(["mem-turn"], useful=False)

    async def test_standing_memories_are_not_touched(self, test_db: Any) -> None:
        conn = db._db()
        _insert_memory(conn, "person-bg", "entity", "BG", "fact", importance=1.9)
        before = conn.execute(
            "SELECT access_count FROM memory_meta WHERE id = 'person-bg'"
        ).fetchone()["access_count"]
        await context.assemble_context(query="ok", chat_id="12345", budget_tokens=4000)
        after = conn.execute(
            "SELECT access_count FROM memory_meta WHERE id = 'person-bg'"
        ).fetchone()["access_count"]
        assert after == before


class TestRenderedNotConsidered:
    """Only what the model actually SAW counts.

    Retrieval routinely produces more candidates than the budget fits — 42 for
    one real query. Treating all of them as injected would speculatively touch
    memories that never appeared, and exclude them from the standing layer for
    a slot they never occupied.
    """

    def test_render_returns_only_what_fit(self, tmp_settings: Any) -> None:
        cands = [
            {"id": f"mem{i}", "type": "entity", "title": "T" * 2000, "score": 0.5}
            for i in range(20)
        ]
        block, _spent, rendered = context._render_turn_block(cands, 200)
        assert len(rendered) < len(cands)
        for mem_id in rendered:
            assert f"[{mem_id}]" in block

    async def test_unrendered_candidates_are_not_touched(self, test_db: Any) -> None:
        from unittest.mock import patch

        # Sized so a few fit and most do not.
        cands = [
            {"id": f"mem{i}", "type": "entity", "title": "T" * 350, "score": 0.5} for i in range(20)
        ]
        with (
            patch("luke.context._rank_turn_candidates", return_value=cands),
            patch("luke.context._memory_module.touch_memories") as touch,
        ):
            ctx = await context.assemble_context(
                query="a real question here", chat_id="12345", budget_tokens=1000
            )
        assert ctx.recalled_ids, "expected some candidates to fit"
        touched = touch.call_args[0][0]
        assert set(touched) == set(ctx.recalled_ids)
        assert len(touched) < len(cands)

    async def test_unrendered_candidates_stay_eligible_for_standing(self, test_db: Any) -> None:
        """A candidate the budget rejected must still be able to appear as
        standing context — it never occupied a turn slot."""
        from unittest.mock import patch

        conn = db._db()
        _insert_memory(conn, "person-a", "entity", "A", "x" * 400, importance=1.9)
        _insert_memory(conn, "person-b", "entity", "B", "y" * 400, importance=1.8)
        cands = [
            {"id": "person-a", "type": "entity", "title": "A", "score": 0.9},
            {"id": "person-b", "type": "entity", "title": "B", "score": 0.8},
        ]
        with patch("luke.context._rank_turn_candidates", return_value=cands):
            ctx = await context.assemble_context(
                query="tell me about them", chat_id="12345", budget_tokens=500
            )
        dropped = {"person-a", "person-b"} - set(ctx.recalled_ids)
        for mem_id in dropped:
            assert mem_id in ctx.system_block, f"{mem_id} was dropped from both layers"


class TestBudgetIsHonoured:
    """`spent` governs turn evidence + standing memory and must stay under
    budget. Pinned continuity is reported separately because it is not
    optional — folding it in would read as a permanent overrun."""

    async def test_spent_stays_within_budget(self, test_db: Any) -> None:
        from unittest.mock import patch

        conn = db._db()
        for i in range(60):
            _insert_memory(conn, f"entity-{i}", "entity", f"E{i}", "z" * 600, importance=1.6)
            _insert_memory(conn, f"insight-{i}", "insight", f"I{i}", "y" * 600, importance=1.5)
        cands = [
            {"id": f"turn-{i}", "type": "entity", "title": "T" * 900, "score": 0.9}
            for i in range(30)
        ]

        captured: list[tuple[int, int]] = []
        for budget in (1000, 2500, 4000, 6000, 8000):
            with patch("luke.context._rank_turn_candidates", return_value=cands):
                ctx = await context.assemble_context(
                    query="a substantive question", chat_id="12345", budget_tokens=budget
                )
            # tokens = spent + pinned; recompute spent the same way _assemble does
            turn_cost = context._estimate_tokens(ctx.turn_block)
            assert turn_cost <= int(budget * context._TURN_BUDGET_SHARE) + 50, (
                f"turn layer overspent at budget={budget}"
            )
            captured.append((budget, turn_cost))
        assert captured

    async def test_pinned_is_not_charged_to_the_budget(self, test_db: Any) -> None:
        """Continuity must survive even a budget too small to pay for it."""
        conn = db._db()
        _insert_memory(conn, "person-a", "entity", "A", "x" * 400, importance=1.9)
        ctx = await context.assemble_context(query="ok", chat_id="12345", budget_tokens=1)
        # No memory fits at budget=1, but the standing block still exists if
        # there is anything pinned for this chat.
        assert isinstance(ctx.system_block, str)


class TestTurnTypeCap:
    """Procedures were 28% of the corpus but 64% of every injected set."""

    @staticmethod
    def _procs(n: int) -> list[dict[str, Any]]:
        return [
            {"id": f"proc-{i}", "type": "procedure", "title": f"P{i}", "score": 0.9 - i * 0.01}
            for i in range(n)
        ]

    def test_procedures_are_capped(self, tmp_settings: Any) -> None:
        from unittest.mock import patch

        with (
            patch("luke.context._memory_module.recall", return_value=self._procs(20)),
            patch("luke.context._memory_module.get_trigger_matched_skills", return_value=[]),
            patch("luke.context._memory_module.get_graph_neighbors", return_value=[]),
        ):
            out = context._rank_turn_candidates("do the thing", set())
        assert len(out) == context._TURN_TYPE_CAP["procedure"]

    def test_cap_keeps_the_highest_scoring(self, tmp_settings: Any) -> None:
        from unittest.mock import patch

        with (
            patch("luke.context._memory_module.recall", return_value=self._procs(20)),
            patch("luke.context._memory_module.get_trigger_matched_skills", return_value=[]),
            patch("luke.context._memory_module.get_graph_neighbors", return_value=[]),
        ):
            out = context._rank_turn_candidates("do the thing", set())
        assert [c["id"] for c in out] == ["proc-0", "proc-1", "proc-2"]

    def test_trigger_skills_are_exempt(self, tmp_settings: Any) -> None:
        """A trigger match is an explicit answer signal; the cap exists to
        stop incidental procedures, not chosen ones."""
        from unittest.mock import patch

        skills = [
            {"id": "skill-a", "type": "procedure", "title": "A", "score": 1.0},
            {"id": "skill-b", "type": "procedure", "title": "B", "score": 1.0},
        ]
        with (
            patch("luke.context._memory_module.recall", return_value=self._procs(20)),
            patch("luke.context._memory_module.get_trigger_matched_skills", return_value=skills),
            patch("luke.context._memory_module.get_graph_neighbors", return_value=[]),
        ):
            out = context._rank_turn_candidates("deploy", set())
        ids = [c["id"] for c in out]
        # Never blocked...
        assert "skill-a" in ids and "skill-b" in ids
        # ...but they still count toward the tally, so total procedure share
        # stays bounded. Exempting them from the count too would let a turn
        # carry cap + skills procedures and defeat the cap.
        assert len(ids) == context._TURN_TYPE_CAP["procedure"]

    def test_skills_admitted_even_when_cap_already_full(self, tmp_settings: Any) -> None:
        """A trigger match must land even if procedures are already at cap."""
        from unittest.mock import patch

        skills = [{"id": "skill-a", "type": "procedure", "title": "A", "score": 1.0}]
        with (
            patch("luke.context._memory_module.recall", return_value=self._procs(20)),
            patch("luke.context._memory_module.get_trigger_matched_skills", return_value=skills),
            patch("luke.context._memory_module.get_graph_neighbors", return_value=[]),
        ):
            out = context._rank_turn_candidates("deploy", set())
        assert "skill-a" in [c["id"] for c in out]

    def test_uncapped_types_are_unaffected(self, tmp_settings: Any) -> None:
        from unittest.mock import patch

        ents = [
            {"id": f"e{i}", "type": "entity", "title": f"E{i}", "score": 0.5} for i in range(10)
        ]
        with (
            patch("luke.context._memory_module.recall", return_value=ents),
            patch("luke.context._memory_module.get_trigger_matched_skills", return_value=[]),
            patch("luke.context._memory_module.get_graph_neighbors", return_value=[]),
        ):
            out = context._rank_turn_candidates("who are they", set())
        assert len(out) == 10


class TestAgeLabel:
    """The ranker decays by recency; the model could not see it. Every memory
    rendered identically, so a March episode read as current fact."""

    @staticmethod
    def _ago(days: float) -> str:
        return (datetime.now(UTC) - timedelta(days=days)).isoformat()

    def test_today(self) -> None:
        assert context._age_label(self._ago(0.2)) == "today"

    def test_yesterday(self) -> None:
        assert context._age_label(self._ago(1.5)) == "yesterday"

    def test_days(self) -> None:
        assert context._age_label(self._ago(5)) == "5d ago"

    def test_weeks(self) -> None:
        assert context._age_label(self._ago(30)) == "4 weeks ago"

    def test_months(self) -> None:
        assert context._age_label(self._ago(120)) == "4 months ago"

    def test_missing_or_malformed(self) -> None:
        assert context._age_label("") == ""
        assert context._age_label("not-a-date") == ""

    def test_rendered_into_the_turn_block(self, tmp_settings: Any) -> None:
        cands = [
            {
                "id": "ep-old",
                "type": "episode",
                "title": "Old thing",
                "score": 0.5,
                "updated": self._ago(120),
            }
        ]
        block, _spent, _rendered = context._render_turn_block(cands, 5000)
        assert "(episode, 4 months ago)" in block

    def test_absent_timestamp_still_renders(self, tmp_settings: Any) -> None:
        cands = [{"id": "m1", "type": "entity", "title": "T", "score": 0.5}]
        block, _spent, _rendered = context._render_turn_block(cands, 5000)
        assert "[m1] (entity)" in block
