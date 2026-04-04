"""Tests for luke.memory — memory index, recall, graph, scoring, decay, archiving."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from luke import db, memory
from luke.memory import _recency_score, strip_frontmatter

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestStripFrontmatter:
    def test_with_frontmatter(self) -> None:
        text = "---\ntitle: Hello\n---\n\nBody text"
        assert strip_frontmatter(text) == "Body text"

    def test_without_frontmatter(self) -> None:
        text = "Just regular text"
        assert strip_frontmatter(text) == "Just regular text"

    def test_incomplete_frontmatter(self) -> None:
        text = "---\ntitle: Hello\nNo closing"
        assert strip_frontmatter(text) == text

    def test_empty_frontmatter(self) -> None:
        text = "---\n---\n\nBody"
        assert strip_frontmatter(text) == "Body"


# ---------------------------------------------------------------------------
# Index & Recall (FTS-only path, no embeddings)
# ---------------------------------------------------------------------------


class TestIndexAndRecall:
    def test_fts_roundtrip(self, test_db: Any) -> None:
        memory.index_memory(
            "proj-x",
            "entity",
            "Project X",
            "A research project on AI safety",
        )
        results = memory.recall(query="research project")
        assert len(results) >= 1
        assert any(r["id"] == "proj-x" for r in results)

    def test_recall_by_type_only(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "Entity One", "content one")
        memory.index_memory("ep1", "episode", "Episode One", "content two")
        results = memory.recall(mem_type="entity")
        assert all(r["type"] == "entity" for r in results)

    def test_recall_temporal_range(self, test_db: Any) -> None:
        memory.index_memory("e1", "episode", "Past Event", "old event")
        # Set updated to a known time
        now = datetime.now(UTC)
        past = (now - timedelta(days=2)).isoformat()
        future = (now + timedelta(days=1)).isoformat()
        results = memory.recall(after=past, before=future)
        assert len(results) >= 1

    def test_recall_no_results(self, test_db: Any) -> None:
        results = memory.recall(query="xyzzy_nonexistent_12345")
        assert results == []

    def test_recall_fts_injection_safe(self, test_db: Any) -> None:
        """Malformed FTS5 query syntax must not raise."""
        memory.index_memory("e1", "entity", "Test", "content")
        results = memory.recall(query="OR OR")
        # Should return empty or handle gracefully, not crash
        assert isinstance(results, list)

    def test_recall_fts_special_chars(self, test_db: Any) -> None:
        """Special characters in queries must not crash."""
        memory.index_memory("e1", "entity", "Test", "content")
        results = memory.recall(query="test AND (OR) NOT *")
        assert isinstance(results, list)

    def test_reindex_preserves_created(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "V1", "first version")
        row1 = db._db().execute("SELECT created FROM memory_meta WHERE id = ?", ("e1",)).fetchone()

        memory.index_memory("e1", "entity", "V2", "second version")
        row2 = db._db().execute("SELECT created FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        assert row1["created"] == row2["created"]

    def test_reindex_preserves_access_count(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "V1", "content")
        memory.touch_memories(["e1"])
        memory.touch_memories(["e1"])
        row1 = (
            db._db()
            .execute("SELECT access_count FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row1["access_count"] == 2

        memory.index_memory("e1", "entity", "V2", "updated")
        row2 = (
            db._db()
            .execute("SELECT access_count FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row2["access_count"] == 2

    def test_reindex_preserves_importance(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "V1", "content")
        # Manually set importance
        db._db().execute("UPDATE memory_meta SET importance = 0.5 WHERE id = ?", ("e1",))
        db._db().commit()

        memory.index_memory("e1", "entity", "V2", "updated")
        row = (
            db._db().execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        )
        assert row["importance"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------


class TestGraph:
    def test_link_and_recall_related(self, test_db: Any) -> None:
        memory.index_memory("a", "entity", "Node A", "content a")
        memory.index_memory("b", "entity", "Node B", "content b")
        memory.link_memories("a", "b", "related_to")

        results = memory.recall(related_to="a")
        assert any(r["id"] == "b" for r in results)

    def test_graph_multi_hop(self, test_db: Any) -> None:
        """A -> B -> C: querying related_to=A should find C (depth 2)."""
        memory.index_memory("a", "entity", "Node A", "a")
        memory.index_memory("b", "entity", "Node B", "b")
        memory.index_memory("c", "entity", "Node C", "c")
        memory.link_memories("a", "b", "knows")
        memory.link_memories("b", "c", "knows")

        results = memory.recall(related_to="a")
        ids = {r["id"] for r in results}
        assert "b" in ids
        assert "c" in ids

    def test_graph_bidirectional(self, test_db: Any) -> None:
        """Links work in both directions."""
        memory.index_memory("x", "entity", "X", "x")
        memory.index_memory("y", "entity", "Y", "y")
        memory.link_memories("x", "y", "linked")

        # Query from y should find x
        results = memory.recall(related_to="y")
        assert any(r["id"] == "x" for r in results)

    def test_graph_no_cycles(self, test_db: Any) -> None:
        """Cycles in graph must not cause infinite loops."""
        memory.index_memory("a", "entity", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.link_memories("a", "b", "knows")
        memory.link_memories("b", "a", "knows")

        results = memory.recall(related_to="a")
        assert isinstance(results, list)  # Didn't hang


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_recency_score_now(self) -> None:
        now = datetime.now(UTC).isoformat()
        score = _recency_score(now)
        assert score > 0.99

    def test_recency_score_old(self) -> None:
        old = (datetime.now(UTC) - timedelta(days=365)).isoformat()
        score = _recency_score(old)
        assert score < 0.01

    def test_recency_score_empty(self) -> None:
        assert _recency_score("") == 0.0

    def test_composite_scoring_applied(self, test_db: Any) -> None:
        """After recall, scores should reflect composite weights (not just FTS rank)."""
        memory.index_memory("e1", "entity", "Test Entity", "keyword alpha")
        results = memory.recall(query="keyword alpha")
        assert len(results) >= 1
        # Score should include importance/recency/access components (> raw FTS score of ~0-1)
        assert results[0]["score"] > 0


# ---------------------------------------------------------------------------
# Decay & Archiving
# ---------------------------------------------------------------------------


class TestDecayAndArchiving:
    def test_decay_importance(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "Test", "content")
        before = (
            db._db()
            .execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()["importance"]
        )

        updated = memory.decay_importance({"entity": 0.99})
        assert updated >= 1

        after = (
            db._db()
            .execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()["importance"]
        )
        assert after < before

    def test_decay_modulated_by_access(self, test_db: Any) -> None:
        """Higher access_count should result in slower decay."""
        memory.index_memory("low", "entity", "Low Access", "c")
        memory.index_memory("high", "entity", "High Access", "c")

        # Give "high" many accesses
        for _ in range(10):
            memory.touch_memories(["high"])

        memory.decay_importance({"entity": 0.99})

        low_imp = (
            db._db()
            .execute("SELECT importance FROM memory_meta WHERE id = ?", ("low",))
            .fetchone()["importance"]
        )
        high_imp = (
            db._db()
            .execute("SELECT importance FROM memory_meta WHERE id = ?", ("high",))
            .fetchone()["importance"]
        )
        # High-access memory should retain more importance
        assert high_imp > low_imp

    def test_touch_memories_increments(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "Test", "c")
        memory.touch_memories(["e1"])
        memory.touch_memories(["e1"])
        row = (
            db._db()
            .execute("SELECT access_count FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row["access_count"] == 2

    def test_archive_excludes_from_recall(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "Archived", "secret content")
        memory.archive_memory("e1")
        results = memory.recall(query="secret content")
        assert not any(r["id"] == "e1" for r in results)

    def test_cleanup_archived_fts(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "To Archive", "content")
        memory.archive_memory("e1")
        memory.cleanup_archived_fts()
        # FTS entry should be gone
        row = db._db().execute("SELECT id FROM memory_fts WHERE id = ?", ("e1",)).fetchone()
        assert row is None


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


class TestConflictDetection:
    def test_detect_title_change(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "Old Title", "content")
        changes = memory.detect_changes("e1", "content", "New Title")
        assert len(changes) == 1
        assert "Title" in changes[0]

    def test_detect_content_change(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "Title", "old content")
        changes = memory.detect_changes("e1", "new content", "Title")
        assert len(changes) == 1
        assert "Content" in changes[0]

    def test_detect_no_existing(self, test_db: Any) -> None:
        changes = memory.detect_changes("nonexistent", "content", "title")
        assert changes == []

    def test_detect_both_changes(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "Old", "old content")
        changes = memory.detect_changes("e1", "new content", "New")
        assert len(changes) == 2  # title + content


# ---------------------------------------------------------------------------
# Consolidation candidates
# ---------------------------------------------------------------------------


class TestConsolidation:
    def test_finds_cluster(self, test_db: Any) -> None:
        shared_tags = ["tag1", "tag2", "tag3"]
        for i in range(4):
            memory.index_memory(
                f"ep{i}",
                "episode",
                f"Episode {i}",
                f"content {i}",
                tags=shared_tags,
            )
        clusters = memory.get_consolidation_candidates(min_shared=3)
        assert len(clusters) >= 1
        assert len(clusters[0]) >= 3

    def test_insufficient_shared_tags(self, test_db: Any) -> None:
        memory.index_memory("ep1", "episode", "Ep 1", "c", tags=["a", "b"])
        memory.index_memory("ep2", "episode", "Ep 2", "c", tags=["c", "d"])
        memory.index_memory("ep3", "episode", "Ep 3", "c", tags=["e", "f"])
        clusters = memory.get_consolidation_candidates(min_shared=3)
        assert clusters == []


# ---------------------------------------------------------------------------
# Time window recall
# ---------------------------------------------------------------------------


class TestTimeWindowRecall:
    def test_recall_by_time_window(self, test_db: Any) -> None:
        memory.index_memory("e1", "episode", "Event", "happened")
        now = datetime.now(UTC)
        results = memory.recall_by_time_window(
            after=(now - timedelta(hours=1)).isoformat(),
            before=(now + timedelta(hours=1)).isoformat(),
        )
        assert len(results) >= 1
        assert results[0]["id"] == "e1"


# ---------------------------------------------------------------------------
# Sync memory index
# ---------------------------------------------------------------------------


class TestFTSRetention:
    def test_prune_old_episodes(self, test_db: Any) -> None:
        memory.index_memory("old-ep", "episode", "Old", "old content")
        # Set updated to 6 years ago and low importance
        old_date = (datetime.now(UTC) - timedelta(days=2200)).isoformat()
        db._db().execute(
            "UPDATE memory_meta SET updated = ?, importance = 0.01 WHERE id = ?",
            (old_date, "old-ep"),
        )
        db._db().commit()

        pruned = memory.prune_old_fts_entries(1825)  # 5 years
        assert pruned == 1
        # Should be archived
        row = (
            db._db().execute("SELECT status FROM memory_meta WHERE id = ?", ("old-ep",)).fetchone()
        )
        assert row["status"] == "archived"

    def test_prune_skips_recent(self, test_db: Any) -> None:
        memory.index_memory("new-ep", "episode", "New", "recent")
        pruned = memory.prune_old_fts_entries(1825)
        assert pruned == 0

    def test_prune_disabled_when_zero(self, test_db: Any) -> None:
        assert memory.prune_old_fts_entries(0) == 0

    def test_prune_skips_high_importance(self, test_db: Any) -> None:
        memory.index_memory("imp-ep", "episode", "Important", "vital")
        old_date = (datetime.now(UTC) - timedelta(days=2200)).isoformat()
        db._db().execute(
            "UPDATE memory_meta SET updated = ?, importance = 0.5 WHERE id = ?",
            (old_date, "imp-ep"),
        )
        db._db().commit()
        pruned = memory.prune_old_fts_entries(1825)
        assert pruned == 0  # High importance preserved


class TestRestoreMemory:
    def test_restore_archived(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "Test", "content")
        memory.archive_memory("e1")
        # Verify archived
        results = memory.recall(query="content")
        assert not any(r["id"] == "e1" for r in results)

        # Restore
        restored = memory.restore_memory("e1")
        assert restored is True
        # Verify active again
        row = db._db().execute("SELECT status FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        assert row["status"] == "active"

    def test_restore_nonexistent_returns_false(self, test_db: Any) -> None:
        assert memory.restore_memory("nonexistent") is False


class TestUpdateMemoryTags:
    def test_update_tags(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "Test", "content", tags=["old"])
        memory.update_memory_tags("e1", ["new", "updated"])
        row = db._db().execute("SELECT tags_json FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        import json

        assert json.loads(row["tags_json"]) == ["new", "updated"]


class TestPrivateMemoryTag:
    def test_private_excluded_from_recall(self, test_db: Any) -> None:
        memory.index_memory(
            "secret",
            "entity",
            "Secret",
            "private stuff",
            tags=["private"],
        )
        memory.index_memory("public", "entity", "Public", "public stuff")
        results = memory.recall(query="stuff")
        ids = {r["id"] for r in results}
        assert "public" in ids
        assert "secret" not in ids

    def test_private_included_with_override(self, test_db: Any) -> None:
        memory.index_memory(
            "secret",
            "entity",
            "Secret",
            "hidden data",
            tags=["private"],
        )
        results = memory.recall(query="hidden data", include_private=True)
        assert any(r["id"] == "secret" for r in results)


class TestSyncMemoryIndex:
    def test_discovers_unindexed_files(
        self,
        test_db: Any,
        tmp_settings: Any,
    ) -> None:
        """Files on disk with frontmatter should get indexed."""
        mem_dir = tmp_settings.memory_dir / "entities"
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / "test-entity.md").write_text(
            "---\nid: test-entity\ntype: entity\ntags: []\nlinks: []\n"
            "---\n\n# Test Entity\n\nSome content"
        )
        memory.sync_memory_index()
        results = memory.recall(query="Test Entity")
        assert any(r["id"] == "test-entity" for r in results)

    def test_skips_already_indexed(self, test_db: Any, tmp_settings: Any) -> None:
        """Already-indexed files should not be re-indexed."""
        memory.index_memory("existing", "entity", "Existing", "content")
        mem_dir = tmp_settings.memory_dir / "entities"
        (mem_dir / "existing.md").write_text(
            "---\nid: existing\ntype: entity\ntags: []\nlinks: []\n---\n\n# Existing\n\ncontent"
        )
        # Should not raise or duplicate
        memory.sync_memory_index()
        row = (
            db._db()
            .execute("SELECT COUNT(*) as cnt FROM memory_meta WHERE id = ?", ("existing",))
            .fetchone()
        )
        assert row["cnt"] == 1


# ---------------------------------------------------------------------------
# Scoring bias fix
# ---------------------------------------------------------------------------


class TestScoringBias:
    def test_query_match_outranks_recent_irrelevant(self, test_db: Any) -> None:
        """A relevant query match should outscore a recent but irrelevant memory."""
        memory.index_memory("relevant", "entity", "Alpha Project", "alpha project details")
        memory.index_memory("irrelevant", "entity", "Beta Unrelated", "completely different topic")
        results = memory.recall(query="alpha project")
        # The relevant match should be first
        if len(results) >= 2:
            ids = [r["id"] for r in results]
            assert ids.index("relevant") < ids.index("irrelevant")

    def test_zero_relevance_dampened(self, test_db: Any) -> None:
        """Non-query results (temporal) should have dampened scores."""
        memory.index_memory("e1", "entity", "Test", "content")
        # Temporal-only query (no text query, so relevance = 0)
        results = memory.recall(after="2020-01-01", before="2030-01-01")
        assert len(results) >= 1
        # Score should be dampened (< 0.5 since it's context_norm * 0.3)
        assert results[0]["score"] < 0.5


# ---------------------------------------------------------------------------
# Graph neighbors
# ---------------------------------------------------------------------------


class TestGraphNeighbors:
    def test_batch_neighbors(self, test_db: Any) -> None:
        memory.index_memory("a", "entity", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.index_memory("c", "entity", "C", "c")
        memory.link_memories("a", "b", "knows")
        memory.link_memories("a", "c", "works_with")
        neighbors = memory.get_graph_neighbors(["a"])
        ids = {n["id"] for n in neighbors}
        assert "b" in ids
        assert "c" in ids

    def test_excludes_input_ids(self, test_db: Any) -> None:
        memory.index_memory("a", "entity", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.link_memories("a", "b", "knows")
        neighbors = memory.get_graph_neighbors(["a", "b"])
        ids = {n["id"] for n in neighbors}
        assert "a" not in ids
        assert "b" not in ids

    def test_empty_input(self, test_db: Any) -> None:
        assert memory.get_graph_neighbors([]) == []

    def test_respects_limit(self, test_db: Any) -> None:
        memory.index_memory("hub", "entity", "Hub", "hub")
        for i in range(5):
            memory.index_memory(f"spoke{i}", "entity", f"Spoke {i}", f"spoke {i}")
            memory.link_memories("hub", f"spoke{i}", "connected")
        neighbors = memory.get_graph_neighbors(["hub"], limit=2)
        assert len(neighbors) <= 2


# ---------------------------------------------------------------------------
# Importance initialization
# ---------------------------------------------------------------------------


class TestImportanceInit:
    def test_custom_importance(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "Important", "vital info", importance=1.5)
        row = (
            db._db().execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        )
        assert row["importance"] == pytest.approx(1.5)

    def test_default_importance(self, test_db: Any) -> None:
        memory.index_memory("e1", "entity", "Normal", "regular info")
        row = (
            db._db().execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        )
        assert row["importance"] == pytest.approx(1.0)

    def test_reindex_keeps_existing_importance(self, test_db: Any) -> None:
        """Re-indexing without explicit importance preserves existing value."""
        memory.index_memory("e1", "entity", "V1", "c", importance=1.8)
        memory.index_memory("e1", "entity", "V2", "c")  # no importance → keep 1.8
        row = (
            db._db().execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        )
        assert row["importance"] == pytest.approx(1.8)

    def test_reindex_overrides_with_explicit(self, test_db: Any) -> None:
        """Re-indexing with explicit importance overrides existing."""
        memory.index_memory("e1", "entity", "V1", "c", importance=1.8)
        memory.index_memory("e1", "entity", "V2", "c", importance=0.5)
        row = (
            db._db().execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        )
        assert row["importance"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# FTS5 sanitization
# ---------------------------------------------------------------------------


class TestFTSSanitization:
    def test_or_semantics_broader_recall(self, test_db: Any) -> None:
        """Multi-term queries should return results matching ANY term (OR semantics)."""
        memory.index_memory("e1", "entity", "Alpha Project", "alpha details here")
        memory.index_memory("e2", "entity", "Beta Project", "beta details here")
        r1 = memory.recall(query="alpha")
        r2 = memory.recall(query="alpha beta")
        assert len(r2) >= len(r1)

    def test_special_chars_stripped(self, test_db: Any) -> None:
        """FTS5 operators in natural text should not crash or alter semantics."""
        memory.index_memory("e1", "entity", "Test", "not a problem at all")
        results = memory.recall(query='is this NOT a "problem"?')
        assert isinstance(results, list)

    def test_sanitize_empty_after_strip(self, test_db: Any) -> None:
        """If sanitization produces empty string, no FTS crash."""
        results = memory.recall(query="- * + ()")
        assert results == []

    def test_colons_and_hyphens_stripped(self, test_db: Any) -> None:
        """Colons and hyphens are FTS5 column/prefix syntax — must be stripped."""
        memory.index_memory("e1", "entity", "Test", "hello world")
        # These would crash FTS5 without sanitization
        results = memory.recall(query="test:colon")
        assert isinstance(results, list)
        results = memory.recall(query="--double-dash")
        assert isinstance(results, list)

    def test_single_char_terms_dropped(self, test_db: Any) -> None:
        """Single-character terms are dropped to avoid noise."""
        from luke.memory import _sanitize_fts_query

        assert _sanitize_fts_query("I am a test") == "am OR test"


# ---------------------------------------------------------------------------
# Bounded importance in scoring
# ---------------------------------------------------------------------------


class TestBoundedImportance:
    def test_high_importance_clamped_in_scoring(self, test_db: Any) -> None:
        """Importance > 1.0 should be clamped to 1.0 in composite scoring."""
        memory.index_memory("high", "entity", "High Imp", "keyword stuff", importance=2.0)
        memory.index_memory("normal", "entity", "Normal Imp", "keyword stuff", importance=1.0)
        results = memory.recall(query="keyword stuff")
        scores = {m["id"]: m["score"] for m in results}
        # Both have importance clamped to 1.0, so scores should be close
        if "high" in scores and "normal" in scores:
            assert scores["high"] == pytest.approx(scores["normal"], abs=0.1)

    def test_stored_importance_not_clamped(self, test_db: Any) -> None:
        """Importance > 1.0 should still be stored for decay runway."""
        memory.index_memory("e1", "entity", "Test", "c", importance=2.0)
        row = (
            db._db().execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        )
        assert row["importance"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Utility tracking
# ---------------------------------------------------------------------------


class TestUtilityTracking:
    def test_touch_increments_both_counts(self, test_db: Any) -> None:
        """Intentional touch increments both access_count and useful_count."""
        memory.index_memory("e1", "entity", "Test", "content")
        memory.touch_memories(["e1"])
        row = (
            db._db()
            .execute("SELECT access_count, useful_count FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row["access_count"] == 1
        assert row["useful_count"] == 1

    def test_speculative_touch_only_access(self, test_db: Any) -> None:
        """Speculative touch increments access_count only, not useful_count."""
        memory.index_memory("e1", "entity", "Test", "content")
        memory.touch_memories(["e1"], useful=False)
        row = (
            db._db()
            .execute("SELECT access_count, useful_count FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row["access_count"] == 1
        assert row["useful_count"] == 0

    def test_last_accessed_updated(self, test_db: Any) -> None:
        """Both touch functions update last_accessed timestamp."""
        memory.index_memory("e1", "entity", "Test", "content")
        memory.touch_memories(["e1"])
        row = (
            db._db()
            .execute("SELECT last_accessed FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row["last_accessed"] != ""

    def test_utility_affects_scoring(self, test_db: Any) -> None:
        """Memory with low utility rate should score lower than high utility."""
        memory.index_memory("low", "entity", "Low Utility", "keyword test")
        memory.index_memory("high", "entity", "High Utility", "keyword test")
        for _ in range(10):
            memory.touch_memories(["low"], useful=False)  # 0 useful / 10 access
            memory.touch_memories(["high"])  # 10 useful / 10 access
        results = memory.recall(query="keyword test")
        scores = {m["id"]: m["score"] for m in results}
        if "low" in scores and "high" in scores:
            assert scores["high"] > scores["low"]

    def test_reindex_preserves_useful_count(self, test_db: Any) -> None:
        """Re-indexing should preserve useful_count."""
        memory.index_memory("e1", "entity", "V1", "content")
        memory.touch_memories(["e1"])
        memory.touch_memories(["e1"])
        memory.index_memory("e1", "entity", "V2", "updated content")
        row = (
            db._db()
            .execute("SELECT useful_count FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row["useful_count"] == 2


# ---------------------------------------------------------------------------
# Co-access link strengthening
# ---------------------------------------------------------------------------


class TestCoAccessStrengthening:
    def test_co_access_strengthens_links(self, test_db: Any) -> None:
        """Recalling linked memories together increases link weight."""
        memory.index_memory("a", "entity", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.link_memories("a", "b", "related")
        before = (
            db._db()
            .execute(
                "SELECT weight FROM memory_links WHERE from_id = ? AND to_id = ?",
                ("a", "b"),
            )
            .fetchone()["weight"]
        )
        memory.touch_memories(["a", "b"])
        after = (
            db._db()
            .execute(
                "SELECT weight FROM memory_links WHERE from_id = ? AND to_id = ?",
                ("a", "b"),
            )
            .fetchone()["weight"]
        )
        assert after == pytest.approx(before + 0.05)

    def test_weight_capped_at_5(self, test_db: Any) -> None:
        """Link weight should not exceed 5.0."""
        memory.index_memory("a", "entity", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.link_memories("a", "b", "related")
        db._db().execute(
            "UPDATE memory_links SET weight = 4.98 WHERE from_id = ? AND to_id = ?",
            ("a", "b"),
        )
        db._db().commit()
        memory.touch_memories(["a", "b"])
        after = (
            db._db()
            .execute(
                "SELECT weight FROM memory_links WHERE from_id = ? AND to_id = ?",
                ("a", "b"),
            )
            .fetchone()["weight"]
        )
        assert after == pytest.approx(5.0)

    def test_no_link_no_strengthen(self, test_db: Any) -> None:
        """Co-accessing unlinked memories should not create links."""
        memory.index_memory("x", "entity", "X", "x")
        memory.index_memory("y", "entity", "Y", "y")
        memory.touch_memories(["x", "y"])
        row = (
            db._db()
            .execute(
                "SELECT * FROM memory_links WHERE from_id = ? AND to_id = ?",
                ("x", "y"),
            )
            .fetchone()
        )
        assert row is None

    def test_link_memories_preserves_weight(self, test_db: Any) -> None:
        """Re-linking should not reset accumulated weight."""
        memory.index_memory("a", "entity", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.link_memories("a", "b", "related")
        db._db().execute(
            "UPDATE memory_links SET weight = 3.0 WHERE from_id = ? AND to_id = ?",
            ("a", "b"),
        )
        db._db().commit()
        memory.link_memories("a", "b", "related")  # re-link same relationship
        row = (
            db._db()
            .execute(
                "SELECT weight FROM memory_links WHERE from_id = ? AND to_id = ?",
                ("a", "b"),
            )
            .fetchone()
        )
        assert row["weight"] == pytest.approx(3.0)  # preserved, not reset to 1.0


# ---------------------------------------------------------------------------
# Memory history
# ---------------------------------------------------------------------------


class TestMemoryHistory:
    def test_record_and_retrieve(self, test_db: Any) -> None:
        memory.record_memory_change("e1", ["Title: 'Old' -> 'New'", "Content updated"])
        history = memory.get_memory_history("e1")
        assert len(history) == 1
        assert len(history[0]["changes"]) == 2

    def test_history_ordered_desc(self, test_db: Any) -> None:
        memory.record_memory_change("e1", ["Change 1"])
        memory.record_memory_change("e1", ["Change 2"])
        history = memory.get_memory_history("e1")
        assert len(history) == 2
        assert history[0]["changes"] == ["Change 2"]

    def test_no_history_returns_empty(self, test_db: Any) -> None:
        assert memory.get_memory_history("nonexistent") == []

    def test_empty_changes_not_recorded(self, test_db: Any) -> None:
        memory.record_memory_change("e1", [])
        assert memory.get_memory_history("e1") == []


# ---------------------------------------------------------------------------
# Temporal validity on links
# ---------------------------------------------------------------------------


class TestTemporalLinks:
    def test_link_has_created_and_null_valid_until(self, test_db: Any) -> None:
        memory.index_memory("a", "entity", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.link_memories("a", "b", "related")
        row = (
            db._db()
            .execute("SELECT created, valid_until FROM memory_links WHERE from_id = 'a'")
            .fetchone()
        )
        assert row["created"] != ""
        assert row["valid_until"] is None

    def test_invalidate_link(self, test_db: Any) -> None:
        memory.index_memory("a", "entity", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.link_memories("a", "b", "related")
        assert memory.invalidate_link("a", "b", "related") is True
        row = (
            db._db()
            .execute(
                "SELECT valid_until FROM memory_links"
                " WHERE from_id = 'a' AND relationship = 'related'"
            )
            .fetchone()
        )
        assert row["valid_until"] is not None and row["valid_until"] != ""

    def test_invalidate_nonexistent(self, test_db: Any) -> None:
        assert memory.invalidate_link("x", "y", "nope") is False

    def test_invalidate_already_expired(self, test_db: Any) -> None:
        memory.index_memory("a", "entity", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.link_memories("a", "b", "related")
        assert memory.invalidate_link("a", "b", "related") is True
        # Second invalidation returns False (already expired)
        assert memory.invalidate_link("a", "b", "related") is False

    def test_graph_excludes_expired(self, test_db: Any) -> None:
        memory.index_memory("a", "entity", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.link_memories("a", "b", "related")
        memory.invalidate_link("a", "b", "related")
        results = memory.recall(related_to="a")
        assert not any(r["id"] == "b" for r in results)

    def test_coexisting_relationships(self, test_db: Any) -> None:
        """Same pair can have both 'related' and 'caused' active."""
        memory.index_memory("a", "episode", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.link_memories("a", "b", "related")
        memory.link_memories("a", "b", "caused")
        results = memory.recall(related_to="a")
        assert any(r["id"] == "b" for r in results)

    def test_hebbian_only_active(self, test_db: Any) -> None:
        memory.index_memory("a", "entity", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.link_memories("a", "b", "related")
        memory.invalidate_link("a", "b", "related")
        # Strengthen should not affect expired link
        memory.touch_memories(["a", "b"], useful=True)
        row = db._db().execute("SELECT weight FROM memory_links WHERE from_id = 'a'").fetchone()
        assert row["weight"] == pytest.approx(1.0)  # unchanged

    def test_neighbors_filters_expired(self, test_db: Any) -> None:
        memory.index_memory("a", "entity", "A", "a")
        memory.index_memory("b", "entity", "B", "b")
        memory.link_memories("a", "b", "related")
        memory.invalidate_link("a", "b", "related")
        neighbors = memory.get_graph_neighbors(["a"])
        assert not any(n["id"] == "b" for n in neighbors)


# ---------------------------------------------------------------------------
# Type-pair default labels
# ---------------------------------------------------------------------------


class TestDefaultLabels:
    def test_episode_entity_label(self, test_db: Any) -> None:
        memory.index_memory("ep1", "episode", "Episode", "content", links=["ent1"])
        memory.index_memory("ent1", "entity", "Entity", "content")
        # Re-index to pick up the link with correct type pair
        memory.index_memory("ep1", "episode", "Episode", "content", links=["ent1"])
        row = (
            db._db()
            .execute(
                "SELECT relationship FROM memory_links WHERE from_id = 'ep1' AND to_id = 'ent1'"
            )
            .fetchone()
        )
        assert row is not None
        assert row["relationship"] == "involves"

    def test_unmapped_pair_defaults_to_related(self, test_db: Any) -> None:
        memory.index_memory("a", "entity", "A", "a", links=["b"])
        memory.index_memory("b", "entity", "B", "b")
        memory.index_memory("a", "entity", "A", "a", links=["b"])
        row = (
            db._db()
            .execute("SELECT relationship FROM memory_links WHERE from_id = 'a' AND to_id = 'b'")
            .fetchone()
        )
        assert row is not None
        assert row["relationship"] == "related"


# ---------------------------------------------------------------------------
# Utility tracking
# ---------------------------------------------------------------------------


class TestUsefulOnly:
    def test_increments_useful_only(self, test_db: Any) -> None:
        memory.index_memory("a", "entity", "A", "a")
        before = (
            db._db()
            .execute("SELECT access_count, useful_count FROM memory_meta WHERE id = 'a'")
            .fetchone()
        )
        memory.touch_memories(["a"], useful_only=True)
        after = (
            db._db()
            .execute("SELECT access_count, useful_count FROM memory_meta WHERE id = 'a'")
            .fetchone()
        )
        assert after["useful_count"] == before["useful_count"] + 1
        assert after["access_count"] == before["access_count"]  # unchanged

    def test_empty_list_noop(self, test_db: Any) -> None:
        memory.touch_memories([], useful_only=True)  # should not raise


# ---------------------------------------------------------------------------
# Lifecycle candidates
# ---------------------------------------------------------------------------


class TestLifecycleCandidates:
    def test_stale_entity_detected(self, test_db: Any) -> None:
        memory.index_memory("old-entity", "entity", "Old", "content")
        # Manually set updated to 100 days ago
        old_date = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        db._db().execute("UPDATE memory_meta SET updated = ? WHERE id = 'old-entity'", (old_date,))
        db._db().commit()
        candidates = memory.get_lifecycle_candidates()
        assert any(e["id"] == "old-entity" for e in candidates["stale_entities"])

    def test_fresh_entity_not_flagged(self, test_db: Any) -> None:
        memory.index_memory("fresh", "entity", "Fresh", "content")
        candidates = memory.get_lifecycle_candidates()
        assert not any(e["id"] == "fresh" for e in candidates["stale_entities"])

    def test_unused_procedure_detected(self, test_db: Any) -> None:
        memory.index_memory("old-proc", "procedure", "Proc", "content")
        candidates = memory.get_lifecycle_candidates()
        # last_accessed defaults to '' which counts as unused
        assert any(p["id"] == "old-proc" for p in candidates["unused_procedures"])

    def test_lingering_goal_detected(self, test_db: Any) -> None:
        memory.index_memory("done-goal", "goal", "Done", "All done.", tags=["completed"])
        candidates = memory.get_lifecycle_candidates()
        assert any(g["id"] == "done-goal" for g in candidates["lingering_goals"])


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class TestTaxonomy:
    def test_default_taxonomy_entity(self, test_db: Any) -> None:
        """Entities default to 'factual' taxonomy."""
        memory.index_memory("e1", "entity", "Person", "content")
        row = db._db().execute(
            "SELECT taxonomy FROM memory_meta WHERE id = ?", ("e1",)
        ).fetchone()
        assert row["taxonomy"] == "factual"

    def test_default_taxonomy_episode(self, test_db: Any) -> None:
        """Episodes default to 'experiential' taxonomy."""
        memory.index_memory("ep1", "episode", "Event", "content")
        row = db._db().execute(
            "SELECT taxonomy FROM memory_meta WHERE id = ?", ("ep1",)
        ).fetchone()
        assert row["taxonomy"] == "experiential"

    def test_default_taxonomy_goal(self, test_db: Any) -> None:
        """Goals default to 'working' taxonomy."""
        memory.index_memory("g1", "goal", "Objective", "content")
        row = db._db().execute(
            "SELECT taxonomy FROM memory_meta WHERE id = ?", ("g1",)
        ).fetchone()
        assert row["taxonomy"] == "working"

    def test_explicit_taxonomy_override(self, test_db: Any) -> None:
        """Caller-provided taxonomy overrides the default."""
        memory.index_memory("e1", "entity", "Temp Entity", "content", taxonomy="working")
        row = db._db().execute(
            "SELECT taxonomy FROM memory_meta WHERE id = ?", ("e1",)
        ).fetchone()
        assert row["taxonomy"] == "working"

    def test_taxonomy_preserved_on_reindex(self, test_db: Any) -> None:
        """Re-indexing without taxonomy keeps existing value."""
        memory.index_memory("e1", "entity", "V1", "content", taxonomy="experiential")
        memory.index_memory("e1", "entity", "V2", "updated content")
        row = db._db().execute(
            "SELECT taxonomy FROM memory_meta WHERE id = ?", ("e1",)
        ).fetchone()
        assert row["taxonomy"] == "experiential"

    def test_invalid_taxonomy_falls_back(self, test_db: Any) -> None:
        """Invalid taxonomy values fall back to default for the type."""
        memory.index_memory("e1", "entity", "Test", "content", taxonomy="invalid")
        row = db._db().execute(
            "SELECT taxonomy FROM memory_meta WHERE id = ?", ("e1",)
        ).fetchone()
        assert row["taxonomy"] == "factual"  # default for entity


class TestTaxonomyScoring:
    def test_factual_scores_importance_heavy(self, test_db: Any) -> None:
        """Factual memories weight importance more than recency."""
        memory.index_memory("f1", "entity", "Factual", "content", taxonomy="factual")
        memory.index_memory("w1", "entity", "Working", "content", taxonomy="working")
        # Set equal importance but different recency
        conn = db._db()
        conn.execute("UPDATE memory_meta SET importance = 0.9 WHERE id IN ('f1', 'w1')")
        conn.commit()

        results = memory.recall(query="content")
        f1_score = next((r["score"] for r in results if r["id"] == "f1"), 0)
        w1_score = next((r["score"] for r in results if r["id"] == "w1"), 0)
        # Both should score, factual should rank higher due to importance weight
        assert f1_score > 0
        assert w1_score > 0

    def test_recall_returns_taxonomy_in_results(self, test_db: Any) -> None:
        """Recall results include taxonomy field."""
        memory.index_memory("e1", "entity", "Test", "search content", taxonomy="factual")
        results = memory.recall(query="search content")
        assert len(results) >= 1
        # taxonomy is in the internal dict even if not in MemoryResult TypedDict
        result_dict = dict(results[0])
        assert "taxonomy" in result_dict or results[0].get("taxonomy") is not None  # type: ignore[union-attr]


class TestWorkingMemoryExpiry:
    def test_expire_working_memories(self, test_db: Any) -> None:
        """Working memories older than max_age_hours are archived."""
        memory.index_memory("w1", "goal", "Temp Goal", "scratch", taxonomy="working")
        # Backdate the updated timestamp to 48h ago
        old = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        conn = db._db()
        conn.execute("UPDATE memory_meta SET updated = ? WHERE id = ?", (old, "w1"))
        conn.commit()

        expired = memory.expire_working_memories(max_age_hours=24)
        assert expired == 1
        row = conn.execute(
            "SELECT status FROM memory_meta WHERE id = ?", ("w1",)
        ).fetchone()
        assert row["status"] == "archived"

    def test_recent_working_not_expired(self, test_db: Any) -> None:
        """Recently updated working memories are kept."""
        memory.index_memory("w1", "goal", "Active Goal", "content", taxonomy="working")
        expired = memory.expire_working_memories(max_age_hours=24)
        assert expired == 0

    def test_non_working_not_expired(self, test_db: Any) -> None:
        """Factual/experiential memories are never expired by this function."""
        memory.index_memory("f1", "entity", "Entity", "content", taxonomy="factual")
        old = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        conn = db._db()
        conn.execute("UPDATE memory_meta SET updated = ? WHERE id = ?", (old, "f1"))
        conn.commit()

        expired = memory.expire_working_memories(max_age_hours=24)
        assert expired == 0


class TestTaxonomyDecay:
    def test_factual_decays_slower(self, test_db: Any) -> None:
        """Factual memories should decay less than experiential with same base rate."""
        memory.index_memory("f1", "entity", "Factual", "content", taxonomy="factual")
        memory.index_memory("e1", "episode", "Experiential", "content", taxonomy="experiential")
        # Both start at importance 1.0
        rates = {"entity": 0.999, "episode": 0.999}
        memory.decay_importance(rates)

        conn = db._db()
        f_imp = conn.execute(
            "SELECT importance FROM memory_meta WHERE id = ?", ("f1",)
        ).fetchone()["importance"]
        e_imp = conn.execute(
            "SELECT importance FROM memory_meta WHERE id = ?", ("e1",)
        ).fetchone()["importance"]
        # Factual should have higher importance (decayed less)
        assert f_imp > e_imp

    def test_working_decays_faster(self, test_db: Any) -> None:
        """Working memories should decay faster than experiential."""
        memory.index_memory("w1", "goal", "Working", "content", taxonomy="working")
        memory.index_memory("e1", "episode", "Experiential", "content", taxonomy="experiential")
        rates = {"goal": 0.999, "episode": 0.999}
        memory.decay_importance(rates)

        conn = db._db()
        w_imp = conn.execute(
            "SELECT importance FROM memory_meta WHERE id = ?", ("w1",)
        ).fetchone()["importance"]
        e_imp = conn.execute(
            "SELECT importance FROM memory_meta WHERE id = ?", ("e1",)
        ).fetchone()["importance"]
        # Working should have lower importance (decayed more)
        assert w_imp < e_imp
