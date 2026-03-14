"""Tests for luke.db — memory index, recall, graph, scoring, decay, archiving."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from luke import db
from luke.db import _recency_score

# ---------------------------------------------------------------------------
# Index & Recall (FTS-only path, no embeddings)
# ---------------------------------------------------------------------------


class TestIndexAndRecall:
    def test_fts_roundtrip(self, test_db: Any) -> None:
        db.index_memory(
            "proj-x",
            "entity",
            "Project X",
            "A research project on AI safety",
        )
        results = db.recall(query="research project")
        assert len(results) >= 1
        assert any(r["id"] == "proj-x" for r in results)

    def test_recall_by_type_only(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "Entity One", "content one")
        db.index_memory("ep1", "episode", "Episode One", "content two")
        results = db.recall(mem_type="entity")
        assert all(r["type"] == "entity" for r in results)

    def test_recall_temporal_range(self, test_db: Any) -> None:
        db.index_memory("e1", "episode", "Past Event", "old event")
        # Set updated to a known time
        now = datetime.now(UTC)
        past = (now - timedelta(days=2)).isoformat()
        future = (now + timedelta(days=1)).isoformat()
        results = db.recall(after=past, before=future)
        assert len(results) >= 1

    def test_recall_no_results(self, test_db: Any) -> None:
        results = db.recall(query="xyzzy_nonexistent_12345")
        assert results == []

    def test_recall_fts_injection_safe(self, test_db: Any) -> None:
        """Malformed FTS5 query syntax must not raise."""
        db.index_memory("e1", "entity", "Test", "content")
        results = db.recall(query="OR OR")
        # Should return empty or handle gracefully, not crash
        assert isinstance(results, list)

    def test_recall_fts_special_chars(self, test_db: Any) -> None:
        """Special characters in queries must not crash."""
        db.index_memory("e1", "entity", "Test", "content")
        results = db.recall(query="test AND (OR) NOT *")
        assert isinstance(results, list)

    def test_reindex_preserves_created(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "V1", "first version")
        row1 = db._db().execute("SELECT created FROM memory_meta WHERE id = ?", ("e1",)).fetchone()

        db.index_memory("e1", "entity", "V2", "second version")
        row2 = db._db().execute("SELECT created FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        assert row1["created"] == row2["created"]

    def test_reindex_preserves_access_count(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "V1", "content")
        db.touch_memories(["e1"])
        db.touch_memories(["e1"])
        row1 = (
            db._db()
            .execute("SELECT access_count FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row1["access_count"] == 2

        db.index_memory("e1", "entity", "V2", "updated")
        row2 = (
            db._db()
            .execute("SELECT access_count FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row2["access_count"] == 2

    def test_reindex_preserves_importance(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "V1", "content")
        # Manually set importance
        db._db().execute("UPDATE memory_meta SET importance = 0.5 WHERE id = ?", ("e1",))
        db._db().commit()

        db.index_memory("e1", "entity", "V2", "updated")
        row = (
            db._db().execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        )
        assert row["importance"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------


class TestGraph:
    def test_link_and_recall_related(self, test_db: Any) -> None:
        db.index_memory("a", "entity", "Node A", "content a")
        db.index_memory("b", "entity", "Node B", "content b")
        db.link_memories("a", "b", "related_to")

        results = db.recall(related_to="a")
        assert any(r["id"] == "b" for r in results)

    def test_graph_multi_hop(self, test_db: Any) -> None:
        """A -> B -> C: querying related_to=A should find C (depth 2)."""
        db.index_memory("a", "entity", "Node A", "a")
        db.index_memory("b", "entity", "Node B", "b")
        db.index_memory("c", "entity", "Node C", "c")
        db.link_memories("a", "b", "knows")
        db.link_memories("b", "c", "knows")

        results = db.recall(related_to="a")
        ids = {r["id"] for r in results}
        assert "b" in ids
        assert "c" in ids

    def test_graph_bidirectional(self, test_db: Any) -> None:
        """Links work in both directions."""
        db.index_memory("x", "entity", "X", "x")
        db.index_memory("y", "entity", "Y", "y")
        db.link_memories("x", "y", "linked")

        # Query from y should find x
        results = db.recall(related_to="y")
        assert any(r["id"] == "x" for r in results)

    def test_graph_no_cycles(self, test_db: Any) -> None:
        """Cycles in graph must not cause infinite loops."""
        db.index_memory("a", "entity", "A", "a")
        db.index_memory("b", "entity", "B", "b")
        db.link_memories("a", "b", "knows")
        db.link_memories("b", "a", "knows")

        results = db.recall(related_to="a")
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
        db.index_memory("e1", "entity", "Test Entity", "keyword alpha")
        results = db.recall(query="keyword alpha")
        assert len(results) >= 1
        # Score should include importance/recency/access components (> raw FTS score of ~0-1)
        assert results[0]["score"] > 0


# ---------------------------------------------------------------------------
# Decay & Archiving
# ---------------------------------------------------------------------------


class TestDecayAndArchiving:
    def test_decay_importance(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "Test", "content")
        before = (
            db._db()
            .execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()["importance"]
        )

        updated = db.decay_importance({"entity": 0.99})
        assert updated >= 1

        after = (
            db._db()
            .execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()["importance"]
        )
        assert after < before

    def test_decay_modulated_by_access(self, test_db: Any) -> None:
        """Higher access_count should result in slower decay."""
        db.index_memory("low", "entity", "Low Access", "c")
        db.index_memory("high", "entity", "High Access", "c")

        # Give "high" many accesses
        for _ in range(10):
            db.touch_memories(["high"])

        db.decay_importance({"entity": 0.99})

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
        db.index_memory("e1", "entity", "Test", "c")
        db.touch_memories(["e1"])
        db.touch_memories(["e1"])
        row = (
            db._db()
            .execute("SELECT access_count FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row["access_count"] == 2

    def test_archive_excludes_from_recall(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "Archived", "secret content")
        db.archive_memory("e1")
        results = db.recall(query="secret content")
        assert not any(r["id"] == "e1" for r in results)

    def test_cleanup_archived_fts(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "To Archive", "content")
        db.archive_memory("e1")
        db.cleanup_archived_fts()
        # FTS entry should be gone
        row = db._db().execute("SELECT id FROM memory_fts WHERE id = ?", ("e1",)).fetchone()
        assert row is None


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


class TestConflictDetection:
    def test_detect_title_change(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "Old Title", "content")
        changes = db.detect_changes("e1", "content", "New Title")
        assert len(changes) == 1
        assert "Title" in changes[0]

    def test_detect_content_change(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "Title", "old content")
        changes = db.detect_changes("e1", "new content", "Title")
        assert len(changes) == 1
        assert "Content" in changes[0]

    def test_detect_no_existing(self, test_db: Any) -> None:
        changes = db.detect_changes("nonexistent", "content", "title")
        assert changes == []

    def test_detect_both_changes(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "Old", "old content")
        changes = db.detect_changes("e1", "new content", "New")
        assert len(changes) == 2  # title + content


# ---------------------------------------------------------------------------
# Consolidation candidates
# ---------------------------------------------------------------------------


class TestConsolidation:
    def test_finds_cluster(self, test_db: Any) -> None:
        shared_tags = ["tag1", "tag2", "tag3"]
        for i in range(4):
            db.index_memory(
                f"ep{i}",
                "episode",
                f"Episode {i}",
                f"content {i}",
                tags=shared_tags,
            )
        clusters = db.get_consolidation_candidates(min_shared=3)
        assert len(clusters) >= 1
        assert len(clusters[0]) >= 3

    def test_insufficient_shared_tags(self, test_db: Any) -> None:
        db.index_memory("ep1", "episode", "Ep 1", "c", tags=["a", "b"])
        db.index_memory("ep2", "episode", "Ep 2", "c", tags=["c", "d"])
        db.index_memory("ep3", "episode", "Ep 3", "c", tags=["e", "f"])
        clusters = db.get_consolidation_candidates(min_shared=3)
        assert clusters == []


# ---------------------------------------------------------------------------
# Time window recall
# ---------------------------------------------------------------------------


class TestTimeWindowRecall:
    def test_recall_by_time_window(self, test_db: Any) -> None:
        db.index_memory("e1", "episode", "Event", "happened")
        now = datetime.now(UTC)
        results = db.recall_by_time_window(
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
        db.index_memory("old-ep", "episode", "Old", "old content")
        # Set updated to 6 years ago and low importance
        old_date = (datetime.now(UTC) - timedelta(days=2200)).isoformat()
        db._db().execute(
            "UPDATE memory_meta SET updated = ?, importance = 0.01 WHERE id = ?",
            (old_date, "old-ep"),
        )
        db._db().commit()

        pruned = db.prune_old_fts_entries(1825)  # 5 years
        assert pruned == 1
        # Should be archived
        row = (
            db._db().execute("SELECT status FROM memory_meta WHERE id = ?", ("old-ep",)).fetchone()
        )
        assert row["status"] == "archived"

    def test_prune_skips_recent(self, test_db: Any) -> None:
        db.index_memory("new-ep", "episode", "New", "recent")
        pruned = db.prune_old_fts_entries(1825)
        assert pruned == 0

    def test_prune_disabled_when_zero(self, test_db: Any) -> None:
        assert db.prune_old_fts_entries(0) == 0

    def test_prune_skips_high_importance(self, test_db: Any) -> None:
        db.index_memory("imp-ep", "episode", "Important", "vital")
        old_date = (datetime.now(UTC) - timedelta(days=2200)).isoformat()
        db._db().execute(
            "UPDATE memory_meta SET updated = ?, importance = 0.5 WHERE id = ?",
            (old_date, "imp-ep"),
        )
        db._db().commit()
        pruned = db.prune_old_fts_entries(1825)
        assert pruned == 0  # High importance preserved


class TestRestoreMemory:
    def test_restore_archived(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "Test", "content")
        db.archive_memory("e1")
        # Verify archived
        results = db.recall(query="content")
        assert not any(r["id"] == "e1" for r in results)

        # Restore
        restored = db.restore_memory("e1")
        assert restored is True
        # Verify active again
        row = db._db().execute("SELECT status FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        assert row["status"] == "active"

    def test_restore_nonexistent_returns_false(self, test_db: Any) -> None:
        assert db.restore_memory("nonexistent") is False


class TestUpdateMemoryTags:
    def test_update_tags(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "Test", "content", tags=["old"])
        db.update_memory_tags("e1", ["new", "updated"])
        row = db._db().execute("SELECT tags_json FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        import json

        assert json.loads(row["tags_json"]) == ["new", "updated"]


class TestPrivateMemoryTag:
    def test_private_excluded_from_recall(self, test_db: Any) -> None:
        db.index_memory(
            "secret",
            "entity",
            "Secret",
            "private stuff",
            tags=["private"],
        )
        db.index_memory("public", "entity", "Public", "public stuff")
        results = db.recall(query="stuff")
        ids = {r["id"] for r in results}
        assert "public" in ids
        assert "secret" not in ids

    def test_private_included_with_override(self, test_db: Any) -> None:
        db.index_memory(
            "secret",
            "entity",
            "Secret",
            "hidden data",
            tags=["private"],
        )
        results = db.recall(query="hidden data", include_private=True)
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
        db.sync_memory_index()
        results = db.recall(query="Test Entity")
        assert any(r["id"] == "test-entity" for r in results)

    def test_skips_already_indexed(self, test_db: Any, tmp_settings: Any) -> None:
        """Already-indexed files should not be re-indexed."""
        db.index_memory("existing", "entity", "Existing", "content")
        mem_dir = tmp_settings.memory_dir / "entities"
        (mem_dir / "existing.md").write_text(
            "---\nid: existing\ntype: entity\ntags: []\nlinks: []\n---\n\n# Existing\n\ncontent"
        )
        # Should not raise or duplicate
        db.sync_memory_index()
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
        db.index_memory("relevant", "entity", "Alpha Project", "alpha project details")
        db.index_memory("irrelevant", "entity", "Beta Unrelated", "completely different topic")
        results = db.recall(query="alpha project")
        # The relevant match should be first
        if len(results) >= 2:
            ids = [r["id"] for r in results]
            assert ids.index("relevant") < ids.index("irrelevant")

    def test_zero_relevance_dampened(self, test_db: Any) -> None:
        """Non-query results (temporal) should have dampened scores."""
        db.index_memory("e1", "entity", "Test", "content")
        # Temporal-only query (no text query, so relevance = 0)
        results = db.recall(after="2020-01-01", before="2030-01-01")
        assert len(results) >= 1
        # Score should be dampened (< 0.5 since it's context_norm * 0.3)
        assert results[0]["score"] < 0.5


# ---------------------------------------------------------------------------
# Graph neighbors
# ---------------------------------------------------------------------------


class TestGraphNeighbors:
    def test_batch_neighbors(self, test_db: Any) -> None:
        db.index_memory("a", "entity", "A", "a")
        db.index_memory("b", "entity", "B", "b")
        db.index_memory("c", "entity", "C", "c")
        db.link_memories("a", "b", "knows")
        db.link_memories("a", "c", "works_with")
        neighbors = db.get_graph_neighbors(["a"])
        ids = {n["id"] for n in neighbors}
        assert "b" in ids
        assert "c" in ids

    def test_excludes_input_ids(self, test_db: Any) -> None:
        db.index_memory("a", "entity", "A", "a")
        db.index_memory("b", "entity", "B", "b")
        db.link_memories("a", "b", "knows")
        neighbors = db.get_graph_neighbors(["a", "b"])
        ids = {n["id"] for n in neighbors}
        assert "a" not in ids
        assert "b" not in ids

    def test_empty_input(self, test_db: Any) -> None:
        assert db.get_graph_neighbors([]) == []

    def test_respects_limit(self, test_db: Any) -> None:
        db.index_memory("hub", "entity", "Hub", "hub")
        for i in range(5):
            db.index_memory(f"spoke{i}", "entity", f"Spoke {i}", f"spoke {i}")
            db.link_memories("hub", f"spoke{i}", "connected")
        neighbors = db.get_graph_neighbors(["hub"], limit=2)
        assert len(neighbors) <= 2


# ---------------------------------------------------------------------------
# Importance initialization
# ---------------------------------------------------------------------------


class TestImportanceInit:
    def test_custom_importance(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "Important", "vital info", importance=1.5)
        row = (
            db._db().execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        )
        assert row["importance"] == pytest.approx(1.5)

    def test_default_importance(self, test_db: Any) -> None:
        db.index_memory("e1", "entity", "Normal", "regular info")
        row = (
            db._db().execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        )
        assert row["importance"] == pytest.approx(1.0)

    def test_reindex_keeps_existing_importance(self, test_db: Any) -> None:
        """Re-indexing without explicit importance preserves existing value."""
        db.index_memory("e1", "entity", "V1", "c", importance=1.8)
        db.index_memory("e1", "entity", "V2", "c")  # no importance → keep 1.8
        row = (
            db._db().execute("SELECT importance FROM memory_meta WHERE id = ?", ("e1",)).fetchone()
        )
        assert row["importance"] == pytest.approx(1.8)

    def test_reindex_overrides_with_explicit(self, test_db: Any) -> None:
        """Re-indexing with explicit importance overrides existing."""
        db.index_memory("e1", "entity", "V1", "c", importance=1.8)
        db.index_memory("e1", "entity", "V2", "c", importance=0.5)
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
        db.index_memory("e1", "entity", "Alpha Project", "alpha details here")
        db.index_memory("e2", "entity", "Beta Project", "beta details here")
        r1 = db.recall(query="alpha")
        r2 = db.recall(query="alpha beta")
        assert len(r2) >= len(r1)

    def test_special_chars_stripped(self, test_db: Any) -> None:
        """FTS5 operators in natural text should not crash or alter semantics."""
        db.index_memory("e1", "entity", "Test", "not a problem at all")
        results = db.recall(query='is this NOT a "problem"?')
        assert isinstance(results, list)

    def test_sanitize_empty_after_strip(self, test_db: Any) -> None:
        """If sanitization produces empty string, no FTS crash."""
        results = db.recall(query="- * + ()")
        assert results == []

    def test_colons_and_hyphens_stripped(self, test_db: Any) -> None:
        """Colons and hyphens are FTS5 column/prefix syntax — must be stripped."""
        db.index_memory("e1", "entity", "Test", "hello world")
        # These would crash FTS5 without sanitization
        results = db.recall(query="test:colon")
        assert isinstance(results, list)
        results = db.recall(query="--double-dash")
        assert isinstance(results, list)

    def test_single_char_terms_dropped(self, test_db: Any) -> None:
        """Single-character terms are dropped to avoid noise."""
        from luke.db import _sanitize_fts_query

        assert _sanitize_fts_query("I am a test") == "am OR test"


# ---------------------------------------------------------------------------
# Bounded importance in scoring
# ---------------------------------------------------------------------------


class TestBoundedImportance:
    def test_high_importance_clamped_in_scoring(self, test_db: Any) -> None:
        """Importance > 1.0 should be clamped to 1.0 in composite scoring."""
        db.index_memory("high", "entity", "High Imp", "keyword stuff", importance=2.0)
        db.index_memory("normal", "entity", "Normal Imp", "keyword stuff", importance=1.0)
        results = db.recall(query="keyword stuff")
        scores = {m["id"]: m["score"] for m in results}
        # Both have importance clamped to 1.0, so scores should be close
        if "high" in scores and "normal" in scores:
            assert scores["high"] == pytest.approx(scores["normal"], abs=0.1)

    def test_stored_importance_not_clamped(self, test_db: Any) -> None:
        """Importance > 1.0 should still be stored for decay runway."""
        db.index_memory("e1", "entity", "Test", "c", importance=2.0)
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
        db.index_memory("e1", "entity", "Test", "content")
        db.touch_memories(["e1"])
        row = (
            db._db()
            .execute("SELECT access_count, useful_count FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row["access_count"] == 1
        assert row["useful_count"] == 1

    def test_speculative_touch_only_access(self, test_db: Any) -> None:
        """Speculative touch increments access_count only, not useful_count."""
        db.index_memory("e1", "entity", "Test", "content")
        db.touch_memories(["e1"], useful=False)
        row = (
            db._db()
            .execute("SELECT access_count, useful_count FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row["access_count"] == 1
        assert row["useful_count"] == 0

    def test_last_accessed_updated(self, test_db: Any) -> None:
        """Both touch functions update last_accessed timestamp."""
        db.index_memory("e1", "entity", "Test", "content")
        db.touch_memories(["e1"])
        row = (
            db._db()
            .execute("SELECT last_accessed FROM memory_meta WHERE id = ?", ("e1",))
            .fetchone()
        )
        assert row["last_accessed"] != ""

    def test_utility_affects_scoring(self, test_db: Any) -> None:
        """Memory with low utility rate should score lower than high utility."""
        db.index_memory("low", "entity", "Low Utility", "keyword test")
        db.index_memory("high", "entity", "High Utility", "keyword test")
        for _ in range(10):
            db.touch_memories(["low"], useful=False)  # 0 useful / 10 access
            db.touch_memories(["high"])  # 10 useful / 10 access
        results = db.recall(query="keyword test")
        scores = {m["id"]: m["score"] for m in results}
        if "low" in scores and "high" in scores:
            assert scores["high"] > scores["low"]

    def test_reindex_preserves_useful_count(self, test_db: Any) -> None:
        """Re-indexing should preserve useful_count."""
        db.index_memory("e1", "entity", "V1", "content")
        db.touch_memories(["e1"])
        db.touch_memories(["e1"])
        db.index_memory("e1", "entity", "V2", "updated content")
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
        db.index_memory("a", "entity", "A", "a")
        db.index_memory("b", "entity", "B", "b")
        db.link_memories("a", "b", "related")
        before = (
            db._db()
            .execute(
                "SELECT weight FROM memory_links WHERE from_id = ? AND to_id = ?",
                ("a", "b"),
            )
            .fetchone()["weight"]
        )
        db.touch_memories(["a", "b"])
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
        db.index_memory("a", "entity", "A", "a")
        db.index_memory("b", "entity", "B", "b")
        db.link_memories("a", "b", "related")
        db._db().execute(
            "UPDATE memory_links SET weight = 4.98 WHERE from_id = ? AND to_id = ?",
            ("a", "b"),
        )
        db._db().commit()
        db.touch_memories(["a", "b"])
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
        db.index_memory("x", "entity", "X", "x")
        db.index_memory("y", "entity", "Y", "y")
        db.touch_memories(["x", "y"])
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
        db.index_memory("a", "entity", "A", "a")
        db.index_memory("b", "entity", "B", "b")
        db.link_memories("a", "b", "related")
        db._db().execute(
            "UPDATE memory_links SET weight = 3.0 WHERE from_id = ? AND to_id = ?",
            ("a", "b"),
        )
        db._db().commit()
        db.link_memories("a", "b", "related")  # re-link same relationship
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
        db.record_memory_change("e1", ["Title: 'Old' -> 'New'", "Content updated"])
        history = db.get_memory_history("e1")
        assert len(history) == 1
        assert len(history[0]["changes"]) == 2

    def test_history_ordered_desc(self, test_db: Any) -> None:
        db.record_memory_change("e1", ["Change 1"])
        db.record_memory_change("e1", ["Change 2"])
        history = db.get_memory_history("e1")
        assert len(history) == 2
        assert history[0]["changes"] == ["Change 2"]

    def test_no_history_returns_empty(self, test_db: Any) -> None:
        assert db.get_memory_history("nonexistent") == []

    def test_empty_changes_not_recorded(self, test_db: Any) -> None:
        db.record_memory_change("e1", [])
        assert db.get_memory_history("e1") == []
