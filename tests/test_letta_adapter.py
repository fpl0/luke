"""Tests for the Letta semantic-recall adapter (letta_adapter.py)."""

from __future__ import annotations

from typing import Any

import pytest

from luke import letta_adapter, memory
from luke.config import settings


def _seed(mem_id: str, title: str, content: str, mem_type: str = "entity") -> None:
    memory.index_memory(
        mem_id=mem_id, mem_type=mem_type, title=title, content=content, tags=[], links=[]
    )


def _passage(luke_id: str, **meta: Any) -> dict[str, Any]:
    return {"text": f"passage for {luke_id}", "metadata": {"luke_id": luke_id, **meta}}


def test_disabled_without_archive_id(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty letta_archive_id disables the letta path entirely — no HTTP attempted."""

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("HTTP must not be attempted without an archive id")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert letta_adapter.letta_semantic_search("anything") is None


def test_server_error_returns_none(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Any transport failure falls back (None) instead of raising."""
    monkeypatch.setattr(settings, "letta_archive_id", "archive-test")

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert letta_adapter.letta_semantic_search("anything") is None


def test_join_filters_and_ranking(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tombstones/archived passages are skipped, unknown ids dropped by the sqlite
    join, duplicates deduped, and rank position becomes the similarity proxy."""
    monkeypatch.setattr(settings, "letta_archive_id", "archive-test")
    _seed("alpha", "Alpha title", "Alpha content")
    _seed("beta", "Beta title", "Beta content")

    passages = [
        _passage("ghost", is_tombstone=True),  # tombstone — skipped
        _passage("alpha", status="archived"),  # archived copy — skipped
        _passage("alpha"),  # rank 0
        _passage("alpha"),  # duplicate — deduped
        _passage("unknown-id"),  # rank 1, but not in memory_meta — dropped by join
        _passage("beta"),  # rank 2
    ]

    def fake_search(q: str, n: int) -> list[dict[str, Any]]:
        return passages

    monkeypatch.setattr(letta_adapter, "_search_passages", fake_search)

    results = letta_adapter.letta_semantic_search("alpha")
    assert results is not None
    ids = [r["id"] for r in results]
    assert ids == ["alpha", "beta"]
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["title"] == "Alpha title"


def test_all_tombstones_falls_back(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "letta_archive_id", "archive-test")
    passages = [_passage("x", is_tombstone=True), _passage("y", is_tombstone=True)]

    def fake_search(q: str, n: int) -> list[dict[str, Any]]:
        return passages

    monkeypatch.setattr(letta_adapter, "_search_passages", fake_search)
    assert letta_adapter.letta_semantic_search("anything") is None


def test_mem_type_filter(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "letta_archive_id", "archive-test")
    _seed("ep1", "An episode", "Episode body", mem_type="episode")
    _seed("en1", "An entity", "Entity body", mem_type="entity")
    passages = [_passage("ep1"), _passage("en1")]

    def fake_search(q: str, n: int) -> list[dict[str, Any]]:
        return passages

    monkeypatch.setattr(letta_adapter, "_search_passages", fake_search)

    results = letta_adapter.letta_semantic_search("body", mem_type="episode")
    assert results is not None
    assert [r["id"] for r in results] == ["ep1"]


def test_recall_uses_letta_candidates(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: recall() on the letta backend sources candidates via the adapter."""
    monkeypatch.setattr(settings, "memory_backend", "letta")
    monkeypatch.setattr(settings, "letta_archive_id", "archive-test")
    monkeypatch.setattr(settings, "letta_write_through", False)
    _seed("gamma", "Gamma fact", "A distinctive gamma fact body")

    def fake_search(q: str, n: int) -> list[dict[str, Any]]:
        return [_passage("gamma")]

    monkeypatch.setattr(letta_adapter, "_search_passages", fake_search)
    results = memory.recall(query="distinctive gamma fact")
    assert any(r["id"] == "gamma" for r in results)
