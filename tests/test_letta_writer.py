"""Tests for the Letta write-through mirror (letta_writer.py)."""

from __future__ import annotations

from typing import Any

import pytest

from luke import letta_writer, memory
from luke.config import settings


def _seed(mem_id: str, title: str = "T", content: str = "Body text") -> None:
    memory.index_memory(
        mem_id=mem_id, mem_type="entity", title=title, content=content, tags=["t1"], links=[]
    )


@pytest.fixture()
def capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture create/delete calls; pretend no stale passages exist."""
    calls: dict[str, Any] = {"created": [], "deleted": [], "found": []}

    def fake_find(surface: str, luke_id: str) -> list[str]:
        calls["found"].append(luke_id)
        return list(calls.get("stale", []))

    def fake_delete(pid: str) -> None:
        calls["deleted"].append(pid)

    def fake_create(text: str, metadata: dict[str, Any]) -> None:
        calls["created"].append((text, metadata))

    monkeypatch.setattr(letta_writer, "_find_passage_ids", fake_find)
    monkeypatch.setattr(letta_writer, "_delete_passage", fake_delete)
    monkeypatch.setattr(letta_writer, "_create_passage", fake_create)
    return calls


def _letta_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "memory_backend", "letta")
    monkeypatch.setattr(settings, "letta_archive_id", "archive-test")
    monkeypatch.setattr(settings, "letta_write_through", True)


def test_noop_off_letta_backend(test_db: Any, capture: dict[str, Any]) -> None:
    _seed("m1")
    letta_writer.letta_write_through("m1")
    assert capture["created"] == []


def test_noop_without_archive_id(
    test_db: Any, capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "memory_backend", "letta")
    monkeypatch.setattr(settings, "letta_write_through", True)
    _seed("m1")
    letta_writer.letta_write_through("m1")
    assert capture["created"] == []


def test_mirrors_committed_row(
    test_db: Any, capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed("m1", title="My title", content="My content")
    _letta_on(monkeypatch)
    letta_writer.letta_write_through("m1")
    assert len(capture["created"]) == 1
    text, meta = capture["created"][0]
    assert text == "My title My content"
    assert meta["luke_id"] == "m1"
    assert meta["status"] == "active"
    assert meta["is_tombstone"] is False
    assert meta["content_source"] == "live_write_through"


def test_upsert_deletes_stale_copies(
    test_db: Any, capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed("m1")
    _letta_on(monkeypatch)
    capture["stale"] = ["p-old-1", "p-old-2"]
    letta_writer.letta_write_through("m1")
    assert capture["deleted"] == ["p-old-1", "p-old-2"]
    assert len(capture["created"]) == 1


def test_links_mirror_live_graph(
    test_db: Any, capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The passage's links metadata comes from memory_links (the live graph),
    including edge expiry — not the creation-time links_json snapshot."""
    _seed("m1")
    _seed("m2")
    memory.link_memories("m1", "m2", "related")
    _letta_on(monkeypatch)

    letta_writer.letta_write_through("m1")
    _, meta = capture["created"][-1]
    assert meta["links"] == ["m2"]

    memory.invalidate_link("m1", "m2", "related")
    letta_writer.letta_write_through("m1")
    _, meta = capture["created"][-1]
    assert meta["links"] == []


def test_invalidate_link_re_mirrors_both_endpoints(
    test_db: Any, capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed("m1")
    _seed("m2")
    memory.link_memories("m1", "m2", "related")
    _letta_on(monkeypatch)

    capture["created"].clear()
    assert memory.invalidate_link("m1", "m2", "related")
    mirrored = {meta["luke_id"] for _, meta in capture["created"]}
    assert mirrored == {"m1", "m2"}


def test_never_raises_on_create_failure(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed("m1")
    _letta_on(monkeypatch)

    def fake_find(surface: str, luke_id: str) -> list[str]:
        return []

    monkeypatch.setattr(letta_writer, "_find_passage_ids", fake_find)

    def boom(text: str, metadata: dict[str, Any]) -> None:
        raise OSError("letta down")

    monkeypatch.setattr(letta_writer, "_create_passage", boom)
    letta_writer.letta_write_through("m1")  # must not raise


def test_contentless_row_not_mirrored(
    test_db: Any, capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed("m1", content="")
    _letta_on(monkeypatch)
    letta_writer.letta_write_through("m1")
    assert capture["created"] == []
