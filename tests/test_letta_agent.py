"""Tests for the Letta agent-loop backend (letta_agent.py)."""

from __future__ import annotations

from typing import Any

import pytest

from luke import letta_agent, memory
from luke.config import settings

# ---------------------------------------------------------------------------
# build_letta_context — core-block assembly
# ---------------------------------------------------------------------------


def test_context_none_without_agent_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("HTTP must not be attempted without an agent id")

    monkeypatch.setattr(letta_agent, "_get_core_blocks", boom)
    assert letta_agent.build_letta_context() is None


def test_context_none_on_fetch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "letta_agent_id", "agent-test")

    def fake_blocks(a: str, timeout: float = 10.0) -> list[dict[str, Any]] | None:
        return None

    monkeypatch.setattr(letta_agent, "_get_core_blocks", fake_blocks)
    assert letta_agent.build_letta_context() is None


def test_context_block_order_and_unknown_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """persona leads, operating-rules trails the known set, unknown blocks are
    appended rather than dropped, and read_only renders as an attribute."""
    blocks: list[dict[str, Any]] = [
        {"label": "operating-rules", "value": "rules", "read_only": True},
        {"label": "custom-extra", "value": "extra"},
        {"label": "persona", "value": "I am Luke"},
    ]
    monkeypatch.setattr(settings, "letta_agent_id", "agent-test")

    def fake_blocks(a: str, timeout: float = 10.0) -> list[dict[str, Any]] | None:
        return blocks

    monkeypatch.setattr(letta_agent, "_get_core_blocks", fake_blocks)

    ctx = letta_agent.build_letta_context()
    assert ctx is not None
    persona_at = ctx.index('label="persona"')
    rules_at = ctx.index('label="operating-rules"')
    extra_at = ctx.index('label="custom-extra"')
    assert persona_at < rules_at < extra_at
    assert 'read_only="true"' in ctx
    assert ctx.startswith("<letta-core-memory>")


# ---------------------------------------------------------------------------
# build_recall_injection / compose_letta_turn_input — the retrieval half
# ---------------------------------------------------------------------------


def _seed(mem_id: str, title: str, content: str) -> None:
    memory.index_memory(
        mem_id=mem_id, mem_type="entity", title=title, content=content, tags=[], links=[]
    )


def _hit(mem_id: str) -> dict[str, Any]:
    return {"id": mem_id, "type": "entity", "title": f"title {mem_id}", "score": 1.0}


def test_injection_empty_query_is_none(test_db: Any) -> None:
    assert letta_agent.build_recall_injection("") is None
    assert letta_agent.build_recall_injection("   ") is None


def test_injection_renders_bodies(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed("a", "Alpha", "Alpha body content")

    def fake_recall(**kw: Any) -> list[dict[str, Any]]:
        return [_hit("a")]

    monkeypatch.setattr(memory, "recall", fake_recall)
    inj = letta_agent.build_recall_injection("alpha")
    assert inj is not None
    assert "<retrieved-memories>" in inj
    assert "Alpha body content" in inj
    assert '<mem id="a"' in inj


def test_injection_respects_total_budget(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Highest-ranked passages that fit are kept; the budget is never blown."""
    for i in range(4):
        _seed(f"m{i}", f"Title {i}", "x" * 600)

    def fake_recall(**kw: Any) -> list[dict[str, Any]]:
        return [_hit(f"m{i}") for i in range(4)]

    monkeypatch.setattr(memory, "recall", fake_recall)
    inj = letta_agent.build_recall_injection(
        "query", k=4, per_passage_cap=700, total_char_budget=1500
    )
    assert inj is not None
    assert len(inj) < 1500 + 400  # wrapper header overhead only
    assert '<mem id="m0"' in inj
    assert '<mem id="m3"' not in inj


def test_injection_recall_error_is_none(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kw: Any) -> Any:
        raise RuntimeError("recall exploded")

    monkeypatch.setattr(memory, "recall", boom)
    assert letta_agent.build_recall_injection("query") is None


# --- as_of time anchoring (the 5.1R replay's fairness guarantee) --------------


def _seed_at(mem_id: str, created: str) -> None:
    """Seed a memory and force its creation timestamp, so the cutoff has something to bite."""
    _seed(mem_id, f"Title {mem_id}", f"body of {mem_id}")
    from luke.db import _db

    _db().execute("UPDATE memory_meta SET created = ? WHERE id = ?", (created, mem_id))
    _db().commit()


def test_as_of_drops_memories_that_did_not_exist_yet(
    test_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the anchor: a fact written after the prompt must not reach the arm.

    Without this the reply-diff gate penalises Letta for being *currently* right about
    something that had not happened on the day of the prompt it is replaying.
    """
    _seed_at("old", "2026-06-01T00:00:00+00:00")
    _seed_at("future", "2026-07-20T00:00:00+00:00")

    def fake_recall(**kw: Any) -> list[dict[str, Any]]:
        return [_hit("old"), _hit("future")]

    monkeypatch.setattr(memory, "recall", fake_recall)

    unanchored = letta_agent.build_recall_injection("q")
    assert unanchored is not None
    assert '<mem id="old"' in unanchored and '<mem id="future"' in unanchored

    anchored = letta_agent.build_recall_injection("q", as_of="2026-07-01T00:00:00+00:00")
    assert anchored is not None
    assert '<mem id="old"' in anchored
    assert '<mem id="future"' not in anchored


def test_as_of_drops_hits_with_unreadable_creation_time(
    test_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-safe direction: unfilterable means dropped, never silently kept."""

    def fake_recall(**kw: Any) -> list[dict[str, Any]]:
        return [_hit("ghost-never-indexed")]

    monkeypatch.setattr(memory, "recall", fake_recall)
    assert letta_agent.build_recall_injection("q", as_of="2026-07-01T00:00:00+00:00") is None


def test_compose_carries_as_of_anchor_even_without_hits(
    test_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anchor covers entities created early but rewritten in place later, so it must
    survive the no-injection path rather than riding along with the retrieved block."""

    def fake_recall(**kw: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(memory, "recall", fake_recall)
    out = letta_agent.compose_letta_turn_input("what's the plan?", as_of="2026-07-01T00:00:00+00:00")
    assert "Answer as of 2026-07-01T00:00:00+00:00" in out
    assert "what's the plan?" in out
    # Live turns pass no anchor and must be byte-identical to before.
    assert letta_agent.compose_letta_turn_input("what's the plan?") == "what's the plan?"


def test_compose_passthrough_when_no_hits(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_recall(**kw: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(memory, "recall", fake_recall)
    msg = "what's the plan?"
    assert letta_agent.compose_letta_turn_input(msg) is msg


def test_compose_prepends_injection(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed("a", "Alpha", "Alpha body content")

    def fake_recall(**kw: Any) -> list[dict[str, Any]]:
        return [_hit("a")]

    monkeypatch.setattr(memory, "recall", fake_recall)
    out = letta_agent.compose_letta_turn_input("what's alpha?")
    assert out.startswith("<retrieved-memories>")
    assert out.endswith("what's alpha?")


# ---------------------------------------------------------------------------
# drive_letta_turn — transport guard rails
# ---------------------------------------------------------------------------


def test_drive_errors_without_agent_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("HTTP must not be attempted without an agent id")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    result = letta_agent.drive_letta_turn("hello")
    assert result["error"] == "no letta_agent_id configured"
    assert result["reply"] == ""


def test_drive_transport_error_returns_dict(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "letta_agent_id", "agent-test")

    def fake_recall(**kw: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(memory, "recall", fake_recall)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    result = letta_agent.drive_letta_turn("hello")
    assert result["error"] is not None
    assert "refused" in result["error"]
    assert result["injected"] is False
