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
    monkeypatch.setattr(letta_agent, "_get_core_blocks", lambda a, timeout=10.0: None)
    assert letta_agent.build_letta_context() is None


def test_context_block_order_and_unknown_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """persona leads, operating-rules trails the known set, unknown blocks are
    appended rather than dropped, and read_only renders as an attribute."""
    blocks = [
        {"label": "operating-rules", "value": "rules", "read_only": True},
        {"label": "custom-extra", "value": "extra"},
        {"label": "persona", "value": "I am Luke"},
    ]
    monkeypatch.setattr(settings, "letta_agent_id", "agent-test")
    monkeypatch.setattr(letta_agent, "_get_core_blocks", lambda a, timeout=10.0: blocks)

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
    monkeypatch.setattr(memory, "recall", lambda **kw: [_hit("a")])
    inj = letta_agent.build_recall_injection("alpha")
    assert inj is not None
    assert "<retrieved-memories>" in inj
    assert "Alpha body content" in inj
    assert '<mem id="a"' in inj


def test_injection_respects_total_budget(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Highest-ranked passages that fit are kept; the budget is never blown."""
    for i in range(4):
        _seed(f"m{i}", f"Title {i}", "x" * 600)
    monkeypatch.setattr(memory, "recall", lambda **kw: [_hit(f"m{i}") for i in range(4)])
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


def test_compose_passthrough_when_no_hits(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "recall", lambda **kw: [])
    msg = "what's the plan?"
    assert letta_agent.compose_letta_turn_input(msg) is msg


def test_compose_prepends_injection(test_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed("a", "Alpha", "Alpha body content")
    monkeypatch.setattr(memory, "recall", lambda **kw: [_hit("a")])
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
    monkeypatch.setattr(memory, "recall", lambda **kw: [])

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    result = letta_agent.drive_letta_turn("hello")
    assert result["error"] is not None
    assert "refused" in result["error"]
    assert result["injected"] is False
