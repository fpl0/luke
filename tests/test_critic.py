"""Tests for luke.critic — outbound message critic (F4)."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator
from typing import Any

import pytest

from luke import critic
from luke.critic import CriticVerdict, _parse_verdict, critique_outbound

# ---------------------------------------------------------------------------
# CriticVerdict dataclass
# ---------------------------------------------------------------------------


class TestCriticVerdict:
    def test_is_frozen(self) -> None:
        v = CriticVerdict("pass", "ok")
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.decision = "block"  # type: ignore[misc]

    def test_has_slots(self) -> None:
        v = CriticVerdict("pass", "ok")
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            v.extra = "nope"  # type: ignore[attr-defined]

    def test_equality(self) -> None:
        a = CriticVerdict("pass", "ok")
        b = CriticVerdict("pass", "ok")
        assert a == b


# ---------------------------------------------------------------------------
# _parse_verdict
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_pass(self) -> None:
        v = _parse_verdict("DECISION: pass")
        assert v.decision == "pass"
        assert v.reason == ""

    def test_revise_with_reason(self) -> None:
        v = _parse_verdict("DECISION: revise tone too cheery")
        assert v.decision == "revise"
        assert v.reason == "tone too cheery"

    def test_block_with_reason(self) -> None:
        v = _parse_verdict("DECISION: block uses 'Absolutely!'")
        assert v.decision == "block"
        assert v.reason == "uses 'Absolutely!'"

    def test_case_insensitive(self) -> None:
        v = _parse_verdict("decision: PASS")
        assert v.decision == "pass"

    def test_unparseable_fails_open(self) -> None:
        v = _parse_verdict("the model wandered off-script entirely")
        assert v.decision == "pass"
        assert "critic-error" in v.reason

    def test_empty_string_fails_open(self) -> None:
        v = _parse_verdict("")
        assert v.decision == "pass"

    def test_reason_truncated_to_200(self) -> None:
        long_reason = "x" * 500
        v = _parse_verdict(f"DECISION: revise {long_reason}")
        assert len(v.reason) <= 200


# ---------------------------------------------------------------------------
# critique_outbound — stubs SDK query
# ---------------------------------------------------------------------------


def _make_fake_query(response_text: str) -> Any:
    """Build a fake `query` coroutine that yields one AssistantMessage."""
    from claude_agent_sdk import AssistantMessage, TextBlock

    async def _fake(
        *,
        prompt: str | Any,
        options: Any = None,
        transport: Any = None,
    ) -> AsyncIterator[Any]:
        yield AssistantMessage(
            content=[TextBlock(text=response_text)],
            model="haiku",
        )

    return _fake


def _make_failing_query(exc: BaseException) -> Any:
    async def _fake(
        *,
        prompt: str | Any,
        options: Any = None,
        transport: Any = None,
    ) -> AsyncIterator[Any]:
        if False:  # make this an async generator
            yield
        raise exc

    return _fake


def _make_hanging_query() -> Any:
    async def _fake(
        *,
        prompt: str | Any,
        options: Any = None,
        transport: Any = None,
    ) -> AsyncIterator[Any]:
        await asyncio.sleep(60)  # longer than any test timeout
        if False:
            yield

    return _fake


class TestCritiqueOutbound:
    async def test_pass_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(critic, "query", _make_fake_query("DECISION: pass"))
        v = await critique_outbound("Heads up, your 3pm moved.", {"tool": "send"})
        assert v.decision == "pass"

    async def test_revise_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            critic,
            "query",
            _make_fake_query("DECISION: revise tone too chipper"),
        )
        v = await critique_outbound(
            "Absolutely! Great question — heads up your 3pm moved.",
            {"tool": "send"},
        )
        assert v.decision == "revise"
        assert "chipper" in v.reason

    async def test_block_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            critic,
            "query",
            _make_fake_query("DECISION: block filler boilerplate"),
        )
        v = await critique_outbound(
            "I apologize for the inconvenience.",
            {"tool": "send"},
        )
        assert v.decision == "block"
        assert "filler" in v.reason

    async def test_network_error_fails_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            critic, "query", _make_failing_query(ConnectionError("boom"))
        )
        v = await critique_outbound("Heads up, your 3pm moved.", {"tool": "send"})
        assert v.decision == "pass"
        assert "critic-error" in v.reason
        assert "ConnectionError" in v.reason

    async def test_parse_error_fails_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Model returns gibberish — should fall through to pass.
        monkeypatch.setattr(
            critic, "query", _make_fake_query("hmm, hard to say")
        )
        v = await critique_outbound("Heads up, your 3pm moved.", {"tool": "send"})
        assert v.decision == "pass"
        assert "critic-error" in v.reason

    async def test_timeout_fails_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from luke.config import settings

        monkeypatch.setattr(critic, "query", _make_hanging_query())
        monkeypatch.setattr(settings, "critic_timeout_s", 0.05)
        v = await critique_outbound("Heads up, your 3pm moved.", {"tool": "send"})
        assert v.decision == "pass"
        assert "timeout" in v.reason
