"""Tests for luke.critic — outbound message critic (F4)."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from typing import Any

import pytest

from luke import critic
from luke.critic import (
    _FRESHNESS_SYSTEM_PROMPT,
    CriticVerdict,
    _critic_system_prompt,
    _parse_verdict,
    check_freshness,
    critique_outbound,
)

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

    async def test_network_error_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(critic, "query", _make_failing_query(ConnectionError("boom")))
        v = await critique_outbound("Heads up, your 3pm moved.", {"tool": "send"})
        assert v.decision == "pass"
        assert "critic-error" in v.reason
        assert "ConnectionError" in v.reason

    async def test_parse_error_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Model returns gibberish — should fall through to pass.
        monkeypatch.setattr(critic, "query", _make_fake_query("hmm, hard to say"))
        v = await critique_outbound("Heads up, your 3pm moved.", {"tool": "send"})
        assert v.decision == "pass"
        assert "critic-error" in v.reason

    async def test_timeout_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from luke.config import settings

        monkeypatch.setattr(critic, "query", _make_hanging_query())
        monkeypatch.setattr(settings, "critic_timeout_s", 0.05)
        v = await critique_outbound("Heads up, your 3pm moved.", {"tool": "send"})
        assert v.decision == "pass"
        assert "timeout" in v.reason


# ---------------------------------------------------------------------------
# Retry before failing open — a single transient blip must not become "pass".
#
# Regression cover for the defect found 2026-08-14: the fail-open rate was
# diagnosed on 2026-05-16, 2026-07-05 and 2026-08-05, and every pass made it
# more VISIBLE while leaving it exactly as frequent, because retry was never
# added. These tests fail if anyone reverts to single-attempt.
# ---------------------------------------------------------------------------


def _make_scripted_query(script: list[Any]) -> tuple[Any, list[int]]:
    """Fake query replaying `script` per call; returns it and a call counter.

    Each entry is either an exception to raise or a string to yield.
    """
    from claude_agent_sdk import AssistantMessage, TextBlock

    calls = [0]

    async def _fake(
        *,
        prompt: str | Any,
        options: Any = None,
        transport: Any = None,
    ) -> AsyncIterator[Any]:
        idx = calls[0]
        calls[0] += 1
        step = script[min(idx, len(script) - 1)]
        if isinstance(step, BaseException):
            raise step
        yield AssistantMessage(content=[TextBlock(text=step)], model="haiku")

    return _fake, calls


class TestGateRetriesBeforeFailingOpen:
    async def test_network_error_then_success_is_judged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake, calls = _make_scripted_query([ConnectionError("boom"), "DECISION: block filler"])
        monkeypatch.setattr(critic, "query", fake)
        v = await critique_outbound("I apologize for the inconvenience.", {"tool": "s"})
        assert v.decision == "block", "a retryable blip must not become a pass"
        assert "critic-error" not in v.reason
        assert calls[0] == 2

    async def test_unparseable_then_success_is_judged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake, calls = _make_scripted_query(["hmm, hard to say", "DECISION: revise too stiff"])
        monkeypatch.setattr(critic, "query", fake)
        v = await critique_outbound("Heads up, your 3pm moved.", {"tool": "s"})
        assert v.decision == "revise"
        assert calls[0] == 2

    async def test_unparseable_retry_carries_the_format_nudge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []
        from claude_agent_sdk import AssistantMessage, TextBlock

        async def _fake(
            *, prompt: str | Any, options: Any = None, transport: Any = None
        ) -> AsyncIterator[Any]:
            seen.append(str(prompt))
            text = "nope" if len(seen) == 1 else "DECISION: pass"
            yield AssistantMessage(content=[TextBlock(text=text)], model="haiku")

        monkeypatch.setattr(critic, "query", _fake)
        await critique_outbound("Heads up.", {"tool": "s"})
        assert "could not be parsed" not in seen[0]
        assert "could not be parsed" in seen[1], "retry should tighten the format ask"

    async def test_timeout_then_success_is_judged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from claude_agent_sdk import AssistantMessage, TextBlock

        from luke.config import settings

        calls = [0]

        async def _fake(
            *, prompt: str | Any, options: Any = None, transport: Any = None
        ) -> AsyncIterator[Any]:
            calls[0] += 1
            if calls[0] == 1:
                await asyncio.sleep(60)
            yield AssistantMessage(
                content=[TextBlock(text="DECISION: block off-voice")], model="haiku"
            )

        monkeypatch.setattr(critic, "query", _fake)
        monkeypatch.setattr(settings, "critic_timeout_s", 0.05)
        v = await critique_outbound("Absolutely!", {"tool": "s"})
        assert v.decision == "block"
        assert calls[0] == 2

    async def test_exhausting_attempts_still_fails_open(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Terminal policy is unchanged: a dead critic never silences Luke.
        fake, calls = _make_scripted_query([ConnectionError("boom")])
        monkeypatch.setattr(critic, "query", fake)
        v = await critique_outbound("Heads up.", {"tool": "s"})
        assert v.decision == "pass"
        assert "ConnectionError" in v.reason
        assert calls[0] == 2, "must stop at critic_attempts, not loop forever"

    async def test_attempts_setting_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from luke.config import settings

        fake, calls = _make_scripted_query([ConnectionError("boom")])
        monkeypatch.setattr(critic, "query", fake)
        monkeypatch.setattr(settings, "critic_attempts", 3)
        v = await critique_outbound("Heads up.", {"tool": "s"})
        assert v.decision == "pass"
        assert calls[0] == 3

    async def test_single_attempt_config_is_respected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from luke.config import settings

        fake, calls = _make_scripted_query([ConnectionError("boom")])
        monkeypatch.setattr(critic, "query", fake)
        monkeypatch.setattr(settings, "critic_attempts", 1)
        await critique_outbound("Heads up.", {"tool": "s"})
        assert calls[0] == 1

    async def test_freshness_gate_retries_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Both gates share _judge; freshness must not be left single-attempt.
        fake, calls = _make_scripted_query(
            [TimeoutError(), "DECISION: block answers a cancelled question"]
        )
        monkeypatch.setattr(critic, "query", fake)
        v = await check_freshness(
            "About that dentist appointment...",
            [{"sender_name": "Filipe", "content": "cancelled it, never mind"}],
        )
        assert v.decision == "block"
        assert calls[0] == 2


# ---------------------------------------------------------------------------
# check_freshness — stubs SDK query, compares drafts vs user-latest
# ---------------------------------------------------------------------------


def _make_prompt_capturing_query(response_text: str) -> tuple[Any, list[str]]:
    """Return a fake query and a list that accumulates prompt arguments."""
    from claude_agent_sdk import AssistantMessage, TextBlock

    captured: list[str] = []

    async def _fake(
        *,
        prompt: str | Any,
        options: Any = None,
        transport: Any = None,
    ) -> AsyncIterator[Any]:
        captured.append(prompt if isinstance(prompt, str) else str(prompt))
        yield AssistantMessage(
            content=[TextBlock(text=response_text)],
            model="haiku",
        )

    return _fake, captured


class TestCheckFreshness:
    async def test_pass_when_aligned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(critic, "query", _make_fake_query("DECISION: pass"))
        user_msgs = [
            {
                "sender_name": "Filipe",
                "content": "what time is the meeting?",
                "timestamp": "2026-05-13T10:00:00+00:00",
            }
        ]
        v = await check_freshness("The meeting is at 3pm in Vega room.", user_msgs)
        assert v.decision == "pass"

    async def test_block_on_retraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            critic,
            "query",
            _make_fake_query("DECISION: block draft answers a cancelled question"),
        )
        user_msgs = [
            {
                "sender_name": "Filipe",
                "content": "what time is the meeting?",
                "timestamp": "2026-05-13T10:00:00+00:00",
            },
            {
                "sender_name": "Filipe",
                "content": "never mind, found it",
                "timestamp": "2026-05-13T10:00:30+00:00",
            },
        ]
        v = await check_freshness("The meeting is at 3pm in Vega room upstairs.", user_msgs)
        assert v.decision == "block"
        assert "cancelled" in v.reason

    async def test_revise_on_stale_question(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            critic,
            "query",
            _make_fake_query("DECISION: revise answers earlier question, ignores latest"),
        )
        user_msgs = [
            {
                "sender_name": "Filipe",
                "content": "actually, what about tomorrow?",
                "timestamp": "2026-05-13T10:01:00+00:00",
            }
        ]
        v = await check_freshness("Today's meeting is at 3pm in Vega room.", user_msgs)
        assert v.decision == "revise"
        assert "earlier" in v.reason

    async def test_block_on_emotional_steamroll(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The Jul-24 miss: Filipe shares a raw last-day voice note and the
        # queued structured Friday review lands on top of it. The freshness
        # gate must block a structure-first draft over a live emotional share.
        monkeypatch.setattr(
            critic,
            "query",
            _make_fake_query("DECISION: block leads with structure over a raw emotional share"),
        )
        user_msgs = [
            {
                "sender_name": "Filipe",
                "content": (
                    "today was my last day, I dropped my laptop in the office "
                    "already, said goodbye to everybody"
                ),
                "timestamp": "2026-07-24T16:21:00+00:00",
            }
        ]
        v = await check_freshness(
            "<b>Friday review</b>\nWhat you did this week / Mood + hours ...",
            user_msgs,
        )
        assert v.decision == "block"
        assert "emotional" in v.reason

    def test_freshness_prompt_covers_emotional_steamroll(self) -> None:
        # Lock the behavioral instruction in place so it can't silently regress.
        p = _FRESHNESS_SYSTEM_PROMPT.lower()
        assert "emotional" in p
        assert "steamroll" in p
        assert "structure" in p and "presence" in p

    async def test_network_error_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(critic, "query", _make_failing_query(ConnectionError("boom")))
        user_msgs = [
            {
                "sender_name": "Filipe",
                "content": "hey",
                "timestamp": "2026-05-13T10:00:00+00:00",
            }
        ]
        v = await check_freshness("Heads up, your 3pm moved.", user_msgs)
        assert v.decision == "pass"
        assert "critic-error" in v.reason
        assert "ConnectionError" in v.reason

    async def test_timeout_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from luke.config import settings

        monkeypatch.setattr(critic, "query", _make_hanging_query())
        monkeypatch.setattr(settings, "critic_timeout_s", 0.05)
        user_msgs = [
            {
                "sender_name": "Filipe",
                "content": "hey",
                "timestamp": "2026-05-13T10:00:00+00:00",
            }
        ]
        v = await check_freshness("Heads up, your 3pm moved.", user_msgs)
        assert v.decision == "pass"
        assert "timeout" in v.reason

    async def test_prompt_includes_user_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake, captured = _make_prompt_capturing_query("DECISION: pass")
        monkeypatch.setattr(critic, "query", fake)
        user_msgs = [
            {
                "sender_name": "Filipe",
                "content": "never mind, found it",
                "timestamp": "2026-05-13T10:00:30+00:00",
            }
        ]
        await check_freshness("The meeting is at 3pm.", user_msgs)
        assert len(captured) == 1
        prompt = captured[0]
        assert "never mind, found it" in prompt
        assert "The meeting is at 3pm." in prompt
        assert "Filipe" in prompt

    async def test_empty_user_messages_still_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Defensive: empty list should still issue a query and pass-through.
        monkeypatch.setattr(critic, "query", _make_fake_query("DECISION: pass"))
        v = await check_freshness("Anything to talk about?", [])
        assert v.decision == "pass"


# ---------------------------------------------------------------------------
# Critic system prompt — date grounding
# ---------------------------------------------------------------------------


class TestCriticSystemPrompt:
    """The critic must not adjudicate a calendar it cannot see.

    Regression: on 2026-08-02 the critic revised a correct "Sunday, 2 August"
    to "Saturday" from parametric memory, vetoing a morning briefing that the
    deterministic weekday gate had already cleared.
    """

    def test_binds_today_and_weekday(self) -> None:
        p = _critic_system_prompt(date(2026, 8, 2))
        assert "2026-08-02" in p
        assert "Sunday" in p

    def test_forbids_flagging_weekday_date_pairs(self) -> None:
        p = _critic_system_prompt(date(2026, 8, 2))
        assert "Never flag a weekday as wrong for its date" in p

    def test_defaults_to_current_date(self) -> None:
        p = _critic_system_prompt()
        today = datetime.now(UTC).date()
        assert today.isoformat() in p
        assert today.strftime("%A") in p

    async def test_outbound_critic_prompt_carries_the_date(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []

        async def fake_collect(prompt: str, system_prompt: str) -> str:
            seen.append(system_prompt)
            return "DECISION: pass"

        monkeypatch.setattr(critic, "_collect_text", fake_collect)
        v = await critique_outbound("Morning — Sunday, 2 August.", {"tool": "send_message"})
        assert v.decision == "pass"
        assert len(seen) == 1
        assert datetime.now(UTC).date().isoformat() in seen[0]
