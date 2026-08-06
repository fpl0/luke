"""Tests for luke.agent — send_long_message, _send_chunk, _build_stop_hook, tools."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot
from claude_agent_sdk import HookCallback
from claude_agent_sdk.types import SyncHookJSONOutput

from luke.agent import (
    _AUTO_SKILL_THRESHOLD,
    _INTERNAL_RE,
    _VALID_MEMORY_TYPES,
    AgentResult,
    _build_stop_hook,
    _context_query,
    _cron_local_time_mismatch,
    _duplicate_pending_task,
    _md_to_html,
    _ok,
    _requests_file_artifact,
    _requests_source_read,
    _task_overlap,
    send_long_message,
)
from luke.config import settings

# ---------------------------------------------------------------------------
# send_long_message
# ---------------------------------------------------------------------------


class TestSendLongMessage:
    @pytest.fixture(autouse=True)
    def _patch_db(self) -> Any:
        with (
            patch("luke.agent.db.store_message"),
            patch("luke.agent.db.is_duplicate_outbound", return_value=False),
            patch("luke.agent.db.log_outbound"),
        ):
            yield

    @pytest.fixture()
    def mock_bot(self) -> AsyncMock:
        return AsyncMock(spec=Bot)

    async def test_short_message(self, mock_bot: AsyncMock) -> None:
        await send_long_message(mock_bot, chat_id=123, text="Hello")
        mock_bot.send_message.assert_called_once()
        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert "Hello" in call_kwargs.get("text", "")

    async def test_splits_at_newline(self, mock_bot: AsyncMock) -> None:
        first_half = "A" * 2000 + "\n"
        second_half = "B" * 3000
        text = first_half + second_half
        assert len(text) > 4096  # Telegram API limit

        await send_long_message(mock_bot, chat_id=123, text=text)
        assert mock_bot.send_message.call_count >= 2

    async def test_hard_cut_no_newline(self, mock_bot: AsyncMock) -> None:
        text = "A" * 10000
        await send_long_message(mock_bot, chat_id=123, text=text)
        assert mock_bot.send_message.call_count >= 2
        first_call_text = mock_bot.send_message.call_args_list[0].kwargs.get(
            "text", mock_bot.send_message.call_args_list[0][1].get("text", "")
        )
        assert first_call_text.endswith("\n…")


# ---------------------------------------------------------------------------
# Markdown never reaches Telegram
#
# parse_mode is HTML, so a stray backtick renders as a literal backtick. The
# persona forbids markdown, but 21 messages carried some in 30 days — the rule
# depends on the model remembering it every turn, and these tests do not.
# ---------------------------------------------------------------------------


class TestMarkdownToHtml:
    def test_inline_code_becomes_code_tag(self) -> None:
        assert _md_to_html("run `pytest -v` now") == "run <code>pytest -v</code> now"

    def test_bold_becomes_b_tag(self) -> None:
        assert _md_to_html("that is **28 of 32** files") == "that is <b>28 of 32</b> files"

    def test_fence_becomes_pre_tag(self) -> None:
        assert _md_to_html("```python\nx = 1\n```") == "<pre>x = 1</pre>"

    def test_fence_wins_over_inline(self) -> None:
        """The inline pattern would otherwise chew through a fence's delimiters."""
        assert "<pre>" in _md_to_html("```\na `b` c\n```")

    def test_tag_inside_code_is_escaped(self) -> None:
        """The older, louder bug: a <tag> in backticks made Telegram reject the
        whole message, dropping it to plaintext with every real tag showing raw."""
        assert _md_to_html("use `<b>` for bold") == "use <code>&lt;b&gt;</code> for bold"

    def test_existing_html_is_left_alone(self) -> None:
        text = "<b>real</b> tags and <code>spans</code> survive"
        assert _md_to_html(text) == text

    def test_bare_asterisks_are_not_bold(self) -> None:
        assert _md_to_html("2 * 3 * 4") == "2 * 3 * 4"

    async def test_conversion_happens_before_the_wire(self) -> None:
        bot = AsyncMock(spec=Bot)
        with (
            patch("luke.agent.db.store_message"),
            patch("luke.agent.db.is_duplicate_outbound", return_value=False),
            patch("luke.agent.db.log_outbound"),
        ):
            await send_long_message(bot, chat_id=123, text="commit `abc123`")
        assert "<code>abc123</code>" in bot.send_message.call_args.kwargs["text"]

    async def test_plaintext_sends_are_not_converted(self) -> None:
        """A caller that explicitly opts out of HTML must get its text verbatim."""
        bot = AsyncMock(spec=Bot)
        with (
            patch("luke.agent.db.store_message"),
            patch("luke.agent.db.is_duplicate_outbound", return_value=False),
            patch("luke.agent.db.log_outbound"),
        ):
            await send_long_message(bot, chat_id=123, text="`raw`", parse_mode=None)
        assert bot.send_message.call_args.kwargs["text"] == "`raw`"


# ---------------------------------------------------------------------------
# _send_chunk
# ---------------------------------------------------------------------------


def _mock_sent() -> MagicMock:
    """Create a mock Telegram message returned by bot.send_message."""
    msg = MagicMock()
    msg.message_id = 1
    msg.date.isoformat.return_value = "2024-01-01T00:00:00"
    return msg


class TestSendChunk:
    @pytest.fixture(autouse=True)
    def _patch_db(self) -> Any:
        with patch("luke.agent.db.store_message"):
            yield

    async def test_html_fallback(self) -> None:
        """When TelegramBadRequest is raised, should retry with parse_mode=None."""
        from aiogram.exceptions import TelegramBadRequest

        from luke.agent import _send_chunk

        mock_bot = AsyncMock(spec=Bot)
        mock_bot.send_message.side_effect = [
            TelegramBadRequest(method=MagicMock(), message="Bad HTML"),
            _mock_sent(),
        ]

        await _send_chunk(mock_bot, chat_id=123, text="<bad>html")
        assert mock_bot.send_message.call_count == 2
        # Second call should have parse_mode=None
        second_call = mock_bot.send_message.call_args_list[1]
        assert second_call.kwargs.get("parse_mode") is None

    async def test_retry_on_transient_error(self) -> None:
        """Transient errors should be retried with backoff."""
        from luke.agent import _send_chunk

        mock_bot = AsyncMock(spec=Bot)
        mock_bot.send_message.side_effect = [
            ConnectionError("network"),
            ConnectionError("network"),
            _mock_sent(),
        ]

        await _send_chunk(mock_bot, chat_id=123, text="Hello")
        assert mock_bot.send_message.call_count == 3

    async def test_retry_exhausted_raises(self) -> None:
        """After max retries, the exception should propagate."""
        from luke.agent import _send_chunk

        mock_bot = AsyncMock(spec=Bot)
        mock_bot.send_message.side_effect = ConnectionError("network")

        with pytest.raises(ConnectionError):
            await _send_chunk(mock_bot, chat_id=123, text="Hello")
        assert mock_bot.send_message.call_count == settings.telegram_send_retries

    async def test_retry_after_handling(self) -> None:
        """TelegramRetryAfter should sleep and retry."""
        from aiogram.exceptions import TelegramRetryAfter

        from luke.agent import _send_chunk

        mock_bot = AsyncMock(spec=Bot)
        exc = TelegramRetryAfter(method=MagicMock(), message="rate limited", retry_after=1)
        mock_bot.send_message.side_effect = [exc, _mock_sent()]

        await _send_chunk(mock_bot, chat_id=123, text="Hello")
        assert mock_bot.send_message.call_count == 2


# ---------------------------------------------------------------------------
# _build_stop_hook
# ---------------------------------------------------------------------------


class TestBuildStopHook:
    async def _call(self, tool_count: int, autonomous: bool) -> SyncHookJSONOutput:
        hook = _build_stop_hook({"n": tool_count}, autonomous)
        return cast(SyncHookJSONOutput, await hook(MagicMock(), None, MagicMock()))

    async def test_returns_system_message(self) -> None:
        result = await self._call(0, False)
        assert "systemMessage" in result
        assert "remember" in result["systemMessage"]

    async def test_no_skill_prompt_below_threshold(self) -> None:
        result = await self._call(_AUTO_SKILL_THRESHOLD - 1, False)
        assert "Skill extraction" not in result.get("systemMessage", "")

    async def test_skill_prompt_at_threshold(self) -> None:
        result = await self._call(_AUTO_SKILL_THRESHOLD, False)
        assert "Skill extraction" in result.get("systemMessage", "")
        assert "procedure" in result.get("systemMessage", "")

    async def test_no_skill_prompt_for_autonomous_runs(self) -> None:
        result = await self._call(_AUTO_SKILL_THRESHOLD + 10, True)
        assert "Skill extraction" not in result.get("systemMessage", "")


# ---------------------------------------------------------------------------
# Artifact-request capture gate
# ---------------------------------------------------------------------------


class TestRequestsFileArtifact:
    @pytest.mark.parametrize(
        "text",
        [
            "Give me a pdf for the visa stuff",
            "can you make me a doc with all the data",
            "put together a brief on the interview rounds",
            "build me a spreadsheet of the options",
            "I need a one-pager on this",
            "could you generate a csv",
            "send me the report when it is done",
        ],
    )
    def test_true_positives(self, text: str) -> None:
        assert _requests_file_artifact(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "draft me an email to Prerna",  # inline draft, not a file
            "write me a message for the team",  # inline
            "give me a second",
            "I need a break honestly",
            "how was your day",
            "I read the document you sent, thanks",  # past ref, not a request
            "thanks for the brief chat earlier",  # 'brief' adjective
            "lets keep it brief",
            "",
        ],
    )
    def test_true_negatives(self, text: str) -> None:
        assert _requests_file_artifact(text) is False


class TestArtifactGate:
    @pytest.fixture(autouse=True)
    def _isolated_db(self, test_db: Any) -> None:
        """The stop hook emits events via bus.emit → db; isolate from the real DB."""

    def _hook(
        self,
        *,
        requested: bool,
        delivered: int = 0,
        scheduled: int = 0,
        fired: int = 0,
        autonomous: bool = False,
    ) -> HookCallback:
        return _build_stop_hook(
            {"n": 3},
            autonomous,
            artifact_requested=requested,
            artifact_delivered_count={"n": delivered},
            work_scheduled_count={"n": scheduled},
            artifact_gate_fired={"n": fired},
        )

    async def _run(self, hook: HookCallback) -> SyncHookJSONOutput:
        return cast(SyncHookJSONOutput, await hook(MagicMock(), None, MagicMock()))

    async def test_blocks_when_requested_and_nothing_shipped(self) -> None:
        result = await self._run(self._hook(requested=True))
        assert result.get("decision") == "block"
        assert "send_document" in result.get("reason", "")

    async def test_passes_when_artifact_delivered(self) -> None:
        result = await self._run(self._hook(requested=True, delivered=1))
        assert result.get("decision") != "block"
        assert "systemMessage" in result

    async def test_passes_when_durable_handle_created(self) -> None:
        result = await self._run(self._hook(requested=True, scheduled=1))
        assert result.get("decision") != "block"
        assert "systemMessage" in result

    async def test_one_shot_does_not_refire(self) -> None:
        # A dropped turn: block once, flip the guard, then never block again
        # so the agent can't be trapped in a Stop loop.
        fired = {"n": 0}
        hook = _build_stop_hook(
            {"n": 3},
            False,
            artifact_requested=True,
            artifact_delivered_count={"n": 0},
            work_scheduled_count={"n": 0},
            artifact_gate_fired=fired,
        )
        first = await self._run(hook)
        assert first.get("decision") == "block"
        assert fired["n"] == 1
        second = await self._run(hook)
        assert second.get("decision") != "block"

    async def test_does_not_fire_when_not_requested(self) -> None:
        result = await self._run(self._hook(requested=False))
        assert result.get("decision") != "block"

    async def test_does_not_fire_for_autonomous_runs(self) -> None:
        result = await self._run(self._hook(requested=True, autonomous=True))
        assert result.get("decision") != "block"


# ---------------------------------------------------------------------------
# Primary-source read gate
# ---------------------------------------------------------------------------


class TestRequestsSourceRead:
    @pytest.mark.parametrize(
        "text",
        [
            "read the email from Prerna and tell me what she means",
            "can you check the doc I shared",
            "what does that pdf say about the FE round",
            "look at the thread and summarise it",
            "go through the letter before you reply",
            "did you even read the email?",
            "have you seen the attachment",
            "pull up the report and check the numbers",
            "read /Users/filipelm/Luke/workspace/foo.pdf",
        ],
    )
    def test_true_positives(self, text: str) -> None:
        assert _requests_source_read(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "thanks for the email, really helpful",  # past ref, no read-verb
            "how was your day",
            "give me a pdf of the visa stuff",  # artifact REQUEST, not a read
            "let's chat about the interview",
            "I already read it, all good",  # no source-noun
            "check in with me later tonight",  # 'check' but no source-noun
            "open to grabbing dinner?",  # 'open' but no source-noun
            "",
        ],
    )
    def test_true_negatives(self, text: str) -> None:
        assert _requests_source_read(text) is False


class TestIsReask:
    """Every true-positive here is a verbatim message Filipe actually sent.

    The 2026-08-03 14:49-14:52 run is the incident that motivated the gate: five
    re-asks in three minutes, answered with three reworded copies of one answer.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "so?",
            "So?",
            "??",
            "???",
            "again?",
            "I mean, how do you evaluate these chanes",
            "Luke, I mean, do you find them useful?1",  # vocative prefix
            "I meant, give me a summary of the day!",
            "I mean how do you know the visa date is correct",
            "Why are you giving me these mechanic answers?",
            "yes, I am forcing you to answer externally. Its ok. "
            "I just want you to say if these changes are useful!",
            "I asked a pdf for the visa stuff",
            "I mean, Luke won't you do anything else? I literally asked you to self-improve",
        ],
    )
    def test_true_positives(self, text: str) -> None:
        from luke.agent import _is_reask

        assert _is_reask(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "how do you find these changes?",  # the FIRST ask, not a re-ask
            "whats the state of the visa thing",
            "Alright, how are you feeling now?",
            "I will do some work on this now",
            "I am the one making the changes, this is not your fault!",
            "Get ride of the remaining letta sessions",
            "how big is your log file atm?!",
            "Amazing",
            "Ok?",
            "So I was thinking we could go on Friday",  # 'so' with content
            "",
        ],
    )
    def test_true_negatives(self, text: str) -> None:
        from luke.agent import _is_reask

        assert _is_reask(text) is None


class TestSourceReadGate:
    @pytest.fixture(autouse=True)
    def _isolated_db(self, test_db: Any) -> None:
        """The stop hook emits events via bus.emit → db; isolate from the real DB."""

    def _hook(
        self,
        *,
        requested: bool,
        read: int = 0,
        fired: int = 0,
        autonomous: bool = False,
    ) -> HookCallback:
        return _build_stop_hook(
            {"n": 3},
            autonomous,
            source_read_requested=requested,
            source_read_count={"n": read},
            source_gate_fired={"n": fired},
        )

    async def _run(self, hook: HookCallback) -> SyncHookJSONOutput:
        return cast(SyncHookJSONOutput, await hook(MagicMock(), None, MagicMock()))

    async def test_blocks_when_requested_and_nothing_read(self) -> None:
        result = await self._run(self._hook(requested=True))
        assert result.get("decision") == "block"
        assert "primary source" in result.get("reason", "").lower()

    async def test_passes_when_source_read(self) -> None:
        result = await self._run(self._hook(requested=True, read=1))
        assert result.get("decision") != "block"
        assert "systemMessage" in result

    async def test_one_shot_does_not_refire(self) -> None:
        fired = {"n": 0}
        hook = _build_stop_hook(
            {"n": 3},
            False,
            source_read_requested=True,
            source_read_count={"n": 0},
            source_gate_fired=fired,
        )
        first = await self._run(hook)
        assert first.get("decision") == "block"
        assert fired["n"] == 1
        second = await self._run(hook)
        assert second.get("decision") != "block"

    async def test_does_not_fire_when_not_requested(self) -> None:
        result = await self._run(self._hook(requested=False))
        assert result.get("decision") != "block"

    async def test_does_not_fire_for_autonomous_runs(self) -> None:
        result = await self._run(self._hook(requested=True, autonomous=True))
        assert result.get("decision") != "block"


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------


class TestAgentResult:
    def test_defaults(self) -> None:
        r = AgentResult()
        assert r.texts == []
        assert r.session_id is None
        assert r.tool_uses == 0


# ---------------------------------------------------------------------------
# _resolve_model_id
# ---------------------------------------------------------------------------


class TestResolveModelId:
    """Tier aliases must pin to explicit model IDs at the SDK boundary."""

    def test_tier_aliases_resolve_to_pinned_ids(self) -> None:
        from luke.agent import _MODEL_IDS, _resolve_model_id

        assert _resolve_model_id("haiku") == _MODEL_IDS["haiku"]
        assert _resolve_model_id("sonnet") == "claude-sonnet-5"
        assert _resolve_model_id("opus") == "claude-opus-5"

    def test_explicit_model_ids_pass_through(self) -> None:
        from luke.agent import _resolve_model_id

        assert _resolve_model_id("claude-haiku-4-5-20251001") == "claude-haiku-4-5-20251001"
        assert _resolve_model_id("claude-opus-4-1-20250805") == "claude-opus-4-1-20250805"

    def test_every_routing_tier_has_a_pinned_id(self) -> None:
        from luke.agent import _MODEL_IDS
        from luke.config import settings

        for tier in (settings.model_low, settings.model_medium, settings.model_high):
            assert tier in _MODEL_IDS


# ---------------------------------------------------------------------------
# _compose_system_append
# ---------------------------------------------------------------------------


class TestComposeSystemAppend:
    """Persona must close the system prompt — voice wins recency over memory."""

    def test_persona_comes_after_working_memory(self) -> None:
        from luke.agent import _compose_system_append

        out = _compose_system_append("You are Luke.", "proc-deploy: run the script")
        assert out.index("proc-deploy") < out.index("You are Luke.")
        assert out.rstrip().endswith("You are Luke.")

    def test_working_memory_is_framed_as_knowledge_not_voice(self) -> None:
        from luke.agent import _compose_system_append

        out = _compose_system_append("persona", "some memories")
        assert out.startswith("<working_memory>")
        assert "never how you sound" in out
        assert "</working_memory>" in out

    def test_no_working_memory_returns_bare_persona(self) -> None:
        from luke.agent import _compose_system_append

        assert _compose_system_append("You are Luke.", None) == "You are Luke."
        assert _compose_system_append("You are Luke.", "") == "You are Luke."

    def test_no_persona_returns_framed_memory(self) -> None:
        from luke.agent import _compose_system_append

        out = _compose_system_append("", "memories")
        assert out.startswith("<working_memory>")
        assert out.endswith("</working_memory>")


class TestComposeTurnPrefix:
    """The gap persona-last could not close.

    _compose_system_append wins recency inside the SYSTEM prompt, but the turn
    block is prepended to the USER message — so it reads after the persona and
    immediately before Filipe. Measured 2026-08-03: 2,098 turn-block tokens
    ahead of a two-token question ("so?"). He called the register "impersonal"
    (08-02) and "mechanic" (08-03) after the ordering fix had already shipped.
    """

    def test_voice_anchor_lands_after_the_evidence(self) -> None:
        from luke.agent import _VOICE_ANCHOR, _compose_turn_prefix

        out = _compose_turn_prefix("<context><memories>\n[m1] stuff\n</memories></context>")
        assert out.index("[m1]") < out.index(_VOICE_ANCHOR)

    def test_nothing_follows_the_anchor_but_the_user(self) -> None:
        from luke.agent import _compose_turn_prefix

        out = _compose_turn_prefix("<context><memories>\n[m1] stuff\n</memories></context>")
        assert out.rstrip().endswith("</voice>")
        full = out + "SO_QUESTIONMARK"
        assert full.index("</voice>") < full.index("SO_QUESTIONMARK")

    def test_anchor_names_the_measured_tics(self) -> None:
        """Adjectives don't survive contact with a 2k-token dossier; the named
        behaviours are each a thing the 30-day log shows Luke doing."""
        from luke.agent import _VOICE_ANCHOR

        low = _VOICE_ANCHOR.lower()
        assert "match his length" in low  # 989-char replies against his 229
        assert "summarise" in low  # a summing-up closer in 33% of messages
        assert "offering" in low  # a trailing offer in 25%


# ---------------------------------------------------------------------------
# _INTERNAL_RE
# ---------------------------------------------------------------------------


class TestInternalRe:
    def test_strips_internal_tags(self) -> None:
        text = "Hello <internal>secret stuff</internal> world"
        result = _INTERNAL_RE.sub("", text).strip()
        assert result == "Hello  world"

    def test_multiline_internal(self) -> None:
        text = "before\n<internal>\nline1\nline2\n</internal>\nafter"
        result = _INTERNAL_RE.sub("", text).strip()
        assert "line1" not in result
        assert "before" in result
        assert "after" in result


# ---------------------------------------------------------------------------
# _safe_path (testing the pattern since it's inside _build_tools closure)
# ---------------------------------------------------------------------------


class TestSafePath:
    """Test the path traversal prevention pattern used in _build_tools."""

    @staticmethod
    def _safe_path(path_str: str, safe_roots: tuple[Path, ...]) -> Path | None:
        resolved = Path(path_str).resolve()
        for root in safe_roots:
            if resolved == root or root in resolved.parents:
                return resolved
        return None

    def test_within_root(self, tmp_path: Path) -> None:
        subfile = tmp_path / "sub" / "file.txt"
        subfile.parent.mkdir(parents=True, exist_ok=True)
        subfile.touch()
        assert self._safe_path(str(subfile), (tmp_path,)) == subfile

    def test_root_itself(self, tmp_path: Path) -> None:
        assert self._safe_path(str(tmp_path), (tmp_path,)) == tmp_path

    def test_outside_rejected(self, tmp_path: Path) -> None:
        assert self._safe_path("/etc/passwd", (tmp_path,)) is None

    def test_traversal_rejected(self, tmp_path: Path) -> None:
        evil = str(tmp_path / ".." / ".." / ".." / "etc" / "passwd")
        assert self._safe_path(evil, (tmp_path,)) is None

    def test_multiple_roots(self, tmp_path: Path) -> None:
        root1 = tmp_path / "root1"
        root2 = tmp_path / "root2"
        root1.mkdir()
        root2.mkdir()
        f = root2 / "file.txt"
        f.touch()
        assert self._safe_path(str(f), (root1, root2)) == f


# ---------------------------------------------------------------------------
# Memory type validation & ID sanitization
# ---------------------------------------------------------------------------


class TestMemoryValidation:
    def test_valid_memory_types(self) -> None:
        assert "entity" in _VALID_MEMORY_TYPES
        assert "episode" in _VALID_MEMORY_TYPES
        assert "procedure" in _VALID_MEMORY_TYPES
        assert "insight" in _VALID_MEMORY_TYPES
        assert "goal" in _VALID_MEMORY_TYPES
        assert "invalid" not in _VALID_MEMORY_TYPES

    def test_id_sanitization(self) -> None:
        raw = "my/dangerous/../id!@#$"
        sanitized = re.sub(r"[^\w-]", "_", raw)
        assert "/" not in sanitized
        assert "." not in sanitized
        assert "!" not in sanitized
        assert sanitized == "my_dangerous____id____"


class TestOk:
    def test_ok_structure(self) -> None:
        result = _ok("success")
        assert result == {"content": [{"type": "text", "text": "success"}]}


# ---------------------------------------------------------------------------
# _build_tools (test tool closures via mocked SDK)
# ---------------------------------------------------------------------------


class TestBuildTools:
    """Test tool functions by calling _build_tools with mocked SDK."""

    @pytest.fixture(autouse=True)
    def _isolated_db(self, test_db: Any) -> None:
        """Tools emit events via bus.emit → db; isolate from the real DB."""

    @pytest.fixture(autouse=True)
    def _patch_db_store(self) -> Any:
        with (
            patch("luke.agent.db.store_message"),
            patch("luke.agent.db.is_duplicate_outbound", return_value=False),
            patch("luke.agent.db.log_outbound"),
        ):
            yield

    @pytest.fixture()
    def tool_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up environment for _build_tools testing."""
        store_dir = tmp_path / "store"
        store_dir.mkdir()

        mock_bot = AsyncMock(spec=Bot)
        captured_tools: list[Any] = []

        def fake_tool(name: str, desc: str, params: dict[str, Any], **_kw: Any) -> Any:
            def decorator(fn: Any) -> Any:
                fn._tool_name = name
                return fn

            return decorator

        def fake_create_server(**kwargs: Any) -> list[Any]:
            captured_tools.extend(kwargs["tools"])
            return captured_tools

        with (
            patch("luke.agent.tool", fake_tool),
            patch("luke.agent.create_sdk_mcp_server", fake_create_server),
            patch("luke.agent.settings") as mock_settings,
            patch("luke.agent.db.store_message"),
        ):
            mock_settings.store_dir = store_dir
            mock_settings.luke_dir = tmp_path
            mock_settings.recall_content_limit = 2000

            from luke.agent import _build_tools

            _build_tools("12345", mock_bot)

        tools_by_name = {t._tool_name: t for t in captured_tools if hasattr(t, "_tool_name")}

        return {
            "tools": tools_by_name,
            "bot": mock_bot,
            "root": tmp_path,
            "store_dir": store_dir,
        }

    async def test_send_message_tool(self, tool_env: dict[str, Any]) -> None:
        t_send = tool_env["tools"]["send_message"]
        result = await t_send({"text": "Hello!"})
        assert result["content"][0]["text"] == "Sent"
        tool_env["bot"].send_message.assert_called()

    async def test_send_photo_path_not_allowed(self, tool_env: dict[str, Any]) -> None:
        t_photo = tool_env["tools"]["send_photo"]
        result = await t_photo({"path": "/etc/passwd"})
        assert "not allowed" in result["content"][0]["text"]

    async def test_send_photo_file_not_found(self, tool_env: dict[str, Any]) -> None:
        t_photo = tool_env["tools"]["send_photo"]
        missing = str(tool_env["root"] / "nonexistent.jpg")
        result = await t_photo({"path": missing})
        assert "not found" in result["content"][0]["text"]

    async def test_send_document_success(self, tool_env: dict[str, Any]) -> None:
        t_doc = tool_env["tools"]["send_document"]
        f = tool_env["root"] / "test.txt"
        f.write_text("hello")
        result = await t_doc({"path": str(f)})
        assert result["content"][0]["text"] == "Document sent"

    async def test_send_voice_success(self, tool_env: dict[str, Any]) -> None:
        t_voice = tool_env["tools"]["send_voice"]
        f = tool_env["root"] / "voice.ogg"
        f.write_bytes(b"audio")
        result = await t_voice({"path": str(f)})
        assert result["content"][0]["text"] == "Voice sent"

    async def test_send_video_success(self, tool_env: dict[str, Any]) -> None:
        t_video = tool_env["tools"]["send_video"]
        f = tool_env["root"] / "video.mp4"
        f.write_bytes(b"video")
        result = await t_video({"path": str(f)})
        assert result["content"][0]["text"] == "Video sent"

    async def test_send_location_tool(self, tool_env: dict[str, Any]) -> None:
        t_loc = tool_env["tools"]["send_location"]
        result = await t_loc({"latitude": 51.5, "longitude": -0.1})
        assert result["content"][0]["text"] == "Location sent"

    async def test_send_poll_tool(self, tool_env: dict[str, Any]) -> None:
        t_poll = tool_env["tools"]["send_poll"]
        result = await t_poll({"question": "Coffee?", "options": ["Yes", "No"]})
        assert result["content"][0]["text"] == "Poll created"

    async def test_react_tool(self, tool_env: dict[str, Any]) -> None:
        t_react = tool_env["tools"]["react"]
        result = await t_react({"message_id": 1, "emoji": "\U0001f44d"})
        assert result["content"][0]["text"] == "Reacted"

    async def test_edit_message_tool(self, tool_env: dict[str, Any]) -> None:
        t_edit = tool_env["tools"]["edit_message"]
        result = await t_edit({"message_id": 1, "text": "edited"})
        assert result["content"][0]["text"] == "Edited"

    async def test_delete_message_tool(self, tool_env: dict[str, Any]) -> None:
        t_del = tool_env["tools"]["delete_message"]
        result = await t_del({"message_id": 1})
        assert result["content"][0]["text"] == "Deleted"

    async def test_pin_tool(self, tool_env: dict[str, Any]) -> None:
        t_pin = tool_env["tools"]["pin"]
        result = await t_pin({"message_id": 1})
        assert result["content"][0]["text"] == "Pinned"

    async def test_reply_tool(self, tool_env: dict[str, Any]) -> None:
        t_reply = tool_env["tools"]["reply"]
        result = await t_reply({"message_id": 1, "text": "reply text"})
        assert result["content"][0]["text"] == "Replied"

    async def test_forward_tool(self, tool_env: dict[str, Any]) -> None:
        t_fwd = tool_env["tools"]["forward"]
        result = await t_fwd({"from_chat_id": "12345", "to_chat_id": "12345", "message_id": 1})
        assert result["content"][0]["text"] == "Forwarded"

    async def test_schedule_task_tool(self, tool_env: dict[str, Any]) -> None:
        sched = tool_env["tools"]["schedule_task"]
        with patch("luke.agent.db") as mock_db:
            mock_db.create_task.return_value = "task-abc"
            result = await sched(
                {"prompt": "remind me", "schedule_type": "once", "schedule_value": "2025-01-01"}
            )
        assert "task-abc" in result["content"][0]["text"]

    async def test_schedule_task_invalid(self, tool_env: dict[str, Any]) -> None:
        sched = tool_env["tools"]["schedule_task"]
        with patch("luke.agent.db") as mock_db:
            mock_db.create_task.side_effect = ValueError("bad cron")
            result = await sched(
                {"prompt": "test", "schedule_type": "cron", "schedule_value": "bad"}
            )
        assert "Error" in result["content"][0]["text"]

    async def test_remember_tool(self, tool_env: dict[str, Any]) -> None:
        remember = tool_env["tools"]["remember"]
        with patch("luke.agent.memory") as mock_memory:
            mock_memory.detect_changes.return_value = []
            result = await remember(
                {"id": "test-mem", "type": "entity", "title": "Test", "content": "body"}
            )
        assert "Remembered" in result["content"][0]["text"]

    async def test_remember_procedure_persists_skill_meta_frontmatter(
        self, tmp_settings: Any, test_db: Any
    ) -> None:
        from luke import memory
        from luke.agent import _build_tools

        captured_tools: list[Any] = []

        def fake_tool(name: str, desc: str, params: dict[str, Any], **_kw: Any) -> Any:
            def decorator(fn: Any) -> Any:
                fn._tool_name = name
                return fn

            return decorator

        def fake_create_server(**kwargs: Any) -> list[Any]:
            captured_tools.extend(kwargs["tools"])
            return captured_tools

        mock_bot = AsyncMock(spec=Bot)
        with (
            patch("luke.agent.tool", fake_tool),
            patch("luke.agent.create_sdk_mcp_server", fake_create_server),
            patch("luke.agent.settings", tmp_settings),
            patch("luke.agent.db.store_message"),
            patch("luke.agent.db.is_duplicate_outbound", return_value=False),
            patch("luke.agent.db.log_outbound"),
        ):
            _build_tools("12345", mock_bot)
            tools = {t._tool_name: t for t in captured_tools if hasattr(t, "_tool_name")}
            remember = tools["remember"]
            await remember(
                {
                    "id": "deploy-docs",
                    "type": "procedure",
                    "title": "Deploy Docs",
                    "content": (
                        "## When to Use\n"
                        "Deploy docs after editing the site.\n\n"
                        "## Steps\n"
                        "1. Build the site\n"
                        "2. Publish the build\n"
                        "3. Verify production\n"
                    ),
                    "tags": ["skill", "docs"],
                }
            )

        path = tmp_settings.memory_dir / "procedures" / "deploy-docs.md"
        frontmatter = memory.read_frontmatter(path)
        assert frontmatter["skill_meta"]["confidence"] == 0.6
        assert "deploy" in frontmatter["skill_meta"]["trigger_pattern"]

    async def test_remember_rejects_non_trivial_auto_extracted_skill(
        self, tmp_settings: Any, test_db: Any
    ) -> None:
        from luke.agent import _build_tools

        captured_tools: list[Any] = []

        def fake_tool(name: str, desc: str, params: dict[str, Any], **_kw: Any) -> Any:
            def decorator(fn: Any) -> Any:
                fn._tool_name = name
                return fn

            return decorator

        def fake_create_server(**kwargs: Any) -> list[Any]:
            captured_tools.extend(kwargs["tools"])
            return captured_tools

        mock_bot = AsyncMock(spec=Bot)
        with (
            patch("luke.agent.tool", fake_tool),
            patch("luke.agent.create_sdk_mcp_server", fake_create_server),
            patch("luke.agent.settings", tmp_settings),
            patch("luke.agent.db.store_message"),
            patch("luke.agent.db.is_duplicate_outbound", return_value=False),
            patch("luke.agent.db.log_outbound"),
        ):
            _build_tools("12345", mock_bot)
            tools = {t._tool_name: t for t in captured_tools if hasattr(t, "_tool_name")}
            remember = tools["remember"]
            result = await remember(
                {
                    "id": "tiny-skill",
                    "type": "procedure",
                    "title": "Tiny Skill",
                    "content": (
                        "## When to Use\n"
                        "When you need a tiny skill.\n\n"
                        "## Steps\n"
                        "1. Do one thing\n"
                        "2. Do the second thing\n"
                    ),
                    "tags": ["skill", "auto-extracted"],
                }
            )

        assert "Skill rejected" in result["content"][0]["text"]
        assert not (tmp_settings.memory_dir / "procedures" / "tiny-skill.md").exists()

    async def test_remember_invalid_type(self, tool_env: dict[str, Any]) -> None:
        remember = tool_env["tools"]["remember"]
        result = await remember(
            {"id": "test", "type": "invalid_type", "title": "T", "content": "c"}
        )
        assert "Invalid type" in result["content"][0]["text"]

    async def test_recall_tool(self, tool_env: dict[str, Any]) -> None:
        recall = tool_env["tools"]["recall"]
        with patch("luke.agent.memory") as mock_memory:
            mock_memory.recall.return_value = []
            result = await recall({"query": "test"})
        assert "No memories" in result["content"][0]["text"]

    async def test_forget_tool(self, tool_env: dict[str, Any]) -> None:
        forget = tool_env["tools"]["forget"]
        with patch("luke.agent.memory"):
            result = await forget({"id": "mem-1"})
        assert "Archived" in result["content"][0]["text"]

    async def test_connect_tool(self, tool_env: dict[str, Any]) -> None:
        connect = tool_env["tools"]["connect"]
        with patch("luke.agent.memory"):
            result = await connect({"from_id": "a", "to_id": "b", "relationship": "related"})
        assert "Linked" in result["content"][0]["text"]

    async def test_recall_conversation_tool(self, tool_env: dict[str, Any]) -> None:
        recall_conv = tool_env["tools"]["recall_conversation"]
        with patch("luke.agent.memory") as mock_memory:
            mock_memory.recall_by_time_window.return_value = []
            result = await recall_conv({"after": "2024-01-01", "before": "2024-12-31"})
        assert "No memories" in result["content"][0]["text"]

    async def test_send_buttons_tool(self, tool_env: dict[str, Any]) -> None:
        t_buttons = tool_env["tools"]["send_buttons"]
        result = await t_buttons(
            {
                "text": "Choose:",
                "buttons": [[{"text": "Yes", "data": "yes"}, {"text": "No", "data": "no"}]],
            }
        )
        assert result["content"][0]["text"] == "Buttons sent"

    async def test_send_buttons_malformed(self, tool_env: dict[str, Any]) -> None:
        t_buttons = tool_env["tools"]["send_buttons"]
        result = await t_buttons(
            {
                "text": "Choose:",
                "buttons": [["not_a_dict"]],
            }
        )
        assert "Error" in result["content"][0]["text"]

    def test_all_mcp_tool_names_matches_registered(self, tool_env: dict[str, Any]) -> None:
        """_ALL_MCP_TOOL_NAMES must match the tools actually registered in _build_tools."""
        from luke.agent import _ALL_MCP_TOOL_NAMES

        registered = set(tool_env["tools"].keys())
        declared = set(_ALL_MCP_TOOL_NAMES)
        assert declared == registered, (
            f"Mismatch: in _ALL_MCP_TOOL_NAMES but not registered: {declared - registered}, "
            f"registered but not in _ALL_MCP_TOOL_NAMES: {registered - declared}"
        )


# ---------------------------------------------------------------------------
# PostToolUse / PostToolUseFailure / Subagent hooks
# ---------------------------------------------------------------------------


class TestPostToolUseHooks:
    """Test the PostToolUse, PostToolUseFailure, SubagentStart, SubagentStop hooks.

    These hooks are closures created inside run_agent(). We can't call them
    directly without starting a full agent session.  Instead we test them
    by importing the hook bodies (they share module-level helpers) and
    verifying the event emission + logging logic.  We re-create the closure
    environment manually.
    """

    @pytest.fixture(autouse=True)
    def _patch_db(self) -> Any:
        """Patch db.emit_event so tests don't hit a real database."""
        with patch("luke.agent.db.emit_event", return_value=1) as mock_emit:
            self.mock_emit = mock_emit
            yield

    async def test_post_tool_hook_emits_event(self) -> None:
        """PostToolUse hook should emit a tool_use event with duration."""
        import json
        import time as _time

        from luke import db

        tool_start_times: dict[str, float] = {"tu_123": _time.monotonic() - 0.5}

        async def _post_tool_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            tool_name = input_data["tool_name"]
            tid = input_data.get("tool_use_id") or tool_use_id
            duration_ms: int | None = None
            if tid and tid in tool_start_times:
                duration_ms = int((_time.monotonic() - tool_start_times.pop(tid)) * 1000)
            agent_id = input_data.get("agent_id")
            agent_type = input_data.get("agent_type")
            payload: dict[str, Any] = {"tool": tool_name, "success": True}
            if duration_ms is not None:
                payload["duration_ms"] = duration_ms
            if agent_id:
                payload["agent_id"] = agent_id
            if agent_type:
                payload["agent_type"] = agent_type
            db.emit_event("tool_use", json.dumps(payload))
            return {}

        result = await _post_tool_hook(
            {"tool_name": "Bash", "tool_use_id": "tu_123", "tool_input": {}, "tool_response": "ok"},
            "tu_123",
            {},
        )
        assert result == {}
        self.mock_emit.assert_called_once()
        call_args = self.mock_emit.call_args
        assert call_args[0][0] == "tool_use"
        payload = json.loads(call_args[0][1])
        assert payload["tool"] == "Bash"
        assert payload["success"] is True
        assert "duration_ms" in payload
        assert payload["duration_ms"] >= 400  # ~500ms with tolerance

    async def test_post_tool_failure_emits_event(self) -> None:
        """PostToolUseFailure hook should emit a tool_failure event."""
        import json

        from luke import db

        async def _post_tool_failure_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            tool_name = input_data["tool_name"]
            error = input_data.get("error", "unknown")
            payload: dict[str, Any] = {
                "tool": tool_name,
                "success": False,
                "error": str(error)[:500],
            }
            db.emit_event("tool_failure", json.dumps(payload))
            return {}

        result = await _post_tool_failure_hook(
            {
                "tool_name": "Read",
                "tool_use_id": "tu_456",
                "tool_input": {},
                "error": "File not found",
            },
            "tu_456",
            {},
        )
        assert result == {}
        self.mock_emit.assert_called_once()
        call_args = self.mock_emit.call_args
        assert call_args[0][0] == "tool_failure"
        payload = json.loads(call_args[0][1])
        assert payload["tool"] == "Read"
        assert payload["success"] is False
        assert "File not found" in payload["error"]

    async def test_subagent_start_emits_event(self) -> None:
        """SubagentStart hook should emit a subagent_start event."""
        import json

        from luke import db

        subagent_start_times: dict[str, float] = {}

        async def _subagent_start_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            import time as _time

            agent_id = input_data["agent_id"]
            agent_type = input_data["agent_type"]
            subagent_start_times[agent_id] = _time.monotonic()
            db.emit_event(
                "subagent_start",
                json.dumps({"agent_id": agent_id, "agent_type": agent_type}),
            )
            return {}

        result = await _subagent_start_hook(
            {"agent_id": "sa_001", "agent_type": "researcher"},
            None,
            {},
        )
        assert result == {}
        assert "sa_001" in subagent_start_times
        self.mock_emit.assert_called_once()
        payload = json.loads(self.mock_emit.call_args[0][1])
        assert payload["agent_id"] == "sa_001"
        assert payload["agent_type"] == "researcher"

    async def test_subagent_stop_emits_event_with_duration(self) -> None:
        """SubagentStop hook should emit a subagent_stop event with duration."""
        import json
        import time as _time

        from luke import db

        subagent_start_times: dict[str, float] = {"sa_002": _time.monotonic() - 2.0}

        async def _subagent_stop_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            agent_id = input_data["agent_id"]
            agent_type = input_data["agent_type"]
            duration_ms: int | None = None
            if agent_id in subagent_start_times:
                duration_ms = int((_time.monotonic() - subagent_start_times.pop(agent_id)) * 1000)
            db.emit_event(
                "subagent_stop",
                json.dumps(
                    {
                        "agent_id": agent_id,
                        "agent_type": agent_type,
                        "duration_ms": duration_ms,
                    }
                ),
            )
            return {}

        result = await _subagent_stop_hook(
            {
                "agent_id": "sa_002",
                "agent_type": "coder",
                "agent_transcript_path": "/tmp/t",
                "stop_hook_active": False,
            },
            None,
            {},
        )
        assert result == {}
        assert "sa_002" not in subagent_start_times  # cleaned up
        payload = json.loads(self.mock_emit.call_args[0][1])
        assert payload["agent_type"] == "coder"
        assert payload["duration_ms"] >= 1800  # ~2000ms with tolerance


# ---------------------------------------------------------------------------
# Active client registry / interrupt
# ---------------------------------------------------------------------------


class TestActiveClientRegistry:
    """Test the active client registry and interrupt_agent function."""

    def test_get_active_agents_empty(self) -> None:
        from luke.agent import _active_clients, get_active_agents

        _active_clients.clear()
        assert get_active_agents() == []

    def test_get_active_agents_with_entries(self) -> None:
        from luke.agent import _active_clients, get_active_agents

        _active_clients.clear()
        _active_clients["123"] = MagicMock()
        _active_clients["456"] = MagicMock()
        assert sorted(get_active_agents()) == ["123", "456"]
        _active_clients.clear()

    async def test_interrupt_agent_no_client(self) -> None:
        from luke.agent import _active_clients, interrupt_agent

        _active_clients.clear()
        result = await interrupt_agent("nonexistent")
        assert result is False

    async def test_interrupt_agent_success(self) -> None:
        from luke.agent import _active_clients, interrupt_agent

        mock_client = AsyncMock()
        _active_clients["123"] = mock_client
        result = await interrupt_agent("123")
        assert result is True
        mock_client.interrupt.assert_awaited_once()
        _active_clients.clear()

    async def test_interrupt_agent_failure(self) -> None:
        from luke.agent import _active_clients, interrupt_agent

        mock_client = AsyncMock()
        mock_client.interrupt.side_effect = Exception("connection lost")
        _active_clients["123"] = mock_client
        result = await interrupt_agent("123")
        assert result is False
        _active_clients.clear()


# ---------------------------------------------------------------------------
# Streaming text cleaner
# ---------------------------------------------------------------------------


class TestRecallBeforeReference:
    """Test the recall-before-reference gate inside _pre_tool_hook.

    The hook is a closure built inside run_agent(). We test the gate logic
    directly by reproducing the relevant closure state and calling a faithful
    reconstruction of the gate, matching the pattern used by TestPostToolUseHooks.
    """

    # ----- _references_past_events helper -----

    @pytest.mark.parametrize(
        "phrase",
        [
            "yesterday",
            "last week",
            "earlier today",
            "last time",
            "the other day",
            "previously",
            "when we talked",
            "you mentioned",
            "you said",
            "you told me",
            "we discussed",
            "the thing we",
            "the topic",
            "remember when",
        ],
    )
    def test_references_past_events_matches_phrase(self, phrase: str) -> None:
        from luke.agent import _references_past_events

        text = f"Quick follow-up: {phrase} that thing — does it still hold?"
        assert _references_past_events(text) is True

    def test_references_past_events_case_insensitive(self) -> None:
        from luke.agent import _references_past_events

        assert _references_past_events("YESTERDAY we talked about the demo plan.") is True

    def test_references_past_events_false_for_fresh_message(self) -> None:
        from luke.agent import _references_past_events

        fresh = "Heads up — your 3pm meeting room just changed to Vega upstairs."
        assert _references_past_events(fresh) is False

    @pytest.mark.parametrize(
        "fresh",
        [
            "One thing worth locking before Aug 10 while we're deep in prep.",
            "Here's the part that comes before you walk in on day one.",
            "Let's nail this down before your start date next month.",
        ],
    )
    def test_references_past_events_false_for_forward_before(self, fresh: str) -> None:
        # Regression: bare "before" used about the FUTURE must not trigger the
        # recall gate. Two such false positives (14:01 + 00:56 on 2026-07-13)
        # blocked legit forward-looking drafts. See reflexion 2026-07-14.
        from luke.agent import _references_past_events

        assert _references_past_events(fresh) is False

    def test_references_past_events_false_for_empty(self) -> None:
        from luke.agent import _references_past_events

        assert _references_past_events("") is False

    def test_references_past_events_false_for_short_text(self) -> None:
        from luke.agent import _references_past_events

        # Even though "yesterday" appears, text is < 30 chars so it's not blocked
        assert _references_past_events("yesterday?") is False

    # ----- _pre_tool_hook recall gate -----
    #
    # We rebuild the same closure structure used by run_agent() so the test
    # exercises the real branching logic.  Only the slice relevant to the
    # recall gate is reproduced — the F4 critic feature will stack on top.

    @staticmethod
    def _build_hook(
        *,
        autonomous: bool,
        initial_recall: int = 0,
    ) -> tuple[Any, dict[str, int], dict[str, int], list[dict[str, Any]]]:
        """Construct a faithful copy of _pre_tool_hook's recall gate.

        Returns (hook, send_count, recall_count, emitted_events).
        """
        from luke.agent import (
            _RECALL_TOOLS,
            _SEND_TOOLS,
            _references_past_events,
        )

        send_count: dict[str, int] = {"n": 0}
        recall_count: dict[str, int] = {"n": initial_recall}
        emitted: list[dict[str, Any]] = []

        async def _hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            tool_name = input_data["tool_name"]
            if tool_name in _RECALL_TOOLS:
                recall_count["n"] += 1
            # Background-work routing gate: harness Task sub-agents die on
            # teardown and never self-report, so in a live (non-autonomous)
            # turn they are blocked in favor of mcp__luke__delegate.
            if tool_name == "Task" and not autonomous:
                emitted.append({"event": "task_blocked_use_delegate"})
                return {
                    "decision": "block",
                    "reason": (
                        "Don't spawn a harness Task sub-agent in a live "
                        "conversation — use mcp__luke__delegate instead."
                    ),
                }
            if tool_name in _SEND_TOOLS:
                send_count["n"] += 1
                if autonomous:
                    tool_input = input_data.get("tool_input", {})
                    msg_text = tool_input.get("text", "") if isinstance(tool_input, dict) else ""
                    if recall_count["n"] == 0 and _references_past_events(msg_text):
                        emitted.append(
                            {
                                "event": "send_blocked_no_recall",
                                "tool": tool_name,
                                "preview": msg_text[:100],
                            }
                        )
                        return {
                            "decision": "block",
                            "reason": (
                                "Reference to past events detected; call "
                                "recall (or recall_conversation) first to "
                                "ground in actual memory."
                            ),
                        }
            return {}

        return _hook, send_count, recall_count, emitted

    async def test_pre_tool_hook_blocks_autonomous_send_with_past_reference(self) -> None:
        hook, _send, _recall, emitted = self._build_hook(autonomous=True)
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": (
                        "Quick check — yesterday you mentioned the talk prep "
                        "was almost done. Is it still on track?"
                    ),
                },
                "tool_use_id": "tu_a",
            },
            "tu_a",
            {},
        )
        assert result.get("decision") == "block"
        assert "recall" in result.get("reason", "").lower()
        assert len(emitted) == 1
        assert emitted[0]["event"] == "send_blocked_no_recall"

    async def test_pre_tool_hook_does_not_block_when_recall_called(self) -> None:
        hook, _send, recall, emitted = self._build_hook(autonomous=True)
        # Simulate recall being invoked earlier in the turn.
        await hook(
            {
                "tool_name": "mcp__luke__recall",
                "tool_input": {"query": "talk prep"},
                "tool_use_id": "tu_r",
            },
            "tu_r",
            {},
        )
        assert recall["n"] == 1

        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": (
                        "Quick check — yesterday you mentioned the talk prep "
                        "was almost done. Is it still on track?"
                    ),
                },
                "tool_use_id": "tu_s",
            },
            "tu_s",
            {},
        )
        assert result == {}
        assert emitted == []

    async def test_pre_tool_hook_does_not_block_non_autonomous(self) -> None:
        hook, _send, _recall, emitted = self._build_hook(autonomous=False)
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": (
                        "Quick check — yesterday you mentioned the talk prep "
                        "was almost done. Is it still on track?"
                    ),
                },
                "tool_use_id": "tu_b",
            },
            "tu_b",
            {},
        )
        assert result == {}
        assert emitted == []

    async def test_pre_tool_hook_blocks_task_in_live_conversation(self) -> None:
        # Harness Task sub-agents die on teardown and never self-report — the
        # "so?" silence Filipe flagged July 15. In a live turn they must be
        # steered to mcp__luke__delegate.
        hook, _send, _recall, emitted = self._build_hook(autonomous=False)
        result = await hook(
            {
                "tool_name": "Task",
                "tool_input": {"description": "research CarGurus financials"},
                "tool_use_id": "tu_task",
            },
            "tu_task",
            {},
        )
        assert result.get("decision") == "block"
        assert "delegate" in result.get("reason", "").lower()
        assert emitted and emitted[0]["event"] == "task_blocked_use_delegate"

    async def test_pre_tool_hook_allows_task_when_autonomous(self) -> None:
        # Autonomous runs (crons/deep work) have no interrupting message, so a
        # within-turn Task is safe and must not be blocked.
        hook, _send, _recall, emitted = self._build_hook(autonomous=True)
        result = await hook(
            {
                "tool_name": "Task",
                "tool_input": {"description": "parallel research fan-out"},
                "tool_use_id": "tu_task2",
            },
            "tu_task2",
            {},
        )
        assert result == {}
        assert emitted == []

    async def test_pre_tool_hook_does_not_block_fresh_text(self) -> None:
        hook, _send, _recall, emitted = self._build_hook(autonomous=True)
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "Heads up — your 3pm meeting moved to Vega room upstairs.",
                },
                "tool_use_id": "tu_c",
            },
            "tu_c",
            {},
        )
        assert result == {}
        assert emitted == []

    async def test_pre_tool_hook_recall_conversation_also_satisfies_gate(self) -> None:
        hook, _send, recall, emitted = self._build_hook(autonomous=True)
        await hook(
            {
                "tool_name": "mcp__luke__recall_conversation",
                "tool_input": {"query": "talk prep"},
                "tool_use_id": "tu_rc",
            },
            "tu_rc",
            {},
        )
        assert recall["n"] == 1

        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": (
                        "About the topic we discussed earlier — I have an "
                        "update worth flagging now."
                    ),
                },
                "tool_use_id": "tu_s2",
            },
            "tu_s2",
            {},
        )
        assert result == {}
        assert emitted == []


class TestOvernightCommitmentGate:
    """Test the overnight-commitment gate inside _pre_tool_hook.

    Blocks outbound sends that commit to future delivery ("overnight",
    "by morning", etc.) when no agent or scheduled task has been spawned
    in the same turn. Runs for both autonomous and interactive runs.
    """

    # ----- _commits_future_work helper -----

    @pytest.mark.parametrize(
        "text",
        [
            "Got it. Sleep — I've got it. I'll have both docs before 8am.",
            "Perfect. I'll deliver the refined prep doc overnight. Sleep.",
            "I'll have the analysis ready by morning — sleep well.",
            "Got you covered. The script will be ready tomorrow morning.",
            "Consider it done before you wake. I'll ship both files overnight.",
            "Working on it tonight — I'll send the refactored module first thing.",
        ],
    )
    def test_commits_future_work_matches_commitment(self, text: str) -> None:
        from luke.agent import _commits_future_work

        assert _commits_future_work(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Heads up — your 3pm meeting room just changed to Vega upstairs.",
            "The earnings release happens tomorrow morning, fyi.",
            "Overnight oats are good but I prefer toast.",
            "By morning the freeze ends — non-critical merges resume.",
            "I'll have a look at this later when I can.",  # no time anchor
        ],
    )
    def test_commits_future_work_false_for_non_commitments(self, text: str) -> None:
        from luke.agent import _commits_future_work

        assert _commits_future_work(text) is False

    def test_commits_future_work_false_for_short_text(self) -> None:
        from luke.agent import _commits_future_work

        # "got it overnight" is too short to count
        assert _commits_future_work("got it overnight") is False

    def test_commits_future_work_false_for_empty(self) -> None:
        from luke.agent import _commits_future_work

        assert _commits_future_work("") is False

    # ----- _pre_tool_hook commitment gate -----

    @staticmethod
    def _build_hook() -> tuple[Any, dict[str, int], dict[str, int], list[dict[str, Any]]]:
        """Reproduce _pre_tool_hook's commitment-gate slice.

        Returns (hook, send_count, work_scheduled_count, emitted_events).
        """
        from luke.agent import (
            _AGENT_SCHEDULE_TOOLS,
            _SEND_TOOLS,
            _commits_future_work,
        )

        send_count: dict[str, int] = {"n": 0}
        work_scheduled_count: dict[str, int] = {"n": 0}
        emitted: list[dict[str, Any]] = []

        async def _hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            tool_name = input_data["tool_name"]
            if tool_name in _AGENT_SCHEDULE_TOOLS:
                work_scheduled_count["n"] += 1
            if tool_name in _SEND_TOOLS:
                send_count["n"] += 1
                tool_input = input_data.get("tool_input", {})
                if isinstance(tool_input, dict):
                    msg_text = tool_input.get("text", "") or tool_input.get("caption", "")
                else:
                    msg_text = ""
                if work_scheduled_count["n"] == 0 and _commits_future_work(msg_text):
                    emitted.append(
                        {
                            "event": "commitment_blocked_no_execution",
                            "tool": tool_name,
                            "preview": msg_text[:100],
                        }
                    )
                    return {
                        "decision": "block",
                        "reason": (
                            "Commitment to future delivery detected but no "
                            "agent or scheduled task was spawned this turn."
                        ),
                    }
            return {}

        return _hook, send_count, work_scheduled_count, emitted

    async def test_blocks_send_with_overnight_commitment_and_no_spawn(self) -> None:
        hook, _send, _work, emitted = self._build_hook()
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": (
                        "Got it. Sleep — I've got it. I'll have both docs "
                        "refined and ready before 8am."
                    ),
                },
                "tool_use_id": "tu_a",
            },
            "tu_a",
            {},
        )
        assert result.get("decision") == "block"
        assert "commitment" in result.get("reason", "").lower()
        assert len(emitted) == 1
        assert emitted[0]["event"] == "commitment_blocked_no_execution"

    async def test_passes_send_when_task_was_spawned_first(self) -> None:
        hook, _send, work, emitted = self._build_hook()
        # Simulate an agent spawn earlier in the turn.
        await hook(
            {
                "tool_name": "Task",
                "tool_input": {"description": "refine prep doc"},
                "tool_use_id": "tu_t",
            },
            "tu_t",
            {},
        )
        assert work["n"] == 1

        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": (
                        "Got it. Sleep — I've got it. I'll have both docs "
                        "refined and ready before 8am."
                    ),
                },
                "tool_use_id": "tu_s",
            },
            "tu_s",
            {},
        )
        assert result == {}
        assert emitted == []

    async def test_passes_send_when_schedule_task_was_called_first(self) -> None:
        hook, _send, work, emitted = self._build_hook()
        await hook(
            {
                "tool_name": "mcp__luke__schedule_task",
                "tool_input": {
                    "prompt": "deliver refined prep doc",
                    "schedule_type": "once",
                    "schedule_value": "2026-05-14T07:30:00+00:00",
                },
                "tool_use_id": "tu_sched",
            },
            "tu_sched",
            {},
        )
        assert work["n"] == 1

        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "I'll have the refined doc ready by morning. Sleep.",
                },
                "tool_use_id": "tu_s2",
            },
            "tu_s2",
            {},
        )
        assert result == {}
        assert emitted == []

    async def test_passes_send_with_no_commitment_language(self) -> None:
        hook, _send, _work, emitted = self._build_hook()
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "Heads up — your 3pm meeting moved to Vega room.",
                },
                "tool_use_id": "tu_c",
            },
            "tu_c",
            {},
        )
        assert result == {}
        assert emitted == []

    async def test_caption_also_checked_for_send_document(self) -> None:
        """send_document carries text in 'caption', not 'text' — must still gate."""
        hook, _send, _work, emitted = self._build_hook()
        result = await hook(
            {
                "tool_name": "mcp__luke__send_document",
                "tool_input": {
                    "path": "/tmp/foo.md",
                    "caption": (
                        "Here's the outline. I'll have the full doc ready before 8am — sleep well."
                    ),
                },
                "tool_use_id": "tu_doc",
            },
            "tu_doc",
            {},
        )
        assert result.get("decision") == "block"
        assert len(emitted) == 1


class TestWeekdayClaimGate:
    """Weekday/date consistency gate.

    Regression coverage for 2026-07-31, when a scheduled reminder told Filipe his
    US visa interview was "Tuesday Aug 7" — Aug 7 2026 is a Friday, and the
    entity memory held the correct day. feedback-dates-accuracy was advisory
    only; this gate makes it enforced on the send path.
    """

    REF = date(2026, 8, 2)  # a Sunday

    @pytest.mark.parametrize(
        "text",
        [
            "Visa in 7 days — Tuesday Aug 7, 07:45, 42 Elgin Road.",  # the real one
            "Interview is <b>Monday, 7 August 2026</b> at 07:45",  # HTML-wrapped
            "Your interview is Sunday, Aug 7th",  # ordinal suffix
            "CarGurus start Tuesday Aug 10",  # yearless, near-term
            "We spoke on Friday Jul 27 about the loop",  # yearless, recent past
            "Talk Saturday 2 Aug 2026",  # day-first with year
        ],
    )
    def test_blocks_wrong_weekday(self, text: str) -> None:
        from luke.agent import _weekday_claim_error

        assert _weekday_claim_error(text, today=self.REF) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "Visa in 7 days — Friday Aug 7, 07:45, 42 Elgin Road.",  # corrected
            "Interview is Fri 7 Aug 2026, 07:45",  # abbreviated weekday, ignored
            "Interview is <b>Friday, 7 August 2026</b> at 07:45",
            "Your interview is Friday, Aug 7th",
            "CarGurus start Monday Aug 10",
            "We spoke on Monday Jul 27 about the loop",
            "Started at Clio on Monday Aug 4 2025",
            "Christopher got his passport Monday, 16 March 2026",
            "Portugal joined on Thursday Jan 1",  # far yearless → any year may vindicate
            "Sunday weekly review",  # weekday with no date
            "Deadline Aug 10 — no weekday here",  # date with no weekday
            "meeting Friday, 29 Feb",  # impossible date must not raise
        ],
    )
    def test_allows_consistent_or_unprovable(self, text: str) -> None:
        from luke.agent import _weekday_claim_error

        assert _weekday_claim_error(text, today=self.REF) is None

    def test_reason_names_the_actual_day(self) -> None:
        from luke.agent import _weekday_claim_error

        reason = _weekday_claim_error("visa is Tuesday Aug 7", today=self.REF)
        assert reason is not None
        assert "Friday" in reason and "2026-08-07" in reason

    def test_empty_text_is_safe(self) -> None:
        from luke.agent import _weekday_claim_error

        assert _weekday_claim_error("", today=self.REF) is None


class TestOutboundQualityGate:
    """Test the outbound quality gate slice inside _pre_tool_hook.

    Regression coverage for the two-month bug (2026-05-16 → 2026-07-13) where
    every send_document was blocked as "empty message": the gate read only the
    'text' field, which documents/media do not carry — their payload is the
    file, with an OPTIONAL caption. Faithfully reproduces the gate slice
    (lines ~1671-1687 of agent.py), including the _TEXT_PRIMARY_TOOLS branch.
    """

    @staticmethod
    def _build_hook() -> tuple[Any, list[dict[str, Any]]]:
        from luke.agent import (
            _SEND_TOOLS,
            _TEXT_PRIMARY_TOOLS,
            _check_outbound_quality,
        )

        emitted: list[dict[str, Any]] = []

        async def _hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            tool_name = input_data["tool_name"]
            if tool_name in _SEND_TOOLS:
                tool_input = input_data.get("tool_input", {})
                if isinstance(tool_input, dict):
                    msg_text = tool_input.get("text", "") or tool_input.get("caption", "")
                else:
                    msg_text = ""
                if tool_name in _TEXT_PRIMARY_TOOLS or msg_text.strip():
                    rejection = _check_outbound_quality(msg_text)
                else:
                    rejection = None
                if rejection:
                    emitted.append(
                        {
                            "reason": rejection,
                            "tool": tool_name,
                            "preview": msg_text[:100],
                        }
                    )
                    return {
                        "decision": "block",
                        "reason": f"Quality gate: {rejection}",
                    }
            return {}

        return _hook, emitted

    async def test_send_document_with_no_caption_passes(self) -> None:
        """The regression: a file with no caption is a valid send, not empty."""
        hook, emitted = self._build_hook()
        result = await hook(
            {
                "tool_name": "mcp__luke__send_document",
                "tool_input": {"path": "/tmp/cargurus-ramp.pdf"},
                "tool_use_id": "tu_doc_empty",
            },
            "tu_doc_empty",
            {},
        )
        assert result == {}
        assert emitted == []

    async def test_send_document_with_empty_caption_passes(self) -> None:
        hook, emitted = self._build_hook()
        result = await hook(
            {
                "tool_name": "mcp__luke__send_document",
                "tool_input": {"path": "/tmp/foo.pdf", "caption": "   "},
                "tool_use_id": "tu_doc_ws",
            },
            "tu_doc_ws",
            {},
        )
        assert result == {}
        assert emitted == []

    async def test_send_document_with_substantive_caption_passes(self) -> None:
        hook, emitted = self._build_hook()
        result = await hook(
            {
                "tool_name": "mcp__luke__send_document",
                "tool_input": {
                    "path": "/tmp/foo.pdf",
                    "caption": "Your day-1 operating guide — the founding-EM build plan.",
                },
                "tool_use_id": "tu_doc_ok",
            },
            "tu_doc_ok",
            {},
        )
        assert result == {}
        assert emitted == []

    async def test_send_document_with_bad_caption_still_blocks(self) -> None:
        """Content quality checks still apply to captions that ARE present."""
        hook, emitted = self._build_hook()
        result = await hook(
            {
                "tool_name": "mcp__luke__send_document",
                "tool_input": {
                    "path": "/tmp/foo.pdf",
                    "caption": "<internal>note to self</internal>",
                },
                "tool_use_id": "tu_doc_bad",
            },
            "tu_doc_bad",
            {},
        )
        assert result.get("decision") == "block"
        assert len(emitted) == 1

    async def test_send_message_empty_text_still_blocks(self) -> None:
        """Text-primary tools must still reject an empty body."""
        hook, emitted = self._build_hook()
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {"text": ""},
                "tool_use_id": "tu_msg_empty",
            },
            "tu_msg_empty",
            {},
        )
        assert result.get("decision") == "block"
        assert emitted[0]["reason"] == "empty message"


class TestCriticGate:
    """Test the critic-agent gate inside _pre_tool_hook (F4).

    Follows the closure-reproduction pattern from TestRecallBeforeReference:
    we rebuild the hook's outer structure faithfully so the test exercises
    the same branching logic the real run_agent installs.

    `critique_outbound` is monkeypatched to control verdicts.
    """

    @staticmethod
    def _build_hook(
        *,
        autonomous: bool,
        critic_enabled: bool,
        verdict_fn: Any,
    ) -> tuple[Any, list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
        """Construct a faithful copy of _pre_tool_hook's critic gate.

        Returns (hook, emitted_events, critic_calls).
        """
        from luke.agent import _RECALL_TOOLS, _SEND_TOOLS, _references_past_events

        send_count: dict[str, int] = {"n": 0}
        recall_count: dict[str, int] = {"n": 0}
        emitted: list[dict[str, Any]] = []
        critic_calls: list[tuple[str, dict[str, Any]]] = []

        async def _hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            tool_name = input_data["tool_name"]
            if tool_name in _RECALL_TOOLS:
                recall_count["n"] += 1
            # Background-work routing gate: harness Task sub-agents die on
            # teardown and never self-report, so in a live (non-autonomous)
            # turn they are blocked in favor of mcp__luke__delegate.
            if tool_name == "Task" and not autonomous:
                emitted.append({"event": "task_blocked_use_delegate"})
                return {
                    "decision": "block",
                    "reason": (
                        "Don't spawn a harness Task sub-agent in a live "
                        "conversation — use mcp__luke__delegate instead."
                    ),
                }
            if tool_name in _SEND_TOOLS:
                send_count["n"] += 1
                if autonomous:
                    tool_input = input_data.get("tool_input", {})
                    msg_text = tool_input.get("text", "") if isinstance(tool_input, dict) else ""
                    # Reproduce the recall gate so we know the critic
                    # only runs when the recall gate passes.
                    if recall_count["n"] == 0 and _references_past_events(msg_text):
                        emitted.append(
                            {
                                "event": "send_blocked_no_recall",
                                "tool": tool_name,
                            }
                        )
                        return {"decision": "block", "reason": "recall first"}

                    # Critic gate — the slice under test.
                    if critic_enabled and msg_text and len(msg_text) >= 20:
                        critic_calls.append((msg_text, {"tool": tool_name}))
                        verdict = await verdict_fn(msg_text, {"tool": tool_name})
                        if verdict.decision != "pass":
                            emitted.append(
                                {
                                    "event": "critic_blocked",
                                    "tool": tool_name,
                                    "verdict": verdict.decision,
                                    "reason": verdict.reason,
                                }
                            )
                            return {
                                "decision": "block",
                                "reason": (f"Critic ({verdict.decision}): {verdict.reason}"),
                            }
            return {}

        return _hook, emitted, critic_calls

    @staticmethod
    def _verdict_factory(decision: str, reason: str = "") -> Any:
        from luke.critic import CriticVerdict

        async def _v(text: str, ctx: dict[str, Any]) -> CriticVerdict:
            return CriticVerdict(decision, reason)

        return _v

    async def test_calls_critic_on_autonomous_send(self) -> None:
        called: list[str] = []

        async def _verdict(text: str, ctx: dict[str, Any]) -> Any:
            from luke.critic import CriticVerdict

            called.append(text)
            return CriticVerdict("pass", "")

        hook, emitted, _calls = self._build_hook(
            autonomous=True, critic_enabled=True, verdict_fn=_verdict
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "Heads up, your 3pm moved to Vega room upstairs.",
                },
                "tool_use_id": "tu_c1",
            },
            "tu_c1",
            {},
        )
        assert result == {}
        assert len(called) == 1
        assert emitted == []

    async def test_blocks_send_on_revise_verdict(self) -> None:
        hook, emitted, _calls = self._build_hook(
            autonomous=True,
            critic_enabled=True,
            verdict_fn=self._verdict_factory("revise", "tone too chipper"),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": ("Absolutely! Great question — heads up your 3pm moved."),
                },
                "tool_use_id": "tu_c2",
            },
            "tu_c2",
            {},
        )
        assert result.get("decision") == "block"
        assert "Critic (revise)" in result.get("reason", "")
        assert "tone too chipper" in result.get("reason", "")
        assert len(emitted) == 1
        assert emitted[0]["event"] == "critic_blocked"
        assert emitted[0]["verdict"] == "revise"

    async def test_blocks_send_on_block_verdict(self) -> None:
        hook, emitted, _calls = self._build_hook(
            autonomous=True,
            critic_enabled=True,
            verdict_fn=self._verdict_factory("block", "filler boilerplate"),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "I apologize for the inconvenience, please bear with us.",
                },
                "tool_use_id": "tu_c3",
            },
            "tu_c3",
            {},
        )
        assert result.get("decision") == "block"
        assert "Critic (block)" in result.get("reason", "")
        assert len(emitted) == 1
        assert emitted[0]["verdict"] == "block"

    async def test_passes_through_on_pass_verdict(self) -> None:
        hook, emitted, calls = self._build_hook(
            autonomous=True,
            critic_enabled=True,
            verdict_fn=self._verdict_factory("pass", ""),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "Heads up — your 3pm moved to Vega room upstairs.",
                },
                "tool_use_id": "tu_c4",
            },
            "tu_c4",
            {},
        )
        assert result == {}
        assert len(calls) == 1
        assert emitted == []

    async def test_skips_critic_when_disabled(self) -> None:
        # If the critic would have blocked but it's disabled, send passes.
        hook, emitted, calls = self._build_hook(
            autonomous=True,
            critic_enabled=False,
            verdict_fn=self._verdict_factory("block", "would have blocked"),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "I apologize for the inconvenience, please bear with us.",
                },
                "tool_use_id": "tu_c5",
            },
            "tu_c5",
            {},
        )
        assert result == {}
        assert calls == []  # verdict_fn never invoked
        assert emitted == []

    async def test_skips_critic_for_short_text(self) -> None:
        hook, emitted, calls = self._build_hook(
            autonomous=True,
            critic_enabled=True,
            verdict_fn=self._verdict_factory("block", "would have blocked"),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {"text": "ok"},  # < 20 chars
                "tool_use_id": "tu_c6",
            },
            "tu_c6",
            {},
        )
        assert result == {}
        assert calls == []
        assert emitted == []

    async def test_skips_critic_when_not_autonomous(self) -> None:
        # Non-autonomous sends bypass all autonomous-only gates including critic.
        hook, emitted, calls = self._build_hook(
            autonomous=False,
            critic_enabled=True,
            verdict_fn=self._verdict_factory("block", "would have blocked"),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "I apologize for the inconvenience, please bear with us.",
                },
                "tool_use_id": "tu_c7",
            },
            "tu_c7",
            {},
        )
        assert result == {}
        assert calls == []
        assert emitted == []


class TestFreshnessGate:
    """Test the freshness gate inside _pre_tool_hook (L1).

    The gate fetches the user's latest inbound messages and asks the
    freshness critic whether the draft is coherent with them. We
    reproduce the gate as a faithful closure (same pattern as
    TestCriticGate) so behavior is exercised in isolation from the SDK.
    """

    @staticmethod
    def _build_hook(
        *,
        autonomous: bool,
        freshness_enabled: bool,
        recent_msgs: list[dict[str, Any]],
        check_fn: Any,
        window_minutes: int = 15,
    ) -> tuple[Any, list[dict[str, Any]], list[tuple[str, list[dict[str, Any]]]]]:
        """Construct a faithful copy of _pre_tool_hook's freshness gate.

        Returns (hook, emitted_events, freshness_calls).
        """
        from datetime import UTC, datetime

        from luke.agent import _SEND_TOOLS
        from luke.config import settings as _s

        send_count: dict[str, int] = {"n": 0}
        emitted: list[dict[str, Any]] = []
        fresh_calls: list[tuple[str, list[dict[str, Any]]]] = []

        async def _hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            tool_name = input_data["tool_name"]
            if tool_name in _SEND_TOOLS:
                send_count["n"] += 1
                if autonomous:
                    tool_input = input_data.get("tool_input", {})
                    msg_text = tool_input.get("text", "") if isinstance(tool_input, dict) else ""
                    if freshness_enabled and msg_text and len(msg_text) >= 30:
                        user_msgs = [
                            r for r in recent_msgs if r.get("sender_name") != _s.assistant_name
                        ][-2:]
                        if user_msgs:
                            try:
                                latest_ts = str(user_msgs[-1].get("timestamp", ""))
                                latest = datetime.fromisoformat(latest_ts)
                                if latest.tzinfo is None:
                                    latest = latest.replace(tzinfo=UTC)
                                age_minutes = (datetime.now(UTC) - latest).total_seconds() / 60
                            except ValueError, TypeError:
                                age_minutes = 999.0
                            if age_minutes <= window_minutes:
                                fresh_calls.append((msg_text, user_msgs))
                                verdict = await check_fn(msg_text, user_msgs)
                                if verdict.decision != "pass":
                                    emitted.append(
                                        {
                                            "event": "freshness_blocked",
                                            "tool": tool_name,
                                            "verdict": verdict.decision,
                                            "reason": verdict.reason,
                                        }
                                    )
                                    return {
                                        "decision": "block",
                                        "reason": (
                                            f"Freshness ({verdict.decision}): {verdict.reason}"
                                        ),
                                    }
            return {}

        return _hook, emitted, fresh_calls

    @staticmethod
    def _verdict_factory(decision: str, reason: str = "") -> Any:
        from luke.critic import CriticVerdict

        async def _v(text: str, user_msgs: list[dict[str, Any]]) -> CriticVerdict:
            return CriticVerdict(decision, reason)

        return _v

    @staticmethod
    def _fresh_user_msg(content: str, age_seconds: float = 30.0) -> dict[str, Any]:
        from datetime import UTC, datetime, timedelta

        ts = (datetime.now(UTC) - timedelta(seconds=age_seconds)).isoformat()
        return {
            "sender_name": "Filipe",
            "content": content,
            "timestamp": ts,
        }

    @staticmethod
    def _stale_user_msg(content: str, age_minutes: float = 60.0) -> dict[str, Any]:
        from datetime import UTC, datetime, timedelta

        ts = (datetime.now(UTC) - timedelta(minutes=age_minutes)).isoformat()
        return {
            "sender_name": "Filipe",
            "content": content,
            "timestamp": ts,
        }

    async def test_calls_freshness_when_window_matches(self) -> None:
        recent = [self._fresh_user_msg("never mind, found it")]
        hook, emitted, calls = self._build_hook(
            autonomous=True,
            freshness_enabled=True,
            recent_msgs=recent,
            check_fn=self._verdict_factory("pass", ""),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": ("The meeting is at 3pm in Vega room upstairs."),
                },
                "tool_use_id": "tu_f1",
            },
            "tu_f1",
            {},
        )
        assert result == {}
        assert len(calls) == 1
        assert calls[0][1][0]["content"] == "never mind, found it"
        assert emitted == []

    async def test_skips_when_no_recent_user_messages(self) -> None:
        # Last user message is far older than the window.
        recent = [self._stale_user_msg("hey", age_minutes=120.0)]
        hook, emitted, calls = self._build_hook(
            autonomous=True,
            freshness_enabled=True,
            recent_msgs=recent,
            check_fn=self._verdict_factory("block", "would have blocked"),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "Heads up — your 3pm moved to Vega room upstairs.",
                },
                "tool_use_id": "tu_f2",
            },
            "tu_f2",
            {},
        )
        assert result == {}
        assert calls == []
        assert emitted == []

    async def test_skips_when_disabled(self) -> None:
        recent = [self._fresh_user_msg("never mind")]
        hook, emitted, calls = self._build_hook(
            autonomous=True,
            freshness_enabled=False,
            recent_msgs=recent,
            check_fn=self._verdict_factory("block", "would have blocked"),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "Heads up — your 3pm moved to Vega room upstairs.",
                },
                "tool_use_id": "tu_f3",
            },
            "tu_f3",
            {},
        )
        assert result == {}
        assert calls == []
        assert emitted == []

    async def test_skips_for_short_drafts(self) -> None:
        recent = [self._fresh_user_msg("never mind")]
        hook, emitted, calls = self._build_hook(
            autonomous=True,
            freshness_enabled=True,
            recent_msgs=recent,
            check_fn=self._verdict_factory("block", "would have blocked"),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {"text": "ok cool"},  # < 30 chars
                "tool_use_id": "tu_f4",
            },
            "tu_f4",
            {},
        )
        assert result == {}
        assert calls == []
        assert emitted == []

    async def test_skips_when_not_autonomous(self) -> None:
        recent = [self._fresh_user_msg("never mind")]
        hook, emitted, calls = self._build_hook(
            autonomous=False,
            freshness_enabled=True,
            recent_msgs=recent,
            check_fn=self._verdict_factory("block", "would have blocked"),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "Heads up — your 3pm moved to Vega room upstairs.",
                },
                "tool_use_id": "tu_f5",
            },
            "tu_f5",
            {},
        )
        assert result == {}
        assert calls == []
        assert emitted == []

    async def test_blocks_send_on_block_verdict(self) -> None:
        recent = [self._fresh_user_msg("never mind, found it")]
        hook, emitted, calls = self._build_hook(
            autonomous=True,
            freshness_enabled=True,
            recent_msgs=recent,
            check_fn=self._verdict_factory("block", "answers a cancelled question"),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "The meeting is at 3pm in Vega room upstairs.",
                },
                "tool_use_id": "tu_f6",
            },
            "tu_f6",
            {},
        )
        assert result.get("decision") == "block"
        assert "Freshness (block)" in result.get("reason", "")
        assert "cancelled" in result.get("reason", "")
        assert len(emitted) == 1
        assert emitted[0]["event"] == "freshness_blocked"
        assert emitted[0]["verdict"] == "block"
        assert len(calls) == 1

    async def test_blocks_send_on_revise_verdict(self) -> None:
        recent = [self._fresh_user_msg("actually, what about tomorrow?")]
        hook, emitted, _calls = self._build_hook(
            autonomous=True,
            freshness_enabled=True,
            recent_msgs=recent,
            check_fn=self._verdict_factory("revise", "answers earlier question"),
        )
        result = await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "Today's meeting is at 3pm in Vega room upstairs.",
                },
                "tool_use_id": "tu_f7",
            },
            "tu_f7",
            {},
        )
        assert result.get("decision") == "block"
        assert "Freshness (revise)" in result.get("reason", "")
        assert len(emitted) == 1
        assert emitted[0]["verdict"] == "revise"

    async def test_excludes_luke_messages_from_user_latest(self) -> None:
        # Even though Luke's message is the most recent, the gate only
        # considers messages from senders != assistant_name.
        from luke.config import settings as _s

        recent = [
            self._fresh_user_msg("never mind, found it", age_seconds=120.0),
            {
                "sender_name": _s.assistant_name,
                "content": "got it, standing down",
                "timestamp": self._fresh_user_msg("x", age_seconds=10.0)["timestamp"],
            },
        ]
        hook, _emitted, calls = self._build_hook(
            autonomous=True,
            freshness_enabled=True,
            recent_msgs=recent,
            check_fn=self._verdict_factory("pass", ""),
        )
        await hook(
            {
                "tool_name": "mcp__luke__send_message",
                "tool_input": {
                    "text": "The meeting is at 3pm in Vega room upstairs.",
                },
                "tool_use_id": "tu_f8",
            },
            "tu_f8",
            {},
        )
        # Only one call, and the latest is Filipe's message.
        assert len(calls) == 1
        user_latest = calls[0][1]
        assert all(m.get("sender_name") != _s.assistant_name for m in user_latest)
        assert user_latest[-1]["content"] == "never mind, found it"


class TestCleanStreamingText:
    def test_strips_complete_internal_tags(self) -> None:
        from luke.agent import _clean_streaming_text

        assert _clean_streaming_text("Hello <internal>secret</internal> world") == "Hello  world"

    def test_strips_unclosed_internal_tag(self) -> None:
        from luke.agent import _clean_streaming_text

        assert _clean_streaming_text("Hello <internal>partial thinking...") == "Hello"

    def test_no_tags(self) -> None:
        from luke.agent import _clean_streaming_text

        assert _clean_streaming_text("Just plain text") == "Just plain text"

    def test_empty(self) -> None:
        from luke.agent import _clean_streaming_text

        assert _clean_streaming_text("") == ""

    def test_only_internal(self) -> None:
        from luke.agent import _clean_streaming_text

        assert _clean_streaming_text("<internal>all hidden</internal>") == ""

    def test_mixed_complete_and_partial(self) -> None:
        from luke.agent import _clean_streaming_text

        text = "A <internal>done</internal> B <internal>still going"
        assert _clean_streaming_text(text) == "A  B"


# ---------------------------------------------------------------------------
# Scheduled-task duplicate gate
# ---------------------------------------------------------------------------


class TestTaskOverlap:
    def test_same_deliverable_high_overlap(self) -> None:
        # Realistic re-stage: same core noun phrase, minor tail divergence.
        a = "Send Filipe the Prerna Monday negotiation call brief with talking points"
        b = "Send Filipe the Prerna Monday negotiation brief and talking points tonight"
        assert _task_overlap(a, b) >= 0.7

    def test_length_asymmetric_duplicate_still_caught(self) -> None:
        # Terse restage vs verbose original — containment beats Jaccard here.
        short = "visa interview reminder tomorrow morning Ballsbridge consulate"
        long = (
            "Remind Filipe the US B1 visa interview is tomorrow at the "
            "Ballsbridge consulate; bring DS-160 confirmation and passport"
        )
        assert _task_overlap(short, long) >= 0.7

    def test_unrelated_prompts_low_overlap(self) -> None:
        a = "Remind Filipe to book the dentist appointment next week"
        b = "Check the CarGurus offer stock-vesting schedule details"
        assert _task_overlap(a, b) < 0.7

    def test_short_prompts_return_zero(self) -> None:
        # Fewer than 4 significant words on either side -> not comparable.
        assert _task_overlap("remind me tonight", "remind me later") == 0.0


class TestDuplicatePendingTask:
    def _existing(self, **over: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "id": "abc123",
            "chat_id": "1",
            "prompt": "Send Filipe the Prerna Monday-call negotiation brief tonight",
            "schedule_type": "once",
            "schedule_value": "2026-07-26T19:00:00+00:00",
            "status": "active",
            "last_run": None,
            "created_at": "2026-07-25T00:00:00+00:00",
        }
        base.update(over)
        return base

    def test_once_near_duplicate_within_window_blocked(self) -> None:
        new = {
            # Re-stage of the same brief; shares the core noun phrase.
            "prompt": "Send Filipe the Prerna Monday negotiation call brief with talking points",
            "schedule_type": "once",
            "schedule_value": "2026-07-26T20:00:00+00:00",  # 1h later
        }
        dup = _duplicate_pending_task(new, [self._existing()])
        assert dup is not None and dup["id"] == "abc123"

    def test_once_same_deliverable_far_apart_allowed(self) -> None:
        new = {
            "prompt": "Deliver the Prerna call negotiation brief to Filipe for dinner",
            "schedule_type": "once",
            "schedule_value": "2026-07-27T09:00:00+00:00",  # ~14h later, > 8h
        }
        assert _duplicate_pending_task(new, [self._existing()]) is None

    def test_different_schedule_type_allowed(self) -> None:
        new = {
            "prompt": "Send Filipe the Prerna Monday-call negotiation brief tonight",
            "schedule_type": "cron",
            "schedule_value": "0 19 * * *",
        }
        assert _duplicate_pending_task(new, [self._existing()]) is None

    def test_low_text_overlap_allowed(self) -> None:
        new = {
            "prompt": "Remind Filipe about the dentist appointment this evening",
            "schedule_type": "once",
            "schedule_value": "2026-07-26T19:30:00+00:00",
        }
        assert _duplicate_pending_task(new, [self._existing()]) is None

    def test_completed_task_ignored(self) -> None:
        new = {
            "prompt": "Deliver the Prerna call negotiation brief to Filipe for dinner",
            "schedule_type": "once",
            "schedule_value": "2026-07-26T20:00:00+00:00",
        }
        done = self._existing(status="completed")
        assert _duplicate_pending_task(new, [done]) is None

    def test_cron_identical_cadence_blocked(self) -> None:
        existing = self._existing(
            schedule_type="cron",
            schedule_value="0 9 * * 1",
            prompt="Weekly Monday review nudge for Filipe strategic priorities",
        )
        new = {
            "prompt": "Monday weekly review nudge covering Filipe strategic priorities",
            "schedule_type": "cron",
            "schedule_value": "0 9 * * 1",
        }
        dup = _duplicate_pending_task(new, [existing])
        assert dup is not None and dup["id"] == "abc123"

    def test_cron_different_cadence_allowed(self) -> None:
        existing = self._existing(
            schedule_type="cron",
            schedule_value="0 9 * * 1",
            prompt="Weekly Monday review nudge for Filipe strategic priorities",
        )
        new = {
            "prompt": "Monday weekly review nudge covering Filipe strategic priorities",
            "schedule_type": "cron",
            "schedule_value": "0 21 * * 5",  # different cadence
        }
        assert _duplicate_pending_task(new, [existing]) is None

    def test_short_new_prompt_skipped(self) -> None:
        new = {
            "prompt": "ping me",
            "schedule_type": "once",
            "schedule_value": "2026-07-26T20:00:00+00:00",
        }
        assert _duplicate_pending_task(new, [self._existing()]) is None


class TestContextQuery:
    """The gate/retrieval query must read the user's actual message, not the
    injected memory blob. Regression for the 2026-07-26 bug where memory context
    was prepended to `prompt` (str) or inserted at index 0 (multimodal list),
    causing memory retrieval and the file-artifact/source-read Stop gates to read
    the memory blob instead of what the user typed — dropping their request.
    """

    MEMORY_BLOB = (
        "## Key Entities\n[entity-cargurus] Prerna is Filipe's manager...\n"
        "## Active Insights\n[dream-...] some long injected memory context\n\n"
    )

    def test_prefers_user_text_over_str_prompt_with_memory(self) -> None:
        # str path: memory_context prepended -> prompt carries the blob
        user_msg = "read the Prerna email and tell me what she wants"
        polluted = f"{self.MEMORY_BLOB}\n\n{user_msg}"
        assert _context_query(polluted, user_text=user_msg) == user_msg

    def test_prefers_user_text_over_list_prompt_with_memory(self) -> None:
        # multimodal path: memory inserted at index 0, real msg pushed to index 1
        user_msg = "make me a PDF of this"
        polluted = [
            {"type": "text", "text": f"{self.MEMORY_BLOB}\n\n"},
            {"type": "text", "text": user_msg},
        ]
        assert _context_query(polluted, user_text=user_msg) == user_msg

    def test_gate_fires_on_user_text_not_blob(self) -> None:
        # The concrete failure: with the blob as query the source-read gate misses;
        # with user_text it fires. Proves the fix restores gate correctness.
        user_msg = "can you read the attached pdf and summarise the thread"
        polluted = [
            {"type": "text", "text": f"{self.MEMORY_BLOB}\n\n"},
            {"type": "text", "text": user_msg},
        ]
        assert _requests_source_read(_context_query(polluted, user_text=None)) is False
        assert _requests_source_read(_context_query(polluted, user_text=user_msg)) is True

    def test_falls_back_to_str_prompt_when_no_user_text(self) -> None:
        # Autonomous/scheduled callers pass no user_text -> use prompt as-is
        assert _context_query("do the nightly reflection", user_text=None) == (
            "do the nightly reflection"
        )

    def test_falls_back_to_first_block_for_list_without_user_text(self) -> None:
        blocks = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert _context_query(blocks, user_text=None) == "hello"

    def test_empty_list_without_user_text_is_safe(self) -> None:
        assert _context_query([], user_text=None) == ""


class TestCronLocalTimeGate:
    """Cron values are UTC; Filipe's stated times are Dublin wall-clock.

    Regression guard for the 2026-08-03 Kagan nightly-lecture near-miss: a
    `30 21 * * *` cron for a "21:30 Dublin" send would have fired at 22:30 IST
    every night for 23 nights.
    """

    # August 2026 — Ireland on IST (UTC+1).
    NOW = datetime(2026, 8, 3, 20, 15, tzinfo=UTC)
    # Late November 2026 — Ireland back on GMT (UTC+0).
    NOW_GMT = datetime(2026, 11, 20, 20, 15, tzinfo=UTC)

    def _inp(self, value: str, prompt: str) -> dict[str, str]:
        return {"schedule_type": "cron", "schedule_value": value, "prompt": prompt}

    def test_utc_hour_used_for_local_time_is_blocked(self) -> None:
        msg = _cron_local_time_mismatch(
            self._inp("30 21 * * *", "Kagan nightly lecture send at 21:30 Dublin."), self.NOW
        )
        assert msg is not None
        assert "22:30 Dublin" in msg
        assert "`30 20 * * *`" in msg

    def test_correct_conversion_passes(self) -> None:
        assert (
            _cron_local_time_mismatch(
                self._inp("30 20 * * *", "Kagan nightly lecture send at 21:30 Dublin."), self.NOW
            )
            is None
        )

    def test_same_wall_clock_flips_across_dst(self) -> None:
        """`30 20` is 21:30 in August but 20:30 in November — the gate follows DST."""
        inp = self._inp("30 20 * * *", "Nightly send at 21:30 Dublin.")
        assert _cron_local_time_mismatch(inp, self.NOW) is None
        winter = _cron_local_time_mismatch(inp, self.NOW_GMT)
        assert winter is not None and "20:30 Dublin" in winter
        assert "`30 21 * * *`" in winter

    def test_no_declared_local_time_passes(self) -> None:
        assert (
            _cron_local_time_mismatch(
                self._inp("30 21 * * *", "Kagan nightly lecture send. Read the plan file."),
                self.NOW,
            )
            is None
        )

    def test_bare_time_without_locality_marker_passes(self) -> None:
        """A time with no locality marker is not a declaration of intent."""
        assert (
            _cron_local_time_mismatch(
                self._inp("30 6 * * *", "Remind Filipe his visa interview is at 07:45."), self.NOW
            )
            is None
        )

    def test_wildcard_and_step_hours_pass(self) -> None:
        for value in ("*/15 * * * *", "30 */2 * * *", "0 8,12,16,20 * * *", "30 8-10 * * *"):
            assert (
                _cron_local_time_mismatch(self._inp(value, "Fires 09:00 Dublin."), self.NOW) is None
            )

    def test_multiple_declared_times_any_match_passes(self) -> None:
        assert (
            _cron_local_time_mismatch(
                self._inp("30 20 * * *", "Fires 21:30 Dublin (cron is UTC: 30 20 = 21:30 IST)."),
                self.NOW,
            )
            is None
        )

    def test_non_cron_types_ignored(self) -> None:
        assert (
            _cron_local_time_mismatch(
                {
                    "schedule_type": "once",
                    "schedule_value": "2026-08-04T21:30:00+01:00",
                    "prompt": "Send at 21:30 Dublin.",
                },
                self.NOW,
            )
            is None
        )

    def test_midnight_crossing_flags_day_fields(self) -> None:
        msg = _cron_local_time_mismatch(
            self._inp("0 8 * * 1", "Monday review at 00:30 Dublin."), self.NOW
        )
        assert msg is not None and "crosses midnight" in msg


class TestCommitmentBlockEscalation:
    """The commitment gate must get LOUDER when I hammer it.

    Root cause, 2026-08-05 18:31: three blocks in 20 seconds on one draft
    (send_message, send_message, send_buttons), then the message was
    abandoned and Filipe never got it. A constant block reason gave the
    retry loop nothing new to react to.
    """

    def test_first_block_is_the_plain_reason(self) -> None:
        from luke.agent import _commitment_block_reason

        r = _commitment_block_reason(1)
        assert "no agent or scheduled task was spawned" in r.replace("\n", " ")
        assert "THIS IS BLOCK" not in r

    def test_repeat_blocks_escalate_and_name_the_two_exits(self) -> None:
        from luke.agent import _commitment_block_reason

        r = _commitment_block_reason(2)
        assert "THIS IS BLOCK #2 THIS TURN" in r
        assert "switching to a different" in r  # tool-switch evasion called out
        assert "schedule_task" in r  # exit (a)
        assert "delete the commitment sentence" in r  # exit (b)
        assert "dropped message is a failure" in r  # no silent abandon

    def test_attempt_number_is_reported(self) -> None:
        from luke.agent import _commitment_block_reason

        assert "THIS IS BLOCK #3 THIS TURN" in _commitment_block_reason(3)
