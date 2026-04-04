"""Tests for luke.agent — send_long_message, _send_chunk, _build_stop_hook, tools."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot

from luke.agent import (
    _AUTO_SKILL_THRESHOLD,
    _INTERNAL_RE,
    _VALID_MEMORY_TYPES,
    AgentResult,
    _build_stop_hook,
    _ok,
    send_long_message,
)
from claude_agent_sdk.types import SyncHookJSONOutput
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
        assert "Skill extraction" not in result["systemMessage"]

    async def test_skill_prompt_at_threshold(self) -> None:
        result = await self._call(_AUTO_SKILL_THRESHOLD, False)
        assert "Skill extraction" in result["systemMessage"]
        assert "procedure" in result["systemMessage"]

    async def test_no_skill_prompt_for_autonomous_runs(self) -> None:
        result = await self._call(_AUTO_SKILL_THRESHOLD + 10, True)
        assert "Skill extraction" not in result["systemMessage"]


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
