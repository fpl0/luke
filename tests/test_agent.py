"""Tests for luke.agent — send_long_message, _send_chunk, _build_stop_hook, tools."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot
from claude_agent_sdk.types import SyncHookJSONOutput

from luke.agent import (
    _AUTO_SKILL_THRESHOLD,
    _INTERNAL_RE,
    _VALID_MEMORY_TYPES,
    AgentResult,
    _build_stop_hook,
    _ok,
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

        from luke.agent import db

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

        from luke.agent import db

        async def _post_tool_failure_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            tool_name = input_data["tool_name"]
            error = input_data.get("error", "unknown")
            payload: dict[str, Any] = {"tool": tool_name, "success": False, "error": str(error)[:500]}
            db.emit_event("tool_failure", json.dumps(payload))
            return {}

        result = await _post_tool_failure_hook(
            {"tool_name": "Read", "tool_use_id": "tu_456", "tool_input": {}, "error": "File not found"},
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

        from luke.agent import db

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

        from luke.agent import db

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
                json.dumps({
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "duration_ms": duration_ms,
                }),
            )
            return {}

        result = await _subagent_stop_hook(
            {"agent_id": "sa_002", "agent_type": "coder", "agent_transcript_path": "/tmp/t", "stop_hook_active": False},
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
            "before",
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
            if tool_name in _SEND_TOOLS:
                send_count["n"] += 1
                if autonomous:
                    tool_input = input_data.get("tool_input", {})
                    msg_text = (
                        tool_input.get("text", "")
                        if isinstance(tool_input, dict)
                        else ""
                    )
                    if recall_count["n"] == 0 and _references_past_events(msg_text):
                        emitted.append({
                            "event": "send_blocked_no_recall",
                            "tool": tool_name,
                            "preview": msg_text[:100],
                        })
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
            if tool_name in _SEND_TOOLS:
                send_count["n"] += 1
                if autonomous:
                    tool_input = input_data.get("tool_input", {})
                    msg_text = (
                        tool_input.get("text", "")
                        if isinstance(tool_input, dict)
                        else ""
                    )
                    # Reproduce the recall gate so we know the critic
                    # only runs when the recall gate passes.
                    if recall_count["n"] == 0 and _references_past_events(msg_text):
                        emitted.append({
                            "event": "send_blocked_no_recall",
                            "tool": tool_name,
                        })
                        return {"decision": "block", "reason": "recall first"}

                    # Critic gate — the slice under test.
                    if critic_enabled and msg_text and len(msg_text) >= 20:
                        critic_calls.append((msg_text, {"tool": tool_name}))
                        verdict = await verdict_fn(msg_text, {"tool": tool_name})
                        if verdict.decision != "pass":
                            emitted.append({
                                "event": "critic_blocked",
                                "tool": tool_name,
                                "verdict": verdict.decision,
                                "reason": verdict.reason,
                            })
                            return {
                                "decision": "block",
                                "reason": (
                                    f"Critic ({verdict.decision}): "
                                    f"{verdict.reason}"
                                ),
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
                    "text": (
                        "Absolutely! Great question — heads up your 3pm moved."
                    ),
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
                    msg_text = (
                        tool_input.get("text", "")
                        if isinstance(tool_input, dict)
                        else ""
                    )
                    if freshness_enabled and msg_text and len(msg_text) >= 30:
                        user_msgs = [
                            r
                            for r in recent_msgs
                            if r.get("sender_name") != _s.assistant_name
                        ][-2:]
                        if user_msgs:
                            try:
                                latest_ts = str(
                                    user_msgs[-1].get("timestamp", "")
                                )
                                latest = datetime.fromisoformat(latest_ts)
                                if latest.tzinfo is None:
                                    latest = latest.replace(tzinfo=UTC)
                                age_minutes = (
                                    datetime.now(UTC) - latest
                                ).total_seconds() / 60
                            except (ValueError, TypeError):
                                age_minutes = 999.0
                            if age_minutes <= window_minutes:
                                fresh_calls.append((msg_text, user_msgs))
                                verdict = await check_fn(msg_text, user_msgs)
                                if verdict.decision != "pass":
                                    emitted.append({
                                        "event": "freshness_blocked",
                                        "tool": tool_name,
                                        "verdict": verdict.decision,
                                        "reason": verdict.reason,
                                    })
                                    return {
                                        "decision": "block",
                                        "reason": (
                                            f"Freshness ({verdict.decision}): "
                                            f"{verdict.reason}"
                                        ),
                                    }
            return {}

        return _hook, emitted, fresh_calls

    @staticmethod
    def _verdict_factory(decision: str, reason: str = "") -> Any:
        from luke.critic import CriticVerdict

        async def _v(
            text: str, user_msgs: list[dict[str, Any]]
        ) -> CriticVerdict:
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
    def _stale_user_msg(
        content: str, age_minutes: float = 60.0
    ) -> dict[str, Any]:
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
                    "text": (
                        "The meeting is at 3pm in Vega room upstairs."
                    ),
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
            check_fn=self._verdict_factory(
                "block", "answers a cancelled question"
            ),
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
            check_fn=self._verdict_factory(
                "revise", "answers earlier question"
            ),
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
            self._fresh_user_msg(
                "never mind, found it", age_seconds=120.0
            ),
            {
                "sender_name": _s.assistant_name,
                "content": "got it, standing down",
                "timestamp": self._fresh_user_msg(
                    "x", age_seconds=10.0
                )["timestamp"],
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
        assert all(
            m.get("sender_name") != _s.assistant_name for m in user_latest
        )
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
