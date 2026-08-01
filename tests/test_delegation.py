"""Tests for the durable delegation loop — delegate → tasks table → relay.

A delegated job is a promise to Filipe, so the loop must ALWAYS close:
- delegate persists the job (survives restarts; the old in-process registry
  lost every in-flight job on deploy/crash, silently)
- the scheduler relays the result deterministically — never by trusting the
  model to remember to send
- a dead job reports its own death
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot

from luke.agent import AgentResult, format_delegation_prompt, parse_delegation
from luke.db import TaskRecord
from luke.scheduler import _run_task

# ---------------------------------------------------------------------------
# Envelope round-trip
# ---------------------------------------------------------------------------


class TestDelegationEnvelope:
    def test_round_trip_with_trigger(self) -> None:
        stored = format_delegation_prompt("research X thoroughly", 42)
        parsed = parse_delegation(stored)
        assert parsed == ("research X thoroughly", 42)

    def test_round_trip_without_trigger(self) -> None:
        stored = format_delegation_prompt("build the report", None)
        parsed = parse_delegation(stored)
        assert parsed == ("build the report", None)

    def test_multi_paragraph_body_preserved(self) -> None:
        body = "step one\n\nstep two\n\nstep three"
        parsed = parse_delegation(format_delegation_prompt(body, 7))
        assert parsed == (body, 7)

    def test_ordinary_prompt_is_not_a_delegation(self) -> None:
        assert parse_delegation("check the mail and report") is None
        assert parse_delegation("[Scheduled task] do stuff") is None

    def test_malformed_trigger_degrades_to_none(self) -> None:
        stored = "[delegation:v1]\ntrigger_msg_id: not-a-number\n\ndo the thing"
        parsed = parse_delegation(stored)
        assert parsed == ("do the thing", None)


# ---------------------------------------------------------------------------
# delegate tool — persists a durable once-task
# ---------------------------------------------------------------------------


class TestDelegateTool:
    @pytest.fixture()
    def tool_env(self, tmp_path: Any) -> dict[str, Any]:
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
        ):
            mock_settings.store_dir = store_dir
            mock_settings.luke_dir = tmp_path
            mock_settings.recall_content_limit = 2000

            from luke.agent import _build_tools

            _build_tools("12345", mock_bot)

        tools = {t._tool_name: t for t in captured_tools if hasattr(t, "_tool_name")}
        return {"tools": tools, "bot": mock_bot}

    async def test_delegate_creates_durable_once_task(self, tool_env: dict[str, Any]) -> None:
        t_delegate = tool_env["tools"]["delegate"]
        with patch("luke.agent.db") as mock_db:
            mock_db.create_task.return_value = "job-777"
            result = await t_delegate({"prompt": "summarize the RFC", "trigger_msg_id": 99})

        assert "job-777" in result["content"][0]["text"]
        chat_id, stored, stype, svalue = mock_db.create_task.call_args.args
        assert chat_id == "12345"
        assert stype == "once"
        # schedule_value is "now" — the job runs on the next wake, not later
        assert datetime.fromisoformat(svalue).tzinfo is not None
        # the stored prompt is a parseable delegation envelope
        assert parse_delegation(stored) == ("summarize the RFC", 99)

    async def test_delegate_without_trigger(self, tool_env: dict[str, Any]) -> None:
        t_delegate = tool_env["tools"]["delegate"]
        with patch("luke.agent.db") as mock_db:
            mock_db.create_task.return_value = "job-888"
            await t_delegate({"prompt": "audit the logs"})

        stored = mock_db.create_task.call_args.args[1]
        assert parse_delegation(stored) == ("audit the logs", None)


# ---------------------------------------------------------------------------
# _run_task — the relay side
# ---------------------------------------------------------------------------


def _delegation_task(prompt: str, trigger: int | None = 42) -> TaskRecord:
    return {
        "id": "job-1",
        "chat_id": "100",
        "prompt": format_delegation_prompt(prompt, trigger),
        "schedule_type": "once",
        "schedule_value": datetime.now(UTC).isoformat(),
        "status": "active",
        "last_run": None,
        "created_at": datetime.now(UTC).isoformat(),
    }


class TestDelegationRelay:
    async def test_result_text_is_relayed_as_reply(self) -> None:
        mock_bot = AsyncMock()
        result = AgentResult(texts=["the findings"])

        with (
            patch("luke.scheduler.run_agent", return_value=result) as mock_agent,
            patch("luke.scheduler.send_long_message") as mock_send,
            patch("luke.scheduler.db"),
        ):
            await _run_task(_delegation_task("research X"), mock_bot)

        # The job prompt tells the agent its text output IS the report
        assert "Delegated background job" in mock_agent.call_args.kwargs["prompt"]
        mock_send.assert_awaited_once()
        assert mock_send.call_args.kwargs["text"] == "the findings"
        assert mock_send.call_args.kwargs["reply_parameters"].message_id == 42

    async def test_job_that_already_sent_is_not_double_reported(self) -> None:
        mock_bot = AsyncMock()
        result = AgentResult(texts=[], sent_messages=2)

        with (
            patch("luke.scheduler.run_agent", return_value=result),
            patch("luke.scheduler.send_long_message") as mock_send,
            patch("luke.scheduler.db"),
        ):
            await _run_task(_delegation_task("send it yourself"), mock_bot)

        mock_send.assert_not_awaited()

    async def test_empty_result_is_flagged_not_silent(self) -> None:
        mock_bot = AsyncMock()
        result = AgentResult(texts=[], sent_messages=0)

        with (
            patch("luke.scheduler.run_agent", return_value=result),
            patch("luke.scheduler.send_long_message") as mock_send,
            patch("luke.scheduler.db"),
        ):
            await _run_task(_delegation_task("do nothing", trigger=None), mock_bot)

        mock_send.assert_awaited_once()
        assert "no output" in mock_send.call_args.kwargs["text"]
        assert "reply_parameters" not in mock_send.call_args.kwargs

    async def test_dead_job_reports_its_own_death(self) -> None:
        mock_bot = AsyncMock()

        with (
            patch("luke.scheduler.run_agent", side_effect=RuntimeError("boom")),
            patch("luke.scheduler.send_long_message") as mock_send,
            patch("luke.scheduler.db") as mock_db,
        ):
            mock_db.increment_task_failures.return_value = 1
            await _run_task(_delegation_task("explode"), mock_bot)

        mock_send.assert_awaited_once()
        text = mock_send.call_args.kwargs["text"]
        assert "died" in text
        assert "boom" in text
        # once-jobs never retry — completed even on failure
        mock_db.update_task_status.assert_called_once_with("job-1", "completed")

    async def test_ordinary_task_untouched_by_delegation_path(self) -> None:
        mock_bot = AsyncMock()
        result = MagicMock()
        result.texts = ["dropped text"]
        result.sent_messages = 0

        task: TaskRecord = {
            "id": "cron-1",
            "chat_id": "100",
            "prompt": "check the mail",
            "schedule_type": "cron",
            "schedule_value": "0 * * * *",
            "status": "active",
            "last_run": None,
            "created_at": datetime.now(UTC).isoformat(),
        }

        with (
            patch("luke.scheduler.run_agent", return_value=result) as mock_agent,
            patch("luke.scheduler.send_long_message") as mock_send,
            patch("luke.scheduler.db"),
        ):
            await _run_task(task, mock_bot)

        assert "Scheduled task" in mock_agent.call_args.kwargs["prompt"]
        mock_send.assert_not_awaited()
