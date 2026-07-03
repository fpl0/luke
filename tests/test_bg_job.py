"""Tests for _run_bg_job — a delegated background job must ALWAYS report back.

Guards commit 460268e: a job that completes with empty output used to `return`
silently. These tests prove all three exit paths message the chat.
"""

from unittest.mock import AsyncMock, patch

from aiogram import Bot

from luke.agent import AgentResult, _run_bg_job


async def test_bg_job_empty_output_still_reports() -> None:
    """A job that runs but produces no text must still send a message."""
    mock_bot = AsyncMock(spec=Bot)
    empty = AgentResult(texts=[])
    with (
        patch("luke.agent.run_agent", AsyncMock(return_value=empty)),
        patch("luke.agent.send_long_message", AsyncMock()) as m_send,
    ):
        await _run_bg_job("job_empty", "123", "do nothing", 42, mock_bot)

    m_send.assert_awaited_once()
    assert "no output" in m_send.call_args.kwargs["text"]


async def test_bg_job_error_reports() -> None:
    """A job that raises must report the error, not vanish."""
    mock_bot = AsyncMock(spec=Bot)
    with (
        patch("luke.agent.run_agent", AsyncMock(side_effect=RuntimeError("boom"))),
        patch("luke.agent.send_long_message", AsyncMock()) as m_send,
    ):
        await _run_bg_job("job_err", "123", "explode", None, mock_bot)

    m_send.assert_awaited_once()
    assert "error" in m_send.call_args.kwargs["text"].lower()


async def test_bg_job_normal_output_sends_text() -> None:
    """The happy path is unchanged: real output is sent verbatim."""
    mock_bot = AsyncMock(spec=Bot)
    res = AgentResult(texts=["the real answer"])
    with (
        patch("luke.agent.run_agent", AsyncMock(return_value=res)),
        patch("luke.agent.send_long_message", AsyncMock()) as m_send,
    ):
        await _run_bg_job("job_ok", "123", "work", 7, mock_bot)

    m_send.assert_awaited_once()
    assert m_send.call_args.kwargs["text"] == "the real answer"
