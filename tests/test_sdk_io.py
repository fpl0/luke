"""The CLI stderr filter — what it drops, what it keeps, what it truncates."""

from __future__ import annotations

import pytest

from luke import sdk_io
from luke.sdk_io import _CLI_STDERR_MAX, cli_stderr


@pytest.fixture
def captured(monkeypatch) -> list[dict]:
    """Collect what the filter would have written to the log."""
    calls: list[dict] = []

    class _Log:
        def warning(self, event: str, **kw) -> None:
            calls.append({"event": event, **kw})

    monkeypatch.setattr(sdk_io, "log", _Log())
    return calls


def test_stack_frames_are_dropped(captured):
    """1800+ of these per err file; they name the CLI's own bundle, not ours."""
    for line in (
        "    at next (/opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js:1:2)",
        "      at sendRequest (file:///cli.js:9:9)",
        "\tat Ev (cli.js:3:4)",
    ):
        cli_stderr(line)
    assert captured == []


def test_blank_lines_are_dropped(captured):
    for line in ("", "   ", "\n"):
        cli_stderr(line)
    assert captured == []


def test_hook_error_keeps_its_first_line(captured):
    """The signal is 'hook_1 blew up' — not the 10KB bundle after it."""
    bundle = "x" * 12_000
    cli_stderr(f"Error in hook callback hook_1: {bundle}")

    assert len(captured) == 1
    entry = captured[0]
    assert entry["event"] == "cli_stderr"
    assert entry["line"].startswith("Error in hook callback hook_1:")
    assert len(entry["line"]) == _CLI_STDERR_MAX
    assert entry["truncated"] is True
    assert entry["original_len"] > 12_000


def test_short_error_passes_through_whole(captured):
    cli_stderr("Received SIGTERM signal")

    assert captured == [
        {
            "event": "cli_stderr",
            "line": "Received SIGTERM signal",
            "truncated": False,
            "original_len": None,
        }
    ]


def test_surrounding_whitespace_is_stripped_but_line_kept(captured):
    """Only leading-whitespace *stack frames* are dropped, not any indent."""
    cli_stderr("  Error output: Check stderr output for details  \n")

    assert len(captured) == 1
    assert captured[0]["line"] == "Error output: Check stderr output for details"


def test_replay_of_live_err_file_collapses_the_noise(captured):
    """End-to-end shape check against the mix seen in the live luke.err."""
    lines = [
        "Error in hook callback hook_1: " + "b" * 9_000,
        "    at next (cli.js:1:1)",
        "    at t (cli.js:2:2)",
        "    at BUm (cli.js:3:3)",
        "",
        "Received SIGTERM signal",
    ]
    for line in lines:
        cli_stderr(line)

    # 6 lines in, 2 log records out, neither carrying a bundle.
    assert len(captured) == 2
    assert all(len(c["line"]) <= _CLI_STDERR_MAX for c in captured)
