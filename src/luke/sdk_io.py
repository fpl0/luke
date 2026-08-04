"""Plumbing shared by every place we hand options to the Claude Agent SDK."""

from __future__ import annotations

import re

import structlog
from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.get_logger()

# A stack frame from the CLI's own bundle — carries no signal for us.
_CLI_STACK_FRAME = re.compile(r"^\s+at\s")
_CLI_STDERR_MAX = 300


def cli_stderr(line: str) -> None:
    """Route the CLI subprocess's stderr through the log instead of luke.err.

    With no callback the SDK lets the subprocess inherit our stderr, so every
    `Error in hook callback` dumps its whole minified bundle straight into
    luke.err — 86% of that file (2.4MB) was those dumps and their stack frames.
    Keep the first line of each error, drop the frames, cap the length.
    """
    if _CLI_STACK_FRAME.match(line):
        return
    line = line.strip()
    if not line:
        return
    truncated = len(line) > _CLI_STDERR_MAX
    log.warning(
        "cli_stderr",
        line=line[:_CLI_STDERR_MAX],
        truncated=truncated,
        original_len=len(line) if truncated else None,
    )
