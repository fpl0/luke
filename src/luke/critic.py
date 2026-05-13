"""Outbound message critic — last-mile quality gate for autonomous sends.

Runs a cheap one-shot Haiku query over a draft message and returns a
pass/revise/block verdict. Wired into ``_pre_tool_hook`` as the final
gate after cheap regex/state checks.

Fail-open: on any error (network, parse, timeout) the critic returns
``pass`` so a misbehaving critic doesn't silence Luke. Single mechanism,
no fallback layers — per Filipe's coherence preference.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

import structlog
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)
from structlog.stdlib import BoundLogger

from .config import settings

log: BoundLogger = structlog.get_logger()


_CRITIC_SYSTEM_PROMPT = (
    "You are a critic for outgoing messages from Luke, a personal AI agent.\n"
    "Luke's voice: warm, unhurried, sometimes wry. Never customer-service.\n"
    'Never "Great question!" or "Absolutely!" or "I apologize for the inconvenience".\n'
    "Substance over performance.\n"
)


_CRITIC_USER_TEMPLATE = (
    "Draft:\n"
    "{text}\n\n"
    'Return one line: "DECISION: pass" or "DECISION: revise <reason>" '
    'or "DECISION: block <reason>"\n'
    "The reason should be under 100 chars. No other output."
)


# Permissive parser — case-insensitive, tolerates leading whitespace/quotes.
_DECISION_RE = re.compile(
    r"DECISION:\s*(pass|revise|block)\b\s*(.*)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CriticVerdict:
    decision: str  # "pass" | "revise" | "block"
    reason: str  # short explanation (<=200 chars)


def _parse_verdict(raw: str) -> CriticVerdict:
    """Parse the model's single-line verdict. Permissive by design."""
    m = _DECISION_RE.search(raw)
    if not m:
        return CriticVerdict("pass", "critic-error: unparseable")
    decision = m.group(1).lower()
    reason = m.group(2).strip().strip(".").strip()
    # Cap reason length to keep downstream block messages compact.
    if len(reason) > 200:
        reason = reason[:200]
    return CriticVerdict(decision, reason)


async def _collect_text(prompt: str) -> str:
    """Run the SDK query and concatenate assistant text blocks."""
    options = ClaudeAgentOptions(
        model=settings.critic_model,
        system_prompt=_CRITIC_SYSTEM_PROMPT,
        max_turns=1,
        max_budget_usd=settings.critic_max_budget_usd,
        permission_mode="bypassPermissions",
        allowed_tools=[],
    )
    chunks: list[str] = []
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
    return "".join(chunks).strip()


async def critique_outbound(text: str, context: dict[str, Any]) -> CriticVerdict:
    """Run a cheap critic pass over an outbound message draft.

    Uses claude-haiku via the Claude Agent SDK so the call participates
    in Luke's existing budget tracking. Returns CriticVerdict.

    The critic checks:
    - Tone fit (warm, unhurried, never customer-service)
    - Filler ("Great question!", "Absolutely!", "I apologize")
    - Factual references that should have been grounded via recall
    - Coherence with claimed context

    On any failure (network, parse error, timeout), returns
    ``CriticVerdict("pass", "critic-error: <type>")`` — fail open so a
    misbehaving critic doesn't silence Luke.
    """
    prompt = _CRITIC_USER_TEMPLATE.format(text=text)
    try:
        raw = await asyncio.wait_for(
            _collect_text(prompt),
            timeout=settings.critic_timeout_s,
        )
    except TimeoutError:
        log.warning("critic_timeout", preview=text[:80])
        return CriticVerdict("pass", "critic-error: timeout")
    except Exception as e:  # fail open on any SDK / network failure
        log.warning("critic_error", error=str(e)[:200], preview=text[:80])
        return CriticVerdict("pass", f"critic-error: {type(e).__name__}")

    verdict = _parse_verdict(raw)
    log.info(
        "critic_verdict",
        decision=verdict.decision,
        reason=verdict.reason,
        tool=context.get("tool"),
    )
    return verdict
