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


_FRESHNESS_SYSTEM_PROMPT = (
    "You compare an outbound draft from Luke against the user's most "
    "recent messages. Decide if the draft is coherent with what the "
    "user just said, or if it would feel stale, contradictory, or like "
    "a response to an earlier state of the conversation."
)


_FRESHNESS_USER_TEMPLATE = (
    "User's most recent messages (oldest first):\n"
    "{user_messages}\n\n"
    "Draft Luke is about to send:\n"
    "{draft}\n\n"
    "Reply with exactly one line:\n"
    '- "DECISION: pass" if the draft is coherent\n'
    '- "DECISION: revise <reason>" if it needs adjustment\n'
    '- "DECISION: block <reason>" if it\'s incompatible (e.g. contradicts '
    "a retraction or answers a cancelled question)\n\n"
    "The reason should be under 100 chars."
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


async def _collect_text(prompt: str, system_prompt: str = _CRITIC_SYSTEM_PROMPT) -> str:
    """Run the SDK query and concatenate assistant text blocks."""
    options = ClaudeAgentOptions(
        model=settings.critic_model,
        system_prompt=system_prompt,
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


def _format_user_messages(user_latest: list[dict[str, Any]]) -> str:
    """Render the user's recent messages for the freshness prompt."""
    if not user_latest:
        return "(none)"
    lines: list[str] = []
    for m in user_latest:
        sender = str(m.get("sender_name", "user"))
        ts = str(m.get("timestamp", ""))
        content = str(m.get("content", "")).strip()
        # Cap each message preview so a long inbound doesn't blow the prompt.
        if len(content) > 800:
            content = content[:800] + "..."
        if ts:
            lines.append(f"[{sender} @ {ts}] {content}")
        else:
            lines.append(f"[{sender}] {content}")
    return "\n".join(lines)


async def check_freshness(
    draft: str, user_latest: list[dict[str, Any]]
) -> CriticVerdict:
    """Compare a draft outbound against the user's latest inbound messages.

    Returns CriticVerdict where:
    - "pass": draft is coherent with what the user just said
    - "revise": draft contradicts or stale-responds to the user
    - "block": draft is fundamentally incompatible (e.g. answering a
       question the user already cancelled)

    Fail-open: errors return pass.
    """
    prompt = _FRESHNESS_USER_TEMPLATE.format(
        user_messages=_format_user_messages(user_latest),
        draft=draft,
    )
    try:
        raw = await asyncio.wait_for(
            _collect_text(prompt, system_prompt=_FRESHNESS_SYSTEM_PROMPT),
            timeout=settings.critic_timeout_s,
        )
    except TimeoutError:
        log.warning("freshness_timeout", preview=draft[:80])
        return CriticVerdict("pass", "critic-error: timeout")
    except Exception as e:  # fail open on any SDK / network failure
        log.warning("freshness_error", error=str(e)[:200], preview=draft[:80])
        return CriticVerdict("pass", f"critic-error: {type(e).__name__}")

    verdict = _parse_verdict(raw)
    log.info(
        "freshness_verdict",
        decision=verdict.decision,
        reason=verdict.reason,
        user_msgs=len(user_latest),
    )
    return verdict
