"""Claude Agent SDK integration with in-process MCP tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

import structlog
import yaml
from aiogram import Bot
from aiogram.types import (
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReactionTypeEmoji,
    ReplyParameters,
)
from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookCallback,
    HookContext,
    HookMatcher,
    NotificationHookInput,
    PostToolUseFailureHookInput,
    PostToolUseHookInput,
    PreCompactHookInput,
    PreToolUseHookInput,
    ResultMessage,
    StopHookInput,
    SubagentStartHookInput,
    SubagentStopHookInput,
    ToolAnnotations,
    UserPromptSubmitHookInput,
    _python_type_to_json_schema,
    create_sdk_mcp_server,
    tool,
)
from claude_agent_sdk.types import (
    HookEvent,
    StreamEvent,
    SyncHookJSONOutput,
    ThinkingConfig,
    ThinkingConfigAdaptive,
)
from structlog.stdlib import BoundLogger

from . import context, db, memory
from . import letta_agent as _letta_agent
from .bus import bus
from .config import settings
from .memory import MEMORY_DIRS, read_frontmatter, read_memory_body, sanitize_memory_id

log: BoundLogger = structlog.get_logger()

_INTERNAL_RE = re.compile(r"<internal>[\s\S]*?</internal>")
_INTERNAL_OPEN_RE = re.compile(r"<internal>[\s\S]*$")  # unclosed tag at end
_LOG_TRUNCATION = 100  # chars for log message truncation

# --- Outbound message quality patterns (autonomous sends only) ---
_CATCHUP_PATTERNS = re.compile(
    r"(?i)(catching up|nothing (?:actionable|to report)|no (?:action|update)s? needed|"
    r"just checking in|no response requested|all good on my end)",
)
_FILLER_PATTERNS = re.compile(
    r"(?i)^(great question!?|absolutely!?|i apologize for the inconvenience|"
    r"let me know if you need anything else|here are the pros and cons)$",
)

# --- Overnight-commitment gate patterns ---
# Triggers when an outbound message commits to future delivery (time anchor
# + commitment verb) AND no agent/schedule_task has been spawned in this turn.
# See: insight-overnight-commitment-no-execution-pattern, luke-code-change-backlog.
_COMMITMENT_TIME_ANCHORS = re.compile(
    r"(?i)\b("
    r"overnight|"
    r"tonight|"
    r"by (?:morning|breakfast|the morning|tomorrow morning|6\s*am|7\s*am|8\s*am|9\s*am)|"
    r"before (?:morning|breakfast|you wake|sleep|8\s*am|7\s*am|9\s*am|6\s*am)|"
    r"while you sleep|"
    r"first thing(?: tomorrow)?|"
    r"tomorrow morning|"
    r"in the morning|"
    r"by sunrise"
    r")\b"
)
_COMMITMENT_VERBS = re.compile(
    r"(?i)\b("
    r"i'?ll have|you'?ll have|i'?ve got (?:it|this|you)|got it|got you(?: covered)?|"
    r"i'?ll ship|i'?ll deliver|i'?ll build|i'?ll send|i'?ll finish|"
    r"i'?ll get (?:it|this) done|i can have (?:it|this)|"
    r"will be ready|will deliver|consider it done"
    r")\b"
)


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# A yearless date resolving within this many days of now is treated as a
# near-term scheduling claim and pinned to that single year.
_WEEKDAY_NEAR_TERM_DAYS = 120

# "Tuesday Aug 7", "Friday, 7 August 2026", "Thu 6 Aug" — weekday adjacent to a
# calendar date, in either order. Year optional.
_WD_THEN_DATE = re.compile(
    r"(?i)\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b[,\s]+"
    r"(?:the\s+)?"
    r"(?:(?P<m1>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?P<d1>\d{1,2})"
    r"|(?P<d2>\d{1,2})(?:st|nd|rd|th)?\s+(?P<m2>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?)"
    r"(?:[,\s]+(?P<y>\d{4}))?"
)


def _weekday_claim_error(text: str, today: date | None = None) -> str | None:
    """Find a stated weekday that contradicts the date it is attached to.

    Deterministic guard against shipping "Tuesday Aug 7" when Aug 7 is a Friday
    (sent to Filipe 2026-07-31 about his US visa interview, while memory held the
    correct day). Conservative by construction: when no year is given the claim is
    only an error if it is wrong for *every* nearby year, so historical or
    forward-looking dates never false-positive.
    """
    if not text:
        return None
    plain = _HTML_TAG_RE.sub(" ", text)
    ref = today or datetime.now(UTC).date()

    for m in _WD_THEN_DATE.finditer(plain):
        claimed = _WEEKDAYS[m.group(1).lower()]
        mon_raw = m.group("m1") or m.group("m2")
        day_raw = m.group("d1") or m.group("d2")
        if not mon_raw or not day_raw:
            continue
        month, day = _MONTHS[mon_raw.lower()], int(day_raw)
        year_raw = m.group("y")
        if year_raw:
            candidates = [int(year_raw)]
        else:
            # Bare "Aug 7": resolve to the nearest occurrence. If that lands
            # inside the near-term window this is a scheduling claim (the
            # damaging class — appointments, deadlines, start dates), so it is
            # checked against that year alone. Outside the window it may be
            # loose historical prose, so any nearby year may vindicate it.
            nearby = []
            for y in (ref.year - 1, ref.year, ref.year + 1):
                try:
                    nearby.append(date(y, month, day))
                except ValueError:
                    continue  # e.g. Feb 29 on a non-leap year
            if not nearby:
                continue
            nearest = min(nearby, key=lambda d: abs((d - ref).days))
            near_term = abs((nearest - ref).days) <= _WEEKDAY_NEAR_TERM_DAYS
            candidates = [nearest.year] if near_term else [d.year for d in nearby]

        actual: list[str] = []
        for y in candidates:
            try:
                d = date(y, month, day)
            except ValueError:
                continue
            if d.weekday() == claimed:
                break  # consistent for at least one plausible year → not an error
            actual.append(f"{d.strftime('%A')} {d.isoformat()}")
        else:
            if not actual:
                continue
            return (
                f"weekday/date mismatch: you wrote {m.group(0).strip()!r} but "
                f"that date falls on {', '.join(actual)}. Verify with datetime "
                f"and correct the weekday before sending."
            )
    return None


def _check_outbound_quality(text: str) -> str | None:
    """Check an outbound message against quality rules. Returns rejection reason or None."""
    if not text or not text.strip():
        return "empty message"

    # Internal tags leaked
    if "<internal>" in text:
        return "contains <internal> tags"

    # Filler / never-say patterns
    stripped = text.strip()
    if _FILLER_PATTERNS.match(stripped):
        return f"matches never-say pattern: {stripped[:50]}"

    # Non-action announcements
    if _CATCHUP_PATTERNS.search(text):
        return "catchup/non-action announcement"

    # Too short to be substantive (but allow emoji reactions)
    if len(stripped) < 15 and not any(ord(c) > 0x1F600 for c in stripped):
        return "message too short to be substantive"

    return None


_TG_MAX_MSG_LEN = 4096  # Telegram API hard limit
_STREAMING_CURSOR = " ▍"  # visual typing indicator


_LEAKED_INTERNAL_PATTERNS = re.compile(
    r"^(?:No response (?:requested|needed)|Nothing to (?:say|send|respond)"
    r"|Silently |I (?:won't|don't need to) (?:send|respond|reply)"
    r"|No (?:message|reply|output) (?:needed|required|necessary))"
    r"\.?$",
    re.IGNORECASE,
)


def _is_leaked_internal(text: str) -> bool:
    """Detect internal reasoning that leaked without <internal> tags."""
    return bool(_LEAKED_INTERNAL_PATTERNS.match(text.strip()))


def _trunc(text: str) -> str:
    """Truncate text for logging."""
    return text[:_LOG_TRUNCATION] + "…" if len(text) > _LOG_TRUNCATION else text


def _clean_streaming_text(raw: str) -> str:
    """Strip complete and partial <internal> tags for streaming display."""
    text = _INTERNAL_RE.sub("", raw)  # strip closed tags
    text = _INTERNAL_OPEN_RE.sub("", text)  # strip unclosed trailing tag
    return text.strip()


def _ok(text: str) -> dict[str, Any]:
    """Build a standard MCP tool success response."""
    return {"content": [{"type": "text", "text": text}]}


async def _send_chunk(bot: Bot, chat_id: int, text: str, **kwargs: Any) -> None:
    """Send a single chunk with retry + exponential backoff.

    Falls back to plaintext on HTML parse failure.
    Stores outbound messages in DB for conversation history.
    """
    from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter

    for attempt in range(settings.telegram_send_retries):
        try:
            sent = await bot.send_message(chat_id=chat_id, text=text, **kwargs)
            db.store_message(
                chat_id=str(chat_id),
                sender_name=settings.assistant_name,
                message_id=sent.message_id,
                content=text,
                timestamp=sent.date.isoformat(),
            )
            return
        except TelegramRetryAfter as exc:
            log.warning(
                "telegram_rate_limited",
                chat_id=chat_id,
                retry_after=exc.retry_after,
            )
            await asyncio.sleep(exc.retry_after)
        except TelegramBadRequest:
            log.warning("html_parse_failed", chat_id=chat_id)
            sent = await bot.send_message(chat_id=chat_id, text=text, parse_mode=None)
            db.store_message(
                chat_id=str(chat_id),
                sender_name=settings.assistant_name,
                message_id=sent.message_id,
                content=text,
                timestamp=sent.date.isoformat(),
            )
            return
        except Exception:
            if attempt == settings.telegram_send_retries - 1:
                raise
            delay = settings.telegram_retry_base_delay * (2**attempt)
            log.warning(
                "telegram_send_retry",
                chat_id=chat_id,
                attempt=attempt + 1,
                delay=delay,
            )
            await asyncio.sleep(delay)


async def send_long_message(bot: Bot, chat_id: int, text: str, **kwargs: Any) -> None:
    """Send a message, splitting into chunks if it exceeds Telegram's 4096 char limit."""
    # Duplicate detection: skip if same content was sent recently
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    if db.is_duplicate_outbound(str(chat_id), content_hash):
        log.warning("duplicate_message_blocked", chat=chat_id, text=_trunc(text))
        return
    log.info("msg_out", chat=chat_id, text=_trunc(text))
    while text:
        if len(text) <= _TG_MAX_MSG_LEN:
            await _send_chunk(bot, chat_id=chat_id, text=text, **kwargs)
            break
        # Split at last newline within limit, fall back to hard cut
        # Reserve 2 chars for the continuation marker "\n…"
        max_len = _TG_MAX_MSG_LEN - 2
        cut = text.rfind("\n", 0, max_len)
        if cut <= 0:
            cut = max_len
        await _send_chunk(bot, chat_id=chat_id, text=text[:cut] + "\n…", **kwargs)
        text = text[cut:].lstrip("\n")
    db.log_outbound(str(chat_id), content_hash)


_VALID_MEMORY_TYPES: frozenset[str] = frozenset(MEMORY_DIRS)

# Tool annotation presets
_OPEN_WORLD = ToolAnnotations(openWorldHint=True)
_READ_ONLY = ToolAnnotations(readOnlyHint=True)
_DESTRUCTIVE = ToolAnnotations(destructiveHint=True)


def _schema(props: dict[str, Any], optional: Sequence[str] = ()) -> dict[str, Any]:
    """Build an explicit JSON Schema so optional params stay optional.

    The SDK's dict-shorthand builder sets ``required = list(properties)`` — every
    declared param becomes mandatory regardless of whether the handler reads it
    with ``args.get()``. That forced values for params the handler defaults
    anyway (``chat_id`` resolves to the current chat; ``supersedes_rel`` only
    invalidates an old edge), so calls either failed validation or shipped a
    guessed value. Emitting the full schema takes the SDK's passthrough branch.
    """
    unknown = set(optional) - set(props)
    if unknown:
        raise ValueError(f"optional params not in schema: {sorted(unknown)}")
    return {
        "type": "object",
        "properties": {k: _python_type_to_json_schema(v) for k, v in props.items()},
        "required": [k for k in props if k not in set(optional)],
    }


# Tools that send outbound Telegram messages (rate-limited by PreToolUse hook)
_SEND_TOOLS: frozenset[str] = frozenset(
    {
        "mcp__luke__send_message",
        "mcp__luke__reply",
        "mcp__luke__send_photo",
        "mcp__luke__send_document",
        "mcp__luke__send_voice",
        "mcp__luke__send_video",
        "mcp__luke__send_location",
        "mcp__luke__send_poll",
        "mcp__luke__send_buttons",
        "mcp__luke__forward",
    }
)

# Send tools whose PRIMARY payload is the text body itself. For these an
# empty message is a real defect. The other _SEND_TOOLS (documents, media,
# location, poll, buttons) carry the payload as a file/attachment with an
# OPTIONAL caption, so an empty caption must NOT be gated as "empty message".
_TEXT_PRIMARY_TOOLS: frozenset[str] = frozenset(
    {
        "mcp__luke__send_message",
        "mcp__luke__reply",
    }
)


# Patterns that suggest the draft references a past event/topic
# Used by _references_past_events to gate autonomous sends behind a recall call.
_TEMPORAL_PHRASES: tuple[str, ...] = (
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
)


def _commits_future_work(text: str) -> bool:
    """Heuristic: does the draft commit to delivering work by a future time?

    Detects co-occurrence of a time anchor ('overnight', 'by morning', etc.)
    with a commitment verb ("I'll have", 'got it', etc.). Used by the
    overnight-commitment gate to require that an agent has been spawned
    or a task scheduled in the same turn before such a message can be sent.

    Conservative by design: both anchors must be present; short texts
    (< 20 chars) are skipped to avoid catching trivial acknowledgments.

    Reference: insight-overnight-commitment-no-execution-pattern.
    """
    if not text or len(text) < 20:
        return False
    return bool(_COMMITMENT_TIME_ANCHORS.search(text)) and bool(_COMMITMENT_VERBS.search(text))


def _references_past_events(text: str) -> bool:
    """Heuristic: does the draft reference a past event Luke should recall first?

    Matches temporal phrases (yesterday/last week/etc) and conversational
    deixis (you mentioned, we discussed).

    Conservative by design: false negatives are fine (worst case: agent sends
    a fresh-looking message that happens to reference past events — current
    behavior).  False positives would block legit fresh messages, so the rule
    requires BOTH a temporal/deictic phrase AND text length > 30 chars
    (filters out trivial acknowledgments like "yesterday's question").
    """
    if not text or len(text) < 30:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in _TEMPORAL_PHRASES)


# --- Explicit file-artifact request capture patterns ---
# The single most-recurring autonomous failure (Jul 2026, ~5 reflexions):
# Filipe asks for a concrete FILE deliverable ("give me a pdf", "make a doc",
# "put together a brief") and the turn ends with only topic-adjacent chatter —
# either the file was never built, or it was built on disk but never SENT.
# Every prior corrective was advisory and never fired in a busy multi-thread
# window. The structural fix (dream-explicit-request-is-the-flinchs-most-
# dangerous-disguise, dream-completion-signal-must-measure-receipt-not-
# production): a forced Stop-hook gate that measures the FAR end of the pipe —
# did an artifact actually ship, or a durable handle get created — before the
# turn is allowed to close. Deliberately targets FILE artifacts only (pdf/doc/
# brief/etc.), NOT inline conversational drafts ("draft an email"), so it fires
# on exactly the recurring class without tripping normal chat.
_ARTIFACT_REQUEST_VERBS = re.compile(
    r"(?i)\b("
    r"give me|send me|make me|build me|write me|get me|"
    r"i need|i want|i'?d like|can you (?:make|build|create|put|do|prepare|write|send|generate)|"
    r"could you (?:make|build|create|put|do|prepare|write|send|generate)|"
    r"put together|prepare|generate|create|build me|draft me"
    r")\b"
)
_ARTIFACT_REQUEST_NOUNS = re.compile(
    r"(?i)\b("
    r"pdf|docx?|document|spreadsheet|csv|excel|"
    r"report|brief|write-?up|deck|slides|presentation|"
    r"one-?pager|cheat\s?sheet|file|doc"
    r")\b"
)


def _requests_file_artifact(text: str) -> bool:
    """Heuristic: does an inbound message explicitly ask for a FILE deliverable?

    Requires BOTH a request verb ("give me", "put together", "can you make")
    AND a file-artifact noun ("pdf", "doc", "brief", "report", "spreadsheet").
    Deliberately excludes inline conversational drafts ("draft an email",
    "write me a message") — those are satisfied by a plain send_message and
    are NOT the failure class. Conservative like ``_commits_future_work``:
    both anchors must be present; short texts (< 12 chars) are skipped.

    Reference: reflexion-built-but-not-sent-bar-brief,
    reflexion-dropped-explicit-request-visa-pdf, insight-artifact-requests-
    need-immediate-tasking.
    """
    if not text or len(text) < 12:
        return False
    return bool(_ARTIFACT_REQUEST_VERBS.search(text)) and bool(_ARTIFACT_REQUEST_NOUNS.search(text))


# Send tools that deliver an actual FILE/attachment to Filipe. Distinct from a
# plain send_message: the recurring failure is a topic-adjacent send_message
# that discusses the artifact while the file itself never ships. Only these
# (or a durable handle via _AGENT_SCHEDULE_TOOLS) satisfy the artifact gate.
_ARTIFACT_SEND_TOOLS: frozenset[str] = frozenset(
    {
        "mcp__luke__send_document",
        "mcp__luke__send_photo",
        "mcp__luke__send_video",
    }
)


# --- Primary-source read gate patterns ---
# The second-most-recurring autonomous failure (Jul 2026, 4 reflexions incl.
# reflexion-primary-source-regression-prerna): Filipe points me at a source he
# KNOWS I can open — "read the email", "check the doc", "what does that pdf say"
# — and I answer from my own prior summary of it instead of reading it. In the
# Prerna case I got progressively wrong across 5 exchanges and had to be told
# "read the fucking email" before I opened it. Every prior corrective was an
# advisory insight that never fired mid-turn. The structural fix mirrors the
# artifact gate: when the inbound message asks me to consult a readable source,
# the turn may not close until I actually called a read/fetch tool this turn.
# Deliberately conservative: requires BOTH a read-verb AND a source-noun (or an
# explicit file path), so casual mentions ("thanks for the email") don't trip it.
_SOURCE_READ_VERBS = re.compile(
    r"(?i)(?:"
    r"\bread\b|\bre-?read\b|\breread\b|\bcheck\b|\bopen\b|\breview\b|"
    r"look (?:at|through|over|into)|take a look|pull up|scroll through|"
    r"go through|go read|did you (?:even )?read|have you (?:read|seen)|"
    r"what does .{0,40}? say|what'?s in\b|see what .{0,40}? say"
    r")"
)
_SOURCE_READ_NOUNS = re.compile(
    r"(?i)(?:"
    r"\be-?mails?\b|\binbox\b|\bthreads?\b|\bmessages?\b|\bdms?\b|"
    r"\bdocs?\b|\bdocuments?\b|\bpdfs?\b|\battachments?\b|\bfiles?\b|"
    r"\bletters?\b|\bmemos?\b|\breports?\b|\bbriefs?\b|\btranscripts?\b|"
    r"\bspreadsheets?\b|\blinks?\b|\barticles?\b|\bpages?\b|\bposts?\b|\bnotes?\b|"
    r"/[\w./-]+\.(?:pdf|docx?|md|txt|csv|xlsx?)|\.(?:pdf|docx?|md|txt|csv|xlsx?)\b"
    r")"
)


def _requests_source_read(text: str) -> bool:
    """Heuristic: does an inbound message ask me to consult a readable SOURCE?

    Requires BOTH a read-verb ("read", "check", "look at", "what does X say")
    AND a source-noun ("email", "doc", "pdf", "thread", a file path). The
    failure class is answering about a specific accessible document from my own
    summary instead of opening it (reflexion-primary-source-regression-prerna-2026-07-21,
    reflexion-advise-on-unreadable-document-name-the-gap-2026-07-23). Conservative like
    ``_requests_file_artifact``: both anchors required; short texts skipped.
    """
    if not text or len(text) < 12:
        return False
    return bool(_SOURCE_READ_VERBS.search(text)) and bool(_SOURCE_READ_NOUNS.search(text))


def _context_query(prompt: str | list[dict[str, Any]], user_text: str | None) -> str:
    """The text that should drive memory retrieval and the Stop gates.

    Prefer ``user_text`` — the caller's clean, envelope-free user message captured
    BEFORE any memory-context injection into ``prompt``. Memory context is prepended
    to str prompts and inserted at index 0 for multimodal list prompts, so reading
    ``prompt`` here would feed the retrieval query and the file-artifact/source-read
    gates the injected memory blob instead of what the user actually typed — masking
    the real message and dropping their request (the proxy-for-referent failure class:
    dream-proxy-for-referent-is-one-disease-three-faces). Only fall back to ``prompt``
    for autonomous/scheduled callers that pass no ``user_text``.
    """
    if user_text is not None:
        return user_text
    if isinstance(prompt, str):
        return prompt
    return str(prompt[0].get("text", "")) if prompt else ""


# Tools that satisfy the primary-source gate: they actually pull CONTENT from a
# document, file, or URL. Deliberately EXCLUDES Bash (too broad — nearly every
# turn runs it, which would neuter the gate) and recall/recall_conversation
# (memory is the WRONG substitute — reaching for my own summary instead of the
# source is the exact failure this gate exists to stop).
_SOURCE_READ_TOOLS: frozenset[str] = frozenset(
    {
        "Read",
        "Grep",
        "WebFetch",
        "mcp__luke__browse",
    }
)


# Tools that count as "recall" for the recall-before-reference gate
_RECALL_TOOLS: frozenset[str] = frozenset(
    {
        "mcp__luke__recall",
        "mcp__luke__recall_conversation",
    }
)

# Tools that count as "work scheduled" for the overnight-commitment gate.
# Task = Claude Code sub-agent spawn; schedule_task = cron/interval/once job.
_AGENT_SCHEDULE_TOOLS: frozenset[str] = frozenset(
    {
        "Task",
        "mcp__luke__schedule_task",
        # delegate spawns a teardown-surviving background job that reports
        # back on completion — it satisfies the overnight-commitment gate the
        # same way Task does. Without this, steering background work to the
        # correct tool (delegate) would trip the "committed but nothing
        # scheduled" block. Added 2026-07-16.
        "mcp__luke__delegate",
    }
)


# --- Scheduled-task duplicate gate ---
# The #2 recurring autonomous failure (scheduling-cron cluster, 18 advisory
# correctives incl. proc-scheduled-task-dedup, insight-scheduled-send-state-
# drift): an auto-staged flow schedules a reminder for an event, then a later
# turn — having lost track of the first — schedules a SECOND task for the same
# deliverable. Filipe then gets the same nudge twice (the Jul 25 2026 Prerna-
# Monday-brief oversend; two live "visa interview TOMORROW" tasks firing 18:00
# AND 20:00 the day before). Every prior corrective was an advisory note that
# never fired at schedule time. The structural fix: before a schedule_task is
# allowed, check the active task list for a near-duplicate — same schedule kind,
# firing close in time (for `once`) or on the identical cadence (cron/interval),
# with high prompt word-overlap — and block, steering to reuse/update the
# existing task. Conservative: requires BOTH strong text overlap AND time/cadence
# proximity, so genuinely distinct same-day reminders are not caught.
# Reference: proc-scheduled-task-dedup, insight-scheduled-send-state-drift.
_TASK_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "for",
        "with",
        "his",
        "her",
        "him",
        "that",
        "this",
        "these",
        "those",
        "then",
        "than",
        "into",
        "onto",
        "from",
        "filipe",
        "before",
        "after",
        "ahead",
        "about",
        "your",
        "yours",
        "them",
        "will",
        "would",
        "should",
        "could",
        "have",
        "has",
        "had",
        "not",
        "you",
        "remind",
        "reminder",
        "nudge",
        "check",
        "run",
        "send",
        "ask",
        "note",
        "once",
        "cron",
        "interval",
        "task",
        "schedule",
        "scheduled",
        "when",
        "day",
        "morning",
        "evening",
        "afternoon",
        "night",
        "today",
        "tomorrow",
    }
)


def _task_sig_words(text: str) -> set[str]:
    """Significant lowercase word set of a task prompt (for overlap scoring)."""
    words = re.findall(r"[a-z0-9]{4,}", (text or "").lower())
    return {w for w in words if w not in _TASK_STOPWORDS}


def _task_overlap(a: str, b: str) -> float:
    """Containment overlap of two task prompts over significant words.

    Uses the overlap (containment) coefficient |A∩B| / min(|A|,|B|) rather than
    Jaccard: task prompts for the SAME deliverable often differ wildly in length
    (a terse restage vs a verbose original), which sinks Jaccard well below any
    useful threshold (real duplicate visa reminders scored 0.11 Jaccard but 0.88
    containment). Both prompts must carry >= 4 significant words, so two short
    unrelated reminders can't spuriously hit a high ratio.
    """
    sa, sb = _task_sig_words(a), _task_sig_words(b)
    if len(sa) < 4 or len(sb) < 4:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def _duplicate_pending_task(
    tool_input: dict[str, Any], existing: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """Return an active task that the new schedule_task would duplicate, else None.

    A duplicate requires the SAME schedule_type, high prompt word-overlap
    (>= 0.7 containment on significant words), AND temporal proximity:
      - once:            both fire within 8 hours of each other
      - cron/interval:   identical schedule_value (same cadence)
    """
    new_prompt = str(tool_input.get("prompt", "") or "")
    new_type = str(tool_input.get("schedule_type", "") or "")
    new_value = str(tool_input.get("schedule_value", "") or "")
    if len(new_prompt) < 12 or not new_type:
        return None
    new_dt: datetime | None = None
    if new_type == "once":
        try:
            new_dt = datetime.fromisoformat(new_value)
        except ValueError:
            new_dt = None
    for t in existing:
        if t.get("status") not in ("active", "pending", "scheduled"):
            continue
        if t.get("schedule_type") != new_type:
            continue
        if _task_overlap(new_prompt, str(t.get("prompt", ""))) < 0.7:
            continue
        if new_type == "once":
            if new_dt is None:
                continue
            try:
                other_dt = datetime.fromisoformat(str(t.get("schedule_value", "")))
            except ValueError:
                continue
            if abs((new_dt - other_dt).total_seconds()) <= 8 * 3600:
                return t
        else:
            if new_value and new_value == str(t.get("schedule_value", "")):
                return t
    return None


_AUTO_SKILL_THRESHOLD: int = 5  # tool calls to trigger procedure extraction in stop hook

# Active client registry — enables external interruption of running agents
_active_clients: dict[str, ClaudeSDKClient] = {}

# Delegation envelope — delegate-created once-tasks carry this header so the
# scheduler can tell "relay the result to Filipe" jobs apart from ordinary
# scheduled tasks. Delegations are DB-backed (tasks table) rather than bare
# asyncio tasks precisely so they survive restarts: the pre-2026-08-01 in-
# process registry lost every in-flight job on deploy/crash, silently.
_DELEGATION_HEADER = "[delegation:v1]"


def format_delegation_prompt(prompt: str, trigger_msg_id: int | None) -> str:
    """Wrap a delegated job's prompt in the durable v1 envelope."""
    return f"{_DELEGATION_HEADER}\ntrigger_msg_id: {trigger_msg_id or ''}\n\n{prompt}"


def parse_delegation(stored: str) -> tuple[str, int | None] | None:
    """Return (body, trigger_msg_id) if `stored` is a delegation envelope, else None."""
    if not stored.startswith(_DELEGATION_HEADER):
        return None
    header, _, body = stored.partition("\n\n")
    trigger: int | None = None
    for line in header.splitlines()[1:]:
        if line.startswith("trigger_msg_id:"):
            value = line.split(":", 1)[1].strip()
            try:
                trigger = int(value) if value else None
            except ValueError:
                trigger = None
    return body, trigger


def _compose_system_append(persona: str, working_ctx: str | None) -> str:
    """Assemble the system-prompt append: working memory FIRST, persona LAST.

    The model's register tracks the most recent instructions it read. With
    the persona first, up to 12k tokens of clinical memory prose landed after
    it and Luke answered like a status report (Filipe, 2026-08-01: "not
    really following his personality"). Working memory is framed as knowledge
    — not voice — and the persona closes the prompt so it wins recency.
    """
    if not working_ctx:
        return persona
    framed = (
        "<working_memory>\n"
        "Background knowledge. It informs what you know — never how you sound.\n"
        + working_ctx
        + "\n</working_memory>"
    )
    if not persona:
        return framed
    return framed + "\n\n" + persona


async def interrupt_agent(chat_id: str) -> bool:
    """Interrupt a running agent for the given chat. Returns True if interrupted."""
    client = _active_clients.get(chat_id)
    if client is None:
        return False
    try:
        await client.interrupt()
        log.info("agent_interrupted", chat_id=chat_id)
        return True
    except Exception:
        log.exception("interrupt_failed", chat_id=chat_id)
        return False


def get_active_agents() -> list[str]:
    """Return chat IDs with currently running agents."""
    return list(_active_clients.keys())


@dataclass(slots=True)
class AgentResult:
    texts: list[str] = field(default_factory=list)
    session_id: str | None = None
    cost_usd: float = 0.0
    num_turns: int = 0
    duration_api_ms: int = 0
    sent_messages: int = 0  # messages sent via MCP tools during this run
    tool_uses: int = 0  # total tool calls made during this run
    streaming_msg_id: int | None = None  # Telegram msg ID if streaming preview was sent
    input_tokens: int = 0
    output_tokens: int = 0
    cache_create_tokens: int = 0
    cache_read_tokens: int = 0


# ---------------------------------------------------------------------------
# MCP tools — built per invocation so they close over group context + bot
# ---------------------------------------------------------------------------


def _build_tools(chat_id: str, bot: Bot) -> Any:
    """Create the in-process MCP server with all tools."""
    root = settings.luke_dir

    # Allowed roots for file-sending tools (prevents arbitrary path access)
    _safe_roots = (
        root.resolve(),
        settings.store_dir.resolve(),
    )

    def _safe_path(path_str: str) -> Path | str:
        """Resolve a path, check it's under an allowed root and exists.

        Returns the resolved Path on success, or an error string on failure.
        """
        resolved = Path(path_str).resolve()
        allowed = any(resolved == root or root in resolved.parents for root in _safe_roots)
        if not allowed:
            return "Error: path not allowed (must be under data directory)"
        if not resolved.is_file():
            return f"Error: file not found: {path_str}"
        return resolved

    def _target(args: dict[str, Any]) -> int:
        """Resolve the target chat ID from tool args, defaulting to current chat."""
        return int(args.get("chat_id", chat_id))

    # --- Telegram (14 tools) ---

    @tool(
        "send_message",
        "Send text. HTML tags (<b>,<i>,<code>,<pre>), NOT markdown. Auto-splits at 4096.",
        _schema({"chat_id": str, "text": str, "silent": bool}, ['chat_id', 'silent']),
        annotations=_OPEN_WORLD,
    )
    async def t_send(args: dict[str, Any]) -> dict[str, Any]:
        await send_long_message(
            bot,
            chat_id=_target(args),
            text=args["text"],
            disable_notification=args.get("silent", False),
        )
        return _ok("Sent")

    @tool(
        "send_photo",
        "Send photo from local path",
        _schema({"chat_id": str, "path": str, "caption": str}, ['chat_id', 'caption']),
        annotations=_OPEN_WORLD,
    )
    async def t_photo(args: dict[str, Any]) -> dict[str, Any]:
        path = _safe_path(args["path"])
        if isinstance(path, str):
            return _ok(path)
        await bot.send_photo(
            chat_id=_target(args),
            photo=FSInputFile(path),
            caption=args.get("caption", ""),
        )
        return _ok("Photo sent")

    @tool(
        "send_document",
        "Send file/document from local path",
        _schema({"chat_id": str, "path": str, "caption": str}, ['chat_id', 'caption']),
        annotations=_OPEN_WORLD,
    )
    async def t_doc(args: dict[str, Any]) -> dict[str, Any]:
        path = _safe_path(args["path"])
        if isinstance(path, str):
            return _ok(path)
        await bot.send_document(
            chat_id=_target(args),
            document=FSInputFile(path),
            caption=args.get("caption", ""),
        )
        return _ok("Document sent")

    @tool(
        "send_voice",
        "Send voice from local .ogg file",
        _schema({"chat_id": str, "path": str}, ['chat_id']),
        annotations=_OPEN_WORLD,
    )
    async def t_voice(args: dict[str, Any]) -> dict[str, Any]:
        path = _safe_path(args["path"])
        if isinstance(path, str):
            return _ok(path)
        await bot.send_voice(chat_id=_target(args), voice=FSInputFile(path))
        return _ok("Voice sent")

    @tool(
        "send_video",
        "Send video from local path",
        _schema({"chat_id": str, "path": str, "caption": str}, ['chat_id', 'caption']),
        annotations=_OPEN_WORLD,
    )
    async def t_video(args: dict[str, Any]) -> dict[str, Any]:
        path = _safe_path(args["path"])
        if isinstance(path, str):
            return _ok(path)
        await bot.send_video(
            chat_id=_target(args),
            video=FSInputFile(path),
            caption=args.get("caption", ""),
        )
        return _ok("Video sent")

    @tool(
        "send_location",
        "Send GPS location",
        _schema({"chat_id": str, "latitude": float, "longitude": float}, ['chat_id']),
        annotations=_OPEN_WORLD,
    )
    async def t_loc(args: dict[str, Any]) -> dict[str, Any]:
        await bot.send_location(
            chat_id=_target(args),
            latitude=args["latitude"],
            longitude=args["longitude"],
        )
        return _ok("Location sent")

    @tool(
        "send_poll",
        "Create poll. options: list of strings e.g. ['Yes','No']",
        _schema({"chat_id": str, "question": str, "options": list, "is_anonymous": bool}, ['chat_id', 'is_anonymous']),
        annotations=_OPEN_WORLD,
    )
    async def t_poll(args: dict[str, Any]) -> dict[str, Any]:
        opts = args["options"]
        if isinstance(opts, str):
            # Agent may pass JSON string or newline-separated
            try:
                opts = json.loads(opts)
            except json.JSONDecodeError, ValueError:
                opts = [o.strip() for o in opts.split("\n") if o.strip()]
        await bot.send_poll(
            chat_id=_target(args),
            question=args["question"],
            options=cast(list[Any], opts),
            is_anonymous=args.get("is_anonymous", True),
        )
        return _ok("Poll created")

    @tool(
        "send_buttons",
        "Send message with inline buttons. buttons: rows of [{text,data}]. "
        "Pressed button sends '[Button pressed: data]' as new message.",
        _schema({"chat_id": str, "text": str, "buttons": list}, ['chat_id']),
        annotations=_OPEN_WORLD,
    )
    async def t_buttons(args: dict[str, Any]) -> dict[str, Any]:
        buttons = args["buttons"]
        if isinstance(buttons, str):
            try:
                buttons = json.loads(buttons)
            except json.JSONDecodeError, ValueError:
                return _ok("Error: buttons must be a JSON list of rows")
        try:
            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text=b["text"], callback_data=b["data"]) for b in row]
                    for row in buttons
                ]
            )
        except (KeyError, TypeError) as exc:
            return _ok(f"Error: malformed buttons structure: {exc}")
        await bot.send_message(chat_id=_target(args), text=args["text"], reply_markup=kb)
        return _ok("Buttons sent")

    @tool(
        "reply",
        "Reply to msg:{id} from prompt. HTML tags, NOT markdown.",
        _schema({"chat_id": str, "message_id": str, "text": str}, ['chat_id']),
        annotations=_OPEN_WORLD,
    )
    async def t_reply(args: dict[str, Any]) -> dict[str, Any]:
        await send_long_message(
            bot,
            chat_id=_target(args),
            text=args["text"],
            reply_parameters=ReplyParameters(message_id=int(args["message_id"])),
        )
        return _ok("Replied")

    @tool(
        "forward",
        "Forward message to another chat",
        {"from_chat_id": str, "to_chat_id": str, "message_id": str},
        annotations=_OPEN_WORLD,
    )
    async def t_fwd(args: dict[str, Any]) -> dict[str, Any]:
        to_id = _target({**args, "chat_id": args["to_chat_id"]})
        from_id = _target({**args, "chat_id": args["from_chat_id"]})
        await bot.forward_message(
            chat_id=to_id,
            from_chat_id=from_id,
            message_id=int(args["message_id"]),
        )
        return _ok("Forwarded")

    @tool(
        "react",
        "React with emoji",
        _schema({"chat_id": str, "message_id": str, "emoji": str}, ['chat_id']),
        annotations=_OPEN_WORLD,
    )
    async def t_react(args: dict[str, Any]) -> dict[str, Any]:
        await bot.set_message_reaction(
            chat_id=_target(args),
            message_id=int(args["message_id"]),
            reaction=[ReactionTypeEmoji(emoji=args["emoji"])],
        )
        return _ok("Reacted")

    @tool(
        "get_reactions",
        "Query emoji reactions received on messages. "
        "Filter by msg_id, sender_id, or sentiment (positive/negative/neutral). "
        "Returns newest first with message previews.",
        _schema({"msg_id": int, "sender_id": str, "sentiment": str, "limit": int}, ['msg_id', 'sender_id', 'sentiment', 'limit']),
        annotations=_READ_ONLY,
    )
    async def t_get_reactions(args: dict[str, Any]) -> dict[str, Any]:
        reactions = db.get_reactions(
            chat_id,
            msg_id=args.get("msg_id"),
            sender_id=args.get("sender_id"),
            sentiment=args.get("sentiment"),
            limit=args.get("limit", 20),
        )
        if not reactions:
            return _ok("No reactions found")
        lines: list[str] = []
        for r in reactions:
            preview = r.get("msg_preview") or "(message not found)"
            sender = r.get("msg_sender") or "?"
            lines.append(
                f"{r['emoji']} ({r['sentiment']}) on msg:{r['msg_id']} "
                f"from {sender}: {preview} "
                f"— reacted by {r['sender_id']} at {r['timestamp']}"
            )
        return _ok("\n".join(lines))

    @tool(
        "edit_message",
        "Edit sent message. HTML tags, NOT markdown.",
        _schema({"chat_id": str, "message_id": str, "text": str}, ['chat_id']),
        annotations=_DESTRUCTIVE,
    )
    async def t_edit(args: dict[str, Any]) -> dict[str, Any]:
        await bot.edit_message_text(
            chat_id=_target(args),
            message_id=int(args["message_id"]),
            text=args["text"],
        )
        return _ok("Edited")

    @tool(
        "delete_message",
        "Delete message",
        _schema({"chat_id": str, "message_id": str}, ['chat_id']),
        annotations=_DESTRUCTIVE,
    )
    async def t_del(args: dict[str, Any]) -> dict[str, Any]:
        await bot.delete_message(chat_id=_target(args), message_id=int(args["message_id"]))
        return _ok("Deleted")

    @tool(
        "pin",
        "Pin message",
        _schema({"chat_id": str, "message_id": str}, ['chat_id']),
        annotations=_OPEN_WORLD,
    )
    async def t_pin(args: dict[str, Any]) -> dict[str, Any]:
        await bot.pin_chat_message(chat_id=_target(args), message_id=int(args["message_id"]))
        return _ok("Pinned")

    # --- Scheduling (3 tools) ---

    @tool(
        "schedule_task",
        "Schedule task. type: cron|interval|once. "
        "value: cron expr, milliseconds, or ISO timestamp.",
        {"prompt": str, "schedule_type": str, "schedule_value": str},
        annotations=_OPEN_WORLD,
    )
    async def sched(args: dict[str, Any]) -> dict[str, Any]:
        try:
            task_id = db.create_task(
                chat_id,
                args["prompt"],
                args["schedule_type"],
                args["schedule_value"],
            )
        except ValueError as exc:
            return _ok(f"Error: {exc}")
        bus.emit(
            "cron_created",
            {
                "task_id": task_id,
                "schedule_type": args["schedule_type"],
                "prompt_preview": args["prompt"][:200],
            },
        )
        return _ok(f"Scheduled: {task_id}")

    @tool(
        "list_tasks",
        "List scheduled tasks",
        {},
        annotations=_OPEN_WORLD,
    )
    async def list_tasks_tool(args: dict[str, Any]) -> dict[str, Any]:
        tasks = db.list_tasks(chat_id)
        if not tasks:
            return _ok("No scheduled tasks.")
        lines = []
        for t in tasks:
            lines.append(
                f"[{t['id']}] {t['schedule_type']}={t['schedule_value']} "
                f"status={t['status']} prompt={t['prompt'][:80]}"
            )
        return _ok("\n".join(lines))

    @tool(
        "delete_task",
        "Delete scheduled task by ID",
        {"task_id": str},
        annotations=_OPEN_WORLD,
    )
    async def delete_task_tool(args: dict[str, Any]) -> dict[str, Any]:
        if db.delete_task(args["task_id"]):
            bus.emit("cron_deleted", {"task_id": args["task_id"]})
            return _ok(f"Deleted task {args['task_id']}")
        return _ok(f"Task {args['task_id']} not found")

    # --- Memory (8 tools) ---

    @tool(
        "remember",
        "Save memory. type: entity|episode|procedure|insight|goal. "
        "importance: 0.1-2.0 (default 1.0). Returns change summary on entity update.",
        # Full JSON Schema (not a plain type-dict): the Agent SDK marks EVERY key
        # of a plain dict as required, which rejected any remember() call omitting
        # tags/links/importance at the MCP validation layer — before mem_save's
        # defaults/coercion could run. Only id/type/title/content are truly required.
        {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string"},
                "title": {"type": "string"},
                "content": {"type": "string"},
                "tags": {"type": "array"},
                "links": {"type": "array"},
                "importance": {"type": "number"},
            },
            "required": ["id", "type", "title", "content"],
        },
    )
    async def mem_save(args: dict[str, Any]) -> dict[str, Any]:
        mem_type: str = args["type"]
        if mem_type not in _VALID_MEMORY_TYPES:
            return _ok(f"Invalid type: {mem_type}")
        mem_id = sanitize_memory_id(args["id"])
        if not mem_id:
            return _ok("Error: id must contain at least one alphanumeric character")
        title = args["title"].replace("\n", " ").replace("\r", " ")
        type_dir = MEMORY_DIRS.get(mem_type, f"{mem_type}s")
        mem_dir = settings.memory_dir / type_dir
        mem_dir.mkdir(parents=True, exist_ok=True)
        path = mem_dir / f"{mem_id}.md"
        now = datetime.now(UTC).isoformat()

        is_update = path.exists() and mem_type == "entity"

        # Defensive coercion: agents sometimes pass tags/links as dicts or
        # lists-of-dicts. The downstream sqlite bind will fail with
        # "type 'dict' is not supported", so normalize to list[str] here and
        # log a warning when we drop non-string entries (or unwrap a dict).
        def _coerce_str_list(raw: Any, field_name: str) -> list[str]:
            if raw is None:
                return []
            if isinstance(raw, dict):
                log.warning(
                    "remember_arg_coerced",
                    field=field_name,
                    received_type="dict",
                    fix="taking keys",
                    mem_id=mem_id,
                )
                raw = list(raw.keys())
            if not isinstance(raw, list):
                log.warning(
                    "remember_arg_coerced",
                    field=field_name,
                    received_type=type(raw).__name__,
                    fix="wrapped",
                    mem_id=mem_id,
                )
                raw = [raw]
            cleaned: list[str] = []
            for item in raw:
                if isinstance(item, str):
                    cleaned.append(item)
                elif isinstance(item, dict):
                    # Common agent mistake: [{"name": "tag1"}, ...]
                    val = item.get("id") or item.get("name") or item.get("value")
                    if isinstance(val, str):
                        cleaned.append(val)
                        continue
                    log.warning(
                        "remember_arg_dropped_item",
                        field=field_name,
                        item_type="dict",
                        mem_id=mem_id,
                    )
                else:
                    cleaned.append(str(item))
            return cleaned

        tags: list[str] = _coerce_str_list(args.get("tags", []), "tags")
        links: list[str] = _coerce_str_list(args.get("links", []), "links")
        raw_imp = args.get("importance")
        imp: float | None = max(0.1, min(2.0, float(raw_imp))) if raw_imp is not None else None
        existing_skill_meta = memory.get_skill_meta(mem_id) if mem_type == "procedure" else None
        skill_meta = existing_skill_meta
        if mem_type == "procedure" and ("skill" in tags or "auto-extracted" in tags):
            import re as _re

            steps = _re.findall(r"^\s*\d+\.\s+.+$", args["content"], _re.MULTILINE)
            passed, reason = memory.skill_gate(
                args["content"],
                steps,
                exclude_id=mem_id if path.exists() else None,
            )
            if not passed:
                return _ok(f"Skill rejected: {reason}")

            trigger_pattern = ""
            content_text = args["content"]
            when_match = _re.search(
                r"## When to Use\s*\n(.+?)(?=\n## |\Z)", content_text, _re.DOTALL
            )
            if when_match:
                words = _re.findall(r"\b[a-z]{4,}\b", when_match.group(1).lower())
                if words:
                    trigger_pattern = "|".join(set(words[:8]))
            skill_meta = {
                "version": (existing_skill_meta["version"] + 1) if existing_skill_meta else 1,
                "source_tasks": (
                    existing_skill_meta.get("source_tasks", []) if existing_skill_meta else []
                ),
                "success_count": (
                    existing_skill_meta.get("success_count", 0) if existing_skill_meta else 0
                ),
                "failure_count": (
                    existing_skill_meta.get("failure_count", 0) if existing_skill_meta else 0
                ),
                "last_applied": (
                    existing_skill_meta.get("last_applied") if existing_skill_meta else None
                ),
                "confidence": (
                    existing_skill_meta.get("confidence", 0.6) if existing_skill_meta else 0.6
                ),
                "trigger_pattern": trigger_pattern,
            }
        fm: dict[str, Any] = {
            "id": mem_id,
            "type": mem_type,
            "tags": tags,
            "created": now if not path.exists() else read_frontmatter(path).get("created", now),
            "updated": now,
            "links": links,
        }
        if skill_meta is not None:
            fm["skill_meta"] = skill_meta
        body = yaml.dump(fm, default_flow_style=False)
        path.write_text(f"---\n{body}---\n\n# {title}\n\n{args['content']}\n")

        # Conflict detection after file write (so changelog is consistent with disk)
        change_note = ""
        if is_update:
            changes = memory.detect_changes(mem_id, args["content"], title)
            if changes:
                change_note = " Changes: " + "; ".join(changes)
                memory.record_memory_change(mem_id, changes)
        emb = await asyncio.to_thread(
            memory.index_memory,
            mem_id,
            mem_type,
            title,
            args["content"],
            tags,
            links,
            imp,
            skill_meta=skill_meta,
        )
        # Linking is a graph-write, not retrieval — track access but not utility
        if links:
            memory.touch_memories(links, useful=False)
        status = f"Remembered: {mem_id}"
        if change_note:
            status += change_note
        # Overlap detection for insights and entities (reuse embedding from index)
        if mem_type in ("insight", "entity"):
            similar = await asyncio.to_thread(
                memory.find_similar,
                mem_id,
                mem_type,
                args["content"],
                limit=3,
                embedding=emb,
            )
            if similar:
                items = "; ".join(
                    f"{s['id']} ({s['similarity']:.0%}): {s['body_preview'][:80]}" for s in similar
                )
                status += (
                    f"\n\nSimilar existing memories — review for overlap, "
                    f"contradiction, or consolidation:\n{items}\n"
                    f"Consider: merge content, archive old + 'supersedes' link, "
                    f"or keep both if complementary."
                )
        # Emit events for event-driven behavior triggers
        _EVENT_TYPES = {"episode": "new_episode", "insight": "new_insight", "goal": "goal_updated"}
        if mem_type in _EVENT_TYPES:
            evt = bus.emit(_EVENT_TYPES[mem_type], {"id": mem_id})
            log.info(
                "event_emitted",
                event_type=_EVENT_TYPES[mem_type],
                event_id=evt.id,
                mem_id=mem_id,
            )
        return _ok(status)

    @tool(
        "recall",
        "Search memories. Combine: query, type, after/before, related_to.",
        # All filters optional and combinable — full JSON Schema so the SDK does
        # not mark every key required (same bug class as remember).
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "type": {"type": "string"},
                "after": {"type": "string"},
                "before": {"type": "string"},
                "related_to": {"type": "string"},
            },
            "required": [],
        },
        annotations=_READ_ONLY,
    )
    async def mem_recall(args: dict[str, Any]) -> dict[str, Any]:
        # Off-loop: recall embeds the query and, on the letta backend, makes a
        # blocking HTTP call — neither may stall the event loop mid-turn.
        results = await asyncio.to_thread(
            memory.recall,
            query=args.get("query", ""),
            mem_type=args.get("type"),
            after=args.get("after"),
            before=args.get("before"),
            related_to=args.get("related_to"),
        )
        if not results:
            return _ok("No memories found")
        lines: list[str] = []
        for r in results:
            body = read_memory_body(r["type"], r["id"], settings.recall_content_limit)
            content = body or r.get("title", "")
            lines.append(f"**{r['id']}** ({r['type']})\n{content}")
        memory.touch_memories([r["id"] for r in results])
        return _ok("\n---\n".join(lines))

    @tool(
        "forget",
        "Archive memory (keeps file, removes from index)",
        {"id": str},
        annotations=_DESTRUCTIVE,
    )
    async def mem_forget(args: dict[str, Any]) -> dict[str, Any]:
        memory.archive_memory(args["id"])
        return _ok(f"Archived: {args['id']}")

    @tool(
        "recall_conversation",
        "Retrieve memories from a time window chronologically.",
        {"after": str, "before": str},
        annotations=_READ_ONLY,
    )
    async def mem_recall_conv(args: dict[str, Any]) -> dict[str, Any]:
        results = memory.recall_by_time_window(
            after=args["after"],
            before=args["before"],
        )
        if not results:
            return _ok("No memories in that time range")
        lines: list[str] = []
        for r in results:
            body = read_memory_body(r["type"], r["id"], settings.recall_content_limit)
            content = body or r.get("title", "")
            created = r.get("created", "")
            lines.append(f"[{created}] **{r['id']}** ({r['type']})\n{content}")
        memory.touch_memories([r["id"] for r in results])
        return _ok("\n---\n".join(lines))

    @tool(
        "connect",
        "Link two memories. Labels: related, involves, contributes_to, derived_from, "
        "uses, about, informed_by, supports, caused, supersedes, contradicts, "
        "blocked_by, enables. Set supersedes_rel to invalidate an old relationship.",
        _schema({"from_id": str, "to_id": str, "relationship": str, "supersedes_rel": str}, ['supersedes_rel']),
    )
    async def mem_link(args: dict[str, Any]) -> dict[str, Any]:
        note = ""
        if supersedes_rel := args.get("supersedes_rel"):
            invalidated = memory.invalidate_link(args["from_id"], args["to_id"], supersedes_rel)
            if invalidated:
                note = f" (invalidated '{supersedes_rel}')"
        created = memory.link_memories(args["from_id"], args["to_id"], args["relationship"])
        # Linking is a graph-write, not retrieval — track access but not utility
        memory.touch_memories([args["from_id"], args["to_id"]], useful=False)
        rel = args["relationship"]
        if created:
            return _ok(f"Linked: {args['from_id']} —{rel}→ {args['to_id']}{note}")
        return _ok(f"Already linked: {args['from_id']} —{rel}→ {args['to_id']}{note}")

    # --- Restore + Bulk memory (2 tools) ---

    @tool(
        "restore",
        "Restore archived memory to active",
        {"id": str},
    )
    async def mem_restore(args: dict[str, Any]) -> dict[str, Any]:
        restored = memory.restore_memory(args["id"])
        if restored:
            return _ok(f"Restored: {args['id']}")
        return _ok(f"Not found or not archived: {args['id']}")

    @tool(
        "bulk_memory",
        "Bulk ops on memories. action: retag|relink|archive. "
        "ids: memory IDs. tags (retag), link_to+relationship (relink).",
        _schema(
            {
                "action": str,
                "ids": list,
                "tags": list,
                "link_to": str,
                "relationship": str,
            },
            ["ids", "tags", "link_to", "relationship"],
        ),
    )
    async def mem_bulk(args: dict[str, Any]) -> dict[str, Any]:
        action: str = args["action"]
        ids: list[str] = args.get("ids", [])
        if not ids:
            return _ok("Error: no IDs provided")
        if action == "retag":
            tags: list[str] = args.get("tags", [])
            for mid in ids:
                memory.update_memory_tags(mid, tags)
        elif action == "relink":
            link_to = args.get("link_to", "")
            rel = args.get("relationship", "related")
            if not link_to:
                return _ok("Error: link_to required for relink")
            for mid in ids:
                memory.link_memories(mid, link_to, rel)
        elif action == "archive":
            for mid in ids:
                memory.archive_memory(mid)
        else:
            return _ok(f"Unknown action: {action}")
        return _ok(f"{action}: {len(ids)} memories updated")

    # --- Memory history (1 tool) ---

    @tool(
        "memory_history",
        "View change history for a memory",
        {"id": str},
        annotations=_READ_ONLY,
    )
    async def mem_history(args: dict[str, Any]) -> dict[str, Any]:
        history = memory.get_memory_history(args["id"])
        if not history:
            return _ok(f"No history for: {args['id']}")
        lines: list[str] = []
        for entry in history:
            changes = "; ".join(entry["changes"])
            lines.append(f"[{entry['timestamp']}] {changes}")
        return _ok("\n".join(lines))

    # --- Correction review (1 tool) ---

    @tool(
        "review_corrections",
        "Review pending memory corrections detected automatically. "
        "Returns pending corrections with original content, proposed correction, "
        "and confidence score. Use action: approve, reject, or modify with corrected_content.",
        _schema({"action": str, "correction_id": int, "corrected_content": str}, ['action', 'correction_id', 'corrected_content']),
        annotations=_DESTRUCTIVE,
    )
    async def mem_review_corrections(args: dict[str, Any]) -> dict[str, Any]:
        action = args.get("action", "list")
        if action == "list":
            pending = memory.get_pending_corrections(limit=5)
            if not pending:
                return _ok("No pending corrections to review.")
            lines: list[str] = []
            for p in pending:
                lines.append(
                    f"**Correction #{p['id']}** (confidence: {p['confidence']:.2f})\n"
                    f"Memory: {p['mem_id']}\n"
                    f"Proposed: {p['corrected_content'][:200]}\n"
                    f"Detected: {p['created_at']}\n"
                    f"Actions: approve (id={p['id']}), reject (id={p['id']}), "
                    f"modify (id={p['id']}, corrected_content=...)"
                )
            return _ok("\n---\n".join(lines))
        elif action in ("approve", "applied"):
            correction_id = args.get("correction_id")
            if not correction_id:
                return _ok("Error: correction_id required for approve/reject")
            result = memory.resolve_correction(correction_id, action)
            return _ok(f"Correction #{correction_id}: {result['status']}")
        elif action == "reject":
            correction_id = args.get("correction_id")
            if not correction_id:
                return _ok("Error: correction_id required for approve/reject")
            result = memory.resolve_correction(correction_id, "rejected")
            return _ok(f"Correction #{correction_id}: rejected")
        elif action == "modify":
            correction_id = args.get("correction_id")
            corrected_content = args.get("corrected_content")
            if not correction_id or not corrected_content:
                return _ok("Error: correction_id and corrected_content required for modify")
            result = memory.apply_correction(
                correction_id,
                corrected_content,
                confidence=0.85,
                source="agent_review",
            )
            memory.resolve_correction(correction_id, "applied")
            return _ok(f"Correction #{correction_id}: modified and applied ({result['status']})")
        else:
            return _ok(f"Unknown action: {action}. Use: list, approve, reject, modify")

    # --- Active attention (2 tools) ---

    @tool(
        "pin_attention",
        "Pin a commitment to active attention so it stays in working context "
        "across sessions. Use when the user says something matters or asks you "
        "to track something. Examples: 'track the Fanatics prep', 'watch for "
        "Naiara's email', 'remember I want to focus on deep work this week'.",
        _schema({"content": str, "related_id": str}, ['related_id']),
        annotations=_OPEN_WORLD,
    )
    async def t_pin_attention(args: dict[str, Any]) -> dict[str, Any]:
        from . import attention

        related = args.get("related_id") or None
        item_id = attention.pin(
            chat_id=chat_id,
            content=args["content"],
            origin="luke",
            related_id=related,
        )
        return _ok(f"Pinned attention #{item_id}: {args['content']}")

    @tool(
        "unpin_attention",
        "Remove an active-attention item by id. Use when a commitment is "
        "complete, cancelled, or resolved.",
        {"attention_id": int},
        annotations=_DESTRUCTIVE,
    )
    async def t_unpin_attention(args: dict[str, Any]) -> dict[str, Any]:
        from . import attention

        removed = attention.unpin(chat_id, int(args["attention_id"]))
        status = "removed" if removed else "not found"
        return _ok(f"Attention #{args['attention_id']}: {status}")

    # --- Cost (1 tool) ---

    @tool(
        "get_cost_report",
        "Cost/usage stats. period: today|week|month|all",
        _schema({"period": str}, ['period']),
        annotations=_READ_ONLY,
    )
    async def t_cost(args: dict[str, Any]) -> dict[str, Any]:
        report = db.get_cost_report(args.get("period", "month"))
        return _ok(report)

    # --- Deep Work Quality (1 tool) ---

    @tool(
        "log_deep_work_quality",
        "Record quality rating (1-5) for a deep work session on a goal. "
        "Call at end of every deep work session.",
        {"goal_id": str, "rating": int},
        annotations=_OPEN_WORLD,
    )
    async def t_quality(args: dict[str, Any]) -> dict[str, Any]:
        goal_id = args["goal_id"]
        rating = int(args["rating"])
        if not 1 <= rating <= 5:
            return _ok("Error: rating must be 1-5")
        db.log_deep_work_quality(goal_id, rating)
        scores = db.get_recent_quality_scores(goal_id, 3)
        avg = sum(scores) / len(scores) if scores else 0
        return _ok(f"Quality logged: {rating}/5 for {goal_id}. Last {len(scores)} avg: {avg:.1f}")

    # --- Browser (1 tool) ---

    @tool(
        "browse",
        "Open a URL and extract page content. Returns title + text. "
        "Optional: CSS selector to narrow extraction, screenshot to save a PNG.",
        _schema({"url": str, "selector": str, "screenshot": bool}, ['selector', 'screenshot']),
        annotations=_OPEN_WORLD,
    )
    async def t_browse(args: dict[str, Any]) -> dict[str, Any]:
        from playwright.async_api import async_playwright

        url: str = args["url"]
        selector: str | None = args.get("selector")
        take_screenshot: bool = args.get("screenshot", False)

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=30_000, wait_until="domcontentloaded")

                title = await page.title()
                final_url = page.url

                if selector:
                    elements = await page.query_selector_all(selector)
                    parts = [await el.inner_text() for el in elements]
                    content = "\n".join(p.strip() for p in parts if p.strip())
                else:
                    content = await page.inner_text("body")

                result_parts = [f"Title: {title}", f"URL: {final_url}", ""]

                if take_screenshot:
                    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                    ss_dir = settings.workspace_dir / "media" / "screenshots"
                    ss_dir.mkdir(parents=True, exist_ok=True)
                    path = ss_dir / f"screenshot_{ts}.png"
                    await page.screenshot(path=str(path))
                    result_parts.append(f"Screenshot: {path}")

                await browser.close()

            max_chars = 15_000
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n[Truncated — {len(content):,} total chars]"

            result_parts.append(content)
            return _ok("\n".join(result_parts))
        except Exception as exc:
            return _ok(f"Browse error: {exc}")

    @tool(
        "delegate",
        (
            "Spawn a self-contained agent job in the background and return "
            "immediately — the conversation stays unblocked while it runs. "
            "The job is durable: it survives restarts, is bounded by the agent "
            "timeout, and ALWAYS closes the loop — its final text output is "
            "relayed to Filipe automatically (as a reply to trigger_msg_id), "
            "and a crash sends an explicit failure notice instead of silence. "
            "prompt must be fully self-contained (no shared context from this turn). "
            "Use for: research that takes >30s, file builds, multi-step analysis."
        ),
        _schema({"prompt": str, "trigger_msg_id": int}, ['trigger_msg_id']),
        annotations=_OPEN_WORLD,
    )
    async def t_delegate(args: dict[str, Any]) -> dict[str, Any]:
        job_id = db.create_task(
            chat_id,
            format_delegation_prompt(args["prompt"], args.get("trigger_msg_id")),
            "once",
            datetime.now(UTC).isoformat(),
        )
        bus.emit(
            "cron_created",
            {
                "task_id": job_id,
                "schedule_type": "once",
                "prompt_preview": args["prompt"][:200],
            },
        )
        return _ok(f"job:{job_id} queued — runs in background and reports back when done.")

    return create_sdk_mcp_server(
        name="luke",
        version="1.0.0",
        tools=[
            t_send,
            t_photo,
            t_doc,
            t_voice,
            t_video,
            t_loc,
            t_poll,
            t_buttons,
            t_reply,
            t_fwd,
            t_react,
            t_get_reactions,
            t_edit,
            t_del,
            t_pin,
            sched,
            list_tasks_tool,
            delete_task_tool,
            mem_save,
            mem_recall,
            mem_recall_conv,
            mem_forget,
            mem_link,
            mem_restore,
            mem_bulk,
            mem_history,
            mem_review_corrections,
            t_pin_attention,
            t_unpin_attention,
            t_cost,
            t_quality,
            t_browse,
            t_delegate,
        ],
    )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _build_stop_hook(
    tool_count: dict[str, int],
    autonomous: bool,
    *,
    artifact_requested: bool = False,
    artifact_delivered_count: dict[str, int] | None = None,
    work_scheduled_count: dict[str, int] | None = None,
    artifact_gate_fired: dict[str, int] | None = None,
    source_read_requested: bool = False,
    source_read_count: dict[str, int] | None = None,
    source_gate_fired: dict[str, int] | None = None,
) -> HookCallback:
    """Factory returning a Stop hook closure with access to the run's tool count.

    When ``artifact_requested`` is set (the inbound message explicitly asked for
    a FILE deliverable), the hook enforces a forced-capture gate: the turn may
    not close until either the artifact actually shipped
    (``artifact_delivered_count`` > 0) or a durable handle was created
    (``work_scheduled_count`` > 0). This measures the FAR end of the pipe —
    receipt, not production — and fires at most ONCE per turn to avoid loops.

    When ``source_read_requested`` is set (the inbound message asked me to
    consult a readable SOURCE — "read the email", "check the doc"), the hook
    enforces a primary-source gate: the turn may not close until a read/fetch
    tool actually ran this turn (``source_read_count`` > 0). Also one-shot.
    """

    async def _stop_hook(
        input_data: StopHookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        # --- Forced artifact-request capture gate (interactive turns only) ---
        # The most-recurring drop: Filipe asks for a file ("give me a pdf") and
        # the turn ends with only topic-adjacent chatter. Block the stop once,
        # forcing delivery or a durable task. Fires at most once (artifact_gate_
        # fired) so a still-uncompleted turn can't trap the agent in a loop.
        if (
            not autonomous
            and artifact_requested
            and artifact_gate_fired is not None
            and artifact_gate_fired["n"] == 0
            and (artifact_delivered_count or {}).get("n", 0) == 0
            and (work_scheduled_count or {}).get("n", 0) == 0
        ):
            artifact_gate_fired["n"] = 1
            log.warning("artifact_gate_blocked_stop", tool_calls=tool_count["n"])
            bus.emit("artifact_gate_blocked_stop", {"tool_calls": tool_count["n"]})
            return {
                "decision": "block",
                "reason": (
                    "You asked-for-artifact check: Filipe explicitly requested a "
                    "FILE deliverable this turn (a pdf / doc / brief / report / "
                    "spreadsheet), but no file was sent (send_document/send_photo/"
                    "send_video) and no durable handle was created "
                    "(delegate / schedule_task / Task). A topic-adjacent message "
                    "is NOT delivery — this is the built-but-not-sent / "
                    "adjacent-helpfulness drop. Before you stop: either (a) build "
                    "and SEND the file now via send_document, or (b) if it "
                    "genuinely can't be finished this turn, create a durable task "
                    "(mcp__luke__delegate or schedule_task) capturing the exact "
                    "request so it can't evaporate. Do one of these now."
                ),
            }

        # --- Primary-source read gate (interactive turns only) ---
        # Filipe pointed me at a readable source ("read the email", "check the
        # doc") and the turn is about to close without any read/fetch tool
        # having run — i.e. I'm answering from my own summary, the exact Prerna
        # failure. Block once (source_gate_fired) so a genuinely-unreadable
        # source can't trap the agent in a Stop loop.
        if (
            not autonomous
            and source_read_requested
            and source_gate_fired is not None
            and source_gate_fired["n"] == 0
            and (source_read_count or {}).get("n", 0) == 0
        ):
            source_gate_fired["n"] = 1
            log.warning("source_read_gate_blocked_stop", tool_calls=tool_count["n"])
            bus.emit("source_read_gate_blocked_stop", {"tool_calls": tool_count["n"]})
            return {
                "decision": "block",
                "reason": (
                    "Primary-source check: Filipe asked you to consult a specific "
                    "readable source this turn (an email / doc / pdf / thread / "
                    "file), but you did NOT open it — no Read / Grep / WebFetch / "
                    "browse ran. Answering from your own prior summary instead of "
                    "the source is the recurring Prerna failure ('read the fucking "
                    "email'). Before you stop: actually open the source now (Read "
                    "the file, WebFetch/browse the link, Grep the thread) and "
                    "answer from what it says. If the source is genuinely NOT "
                    "accessible to you, do not guess — say so plainly and name the "
                    "access gap. Recall of your own memory does NOT satisfy this; "
                    "go to the primary source."
                ),
            }

        skill_prompt = ""
        if not autonomous and tool_count["n"] >= _AUTO_SKILL_THRESHOLD:
            skill_prompt = (
                f"\n8. Skill extraction: this conversation used {tool_count['n']} tool calls "
                "— it was complex. Before stopping, check whether a reusable "
                "procedure can be extracted:\n"
                "   - Did you solve something that is likely to come up again?\n"
                "   - Is there a clear sequence of steps that another session could follow?\n"
                "   If yes, save a procedure memory (type: procedure):\n"
                "     - ID: descriptive kebab-case "
                "(e.g. how-to-deploy-app, research-flight-options)\n"
                "     - Include: trigger condition, step-by-step approach, gotchas, example\n"
                "   If no clear reusable pattern exists, skip this — don't save noise."
            )
        return {
            "systemMessage": (
                "Session ending. Before you stop:\n"
                "1. Did you learn anything new about the user? Save it with 'remember'.\n"
                "2. Did any entity (person, project) change? Update their db.\n"
                "3. Was this conversation significant? Save an episode — "
                "include your reasoning: what approaches you considered, "
                "why you chose your solution, what worked or didn't.\n"
                "4. Did you notice a pattern or preference? Save an insight.\n"
                "5. Is there anything pending that needs follow-up? Schedule a reminder.\n"
                "6. Did the user mention wanting to achieve something? Create or update a goal."
                + skill_prompt
                + (
                    "\n7. Correction check: did any recalled information get corrected "
                    "during this conversation? If the user corrected a fact you remembered, "
                    "or you realized something you stored was wrong, use the remember tool "
                    "to update the entity with the corrected content. Corrections include: "
                    "factual updates, changed preferences, outdated information, "
                    "or anything you previously stored that is no longer accurate."
                )
            )
        }

    return cast(HookCallback, _stop_hook)


async def _pre_compact_hook(
    input_data: PreCompactHookInput,
    tool_use_id: str | None,
    hook_context: HookContext,
) -> SyncHookJSONOutput:
    log.info("session_compact", trigger=input_data["trigger"])
    manifest = context.build_preservation_manifest()
    return {"systemMessage": manifest}


async def _notification_hook(
    input_data: NotificationHookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    log.info(
        "sdk_notification",
        title=input_data.get("title", ""),
        message=input_data["message"],
    )
    return {}


_DUBLIN_TZ = ZoneInfo("Europe/Dublin")


async def _user_prompt_submit_hook(
    input_data: UserPromptSubmitHookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    """Inject current Dublin local time so day-of-week and schedule reasoning is always accurate."""
    now = datetime.now(_DUBLIN_TZ)
    time_str = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    day_name = now.strftime("%A")
    log.debug("user_prompt_submit", local_time=time_str, day=day_name)
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                f"Current local time (Dublin): {time_str}. Day of week: {day_name}."
            ),
        }
    }


# ---------------------------------------------------------------------------
# Tool scoping per model tier
# ---------------------------------------------------------------------------

_BUILTINS_ALL: list[str] = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "WebSearch",
    "WebFetch",
    "Task",
    "TaskOutput",
    "TaskStop",
    "TeamCreate",
    "TeamDelete",
    "SendMessage",
    "TodoWrite",
    "ToolSearch",
    "Skill",
    "NotebookEdit",
]

_BUILTINS_HAIKU: list[str] = ["Read", "Glob", "Grep"]

# All bare tool names (no prefix) — must match tools in _build_tools().
# Validated by test_all_mcp_tool_names_matches_registered.
_ALL_MCP_TOOL_NAMES: list[str] = [
    "send_message",
    "send_photo",
    "send_document",
    "send_voice",
    "send_video",
    "send_location",
    "send_poll",
    "send_buttons",
    "reply",
    "forward",
    "react",
    "get_reactions",
    "edit_message",
    "delete_message",
    "pin",
    "schedule_task",
    "list_tasks",
    "delete_task",
    "remember",
    "recall",
    "recall_conversation",
    "forget",
    "connect",
    "restore",
    "bulk_memory",
    "memory_history",
    "review_corrections",
    "pin_attention",
    "unpin_attention",
    "get_cost_report",
    "log_deep_work_quality",
    "browse",
    "delegate",
]

_MCP_TOOLS_HAIKU: frozenset[str] = frozenset(
    {"send_message", "reply", "react", "remember", "recall", "recall_conversation"}
)
_MCP_TOOLS_SONNET_EXCLUDE: frozenset[str] = frozenset({"schedule_task", "bulk_memory"})


def _mcp(name: str) -> str:
    return f"mcp__luke__{name}"


# Pre-computed per-tier tool lists (inputs are module-level constants)
_ALLOWED_HAIKU: list[str] = _BUILTINS_HAIKU + [_mcp(n) for n in _MCP_TOOLS_HAIKU]
_ALLOWED_SONNET: list[str] = _BUILTINS_ALL + [
    _mcp(n) for n in _ALL_MCP_TOOL_NAMES if n not in _MCP_TOOLS_SONNET_EXCLUDE
]
_ALLOWED_OPUS: list[str] = [*_BUILTINS_ALL, "mcp__luke__*"]


def _allowed_tools_for_model(model: str) -> list[str]:
    """Return the allowed_tools list scoped to the model tier."""
    if model == "haiku":
        return _ALLOWED_HAIKU
    if model == "sonnet":
        return _ALLOWED_SONNET
    return _ALLOWED_OPUS


# Tier aliases ("haiku"/"sonnet"/"opus") are routing keys throughout luke; the
# SDK gets explicit model IDs so a model generation change is a deliberate edit
# here — not a side effect of whichever CLI version happens to be installed.
_MODEL_IDS: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
}


def _resolve_model_id(model: str) -> str:
    """Map a tier alias to its pinned model ID; pass explicit IDs through."""
    return _MODEL_IDS.get(model, model)


# ---------------------------------------------------------------------------
# Run agent
# ---------------------------------------------------------------------------


async def run_agent(
    *,
    chat_id: str,
    prompt: str | list[dict[str, Any]],
    session_id: str | None,
    bot: Bot,
    model: str | None = None,
    max_turns: int | None = None,
    max_sends: int | None = None,
    effort: Literal["low", "medium", "high", "max"] | None = None,
    thinking: ThinkingConfig | None = None,
    autonomous: bool = False,
    urgent: bool = False,
    user_text: str | None = None,
) -> AgentResult:
    # Mark this run's access as human-driven (or not) so the injection ranker's
    # relevance signal (human_last_accessed) isn't polluted by autonomous churn.
    memory.human_turn.set(not autonomous)

    root = settings.luke_dir
    effective_model = model or settings.agent_model

    # Load LUKE.md persona (separate from project CLAUDE.md which is dev instructions)
    persona_path = root / "LUKE.md"
    persona = persona_path.read_text() if persona_path.exists() else ""

    # Load constitutional layer — non-compressible behavioral anchors
    constitutional_path = root / "constitutional.yaml"
    if constitutional_path.exists():
        persona += (
            "\n\n<constitutional>\n" + constitutional_path.read_text() + "\n</constitutional>"
        )

    # Inject working memory — priority memories scored and selected at session start
    # Adaptive budget: low-effort messages get less context (saves tokens + noise)
    # user_text is the caller's clean, envelope-free user message, captured BEFORE any
    # memory-context injection into `prompt`. Prefer it so retrieval queries and the
    # file-artifact/source-read Stop gates read what the user actually typed — not the
    # injected memory blob (which is prepended for str prompts and inserted at index 0
    # for multimodal list prompts, masking the real message entirely).
    prompt_text_for_context = _context_query(prompt, user_text)
    _EFFORT_BUDGET = {"low": 3_000, "medium": 6_000, "high": 12_000, "max": 12_000}
    ctx_budget = 12_000 if autonomous else _EFFORT_BUDGET.get(effort or "high", 12_000)
    # Phase 4.1: when agent_backend="letta", source the always-in-context world model from the
    # luke-agent-claude core blocks (self-editing, packed in 2.2a) instead of re-injecting the
    # sqlite working-memory blob. Fail-safe: a None assembly falls through to the sqlite blob,
    # so a down/empty Letta agent degrades to current SDK behavior. Defaults off (backend="sdk").
    working_ctx: str | None = None
    if settings.agent_backend == "letta":
        working_ctx = _letta_agent.build_letta_context()
    if working_ctx is None:
        working_ctx = context.build_working_context(
            query=prompt_text_for_context, budget_tokens=ctx_budget
        )
    system_append = _compose_system_append(persona, working_ctx)

    # Per-run counters and timing state (closed over by hooks)
    send_count = {"n": 0}
    recall_count = {"n": 0}  # incremented when recall/recall_conversation runs
    work_scheduled_count = {"n": 0}  # incremented when Task/schedule_task runs
    artifact_delivered_count = {"n": 0}  # incremented when a file/attachment ships
    artifact_gate_fired = {"n": 0}  # one-shot guard for the Stop-hook artifact gate
    source_read_count = {"n": 0}  # incremented when a read/fetch tool runs
    source_gate_fired = {"n": 0}  # one-shot guard for the Stop-hook source gate
    tool_count: dict[str, int] = {"n": 0}
    # Detect an explicit inbound FILE-artifact request so the Stop hook can
    # enforce delivery-or-durable-handle before the turn closes.
    artifact_requested = not autonomous and _requests_file_artifact(prompt_text_for_context)
    # Detect an explicit inbound request to consult a readable SOURCE so the
    # Stop hook can enforce an actual read before the turn closes.
    source_read_requested = not autonomous and _requests_source_read(prompt_text_for_context)
    tool_start_times: dict[str, float] = {}  # tool_use_id -> monotonic start
    subagent_start_times: dict[str, float] = {}  # agent_id -> monotonic start
    effective_max_sends = max_sends if max_sends is not None else settings.max_sends_per_run

    async def _pre_tool_hook(
        input_data: PreToolUseHookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        tool_name = input_data["tool_name"]
        log.info("tool_use", tool=tool_name, input=_trunc(str(input_data["tool_input"])))
        tool_count["n"] += 1
        # Record start time for latency tracking in PostToolUse
        tid = input_data.get("tool_use_id") or tool_use_id
        if tid:
            tool_start_times[tid] = time.monotonic()
        # Track recall calls to gate references-to-past-events in send tools.
        if tool_name in _RECALL_TOOLS:
            recall_count["n"] += 1
        # Track agent/schedule spawns for the overnight-commitment gate.
        if tool_name in _AGENT_SCHEDULE_TOOLS:
            work_scheduled_count["n"] += 1
        # Track actual file/attachment deliveries for the artifact-request gate.
        if tool_name in _ARTIFACT_SEND_TOOLS:
            artifact_delivered_count["n"] += 1
        # Track read/fetch calls for the primary-source gate.
        if tool_name in _SOURCE_READ_TOOLS:
            source_read_count["n"] += 1
        # --- Background-work routing gate (interactive turns only) ---
        # Harness `Task` sub-agents are children of the per-turn client: they
        # die the moment Filipe sends his next message (the July 3 2026
        # teardown bug) and never report back on their own — the repeated
        # "so?" silence he flagged on July 15 ("run subagents and let me know
        # when they are done!", "How will you enforce that?!"). The `delegate`
        # MCP tool survives teardown AND always sends its result back. When
        # Filipe is live (autonomous=False), force background work through it.
        # Autonomous runs (crons/deep work) have no interrupting message, so a
        # within-turn Task is safe there.
        if tool_name == "Task" and not autonomous:
            log.warning("task_blocked_use_delegate", chat_id=chat_id)
            bus.emit("task_blocked_use_delegate", {"chat_id": chat_id})
            return {
                "decision": "block",
                "reason": (
                    "Don't spawn a harness Task sub-agent in a live "
                    "conversation — it dies when Filipe sends his next message "
                    "and never reports back, which is the 'so?' silence. Use "
                    "mcp__luke__delegate(prompt=<self-contained>, "
                    "trigger_msg_id=<the msg that asked>) instead: it survives "
                    "teardown and always sends its result back. For a quick "
                    "lookup that finishes this turn, just do it inline "
                    "(WebSearch/WebFetch/Read) — no sub-agent needed."
                ),
            }
        # --- Scheduled-task duplicate gate (all runs) ---
        # Block a schedule_task that would double-book an existing active task
        # for the same deliverable (see _duplicate_pending_task). Steers to
        # reuse/update the existing task instead of stacking a second nudge.
        if tool_name == "mcp__luke__schedule_task":
            tool_input = cast(object, input_data.get("tool_input", {}))
            if isinstance(tool_input, dict):
                try:
                    dup = _duplicate_pending_task(tool_input, db.list_tasks(chat_id))
                except Exception as e:  # never let the gate crash a schedule
                    log.warning("task_dedup_gate_error", error=str(e))
                    dup = None
                if dup is not None:
                    log.warning(
                        "duplicate_task_blocked",
                        chat_id=chat_id,
                        existing_id=dup.get("id"),
                        new_preview=str(tool_input.get("prompt", ""))[:80],
                    )
                    bus.emit(
                        "duplicate_task_blocked",
                        {
                            "existing_id": dup.get("id"),
                            "existing_value": dup.get("schedule_value"),
                            "new_preview": str(tool_input.get("prompt", ""))[:100],
                        },
                    )
                    return {
                        "decision": "block",
                        "reason": (
                            "Near-duplicate of an existing active task "
                            f"(id={dup.get('id')}, fires {dup.get('schedule_value')}): "
                            f'"{str(dup.get("prompt", ""))[:80]}". Filipe would get '
                            "the same nudge twice. Don't stack a second task — "
                            "either leave the existing one, or delete_task(it) and "
                            "reschedule once with the corrected time/content. "
                            "(proc-scheduled-task-dedup)"
                        ),
                    }

        if tool_name in _SEND_TOOLS:
            send_count["n"] += 1
            if send_count["n"] > effective_max_sends:
                return {"decision": "block", "reason": "Rate limit: too many outbound messages"}

            # --- Overnight-commitment gate (all runs) ---
            # Block sends that commit to future delivery when no agent/
            # schedule_task has been spawned this turn. Without this, the
            # commitment exists only as text and no execution path runs
            # (the May 14 2026 overnight-prep failure).
            tool_input = cast(object, input_data.get("tool_input", {}))
            if isinstance(tool_input, dict):
                msg_text = tool_input.get("text", "") or tool_input.get("caption", "")
            else:
                msg_text = ""
            if work_scheduled_count["n"] == 0 and _commits_future_work(msg_text):
                log.warning(
                    "commitment_blocked_no_execution",
                    chat_id=chat_id,
                    tool=tool_name,
                    preview=msg_text[:100],
                )
                bus.emit(
                    "commitment_blocked_no_execution",
                    {
                        "tool": tool_name,
                        "preview": msg_text[:100],
                    },
                )
                return {
                    "decision": "block",
                    "reason": (
                        "Commitment to future delivery detected but no "
                        "agent or scheduled task was spawned this turn. "
                        "Either spawn the work via Task / schedule_task "
                        "before sending, or rephrase the message to "
                        "remove the commitment."
                    ),
                }

            # --- Weekday/date consistency gate (all runs) ---
            # Deterministic. Blocks "Tuesday Aug 7" when Aug 7 is a Friday —
            # the error shipped to Filipe 2026-07-31 about his visa interview
            # while memory held the correct day. feedback-dates-accuracy was
            # advisory only; this makes it enforced.
            weekday_error = _weekday_claim_error(msg_text)
            if weekday_error:
                log.warning(
                    "weekday_mismatch_blocked",
                    chat_id=chat_id,
                    tool=tool_name,
                    reason=weekday_error,
                    preview=msg_text[:100],
                )
                bus.emit(
                    "weekday_mismatch_blocked",
                    {
                        "tool": tool_name,
                        "reason": weekday_error,
                        "preview": msg_text[:100],
                    },
                )
                return {"decision": "block", "reason": weekday_error}

            # Global hourly attention budget for autonomous runs (behaviors + crons).
            # Urgent behaviors (e.g. proactive_scan) can draw from a small reserve
            # beyond the normal cap so they can still reach the user when the normal
            # budget is exhausted.  Non-urgent behaviors are blocked at max_sends_per_hour.
            if autonomous:
                hourly_count = db.count_recent_outbound(chat_id)
                normal_limit = settings.max_sends_per_hour
                urgent_limit = normal_limit + settings.attention_urgent_reserve
                effective_limit = urgent_limit if urgent else normal_limit
                if hourly_count >= effective_limit:
                    log.warning(
                        "hourly_budget_exceeded",
                        chat_id=chat_id,
                        hourly_count=hourly_count,
                        limit=effective_limit,
                        urgent=urgent,
                    )
                    return {"decision": "block", "reason": "Hourly message budget exceeded"}

                # --- Outbound message quality gate (autonomous only) ---
                # Read the caption as a fallback so document/media sends are
                # evaluated on their caption, not a non-existent "text" field.
                # Only enforce the check on text-primary tools OR when a caption
                # is actually present — a file with no caption is a valid send,
                # not an "empty message" (bug fixed 2026-07-13; false-positive
                # blocked every document send since 2026-05-16).
                tool_input = cast(object, input_data.get("tool_input", {}))
                if isinstance(tool_input, dict):
                    msg_text = tool_input.get("text", "") or tool_input.get("caption", "")
                else:
                    msg_text = ""
                if tool_name in _TEXT_PRIMARY_TOOLS or msg_text.strip():
                    rejection = _check_outbound_quality(msg_text)
                else:
                    rejection = None
                if rejection:
                    log.warning(
                        "message_rejected",
                        chat_id=chat_id,
                        reason=rejection,
                        preview=msg_text[:100],
                    )
                    bus.emit(
                        "message_rejected",
                        {
                            "reason": rejection,
                            "tool": tool_name,
                            "preview": msg_text[:100],
                        },
                    )
                    return {"decision": "block", "reason": f"Quality gate: {rejection}"}

                # --- Recall-before-reference gate (autonomous only) ---
                # If the draft references past events but no recall was called
                # this turn, block so the agent grounds in actual memory before
                # asking about something it should already know.
                if recall_count["n"] == 0 and _references_past_events(msg_text):
                    log.warning(
                        "send_blocked_no_recall",
                        chat_id=chat_id,
                        tool=tool_name,
                        preview=msg_text[:100],
                    )
                    bus.emit(
                        "send_blocked_no_recall",
                        {
                            "tool": tool_name,
                            "preview": msg_text[:100],
                        },
                    )
                    return {
                        "decision": "block",
                        "reason": (
                            "Reference to past events detected; call recall "
                            "(or recall_conversation) first to ground in "
                            "actual memory."
                        ),
                    }

                # --- Freshness gate (autonomous only) — L1 ---
                # Pre-send structural check: re-read user's latest message
                # and abort if the draft contradicts or stale-responds to
                # it. Runs BEFORE the critic so we only spend on drafts
                # that survived everything else. Fail-open via check_freshness.
                if settings.freshness_enabled and msg_text and len(msg_text) >= 30:
                    recent_rows = db.get_recent_messages(chat_id, limit=10)
                    user_msgs = [
                        r for r in recent_rows if r.get("sender_name") != settings.assistant_name
                    ][-4:]
                    if user_msgs:
                        try:
                            latest_ts = str(user_msgs[-1].get("timestamp", ""))
                            latest = datetime.fromisoformat(latest_ts)
                            if latest.tzinfo is None:
                                latest = latest.replace(tzinfo=UTC)
                            age_minutes = (datetime.now(UTC) - latest).total_seconds() / 60
                        except ValueError, TypeError:
                            age_minutes = 999.0
                        if age_minutes <= settings.freshness_window_minutes:
                            from .critic import check_freshness

                            fresh_verdict = await check_freshness(msg_text, user_msgs)
                            bus.emit(
                                "freshness_ran",
                                {
                                    "tool": tool_name,
                                    "decision": fresh_verdict.decision,
                                    "reason": fresh_verdict.reason,
                                    "age_minutes": round(age_minutes, 1),
                                },
                            )
                            if fresh_verdict.decision != "pass":
                                log.warning(
                                    "freshness_blocked",
                                    chat_id=chat_id,
                                    tool=tool_name,
                                    verdict=fresh_verdict.decision,
                                    reason=fresh_verdict.reason,
                                    preview=msg_text[:100],
                                )
                                bus.emit(
                                    "freshness_blocked",
                                    {
                                        "tool": tool_name,
                                        "verdict": fresh_verdict.decision,
                                        "reason": fresh_verdict.reason,
                                        "preview": msg_text[:100],
                                    },
                                )
                                return {
                                    "decision": "block",
                                    "reason": (
                                        f"Freshness ({fresh_verdict.decision}): "
                                        f"{fresh_verdict.reason}"
                                    ),
                                }

                # --- Outbound critic (autonomous only) — F4 ---
                # Last gate: cheap haiku pass for tone/factuality/fit. Runs
                # AFTER cheap regex/state gates so we only spend on drafts
                # that survived everything else. Fail-open semantics live
                # inside critique_outbound itself.
                if settings.critic_enabled and msg_text and len(msg_text) >= 20:
                    from .critic import critique_outbound

                    verdict = await critique_outbound(msg_text, {"tool": tool_name})
                    bus.emit(
                        "critic_ran",
                        {
                            "tool": tool_name,
                            "decision": verdict.decision,
                            "reason": verdict.reason,
                            "msg_len": len(msg_text),
                        },
                    )
                    if verdict.decision != "pass":
                        log.warning(
                            "critic_blocked",
                            chat_id=chat_id,
                            tool=tool_name,
                            verdict=verdict.decision,
                            reason=verdict.reason,
                            preview=msg_text[:100],
                        )
                        bus.emit(
                            "critic_blocked",
                            {
                                "tool": tool_name,
                                "verdict": verdict.decision,
                                "reason": verdict.reason,
                                "preview": msg_text[:100],
                            },
                        )
                        return {
                            "decision": "block",
                            "reason": (f"Critic ({verdict.decision}): {verdict.reason}"),
                        }
        return {}

    async def _post_tool_hook(
        input_data: PostToolUseHookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        tool_name = input_data["tool_name"]
        tid = input_data.get("tool_use_id") or tool_use_id
        duration_ms: int | None = None
        if tid and tid in tool_start_times:
            duration_ms = int((time.monotonic() - tool_start_times.pop(tid)) * 1000)
        agent_id = input_data.get("agent_id")
        agent_type = input_data.get("agent_type")
        payload: dict[str, Any] = {"tool": tool_name, "success": True}
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if agent_id:
            payload["agent_id"] = agent_id
        if agent_type:
            payload["agent_type"] = agent_type
        log.info("tool_complete", tool=tool_name, duration_ms=duration_ms, agent_id=agent_id)
        bus.emit("tool_use", payload)
        return {}

    async def _post_tool_failure_hook(
        input_data: PostToolUseFailureHookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        tool_name = input_data["tool_name"]
        error = input_data.get("error", "unknown")
        tid = input_data.get("tool_use_id") or tool_use_id
        # Clean up start time if present
        if tid:
            tool_start_times.pop(tid, None)
        agent_id = input_data.get("agent_id")
        agent_type = input_data.get("agent_type")
        payload = {"tool": tool_name, "success": False, "error": str(error)[:500]}
        if agent_id:
            payload["agent_id"] = agent_id
        if agent_type:
            payload["agent_type"] = agent_type
        log.warning("tool_failure", tool=tool_name, error=_trunc(str(error)), agent_id=agent_id)
        bus.emit("tool_failure", payload)
        return {}

    async def _subagent_start_hook(
        input_data: SubagentStartHookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        agent_id = input_data["agent_id"]
        agent_type = input_data["agent_type"]
        subagent_start_times[agent_id] = time.monotonic()
        log.info("subagent_start", agent_id=agent_id, agent_type=agent_type)
        bus.emit("subagent_start", {"agent_id": agent_id, "agent_type": agent_type})
        return {}

    async def _subagent_stop_hook(
        input_data: SubagentStopHookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        agent_id = input_data["agent_id"]
        agent_type = input_data["agent_type"]
        duration_ms: int | None = None
        if agent_id in subagent_start_times:
            duration_ms = int((time.monotonic() - subagent_start_times.pop(agent_id)) * 1000)
        log.info("subagent_stop", agent_id=agent_id, agent_type=agent_type, duration_ms=duration_ms)
        bus.emit(
            "subagent_stop",
            {"agent_id": agent_id, "agent_type": agent_type, "duration_ms": duration_ms},
        )
        return {}

    hooks: dict[HookEvent, list[HookMatcher]] = {
        "Stop": [
            HookMatcher(
                hooks=[
                    _build_stop_hook(
                        tool_count,
                        autonomous,
                        artifact_requested=artifact_requested,
                        artifact_delivered_count=artifact_delivered_count,
                        work_scheduled_count=work_scheduled_count,
                        artifact_gate_fired=artifact_gate_fired,
                        source_read_requested=source_read_requested,
                        source_read_count=source_read_count,
                        source_gate_fired=source_gate_fired,
                    )
                ]
            )
        ],
        "PreToolUse": [HookMatcher(hooks=[cast(HookCallback, _pre_tool_hook)])],
        "PostToolUse": [HookMatcher(hooks=[cast(HookCallback, _post_tool_hook)])],
        "PostToolUseFailure": [HookMatcher(hooks=[cast(HookCallback, _post_tool_failure_hook)])],
        "PreCompact": [HookMatcher(hooks=[cast(HookCallback, _pre_compact_hook)])],
        "Notification": [HookMatcher(hooks=[cast(HookCallback, _notification_hook)])],
        "SubagentStart": [HookMatcher(hooks=[cast(HookCallback, _subagent_start_hook)])],
        "SubagentStop": [HookMatcher(hooks=[cast(HookCallback, _subagent_stop_hook)])],
        "UserPromptSubmit": [HookMatcher(hooks=[cast(HookCallback, _user_prompt_submit_hook)])],
    }

    allowed = _allowed_tools_for_model(effective_model)

    # Fallback model must differ from main model (SDK requirement)
    fallback: str | None = settings.agent_fallback_model
    if fallback == effective_model:
        fallback = None

    options = ClaudeAgentOptions(
        cwd=str(root),
        resume=session_id,
        model=_resolve_model_id(effective_model),
        fallback_model=_resolve_model_id(fallback) if fallback else None,
        system_prompt={"type": "preset", "preset": "claude_code", "append": system_append},
        allowed_tools=allowed,
        permission_mode="bypassPermissions",
        setting_sources=["project", "user"],
        mcp_servers={
            "luke": _build_tools(chat_id, bot),
        },
        thinking=thinking if thinking is not None else ThinkingConfigAdaptive(type="adaptive"),
        effort=effort if effort is not None else "high",
        max_turns=max_turns if max_turns is not None else settings.agent_max_turns,
        include_partial_messages=settings.streaming_enabled,
        enable_file_checkpointing=True,
        sandbox={"enabled": True, "autoAllowBashIfSandboxed": True},
        hooks=hooks,
        agents={
            "researcher": AgentDefinition(
                description=(
                    "Web research agent — search, fetch, and synthesize information from the web"
                ),
                prompt=(
                    "Search the web and gather information. Be thorough: check "
                    "multiple sources, cross-reference facts, and return "
                    "structured findings with sources cited."
                ),
                tools=["WebSearch", "WebFetch", "Read", "Grep"],
                model=_MODEL_IDS["opus"],
            ),
            "coder": AgentDefinition(
                description=(
                    "Code and file worker — write, edit, and test Python code "
                    "or process files in the workspace"
                ),
                prompt=(
                    "Write clean, well-organized Python code. Prefer Python for "
                    "all scripting tasks. Read existing files before modifying. "
                    "Keep code modular with clear separation of concerns. "
                    "Run tests or validation after changes. "
                    "Work in the luke/workspace/ directory."
                ),
                tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
                model=_MODEL_IDS["opus"],
            ),
            "memory_curator": AgentDefinition(
                description=("Memory organizer — consolidate, retag, link, and clean up memories"),
                prompt=(
                    "You manage Luke's memory system. Your job: find related "
                    "memories, create connections, retag for better retrieval, "
                    "archive redundant entries, and synthesize insights from "
                    "clusters. Use the recall, remember, connect, forget, and "
                    "bulk_memory tools."
                ),
                tools=[
                    "mcp__luke__recall",
                    "mcp__luke__recall_conversation",
                    "mcp__luke__remember",
                    "mcp__luke__connect",
                    "mcp__luke__forget",
                    "mcp__luke__restore",
                    "mcp__luke__bulk_memory",
                ],
                model=_MODEL_IDS["haiku"],
            ),
        },
    )

    prompt_text = prompt if isinstance(prompt, str) else str(prompt[0].get("text", ""))
    log.info(
        "agent_run",
        chat=chat_id,
        prompt=_trunc(prompt_text),
        resume=bool(session_id),
    )

    result = AgentResult()
    async with ClaudeSDKClient(options=options) as client:
        _active_clients[chat_id] = client
        try:
            if isinstance(prompt, list):

                async def _multimodal() -> AsyncIterator[dict[str, Any]]:
                    yield {
                        "type": "user",
                        "message": {"role": "user", "content": prompt},
                        "session_id": session_id or "default",
                    }

                await client.query(_multimodal())
            else:
                await client.query(prompt)
            # Streaming state — progressive delivery to Telegram
            _stream_buf = ""  # accumulated raw text from stream events
            _stream_msg_id: int | None = None  # Telegram message ID for edits
            _stream_last_edit = 0.0  # monotonic time of last edit
            _stream_enabled = settings.streaming_enabled and not autonomous

            async for msg in client.receive_response():
                if _stream_enabled and isinstance(msg, StreamEvent):
                    event = msg.event
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            _stream_buf += delta.get("text", "")
                            clean = _clean_streaming_text(_stream_buf)
                            now = time.monotonic()
                            if (
                                clean
                                and len(clean) >= settings.streaming_min_chars
                                and now - _stream_last_edit >= settings.streaming_edit_interval
                            ):
                                # Truncate for Telegram limit, add cursor
                                display = clean[: _TG_MAX_MSG_LEN - 10]
                                if len(clean) > _TG_MAX_MSG_LEN - 10:
                                    display += "…"
                                else:
                                    display += _STREAMING_CURSOR
                                try:
                                    if _stream_msg_id:
                                        await bot.edit_message_text(
                                            text=display,
                                            chat_id=int(chat_id),
                                            message_id=_stream_msg_id,
                                            parse_mode=None,
                                        )
                                    else:
                                        sent = await bot.send_message(
                                            int(chat_id), display, parse_mode=None
                                        )
                                        _stream_msg_id = sent.message_id
                                    _stream_last_edit = now
                                except Exception:
                                    pass  # best-effort — don't break agent run

                elif isinstance(msg, ResultMessage):
                    result.session_id = result.session_id or msg.session_id
                    if msg.total_cost_usd is not None:
                        result.cost_usd = msg.total_cost_usd
                    result.num_turns = msg.num_turns
                    result.duration_api_ms = msg.duration_api_ms
                    if msg.usage:
                        result.input_tokens = msg.usage.get("input_tokens", 0)
                        result.output_tokens = msg.usage.get("output_tokens", 0)
                        result.cache_create_tokens = msg.usage.get("cache_creation_input_tokens", 0)
                        result.cache_read_tokens = msg.usage.get("cache_read_input_tokens", 0)
                        log.info(
                            "agent_usage",
                            input=result.input_tokens,
                            output=result.output_tokens,
                            cache_create=result.cache_create_tokens,
                            cache_read=result.cache_read_tokens,
                        )
                    if msg.result:
                        text = _INTERNAL_RE.sub("", msg.result).strip()
                        text = _INTERNAL_OPEN_RE.sub("", text).strip()
                        if text and not _is_leaked_internal(text):
                            result.texts.append(text)

            result.streaming_msg_id = _stream_msg_id
        finally:
            _active_clients.pop(chat_id, None)

    result.sent_messages = send_count["n"]
    result.tool_uses = tool_count["n"]
    # Subagents that started but never stopped were killed by the turn ending
    # (timeout/interrupt/teardown). Surface it — a dead subagent must never be
    # indistinguishable from a finished one.
    if subagent_start_times:
        log.warning(
            "subagents_orphaned",
            chat_id=chat_id,
            count=len(subagent_start_times),
            agent_ids=list(subagent_start_times),
        )
    return result
