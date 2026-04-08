"""Claude Agent SDK integration with in-process MCP tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    create_sdk_mcp_server,
    tool,
)
from claude_agent_sdk.types import (
    HookEvent,
    StreamEvent,
    SyncHookJSONOutput,
    TextBlock,
    ThinkingConfig,
    ThinkingConfigAdaptive,
)
from structlog.stdlib import BoundLogger

from . import context, db, memory
from .bus import bus
from .config import settings
from .memory import MEMORY_DIRS, read_frontmatter, read_memory_body, sanitize_memory_id

log: BoundLogger = structlog.get_logger()

_INTERNAL_RE = re.compile(r"<internal>[\s\S]*?</internal>")
_INTERNAL_OPEN_RE = re.compile(r"<internal>[\s\S]*$")  # unclosed tag at end
_LOG_TRUNCATION = 100  # chars for log message truncation
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


_AUTO_SKILL_THRESHOLD: int = 5  # tool calls to trigger procedure extraction in stop hook

# Active client registry — enables external interruption of running agents
_active_clients: dict[str, ClaudeSDKClient] = {}


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
        {"chat_id": str, "text": str, "silent": bool},
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
        {"chat_id": str, "path": str, "caption": str},
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
        {"chat_id": str, "path": str, "caption": str},
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
        {"chat_id": str, "path": str},
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
        {"chat_id": str, "path": str, "caption": str},
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
        {"chat_id": str, "latitude": float, "longitude": float},
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
        {"chat_id": str, "question": str, "options": list, "is_anonymous": bool},
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
        {"chat_id": str, "text": str, "buttons": list},
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
        {"chat_id": str, "message_id": str, "text": str},
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
        {"chat_id": str, "message_id": str, "emoji": str},
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
        {"msg_id": int, "sender_id": str, "sentiment": str, "limit": int},
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
        {"chat_id": str, "message_id": str, "text": str},
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
        {"chat_id": str, "message_id": str},
        annotations=_DESTRUCTIVE,
    )
    async def t_del(args: dict[str, Any]) -> dict[str, Any]:
        await bot.delete_message(chat_id=_target(args), message_id=int(args["message_id"]))
        return _ok("Deleted")

    @tool(
        "pin",
        "Pin message",
        {"chat_id": str, "message_id": str},
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
            return _ok(f"Deleted task {args['task_id']}")
        return _ok(f"Task {args['task_id']} not found")

    # --- Memory (8 tools) ---

    @tool(
        "remember",
        "Save memory. type: entity|episode|procedure|insight|goal. "
        "importance: 0.1-2.0 (default 1.0). Returns change summary on entity update.",
        {
            "id": str,
            "type": str,
            "title": str,
            "content": str,
            "tags": list,
            "links": list,
            "importance": float,
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

        tags: list[str] = args.get("tags", [])
        links: list[str] = args.get("links", [])
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
        {"query": str, "type": str, "after": str, "before": str, "related_to": str},
        annotations=_READ_ONLY,
    )
    async def mem_recall(args: dict[str, Any]) -> dict[str, Any]:
        results = memory.recall(
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
        {"from_id": str, "to_id": str, "relationship": str, "supersedes_rel": str},
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
        {
            "action": str,
            "ids": list,
            "tags": list,
            "link_to": str,
            "relationship": str,
        },
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
        {"action": str, "correction_id": int, "corrected_content": str},
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

    # --- Cost (1 tool) ---

    @tool(
        "get_cost_report",
        "Cost/usage stats. period: today|week|month|all",
        {"period": str},
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
        {"url": str, "selector": str, "screenshot": bool},
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
            t_cost,
            t_quality,
            t_browse,
        ],
    )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _build_stop_hook(tool_count: dict[str, int], autonomous: bool) -> HookCallback:
    """Factory returning a Stop hook closure with access to the run's tool count."""

    async def _stop_hook(
        input_data: StopHookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
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
        "hookEventName": "UserPromptSubmit",
        "additionalContext": f"Current local time (Dublin): {time_str}. Day of week: {day_name}.",
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
    "get_cost_report",
    "log_deep_work_quality",
    "browse",
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
    max_budget_usd: float | None = None,
    max_sends: int | None = None,
    effort: Literal["low", "medium", "high", "max"] | None = None,
    thinking: ThinkingConfig | None = None,
    autonomous: bool = False,
    urgent: bool = False,
) -> AgentResult:
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
    prompt_text_for_context = prompt if isinstance(prompt, str) else str(prompt[0].get("text", ""))
    working_ctx = context.build_working_context(query=prompt_text_for_context)
    if working_ctx:
        persona += "\n\n" + working_ctx

    # Per-run counters and timing state (closed over by hooks)
    send_count = {"n": 0}
    tool_count: dict[str, int] = {"n": 0}
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
        if tool_name in _SEND_TOOLS:
            send_count["n"] += 1
            if send_count["n"] > effective_max_sends:
                return {"decision": "block", "reason": "Rate limit: too many outbound messages"}
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
        payload = {"tool": tool_name, "success": True}
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
        "Stop": [HookMatcher(hooks=[_build_stop_hook(tool_count, autonomous)])],
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
        model=effective_model,
        fallback_model=fallback,
        system_prompt={"type": "preset", "preset": "claude_code", "append": persona},
        allowed_tools=allowed,
        permission_mode="bypassPermissions",
        setting_sources=["project", "user"],
        mcp_servers={
            "luke": _build_tools(chat_id, bot),
        },
        thinking=thinking if thinking is not None else ThinkingConfigAdaptive(type="adaptive"),
        effort=effort if effort is not None else "high",
        max_turns=max_turns if max_turns is not None else settings.agent_max_turns,
        max_budget_usd=(
            max_budget_usd if max_budget_usd is not None else settings.agent_max_budget_usd
        ),
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
                model="opus",
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
                model="opus",
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
                model="haiku",
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
    return result
