"""Claude Agent SDK integration with in-process MCP tools."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

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
    PreCompactHookInput,
    PreToolUseHookInput,
    ResultMessage,
    StopHookInput,
    ToolAnnotations,
    create_sdk_mcp_server,
    tool,
)
from claude_agent_sdk.types import (
    HookEvent,
    SyncHookJSONOutput,
    ThinkingConfig,
    ThinkingConfigAdaptive,
)
from structlog.stdlib import BoundLogger

from . import db
from .config import settings
from .db import MEMORY_DIRS, read_frontmatter, read_memory_body

log: BoundLogger = structlog.get_logger()

_INTERNAL_RE = re.compile(r"<internal>[\s\S]*?</internal>")
_LOG_TRUNCATION = 100  # chars for log message truncation
_TG_MAX_MSG_LEN = 4096  # Telegram API hard limit


def _trunc(text: str) -> str:
    """Truncate text for logging."""
    return text[:_LOG_TRUNCATION] + "…" if len(text) > _LOG_TRUNCATION else text


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


@dataclass(slots=True)
class AgentResult:
    texts: list[str] = field(default_factory=list)
    session_id: str | None = None
    cost_usd: float = 0.0
    num_turns: int = 0
    duration_api_ms: int = 0
    sent_messages: int = 0  # messages sent via MCP tools during this run


# ---------------------------------------------------------------------------
# MCP tools — built per invocation so they close over group context + bot
# ---------------------------------------------------------------------------


def _build_tools(chat_id: str, bot: Bot) -> Any:
    """Create the in-process MCP server with all 24 tools."""
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
        "Send a text message. Auto-splits if >4096 chars. "
        "Use Telegram HTML (<b>, <i>, <code>, <pre>), NOT markdown.",
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
        "Send a photo from a local file path",
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
        "Send a file/document",
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
        "Send a voice message from a local .ogg file",
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
        "Send a video file",
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
        "Send a GPS location",
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
        "Create a poll in a chat. options: list of strings, e.g. ['Yes', 'No']",
        {"chat_id": str, "question": str, "options": list, "is_anonymous": bool},
        annotations=_OPEN_WORLD,
    )
    async def t_poll(args: dict[str, Any]) -> dict[str, Any]:
        opts = args["options"]
        if isinstance(opts, str):
            # Agent may pass JSON string or newline-separated
            try:
                opts = json.loads(opts)
            except (json.JSONDecodeError, ValueError):
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
        "Send a message with inline keyboard buttons. "
        "buttons: list of rows, each row a list of {text, data} objects. "
        "When pressed, you receive '[Button pressed: data]' as a new message.",
        {"chat_id": str, "text": str, "buttons": list},
        annotations=_OPEN_WORLD,
    )
    async def t_buttons(args: dict[str, Any]) -> dict[str, Any]:
        buttons = args["buttons"]
        if isinstance(buttons, str):
            try:
                buttons = json.loads(buttons)
            except (json.JSONDecodeError, ValueError):
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
        "Reply to a specific message by its msg: ID from the prompt. "
        "Use Telegram HTML formatting.",
        {"chat_id": str, "message_id": int, "text": str},
        annotations=_OPEN_WORLD,
    )
    async def t_reply(args: dict[str, Any]) -> dict[str, Any]:
        await send_long_message(
            bot,
            chat_id=_target(args),
            text=args["text"],
            reply_parameters=ReplyParameters(message_id=args["message_id"]),
        )
        return _ok("Replied")

    @tool(
        "forward",
        "Forward a message to another chat",
        {"from_chat_id": str, "to_chat_id": str, "message_id": int},
        annotations=_OPEN_WORLD,
    )
    async def t_fwd(args: dict[str, Any]) -> dict[str, Any]:
        to_id = _target({**args, "chat_id": args["to_chat_id"]})
        from_id = _target({**args, "chat_id": args["from_chat_id"]})
        await bot.forward_message(
            chat_id=to_id,
            from_chat_id=from_id,
            message_id=args["message_id"],
        )
        return _ok("Forwarded")

    @tool(
        "react",
        "React to a message with an emoji",
        {"chat_id": str, "message_id": int, "emoji": str},
        annotations=_OPEN_WORLD,
    )
    async def t_react(args: dict[str, Any]) -> dict[str, Any]:
        await bot.set_message_reaction(
            chat_id=_target(args),
            message_id=args["message_id"],
            reaction=[ReactionTypeEmoji(emoji=args["emoji"])],
        )
        return _ok("Reacted")

    @tool(
        "edit_message",
        "Edit a previously sent message. Use Telegram HTML formatting.",
        {"chat_id": str, "message_id": int, "text": str},
        annotations=_DESTRUCTIVE,
    )
    async def t_edit(args: dict[str, Any]) -> dict[str, Any]:
        await bot.edit_message_text(
            chat_id=_target(args),
            message_id=args["message_id"],
            text=args["text"],
        )
        return _ok("Edited")

    @tool(
        "delete_message",
        "Delete a message",
        {"chat_id": str, "message_id": int},
        annotations=_DESTRUCTIVE,
    )
    async def t_del(args: dict[str, Any]) -> dict[str, Any]:
        await bot.delete_message(chat_id=_target(args), message_id=args["message_id"])
        return _ok("Deleted")

    @tool(
        "pin",
        "Pin a message in a chat",
        {"chat_id": str, "message_id": int},
        annotations=_OPEN_WORLD,
    )
    async def t_pin(args: dict[str, Any]) -> dict[str, Any]:
        await bot.pin_chat_message(chat_id=_target(args), message_id=args["message_id"])
        return _ok("Pinned")

    # --- Scheduling (1 tool) ---

    @tool(
        "schedule_task",
        "Schedule a recurring or one-time task. schedule_type: 'cron', 'interval', or 'once'. "
        "schedule_value: cron expression, milliseconds, or ISO timestamp.",
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

    # --- Memory (8 tools) ---

    @tool(
        "remember",
        "Save a memory. type: 'entity' | 'episode' | 'procedure' | 'insight' | 'goal'. "
        "importance: 0.1-2.0 (default 1.0). Higher = decays slower. "
        "Returns change summary if updating an existing entity.",
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
        mem_id = re.sub(r"[^\w-]", "_", args["id"]).strip("_-")
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
        fm: dict[str, Any] = {
            "id": mem_id,
            "type": mem_type,
            "tags": tags,
            "created": now if not path.exists() else read_frontmatter(path).get("created", now),
            "updated": now,
            "links": links,
        }
        body = yaml.dump(fm, default_flow_style=False)
        path.write_text(f"---\n{body}---\n\n# {title}\n\n{args['content']}\n")

        # Conflict detection after file write (so changelog is consistent with disk)
        change_note = ""
        if is_update:
            changes = db.detect_changes(mem_id, args["content"], title)
            if changes:
                change_note = " Changes: " + "; ".join(changes)
                db.record_memory_change(mem_id, changes)
        raw_imp = args.get("importance")
        imp: float | None = max(0.1, min(2.0, float(raw_imp))) if raw_imp is not None else None
        await asyncio.to_thread(
            db.index_memory,
            mem_id,
            mem_type,
            title,
            args["content"],
            tags,
            links,
            imp,
        )
        # Linking is a graph-write, not retrieval — track access but not utility
        if links:
            db.touch_memories(links, useful=False)
        # Invalidate recall cache so next message gets fresh results
        from .app import invalidate_recall_cache

        invalidate_recall_cache()
        status = f"Remembered: {mem_id}"
        if change_note:
            status += change_note
        return _ok(status)

    @tool(
        "recall",
        "Search memories. Combine: query, type, after/before, related_to. "
        "Uses hybrid FTS + semantic search with intelligent ranking.",
        {"query": str, "type": str, "after": str, "before": str, "related_to": str},
        annotations=_READ_ONLY,
    )
    async def mem_recall(args: dict[str, Any]) -> dict[str, Any]:
        results = db.recall(
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
        db.touch_memories([r["id"] for r in results])
        return _ok("\n---\n".join(lines))

    @tool(
        "forget",
        "Archive a memory (keeps file, removes from active index)",
        {"id": str},
        annotations=_DESTRUCTIVE,
    )
    async def mem_forget(args: dict[str, Any]) -> dict[str, Any]:
        db.archive_memory(args["id"])
        from .app import invalidate_recall_cache

        invalidate_recall_cache()
        return _ok(f"Archived: {args['id']}")

    @tool(
        "recall_conversation",
        "Retrieve all memories from a time window chronologically. "
        "Useful for 'what happened last Tuesday?' queries.",
        {"after": str, "before": str},
        annotations=_READ_ONLY,
    )
    async def mem_recall_conv(args: dict[str, Any]) -> dict[str, Any]:
        results = db.recall_by_time_window(
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
        db.touch_memories([r["id"] for r in results])
        return _ok("\n---\n".join(lines))

    @tool(
        "connect",
        "Link two memories with a named relationship",
        {"from_id": str, "to_id": str, "relationship": str},
    )
    async def mem_link(args: dict[str, Any]) -> dict[str, Any]:
        created = db.link_memories(args["from_id"], args["to_id"], args["relationship"])
        # Linking is a graph-write, not retrieval — track access but not utility
        db.touch_memories([args["from_id"], args["to_id"]], useful=False)
        rel = args["relationship"]
        if created:
            return _ok(f"Linked: {args['from_id']} —{rel}→ {args['to_id']}")
        return _ok(f"Already linked: {args['from_id']} —{rel}→ {args['to_id']}")

    # --- Restore + Bulk memory (2 tools) ---

    @tool(
        "restore",
        "Restore a previously archived memory back to active status",
        {"id": str},
    )
    async def mem_restore(args: dict[str, Any]) -> dict[str, Any]:
        restored = db.restore_memory(args["id"])
        if restored:
            from .app import invalidate_recall_cache

            invalidate_recall_cache()
            return _ok(f"Restored: {args['id']}")
        return _ok(f"Not found or not archived: {args['id']}")

    @tool(
        "bulk_memory",
        "Bulk operations on memories. action: 'retag' | 'relink' | 'archive'. "
        "ids: list of memory IDs. tags: new tags (for retag). "
        "link_to: target ID, relationship: link name (for relink).",
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
                db.update_memory_tags(mid, tags)
        elif action == "relink":
            link_to = args.get("link_to", "")
            rel = args.get("relationship", "related")
            if not link_to:
                return _ok("Error: link_to required for relink")
            for mid in ids:
                db.link_memories(mid, link_to, rel)
        elif action == "archive":
            for mid in ids:
                db.archive_memory(mid)
        else:
            return _ok(f"Unknown action: {action}")
        from .app import invalidate_recall_cache

        invalidate_recall_cache()
        return _ok(f"{action}: {len(ids)} memories updated")

    # --- Memory history (1 tool) ---

    @tool(
        "memory_history",
        "View change history for a memory (entity updates, content changes over time)",
        {"id": str},
        annotations=_READ_ONLY,
    )
    async def mem_history(args: dict[str, Any]) -> dict[str, Any]:
        history = db.get_memory_history(args["id"])
        if not history:
            return _ok(f"No history for: {args['id']}")
        lines: list[str] = []
        for entry in history:
            changes = "; ".join(entry["changes"])
            lines.append(f"[{entry['timestamp']}] {changes}")
        return _ok("\n".join(lines))

    # --- Cost (1 tool) ---

    @tool(
        "get_cost_report",
        "Get cost and usage statistics. period: 'today' | 'week' | 'month' | 'all'",
        {"period": str},
        annotations=_READ_ONLY,
    )
    async def t_cost(args: dict[str, Any]) -> dict[str, Any]:
        report = db.get_cost_report(args.get("period", "month"))
        return _ok(report)

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
            t_edit,
            t_del,
            t_pin,
            sched,
            mem_save,
            mem_recall,
            mem_recall_conv,
            mem_forget,
            mem_link,
            mem_restore,
            mem_bulk,
            mem_history,
            t_cost,
        ],
    )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


async def _stop_hook(
    input_data: StopHookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    return {
        "systemMessage": (
            "Session ending. Before you stop:\n"
            "1. Did you learn anything new about the user? Save it with 'remember'.\n"
            "2. Did any entity (person, project) change? Update their db.\n"
            "3. Was this conversation significant? Save an episode — include your reasoning: "
            "what approaches you considered, why you chose your solution, what worked or didn't.\n"
            "4. Did you notice a pattern or preference? Save an insight.\n"
            "5. Is there anything pending that needs follow-up? Schedule a reminder.\n"
            "6. Did the user mention wanting to achieve something? Create or update a goal."
        )
    }


async def _pre_compact_hook(
    input_data: PreCompactHookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    log.info("session_compact", trigger=input_data["trigger"])
    return {
        "systemMessage": (
            "CRITICAL — preserve in your compaction summary:\n"
            "1. All memory IDs you've referenced or created (needed for follow-ups)\n"
            "2. The user's most recent request and any pending actions\n"
            "3. Active goals and their current status\n"
            "4. Key facts about the user (preferences, context) from injected memories\n"
            "5. Any tool results not yet communicated to the user\n"
            "6. Relationship links between memories you've established\n"
        )
    }


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


# ---------------------------------------------------------------------------
# Run agent
# ---------------------------------------------------------------------------


async def run_agent(
    *,
    chat_id: str,
    prompt: str | list[dict[str, Any]],
    session_id: str | None,
    bot: Bot,
    max_turns: int | None = None,
    max_budget_usd: float | None = None,
    effort: Literal["low", "medium", "high", "max"] | None = None,
    thinking: ThinkingConfig | None = None,
) -> AgentResult:
    root = settings.luke_dir

    # Load LUKE.md persona (separate from project CLAUDE.md which is dev instructions)
    persona_path = root / "LUKE.md"
    persona = persona_path.read_text() if persona_path.exists() else ""

    # Per-run send rate-limit counter (closed over by PreToolUse hook)
    send_count = {"n": 0}

    async def _pre_tool_hook(
        input_data: PreToolUseHookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        tool_name = input_data["tool_name"]
        log.info("tool_use", tool=tool_name, input=_trunc(str(input_data["tool_input"])))
        if tool_name in _SEND_TOOLS:
            send_count["n"] += 1
            if send_count["n"] > settings.max_sends_per_run:
                return {"decision": "block", "reason": "Rate limit: too many outbound messages"}
        return {}

    hooks: dict[HookEvent, list[HookMatcher]] = {
        "Stop": [HookMatcher(hooks=[cast(HookCallback, _stop_hook)])],
        "PreToolUse": [HookMatcher(hooks=[cast(HookCallback, _pre_tool_hook)])],
        "PreCompact": [HookMatcher(hooks=[cast(HookCallback, _pre_compact_hook)])],
        "Notification": [HookMatcher(hooks=[cast(HookCallback, _notification_hook)])],
    }

    options = ClaudeAgentOptions(
        cwd=str(root),
        resume=session_id,
        model=settings.agent_model,
        fallback_model=settings.agent_fallback_model,
        system_prompt={"type": "preset", "preset": "claude_code", "append": persona},
        allowed_tools=[
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
            "mcp__luke__*",
        ],
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
        async for msg in client.receive_response():
            if isinstance(msg, ResultMessage):
                result.session_id = result.session_id or msg.session_id
                if msg.total_cost_usd is not None:
                    result.cost_usd = msg.total_cost_usd
                result.num_turns = msg.num_turns
                result.duration_api_ms = msg.duration_api_ms
                if msg.usage:
                    log.info(
                        "agent_usage",
                        input=msg.usage.get("input_tokens", 0),
                        output=msg.usage.get("output_tokens", 0),
                        cache_create=msg.usage.get("cache_creation_input_tokens", 0),
                        cache_read=msg.usage.get("cache_read_input_tokens", 0),
                    )
                if msg.result:
                    text = _INTERNAL_RE.sub("", msg.result).strip()
                    if text:
                        result.texts.append(text)

    result.sent_messages = send_count["n"]
    return result
