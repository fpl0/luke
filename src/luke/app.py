"""Luke — Telegram bot + Claude Agent SDK orchestrator."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import shutil
import signal
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import structlog
from aiogram import Bot, Dispatcher, F, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand, ReactionTypeEmoji
from claude_agent_sdk.types import (
    ThinkingConfig,
    ThinkingConfigAdaptive,
    ThinkingConfigDisabled,
    ThinkingConfigEnabled,
)
from structlog.stdlib import BoundLogger

from . import db
from .agent import _trunc, run_agent, send_long_message
from .config import settings
from .db import MemoryResult, get_graph_neighbors, recall, touch_memories
from .media import build_prompt, extract_frame, transcribe
from .scheduler import start_scheduler_loop

log: BoundLogger = structlog.get_logger()


bot = Bot(
    token=settings.telegram_bot_token.get_secret_value(),
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()

_sem = asyncio.Semaphore(settings.max_concurrent)
_active: dict[str, asyncio.Lock] = {}
_notified_unregistered: set[str] = set()
_last_error: dict[str, float] = {}  # chat_id → monotonic timestamp
_retry_counts: dict[str, int] = {}  # chat_id → consecutive failure count

# Recall cache: (memories, formatted_context, monotonic_time, query_hash)
_recall_cache: tuple[list[MemoryResult], str, float, int] | None = None

_TRIVIAL_WORDS = frozenset(
    {
        "ok",
        "okay",
        "yes",
        "no",
        "yeah",
        "yep",
        "nope",
        "sure",
        "thanks",
        "thank",
        "lol",
        "haha",
        "hehe",
        "wow",
        "cool",
        "nice",
        "great",
        "hi",
        "hey",
        "hello",
        "bye",
        "goodnight",
        "gn",
        "gm",
    }
)


def _needs_recall(text: str) -> bool:
    """Heuristic: skip recall for trivial/short messages."""
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    words = stripped.lower().split()
    return not (len(words) <= 2 and all(w.strip("!?.,:") in _TRIVIAL_WORDS for w in words))


def invalidate_recall_cache() -> None:
    """Clear the recall cache."""
    global _recall_cache
    _recall_cache = None


_COMPLEX_KEYWORDS: frozenset[str] = frozenset(
    {"research", "analyze", "compare", "build", "create", "plan", "implement", "design"}
)


def _classify_effort(
    prompt: str | list[dict[str, Any]],
) -> tuple[Literal["low", "medium", "high", "max"], ThinkingConfig]:
    """Classify message complexity to scale thinking effort dynamically."""
    if isinstance(prompt, str):
        text = prompt
        has_media = False
    else:
        text = " ".join(b.get("text", "") for b in prompt)
        has_media = any(b.get("type") == "image" for b in prompt)
    words = text.split()
    word_count = len(words)
    has_question = "?" in text

    # Trivial: short, no questions, no media
    if word_count < 15 and not has_question and not has_media:
        return "low", ThinkingConfigDisabled(type="disabled")

    # Complex: long messages, multiple questions, media, code blocks
    text_lower = text.lower()
    is_complex = (
        word_count > 150
        or text.count("?") > 2
        or has_media
        or "```" in text
        or any(kw in text_lower for kw in _COMPLEX_KEYWORDS)
    )
    if is_complex:
        return "high", ThinkingConfigEnabled(type="enabled", budget_tokens=16_000)

    # Normal: everything else
    return "medium", ThinkingConfigAdaptive(type="adaptive")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


async def _keep_typing(chat_id: int) -> None:
    """Refresh the typing indicator every 4 seconds until cancelled."""
    try:
        while True:
            await asyncio.sleep(4)
            await bot.send_chat_action(chat_id=chat_id, action="typing")
    except asyncio.CancelledError:
        pass
    except Exception:
        pass  # non-critical — silently stop


def _should_send_error(chat_id: str) -> bool:
    """Rate-limit error messages: one per chat per cooldown period."""
    now = time.monotonic()
    last = _last_error.pop(chat_id, 0.0)
    if now - last < settings.error_cooldown:
        _last_error[chat_id] = last  # still within window, restore
        return False
    _last_error[chat_id] = now
    return True


async def process(chat_id: str) -> None:
    """Process all pending messages for a chat."""
    global _recall_cache
    lock = _active.setdefault(chat_id, asyncio.Lock())

    async with lock:
        if settings.chat_id and chat_id != settings.chat_id:
            log.info("unregistered_chat", chat_id=chat_id)
            if chat_id not in _notified_unregistered:
                _notified_unregistered.add(chat_id)
                try:
                    await bot.send_message(
                        chat_id=int(chat_id),
                        text=f"This chat isn't registered with {settings.assistant_name} yet.",
                    )
                except Exception:
                    log.debug("unregistered_notify_failed", chat_id=chat_id)
            return

        messages = db.get_pending_messages(chat_id)
        if not messages:
            return

        prompt: str | list[dict[str, Any]] = await build_prompt(messages, chat_id)

        # Auto memory injection — proactively surface relevant memories
        combined_text = " ".join(m.content for m in messages)
        memory_context = ""

        if _needs_recall(combined_text):
            recall_start = time.monotonic()
            now_mono = recall_start
            cache_hit = False
            query_hash = hash(combined_text)
            if (
                _recall_cache
                and (now_mono - _recall_cache[2]) < settings.auto_recall_cache_ttl
                and _recall_cache[3] == query_hash
            ):
                memory_context = _recall_cache[1]
                cache_hit = True
            else:
                # Runs in thread — embedding inference is CPU-bound
                memories = await asyncio.to_thread(
                    recall,
                    query=combined_text,
                    limit=settings.auto_recall_limit,
                )
                # Graph augmentation: expand 1-hop from recalled memories
                if memories:
                    mem_ids = [m["id"] for m in memories]
                    neighbors = await asyncio.to_thread(
                        get_graph_neighbors,
                        mem_ids,
                        limit=3,
                    )
                    seen = set(mem_ids)
                    for n in neighbors:
                        if n["id"] not in seen:
                            memories.append(n)
                            seen.add(n["id"])
                    memory_context = _format_memory_context(memories)
                    # Speculative touch: track auto-injection access
                    await asyncio.to_thread(touch_memories, list(seen), useful=False)
                _recall_cache = (memories, memory_context, now_mono, query_hash)
            recall_ms = round((time.monotonic() - recall_start) * 1000)
            recalled_ids = [m["id"] for m in (_recall_cache[0] if _recall_cache else [])]
            log.info(
                "recall_done",
                chat_id=chat_id,
                cache_hit=cache_hit,
                duration_ms=recall_ms,
                count=len(recalled_ids),
                ids=recalled_ids,
            )

        # Classify effort BEFORE memory injection so injected context
        # doesn't inflate word count and skew the complexity heuristic.
        effort, thinking = _classify_effort(prompt)

        if memory_context:
            log.info("memories_injected", chat=chat_id)
            if isinstance(prompt, list):
                prompt.insert(0, {"type": "text", "text": f"{memory_context}\n\n"})
            else:
                prompt = f"{memory_context}\n\n{prompt}"

        await bot.send_chat_action(chat_id=int(chat_id), action="typing")
        typing_task = asyncio.create_task(_keep_typing(int(chat_id)))

        prompt_chars = (
            len(prompt) if isinstance(prompt, str) else sum(len(b.get("text", "")) for b in prompt)
        )
        log.info(
            "agent_start",
            chat_id=chat_id,
            messages=len(messages),
            prompt_chars=prompt_chars,
            effort=effort,
        )
        agent_start_mono = time.monotonic()
        try:
            async with _sem:
                result = await asyncio.wait_for(
                    run_agent(
                        chat_id=chat_id,
                        prompt=prompt,
                        session_id=db.get_session(chat_id),
                        bot=bot,
                        effort=effort,
                        thinking=thinking,
                    ),
                    timeout=settings.agent_timeout,
                )
        except Exception as exc:
            is_timeout = isinstance(exc, TimeoutError)
            if is_timeout:
                log.error("agent_timeout", chat_id=chat_id, timeout=settings.agent_timeout)
            else:
                log.exception("agent_failed", chat_id=chat_id)
            # Clear stale session so retries start fresh
            db.set_session(chat_id, "")
            count = _retry_counts.get(chat_id, 0) + 1
            _retry_counts[chat_id] = count
            if count >= settings.max_retries:
                log.error("max_retries_exhausted", chat_id=chat_id, retries=count)
                db.set_session(chat_id, "")
                db.advance_cursor(chat_id, messages[-1].id)
                _retry_counts.pop(chat_id, None)
                if _should_send_error(chat_id):
                    msg_text = (
                        "Your messages were skipped after repeated timeouts."
                        if is_timeout
                        else f"Your messages were skipped after {count} failures "
                        f"({type(exc).__name__})."
                    )
                    await bot.send_message(chat_id=int(chat_id), text=msg_text)
            else:
                log.warning("agent_retry_pending", chat_id=chat_id, attempt=count)
                # Exponential backoff: 30s, 60s, 120s, ...
                delay = 30.0 * (2 ** (count - 1))
                asyncio.get_running_loop().call_later(delay, _dispatch, chat_id)
            return
        finally:
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task

        agent_dur = round(time.monotonic() - agent_start_mono, 1)
        log.info(
            "agent_done",
            chat_id=chat_id,
            responses=len(result.texts),
            cost_usd=result.cost_usd,
            turns=result.num_turns,
            duration_s=agent_dur,
        )
        # Single commit for cost log + cursor advance
        with db.batch():
            db.log_cost(
                chat_id, result.cost_usd, result.num_turns, result.duration_api_ms, "message"
            )
            # Advance cursor only after successful agent run
            db.advance_cursor(chat_id, messages[-1].id)
        _retry_counts.pop(chat_id, None)

        if result.session_id:
            db.set_session(chat_id, result.session_id)

        # Only send result text if the agent didn't already message via MCP tools
        if result.sent_messages == 0:
            for text in result.texts:
                await send_long_message(bot, chat_id=int(chat_id), text=text)


_background_tasks: set[asyncio.Task[None]] = set()


def _on_task_done(task: asyncio.Task[None]) -> None:
    _background_tasks.discard(task)
    if not task.cancelled() and (exc := task.exception()):
        log.error("process() failed", exc_info=exc)


def _dispatch(chat_id: str) -> None:
    """Fire-and-forget message processing."""
    task = asyncio.create_task(process(chat_id))
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reply_to(msg: types.Message) -> str | None:
    return str(msg.reply_to_message.message_id) if msg.reply_to_message else None


def _store(
    msg: types.Message,
    content: str,
    *,
    reply_to: str | None = None,
    media_file_id: str | None = None,
    timestamp: str | None = None,
) -> bool:
    """Store a message. Returns False if duplicate detected."""
    if not msg.from_user:
        return False
    log.info(
        "msg_in",
        chat=msg.chat.id,
        sender=msg.from_user.full_name,
        text=_trunc(content),
    )
    stored = db.store_message(
        chat_id=str(msg.chat.id),
        sender_name=msg.from_user.full_name,
        sender_id=str(msg.from_user.id),
        message_id=msg.message_id,
        content=content,
        timestamp=timestamp or msg.date.isoformat(),
        reply_to=reply_to,
        media_file_id=media_file_id,
    )
    if not stored:
        log.debug("duplicate_message", chat=msg.chat.id, msg_id=msg.message_id)
    return stored


async def _handle_media(
    msg: types.Message,
    media: Any,
    filename: str,
    content: str,
    *,
    reply_to: str | None = None,
    media_file_id: str | None = None,
    post_download: Callable[[Path], Awaitable[str]] | None = None,
) -> None:
    """Download media to workspace, store message, and dispatch for processing."""
    dest = settings.workspace_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        await bot.download(media, dest)
    except Exception:
        log.exception("media_download_failed", filename=filename)
        _store(
            msg,
            content.format(dest=dest) + "\n[Download failed]",
            reply_to=reply_to,
            media_file_id=media_file_id,
        )
        _dispatch(str(msg.chat.id))
        return
    extra = await post_download(dest) if post_download else ""
    _store(
        msg,
        content.format(dest=dest) + extra,
        reply_to=reply_to,
        media_file_id=media_file_id,
    )
    _dispatch(str(msg.chat.id))


async def _video_thumbnail(dest: Path) -> str:
    """Extract a thumbnail frame from a video file. Returns content suffix."""
    thumb = dest.with_suffix(".jpg")
    if await extract_frame(dest, thumb):
        return f"\n[Video thumbnail saved: {thumb}]"
    return ""


async def _animation_frame(dest: Path) -> str:
    """Extract first frame from an animation. Returns content suffix."""
    frame = dest.with_suffix(".jpg")
    if await extract_frame(dest, frame):
        return f"\n[Animation frame saved: {frame}]"
    return ""


async def _transcribe_post(dest: Path) -> str:
    """Transcribe audio after download. Returns content suffix."""
    transcript = await transcribe(dest)
    if transcript:
        log.info("transcription", file=dest.name, text=transcript)
        return f"\n[Audio transcript]: {transcript}"
    log.warning("transcription_failed", file=dest.name)
    return "\n[Voice transcription failed]"


def _format_memory_context(memories: list[MemoryResult]) -> str:
    """Format recalled memories as context prefix for the agent."""
    lines: list[str] = []
    for m in memories:
        body = db.read_memory_body(m["type"], m["id"], settings.recall_content_limit)
        content = body or m.get("title", "")
        lines.append(f"[{m['id']}] ({m['type']}) {content}")
    body = "\n---\n".join(lines)
    return f"<context><memories>\n{body}\n</memories></context>"


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------


@dp.message(F.text.startswith("/restart"))
async def on_restart(msg: types.Message) -> None:
    if not msg.from_user:
        return
    chat_id = str(msg.chat.id)
    if settings.chat_id and chat_id != settings.chat_id:
        return
    log.info("restart_requested", chat_id=chat_id, user=msg.from_user.full_name)
    await bot.send_message(int(chat_id), "Restarting... 🔄")
    # Use launchctl kickstart to bypass launchd's throttle interval
    uid = os.getuid()
    await asyncio.create_subprocess_exec(
        "launchctl", "kickstart", "-k", f"gui/{uid}/com.luke",
    )


@dp.message(F.text.startswith("/start"))
async def on_start(msg: types.Message) -> None:
    if not msg.from_user:
        return
    chat_id = str(msg.chat.id)
    if not settings.chat_id or chat_id == settings.chat_id:
        _store(msg, msg.text or "/start", reply_to=_reply_to(msg))
        _dispatch(chat_id)
    else:
        log.info("unregistered_start", chat_id=chat_id, user=msg.from_user.full_name)
        await bot.send_message(
            chat_id=int(chat_id),
            text=(
                f"Hi {msg.from_user.first_name}! I'm {settings.assistant_name}, "
                f"but I'm not set up for this chat yet.\n\n"
                f"Chat ID: <code>{chat_id}</code>"
            ),
        )


@dp.message(F.text)
async def on_text(msg: types.Message) -> None:
    if not msg.from_user:
        return
    if _store(msg, msg.text or "", reply_to=_reply_to(msg)):
        _dispatch(str(msg.chat.id))


@dp.message(F.photo)
async def on_photo(msg: types.Message) -> None:
    if not msg.from_user or not msg.photo:
        return
    await _handle_media(
        msg,
        msg.photo[-1],
        f"photo_{msg.message_id}.jpg",
        f"{msg.caption or ''}\n[Photo saved: {{dest}}]",
        reply_to=_reply_to(msg),
    )


@dp.message(F.voice)
async def on_voice(msg: types.Message) -> None:
    if not msg.from_user or not msg.voice:
        return

    await _handle_media(
        msg,
        msg.voice,
        f"voice_{msg.message_id}.ogg",
        "[Voice message saved: {dest}]",
        reply_to=_reply_to(msg),
        media_file_id=msg.voice.file_id,
        post_download=_transcribe_post,
    )


@dp.message(F.document)
async def on_document(msg: types.Message) -> None:
    if not msg.from_user or not msg.document:
        return
    raw_name = Path(msg.document.file_name or f"doc_{msg.message_id}").name
    await _handle_media(
        msg,
        msg.document,
        f"{msg.message_id}_{raw_name}",
        f"{msg.caption or ''}\n[Document saved: {{dest}}]",
        reply_to=_reply_to(msg),
    )


@dp.message(F.video)
async def on_video(msg: types.Message) -> None:
    if not msg.from_user or not msg.video:
        return
    await _handle_media(
        msg,
        msg.video,
        f"video_{msg.message_id}.mp4",
        f"{msg.caption or ''}\n[Video saved: {{dest}}]",
        reply_to=_reply_to(msg),
        media_file_id=msg.video.file_id,
        post_download=_video_thumbnail,
    )


@dp.message(F.video_note)
async def on_video_note(msg: types.Message) -> None:
    if not msg.from_user or not msg.video_note:
        return
    await _handle_media(
        msg,
        msg.video_note,
        f"videonote_{msg.message_id}.mp4",
        "[Video note saved: {dest}]",
        reply_to=_reply_to(msg),
        media_file_id=msg.video_note.file_id,
        post_download=_video_thumbnail,
    )


@dp.message(F.sticker)
async def on_sticker(msg: types.Message) -> None:
    if not msg.from_user or not msg.sticker:
        return
    emoji = msg.sticker.emoji or ""
    # Download static stickers (WebP) for vision; skip animated (TGS) and video (WebM)
    if not msg.sticker.is_animated and not msg.sticker.is_video:
        await _handle_media(
            msg,
            msg.sticker,
            f"sticker_{msg.message_id}.webp",
            f"[Sticker: {emoji}]\n[Sticker image saved: {{dest}}]",
        )
        return
    if _store(msg, f"[Sticker: {emoji}]"):
        _dispatch(str(msg.chat.id))


@dp.message(F.animation)
async def on_animation(msg: types.Message) -> None:
    if not msg.from_user or not msg.animation:
        return
    await _handle_media(
        msg,
        msg.animation,
        f"animation_{msg.message_id}.mp4",
        f"{msg.caption or '[GIF/Animation]'}\n[Animation saved: {{dest}}]",
        reply_to=_reply_to(msg),
        post_download=_animation_frame,
    )


@dp.message(F.audio)
async def on_audio(msg: types.Message) -> None:
    if not msg.from_user or not msg.audio:
        return
    raw_name = Path(msg.audio.file_name or f"audio_{msg.message_id}").name
    await _handle_media(
        msg,
        msg.audio,
        f"{msg.message_id}_{raw_name}",
        f"{msg.caption or ''}\n[Audio saved: {{dest}}]",
        reply_to=_reply_to(msg),
        post_download=_transcribe_post,
    )


@dp.message(F.location)
async def on_location(msg: types.Message) -> None:
    if not msg.from_user or not msg.location:
        return
    if _store(msg, f"[Location: {msg.location.latitude}, {msg.location.longitude}]"):
        _dispatch(str(msg.chat.id))


@dp.message(F.contact)
async def on_contact(msg: types.Message) -> None:
    if not msg.from_user or not msg.contact:
        return
    c = msg.contact
    if _store(msg, f"[Contact: {c.first_name} {c.last_name or ''}, {c.phone_number}]"):
        _dispatch(str(msg.chat.id))


@dp.message(F.poll)
async def on_poll(msg: types.Message) -> None:
    if not msg.from_user or not msg.poll:
        return
    options = ", ".join(o.text for o in msg.poll.options)
    if _store(msg, f"[Poll: {msg.poll.question} — Options: {options}]"):
        _dispatch(str(msg.chat.id))


@dp.message_reaction()
async def on_reaction(event: types.MessageReactionUpdated) -> None:
    if not event.user or not event.new_reaction:
        return
    for r in event.new_reaction:
        if isinstance(r, ReactionTypeEmoji):
            db.store_reaction_feedback(
                chat_id=str(event.chat.id),
                msg_id=event.message_id,
                sender_id=str(event.user.id),
                emoji=r.emoji,
                timestamp=event.date.isoformat(),
            )
    emojis = " ".join(r.emoji for r in event.new_reaction if isinstance(r, ReactionTypeEmoji))
    if emojis:
        log.info(
            "reaction_tracked",
            chat=event.chat.id,
            sender=event.user.full_name,
            emojis=emojis,
            msg_id=event.message_id,
        )


@dp.edited_message()
async def on_edit(msg: types.Message) -> None:
    if not msg.from_user:
        return
    edit_dt = msg.edit_date if isinstance(msg.edit_date, datetime) else None
    if _store(
        msg,
        f"[Edited message {msg.message_id}]: {msg.text or msg.caption or ''}",
        timestamp=(edit_dt or msg.date).isoformat(),
    ):
        _dispatch(str(msg.chat.id))


@dp.callback_query()
async def on_callback(cb: types.CallbackQuery) -> None:
    await cb.answer()
    if cb.data and cb.message and isinstance(cb.message, types.Message):
        sender_name = cb.from_user.full_name if cb.from_user else "Unknown"
        sender_id = str(cb.from_user.id) if cb.from_user else ""
        log.info("msg_in", chat=cb.message.chat.id, sender=sender_name, text=f"[Button: {cb.data}]")
        stored = db.store_message(
            chat_id=str(cb.message.chat.id),
            sender_name=sender_name,
            sender_id=sender_id,
            message_id=cb.message.message_id,
            content=f"[Button pressed: {cb.data}]",
            timestamp=datetime.now(UTC).isoformat(),
        )
        if stored:
            _dispatch(str(cb.message.chat.id))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _notify_main(text: str) -> None:
    """Send a message to the main chat (best-effort)."""
    if not settings.chat_id:
        return
    try:
        await bot.send_message(int(settings.chat_id), text)
    except Exception:
        log.debug("notify_main_failed", chat_id=settings.chat_id)


def _ensure_dirs() -> None:
    """Create Luke's directory tree and seed templates if absent."""
    settings.luke_dir.mkdir(parents=True, exist_ok=True)
    settings.workspace_dir.mkdir(exist_ok=True)
    settings.memory_dir.mkdir(exist_ok=True)
    for subdir in db.MEMORY_DIRS.values():
        (settings.memory_dir / subdir).mkdir(exist_ok=True)
    # Seed LUKE.md persona from package template if not present
    persona_dest = settings.luke_dir / "LUKE.md"
    if not persona_dest.exists():
        template = Path(__file__).parent / "templates" / "LUKE.md"
        if template.exists():
            shutil.copy2(template, persona_dest)


async def main() -> None:
    log.info("starting", phase="init")
    _ensure_dirs()
    db.init()
    db.clear_sessions()  # no session survives process restart
    log.info("starting", phase="memory_sync")
    await asyncio.to_thread(db.sync_memory_index)
    log.info("started", chat_id=settings.chat_id)

    await bot.set_my_commands(
        [
            BotCommand(command="cost", description="Show cost & usage stats"),
            BotCommand(command="tasks", description="List scheduled tasks"),
            BotCommand(command="recall", description="Search memories"),
            BotCommand(command="restart", description="Restart Luke"),
        ]
    )

    await _notify_main("Back online.")

    # Replay pending messages from before restart
    if settings.chat_id:
        pending = db.get_pending_messages(settings.chat_id)
        if pending:
            log.info("startup_replay", chat_id=settings.chat_id, count=len(pending))
            _dispatch(settings.chat_id)

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _on_signal() -> None:
        if not shutdown_event.is_set():
            shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _on_signal)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(dp.start_polling(bot))
        tg.create_task(start_scheduler_loop(bot, _sem, shutdown=shutdown_event))

        async def _wait_for_shutdown() -> None:
            await shutdown_event.wait()
            log.info("stopping", phase="dispatcher")
            await dp.stop_polling()
            pending = set(_background_tasks)
            if pending:
                log.info("stopping", phase="drain_tasks", count=len(pending))
                await asyncio.gather(*pending, return_exceptions=True)
            log.info("stopping", phase="notify")
            await _notify_main("Going offline.")
            log.info("stopped")

        tg.create_task(_wait_for_shutdown())


def _configure_logging() -> None:
    """Configure structlog with timestamped JSON output."""
    import structlog as _structlog

    _structlog.configure(
        processors=[
            _structlog.contextvars.merge_contextvars,
            _structlog.processors.add_log_level,
            _structlog.processors.StackInfoRenderer(),
            _structlog.dev.set_exc_info,
            _structlog.processors.TimeStamper(fmt="iso"),
            _structlog.processors.JSONRenderer(),
        ],
        wrapper_class=_structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=_structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


_lock_fd: int | None = None


def _try_flock(fd: int) -> bool:
    """Try to acquire exclusive flock. Returns True on success."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _acquire_lock() -> None:
    """Acquire an exclusive lock, killing stale holders if needed.

    Uses flock for the lock itself. On failure, reads the PID from the lock
    file, checks if it's alive, and kills it if it's a stale Luke process.
    This handles orphan processes that survive outside launchd's control.
    """
    global _lock_fd
    lock_path = settings.store_dir / "luke.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)

    # Fast path: lock available
    if _try_flock(_lock_fd):
        os.ftruncate(_lock_fd, 0)
        os.lseek(_lock_fd, 0, os.SEEK_SET)
        os.write(_lock_fd, str(os.getpid()).encode())
        return

    # Lock held — read PID and check if holder is alive
    os.lseek(_lock_fd, 0, os.SEEK_SET)
    pid_str = os.read(_lock_fd, 32).decode().strip()
    if pid_str.isdigit():
        old_pid = int(pid_str)
        try:
            os.kill(old_pid, signal.SIGTERM)
            print(f"Killed stale Luke process {old_pid}", file=sys.stderr)
            time.sleep(2)
            # Force-kill if still alive
            with contextlib.suppress(ProcessLookupError):
                os.kill(old_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # already dead

    # Retry after killing
    time.sleep(1)
    if _try_flock(_lock_fd):
        os.ftruncate(_lock_fd, 0)
        os.lseek(_lock_fd, 0, os.SEEK_SET)
        os.write(_lock_fd, str(os.getpid()).encode())
        return

    print("Another Luke process is already running and could not be killed.", file=sys.stderr)
    raise SystemExit(1)


def cli() -> None:
    """Entry point for `pyproject.toml` [project.scripts]."""
    _configure_logging()
    _acquire_lock()
    asyncio.run(main())
