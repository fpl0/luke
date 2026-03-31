"""Luke — Telegram bot + Claude Agent SDK orchestrator."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import re
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
    ThinkingConfigDisabled,
    ThinkingConfigEnabled,
)
from structlog.stdlib import BoundLogger

from . import db, memory
from .agent import _trunc, run_agent, send_long_message
from .config import settings
from .db import ensure_utc
from .media import build_prompt, extract_frame, transcribe
from .memory import (
    MEMORY_DIRS,
    MemoryResult,
    get_graph_neighbors,
    read_frontmatter,
    recall,
    touch_memories,
)
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

# Model routing: one-way ratchet within a session (never downgrade mid-conversation)
_MODEL_RANK: dict[str, int] = {"haiku": 0, "sonnet": 1, "opus": 2}
_session_models: dict[str, str] = {}  # chat_id → highest model used in session

_session_lost: dict[str, bool] = {}  # chat_id → whether last session was lost


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


_COMPLEX_KEYWORDS: frozenset[str] = frozenset(
    {"research", "analyze", "compare", "build", "create", "plan", "implement", "design"}
)


def _classify_effort(
    prompt: str | list[dict[str, Any]],
) -> tuple[Literal["low", "medium", "high", "max"], ThinkingConfig, str]:
    """Classify message complexity and select model tier dynamically."""
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
        return "low", ThinkingConfigDisabled(type="disabled"), settings.model_low

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
        thinking_cfg = ThinkingConfigEnabled(type="enabled", budget_tokens=16_000)
        return "high", thinking_cfg, settings.model_high

    # Normal: everything else — no thinking to keep latency low
    return "medium", ThinkingConfigDisabled(type="disabled"), settings.model_medium


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


_CONV_STATE_ID = "conversation-state-latest"
_STALE_HOURS = 24.0
_COST_ANOMALY_MIN = 2.0  # minimum cost to trigger anomaly check
_COST_ANOMALY_MULTIPLIER = 3  # times rolling average


def _load_conv_state() -> tuple[str, str | None]:
    """Read conversation state body and timestamp (sync, for use in to_thread)."""
    body = memory.read_memory_body("episode", _CONV_STATE_ID, settings.recall_content_limit)
    updated = memory.get_memory_updated(_CONV_STATE_ID) if body else None
    return body, updated


async def _get_conversation_state(chat_id: str) -> str:
    """Load conversation state for continuity injection.

    Returns formatted context string (empty if nothing to inject).
    Falls back to recent message synthesis if no saved state exists.
    """
    body, updated = await asyncio.to_thread(_load_conv_state)
    if body:
        # Add session recovery notice if applicable
        if _session_lost.pop(chat_id, False):
            body = "[Session was reset — use this context to resume seamlessly]\n" + body
        if updated:
            try:
                ts = ensure_utc(datetime.fromisoformat(updated))
                hours_ago = (datetime.now(UTC) - ts).total_seconds() / 3600
                if hours_ago > _STALE_HOURS:
                    body = (
                        f"[Last conversation was {int(hours_ago)}h ago "
                        f"— context may be outdated]\n{body}"
                    )
            except ValueError:
                pass
        return f"<conversation-state>\n{body}\n</conversation-state>\n"

    # Fallback: synthesize from recent messages
    recent = await asyncio.to_thread(db.get_recent_messages, chat_id, limit=20)
    if not recent:
        return ""
    lines = [f"{m['sender_name']}: {m['content'][:500]}" for m in recent[-10:]]
    ctx = "\n".join(lines)
    return (
        "<conversation-state>\n"
        "[Recent conversation context (no saved state available)]\n"
        f"{ctx}\n</conversation-state>\n"
    )


async def _auto_recall(combined_text: str, chat_id: str) -> tuple[str, list[MemoryResult]]:
    """Run memory recall + graph augmentation. Returns (formatted_context, raw_memories)."""
    recall_start = time.monotonic()
    # Runs in thread — embedding inference is CPU-bound
    memories = await asyncio.to_thread(
        recall,
        query=combined_text,
        limit=settings.auto_recall_limit,
    )
    memory_context = ""
    seen: set[str] = set()
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
        memory_context = await asyncio.to_thread(_format_memory_context, memories)
        # Speculative touch: non-critical bookkeeping, fire-and-forget
        _fire_and_forget(asyncio.to_thread(touch_memories, list(seen), useful=False))
    recall_ms = round((time.monotonic() - recall_start) * 1000)
    all_ids = list(seen)
    log.info(
        "recall_done",
        chat_id=chat_id,
        duration_ms=recall_ms,
        count=len(all_ids),
        ids=all_ids,
    )
    return memory_context, memories


async def process(chat_id: str) -> None:
    """Process all pending messages for a chat."""
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

        # Emit user_message event for event-driven behavior triggers
        word_count = sum(len(m.content.split()) for m in messages)
        db.emit_event("user_message", f'{{"word_count": {word_count}}}')

        # Run build_prompt, memory recall, and conversation state concurrently
        combined_text = " ".join(m.content for m in messages)
        _recalled: list[MemoryResult] = []
        if _needs_recall(combined_text):
            prompt, (memory_context, _recalled), conv_state = await asyncio.gather(
                build_prompt(messages, chat_id),
                _auto_recall(combined_text, chat_id),
                _get_conversation_state(chat_id),
            )
            # Prepend conversation state before recalled memories for priority
            memory_context = conv_state + memory_context
        else:
            prompt = await build_prompt(messages, chat_id)
            memory_context = ""

        # Classify effort and select model tier dynamically
        effort, thinking, routed_model = _classify_effort(prompt)
        # Memory-aware boost: if haiku was selected but substantial context was
        # injected, upgrade to sonnet so the model can process the memories.
        # Only triggers on trivial messages with heavy context — avoids always-fire.
        if routed_model == "haiku" and memory_context and len(memory_context) > 500:
            routed_model = "sonnet"
        # One-way ratchet: never downgrade within a session
        prev_model = _session_models.get(chat_id)
        if prev_model and _MODEL_RANK.get(prev_model, 0) > _MODEL_RANK.get(routed_model, 0):
            model = prev_model
        else:
            model = routed_model

        session_id = db.get_session(chat_id)
        # Non-opus models crash on session resume (SDK bug) — start fresh
        if model != "opus" and session_id:
            session_id = None

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
            model=model,
        )
        agent_start_mono = time.monotonic()
        try:
            async with _sem:
                result = await asyncio.wait_for(
                    run_agent(
                        chat_id=chat_id,
                        prompt=prompt,
                        session_id=session_id,
                        bot=bot,
                        model=model,
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
            # Clear stale session + model ratchet so retries start fresh
            db.set_session(chat_id, "")
            _session_models.pop(chat_id, None)
            count = _retry_counts.get(chat_id, 0) + 1
            _retry_counts[chat_id] = count
            if count >= settings.max_retries:
                log.error("max_retries_exhausted", chat_id=chat_id, retries=count)
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
            model=model,
            effort=effort,
            responses=len(result.texts),
            cost_usd=result.cost_usd,
            turns=result.num_turns,
            duration_s=agent_dur,
        )
        # Single commit for cost log + cursor advance
        with db.batch():
            db.log_cost(
                chat_id, result.cost_usd, result.num_turns, result.duration_api_ms,
                f"message:{model}",
            )
            # Advance cursor only after successful agent run
            db.advance_cursor(chat_id, messages[-1].id)
        _retry_counts.pop(chat_id, None)

        # Upgrade utility for auto-recalled memories the agent actually referenced
        if _recalled:
            recalled_ids = [m["id"] for m in _recalled]
            combined_response = " ".join(result.texts)
            # Single compiled regex to find all referenced memory IDs at once
            pattern = re.compile(r"\b(" + "|".join(re.escape(mid) for mid in recalled_ids) + r")\b")
            found = set(pattern.findall(combined_response))
            referenced = [mid for mid in recalled_ids if mid in found]
            if referenced:
                _fire_and_forget(asyncio.to_thread(touch_memories, referenced, useful_only=True))

        # Session loss detection
        if session_id and result.session_id != session_id:
            log.warning(
                "session_changed",
                chat_id=chat_id,
                old=session_id[:8],
                new=result.session_id[:8] if result.session_id else "none",
            )
            _session_lost[chat_id] = True

        # Cost anomaly detection
        if result.cost_usd > _COST_ANOMALY_MIN:
            avg = db.get_rolling_avg_cost(days=7)
            if avg > 0 and result.cost_usd > avg * _COST_ANOMALY_MULTIPLIER:
                log.warning(
                    "cost_anomaly",
                    chat_id=chat_id,
                    cost=result.cost_usd,
                    rolling_avg=round(avg, 4),
                )

        if result.session_id:
            db.set_session(chat_id, result.session_id)
        # Update model ratchet after successful run
        _session_models[chat_id] = model

        # Only send result text if the agent didn't already message via MCP tools
        if result.sent_messages == 0:
            for text in result.texts:
                await send_long_message(bot, chat_id=int(chat_id), text=text)

        # Save conversation state for continuity (non-trivial conversations only)
        if effort != "low":
            _fire_and_forget(asyncio.to_thread(_save_conv_state, messages, result.texts))


def _extract_topics(messages: list[db.StoredMessage], agent_texts: list[str]) -> list[str]:
    """Extract active topics from conversation using keyword frequency."""
    from collections import Counter

    # Combine all text
    all_text = " ".join(m.content.lower() for m in messages)
    if agent_texts:
        all_text += " " + " ".join(t.lower() for t in agent_texts)
    # Simple stopword filter + word frequency
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "don", "now", "and", "but", "or", "if", "that", "this", "it", "i",
        "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "what", "which", "who",
        "whom", "these", "those", "am", "about", "up", "like", "yeah", "ok",
        "okay", "hey", "hi", "um", "uh", "well", "got", "get", "going",
        "thing", "things", "want", "think", "know", "see", "look", "make",
    }
    words = [w for w in all_text.split() if len(w) > 2 and w.isalpha() and w not in stopwords]
    counter = Counter(words)
    return [word for word, _ in counter.most_common(5) if _ >= 2]


def _extract_pending_actions(agent_texts: list[str]) -> list[str]:
    """Extract pending actions from agent responses."""
    patterns = [
        r"(?:I'll|I will|I'm going to|next step[s]?[:\s]|still need to|working on|TODO:?|remaining:?|will follow up)\s*(.{10,100}?)(?:\.|$|\n)",
    ]
    actions: list[str] = []
    combined = " ".join(agent_texts)
    for pat in patterns:
        for match in re.finditer(pat, combined, re.IGNORECASE):
            action = match.group(1).strip().rstrip(".")
            if action and action not in actions:
                actions.append(action)
            if len(actions) >= 5:
                break
    return actions[:5]


def _save_conv_state(
    messages: list[db.StoredMessage],
    agent_texts: list[str],
) -> None:
    """Save conversation-state-latest memory (sync, runs in to_thread).

    Captures structured context + message history for seamless continuity.
    """
    now = datetime.now(UTC).isoformat(timespec="minutes")
    # Pull recent messages for broader context (not just current batch)
    recent = db.get_recent_messages(settings.chat_id, limit=20) if messages else []

    # Extract structured metadata
    topics = _extract_topics(messages, agent_texts)
    user_msgs = [m for m in messages if m.sender_name != settings.assistant_name]
    last_user_active = user_msgs[-1].timestamp[:16] if user_msgs else "unknown"

    # Build conversation thread: recent history + current exchange
    lines: list[str] = []
    for m in recent[-10:]:
        preview = m["content"][:500]
        lines.append(f"**{m['sender_name']}** ({m['timestamp'][:16]}): {preview}")

    # Add current batch messages if not already in recent
    recent_contents = {r["content"][:50] for r in recent}
    for msg in messages[-5:]:
        if msg.content[:50] not in recent_contents:
            lines.append(f"**{msg.sender_name}** ({msg.timestamp[:16]}): {msg.content[:500]}")

    # Add agent response
    if agent_texts:
        lines.append(f"**Luke** ({now}): {agent_texts[-1][:800]}")

    # Structured header + message thread
    structured = ""
    if topics:
        structured += f"**Active topics:** {', '.join(topics)}\n"
    structured += f"**User last active:** {last_user_active}\n"

    # Extract pending actions from agent responses
    pending = _extract_pending_actions(agent_texts)
    if pending:
        structured += f"**Pending actions:** {' | '.join(pending)}\n"

    body = f"**Last exchange:** {now}\n{structured}" + "\n".join(lines)
    # Write memory file
    import yaml

    type_dir = MEMORY_DIRS["episode"]
    mem_dir = settings.memory_dir / type_dir
    mem_dir.mkdir(parents=True, exist_ok=True)
    path = mem_dir / f"{_CONV_STATE_ID}.md"
    created = read_frontmatter(path).get("created", now) if path.exists() else now
    fm = yaml.dump(
        {
            "id": _CONV_STATE_ID,
            "type": "episode",
            "tags": ["conversation", "state"],
            "created": created,
            "updated": now,
            "links": [],
        },
        default_flow_style=False,
    )
    path.write_text(f"---\n{fm}---\n\n# Conversation State\n\n{body}\n")
    memory.index_memory(
        _CONV_STATE_ID,
        "episode",
        "Last conversation state",
        body,
        tags=["conversation", "state"],
        importance=0.3,
    )


_background_tasks: set[asyncio.Task[None]] = set()


def _on_task_done(task: asyncio.Task[None]) -> None:
    _background_tasks.discard(task)
    if not task.cancelled() and (exc := task.exception()):
        log.error("process() failed", exc_info=exc)


def _fire_and_forget(coro: Awaitable[None]) -> None:
    """Schedule a coroutine as a background task without awaiting it."""
    task: asyncio.Task[None] = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


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
        body = memory.read_memory_body(m["type"], m["id"], settings.recall_content_limit)
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
        "launchctl",
        "kickstart",
        "-k",
        f"gui/{uid}/com.luke",
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
        f"media/photos/photo_{msg.message_id}.jpg",
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
        f"media/voice/voice_{msg.message_id}.ogg",
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
        f"media/documents/{msg.message_id}_{raw_name}",
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
        f"media/video/video_{msg.message_id}.mp4",
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
        f"media/video/videonote_{msg.message_id}.mp4",
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
            f"media/stickers/sticker_{msg.message_id}.webp",
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
        f"media/animations/animation_{msg.message_id}.mp4",
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
        f"media/audio/{msg.message_id}_{raw_name}",
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
    for subdir in MEMORY_DIRS.values():
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
    await asyncio.to_thread(memory.sync_memory_index)
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
            _structlog.processors.format_exc_info,
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
