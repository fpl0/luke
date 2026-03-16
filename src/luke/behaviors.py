"""Autonomous behaviors: consolidation, reflection, proactive scan, goal execution."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from aiogram import Bot
from structlog.stdlib import BoundLogger

from . import db
from .agent import AgentResult, run_agent
from .config import settings
from .db import read_memory_body

log: BoundLogger = structlog.get_logger()

_LOG_PREVIEW = 200


async def _run_behavior(
    name: str, prompt: str, bot: Bot, sem: asyncio.Semaphore, **log_fields: Any
) -> AgentResult | None:
    """Run an agent behavior with standard timeout, logging, and error handling."""
    chat_id = settings.chat_id
    if not chat_id:
        return None
    started = time.monotonic()
    try:
        async with sem:
            result = await asyncio.wait_for(
                run_agent(
                    chat_id=chat_id,
                    prompt=prompt,
                    session_id=None,
                    bot=bot,
                    max_turns=settings.behavior_max_turns,
                    max_budget_usd=settings.behavior_max_budget_usd,
                    effort="high",
                ),
                timeout=settings.agent_timeout,
            )
        log.info(
            f"{name}_done",
            responses=len(result.texts),
            response_preview=result.texts[0][:_LOG_PREVIEW] if result.texts else "",
            cost_usd=result.cost_usd,
            turns=result.num_turns,
            duration_s=round(time.monotonic() - started, 1),
            **log_fields,
        )
        db.log_cost(
            chat_id, result.cost_usd, result.num_turns, result.duration_api_ms, f"behavior:{name}"
        )
        return result
    except Exception:
        log.exception(f"{name}_failed")
        return None


async def run_consolidation(bot: Bot, sem: asyncio.Semaphore) -> None:
    """Find episode clusters and consolidate them into insights via agent."""
    if not settings.chat_id:
        return

    clusters = db.get_consolidation_candidates(settings.consolidation_min_cluster)
    if not clusters:
        return

    for cluster in clusters[: settings.max_consolidation_clusters]:
        episode_ids = [ep["id"] for ep in cluster]
        contents: list[str] = []
        for ep in cluster:
            body = read_memory_body("episode", ep["id"], 500)
            if body:
                contents.append(f"[{ep['id']}]: {body}")
        if not contents:
            continue
        prompt = (
            "Memory consolidation task. Review these related episodes and:\n"
            "1. Create an insight that captures the common pattern\n"
            "2. Connect the new insight to relevant entities\n"
            "3. Archive redundant episodes with 'forget'\n\n" + "\n---\n".join(contents)
        )
        await _run_behavior("consolidation", prompt, bot, sem, episodes=episode_ids)


async def run_reflection(bot: Bot, sem: asyncio.Semaphore) -> None:
    """Weekly self-reflection: review recent memories and generate meta-cognitive insights."""
    chat_id = settings.chat_id
    if not chat_id:
        return

    now = datetime.now(UTC)
    week_ago = (now - timedelta(days=7)).isoformat()

    recent = db.recall_by_time_window(after=week_ago, before=now.isoformat())
    if not recent:
        return

    contents: list[str] = []
    for r in recent[:20]:
        body = read_memory_body(r["type"], r["id"], 300)
        if body:
            contents.append(f"[{r['id']}] ({r['type']}): {body}")

    if not contents:
        return

    # Include conversation history for feedback analysis
    msg_lines: list[str] = []
    for m in db.get_recent_messages(chat_id, limit=50):
        msg_lines.append(f"[{m['sender_name']} {m['timestamp']}] {m['content'][:150]}")

    prompt = (
        "Weekly reflection task. Review these recent memories AND conversation "
        "history:\n\n"
        "=== Recent Memories ===\n"
        + "\n---\n".join(contents)
        + "\n\n=== Recent Conversations ===\n"
        + "\n".join(msg_lines[-50:])
        + "\n\nReflect on:\n"
        "1. Which responses did the user react positively to? "
        "(reactions, thanks, follow-through)\n"
        "2. Which responses were corrected or ignored?\n"
        "3. What patterns in user satisfaction do you notice?\n"
        "4. What should you do differently?\n"
        "5. Are there recurring requests that should become procedures or tools?\n"
        "6. Active goals: what progress was made this week?\n\n"
        "Save actionable insights. Be specific about what to change."
    )

    await _run_behavior(
        "reflection",
        prompt,
        bot,
        sem,
        memories_reviewed=len(recent),
        messages_reviewed=len(msg_lines),
    )


async def run_proactive_scan(bot: Bot, sem: asyncio.Semaphore) -> None:
    """Daily proactive scan: check for actionable items and message the user."""
    chat_id = settings.chat_id
    if not chat_id:
        return

    now = datetime.now(UTC)
    week_ago = (now - timedelta(days=7)).isoformat()
    sections: list[str] = []

    # Active goals
    goals = db.recall(mem_type="goal", limit=10)
    if goals:
        goal_lines: list[str] = []
        for g in goals:
            body = read_memory_body(g["type"], g["id"], 300)
            if body:
                goal_lines.append(f"[{g['id']}]: {body}")
        if goal_lines:
            sections.append("Active goals:\n" + "\n---\n".join(goal_lines))

    # Recent insights (type-scoped query avoids fetching all types)
    recent_insights = db.recall(
        mem_type="insight",
        after=week_ago,
        before=now.isoformat(),
        limit=5,
    )
    if recent_insights:
        insight_lines: list[str] = []
        for r in recent_insights:
            body = read_memory_body(r["type"], r["id"], 200)
            if body:
                insight_lines.append(f"[{r['id']}]: {body}")
        if insight_lines:
            sections.append("Recent insights:\n" + "\n---\n".join(insight_lines))

    # Add conversation summaries for anticipatory pattern recognition
    summaries = db.get_message_summaries(chat_id, days=14)
    if summaries:
        summary_lines = [f"{s['date']}:\n" + "\n".join(s["messages"][:10]) for s in summaries]
        sections.append("Recent conversation topics (last 14 days):\n" + "\n\n".join(summary_lines))

    if not sections:
        return

    prompt = (
        "Daily proactive scan. Review your goals, insights, AND recent conversation "
        "patterns below. Decide if the user needs to hear from you:\n"
        "- A goal deadline approaching (within 3 days)\n"
        "- A goal that hasn't been updated in a week\n"
        "- An insight that suggests a timely action\n"
        "- A recurring pattern suggesting the user will need something soon\n"
        "- An unresolved item from a recent conversation\n"
        "- A follow-up you promised but haven't delivered\n\n"
        "If nothing warrants a message, do nothing. Don't message just to check in.\n\n"
        + "\n\n".join(sections)
    )

    await _run_behavior("proactive_scan", prompt, bot, sem, sections=len(sections))


async def run_goal_execution(bot: Bot, sem: asyncio.Semaphore) -> None:
    """Autonomously advance active goals — execute ONE concrete next step per goal."""
    if not settings.chat_id:
        return

    goals = db.recall(mem_type="goal", limit=10)
    if not goals:
        return

    goal_sections: list[str] = []
    for g in goals:
        body = read_memory_body(g["type"], g["id"], 1000)
        if body:
            goal_sections.append(f"[{g['id']}]:\n{body}")

    if not goal_sections:
        return

    prompt = (
        "Goal execution task. You have active goals to work on. "
        "Pick the highest-priority goal and execute ONE concrete next step:\n"
        "- If it requires research, search the web\n"
        "- If it requires code or files, write them\n"
        "- If it requires the user's input and the goal is BLOCKED without it, "
        "message them with a specific question\n"
        "- If it requires scheduling, create reminders or tasks\n"
        "- If you can make progress autonomously, do it\n\n"
        "IMPORTANT: Do NOT send check-in messages, status updates, briefings, "
        "or any messages that belong to scheduled tasks (morning briefing, "
        "midday check-in, evening check-in, reminders). Those are handled by "
        "their own cron jobs. Only message if the goal is truly blocked and "
        "requires input, or you discovered something urgent. When in doubt, "
        "do the work silently and update the goal memory.\n\n"
        "After acting:\n"
        "1. Save an episode documenting what you did and what you learned\n"
        "2. Update the goal memory with new progress and status\n"
        "3. If a follow-up action is needed, schedule it\n\n"
        "Do NOT try to complete the entire goal at once. One step at a time.\n"
        "If no goal needs attention right now, do nothing.\n\n" + "\n\n".join(goal_sections)
    )

    await _run_behavior("goal_execution", prompt, bot, sem, goals_reviewed=len(goals))
