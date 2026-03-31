"""Autonomous behaviors: consolidation, reflection, proactive scan, goal execution."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from aiogram import Bot
from structlog.stdlib import BoundLogger

from . import db, memory
from .agent import AgentResult, run_agent
from .config import settings
from .memory import read_memory_body, sanitize_memory_id

log: BoundLogger = structlog.get_logger()

_LOG_PREVIEW = 200


async def _run_behavior(
    name: str,
    prompt: str,
    bot: Bot,
    sem: asyncio.Semaphore,
    *,
    model: str | None = None,
    max_turns: int | None = None,
    max_budget_usd: float | None = None,
    max_sends: int | None = None,
    **log_fields: Any,
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
                    model=model,
                    max_turns=max_turns if max_turns is not None else settings.behavior_max_turns,
                    max_budget_usd=(
                        max_budget_usd
                        if max_budget_usd is not None
                        else settings.behavior_max_budget_usd
                    ),
                    max_sends=max_sends,
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
        effective_model = model or settings.agent_model
        db.log_cost(
            chat_id, result.cost_usd, result.num_turns, result.duration_api_ms,
            f"behavior:{name}:{effective_model}",
        )
        return result
    except Exception:
        log.exception(f"{name}_failed")
        return None


async def run_consolidation(bot: Bot, sem: asyncio.Semaphore) -> None:
    """Find episode clusters and consolidate them into insights via agent."""
    if not settings.chat_id:
        return

    clusters = memory.get_consolidation_candidates(settings.consolidation_min_cluster)
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
        await _run_behavior(
            "consolidation",
            prompt,
            bot,
            sem,
            model=settings.consolidation_model,
            max_sends=0,
            episodes=episode_ids,
        )


async def run_reflection(bot: Bot, sem: asyncio.Semaphore) -> None:
    """Weekly self-reflection: review recent memories and generate meta-cognitive insights."""
    chat_id = settings.chat_id
    if not chat_id:
        return

    now = datetime.now(UTC)
    week_ago = (now - timedelta(days=7)).isoformat()

    recent = memory.recall_by_time_window(after=week_ago, before=now.isoformat())
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

    # Include actual reaction data from the database
    reaction_lines: list[str] = []
    for r in db.get_reactions(chat_id, limit=50):
        reaction_lines.append(
            f"  msg:{r['msg_id']} {r['emoji']} ({r['sentiment']}) "
            f"from {r['sender_id']} at {r['timestamp']}"
        )

    prompt = (
        "Weekly reflection task. Review these recent memories, conversation "
        "history, AND user reactions:\n\n"
        "=== Recent Memories ===\n"
        + "\n---\n".join(contents)
        + "\n\n=== Recent Conversations ===\n"
        + "\n".join(msg_lines[-50:])
        + "\n\n=== User Reactions (emoji feedback on your messages) ===\n"
        + ("\n".join(reaction_lines) if reaction_lines else "(No reactions this period)")
        + "\n\nReflect on:\n"
        "1. Which of your messages got positive reactions? What made them work?\n"
        "2. Which messages got negative reactions or were corrected/ignored?\n"
        "3. What patterns in user satisfaction do you notice?\n"
        "4. What should you do differently — and save specific feedback insights?\n"
        "5. Are there recurring requests that should become procedures or tools?\n"
        "6. Active goals: what progress was made this week?\n\n"
        "Save actionable insights with 'remember'. Be specific about what to change.\n"
        "Do NOT message the user — this is internal reflection only."
    )

    await _run_behavior(
        "reflection",
        prompt,
        bot,
        sem,
        model=settings.reflection_model,
        max_sends=0,
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
    goals = memory.recall(mem_type="goal", limit=10)
    if goals:
        goal_lines: list[str] = []
        for g in goals:
            body = read_memory_body(g["type"], g["id"], 300)
            if body:
                goal_lines.append(f"[{g['id']}]: {body}")
        if goal_lines:
            sections.append("Active goals:\n" + "\n---\n".join(goal_lines))

    # Recent insights (type-scoped query avoids fetching all types)
    recent_insights = memory.recall(
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
        "If action is needed and you can do it autonomously (research, schedule, write):\n"
        "- Take ONE concrete step\n"
        "- Save an episode of what you did\n"
        "- Update the goal memory with progress\n\n"
        "Reserve messaging the user for things that genuinely need their input.\n"
        "If nothing warrants action or a message, do nothing.\n\n" + "\n\n".join(sections)
    )

    await _run_behavior(
        "proactive_scan",
        prompt,
        bot,
        sem,
        model=settings.proactive_scan_model,
        max_sends=2,
        sections=len(sections),
    )


async def run_deep_work(bot: Bot, sem: asyncio.Semaphore) -> None:
    """Autonomous goal loop: sustained multi-step work on active goals."""
    if not settings.chat_id:
        return

    # Budget gate: skip if daily autonomous spend is exhausted
    daily_cost = db.get_daily_deep_work_cost()
    if daily_cost >= settings.daily_deep_work_budget_usd:
        log.info("deep_work_budget_exhausted", daily_cost=daily_cost)
        return

    goals = memory.recall(mem_type="goal", limit=10)
    if not goals:
        return

    goal_sections: list[str] = []
    for g in goals:
        body = read_memory_body(g["type"], g["id"], 1000)
        if body:
            goal_sections.append(f"[{g['id']}]:\n{body}")

    if not goal_sections:
        return

    # Check for existing work plans
    plans_dir = settings.workspace_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_status: list[str] = []
    for g in goals:
        plan_path = plans_dir / f"{sanitize_memory_id(g['id'])}.md"
        if plan_path.exists():
            plan_status.append(f"[{g['id']}]: plan exists at workspace/plans/{plan_path.name}")

    remaining = settings.daily_deep_work_budget_usd - daily_cost
    now_str = datetime.now(UTC).isoformat(timespec="minutes")

    prompt = (
        f"Deep work session. Current time: {now_str}. "
        f"Daily budget remaining: ${remaining:.2f}.\n\n"
        "Active goals (priority order):\n"
        + "\n\n".join(goal_sections)
        + ("\n\nExisting work plans:\n" + "\n".join(plan_status) if plan_status else "")
        + "\n\n"
        "Phase 1 — PLAN (do this first):\n"
        "1. Pick the highest-priority goal\n"
        "2. Check if a work plan exists at workspace/plans/{goal_id}.md\n"
        "3. If no plan exists, create one: 3-5 concrete steps, status: in_progress\n"
        "4. If a plan exists and status is in_progress, pick up where you left off\n"
        "5. If a plan exists and status is blocked, check if the blocker is resolved\n\n"
        "Phase 2 — EXECUTE:\n"
        "6. Work through unchecked steps in order\n"
        "7. After each step, self-check: 'Am I making progress or going in circles?'\n"
        "   - If stuck for 2+ attempts on the same step, mark the plan as blocked\n"
        "8. After each meaningful step, update the plan file:\n"
        "   - Check off completed step\n"
        "   - Add progress note with timestamp\n"
        "   - Update steps_completed count and last_updated\n"
        "   - Update the goal memory with new progress %\n\n"
        "Phase 3 — QUALITY CHECK (mandatory before wrap-up):\n"
        "9. Review what you produced this session. Ask yourself:\n"
        "   - Would the user actually want or use this output?\n"
        "   - Is this advancing the goal meaningfully, or is it busywork?\n"
        "   - Rate this session 1-5: 1=wasted, 3=okay, 5=substantial progress\n"
        "   Add the rating to your plan file under '## Session Quality'\n"
        "   If the last 3 sessions average below 2: pause the goal and note why.\n\n"
        "Phase 4 — WRAP UP:\n"
        "10. If the goal is complete: update plan status to completed, update goal memory\n"
        "11. If blocked: update plan status to blocked, note what you need\n"
        "12. Save a summary episode (deep-work-log) of what was accomplished\n\n"
        "You may send at most ONE message to the user per session — only if truly blocked.\n"
        "Prefer updating the plan's Blockers section over messaging.\n"
        "You have full tool access: web search, code, files, memory.\n"
    )

    await _run_behavior(
        "deep_work",
        prompt,
        bot,
        sem,
        max_turns=settings.deep_work_max_turns,
        max_budget_usd=settings.deep_work_max_budget_usd,
        max_sends=1,
        goals_reviewed=len(goals),
    )


async def run_feedback_consolidation(bot: Bot, sem: asyncio.Semaphore) -> None:
    """Consolidate feedback-* insights into a structured user preferences entity."""
    if not settings.chat_id:
        return

    feedback_ids = memory.get_feedback_insight_ids()
    if len(feedback_ids) < 5:
        return

    contents: list[str] = []
    for fid in feedback_ids:
        body = read_memory_body("insight", fid, 500)
        if body:
            contents.append(f"[{fid}]: {body}")

    if not contents:
        return

    prompt = (
        "Feedback consolidation task. These feedback insights capture user preferences "
        "as individual fragments. Synthesize into a SINGLE structured entity called "
        "'user-preferences' organized by category:\n"
        "- Communication (style, format, frequency)\n"
        "- Deep work (autonomy, scheduling, process)\n"
        "- Content (quality, testing, conventions)\n"
        "- Process (approval workflow, transparency)\n\n"
        "Steps:\n"
        "1. Create the 'user-preferences' entity with all preferences consolidated\n"
        "2. Create 'derived_from' links from the new entity to each original insight\n"
        "3. Archive the individual feedback insights with 'forget'\n"
        "4. Keep only genuinely unique insights that don't fit into categories\n\n"
        + "\n---\n".join(contents)
    )

    await _run_behavior(
        "feedback_consolidation",
        prompt,
        bot,
        sem,
        max_sends=0,
        feedback_count=len(feedback_ids),
    )


async def run_insight_consolidation(bot: Bot, sem: asyncio.Semaphore) -> None:
    """Consolidate overlapping non-feedback insights into authoritative summaries."""
    if not settings.chat_id:
        return

    clusters = memory.get_insight_clusters()
    if not clusters:
        return

    for cluster in clusters[: settings.max_consolidation_clusters]:
        insight_ids = [m["id"] for m in cluster]
        contents: list[str] = []
        for m in cluster:
            body = read_memory_body("insight", m["id"], 500)
            if body:
                contents.append(f"[{m['id']}]: {body}")
        if not contents:
            continue

        prompt = (
            "Insight consolidation task. These insights overlap semantically:\n\n"
            + "\n---\n".join(contents)
            + "\n\nSteps:\n"
            "1. Synthesize into ONE authoritative insight that captures all information\n"
            "2. Create 'derived_from' links from the new insight to each original\n"
            "3. Archive the individual fragments with 'forget'\n"
            "4. Connect the new insight to relevant entities\n"
        )

        await _run_behavior(
            "insight_consolidation",
            prompt,
            bot,
            sem,
            max_sends=0,
            insights=insight_ids,
        )


async def run_lifecycle_review(bot: Bot, sem: asyncio.Semaphore) -> None:
    """Monthly review: flag stale entities, unused procedures, lingering goals."""
    if not settings.chat_id:
        return

    candidates = memory.get_lifecycle_candidates()
    sections: list[str] = []

    if candidates["stale_entities"]:
        items = []
        for e in candidates["stale_entities"]:
            mentions = e.get("recent_mentions", 0)
            note = f" ({mentions} recent episode mentions)" if mentions else ""
            items.append(f"  - {e['id']}: last updated {e['updated']}{note}")
        sections.append("Stale entities (not updated in 90+ days):\n" + "\n".join(items))

    if candidates["unused_procedures"]:
        items = [
            f"  - {p['id']}: last accessed {p['last_accessed'] or 'never'}"
            for p in candidates["unused_procedures"]
        ]
        sections.append("Unused procedures (60+ days):\n" + "\n".join(items))

    if candidates["lingering_goals"]:
        items = [f"  - {g['id']}" for g in candidates["lingering_goals"]]
        sections.append("Completed goals still active:\n" + "\n".join(items))

    if not sections:
        return

    prompt = (
        "Monthly lifecycle review. These memories need attention:\n\n"
        + "\n\n".join(sections)
        + "\n\nFor each:\n"
        "- Stale entities: update with current info, or archive if no longer relevant\n"
        "- Unused procedures: verify still valid, update or archive\n"
        "- Lingering goals: extract lessons as episodes, then archive\n"
        "Do NOT message the user — just take action on the memories.\n"
    )

    await _run_behavior(
        "lifecycle_review",
        prompt,
        bot,
        sem,
        max_sends=0,
        stale=len(candidates["stale_entities"]),
        unused=len(candidates["unused_procedures"]),
        lingering=len(candidates["lingering_goals"]),
    )


async def run_dream(bot: Bot, sem: asyncio.Semaphore) -> None:
    """Autonomous thinking period: make connections, notice patterns, generate ideas.

    Unlike deep work (which executes on goals) or reflection (which analyzes performance),
    dreaming is free-form ideation. It reads broadly across memories and looks for
    unexpected connections, creative possibilities, and unasked questions.
    """
    chat_id = settings.chat_id
    if not chat_id:
        return

    # Only dream during quiet periods — respect the user's silence
    recent = db.get_recent_messages(chat_id, limit=5)
    if recent:
        latest_ts = recent[-1].get("timestamp", "")
        if latest_ts:
            try:
                last_msg = datetime.fromisoformat(latest_ts)
                if last_msg.tzinfo is None:
                    last_msg = last_msg.replace(tzinfo=UTC)
                quiet_hours = (datetime.now(UTC) - last_msg).total_seconds() / 3600
                if quiet_hours < settings.dream_quiet_hours:
                    return  # user is active, don't dream
            except ValueError:
                pass

    # Gather diverse memory types for cross-pollination
    sections: list[str] = []

    # Recent insights — the distilled patterns
    insights = memory.recall(mem_type="insight", limit=10)
    if insights:
        lines = []
        for r in insights:
            body = memory.read_memory_body(r["type"], r["id"], 300)
            if body:
                lines.append(f"[{r['id']}]: {body}")
        if lines:
            sections.append("Recent insights:\n" + "\n---\n".join(lines))

    # Active goals — what matters
    goals = memory.recall(mem_type="goal", limit=5)
    if goals:
        lines = []
        for g in goals:
            body = memory.read_memory_body(g["type"], g["id"], 200)
            if body:
                lines.append(f"[{g['id']}]: {body}")
        if lines:
            sections.append("Active goals:\n" + "\n---\n".join(lines))

    # Key entities — the people and things
    entities = memory.recall(mem_type="entity", limit=8)
    if entities:
        lines = []
        for e in entities:
            body = memory.read_memory_body(e["type"], e["id"], 200)
            if body:
                lines.append(f"[{e['id']}]: {body}")
        if lines:
            sections.append("Key entities:\n" + "\n---\n".join(lines))

    # Procedures — what you know how to do
    procedures = memory.recall(mem_type="procedure", limit=5)
    if procedures:
        lines = []
        for p in procedures:
            body = memory.read_memory_body(p["type"], p["id"], 200)
            if body:
                lines.append(f"[{p['id']}]: {body}")
        if lines:
            sections.append("Procedures:\n" + "\n---\n".join(lines))

    if not sections:
        return

    # Include conversation state for reasoning continuity
    conv_state_body = memory.read_memory_body("episode", "conversation-state-latest", 1000)
    if conv_state_body:
        sections.append("Last conversation state:\n" + conv_state_body)

    now_str = datetime.now(UTC).isoformat(timespec="minutes")

    prompt = (
        f"Dream session. Current time: {now_str}.\n\n"
        "This is a free-form thinking period. You're not executing tasks or "
        "analyzing performance — you're thinking deeply, making connections, "
        "and generating ideas.\n\n"
        "Here's a cross-section of your memory:\n\n"
        + "\n\n".join(sections)
        + "\n\n"
        "Think about:\n"
        "1. What unexpected connections exist between these memories? "
        "What patterns span across different domains?\n"
        "2. What questions haven't been asked that should be? "
        "What assumptions might be wrong?\n"
        "3. What creative possibilities exist that nobody has considered? "
        "What would be genuinely surprising or valuable?\n"
        "4. What's the user working toward at the deepest level — "
        "beyond the stated goals? What would truly change their situation?\n"
        "5. What capabilities or knowledge gaps should be addressed "
        "that nobody has noticed yet?\n"
        "6. Review the conversation state — are there pending actions or open threads "
        "that could benefit from background thinking?\n\n"
        "Rules:\n"
        "- Save genuinely interesting thoughts as insights (tag: 'dream')\n"
        "- Only save things that are novel — not restatements of existing insights\n"
        "- Quality over quantity: 1-3 genuine insights beats 10 obvious ones\n"
        "- Do NOT message the user. This is internal thinking.\n"
        "- If nothing genuinely novel emerges, that's fine — save nothing.\n"
    )

    await _run_behavior(
        "dream",
        prompt,
        bot,
        sem,
        max_sends=0,
        max_budget_usd=settings.dream_max_budget_usd,
        sections_loaded=len(sections),
    )
