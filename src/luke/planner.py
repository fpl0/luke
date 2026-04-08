"""Intent-driven planning: generate, prioritize, and execute work intents.

Replaces the scheduler's per-behavior "am I due?" blocks with a unified
planning layer that asks: "What's the most valuable thing to do right now?"

Intent sources:
  - Goals: active goals generate 'deep_work' intents (priority 0.75-0.85)
  - Events: accumulated events generate maintenance intents (priority 0.3-0.55)
  - Time: overdue behaviors generate fallback intents (priority 0.2-0.4)

The planner generates intents, scores them, and returns a prioritized plan.
The scheduler executes the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from structlog.stdlib import BoundLogger

from . import db
from .config import settings

log: BoundLogger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Intent model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Intent:
    """A declared intention to do work, with priority and budget."""

    kind: str  # behavior name: "deep_work", "consolidation", etc.
    priority: float  # 0.0-1.0 (higher = more urgent)
    source: str  # "goal", "event", "time"
    budget_usd: float  # estimated cost
    context: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Behavior classification
# ---------------------------------------------------------------------------

# Maintenance behaviors run concurrently (short-lived, awaited)
MAINTENANCE_BEHAVIORS: frozenset[str] = frozenset(
    {
        "consolidation",
        "reflection",
        "proactive_scan",
        "insight_consolidation",
        "feedback_consolidation",
        "lifecycle_review",
        "skill_extraction",
        "dream",
    }
)

# Events consumed by each behavior after completion
BEHAVIOR_EVENTS: dict[str, tuple[str, ...]] = {
    "consolidation": ("new_episode",),
    "reflection": ("feedback_negative", "user_message"),
    "proactive_scan": ("goal_updated",),
    "insight_consolidation": ("new_insight",),
    "feedback_consolidation": ("feedback_negative",),
    "lifecycle_review": (),
    "skill_extraction": (),
    "dream": (),
}

# Per-behavior backoff caps: high-frequency behaviors cap lower
_BACKOFF_CAPS: dict[str, int] = {
    "proactive_scan": 1,  # max 2x (6h) — must hit 2/day target
    "consolidation": 2,  # max 4x (24h) — must hit 1/day target
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seconds_since_last_run(name: str) -> float:
    """How many seconds since this behavior last ran. Returns inf if never ran."""
    iso = db.get_behavior_last_run(name)
    if iso is None:
        return float("inf")
    last = db.ensure_utc(datetime.fromisoformat(iso))
    return (datetime.now(UTC) - last).total_seconds()


def effective_interval(name: str, base: float) -> float:
    """Apply exponential backoff based on consecutive no-op runs.

    Formula: base * (2 ** min(no_ops, cap))
    Default cap is 4 (16x max), but some behaviors have tighter caps.
    """
    try:
        no_ops = int(db.get_behavior_no_ops(name))
    except (TypeError, ValueError):
        no_ops = 0
    max_exp = _BACKOFF_CAPS.get(name, 4)
    return float(base * (2 ** min(no_ops, max_exp)))


# ---------------------------------------------------------------------------
# Intent generators
# ---------------------------------------------------------------------------


def _deep_work_intents() -> list[Intent]:
    """Active goals generate 'deep_work' intents."""
    elapsed = _seconds_since_last_run("deep_work")
    if elapsed < settings.deep_work_interval:
        return []

    # Budget gate
    daily_cost = db.get_daily_deep_work_cost()
    remaining = settings.daily_deep_work_budget_usd - daily_cost
    if remaining <= 0:
        return []

    # Event-driven: goal_updated events → higher priority
    has_goal_events = db.count_unconsumed_events("goal_updated") >= 1
    if has_goal_events:
        return [
            Intent(
                kind="deep_work",
                priority=0.85,
                source="goal",
                budget_usd=min(settings.deep_work_max_budget_usd, remaining),
                context={"daily_remaining_usd": remaining},
            )
        ]

    # Time-based fallback: run even without events after 2x interval
    if elapsed >= settings.deep_work_interval * 2:
        return [
            Intent(
                kind="deep_work",
                priority=0.75,
                source="time",
                budget_usd=min(settings.deep_work_max_budget_usd, remaining),
                context={"daily_remaining_usd": remaining},
            )
        ]

    return []


def _maintenance_intents() -> list[Intent]:
    """Time+event hybrid intents for maintenance behaviors."""
    intents: list[Intent] = []
    cost = settings.behavior_max_budget_usd

    # --- Consolidation ---
    interval = effective_interval("consolidation", settings.episode_consolidation_interval)
    elapsed = _seconds_since_last_run("consolidation")
    if elapsed >= interval:
        has_episodes = (
            db.count_unconsumed_events("new_episode") >= settings.consolidation_min_cluster
        )
        if has_episodes:
            intents.append(Intent("consolidation", 0.50, "event", cost))
        elif elapsed >= interval * 2:
            intents.append(Intent("consolidation", 0.35, "time", cost))

    # --- Proactive scan ---
    interval = effective_interval("proactive_scan", settings.proactive_scan_interval)
    elapsed = _seconds_since_last_run("proactive_scan")
    if elapsed >= interval:
        has_activity = db.count_unconsumed_events("goal_updated", "user_message") >= 1
        if has_activity:
            intents.append(Intent("proactive_scan", 0.55, "event", cost))
        elif elapsed >= interval * 3:
            intents.append(Intent("proactive_scan", 0.40, "time", cost))

    # --- Reflection (weekly) ---
    interval = effective_interval("reflection", settings.reflection_interval)
    elapsed = _seconds_since_last_run("reflection")
    if elapsed >= interval:
        has_feedback = db.count_unconsumed_events("feedback_negative", "user_message") >= 5
        if has_feedback:
            intents.append(Intent("reflection", 0.45, "event", cost))
        elif elapsed >= interval * 6:
            intents.append(Intent("reflection", 0.30, "time", cost))

    # --- Insight consolidation ---
    interval = effective_interval(
        "insight_consolidation", settings.insight_consolidation_interval
    )
    elapsed = _seconds_since_last_run("insight_consolidation")
    if elapsed >= interval:
        has_insights = db.count_unconsumed_events("new_insight") >= 3
        if has_insights:
            intents.append(Intent("insight_consolidation", 0.45, "event", cost))
        elif elapsed >= interval * 6:
            intents.append(Intent("insight_consolidation", 0.30, "time", cost))

    # --- Feedback consolidation ---
    interval = effective_interval(
        "feedback_consolidation", settings.feedback_consolidation_interval
    )
    elapsed = _seconds_since_last_run("feedback_consolidation")
    if elapsed >= interval:
        has_negatives = db.count_unconsumed_events("feedback_negative") >= 2
        if has_negatives:
            intents.append(Intent("feedback_consolidation", 0.45, "event", cost))
        elif elapsed >= interval * 6:
            intents.append(Intent("feedback_consolidation", 0.30, "time", cost))

    # --- Lifecycle review ---
    elapsed = _seconds_since_last_run("lifecycle_review")
    if elapsed >= settings.lifecycle_review_interval:
        has_activity = db.count_unconsumed_events("new_episode", "new_insight") >= 5
        if has_activity:
            intents.append(Intent("lifecycle_review", 0.35, "event", cost))
        elif elapsed >= settings.lifecycle_review_interval * 6:
            intents.append(Intent("lifecycle_review", 0.25, "time", cost))

    # --- Skill extraction ---
    interval = effective_interval("skill_extraction", settings.skill_extraction_interval)
    elapsed = _seconds_since_last_run("skill_extraction")
    if elapsed >= interval:
        has_episodes = db.count_unconsumed_events("new_episode") >= 2
        if has_episodes:
            intents.append(Intent("skill_extraction", 0.40, "event", cost))
        elif elapsed >= interval * 6:
            intents.append(Intent("skill_extraction", 0.25, "time", cost))

    # --- Dream ---
    elapsed = _seconds_since_last_run("dream")
    if elapsed >= settings.dream_interval:
        has_material = db.count_unconsumed_events("new_insight", "new_episode") >= 1
        if has_material:
            intents.append(Intent("dream", 0.30, "event", settings.dream_max_budget_usd))
        elif elapsed >= settings.dream_interval * 6:
            intents.append(Intent("dream", 0.20, "time", settings.dream_max_budget_usd))

    return intents


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_intents() -> list[Intent]:
    """Collect all pending intents from all sources."""
    intents: list[Intent] = []
    intents.extend(_deep_work_intents())
    intents.extend(_maintenance_intents())
    return intents


def plan(intents: list[Intent]) -> tuple[list[Intent], Intent | None]:
    """Partition intents into maintenance work and deep work.

    Returns (maintenance_intents, deep_work_intent_or_none).
    Both lists are sorted by priority descending.
    """
    maintenance = sorted(
        [i for i in intents if i.kind in MAINTENANCE_BEHAVIORS],
        key=lambda i: i.priority,
        reverse=True,
    )
    deep_work = next((i for i in intents if i.kind == "deep_work"), None)
    return maintenance, deep_work
