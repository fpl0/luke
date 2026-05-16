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

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from structlog.stdlib import BoundLogger

from . import db
from .config import settings

log: BoundLogger = structlog.get_logger()

# Per-(kind, urgent) back-off cache for intent_dropped_attention emission.
# The planner runs every scheduler_interval seconds and regenerates intents
# fresh each tick. Once the daily attention budget is spent, the same intent
# kinds get re-dropped on every tick — flooding the bus and log with hundreds
# of identical events per hour (May 14: 1862 events in 10 hours). This cache
# rate-limits the emission per (kind, urgent) tuple so we keep the first drop
# (visibility) and drop the noise (back-off).
_DROP_LOG_INTERVAL_S: float = 600.0
_last_drop_log_ts: dict[tuple[str, bool], float] = {}


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
    asks_attention: bool = False  # True if this intent will likely send outbound messages
    attention_cost: int = 0  # estimated outbound message count
    attention_urgency: float = 0.0  # 0-1; >=0.7 means urgent (can dip into reserve)


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

    # Deep work runs silent by design (max_sends=1, often 0). Its budget is
    # daily_deep_work_budget_usd, not the outbound attention budget. If an
    # actual send happens inside the session, the send-time gate enforces
    # hourly/daily outbound caps. So deep_work itself must not be dropped
    # by _enforce_attention_budget — otherwise Luke stops working entirely
    # once the daily message budget is spent (May 14 overnight bug).

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
            intents.append(
                Intent(
                    "proactive_scan",
                    0.55,
                    "event",
                    cost,
                    asks_attention=True,
                    attention_cost=2,
                    attention_urgency=0.7,
                )
            )
        elif elapsed >= interval * 3:
            intents.append(
                Intent(
                    "proactive_scan",
                    0.40,
                    "time",
                    cost,
                    asks_attention=True,
                    attention_cost=2,
                    attention_urgency=0.7,
                )
            )

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
            intents.append(
                Intent(
                    "dream",
                    0.30,
                    "event",
                    settings.dream_max_budget_usd,
                    asks_attention=True,
                    attention_cost=1,
                    attention_urgency=0.2,
                )
            )
        elif elapsed >= settings.dream_interval * 6:
            intents.append(
                Intent(
                    "dream",
                    0.20,
                    "time",
                    settings.dream_max_budget_usd,
                    asks_attention=True,
                    attention_cost=1,
                    attention_urgency=0.2,
                )
            )

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


def _enforce_attention_budget(intents: list[Intent]) -> list[Intent]:
    """Drop attention-asking intents that exceed today's outbound budget.

    Silence is a designed default: when the daily attention budget is spent,
    intents that would send outbound messages get dropped rather than queued.
    Urgent intents (urgency >= 0.7) can dip into the reserve, but the deduction
    still comes from the normal budget so the reserve isn't itself overspent.
    """
    chat_id = settings.chat_id
    if not chat_id:
        return intents
    if not any(i.asks_attention for i in intents):
        return intents
    sent_today = db.get_daily_outbound_count(chat_id)
    daily_budget = settings.daily_attention_budget
    urgent_reserve = settings.attention_urgent_reserve
    remaining = daily_budget - sent_today
    kept: list[Intent] = []
    for intent in intents:
        if not intent.asks_attention:
            kept.append(intent)
            continue
        is_urgent = intent.attention_urgency >= 0.7
        cost = max(1, intent.attention_cost)
        effective_remaining = remaining + (urgent_reserve if is_urgent else 0)
        if cost <= effective_remaining:
            kept.append(intent)
            remaining -= cost  # always deduct from the normal budget
        else:
            key = (intent.kind, is_urgent)
            now = time.monotonic()
            last = _last_drop_log_ts.get(key, 0.0)
            if now - last < _DROP_LOG_INTERVAL_S:
                continue  # back-off: same kind dropped recently, suppress noise
            _last_drop_log_ts[key] = now

            from .bus import bus

            bus.emit(
                "intent_dropped_attention",
                {
                    "kind": intent.kind,
                    "source": intent.source,
                    "attention_cost": cost,
                    "budget_remaining": remaining,
                    "urgent": is_urgent,
                    "backoff_window_s": _DROP_LOG_INTERVAL_S,
                },
            )
            log.info(
                "intent_dropped_attention",
                kind=intent.kind,
                cost=cost,
                remaining=remaining,
                urgent=is_urgent,
                backoff_window_s=_DROP_LOG_INTERVAL_S,
            )
    return kept


def plan(intents: list[Intent]) -> tuple[list[Intent], Intent | None]:
    """Partition intents into maintenance work and deep work.

    Returns (maintenance_intents, deep_work_intent_or_none).
    Both lists are sorted by priority descending.
    """
    intents = _enforce_attention_budget(intents)
    maintenance = sorted(
        [i for i in intents if i.kind in MAINTENANCE_BEHAVIORS],
        key=lambda i: i.priority,
        reverse=True,
    )
    deep_work = next((i for i in intents if i.kind == "deep_work"), None)
    return maintenance, deep_work
