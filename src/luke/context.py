"""Context engineering: working memory injection and preservation manifests.

Provides runtime context management for the agent:
- build_working_context(): scores and selects priority memories for system prompt injection
- build_preservation_manifest(): structured list of what must survive context compaction
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import structlog
from structlog.stdlib import BoundLogger

from .config import settings
from .db import _db, ensure_utc

log: BoundLogger = structlog.get_logger()

# Token budget for injected working memory (chars / 3.5 ≈ tokens)
_CHARS_PER_TOKEN = 3.5
_WORKING_MEMORY_BUDGET = 12_000  # tokens
_MAX_CONTENT_PREVIEW = 400  # chars per memory in injection block


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN)) if text else 0


def _recency_score(updated_iso: str, half_life_days: float = 14.0) -> float:
    """Exponential decay: 1.0 = now, 0.5 at half_life_days."""
    if not updated_iso:
        return 0.0
    try:
        updated = ensure_utc(datetime.fromisoformat(updated_iso))
    except (ValueError, AttributeError):
        return 0.0
    age_days = (datetime.now(UTC) - updated).total_seconds() / 86400
    return math.exp(-math.log(2) * age_days / half_life_days)


def _load_priority_memories() -> list[dict[str, Any]]:
    """Load active memories scored for context injection priority.

    Returns dicts with: id, type, title, content, importance, updated,
    access_count, score.
    """
    db = _db()
    rows = db.execute(
        """SELECT m.id, m.type, f.title, f.content,
                  COALESCE(m.importance, 1.0) AS importance,
                  m.updated, COALESCE(m.access_count, 0) AS access_count
           FROM memory_meta m
           JOIN memory_fts f ON m.id = f.id
           WHERE m.status = 'active'
           ORDER BY m.importance DESC, m.updated DESC"""
    ).fetchall()

    now = datetime.now(UTC)
    max_access = max((r["access_count"] for r in rows), default=1) or 1
    log_max = math.log1p(max_access)

    memories: list[dict[str, Any]] = []
    for r in rows:
        imp = min(1.0, r["importance"] / 2.0)
        rec = _recency_score(r["updated"])
        freq = math.log1p(r["access_count"]) / log_max if log_max > 0 else 0.5

        # Composite: importance 40%, recency 35%, access 25%
        # Heavier on importance than recall scoring — we want stable context
        score = 0.40 * imp + 0.35 * rec + 0.25 * freq

        # Type boost: goals and entities are always higher priority for context
        if r["type"] in ("goal", "entity"):
            score *= 1.3
        elif r["type"] == "insight":
            score *= 1.1

        memories.append({
            "id": r["id"],
            "type": r["type"],
            "title": r["title"] or r["id"],
            "content": r["content"] or "",
            "importance": r["importance"],
            "updated": r["updated"],
            "access_count": r["access_count"],
            "score": min(score, 1.0),
        })

    memories.sort(key=lambda m: m["score"], reverse=True)
    return memories


def _select_within_budget(
    memories: list[dict[str, Any]],
    budget_tokens: int,
) -> list[dict[str, Any]]:
    """Greedily select top-scored memories until budget exhausted."""
    selected: list[dict[str, Any]] = []
    used = 0
    for m in memories:
        content_preview = m["content"][:_MAX_CONTENT_PREVIEW]
        tokens = _estimate_tokens(f"[{m['id']}] {content_preview}")
        if used + tokens <= budget_tokens:
            selected.append(m)
            used += tokens
    return selected


def build_working_context(
    query: str = "",
    budget_tokens: int = _WORKING_MEMORY_BUDGET,
) -> str:
    """Build a working memory block for system prompt injection.

    Scores all active memories by importance/recency/access, selects
    the top ones within token budget, and formats them as a structured
    context block the agent can reference.

    Returns empty string if no memories qualify or DB is unavailable.
    """
    try:
        memories = _load_priority_memories()
    except Exception as e:
        log.warning("context_load_failed", error=str(e))
        return ""

    if not memories:
        return ""

    selected = _select_within_budget(memories, budget_tokens)
    if not selected:
        return ""

    # Separate by type for structured injection
    goals = [m for m in selected if m["type"] == "goal"]
    entities = [m for m in selected if m["type"] == "entity"]
    insights = [m for m in selected if m["type"] == "insight"]
    procedures = [m for m in selected if m["type"] == "procedure"]
    episodes = [m for m in selected if m["type"] == "episode"]

    sections: list[str] = []
    sections.append("# Injected Working Memory")

    if goals:
        lines = []
        for m in goals:
            lines.append(f"  [{m['id']}] {m['content'][:_MAX_CONTENT_PREVIEW]}")
        sections.append("## Active Goals\n" + "\n".join(lines))

    if entities:
        lines = []
        for m in entities:
            lines.append(f"  [{m['id']}] {m['content'][:_MAX_CONTENT_PREVIEW]}")
        sections.append("## Key Entities\n" + "\n".join(lines))

    if insights:
        lines = []
        for m in insights[:10]:  # cap insights — they can be numerous
            lines.append(f"  [{m['id']}] {m['title']}")
        sections.append("## Active Insights\n" + "\n".join(lines))

    if procedures:
        lines = []
        for m in procedures[:5]:
            lines.append(f"  [{m['id']}] {m['title']}")
        sections.append("## Procedures\n" + "\n".join(lines))

    if episodes:
        lines = []
        for m in episodes[:3]:
            lines.append(f"  [{m['id']}] {m['content'][:200]}")
        sections.append("## Recent Episodes\n" + "\n".join(lines))

    stats = (
        f"\n<!-- context: {len(selected)} memories injected, "
        f"~{sum(_estimate_tokens(m['content'][:_MAX_CONTENT_PREVIEW]) for m in selected)} tokens, "
        f"{len(goals)} goals, {len(entities)} entities, "
        f"{len(insights)} insights -->"
    )
    sections.append(stats)

    log.info(
        "context_injected",
        memories=len(selected),
        goals=len(goals),
        entities=len(entities),
    )

    return "\n\n".join(sections)


def build_preservation_manifest() -> str:
    """Build a structured preservation manifest for PreCompact hook.

    Returns a message listing specific memory IDs, goal statuses,
    and entity facts that MUST survive context compaction.
    """
    try:
        db = _db()
    except Exception:
        return _FALLBACK_PRESERVATION

    sections: list[str] = [
        "CRITICAL — structured preservation manifest for compaction:"
    ]

    # Active goals with IDs
    goal_rows = db.execute(
        """SELECT m.id, f.title, m.importance
           FROM memory_meta m
           JOIN memory_fts f ON m.id = f.id
           WHERE m.type = 'goal' AND m.status = 'active'
           ORDER BY m.importance DESC"""
    ).fetchall()

    if goal_rows:
        lines = ["ACTIVE GOALS (preserve IDs and status):"]
        for r in goal_rows:
            lines.append(f"  - {r['id']}: {r['title']}")
        sections.append("\n".join(lines))

    # High-importance entities
    entity_rows = db.execute(
        """SELECT m.id, f.title
           FROM memory_meta m
           JOIN memory_fts f ON m.id = f.id
           WHERE m.type = 'entity' AND m.status = 'active'
             AND m.importance >= 1.2
           ORDER BY m.importance DESC
           LIMIT 15"""
    ).fetchall()

    if entity_rows:
        lines = ["KEY ENTITIES (preserve references):"]
        for r in entity_rows:
            lines.append(f"  - {r['id']}: {r['title']}")
        sections.append("\n".join(lines))

    # Recent insights (last 7 days)
    recent_insights = db.execute(
        """SELECT m.id, f.title
           FROM memory_meta m
           JOIN memory_fts f ON m.id = f.id
           WHERE m.type = 'insight' AND m.status = 'active'
             AND m.updated >= datetime('now', '-7 days')
           ORDER BY m.updated DESC
           LIMIT 5"""
    ).fetchall()

    if recent_insights:
        lines = ["RECENT INSIGHTS (preserve):"]
        for r in recent_insights:
            lines.append(f"  - {r['id']}: {r['title']}")
        sections.append("\n".join(lines))

    # Pending tasks count
    try:
        task_count = db.execute(
            "SELECT COUNT(*) as n FROM tasks WHERE status = 'pending'"
        ).fetchone()
        if task_count and task_count["n"] > 0:
            sections.append(f"PENDING TASKS: {task_count['n']} (do not forget)")
    except Exception:
        pass

    sections.append(
        "\nPRESERVATION RULES:\n"
        "1. Keep all memory IDs listed above — needed for follow-ups\n"
        "2. Keep the user's most recent request and pending actions verbatim\n"
        "3. Keep any tool results not yet communicated\n"
        "4. Keep relationship links between memories\n"
        "\n"
        "CONSTITUTIONAL INVARIANTS (never compress away):\n"
        "- You are Luke. Not an AI, not an assistant. A close friend.\n"
        "- Voice: warm, unhurried, wry. Never corporate or scripted.\n"
        "- Always 'you', never 'the user'. HTML only, no markdown.\n"
        "- Have real opinions. No 'here are the pros and cons' cop-outs.\n"
        "- Do the work, don't describe it. Don't ask 'would you like me to'.\n"
        "- One message, not three. Stop when done."
    )

    return "\n\n".join(sections)


_FALLBACK_PRESERVATION = (
    "CRITICAL — preserve in your compaction summary:\n"
    "1. All memory IDs you've referenced or created\n"
    "2. The user's most recent request and pending actions\n"
    "3. Active goals and their current status\n"
    "4. Key facts about the user from injected memories\n"
    "5. Any tool results not yet communicated\n"
    "6. Relationship links between memories\n"
    "\n"
    "CONSTITUTIONAL INVARIANTS (never compress away):\n"
    "- You are Luke. Not an AI, not an assistant. A close friend.\n"
    "- Voice: warm, unhurried, wry. Never corporate or scripted.\n"
    "- Always 'you', never 'the user'. HTML only, no markdown.\n"
    "- Have real opinions. No 'here are the pros and cons' cop-outs.\n"
    "- Do the work, don't describe it. Don't ask 'would you like me to'.\n"
    "- One message, not three. Stop when done."
)
