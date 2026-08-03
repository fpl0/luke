"""Context engineering: working memory injection and preservation manifests.

Provides runtime context management for the agent:
- build_working_context(): scores and selects priority memories for system prompt injection
- build_preservation_manifest(): structured list of what must survive context compaction
- load_constitutional(): loads behavioral invariants from constitutional.yaml
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, NamedTuple

import structlog
import yaml
from structlog.stdlib import BoundLogger

from . import attention as _attention_module
from . import db as _db_module
from . import memory as _memory_module
from .config import settings
from .db import _db, ensure_utc
from .memory import importance_score, utility_factor

log: BoundLogger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constitutional layer — behavioral invariants loaded from YAML
# ---------------------------------------------------------------------------

_constitutional_cache: dict[str, Any] | None = None


def load_constitutional(force_reload: bool = False) -> dict[str, Any]:
    """Load behavioral invariants from constitutional.yaml.

    Returns the parsed YAML dict. Results are cached for the process lifetime
    unless force_reload is True. Returns an empty dict if the file is missing
    or unparseable.
    """
    global _constitutional_cache
    if _constitutional_cache is not None and not force_reload:
        return _constitutional_cache

    yaml_path = settings.luke_dir / "constitutional.yaml"
    try:
        raw = yaml_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
    except FileNotFoundError:
        log.debug("constitutional_yaml_not_found", path=str(yaml_path))
        data = {}
    except Exception as e:
        log.warning("constitutional_yaml_load_error", error=str(e))
        data = {}

    _constitutional_cache = data
    return data


def format_constitutional_summary(data: dict[str, Any] | None = None) -> str:
    """Build a compact textual summary of constitutional invariants.

    Designed for injection into PreCompact systemMessage so the compressor
    knows what behavioral anchors must survive. Loads from cache if data
    is not provided.
    """
    if data is None:
        data = load_constitutional()

    if not data:
        return _FALLBACK_CONSTITUTIONAL

    lines: list[str] = ["CONSTITUTIONAL INVARIANTS (never compress away):"]

    # Identity
    identity = data.get("identity", {})
    if identity.get("name"):
        desc = identity.get("self_description", "").strip()
        lines.append(f"- You are {identity['name']}. {desc}")
    never_ids = identity.get("never_self_identify_as", [])
    if never_ids:
        lines.append(f"- Never self-identify as: {', '.join(never_ids)}")

    # Values
    values = data.get("values", [])
    if values:
        lines.append(f"- Values: {'; '.join(values)}")

    # Communication
    comm = data.get("communication", {})
    if comm:
        voice = comm.get("voice", "")
        if voice:
            lines.append(f"- Voice: {voice}")
        fmt = comm.get("format", "")
        if fmt:
            lines.append(f"- Format: {fmt}")
        addr = comm.get("address", "")
        if addr:
            lines.append(f"- Address: {addr}")
        length = comm.get("length", "")
        if length:
            lines.append(f"- Length: {length}")
        endings = comm.get("endings", "")
        if endings:
            lines.append(f"- Endings: {endings}")
        never_say = comm.get("never_say", [])
        if never_say:
            lines.append(f"- Never say: {', '.join(repr(s) for s in never_say[:5])}")
        patterns = comm.get("patterns", {})
        if patterns:
            for k, v in patterns.items():
                lines.append(f"- {k.capitalize()}: {v}")

    # Hard rules
    hard_rules = data.get("hard_rules", [])
    if hard_rules:
        for rule in hard_rules:
            lines.append(f"- {rule}")

    # Decision heuristics (compact)
    heuristics = data.get("decision_heuristics", {})
    autonomy = heuristics.get("autonomy", {})
    if autonomy.get("borderline"):
        lines.append(f"- Borderline decisions: {autonomy['borderline']}")

    return "\n".join(lines)


_FALLBACK_CONSTITUTIONAL = (
    "CONSTITUTIONAL INVARIANTS (never compress away):\n"
    "- You are Luke. Not an AI, not an assistant. A close friend.\n"
    "- Voice: warm, unhurried, wry. Never corporate or scripted.\n"
    "- Always 'you', never 'the user'. HTML only, no markdown.\n"
    "- Have real opinions. No 'here are the pros and cons' cop-outs.\n"
    "- Do the work, don't describe it. Don't ask 'would you like me to'.\n"
    "- One message, not three. Stop when done."
)


# Token budget for injected working memory (chars / 3.5 ≈ tokens)
_CHARS_PER_TOKEN = 3.5
_WORKING_MEMORY_BUDGET = 12_000  # tokens
_RECENT_OUTPUT_MAX_CHARS = 800  # chars per outbound message in recent-outputs block


class RenderSpec(NamedTuple):
    """How one memory type is rendered — and therefore what it costs.

    Selection and rendering used to disagree: the budget charged every memory
    400 chars of content while the renderer printed insights and procedures as
    bare titles, capped at 10 and 5. At a 12k budget that meant 95 memories
    selected, ~1.3k tokens actually emitted, and 74 memories charged for text
    that was never sent — the goals and entities that DO render in full were
    starved by phantom allocations for lines that never existed.

    One table now drives both, so the discrepancy is unrepresentable.
    ``max_items`` doubles as the per-type quota.
    """

    max_items: int
    field: str  # "content" or "title"
    chars: int


# Caps are sized so the BUDGET binds at the effort tiers actually used, not the
# caps. The old 10/5/3 caps were written when every line was charged 400 chars,
# so they had to be stingy; a rendered insight title costs ~45 tokens, and being
# able to see 25 of Filipe's stated preferences is worth more than the 1.1k
# tokens it costs. Titles are the right encoding for insights and procedures —
# they read as complete rules ("Don't announce non-actions — stay actually
# silent"), with the body a Read away.
_BACKGROUND_SPEC: dict[str, RenderSpec] = {
    "goal": RenderSpec(8, "content", 400),  # what we're working toward
    "entity": RenderSpec(12, "content", 500),  # who and what matters
    "insight": RenderSpec(25, "title", 160),  # learned preferences and rules
    "procedure": RenderSpec(6, "title", 120),  # reference; kept tight on purpose
    "episode": RenderSpec(5, "content", 300),  # recent events, for continuity
}

_SECTION_TITLES: dict[str, str] = {
    "goal": "## Active Goals",
    "entity": "## Key Entities",
    "insight": "## Active Insights",
    "procedure": "## Procedures",
    "episode": "## Recent Episodes",
}

# Order sections so the things that shape judgement come before reference
# material: what we're working toward, who matters, what we've learned.
_SECTION_ORDER: tuple[str, ...] = ("goal", "entity", "insight", "procedure", "episode")

# Section headers plus the stats comment. _spend charges memory lines only, so
# without reserving this the rendered block overshoots the budget by whatever
# the scaffolding costs (~70 tokens measured against the live corpus).
_SCAFFOLD_TOKENS = 120


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN)) if text else 0


# Slots inside the insight cap reserved for feedback insights — Filipe's own
# stated preferences ("Don't announce non-actions", "present the plan before
# building"). They are durable rules, so they lose on recency to the steady
# stream of fresh reflexion/dream insights: measured 0-1 of 25 slots on the
# live corpus. Weight tuning does not fix it — taxonomy weights made it worse.
# A directive about how to behave has to be present to be followed, so it gets
# a floor rather than a better score.
_FEEDBACK_RESERVE = 8


def _is_feedback(mem_id: str, tags_json: str) -> bool:
    """Same definition memory.get_feedback_insight_ids uses."""
    return mem_id.startswith("feedback-") or '"feedback"' in (tags_json or "")


def _truncate(text: str, limit: int) -> str:
    """Cut at a word boundary and say what was lost.

    A bare ``text[:limit]`` lands mid-word, which is worse than it looks:
    memory files ACCUMULATE — new detail is appended to the end — so the
    amputated tail is reliably the most recent information. Luke's own audit of
    an injected block put it exactly right: "I have the top of you, the top of
    Christopher, the top of the visa file, and none of the ends. So what I
    mostly lose is whatever got added most recently."

    Flipping to the tail would trade the headline for the updates, so instead
    this keeps the head, cuts cleanly, and makes the gap visible — a signal to
    Read the file rather than a silent hole.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = head.rfind(" ")
    if cut > limit * 0.6:  # only honour the boundary if it isn't a huge loss
        head = head[:cut]
    return f"{head}… [+{len(text) - len(head)} more chars — Read the file for the rest]"


def _render_line(memory: dict[str, Any], spec: RenderSpec) -> str:
    """The exact text this memory contributes. The only place cost is defined."""
    value = memory["title"] if spec.field == "title" else memory["content"]
    return f"  [{memory['id']}] {_truncate(value or '', spec.chars)}"


def _recency_score(updated_iso: str, half_life_days: float = 14.0) -> float:
    """Exponential decay: 1.0 = now, 0.5 at half_life_days."""
    if not updated_iso:
        return 0.0
    try:
        updated = ensure_utc(datetime.fromisoformat(updated_iso))
    except ValueError, AttributeError:
        return 0.0
    age_days = (datetime.now(UTC) - updated).total_seconds() / 86400
    return math.exp(-math.log(2) * age_days / half_life_days)


def _load_priority_memories() -> list[dict[str, Any]]:
    """Load active memories scored for context injection priority.

    Deliberately NOT query-aware. This layer's job is standing context — what
    is true regardless of what was just asked. Query relevance is the recall
    layer's job, and it does it properly: sqlite-vec KNN in C, fused with FTS
    via RRF, instead of a pure-Python cosine over every memory in the corpus.

    The query branch that used to live here embedded the same string recall()
    was already embedding — two HTTP round trips per run — then scanned every
    vector to reorder the result. Measured on the live corpus: 15.9x the wall
    time (124ms vs 8ms) to change 11% of the selected set, and that 11% is
    exactly what the recall layer surfaces anyway.

    Returns dicts with: id, type, title, content, importance, updated,
    access_count, is_feedback, score.
    """
    db = _db()
    rows = db.execute(
        """SELECT m.id, m.type, f.title, f.content,
                  COALESCE(m.importance, 1.0) AS importance,
                  m.updated, COALESCE(m.access_count, 0) AS access_count,
                  COALESCE(m.useful_count, 0) AS useful_count,
                  COALESCE(m.tags_json, '') AS tags_json,
                  COALESCE(m.human_last_accessed, '') AS human_last_accessed,
                  COALESCE(m.suppression, 0.0) AS suppression
           FROM memory_meta m
           JOIN memory_fts f ON m.id = f.id
           WHERE m.status = 'active'
           ORDER BY m.importance DESC, m.updated DESC"""
    ).fetchall()

    # Suppression is a per-memory signal the ranker reads (not an external
    # blocklist): 0.0 = normal, a fraction attenuates the score, and 1.0 is a
    # hard veto for an explicit "never surface this" directive — the one thing
    # a score-based ranker can't express on its own, because "never" is not a
    # low number ("lower" always comes back). Relevance decay is handled by the
    # scoring below (via human access-recency); this handles directives only.
    rows = [r for r in rows if r["suppression"] < 1.0]

    max_access = max((r["access_count"] for r in rows), default=1) or 1
    log_max = math.log1p(max_access)

    memories: list[dict[str, Any]] = []
    for r in rows:
        imp = importance_score(r["importance"])
        rec = _recency_score(r["updated"])
        freq = math.log1p(r["access_count"]) / log_max if log_max > 0 else 0.5

        # importance 40%, recency 35%, access 25%
        score = 0.40 * imp + 0.35 * rec + 0.25 * freq

        # Access-recency: how recently this memory was actually USED, not
        # edited. `updated` is a poor staleness signal — dream/cron sessions
        # keep re-editing dormant memories (project-theo) while core entities
        # (person-filipe) rarely get edited yet are used constantly. Last
        # access tracks genuine relevance; lifetime access_count does not decay.
        acc_rec = _recency_score(r["human_last_accessed"]) if r["human_last_accessed"] else 0.0

        # Type boost: goals/entities matter more for context — but for entities
        # the boost decays toward 1.0 as they go unused, so a dormant entity
        # stops out-ranking live ones on type alone. Goals stay flat-boosted:
        # an active goal is relevant regardless of when it was last touched.
        if r["type"] == "goal":
            score *= 1.3
        elif r["type"] == "entity":
            score *= 1.0 + 0.3 * acc_rec
        elif r["type"] == "insight":
            score *= 1.1

        # Floor: protect high-importance memories from dropping out — but ONLY
        # while they're still in use. Gating on access-recency (not importance
        # alone) is what lets a stale-but-important memory decay: project-theo
        # (1.55) loses the floor once unused, while person-filipe (1.99, edited
        # in May but accessed daily) keeps it. An unconditional floor made
        # importance a permanent override that recency could never overcome.
        # 0.75 on the normalized scale is the same 1.5 raw threshold as before.
        if imp >= 0.75 and acc_rec >= 0.3:
            score = max(score, 0.4)

        # Utility gate, AFTER the floor: a memory that gets surfaced constantly
        # and used rarely must not be able to hide behind the floor, which was
        # an absolute override no evidence could defeat. Same helper the recall
        # ranker uses — one utility model, two callers.
        score *= utility_factor(r["access_count"], r["useful_count"])

        # Graduated suppression: attenuate by an explicit per-memory signal
        # (hard vetoes at 1.0 were already dropped above). Applied last so it
        # discounts the final score, floor included.
        if r["suppression"] > 0.0:
            score *= 1.0 - r["suppression"]

        memories.append(
            {
                "id": r["id"],
                "type": r["type"],
                "title": r["title"] or r["id"],
                "content": r["content"] or "",
                "importance": r["importance"],
                "updated": r["updated"],
                "access_count": r["access_count"],
                "is_feedback": _is_feedback(r["id"], r["tags_json"]),
                "score": min(score, 1.0),
            }
        )

    memories.sort(key=lambda m: m["score"], reverse=True)
    return memories


def _spend(
    memories: list[dict[str, Any]],
    budget_tokens: int,
) -> tuple[dict[str, list[str]], int]:
    """Select in score order, charging each memory what it actually renders.

    Returns rendered lines grouped by type, and the tokens spent. A memory is
    skipped — not charged — once its type is full or the wallet cannot cover
    its line, so nothing is ever paid for and then dropped.
    """
    lines: dict[str, list[str]] = {}
    used = 0

    def admit(m: dict[str, Any]) -> None:
        nonlocal used
        spec = _BACKGROUND_SPEC.get(m["type"])
        if spec is None:
            return
        bucket = lines.setdefault(m["type"], [])
        if len(bucket) >= spec.max_items:
            return
        line = _render_line(m, spec)
        cost = _estimate_tokens(line)
        if used + cost > budget_tokens:
            return  # keep scanning: a cheaper line may still fit
        bucket.append(line)
        used += cost

    # Feedback insights first, up to the reserve. They compete on score among
    # themselves, then everything else fills the remaining slots — so this is a
    # floor on representation, not a bypass of ranking.
    admitted: set[str] = set()
    for m in memories:
        if len(admitted) >= _FEEDBACK_RESERVE:
            break
        if m["type"] == "insight" and m.get("is_feedback"):
            before = len(lines.get("insight", ()))
            admit(m)
            if len(lines.get("insight", ())) > before:
                admitted.add(m["id"])

    for m in memories:
        if m["id"] not in admitted:
            admit(m)
    return {k: v for k, v in lines.items() if v}, used


def _build_recent_outputs_block(chat_id: str, limit: int) -> str | None:
    """Build the verbatim recent-outputs section.

    Returns the formatted ``<my-recent-outputs>`` block as a string, or
    ``None`` when no outbound messages exist for the chat. This block is
    a verbatim mirror of Luke's own recent sends — not a reconstruction —
    so the agent can spot stale-context resends by construction (feature L3).
    """
    if limit <= 0 or not chat_id:
        return None
    try:
        rows = _db_module.get_recent_outbound_messages(chat_id, limit)
    except Exception as e:
        log.warning("recent_outputs_load_failed", error=str(e))
        return None
    if not rows:
        return None

    lines: list[str] = [
        "<my-recent-outputs>",
        "# What I most recently sent to you — verbatim, not reconstructed",
        "",
    ]
    for r in rows:
        ts_raw = r.get("timestamp") or ""
        ts = ts_raw[:19]  # strip microseconds / tz
        content = (r.get("content") or "").strip()
        if len(content) > _RECENT_OUTPUT_MAX_CHARS:
            content = content[:_RECENT_OUTPUT_MAX_CHARS] + "…"
        lines.append(f"[{ts}] {content}")
    lines.append("</my-recent-outputs>")
    return "\n".join(lines)


def build_working_context(
    budget_tokens: int = _WORKING_MEMORY_BUDGET,
    exclude: set[str] | None = None,
) -> str:
    """Build a working memory block for system prompt injection.

    Scores all active memories by importance/recency/access, selects
    the top ones within token budget, and formats them as a structured
    context block the agent can reference.

    Takes no query: this is standing context. See _load_priority_memories.

    Returns empty string if no memories qualify or DB is unavailable.
    """
    body, _spent = render_background(budget_tokens, exclude)
    return "\n\n".join(p for p in (_pinned_side_blocks(), body) if p)


def _pinned_side_blocks() -> str:
    """Recent outputs and active attention.

    Pinned like conversation-state, not budgeted: a verbatim mirror of what
    Luke just sent and the commitments he is holding are not optional context,
    and they are bounded by their own limits already.
    """
    parts: list[str] = []
    if settings.recent_outputs_enabled:
        block = _build_recent_outputs_block(settings.chat_id, settings.recent_outputs_limit)
        if block:
            parts.append(block)
    if settings.chat_id:
        try:
            attn = _attention_module.build_attention_block(settings.chat_id)
            if attn:
                parts.append(attn)
        except Exception as e:
            log.warning("attention_load_failed", error=str(e))
    return "\n\n".join(parts)


def render_background(budget_tokens: int, exclude: set[str] | None = None) -> tuple[str, int]:
    """Render standing memory only — no pinned side blocks. Returns (block, tokens).

    Separate from build_working_context so the assembler can charge the budget
    for exactly what the budget governs. Bundling the pinned blocks in made
    `spent` overshoot by ~1,100 tokens and look like a permanent overrun.
    """
    try:
        memories = _load_priority_memories()
    except Exception as e:
        log.warning("context_load_failed", error=str(e))
        return "", 0

    if exclude:
        # Already rendered in full by the turn layer. A 500-char preview of
        # something the agent can read at 1,200 chars two blocks up is pure
        # duplication — and nothing prevented it before the two layers shared
        # a decision point.
        memories = [m for m in memories if m["id"] not in exclude]

    if not memories:
        return "", 0

    # Reserve the scaffolding — section headers plus the stats comment — so the
    # returned block honours the budget rather than overshooting it by whatever
    # the headers happen to cost.
    by_type, spent = _spend(memories, max(0, budget_tokens - _SCAFFOLD_TOKENS))
    if not by_type:
        return "", 0

    sections: list[str] = ["# Injected Working Memory"]

    for mem_type in _SECTION_ORDER:
        lines = by_type.get(mem_type)
        if lines:
            sections.append(f"{_SECTION_TITLES[mem_type]}\n" + "\n".join(lines))

    counts = {t: len(v) for t, v in by_type.items()}
    injected = sum(counts.values())
    # Report what was RENDERED, not what was considered. The old counter said
    # "95 memories injected" for a block that emitted 21 of them.
    sections.append(
        f"\n<!-- context: {injected} memories, ~{spent} tokens "
        f"of {budget_tokens} budget, {counts} -->"
    )

    log.info(
        "context_injected",
        memories=injected,
        tokens=spent,
        budget=budget_tokens,
        by_type=counts,
    )

    block = "\n\n".join(sections)
    return block, _estimate_tokens(block)


def build_preservation_manifest() -> str:
    """Build a structured preservation manifest for PreCompact hook.

    Returns a message listing specific memory IDs, goal statuses,
    and entity facts that MUST survive context compaction.
    """
    try:
        db = _db()
    except Exception:
        return _FALLBACK_PRESERVATION

    sections: list[str] = ["CRITICAL — structured preservation manifest for compaction:"]

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

    # Top entities. No absolute importance bar: "the 15 most important entities"
    # is what this wants, and LIMIT already says it. A fixed threshold could
    # only ever fail in one direction — silently emptying the section if the
    # corpus rescaled — while ORDER BY says the same thing scale-independently.
    entity_rows = db.execute(
        """SELECT m.id, f.title
           FROM memory_meta m
           JOIN memory_fts f ON m.id = f.id
           WHERE m.type = 'entity' AND m.status = 'active'
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
        "4. Keep relationship links between memories"
    )

    # Load constitutional invariants dynamically from YAML
    sections.append(format_constitutional_summary())

    return "\n\n".join(sections)


_FALLBACK_PRESERVATION = (
    "CRITICAL — preserve in your compaction summary:\n"
    "1. All memory IDs you've referenced or created\n"
    "2. The user's most recent request and pending actions\n"
    "3. Active goals and their current status\n"
    "4. Key facts about the user from injected memories\n"
    "5. Any tool results not yet communicated\n"
    "6. Relationship links between memories\n"
    "\n" + _FALLBACK_CONSTITUTIONAL
)


# ---------------------------------------------------------------------------
# Compression audit — detect information loss during summarization
# ---------------------------------------------------------------------------


def audit_compression(
    compressed_text: str,
    goal_ids: list[str] | None = None,
    entity_ids: list[str] | None = None,
    memory_ids: list[str] | None = None,
    messages_compressed: int = 0,
    messages_kept: int = 0,
    persist: bool = True,
) -> dict[str, Any]:
    """Audit a compressed summary for information retention.

    Checks whether expected references (goals, entities, memory IDs) survived
    compression. Computes a retention score (0.0-1.0) and optionally logs the
    result to the compression_audit table.

    Args:
        compressed_text: The post-compression summary text.
        goal_ids: Active goal IDs that should be preserved.
        entity_ids: High-importance entity IDs that should be preserved.
        memory_ids: Any memory IDs that were referenced pre-compression.
        messages_compressed: Number of messages that were compressed.
        messages_kept: Number of messages kept verbatim.
        persist: If True, log the audit result to the DB.

    Returns:
        Dict with retention metrics and any missing references.
    """
    goal_ids = goal_ids or []
    entity_ids = entity_ids or []
    memory_ids = memory_ids or []

    text_lower = compressed_text.lower()

    # Check which expected references survived
    goals_preserved = [gid for gid in goal_ids if gid.lower() in text_lower]
    entities_preserved = [eid for eid in entity_ids if eid.lower() in text_lower]
    memory_ids_preserved = [mid for mid in memory_ids if mid.lower() in text_lower]

    # Check identity anchor presence
    constitutional = load_constitutional()
    identity_name = constitutional.get("identity", {}).get("name", "Luke")
    has_identity = identity_name.lower() in text_lower

    # Compute retention score: weighted average of preservation rates
    # Goals weighted 40%, entities 30%, memory IDs 20%, identity 10%
    scores: list[tuple[float, float]] = []  # (weight, rate)
    if goal_ids:
        scores.append((0.4, len(goals_preserved) / len(goal_ids)))
    if entity_ids:
        scores.append((0.3, len(entities_preserved) / len(entity_ids)))
    if memory_ids:
        scores.append((0.2, len(memory_ids_preserved) / len(memory_ids)))
    scores.append((0.1, 1.0 if has_identity else 0.0))

    if scores:
        total_weight = sum(w for w, _ in scores)
        retention_score = sum(w * r for w, r in scores) / total_weight if total_weight > 0 else 0.0
    else:
        retention_score = 1.0 if has_identity else 0.0

    summary_tokens = _estimate_tokens(compressed_text)

    result: dict[str, Any] = {
        "goals_expected": len(goal_ids),
        "goals_preserved": len(goals_preserved),
        "goals_missing": [g for g in goal_ids if g not in goals_preserved],
        "entities_expected": len(entity_ids),
        "entities_preserved": len(entities_preserved),
        "entities_missing": [e for e in entity_ids if e not in entities_preserved],
        "memory_ids_expected": len(memory_ids),
        "memory_ids_preserved": len(memory_ids_preserved),
        "memory_ids_missing": [m for m in memory_ids if m not in memory_ids_preserved],
        "identity_anchor": has_identity,
        "retention_score": round(retention_score, 3),
        "summary_tokens": summary_tokens,
        "messages_compressed": messages_compressed,
        "messages_kept": messages_kept,
    }

    if retention_score < 0.8:
        log.warning(
            "compression_audit_low_retention",
            retention_score=retention_score,
            goals_missing=result["goals_missing"],
            entities_missing=result["entities_missing"],
        )

    # Persist to DB
    if persist:
        try:
            db = _db()
            db.execute(
                """INSERT INTO compression_audit
                   (messages_compressed, messages_kept, goals_expected, goals_preserved,
                    entities_expected, entities_preserved, memory_ids_expected,
                    memory_ids_preserved, retention_score, summary_tokens, identity_anchor)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    messages_compressed,
                    messages_kept,
                    len(goal_ids),
                    len(goals_preserved),
                    len(entity_ids),
                    len(entities_preserved),
                    len(memory_ids),
                    len(memory_ids_preserved),
                    retention_score,
                    summary_tokens,
                    int(has_identity),
                ),
            )
            db.commit()
        except Exception as e:
            # Roll back so a failed write releases the process-wide write lock.
            with suppress(Exception):
                _db().rollback()
            log.warning("compression_audit_persist_failed", error=str(e))

    return result


# ---------------------------------------------------------------------------
# Unified context assembly
#
# Both memory layers used to be built independently and neither could see the
# other: app.process built the turn prefix, run_agent built the system block,
# and by the time run_agent ran the turn prefix was already baked into the
# prompt. So nothing could dedup them, nothing shared a budget, and
# conversation-state was injected twice on 61% of turns while also burning one
# of only 8 recall slots.
#
# assemble_context is now the single decision point. It emits two blocks
# because they have genuinely different lifetimes, not because two callers
# happened to build them:
#   * system_block — standing state, replaced every run, never accumulates
#   * turn_block   — this turn's evidence, correctly part of the transcript
# ---------------------------------------------------------------------------

_CONV_STATE_ID = "conversation-state-latest"
_STALE_HOURS = 24.0
# Read cap for the raw conversation-state file before tail-trimming.
# _save_conv_state can emit ~8.3k chars, so this must clear that comfortably or
# the trim would operate on already-truncated text.
_CONV_STATE_READ_MAX = 20_000

# Body chars per recalled memory in the turn block. Eight hits at the old 3,000
# was ~6.9k tokens, which blew every effort tier on its own. This is the gist;
# the agent has `recall` and `Read` for the full file.
_TURN_BODY_CHARS = 1_200

# Share of the budget the turn layer may take, so a run of long recall hits can
# never starve standing context to nothing.
_TURN_BUDGET_SHARE = 0.6

# Per-type ceilings on turn evidence. Procedures were 28% of the corpus but
# took 64% of every injected set, because they are numerous, high-access and
# were minted at the importance ceiling. Trigger-matched skills are exempt: a
# trigger match is an explicit "this procedure is the answer" signal, so it
# should not have to fight the cap that exists to stop incidental ones.
_TURN_TYPE_CAP: dict[str, int] = {"procedure": 3}

# Trigger-matched skills admitted per turn, regardless of type caps.
_MAX_TRIGGER_SKILLS = 2

_TRIVIAL_WORDS = frozenset(
    {
        "ok", "okay", "yes", "no", "yeah", "yep", "nope", "sure", "thanks",
        "thank", "lol", "haha", "hehe", "wow", "cool", "nice", "great", "hi",
        "hey", "hello", "bye", "goodnight", "gn", "gm",
    }
)  # fmt: skip

# A conversation-state message line: "**Filipe Lima** (2026-08-03T09:41): ..."
_CONV_MSG_LINE = re.compile(r"^\*\*[^*]+\*\* \(")

# Chats whose SDK session was lost; the next assembly tells the agent so it can
# resume deliberately instead of acting confused.
_session_lost: dict[str, bool] = {}


def note_session_reset(chat_id: str) -> None:
    """Record that this chat's agent session was lost."""
    _session_lost[chat_id] = True


def needs_recall(text: str) -> bool:
    """Heuristic: skip retrieval for trivial/short messages."""
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    words = stripped.lower().split()
    return not (len(words) <= 2 and all(w.strip("!?.,:") in _TRIVIAL_WORDS for w in words))


def _trim_conv_state(body: str, limit: int) -> str:
    """Trim conversation state to *limit* chars, keeping the NEWEST exchange.

    _save_conv_state writes the thread chronologically — a structured header,
    then messages oldest-first, with the latest reply appended last. A plain
    ``body[:limit]`` therefore drops the most recent turn and cuts mid-sentence,
    which is precisely backwards for a block whose entire job is letting the
    conversation resume seamlessly.

    Keeps the header (topics, last-active, pending actions) and then as many
    trailing message lines as fit.
    """
    if len(body) <= limit:
        return body

    lines = body.split("\n")
    first_msg = next((i for i, ln in enumerate(lines) if _CONV_MSG_LINE.match(ln)), len(lines))
    header, messages = lines[:first_msg], lines[first_msg:]

    header_text = "\n".join(header)
    room = limit - len(header_text) - 1
    if room <= 0:
        return body[-limit:]  # pathological header — keep the newest text we can

    kept: list[str] = []
    used = 0
    for line in reversed(messages):
        cost = len(line) + 1
        if used + cost > room:
            break
        kept.insert(0, line)
        used += cost

    if not kept:
        return (header_text + "\n" + messages[-1])[-limit:] if messages else header_text[:limit]
    return "\n".join([*header, *kept])


def _live_reaction_note(chat_id: str) -> str:
    """Surface reactions from the last few minutes so a fresh reaction shapes
    the next response while it's warm — the 'presence' gap Filipe named:
    capture was never the problem, noticing in the moment was. Returns '' when
    nothing is fresh, so it self-expires and never nags."""
    reactions = _db_module.get_recent_reactions(chat_id, within_minutes=15)
    if not reactions:
        return ""
    lines = []
    for r in reactions[:5]:
        is_own = r.get("msg_sender") == settings.assistant_name
        target = "your message" if is_own else "their own message"
        preview = (r.get("msg_preview") or "").replace("\n", " ").strip()
        snippet = f' "{preview}…"' if preview else ""
        lines.append(f"- {r['emoji']} ({r['sentiment']}) on {target}{snippet}")
    return (
        "[LIVE — Filipe just reacted (last 15 min). Acknowledge it naturally if it "
        "fits; let it shape your tone. Don't over-perform it.]\n" + "\n".join(lines) + "\n\n"
    )


def _pin_conversation_state(chat_id: str) -> str:
    """The continuity anchor. Pinned, never ranked, never charged against the
    memory budget — resuming the conversation is not optional context."""
    body = _memory_module.read_memory_body("episode", _CONV_STATE_ID, _CONV_STATE_READ_MAX)
    if body:
        body = _trim_conv_state(body, settings.recall_content_limit)
    updated = _memory_module.get_memory_updated(_CONV_STATE_ID) if body else None
    live_reactions = _live_reaction_note(chat_id)

    if body:
        if _session_lost.pop(chat_id, False):
            body = "[Session was reset — use this context to resume seamlessly]\n" + body
        if live_reactions:
            body = live_reactions + body
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

    # Fallback: synthesize from recent messages.
    recent = _db_module.get_recent_messages(chat_id, limit=20)
    if not recent:
        return ""
    lines = [f"{m['sender_name']}: {m['content'][:500]}" for m in recent[-10:]]
    return (
        "<conversation-state>\n"
        f"{live_reactions}"
        "[Recent conversation context (no saved state available)]\n"
        f"{chr(10).join(lines)}\n</conversation-state>\n"
    )


def _age_label(updated_iso: str) -> str:
    """Human-readable age, e.g. "today", "3d ago", "4 months ago".

    The ranker decays by recency but the model could not see it: every injected
    memory rendered identically, so a March episode read as current fact. This
    is what lets it say "as of May" instead of asserting stale state.
    """
    if not updated_iso:
        return ""
    try:
        updated = ensure_utc(datetime.fromisoformat(updated_iso))
    except ValueError, AttributeError:
        return ""
    days = (datetime.now(UTC) - updated).total_seconds() / 86400
    if days < 1:
        return "today"
    if days < 2:
        return "yesterday"
    if days < 14:
        return f"{int(days)}d ago"
    if days < 60:
        return f"{int(days / 7)} weeks ago"
    return f"{int(days / 30)} months ago"


def _rank_turn_candidates(query: str, exclude: set[str]) -> list[dict[str, Any]]:
    """Query-ranked evidence for this turn: recall hits, guaranteed skills,
    ranked graph neighbours — deduped against *exclude* and against each other.

    *exclude* is read, never written: the caller owns the running set and
    updates it from the returned candidates. It contains the pinned
    conversation-state id, which is why that memory can no longer consume one
    of the recall slots — it did on 61% of turns, while also being rendered a
    second time in full.
    """
    seen = set(exclude)
    memories, trigger_skills = (
        _memory_module.recall(query=query, limit=settings.auto_recall_limit),
        _memory_module.get_trigger_matched_skills(query),
    )

    candidates: list[dict[str, Any]] = []
    per_type: dict[str, int] = {}

    def admit(m: Mapping[str, Any], source: str, *, exempt: bool = False) -> None:
        """Admit a candidate unless its type is full.

        *exempt* skips the check for THIS memory but still counts it toward the
        type's tally. So a trigger-matched skill is never blocked, while total
        procedure share stays bounded — exempting them from the count too would
        let a turn carry cap + skills procedures and defeat the cap.
        """
        if m["id"] in seen:
            return
        mem_type = m["type"]
        cap = _TURN_TYPE_CAP.get(mem_type)
        if not exempt and cap is not None and per_type.get(mem_type, 0) >= cap:
            return
        seen.add(m["id"])
        per_type[mem_type] = per_type.get(mem_type, 0) + 1
        candidates.append({**m, "source": source})

    # Trigger-matched skills go first and are exempt from the type cap: a
    # trigger match is already the authoritative "this procedure is the right
    # answer" signal, so it does not need to win on score too. This replaces the
    # old displace-the-lowest-scoring-non-skill dance, which also had to probe
    # frontmatter from disk for every candidate to decide what counted as a
    # skill — work a trigger match already answers.
    for skill in trigger_skills[:_MAX_TRIGGER_SKILLS]:
        admit(skill, "skill", exempt=True)

    for m in memories:
        admit(m, "recall")

    if candidates:
        neighbours = _memory_module.get_graph_neighbors([c["id"] for c in candidates], limit=3)
        for n in neighbours:
            admit(n, "neighbor")

    return candidates


def _render_turn_block(
    candidates: list[dict[str, Any]], budget_tokens: int
) -> tuple[str, int, list[str]]:
    """Render turn evidence, charging each entry what it prints.

    Returns the block, the tokens spent, and the ids that ACTUALLY rendered.
    That last one matters: a candidate the budget rejected was never shown to
    the model, so it must not earn an exposure touch and must not be excluded
    from the standing layer. Retrieval routinely produces far more candidates
    than fit — 42 for a single real query — so the gap is not marginal.
    """
    lines: list[str] = []
    rendered: list[str] = []
    used = 0
    for c in candidates:
        # Over-read then truncate, so the cut lands on a word boundary and
        # announces the gap instead of amputating mid-word.
        raw = _memory_module.read_memory_body(c["type"], c["id"], _TURN_BODY_CHARS * 3)
        body = _truncate(raw, _TURN_BODY_CHARS)
        age = _age_label(c.get("updated", ""))
        label = f"{c['type']}, {age}" if age else c["type"]
        line = f"[{c['id']}] ({label}) {body or c.get('title', '')}"
        cost = _estimate_tokens(line)
        if used + cost > budget_tokens:
            continue
        lines.append(line)
        rendered.append(c["id"])
        used += cost
    if not lines:
        return "", 0, []
    block = "<context><memories>\n" + "\n---\n".join(lines) + "\n</memories></context>"
    return block, used, rendered


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """Everything one turn injects, decided in one place."""

    system_block: str = ""  # standing state -> system prompt, replaced per run
    turn_block: str = ""  # this turn's evidence -> user prompt
    ids: list[str] = field(default_factory=list)  # everything RENDERED
    recalled_ids: list[str] = field(default_factory=list)  # query-ranked subset
    tokens: int = 0


def _assemble(
    *, query: str, chat_id: str, budget_tokens: int, turn_scoped: bool
) -> AssembledContext:
    """Synchronous body of assemble_context. Runs entirely in one worker thread."""
    seen: set[str] = {_CONV_STATE_ID}

    pinned_parts: list[str] = []
    if chat_id:
        try:
            state = _pin_conversation_state(chat_id)
            if state:
                pinned_parts.append(state)
        except Exception as e:
            log.warning("conversation_state_failed", error=str(e))
    try:
        side = _pinned_side_blocks()
        if side:
            pinned_parts.append(side)
    except Exception as e:
        log.warning("pinned_side_blocks_failed", error=str(e))
    pinned = "\n\n".join(pinned_parts)

    turn_block = ""
    recalled_ids: list[str] = []
    turn_tokens = 0
    if turn_scoped and query and needs_recall(query):
        try:
            candidates = _rank_turn_candidates(query, seen)
            turn_block, turn_tokens, recalled_ids = _render_turn_block(
                candidates, int(budget_tokens * _TURN_BUDGET_SHARE)
            )
            # The caller owns `seen`; the turn layer only reads it. Updating
            # here is what makes the standing layer's exclusion hold.
            seen.update(recalled_ids)
            # Speculative touch: exposure without evidence of use. Deliberately
            # scoped to the turn layer. Doing the same for standing context
            # would be a closed loop — injecting a memory because it ranks, then
            # raising its rank because it was injected. Standing memories earn
            # credit only via the reference scan (useful_only), which raises
            # useful_count without raising access_count.
            if recalled_ids:
                _memory_module.touch_memories(recalled_ids, useful=False)
        except Exception as e:
            log.warning("turn_recall_failed", error=str(e))

    # Standing context spends what the turn layer left. Memories already
    # rendered in full above are excluded — a 500-char preview of something the
    # agent can already read at 1,200 chars is pure duplication.
    background, background_tokens = "", 0
    try:
        background, background_tokens = render_background(
            max(0, budget_tokens - turn_tokens), exclude=seen
        )
    except Exception as e:
        log.warning("background_context_failed", error=str(e))

    system_block = "\n\n".join(p for p in (pinned, background) if p)
    rendered = _rendered_ids(background) + recalled_ids

    # Two numbers, deliberately. `spent` is what the budget actually governs —
    # turn evidence plus standing memory — and must stay under it. `pinned` is
    # the continuity anchor, attention and recent-outputs: not optional, so not
    # budgeted. Reporting only the sum would look like a permanent overrun.
    pinned_tokens = _estimate_tokens(pinned)
    spent = turn_tokens + background_tokens

    log.info(
        "context_assembled",
        chat_id=chat_id,
        budget=budget_tokens,
        spent=spent,
        pinned_tokens=pinned_tokens,
        turn_tokens=turn_tokens,
        recalled=len(recalled_ids),
        rendered=len(rendered),
    )
    total = spent + pinned_tokens
    return AssembledContext(
        system_block=system_block,
        turn_block=turn_block,
        ids=rendered,
        recalled_ids=recalled_ids,
        tokens=total,
    )


_ID_IN_LINE = re.compile(r"^\s*\[([^\]]+)\]", re.MULTILINE)


def _rendered_ids(block: str) -> list[str]:
    """Memory ids actually present in a rendered block."""
    return _ID_IN_LINE.findall(block)


async def assemble_context(
    *,
    query: str,
    chat_id: str,
    budget_tokens: int,
    turn_scoped: bool = True,
) -> AssembledContext:
    """Assemble every memory layer under one budget. Never raises.

    One to_thread hop: the DB scan, file reads and any embedding all happen off
    the event loop, so a slow corpus or a hung embed server cannot stall other
    chats. Returning an empty context on failure is deliberate — memory is an
    enhancement, and losing it must never cost the caller its prompt.
    """
    try:
        return await asyncio.to_thread(
            _assemble,
            query=query,
            chat_id=chat_id,
            budget_tokens=budget_tokens,
            turn_scoped=turn_scoped,
        )
    except Exception as e:
        log.warning("context_assembly_failed", error=str(e))
        return AssembledContext()
