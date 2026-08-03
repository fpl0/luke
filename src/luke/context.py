"""Context engineering: working memory injection and preservation manifests.

Provides runtime context management for the agent:
- build_working_context(): scores and selects priority memories for system prompt injection
- build_preservation_manifest(): structured list of what must survive context compaction
- load_constitutional(): loads behavioral invariants from constitutional.yaml
"""

from __future__ import annotations

import math
import struct
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, NamedTuple

import structlog
import yaml
from structlog.stdlib import BoundLogger

from . import attention as _attention_module
from . import db as _db_module
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


def _render_line(memory: dict[str, Any], spec: RenderSpec) -> str:
    """The exact text this memory contributes. The only place cost is defined."""
    value = memory["title"] if spec.field == "title" else memory["content"]
    return f"  [{memory['id']}] {(value or '')[: spec.chars]}"


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


def _cosine_similarity(a: list[float], b_blob: bytes) -> float:
    """Cosine similarity between a list[float] and a packed-float blob."""
    dim = len(a)
    try:
        b: tuple[float, ...] = struct.unpack(f"{dim}f", b_blob)
    except struct.error:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _load_priority_memories(query: str = "") -> list[dict[str, Any]]:
    """Load active memories scored for context injection priority.

    When *query* is non-empty, embeds it and adds semantic similarity as a
    scoring factor (25% weight). Falls back to static scoring when empty.

    Returns dicts with: id, type, title, content, importance, updated,
    access_count, score.
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

    # --- Query-aware scoring: embed query and load memory vectors ---
    query_vec: list[float] | None = None
    vec_map: dict[str, bytes] = {}  # mem_id → packed embedding blob
    if query.strip():
        try:
            from .memory import _embed_query

            query_vec = _embed_query(query)
        except Exception:
            log.warning("context_query_embed_failed")
            query_vec = None

    if query_vec is not None:
        id_set = [r["id"] for r in rows]
        if id_set:
            placeholders = ",".join("?" for _ in id_set)
            vec_rows = db.execute(
                f"SELECT memory_id, embedding FROM memory_vec WHERE memory_id IN ({placeholders})",
                id_set,
            ).fetchall()
            for vr in vec_rows:
                vec_map[vr["memory_id"]] = vr["embedding"]

    has_query = query_vec is not None and len(vec_map) > 0

    max_access = max((r["access_count"] for r in rows), default=1) or 1
    log_max = math.log1p(max_access)

    memories: list[dict[str, Any]] = []
    for r in rows:
        imp = importance_score(r["importance"])
        rec = _recency_score(r["updated"])
        freq = math.log1p(r["access_count"]) / log_max if log_max > 0 else 0.5

        if has_query:
            # Query-aware: importance 30%, recency 25%, access 20%, similarity 25%
            sim = 0.0
            blob = vec_map.get(r["id"])
            if blob is not None and query_vec is not None:
                sim = max(0.0, _cosine_similarity(query_vec, blob))
            score = 0.30 * imp + 0.25 * rec + 0.20 * freq + 0.25 * sim
        else:
            # Static fallback: importance 40%, recency 35%, access 25%
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
    query: str = "",
    budget_tokens: int = _WORKING_MEMORY_BUDGET,
) -> str:
    """Build a working memory block for system prompt injection.

    Scores all active memories by importance/recency/access, selects
    the top ones within token budget, and formats them as a structured
    context block the agent can reference.

    Returns empty string if no memories qualify or DB is unavailable.
    """
    # Recent outputs (L3) — verbatim mirror of Luke's own recent sends.
    # Built first so we can prepend it even if memory selection fails.
    recent_outputs: str | None = None
    if settings.recent_outputs_enabled:
        recent_outputs = _build_recent_outputs_block(
            settings.chat_id, settings.recent_outputs_limit
        )

    # Active attention (L2) — persistent foreground commitments.
    attn_block: str | None = None
    if settings.chat_id:
        try:
            attn_block = _attention_module.build_attention_block(settings.chat_id)
        except Exception as e:
            log.warning("attention_load_failed", error=str(e))

    def _prepended(body: str | None) -> str:
        """Combine recent-outputs + attention with a base body string."""
        parts = [p for p in (recent_outputs, attn_block, body) if p]
        return "\n\n".join(parts)

    try:
        memories = _load_priority_memories(query=query)
    except Exception as e:
        log.warning("context_load_failed", error=str(e))
        return _prepended(None)

    if not memories:
        return _prepended(None)

    by_type, spent = _spend(memories, budget_tokens)
    if not by_type:
        return _prepended(None)

    sections: list[str] = []
    # Verbatim recent outputs go above the reconstructed working memory so the
    # agent sees its actual prior sends before any compressed/summarized state.
    if recent_outputs:
        sections.append(recent_outputs)
    # Active attention (L2) sits above memory injection so foreground
    # commitments have visual priority over reconstructed working memory.
    if attn_block:
        sections.append(attn_block)
    sections.append("# Injected Working Memory")

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
