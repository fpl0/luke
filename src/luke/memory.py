"""Memory subsystem: FTS5 indexing, semantic search, graph traversal, scoring, decay."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import sqlite3
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import structlog
import yaml
from structlog.stdlib import BoundLogger

from .config import settings
from .db import _commit, _db, batch, ensure_utc

log: BoundLogger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Constants & types
# ---------------------------------------------------------------------------

MEMORY_DIRS: dict[str, str] = {
    "entity": "entities",
    "episode": "episodes",
    "procedure": "procedures",
    "insight": "insights",
    "goal": "goals",
}
_DIR_TO_TYPE: dict[str, str] = {v: k for k, v in MEMORY_DIRS.items()}

# Taxonomy classification: factual, experiential, working
TAXONOMY_VALUES: frozenset[str] = frozenset({"factual", "experiential", "working"})

# Default taxonomy per memory type — used when caller doesn't specify
_DEFAULT_TAXONOMY: dict[str, str] = {
    "entity": "factual",
    "procedure": "factual",
    "insight": "factual",
    "episode": "experiential",
    "goal": "working",
}

# Causal labels — prioritized in graph traversal for "why" queries
CAUSAL_RELATIONSHIPS: frozenset[str] = frozenset(
    {"caused", "derived_from", "supersedes", "contradicts", "supports", "blocked_by", "enables"}
)

# All standard relationship labels — built from causal + contextual to avoid maintaining two lists
RELATIONSHIP_LABELS: frozenset[str] = CAUSAL_RELATIONSHIPS | frozenset(
    {"related", "involves", "contributes_to", "uses", "about", "informed_by"}
)

# Default label based on (from_type, to_type) pair — fallback is "related"
_DEFAULT_RELATIONSHIP: dict[tuple[str, str], str] = {
    ("episode", "entity"): "involves",
    ("episode", "goal"): "contributes_to",
    ("episode", "insight"): "derived_from",
    ("episode", "procedure"): "uses",
    ("insight", "entity"): "about",
    ("insight", "goal"): "about",
    ("insight", "insight"): "supports",
    ("goal", "entity"): "involves",
    ("goal", "insight"): "informed_by",
    ("procedure", "entity"): "about",
}

_CAUSAL_QUERY_RE = re.compile(
    r"\b(why|because|reason|cause[ds]?|led to|result of|due to|motivated)\b", re.IGNORECASE
)

assert CAUSAL_RELATIONSHIPS <= RELATIONSHIP_LABELS, "causal labels must be a subset of all labels"


def _valid_link_clause(alias: str = "") -> str:
    """SQL clause for filtering active (non-expired) links."""
    prefix = f"{alias}." if alias else ""
    return f"AND ({prefix}valid_until IS NULL OR {prefix}valid_until = '')"


# Insight consolidation thresholds — domain constants, not user settings
_INSIGHT_SIMILARITY_THRESHOLD = 0.65
_INSIGHT_MIN_CLUSTER = 3

# Lifecycle review thresholds
_STALE_ENTITY_DAYS = 90
_UNUSED_PROCEDURE_DAYS = 60


class MemoryResult(TypedDict):
    id: str
    type: str
    title: str
    score: float
    relationship: NotRequired[str]
    created: NotRequired[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_memory_id(raw_id: str) -> str:
    """Sanitize a memory ID for filesystem and index use."""
    return re.sub(r"[^\w-]", "_", raw_id).strip("_-")


def _find_frontmatter_end(text: str) -> int:
    """Find the closing --- delimiter of YAML frontmatter. Returns -1 if not found."""
    # Prefer \n---\n (own line); fall back to \n--- at EOF
    end = text.find("\n---\n", 3)
    if end != -1:
        return end
    end = text.find("\n---", 3)
    if end != -1 and end + 4 >= len(text):
        return end  # --- at very end of file
    return -1


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (---...---) from markdown text."""
    if text.startswith("---"):
        end = _find_frontmatter_end(text)
        if end != -1:
            # Skip past \n---\n (5 chars) or \n--- at EOF (4 chars)
            skip = 5 if text[end : end + 5] == "\n---\n" else 4
            return text[end + skip :].strip()
    return text


def read_memory_body(mem_type: str, mem_id: str, limit: int = 2000) -> str:
    """Read a memory file's body (sans frontmatter), truncated to *limit* chars."""
    type_dir = MEMORY_DIRS.get(mem_type, f"{mem_type}s")
    path = settings.memory_dir / type_dir / f"{mem_id}.md"
    try:
        return strip_frontmatter(path.read_text())[:limit]
    except FileNotFoundError:
        return ""


def get_memory_updated(mem_id: str) -> str | None:
    """Return the 'updated' ISO timestamp for a memory, or None if not found."""
    row = _db().execute("SELECT updated FROM memory_meta WHERE id = ?", (mem_id,)).fetchone()
    return str(row["updated"]) if row else None


def read_frontmatter(path: Path) -> dict[str, Any]:
    """Read YAML frontmatter from a markdown file. Returns {} if absent or malformed."""
    try:
        text = path.read_text()
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    end = _find_frontmatter_end(text)
    if end == -1:
        return {}
    result: dict[str, Any] = yaml.safe_load(text[3:end]) or {}
    return result


# ---------------------------------------------------------------------------
# Embeddings (fastembed + sqlite-vec)
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
_embedder: Any = None


def _get_embedder() -> Any:
    """Lazy-init fastembed TextEmbedding. Lazy because model load is heavy (~1s)."""
    global _embedder
    if _embedder is not None:
        return _embedder
    from fastembed import TextEmbedding

    _embedder = TextEmbedding(model_name=_EMBEDDING_MODEL)
    log.info("embedder_loaded")
    return _embedder


def _embed_passage(text: str) -> list[float] | None:
    """Embed a document/memory for storage. Uses passage prefix for asymmetric retrieval."""
    try:
        results = list(_get_embedder().passage_embed([text]))
        return [float(x) for x in results[0]]
    except Exception:
        log.warning("embedding_failed", func="passage")
        return None


def _embed_query(text: str) -> list[float] | None:
    """Embed a search query. Uses query prefix for asymmetric retrieval."""
    try:
        results = list(_get_embedder().query_embed(text))
        return [float(x) for x in results[0]]
    except Exception:
        log.warning("embedding_failed", func="query")
        return None


def _semantic_search(
    query_embedding: list[float],
    *,
    mem_type: str | None = None,
    limit: int = 20,
    include_private: bool = False,
) -> list[dict[str, Any]]:
    """KNN search using sqlite-vec — native C distance computation."""
    db = _db()
    query_blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)
    # vec0 KNN: auxiliary columns can be SELECTed but not filtered in WHERE.
    # Over-fetch 4x when type-filtered (more rows eliminated), 2x otherwise.
    fetch_limit = limit * (4 if mem_type else 2)
    knn_rows = db.execute(
        "SELECT memory_id, distance FROM memory_vec WHERE embedding MATCH ? AND k = ?",
        (query_blob, fetch_limit),
    ).fetchall()
    if not knn_rows:
        return []

    # Build memory_id → distance map, then enrich via memory_meta + memory_fts
    mid_dist = {r["memory_id"]: r["distance"] for r in knn_rows}
    conditions = ["m.status = 'active'"]
    if not include_private:
        conditions.append("m.is_private = 0")
    if mem_type:
        conditions.append("m.type = ?")
    filter_clause = " AND ".join(conditions)
    mid_placeholders = ",".join("?" for _ in mid_dist)
    params: list[Any] = list(mid_dist.keys())
    if mem_type:
        params.append(mem_type)

    meta_rows = db.execute(
        f"""SELECT m.id, m.type, f.title,
                   m.importance, m.access_count, m.useful_count, m.updated,
                   m.taxonomy
            FROM memory_meta m
            JOIN memory_fts f ON m.id = f.id
            WHERE m.id IN ({mid_placeholders}) AND {filter_clause}""",
        params,
    ).fetchall()

    scored: list[dict[str, Any]] = []
    for r in meta_rows:
        dist = mid_dist.get(r["id"], 0.0)
        similarity = 1.0 / (1.0 + dist)
        scored.append(
            {
                "id": r["id"],
                "type": r["type"],
                "title": r["title"],
                "score": similarity,
                "importance": r["importance"],
                "access_count": r["access_count"],
                "useful_count": r["useful_count"],
                "updated": r["updated"],
                "taxonomy": r["taxonomy"],
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def detect_changes(mem_id: str, new_content: str, new_title: str) -> list[str]:
    """Compare new content against existing memory. Return change descriptions."""
    conn = _db()
    row = conn.execute("SELECT content, title FROM memory_fts WHERE id = ?", (mem_id,)).fetchone()
    if not row:
        return []

    changes: list[str] = []
    if row["title"] != new_title:
        changes.append(f"Title: '{row['title']}' -> '{new_title}'")
    if row["content"] != new_content:
        old_preview = row["content"][:100]
        changes.append(f"Content updated (was: '{old_preview}...')")
    return changes


# ---------------------------------------------------------------------------
# Adaptive forgetting (spaced repetition)
# ---------------------------------------------------------------------------


def decay_importance(rates: dict[str, float]) -> int:
    """Apply type-aware importance decay modulated by access_count and taxonomy.

    Working-taxonomy memories decay 3x faster than their type's base rate.
    Factual-taxonomy memories decay at half the base rate (more stable).
    """
    db = _db()
    total = 0
    for mem_type, base_rate in rates.items():
        # Default decay for memories without taxonomy or experiential taxonomy
        cur = db.execute(
            """UPDATE memory_meta
               SET importance = importance * (1.0 - (1.0 - ?) / (1.0 + access_count * 0.1))
               WHERE type = ? AND status = 'active'
               AND taxonomy NOT IN ('factual', 'working')""",
            (base_rate, mem_type),
        )
        total += cur.rowcount
        # Factual: slower decay (half rate — stable knowledge)
        factual_rate = 1.0 - (1.0 - base_rate) * 0.5
        cur = db.execute(
            """UPDATE memory_meta
               SET importance = importance * (1.0 - (1.0 - ?) / (1.0 + access_count * 0.1))
               WHERE type = ? AND status = 'active' AND taxonomy = 'factual'""",
            (factual_rate, mem_type),
        )
        total += cur.rowcount
        # Working: faster decay (3x rate — ephemeral)
        working_rate = 1.0 - (1.0 - base_rate) * 3.0
        working_rate = max(working_rate, 0.5)  # floor at 0.5 to avoid instant zeroing
        cur = db.execute(
            """UPDATE memory_meta
               SET importance = importance * (1.0 - (1.0 - ?) / (1.0 + access_count * 0.1))
               WHERE type = ? AND status = 'active' AND taxonomy = 'working'""",
            (working_rate, mem_type),
        )
        total += cur.rowcount
    _commit(db)
    return total


# ---------------------------------------------------------------------------
# Temporal episodic clustering
# ---------------------------------------------------------------------------


def recall_by_time_window(
    *,
    after: str,
    before: str,
    limit: int = 50,
) -> list[MemoryResult]:
    """Retrieve all memories within a time window, ordered chronologically."""
    db = _db()
    rows = db.execute(
        """SELECT m.id, m.type, f.title, m.created, m.updated, m.importance
           FROM memory_meta m
           JOIN memory_fts f ON m.id = f.id
           WHERE m.status = 'active' AND m.created >= ? AND m.created <= ?
           ORDER BY m.created ASC
           LIMIT ?""",
        (after, before, limit),
    ).fetchall()
    return cast(
        list[MemoryResult],
        [dict(r) | {"score": 0.0} for r in rows],
    )


# ---------------------------------------------------------------------------
# Consolidation candidates
# ---------------------------------------------------------------------------


def get_consolidation_candidates(min_shared: int = 3) -> list[list[dict[str, Any]]]:
    """Find episode clusters sharing >= min_shared tags or >= 2 links."""
    db = _db()
    rows = db.execute(
        """SELECT id, tags_json, links_json, created, updated
           FROM memory_meta
           WHERE type = 'episode' AND status = 'active'
           ORDER BY created""",
    ).fetchall()

    if len(rows) < min_shared:
        return []

    episodes: list[dict[str, Any]] = [
        {
            "id": r["id"],
            "tags": set(json.loads(r["tags_json"])),
            "links": set(json.loads(r["links_json"])),
            "created": r["created"],
            "updated": r["updated"],
        }
        for r in rows
    ]

    clusters: list[list[dict[str, Any]]] = []
    used: set[str] = set()

    for i, ep in enumerate(episodes):
        if ep["id"] in used:
            continue
        cluster = [ep]
        for j in range(i + 1, len(episodes)):
            other = episodes[j]
            if other["id"] in used:
                continue
            shared_tags = len(ep["tags"] & other["tags"])
            shared_links = len(ep["links"] & other["links"])
            if shared_tags >= min_shared or shared_links >= 2:
                cluster.append(other)
        if len(cluster) >= min_shared:
            for c in cluster:
                used.add(c["id"])
            clusters.append(cluster)

    return clusters


def get_factual_duplicate_candidates(
    similarity_threshold: float = 0.7,
) -> list[list[dict[str, Any]]]:
    """Find factual-taxonomy memories with high semantic overlap for merging.

    Groups factual memories (entities, procedures, insights) that are semantically
    similar enough to warrant deduplication. Returns clusters of 2+ similar memories.
    """
    db = _db()
    rows = db.execute(
        """SELECT m.id, m.type, f.title
           FROM memory_meta m
           JOIN memory_fts f ON m.id = f.id
           WHERE m.taxonomy = 'factual' AND m.status = 'active'""",
    ).fetchall()

    if len(rows) < 2:
        return []

    # Build embedding map
    id_set = {r["id"] for r in rows}
    embedding_rows = db.execute(
        f"""SELECT memory_id, embedding FROM memory_vec
            WHERE memory_id IN ({",".join("?" for _ in id_set)})""",
        list(id_set),
    ).fetchall()

    embeddings: dict[str, list[float]] = {}
    for row in embedding_rows:
        if row["embedding"]:
            vec = list(struct.unpack(f"{len(row['embedding']) // 4}f", row["embedding"]))
            embeddings[row["memory_id"]] = vec

    # Find similar pairs via semantic search
    clusters: list[list[dict[str, Any]]] = []
    used: set[str] = set()
    meta_map = {r["id"]: dict(r) for r in rows}

    for mid in id_set:
        if mid in used or mid not in embeddings:
            continue
        neighbors = _semantic_search(embeddings[mid], mem_type=None, limit=6, include_private=True)
        cluster = [meta_map[mid]]
        for n in neighbors:
            nid = n["id"]
            eligible = nid != mid and nid in id_set and nid not in used
            if eligible and n["score"] >= similarity_threshold:
                cluster.append(meta_map[nid])
        if len(cluster) >= 2:
            for c in cluster:
                used.add(c["id"])
            clusters.append(cluster)

    return clusters


# ---------------------------------------------------------------------------
# Memory index
# ---------------------------------------------------------------------------


def _vec_rowid(mem_id: str) -> int:
    """Deterministic rowid for sqlite-vec from a memory ID (stable across restarts)."""
    digest = hashlib.sha256(mem_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def index_memory(
    mem_id: str,
    mem_type: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    links: list[str] | None = None,
    importance: float | None = None,
    taxonomy: str | None = None,
    skill_meta: dict[str, Any] | None = None,
) -> list[float] | None:
    # Frontmatter may store tags/links as JSON strings — parse them
    if isinstance(tags, str):
        tags = json.loads(tags)
    if isinstance(links, str):
        links = json.loads(links)
    tags = tags or []
    links = links or []
    now = datetime.now(UTC).isoformat()
    db = _db()

    # Upsert FTS
    db.execute("DELETE FROM memory_fts WHERE id = ?", (mem_id,))
    db.execute(
        "INSERT INTO memory_fts (id, type, title, content, tags) VALUES (?, ?, ?, ?, ?)",
        (mem_id, mem_type, title, content, " ".join(tags)),
    )

    # Resolve taxonomy: caller-provided > existing > default for type
    resolved_taxonomy = taxonomy or ""
    if resolved_taxonomy and resolved_taxonomy not in TAXONOMY_VALUES:
        log.warning("invalid_taxonomy", taxonomy=resolved_taxonomy, mem_id=mem_id)
        resolved_taxonomy = ""

    # Upsert metadata — read existing values BEFORE INSERT OR REPLACE deletes the row
    existing = db.execute(
        "SELECT created, access_count, useful_count, importance, "
        "last_accessed, taxonomy, skill_meta"
        " FROM memory_meta WHERE id = ?",
        (mem_id,),
    ).fetchone()
    created = existing["created"] if existing else now
    access_count = existing["access_count"] if existing else 0
    useful_count = existing["useful_count"] if existing else 0
    last_accessed: str = existing["last_accessed"] if existing else ""
    # Use caller-provided importance, then existing, then default 1.0
    if importance is None:
        importance = existing["importance"] if existing else 1.0
    # Taxonomy: caller > existing > default for type
    if not resolved_taxonomy:
        resolved_taxonomy = existing["taxonomy"] if existing and existing["taxonomy"] else ""
    if not resolved_taxonomy:
        resolved_taxonomy = _DEFAULT_TAXONOMY.get(mem_type, "")
    # Skill meta: caller-provided > existing > None
    resolved_skill_meta: str | None = None
    if skill_meta is not None:
        resolved_skill_meta = json.dumps(skill_meta)
    elif existing and existing["skill_meta"]:
        resolved_skill_meta = existing["skill_meta"]
    tags_json = json.dumps(tags)
    links_json = json.dumps(links)
    is_private = 1 if "private" in tags else 0
    db.execute(
        """INSERT OR REPLACE INTO memory_meta
           (id, type, taxonomy, created, updated,
            access_count, useful_count, importance, status,
            tags_json, links_json, is_private, last_accessed, skill_meta)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)""",
        (
            mem_id,
            mem_type,
            resolved_taxonomy,
            created,
            now,
            access_count,
            useful_count,
            importance,
            tags_json,
            links_json,
            is_private,
            last_accessed,
            resolved_skill_meta,
        ),
    )

    # Sync skill_triggers table when procedure has skill_meta with trigger_pattern
    if skill_meta and mem_type == "procedure" and skill_meta.get("trigger_pattern"):
        from .db import upsert_skill_trigger

        upsert_skill_trigger(
            procedure_id=mem_id,
            trigger_pattern=skill_meta["trigger_pattern"],
            confidence=skill_meta.get("confidence", 0.5),
            success_count=skill_meta.get("success_count", 0),
            failure_count=skill_meta.get("failure_count", 0),
        )

    # Update links — determine relationship from type pair
    if links:
        link_types: dict[str, str] = {}
        ph = ",".join("?" for _ in links)
        for row in db.execute(
            f"SELECT id, type FROM memory_meta WHERE id IN ({ph})", links
        ).fetchall():
            link_types[row["id"]] = row["type"]
        for link_id in links:
            if not link_id or not link_id[0].isalnum():
                continue  # skip corrupt IDs (e.g. from string iteration)
            target_type = link_types.get(link_id, "")
            rel = _DEFAULT_RELATIONSHIP.get((mem_type, target_type), "related")
            db.execute(
                """INSERT OR IGNORE INTO memory_links
                   (from_id, to_id, relationship, weight, created)
                   VALUES (?, ?, ?, 1.0, ?)""",
                (mem_id, link_id, rel, now),
            )

    _commit(db)

    # Embed AFTER committing the main writes — CPU-intensive, don't hold the lock
    embedding = _embed_passage(f"{title} {content}")
    if embedding is not None:
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        rowid = _vec_rowid(mem_id)
        db.execute("DELETE FROM memory_vec WHERE rowid = ?", (rowid,))
        db.execute(
            "INSERT INTO memory_vec (rowid, embedding, memory_id) VALUES (?, ?, ?)",
            (rowid, blob, mem_id),
        )
        _commit(db)

    # Assign to cluster (online BIRCH)
    assign_cluster_online(mem_id, embedding)

    # --- Memory evolution hook (A-MEM pattern) ---
    # When a new memory is indexed, auto-link to semantically similar existing
    # memories.  This builds the knowledge graph automatically instead of
    # relying on the agent to specify links manually every time.
    if embedding is not None:
        try:
            similar = _semantic_search(
                embedding, limit=4, include_private=True
            )
            for cand in similar:
                if cand["id"] == mem_id:
                    continue
                if cand["score"] < 0.40:  # too dissimilar
                    continue
                target_type_row = _db().execute(
                    "SELECT type FROM memory_meta WHERE id = ?", (cand["id"],)
                ).fetchone()
                target_type = target_type_row["type"] if target_type_row else ""
                rel = _DEFAULT_RELATIONSHIP.get(
                    (mem_type, target_type), "related"
                )
                _db().execute(
                    """INSERT OR IGNORE INTO memory_links
                       (from_id, to_id, relationship, weight, created)
                       VALUES (?, ?, ?, 1.0, ?)""",
                    (mem_id, cand["id"], rel, datetime.now(UTC).isoformat()),
                )
            _commit(_db())
        except Exception:
            log.warning("memory_evolution_failed", mem_id=mem_id)

    return embedding


_FTS_STRIP = re.compile(r'[*"^(){}\[\]:+\-~]')
_FTS_OPS = re.compile(r"\b(NOT|OR|AND|NEAR)\b", re.IGNORECASE)


def _sanitize_fts_query(query: str) -> str:
    """Convert user text to safe FTS5 OR query.

    Strips FTS5 operator characters and joins terms with OR so that longer
    queries return MORE results (ranked by BM25) instead of fewer (implicit AND).
    """
    cleaned = _FTS_STRIP.sub(" ", query)
    cleaned = _FTS_OPS.sub("", cleaned)
    terms = [t for t in cleaned.split() if len(t) > 1]
    return " OR ".join(terms) if terms else ""


def recall(
    *,
    query: str = "",
    mem_type: str | None = None,
    after: str | None = None,
    before: str | None = None,
    related_to: str | None = None,
    limit: int = 20,
    include_private: bool = False,
    cluster_ids: list[int] | None = None,
) -> list[MemoryResult]:
    """Unified memory search: FTS5 + embeddings + temporal + multi-hop graph + composite scoring.

    When *cluster_ids* are provided, search is scoped to those clusters first.
    If results are insufficient, falls back to full corpus search.
    """
    db = _db()
    results: dict[str, dict[str, Any]] = {}
    priv_clause = "" if include_private else " AND m.is_private = 0"

    # Build cluster filter clause if provided
    cluster_clause = ""
    cluster_params: tuple[int, ...] = ()
    if cluster_ids:
        ph = ",".join("?" for _ in cluster_ids)
        cluster_clause = f" AND m.cluster_id IN ({ph})"
        cluster_params = tuple(cluster_ids)

    # --- Strategy 1: FTS5 ranked search ---
    fts_ranked: list[str] = []
    fts_query = _sanitize_fts_query(query) if query else ""
    if fts_query:
        try:
            type_clause = " AND m.type = ?" if mem_type else ""
            type_params: tuple[str, ...] = (mem_type,) if mem_type else ()
            rows = db.execute(
                f"""SELECT f.id, f.type, f.title, rank,
                           m.importance, m.access_count, m.useful_count, m.updated,
                           m.taxonomy
                    FROM memory_fts f
                    JOIN memory_meta m ON f.id = m.id
                    WHERE memory_fts MATCH ? AND m.status = 'active'
                    {type_clause}{priv_clause}{cluster_clause}
                    ORDER BY rank
                    LIMIT ?""",
                (fts_query, *type_params, *cluster_params, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []  # Malformed FTS5 query syntax
        # Normalize FTS scores to 0-1
        max_score = max((-r["rank"] for r in rows), default=1.0) or 1.0
        for r in rows:
            fts_ranked.append(r["id"])
            results[r["id"]] = {
                "id": r["id"],
                "type": r["type"],
                "title": r["title"],
                "score": -r["rank"] / max_score,
                "importance": r["importance"],
                "access_count": r["access_count"],
                "useful_count": r["useful_count"],
                "updated": r["updated"],
                "taxonomy": r["taxonomy"],
            }

    # --- Strategy 2: Semantic search (embedding-based) ---
    sem_ranked: list[str] = []
    query_embedding: list[float] | None = None
    if query:
        query_embedding = _embed_query(query)
        if query_embedding is not None:
            sem_results = _semantic_search(
                query_embedding,
                mem_type=mem_type,
                limit=limit,
                include_private=include_private,
            )
            # Filter by cluster if provided
            if cluster_ids:
                cluster_id_set = set(cluster_ids)
                sem_results = [
                    sr for sr in sem_results if _get_memory_cluster_id(sr["id"]) in cluster_id_set
                ]
            for sr in sem_results:
                sem_ranked.append(sr["id"])
                if sr["id"] not in results:
                    results[sr["id"]] = sr

    # --- Reciprocal Rank Fusion (when both FTS and semantic results exist) ---
    if fts_ranked and sem_ranked:
        rrf_k = settings.rrf_k
        rrf_scores: dict[str, float] = {}
        for rank_idx, mem_id in enumerate(fts_ranked):
            rrf_scores[mem_id] = rrf_scores.get(mem_id, 0) + 1.0 / (rrf_k + rank_idx)
        for rank_idx, mem_id in enumerate(sem_ranked):
            rrf_scores[mem_id] = rrf_scores.get(mem_id, 0) + 1.0 / (rrf_k + rank_idx)
        # Normalize RRF to 0-1
        max_rrf = max(rrf_scores.values()) or 1.0
        for mem_id, rrf_score in rrf_scores.items():
            if mem_id in results:
                results[mem_id]["score"] = rrf_score / max_rrf

    # --- Strategy 3: Temporal range ---
    if after or before:
        conditions = ["m.status = 'active'"]
        params: list[Any] = []
        if after:
            conditions.append("m.updated >= ?")
            params.append(after)
        if before:
            conditions.append("m.updated <= ?")
            params.append(before)
        if mem_type:
            conditions.append("m.type = ?")
            params.append(mem_type)
        if cluster_ids:
            conditions.append(f"m.cluster_id IN ({','.join('?' for _ in cluster_ids)})")
            params.extend(cluster_ids)

        rows = db.execute(
            f"""SELECT m.id, m.type, f.title,
                       m.importance, m.access_count, m.useful_count, m.updated,
                       m.taxonomy
                FROM memory_meta m
                JOIN memory_fts f ON m.id = f.id
                WHERE {" AND ".join(conditions)}{priv_clause}
                ORDER BY m.updated DESC
                LIMIT ?""",
            (*params, limit),
        ).fetchall()
        for r in rows:
            if r["id"] not in results:
                results[r["id"]] = {
                    "id": r["id"],
                    "type": r["type"],
                    "title": r["title"],
                    "score": 0.0,
                    "importance": r["importance"],
                    "access_count": r["access_count"],
                    "useful_count": r["useful_count"],
                    "updated": r["updated"],
                    "taxonomy": r["taxonomy"],
                }

    # --- Strategy 4: Multi-hop graph traversal (BFS, depth up to graph_max_depth) ---
    if related_to:
        # For causal queries, filter graph traversal to causal edges
        causal_filter = CAUSAL_RELATIONSHIPS if query and _CAUSAL_QUERY_RE.search(query) else None
        visited: set[str] = {related_to}
        frontier = [related_to]
        for depth in range(settings.graph_max_depth):
            hop_weight = settings.graph_decay_per_hop ** (depth + 1)
            neighbors = _get_neighbors_batch(db, frontier, relationship_filter=causal_filter)
            next_frontier: list[str] = []
            for n in neighbors:
                if n["id"] not in visited:
                    visited.add(n["id"])
                    # Skip if cluster filter is active and neighbor not in clusters
                    if cluster_ids:
                        n_cluster = _get_memory_cluster_id(n["id"])
                        if n_cluster not in set(cluster_ids):
                            next_frontier.append(n["id"])
                            continue
                    next_frontier.append(n["id"])
                    if n["id"] not in results:
                        results[n["id"]] = {
                            "id": n["id"],
                            "type": n["type"],
                            "title": n["title"],
                            "score": n["weight"] * hop_weight,
                            "relationship": n["relationship"],
                            "importance": n["importance"],
                            "access_count": n["access_count"],
                            "useful_count": n["useful_count"],
                            "updated": n["updated"],
                        }
            frontier = next_frontier

    # --- Type filter (when only type is specified, no query/temporal/graph) ---
    if mem_type and not query and not after and not before and not related_to:
        rows = db.execute(
            f"""SELECT m.id, m.type, f.title,
                       m.importance, m.access_count, m.useful_count, m.updated,
                       m.taxonomy
                FROM memory_meta m
                JOIN memory_fts f ON m.id = f.id
                WHERE m.type = ? AND m.status = 'active'{priv_clause}{cluster_clause}
                ORDER BY m.updated DESC
                LIMIT ?""",
            (mem_type, *cluster_params, limit),
        ).fetchall()
        for r in rows:
            results[r["id"]] = {
                "id": r["id"],
                "type": r["type"],
                "title": r["title"],
                "score": 0.0,
                "importance": r["importance"],
                "access_count": r["access_count"],
                "useful_count": r["useful_count"],
                "updated": r["updated"],
                "taxonomy": r["taxonomy"],
            }

    # Apply type filter to combined results
    if mem_type and (query or after or before or related_to):
        results = {k: v for k, v in results.items() if v["type"] == mem_type}

    # --- Composite scoring: relevance * importance * recency * access frequency ---
    _apply_composite_scores(results)

    ranked = sorted(results.values(), key=lambda x: x.get("score", 0), reverse=True)[:limit]
    return cast(list[MemoryResult], ranked)


def _get_memory_cluster_id(mem_id: str) -> int | None:
    """Get the cluster_id for a memory, or None."""
    row = _db().execute("SELECT cluster_id FROM memory_meta WHERE id = ?", (mem_id,)).fetchone()
    return row["cluster_id"] if row and row["cluster_id"] is not None else None


def generate_cluster_summary(cluster_id: int) -> dict[str, Any] | None:
    """Generate an LLM summary for a cluster and store it.

    Returns the summary dict, or None if generation failed.
    """
    db = _db()
    now = datetime.now(UTC).isoformat()

    # Get cluster members
    member_rows = db.execute(
        """SELECT m.id, m.type, f.title
           FROM memory_meta m
           JOIN memory_fts f ON m.id = f.id
           WHERE m.cluster_id = ? AND m.status = 'active'""",
        (cluster_id,),
    ).fetchall()

    if not member_rows:
        return None

    type_dist: dict[str, int] = {}
    for r in member_rows:
        type_dist[r["type"]] = type_dist.get(r["type"], 0) + 1

    # Build bodies for context
    bodies = []
    for r in member_rows[:8]:
        body = read_memory_body(r["type"], r["id"], 200)
        if body:
            bodies.append(f"[{r['title']}]: {body[:150]}")

    try:
        types = ", ".join(type_dist.keys())
        summary = f"Cluster {cluster_id}: {len(member_rows)} memories covering {types}"

        # Store summary
        summary_embedding = _embed_passage(summary)
        summary_blob = None
        if summary_embedding:
            summary_blob = struct.pack(f"{len(summary_embedding)}f", *summary_embedding)

        db.execute(
            """INSERT OR REPLACE INTO cluster_summaries
               (cluster_id, summary_text, summary_embedding,
                memory_count, type_distribution, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?,
                COALESCE((SELECT created_at FROM cluster_summaries
                          WHERE cluster_id = ?), ?), ?)""",
            (
                cluster_id,
                summary,
                summary_blob,
                len(member_rows),
                json.dumps(type_dist),
                cluster_id,
                now,
                now,
            ),
        )
        _commit(db)

        return {
            "cluster_id": cluster_id,
            "summary": summary,
            "memory_count": len(member_rows),
            "type_distribution": type_dist,
        }
    except Exception:
        log.warning("cluster_summary_failed", cluster_id=cluster_id)
        return None


def generate_all_cluster_summaries() -> list[dict[str, Any]]:
    """Generate summaries for all clusters that don't have one.

    Returns list of summary dicts.
    """
    db = _db()
    cluster_rows = db.execute(
        "SELECT DISTINCT cluster_id FROM memory_meta "
        "WHERE cluster_id IS NOT NULL AND status = 'active'"
    ).fetchall()

    summaries = []
    for cr in cluster_rows:
        cid = cr["cluster_id"]
        existing = db.execute(
            "SELECT 1 FROM cluster_summaries WHERE cluster_id = ?", (cid,)
        ).fetchone()
        if not existing:
            result = generate_cluster_summary(cid)
            if result:
                summaries.append(result)

    return summaries


# ---------------------------------------------------------------------------
# Graph & scoring
# ---------------------------------------------------------------------------
# Graph & scoring
# ---------------------------------------------------------------------------


def _get_neighbors_batch(
    db: sqlite3.Connection,
    node_ids: list[str],
    *,
    relationship_filter: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Get bidirectional graph neighbors of multiple memory nodes in one query.

    Only returns currently-valid links (valid_until IS NULL or empty).
    Optionally filters to specific relationship types.
    """
    if not node_ids:
        return []
    ph = ",".join("?" for _ in node_ids)
    valid_clause = _valid_link_clause("ml")
    rel_clause = ""
    rel_params: tuple[str, ...] = ()
    if relationship_filter:
        rel_ph = ",".join("?" for _ in relationship_filter)
        rel_clause = f"AND ml.relationship IN ({rel_ph})"
        rel_params = tuple(relationship_filter)
    rows = db.execute(
        f"""SELECT ml.to_id AS id, m.type, f.title, ml.relationship, ml.weight,
                   m.importance, m.access_count, m.useful_count, m.updated,
                   m.taxonomy
            FROM memory_links ml
            JOIN memory_meta m ON ml.to_id = m.id
            JOIN memory_fts f ON ml.to_id = f.id
            WHERE ml.from_id IN ({ph}) AND m.status = 'active'
            {valid_clause} {rel_clause}
            UNION
            SELECT ml.from_id AS id, m.type, f.title, ml.relationship, ml.weight,
                   m.importance, m.access_count, m.useful_count, m.updated,
                   m.taxonomy
            FROM memory_links ml
            JOIN memory_meta m ON ml.from_id = m.id
            JOIN memory_fts f ON ml.from_id = f.id
            WHERE ml.to_id IN ({ph}) AND m.status = 'active'
            {valid_clause} {rel_clause}""",
        (*node_ids, *rel_params, *node_ids, *rel_params),
    ).fetchall()
    return [dict(r) for r in rows]


def _recency_score(updated_iso: str) -> float:
    """Score 0-1 where 1.0 = now, decays exponentially over ~30 days."""
    if not updated_iso:
        return 0.0
    updated = ensure_utc(datetime.fromisoformat(updated_iso))
    age_days = (datetime.now(UTC) - updated).total_seconds() / 86400
    return math.exp(-age_days / settings.recency_decay_days)


def _apply_composite_scores(results: dict[str, dict[str, Any]]) -> None:
    """Apply composite scoring (relevance * importance * recency * access) in place.

    Metadata (importance, access_count, useful_count, updated, taxonomy) is expected to
    be pre-fetched in each entry dict by the query strategies in recall().

    Taxonomy adjusts the weight distribution:
    - factual: importance-heavy (stable facts matter most)
    - experiential: recency-heavy (recent experiences matter most)
    - working: recency-dominant (ephemeral, expire fast)
    """
    if not results:
        return

    # Loop-invariant constants
    context_denom = 1.0 - settings.score_weight_relevance
    log_101 = math.log(101)

    # Base weights (used when taxonomy is empty/unset)
    base_w_imp = settings.score_weight_importance
    base_w_rec = settings.score_weight_recency
    base_w_acc = settings.score_weight_access

    # Taxonomy-specific weight overrides (importance, recency, access)
    _TAXONOMY_WEIGHTS: dict[str, tuple[float, float, float]] = {
        "factual": (0.40, 0.10, 0.10),  # importance-heavy: stable facts
        "experiential": (0.15, 0.35, 0.10),  # recency-heavy: recent experiences
        "working": (0.05, 0.45, 0.10),  # recency-dominant: ephemeral
    }

    for entry in results.values():
        relevance = entry.get("score", 0)
        importance: float = min(entry["importance"], 1.0)
        access_count: int = entry["access_count"]
        useful_count: int = entry["useful_count"]
        updated: str = entry["updated"]
        taxonomy: str = entry.get("taxonomy", "")
        recency = _recency_score(updated)
        access_score = math.log(1 + access_count) / log_101  # ~1.0 at 100 accesses
        # Utility modulation: penalize memories that get accessed but never used
        utility_rate = useful_count / max(access_count, 1)
        access_score *= settings.utility_floor + settings.utility_weight * utility_rate

        # Select weights based on taxonomy
        w_imp, w_rec, w_acc = _TAXONOMY_WEIGHTS.get(taxonomy, (base_w_imp, base_w_rec, base_w_acc))

        # Context quality: importance, recency, and access frequency
        context = w_imp * importance + w_rec * min(recency, 1.0) + w_acc * min(access_score, 1.0)
        # Normalize context to 0-1 (weights for non-relevance factors sum to 0.6)
        context_norm = context / context_denom if context_denom > 0 else context

        if relevance > 0:
            # Query-matched: relevance gates the context score
            entry["score"] = relevance * context_norm
        else:
            # Non-query (temporal/graph): rank below query matches
            entry["score"] = context_norm * 0.3


def touch_memories(mem_ids: list[str], *, useful: bool = True, useful_only: bool = False) -> None:
    """Increment access_count and update last_accessed. If *useful*, also increment
    useful_count and strengthen co-access graph links (Hebbian learning).
    If *useful_only*, increment only useful_count (for retroactive utility upgrade)."""
    if not mem_ids:
        return
    conn = _db()
    now = datetime.now(UTC).isoformat()
    ph = ",".join("?" for _ in mem_ids)
    if useful_only:
        set_clause = "useful_count = useful_count + 1"
        conn.execute(f"UPDATE memory_meta SET {set_clause} WHERE id IN ({ph})", mem_ids)
    else:
        set_clause = (
            "access_count = access_count + 1, useful_count = useful_count + 1, last_accessed = ?"
            if useful
            else "access_count = access_count + 1, last_accessed = ?"
        )
        conn.execute(f"UPDATE memory_meta SET {set_clause} WHERE id IN ({ph})", (now, *mem_ids))
    # Hebbian co-access: strengthen existing active links between co-recalled memories
    if useful and len(mem_ids) > 1:
        conn.execute(
            f"""UPDATE memory_links SET weight = MIN(weight + 0.05, 5.0)
                WHERE from_id IN ({ph}) AND to_id IN ({ph})
                {_valid_link_clause()}""",
            (*mem_ids, *mem_ids),
        )
    _commit(conn)


def get_graph_neighbors(mem_ids: list[str], *, limit: int = 3) -> list[MemoryResult]:
    """Get unique 1-hop neighbors of the given memory IDs, excluding the inputs."""
    if not mem_ids:
        return []
    conn = _db()
    ph = ",".join("?" for _ in mem_ids)
    valid = _valid_link_clause("ml")
    rows = conn.execute(
        f"""SELECT DISTINCT m.id, m.type, f.title, ml.relationship
            FROM memory_links ml
            JOIN memory_meta m ON ml.to_id = m.id
            JOIN memory_fts f ON ml.to_id = f.id
            WHERE ml.from_id IN ({ph}) AND m.status = 'active'
              AND ml.to_id NOT IN ({ph}) {valid}
            UNION
            SELECT DISTINCT m.id, m.type, f.title, ml.relationship
            FROM memory_links ml
            JOIN memory_meta m ON ml.from_id = m.id
            JOIN memory_fts f ON ml.from_id = f.id
            WHERE ml.to_id IN ({ph}) AND m.status = 'active'
              AND ml.from_id NOT IN ({ph}) {valid}
            LIMIT ?""",
        (*mem_ids, *mem_ids, *mem_ids, *mem_ids, limit),
    ).fetchall()
    return cast(
        list[MemoryResult],
        [
            {
                "id": r["id"],
                "type": r["type"],
                "title": r["title"],
                "score": 0.0,
                "relationship": r["relationship"],
            }
            for r in rows
        ],
    )


# ---------------------------------------------------------------------------
# Archive, tags, links
# ---------------------------------------------------------------------------


def archive_memory(mem_id: str) -> None:
    conn = _db()
    conn.execute(
        "UPDATE memory_meta SET status = 'archived' WHERE id = ?",
        (mem_id,),
    )
    _commit(conn)


def restore_memory(mem_id: str) -> bool:
    """Restore an archived memory to active status. Returns True if found."""
    conn = _db()
    cur = conn.execute(
        "UPDATE memory_meta SET status = 'active' WHERE id = ? AND status = 'archived'",
        (mem_id,),
    )
    if cur.rowcount == 0:
        return False
    # Re-index in FTS from existing meta (file may still be on disk)
    row = conn.execute("SELECT type FROM memory_meta WHERE id = ?", (mem_id,)).fetchone()
    if row:
        fts_exists = conn.execute("SELECT 1 FROM memory_fts WHERE id = ?", (mem_id,)).fetchone()
        if not fts_exists:
            body = read_memory_body(row["type"], mem_id, 50000)
            if body:
                fm = read_frontmatter(
                    settings.memory_dir
                    / MEMORY_DIRS.get(row["type"], f"{row['type']}s")
                    / f"{mem_id}.md"
                )
                conn.execute(
                    "INSERT INTO memory_fts (id, type, title, content, tags) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        mem_id,
                        row["type"],
                        fm.get("title", mem_id),
                        body,
                        " ".join(fm.get("tags", [])),
                    ),
                )
    _commit(conn)
    return True


def update_memory_tags(mem_id: str, tags: list[str]) -> None:
    """Update tags for a memory in both meta and FTS."""
    conn = _db()
    conn.execute(
        "UPDATE memory_meta SET tags_json = ? WHERE id = ?",
        (json.dumps(tags), mem_id),
    )
    conn.execute(
        "UPDATE memory_fts SET tags = ? WHERE id = ?",
        (" ".join(tags), mem_id),
    )
    _commit(conn)


def link_memories(from_id: str, to_id: str, relationship: str) -> bool:
    """Create a link between memories, preserving accumulated weight if it already exists.

    Returns True if a new link was created, False if it already existed.
    """
    if not from_id or not to_id or not from_id[0].isalnum() or not to_id[0].isalnum():
        log.warning("link_rejected_invalid_id", from_id=from_id, to_id=to_id)
        return False
    conn = _db()
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        """INSERT OR IGNORE INTO memory_links
           (from_id, to_id, relationship, weight, created)
           VALUES (?, ?, ?, 1.0, ?)""",
        (from_id, to_id, relationship, now),
    )
    _commit(conn)
    return cur.rowcount > 0


def invalidate_link(from_id: str, to_id: str, relationship: str) -> bool:
    """Set valid_until on a specific link. Returns True if found and invalidated."""
    conn = _db()
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        f"""UPDATE memory_links SET valid_until = ?
           WHERE from_id = ? AND to_id = ? AND relationship = ?
           {_valid_link_clause()}""",
        (now, from_id, to_id, relationship),
    )
    _commit(conn)
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


def cleanup_archived_fts() -> None:
    """Remove archived memories from FTS5 index."""
    db = _db()
    try:
        db.execute(
            "DELETE FROM memory_fts WHERE id IN (SELECT id FROM memory_meta WHERE status = 'archived')"
        )
        _commit(db)
    except sqlite3.OperationalError:
        db.rollback()  # release the lock, retry next cycle


def prune_old_fts_entries(retention_days: int) -> int:
    """Archive low-importance episodes older than retention_days. Returns count."""
    if retention_days <= 0:
        return 0
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
    conn = _db()
    rows = conn.execute(
        """SELECT id FROM memory_meta
           WHERE type = 'episode' AND status = 'active'
           AND updated < ? AND importance < 0.1""",
        (cutoff,),
    ).fetchall()
    if not rows:
        return 0
    ids = [r["id"] for r in rows]
    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"UPDATE memory_meta SET status = 'archived' WHERE id IN ({placeholders})",
        ids,
    )
    conn.execute(f"DELETE FROM memory_fts WHERE id IN ({placeholders})", ids)
    _commit(conn)
    return len(ids)


def expire_working_memories(max_age_hours: int = 24) -> int:
    """Auto-archive working-taxonomy memories older than max_age_hours.

    Working memories are ephemeral by design — they represent transient context
    (scratch notes, in-progress state, temporary plans) that should not persist.
    Returns the number of archived memories.
    """
    if max_age_hours <= 0:
        return 0
    cutoff = (datetime.now(UTC) - timedelta(hours=max_age_hours)).isoformat()
    conn = _db()
    rows = conn.execute(
        """SELECT id FROM memory_meta
           WHERE taxonomy = 'working' AND status = 'active'
           AND updated < ?""",
        (cutoff,),
    ).fetchall()
    if not rows:
        return 0
    ids = [r["id"] for r in rows]
    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"UPDATE memory_meta SET status = 'archived' WHERE id IN ({placeholders})",
        ids,
    )
    conn.execute(f"DELETE FROM memory_fts WHERE id IN ({placeholders})", ids)
    _commit(conn)
    log.info("working_memories_expired", count=len(ids))
    return len(ids)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def record_memory_change(mem_id: str, changes: list[str]) -> None:
    """Record a change event in memory_history."""
    if not changes:
        return
    conn = _db()
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO memory_history (mem_id, changed_fields, timestamp) VALUES (?, ?, ?)",
        (mem_id, json.dumps(changes), now),
    )
    _commit(conn)


def get_memory_history(mem_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Get change history for a memory, most recent first."""
    rows = (
        _db()
        .execute(
            """SELECT changed_fields, timestamp
           FROM memory_history WHERE mem_id = ?
           ORDER BY timestamp DESC LIMIT ?""",
            (mem_id, limit),
        )
        .fetchall()
    )
    return [{"changes": json.loads(r["changed_fields"]), "timestamp": r["timestamp"]} for r in rows]


# ---------------------------------------------------------------------------
# Overlap / contradiction detection
# ---------------------------------------------------------------------------


def find_similar(
    mem_id: str,
    mem_type: str,
    content: str,
    *,
    limit: int = 5,
    embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Find existing memories of the same type that are semantically similar.

    Used to surface potential overlaps, contradictions, or consolidation
    candidates when saving a new insight or entity. The caller (agent)
    decides what to do — no automated threshold-based judgment.

    If *embedding* is provided, skip re-embedding (saves ~100ms CPU).
    """
    if embedding is None:
        embedding = _embed_passage(content)
    if embedding is None:
        return []
    candidates = _semantic_search(
        embedding,
        mem_type=mem_type,
        limit=limit + 1,  # +1 to account for self-match
        include_private=True,
    )
    results: list[dict[str, Any]] = []
    for c in candidates:
        if c["id"] == mem_id:
            continue
        body = read_memory_body(c["type"], c["id"], 300)
        results.append(
            {
                "id": c["id"],
                "title": c["title"],
                "similarity": round(c["score"], 3),
                "body_preview": body[:200] if body else "",
            }
        )
        if len(results) >= limit:
            break
    return results


# ---------------------------------------------------------------------------
# Correction detection and reconsolidation
# ---------------------------------------------------------------------------

CORRECTION_SIGNALS = {
    "explicit_language": 0.3,
    "source_reliability": 0.3,
    "semantic_strength": 0.2,
    "temporal_recency": 0.2,
}

SOURCE_RELIABILITY = {
    "user_direct": 1.0,
    "agent_self_correction": 0.85,
    "agent_inferred": 0.6,
    "external_tool": 0.5,
    "system_detected": 0.4,
}

CORRECTION_PATTERNS = re.compile(
    r"\b(actually|correction|that'?s not right|that'?s wrong|i was wrong|"
    r"i meant|let me correct|no that'?s incorrect|correction:|wait,? )\b",
    re.IGNORECASE,
)


def compute_correction_confidence(
    explicit_language: bool = False,
    source: str = "system_detected",
    semantic_strength: float = 0.0,
    temporal_recency: float = 0.5,
) -> float:
    """Score how confident we are that this is a genuine correction.

    Returns a value between 0.0 and 1.0.
    """
    signals = {
        "explicit_language": 1.0 if explicit_language else 0.0,
        "source_reliability": SOURCE_RELIABILITY.get(source, 0.4),
        "semantic_strength": min(semantic_strength, 1.0),
        "temporal_recency": min(temporal_recency, 1.0),
    }
    score = sum(CORRECTION_SIGNALS[k] * signals[k] for k in CORRECTION_SIGNALS)
    return round(score, 3)


def classify_relationship(existing_content: str, new_content: str) -> str:
    """Classify relationship between existing and new content.

    Uses semantic similarity with thresholds:
    - < 0.3: independent (no meaningful overlap)
    - 0.3-0.7: extendable (complementary information)
    - > 0.7: contradictory (conflicting — needs LLM verification)

    Returns: 'independent', 'extendable', or 'contradictory'
    """
    try:
        existing_emb = _embed_passage(existing_content)
        new_emb = _embed_passage(new_content)
    except Exception:
        return "extendable"  # conservative default

    if existing_emb is None or new_emb is None:
        return "extendable"

    sim = 1.0 / (1.0 + _cosine_distance(existing_emb, new_emb))

    if sim < 0.3:
        return "independent"
    elif sim > 0.7:
        return "contradictory"
    else:
        return "extendable"


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))


def apply_correction(
    mem_id: str,
    corrected_content: str,
    *,
    confidence: float = 0.7,
    source: str = "auto_detection",
    change_type: str = "correction",
) -> dict[str, Any]:
    """Apply a detected correction to a memory.

    1. Fetch current memory content
    2. Classify relationship (extendable vs contradictory)
    3. If extendable: merge content
    4. If contradictory: replace with new, link with supersedes
    5. Re-embed and re-index
    6. Log to memory_history with content snapshots
    """
    conn = _db()
    row = conn.execute("SELECT content FROM memory_fts WHERE id = ?", (mem_id,)).fetchone()
    if not row:
        return {"status": "not_found", "mem_id": mem_id}

    existing_content = row["content"]

    relationship = classify_relationship(existing_content, corrected_content)

    if relationship == "extendable":
        new_content = existing_content + "\n\n" + corrected_content
        change_desc = "Extended with new information"
    elif relationship == "contradictory":
        new_content = corrected_content
        change_desc = "Contradiction resolved — replaced content"
    else:
        new_content = corrected_content
        change_desc = "Updated content"

    # Compute embedding BEFORE acquiring write lock (CPU-intensive)
    embedding = _embed_passage(new_content)
    packed_embedding: bytes | None = None
    rowid: int | None = None
    if embedding is not None:
        rowid = _vec_rowid(mem_id)
        packed_embedding = struct.pack(f"{len(embedding)}f", *embedding)

    # All writes in one fast batch — minimise time holding the write lock
    now = datetime.now(UTC).isoformat()
    changes = [change_desc, f"Confidence: {confidence:.2f}, Source: {source}"]

    # Note: contradictory corrections replace content in-place on the same
    # mem_id, so there is no separate "old" memory to link.  The history
    # table already captures the before/after snapshot.

    conn.execute("UPDATE memory_fts SET content = ? WHERE id = ?", (new_content, mem_id))
    if packed_embedding is not None and rowid is not None:
        conn.execute(
            "INSERT OR REPLACE INTO memory_vec (rowid, memory_id, embedding) VALUES (?, ?, ?)",
            (rowid, mem_id, packed_embedding),
        )
    conn.execute("UPDATE memory_meta SET updated = ? WHERE id = ?", (now, mem_id))
    conn.execute(
        """INSERT INTO memory_history
           (mem_id, changed_fields, timestamp, old_content,
            new_content, change_type, source, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            mem_id,
            json.dumps(changes),
            now,
            existing_content[:500],
            new_content[:500],
            change_type,
            source,
            confidence,
        ),
    )
    _commit(conn)

    return {
        "status": "applied",
        "mem_id": mem_id,
        "relationship": relationship,
        "confidence": confidence,
    }


def flag_for_review(
    mem_id: str,
    corrected_content: str,
    *,
    confidence: float = 0.6,
    source: str = "auto_detection",
) -> dict[str, Any]:
    """Flag a potential correction for agent review."""
    conn = _db()
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """INSERT INTO pending_corrections
           (mem_id, corrected_content, confidence,
            source, status, created_at)
           VALUES (?, ?, ?, ?, 'pending', ?)""",
        (mem_id, corrected_content, confidence, source, now),
    )
    _commit(conn)
    return {"status": "flagged", "mem_id": mem_id, "confidence": confidence}


def get_pending_corrections(
    mem_id: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get pending corrections for agent review."""
    conn = _db()
    if mem_id:
        rows = conn.execute(
            """SELECT * FROM pending_corrections
               WHERE mem_id = ? AND status = 'pending'
               ORDER BY created_at DESC LIMIT ?""",
            (mem_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM pending_corrections
               WHERE status = 'pending'
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def resolve_correction(correction_id: int, status: str) -> dict[str, Any]:
    """Resolve a pending correction (approved, rejected, applied)."""
    conn = _db()
    now = datetime.now(UTC).isoformat()
    row = conn.execute(
        "SELECT * FROM pending_corrections WHERE id = ?", (correction_id,)
    ).fetchone()
    if not row:
        return {"status": "not_found"}

    conn.execute(
        "UPDATE pending_corrections SET status = ?, resolved_at = ? WHERE id = ?",
        (status, now, correction_id),
    )
    _commit(conn)

    if status == "approved" or status == "applied":
        return apply_correction(
            row["mem_id"],
            row["corrected_content"],
            confidence=row["confidence"],
            source=row["source"],
        )

    return {"status": status, "correction_id": correction_id}


def detect_corrections(
    recalled_memory_ids: list[str],
    agent_response: str,
    conversation_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Detect when the agent or user corrects information from recalled memories.

    Three-layer detection:
    1. Explicit correction pattern matching
    2. Semantic contradiction detection
    3. Confidence scoring

    Returns list of correction dicts with memory_id, original, corrected, confidence.
    """
    corrections = []
    has_explicit_signal = bool(CORRECTION_PATTERNS.search(agent_response))

    for mem_id in recalled_memory_ids:
        conn = _db()
        row = conn.execute(
            "SELECT content, title FROM memory_fts WHERE id = ?", (mem_id,)
        ).fetchone()
        if not row:
            continue

        original_content = row["content"]
        semantic_sim = _compute_semantic_similarity(original_content, agent_response)

        if semantic_sim > 0.5:
            confidence = compute_correction_confidence(
                explicit_language=has_explicit_signal,
                source="agent_inferred",
                semantic_strength=semantic_sim,
            )

            corrections.append(
                {
                    "memory_id": mem_id,
                    "original": original_content,
                    "corrected": agent_response[:500],
                    "confidence": confidence,
                    "source": "agent_response",
                    "explicit_signal": has_explicit_signal,
                }
            )

    return corrections


def _compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity between two texts using embeddings."""
    try:
        emb_a = _embed_passage(text_a)
        emb_b = _embed_passage(text_b)
    except Exception:
        return 0.0

    if emb_a is None or emb_b is None:
        return 0.0

    sim = 1.0 / (1.0 + _cosine_distance(emb_a, emb_b))
    return round(sim, 3)


# ---------------------------------------------------------------------------
# Insight clustering (for consolidation)
# ---------------------------------------------------------------------------


def get_insight_clusters(
    *,
    similarity_threshold: float | None = None,
    min_cluster: int | None = None,
    exclude_tags: frozenset[str] | None = None,
) -> list[list[dict[str, Any]]]:
    """Find clusters of semantically similar active insights.

    Uses KNN via sqlite-vec (O(n·K), not O(n²) pairwise). Scalable to thousands.
    """
    if similarity_threshold is None:
        similarity_threshold = _INSIGHT_SIMILARITY_THRESHOLD
    if min_cluster is None:
        min_cluster = _INSIGHT_MIN_CLUSTER
    if exclude_tags is None:
        exclude_tags = frozenset({"feedback"})

    db = _db()
    rows = db.execute(
        "SELECT id, tags_json FROM memory_meta WHERE type = 'insight' AND status = 'active'"
    ).fetchall()

    # Filter out excluded tags
    insight_ids: list[str] = []
    for r in rows:
        tags = set(json.loads(r["tags_json"]))
        if not (exclude_tags & tags):
            insight_ids.append(r["id"])

    if len(insight_ids) < min_cluster:
        return []

    # Build neighbor graph via KNN — for each insight, find top-5 nearest among others
    adjacency: dict[str, set[str]] = {mid: set() for mid in insight_ids}
    id_set = set(insight_ids)

    for mid in insight_ids:
        # Fetch this memory's embedding
        row = db.execute("SELECT embedding FROM memory_vec WHERE memory_id = ?", (mid,)).fetchone()
        if row is None:
            continue
        vec = list(struct.unpack(f"{len(row['embedding']) // 4}f", row["embedding"]))
        neighbors = _semantic_search(vec, mem_type="insight", limit=6, include_private=True)
        for n in neighbors:
            if n["id"] != mid and n["id"] in id_set and n["score"] >= similarity_threshold:
                adjacency[mid].add(n["id"])
                adjacency.setdefault(n["id"], set()).add(mid)

    # BFS to find connected components
    visited: set[str] = set()
    clusters: list[list[dict[str, Any]]] = []

    for start in insight_ids:
        if start in visited:
            continue
        component: list[str] = []
        queue = collections.deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            queue.extend(adjacency.get(node, set()) - visited)
        if len(component) >= min_cluster:
            # Fetch metadata for cluster members
            ph = ",".join("?" for _ in component)
            meta_rows = db.execute(
                f"""SELECT m.id, m.type, f.title, m.created, m.updated, m.importance
                    FROM memory_meta m JOIN memory_fts f ON m.id = f.id
                    WHERE m.id IN ({ph})""",
                component,
            ).fetchall()
            clusters.append([dict(r) for r in meta_rows])

    return clusters


# ---------------------------------------------------------------------------
# Lifecycle review
# ---------------------------------------------------------------------------


def get_lifecycle_candidates() -> dict[str, list[dict[str, Any]]]:
    """Find memories needing review, with context from recent activity.

    Returns dict with keys: stale_entities, unused_procedures, lingering_goals.
    """
    db = _db()
    result: dict[str, list[dict[str, Any]]] = {
        "stale_entities": [],
        "unused_procedures": [],
        "lingering_goals": [],
    }

    stale_cutoff = (datetime.now(UTC) - timedelta(days=_STALE_ENTITY_DAYS)).isoformat()
    unused_cutoff = (datetime.now(UTC) - timedelta(days=_UNUSED_PROCEDURE_DAYS)).isoformat()

    # Stale entities: not updated in N days + cross-reference with recent episodes
    stale_rows = db.execute(
        """SELECT id, type, updated FROM memory_meta
           WHERE type = 'entity' AND status = 'active' AND updated < ?""",
        (stale_cutoff,),
    ).fetchall()
    for r in stale_rows:
        # Count recent episodes mentioning this entity via FTS
        try:
            mention_row = db.execute(
                "SELECT COUNT(*) AS cnt FROM memory_fts WHERE type = 'episode' AND content MATCH ?",
                (r["id"],),
            ).fetchone()
            mentions = mention_row["cnt"] if mention_row else 0
        except sqlite3.OperationalError:
            mentions = 0
        result["stale_entities"].append(
            {
                "id": r["id"],
                "type": r["type"],
                "updated": r["updated"],
                "recent_mentions": mentions,
            }
        )

    # Unused procedures: not accessed in N days
    unused_rows = db.execute(
        """SELECT id, type, last_accessed FROM memory_meta
           WHERE type = 'procedure' AND status = 'active'
           AND (last_accessed = '' OR last_accessed < ?)""",
        (unused_cutoff,),
    ).fetchall()
    result["unused_procedures"] = [dict(r) for r in unused_rows]

    # Lingering goals: completed tag but still active status
    goal_rows = db.execute(
        """SELECT id, type FROM memory_meta
           WHERE type = 'goal' AND status = 'active'
           AND tags_json LIKE '%"completed"%'"""
    ).fetchall()
    result["lingering_goals"] = [dict(r) for r in goal_rows]

    return result


def get_feedback_insight_ids() -> list[str]:
    """Return IDs of active feedback insights (prefixed 'feedback-' or tagged 'feedback')."""
    rows = (
        _db()
        .execute(
            """SELECT id FROM memory_meta
           WHERE type = 'insight' AND status = 'active'
           AND (id LIKE 'feedback-%' OR tags_json LIKE '%"feedback"%')"""
        )
        .fetchall()
    )
    return [r["id"] for r in rows]


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def sync_memory_index() -> None:
    """Sync memory index with markdown files on disk. Indexes any unindexed files."""
    db = _db()
    indexed_ids: set[str] = {row[0] for row in db.execute("SELECT id FROM memory_meta").fetchall()}
    memory_dir = settings.memory_dir
    if not memory_dir.exists():
        return

    with batch():
        for type_dir in memory_dir.iterdir():
            if not type_dir.is_dir():
                continue
            for md_file in type_dir.glob("*.md"):
                try:
                    fm = read_frontmatter(md_file)
                    if not fm or "id" not in fm:
                        continue
                    if fm["id"] in indexed_ids:
                        continue
                    body = strip_frontmatter(md_file.read_text())
                    title = ""
                    for line in body.split("\n"):
                        if line.startswith("# "):
                            title = line[2:].strip()
                            break
                    index_memory(
                        fm["id"],
                        fm.get("type", _DIR_TO_TYPE.get(type_dir.name, type_dir.name)),
                        title,
                        body,
                        fm.get("tags", []),
                        fm.get("links", []),
                        skill_meta=fm.get("skill_meta"),
                    )
                except Exception:
                    log.warning("memory_index_skip", file=str(md_file))
                    continue

    # After indexing, backfill any procedures missing skill_meta
    backfill_skill_meta()


# ---------------------------------------------------------------------------
# Skill loop: quality gate, trigger matching, skill metadata
# ---------------------------------------------------------------------------

# Quality gate thresholds
_SKILL_MIN_STEPS = 3
_SKILL_DUPLICATE_THRESHOLD = 0.85


def get_skill_meta(mem_id: str) -> dict[str, Any] | None:
    """Get parsed skill_meta for a procedure memory, or None."""
    row = _db().execute("SELECT skill_meta FROM memory_meta WHERE id = ?", (mem_id,)).fetchone()
    if row and row["skill_meta"]:
        return cast(dict[str, Any], json.loads(row["skill_meta"]))
    return None


def update_skill_meta(mem_id: str, updates: dict[str, Any]) -> None:
    """Merge updates into existing skill_meta for a procedure memory."""
    db = _db()
    row = db.execute("SELECT skill_meta FROM memory_meta WHERE id = ?", (mem_id,)).fetchone()
    if not row:
        return
    existing = json.loads(row["skill_meta"]) if row["skill_meta"] else {}
    existing.update(updates)
    db.execute(
        "UPDATE memory_meta SET skill_meta = ? WHERE id = ?",
        (json.dumps(existing), mem_id),
    )
    trigger_fields = {"trigger_pattern", "confidence", "success_count", "failure_count"}
    if trigger_fields & updates.keys():
        if existing.get("trigger_pattern"):
            from .db import upsert_skill_trigger

            upsert_skill_trigger(
                procedure_id=mem_id,
                trigger_pattern=existing["trigger_pattern"],
                confidence=existing.get("confidence", 0.5),
                success_count=existing.get("success_count", 0),
                failure_count=existing.get("failure_count", 0),
            )
        else:
            from .db import delete_skill_trigger

            delete_skill_trigger(mem_id)
    _commit(db)


def skill_gate(
    content: str,
    steps: list[str],
    *,
    exclude_id: str | None = None,
) -> tuple[bool, str]:
    """Quality gate for skill extraction. Returns (pass, reason).

    Checks:
    1. Non-trivial: >= 3 steps
    2. Non-duplicate: < 0.85 cosine similarity to existing procedures
    """
    # Check 1: Non-trivial
    if len(steps) < _SKILL_MIN_STEPS:
        return False, f"too few steps ({len(steps)} < {_SKILL_MIN_STEPS})"

    # Check 2: Non-duplicate
    embedding = _embed_passage(content)
    if embedding is not None:
        candidates = _semantic_search(
            embedding, mem_type="procedure", limit=3, include_private=True
        )
        for c in candidates:
            if exclude_id and c["id"] == exclude_id:
                continue
            if c["score"] >= _SKILL_DUPLICATE_THRESHOLD:
                return False, f"duplicate of {c['id']} (similarity {c['score']:.2f})"

    return True, "passed"


def backfill_skill_meta() -> int:
    """Backfill existing procedure memories with default skill_meta.

    Sets confidence to 0.4 (lower than auto-extracted, per spec) for procedures
    that don't already have skill_meta. Returns count of backfilled procedures.
    """
    db = _db()
    rows = db.execute(
        """SELECT m.id, f.title, f.content
           FROM memory_meta m
           JOIN memory_fts f ON m.id = f.id
           WHERE m.type = 'procedure' AND m.status = 'active'
           AND (m.skill_meta IS NULL OR m.skill_meta = '')""",
    ).fetchall()
    if not rows:
        return 0

    count = 0
    for r in rows:
        # Generate a basic trigger pattern from the title
        title_words = re.findall(r"\b[a-z]{4,}\b", r["title"].lower())
        trigger_pattern = "|".join(set(title_words[:6])) if title_words else ""

        sm = {
            "version": 1,
            "source_tasks": [],
            "success_count": 0,
            "failure_count": 0,
            "last_applied": None,
            "confidence": 0.4,
            "trigger_pattern": trigger_pattern,
        }
        db.execute(
            "UPDATE memory_meta SET skill_meta = ? WHERE id = ?",
            (json.dumps(sm), r["id"]),
        )
        # Sync trigger table
        if trigger_pattern:
            from .db import upsert_skill_trigger

            upsert_skill_trigger(
                procedure_id=r["id"],
                trigger_pattern=trigger_pattern,
                confidence=0.4,
            )
        count += 1
    _commit(db)
    log.info("skill_meta_backfill", count=count)
    return count


def get_trigger_matched_skills(text: str) -> list[MemoryResult]:
    """Find procedure memories whose trigger patterns match the given text.

    Returns enriched MemoryResult entries for injection into auto_recall.
    """
    from .db import get_matching_skill_triggers

    matches = get_matching_skill_triggers(text)
    if not matches:
        return []

    db = _db()
    results: list[MemoryResult] = []
    for m in matches:
        row = db.execute(
            """SELECT m.id, m.type, f.title
               FROM memory_meta m
               JOIN memory_fts f ON m.id = f.id
               WHERE m.id = ? AND m.status = 'active'""",
            (m["procedure_id"],),
        ).fetchone()
        if row:
            results.append(
                cast(
                    MemoryResult,
                    {
                        "id": row["id"],
                        "type": row["type"],
                        "title": row["title"],
                        "score": m["confidence"],
                    },
                )
            )
    return results


# ---------------------------------------------------------------------------
# Semantic clustering: BIRCH (online) + HDBSCAN (offline)
# ---------------------------------------------------------------------------

_birch_model: Any = None


def _get_birch() -> Any:
    """Lazy-init BIRCH model."""
    global _birch_model
    if _birch_model is not None:
        return _birch_model
    from sklearn.cluster import Birch  # type: ignore[import-not-found]

    _birch_model = Birch(
        threshold=settings.birch_threshold,
        branching_factor=settings.birch_branching_factor,
        n_clusters=None,
        compute_labels=True,
    )
    return _birch_model


def _normalize_embeddings(embeddings: list[list[float]]) -> list[list[float]]:
    """L2-normalize embeddings so euclidean distance ≈ cosine distance."""
    result = []
    for vec in embeddings:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            result.append([x / norm for x in vec])
        else:
            result.append(vec)
    return result


def _embeddings_to_numpy_array(embeddings: list[list[float]]) -> Any:
    """Convert list of embeddings to numpy array for sklearn."""
    import numpy as np

    return np.array(embeddings, dtype=np.float32)


def assign_cluster_online(mem_id: str, embedding: list[float] | None) -> int | None:
    """Assign a new memory to a cluster using BIRCH (online/streaming).

    Returns the cluster_id, or None if clustering failed.
    """
    if embedding is None:
        return None

    db = _db()
    now = datetime.now(UTC).isoformat()

    try:
        birch = _get_birch()

        # Load existing cluster state from DB to reinitialize BIRCH
        existing = db.execute(
            "SELECT id, cluster_id FROM memory_meta "
            "WHERE cluster_id IS NOT NULL AND status = 'active'"
        ).fetchall()

        if existing:
            # Load embeddings for already-clustered memories to warm-start BIRCH
            clustered_ids = [r["id"] for r in existing]
            placeholders = ",".join("?" for _ in clustered_ids)
            vec_rows = db.execute(
                f"SELECT memory_id, embedding FROM memory_vec WHERE memory_id IN ({placeholders})",
                clustered_ids,
            ).fetchall()

            embed_map: dict[str, list[float]] = {}
            for vr in vec_rows:
                if vr["embedding"]:
                    dim = len(vr["embedding"]) // 4
                    embed_map[vr["memory_id"]] = list(struct.unpack(f"{dim}f", vr["embedding"]))

            if embed_map:
                ordered_embeddings = []
                ordered_ids = []
                for r in existing:
                    if r["id"] in embed_map:
                        ordered_embeddings.append(embed_map[r["id"]])
                        ordered_ids.append(r["id"])

                if ordered_embeddings:
                    normalized = _normalize_embeddings(ordered_embeddings)
                    np_embeddings = _embeddings_to_numpy_array(normalized)
                    birch.fit(np_embeddings)

        # Assign the new memory
        normalized_new = _normalize_embeddings([embedding])
        np_new = _embeddings_to_numpy_array(normalized_new)

        # If BIRCH hasn't been fitted yet, partial_fit first to initialize
        from sklearn.exceptions import NotFittedError  # type: ignore[import-not-found]

        try:
            label = int(birch.predict(np_new)[0])
            birch.partial_fit(np_new)
        except NotFittedError:
            birch.partial_fit(np_new)
            label = int(birch.predict(np_new)[0])

        # Store cluster assignment
        db.execute(
            """UPDATE memory_meta
               SET cluster_id = ?, cluster_updated_at = ?,
                   cluster_assignment_method = 'birch_online'
               WHERE id = ?""",
            (label, now, mem_id),
        )

        # Store/update centroid for this cluster
        _update_cluster_centroid(label, birch, db, now, "birch")

        _commit(db)
        return label

    except Exception:
        log.warning("cluster_assignment_failed", mem_id=mem_id)
        return None


def _update_cluster_centroid(
    cluster_id: int, birch: Any, db: sqlite3.Connection, now: str, method: str
) -> None:
    """Store or update the centroid for a cluster."""
    try:
        centroids = birch.subcluster_centers_
        if hasattr(birch, "subcluster_labels_"):
            labels = birch.subcluster_labels_
            # Find the centroid for this specific cluster label
            mask = [i for i, lbl in enumerate(labels) if int(lbl) == cluster_id]
            if mask:
                centroid = centroids[mask[0]].tolist()
            else:
                # Fallback: use the cluster_id as index if labels don't match
                if cluster_id < len(centroids):
                    centroid = centroids[cluster_id].tolist()
                else:
                    return
        else:
            if cluster_id < len(centroids):
                centroid = centroids[cluster_id].tolist()
            else:
                return

        centroid_blob = struct.pack(f"{len(centroid)}f", *centroid)

        # Count members
        count_row = db.execute(
            "SELECT COUNT(*) AS cnt FROM memory_meta WHERE cluster_id = ? AND status = 'active'",
            (cluster_id,),
        ).fetchone()
        member_count = count_row["cnt"] if count_row else 0

        db.execute(
            """INSERT OR REPLACE INTO cluster_centroids
               (cluster_id, centroid_embedding, member_count,
                created_at, updated_at, method)
               VALUES (?, ?, ?,
                COALESCE((SELECT created_at FROM cluster_centroids
                          WHERE cluster_id = ?), ?), ?, ?)""",
            (cluster_id, centroid_blob, member_count, cluster_id, now, now, method),
        )

    except Exception:
        log.warning("centroid_update_failed", cluster_id=cluster_id)


def recluster_offline() -> dict[str, Any]:
    """Run HDBSCAN re-clustering on all active memories with embeddings.

    Returns a report with cluster quality metrics.
    """
    try:
        import hdbscan  # type: ignore[import-not-found]
    except ImportError:
        log.warning("hdbscan_not_installed")
        return {"error": "hdbscan not installed", "n_clusters": 0, "n_noise": 0}

    db = _db()
    now = datetime.now(UTC).isoformat()

    # Load all active memory embeddings
    rows = db.execute(
        """SELECT m.id, v.embedding
           FROM memory_meta m
           JOIN memory_vec v ON m.id = v.memory_id
           WHERE m.status = 'active' AND v.embedding IS NOT NULL"""
    ).fetchall()

    if len(rows) < settings.hdbscan_min_cluster_size:
        return {
            "error": "too few memories",
            "n_clusters": 0,
            "n_noise": 0,
            "total_memories": len(rows),
        }

    # Build embedding array
    memory_ids: list[str] = []
    embeddings: list[list[float]] = []
    for r in rows:
        if r["embedding"]:
            dim = len(r["embedding"]) // 4
            vec = list(struct.unpack(f"{dim}f", r["embedding"]))
            memory_ids.append(r["id"])
            embeddings.append(vec)

    if len(embeddings) < settings.hdbscan_min_cluster_size:
        return {
            "error": "too few embeddings",
            "n_clusters": 0,
            "n_noise": 0,
            "total_memories": len(embeddings),
        }

    # Normalize and cluster
    normalized = _normalize_embeddings(embeddings)
    np_embeddings = _embeddings_to_numpy_array(normalized)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=settings.hdbscan_min_cluster_size,
        min_samples=settings.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(np_embeddings)

    # Update assignments
    updated = 0
    noise_count = 0
    cluster_sizes: dict[int, int] = {}

    for i, mem_id in enumerate(memory_ids):
        label = int(labels[i])
        if label == -1:
            noise_count += 1
            db.execute(
                """UPDATE memory_meta
                   SET cluster_id = NULL, cluster_updated_at = ?,
                       cluster_assignment_method = 'hdbscan_offline'
                   WHERE id = ?""",
                (now, mem_id),
            )
        else:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
            db.execute(
                """UPDATE memory_meta
                   SET cluster_id = ?, cluster_updated_at = ?,
                       cluster_assignment_method = 'hdbscan_offline'
                   WHERE id = ?""",
                (label, now, mem_id),
            )
            updated += 1

    # Recompute centroids from actual member embeddings
    for cluster_id in cluster_sizes:
        member_ids = [memory_ids[i] for i in range(len(memory_ids)) if int(labels[i]) == cluster_id]
        if not member_ids:
            continue

        # Compute centroid as mean of member embeddings
        member_embeddings = []
        for mid in member_ids:
            row = db.execute(
                "SELECT embedding FROM memory_vec WHERE memory_id = ?", (mid,)
            ).fetchone()
            if row and row["embedding"]:
                dim = len(row["embedding"]) // 4
                vec = list(struct.unpack(f"{dim}f", row["embedding"]))
                member_embeddings.append(vec)

        if member_embeddings:
            dim = len(member_embeddings[0])
            centroid = [
                sum(e[d] for e in member_embeddings) / len(member_embeddings) for d in range(dim)
            ]
            centroid_blob = struct.pack(f"{dim}f", *centroid)

            db.execute(
                """INSERT OR REPLACE INTO cluster_centroids
                   (cluster_id, centroid_embedding, member_count, created_at, updated_at, method)
                   VALUES (?, ?, ?,
                    COALESCE((SELECT created_at FROM cluster_centroids
                              WHERE cluster_id = ?), ?),
                    ?, 'hdbscan')""",
                (cluster_id, centroid_blob, cluster_sizes[cluster_id], cluster_id, now, now),
            )

    _commit(db)

    n_clusters = len(cluster_sizes)
    log.info(
        "offline_reclustering_done",
        n_clusters=n_clusters,
        n_noise=noise_count,
        n_updated=updated,
        cluster_sizes=cluster_sizes,
    )

    return {
        "n_clusters": n_clusters,
        "n_noise": noise_count,
        "n_updated": updated,
        "cluster_sizes": cluster_sizes,
        "total_memories": len(memory_ids),
    }
