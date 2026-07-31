"""Letta backend adapter for Luke's semantic recall.

Isolated, reversible integration point. When ``settings.memory_backend == "letta"``,
``recall()`` sources its semantic-candidate set from a running Letta server's vector
store instead of the in-process sqlite-vec index. Everything else about recall — FTS5,
RRF fusion, temporal, graph multi-hop, composite scoring — is unchanged, because this
adapter returns the exact same enriched-dict shape as ``memory._semantic_search`` by
re-joining Letta's candidate ids back onto Luke's own ``memory_meta`` / ``memory_fts``.

Design invariants (this is Luke's most silent-fail-prone component):
  * FAIL-SAFE: any error, timeout, or empty/malformed response returns ``None`` so the
    caller transparently falls back to sqlite-vec. Letta being down must never break recall.
  * SHORT TIMEOUT: a hung server cannot stall a user-facing recall.
  * DELETABLE: remove this file + the two-line branch in memory.py to fully revert.

The Letta passages were loaded carrying ``metadata.luke_id`` (see scripts/letta_full_load.py),
which is the join key back to Luke's memory rows.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import structlog

from .config import settings

log = structlog.get_logger()

_TIMEOUT_S = 2.5


def _search_passages(query: str, limit: int) -> list[dict[str, Any]] | None:
    """POST to Letta's passage-search endpoint. Returns raw passage dicts or None on failure."""
    url = f"{settings.letta_base_url.rstrip('/')}/v1/passages/search"
    body = json.dumps(
        {"query": query, "archive_id": settings.letta_archive_id, "limit": limit}
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            data = json.load(resp)
    except Exception as exc:  # network, timeout, JSON, HTTP — all fall back to sqlite
        log.warning("letta_search_failed", err=str(exc)[:120])
        return None
    # Response is a flat list of passage objects, or a wrapper with results/passages.
    if isinstance(data, dict):
        data = data.get("results") or data.get("passages") or []
    if not isinstance(data, list):
        return None
    return data


def letta_semantic_search(
    query: str,
    *,
    mem_type: str | None = None,
    limit: int = 20,
    include_private: bool = False,
) -> list[dict[str, Any]] | None:
    """Letta-backed drop-in for ``memory._semantic_search``.

    Returns the same enriched-dict shape (id/type/title/score/importance/access_count/
    useful_count/updated/taxonomy), ranked by Letta vector similarity but scored and
    filtered against Luke's own memory_meta. Returns ``None`` to signal fall-through
    to sqlite-vec (server down, no candidates, or nothing joinable).
    """
    from .memory import _db  # local import avoids a circular dependency

    # Over-fetch: ~40% of the archive is tombstone placeholders (Phase 2.4) that get
    # skipped below, so widen the window to keep enough real candidates after attrition.
    passages = _search_passages(query, limit * (5 if mem_type else 3))
    if not passages:
        return None

    # Letta returns results already ranked by similarity; preserve that order and turn
    # rank position into a monotonic similarity proxy (RRF downstream only needs order).
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for p in passages:
        obj = p.get("passage", p) if isinstance(p, dict) else {}
        meta = obj.get("metadata") or {}
        # Tombstones are metadata-only placeholders loaded for store completeness /
        # id-resolution (Phase 2.4). Their placeholder text embeds the id slug, which
        # can outrank real content on some queries — never let them into recall. The
        # sqlite active-status join below is a second backstop, but skipping here keeps
        # tombstones from consuming candidate slots and shifting real ranks.
        if meta.get("is_tombstone"):
            continue
        luke_id = meta.get("luke_id")
        if luke_id and luke_id not in seen:
            seen.add(luke_id)
            ordered_ids.append(luke_id)
    if not ordered_ids:
        return None
    rank_similarity = {mid: 1.0 / (1 + i) for i, mid in enumerate(ordered_ids)}

    # Enrich against Luke's own rows — identical filter/shape to _semantic_search.
    db = _db()
    conditions = ["m.status = 'active'"]
    if not include_private:
        conditions.append("m.is_private = 0")
    if mem_type:
        conditions.append("m.type = ?")
    filter_clause = " AND ".join(conditions)
    placeholders = ",".join("?" for _ in ordered_ids)
    params: list[Any] = list(ordered_ids)
    if mem_type:
        params.append(mem_type)

    meta_rows = db.execute(
        f"""SELECT m.id, m.type, f.title,
                   m.importance, m.access_count, m.useful_count, m.updated, m.taxonomy
            FROM memory_meta m
            JOIN memory_fts f ON m.id = f.id
            WHERE m.id IN ({placeholders}) AND {filter_clause}""",
        params,
    ).fetchall()
    if not meta_rows:
        return None

    scored: list[dict[str, Any]] = []
    for r in meta_rows:
        scored.append(
            {
                "id": r["id"],
                "type": r["type"],
                "title": r["title"],
                "score": rank_similarity.get(r["id"], 0.0),
                "importance": r["importance"],
                "access_count": r["access_count"],
                "useful_count": r["useful_count"],
                "updated": r["updated"],
                "taxonomy": r["taxonomy"],
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]
