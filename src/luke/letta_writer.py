"""Letta write-through for Luke's memory store (Phase 2.2c).

The read side (``letta_adapter``) lets ``recall`` source semantic candidates from Letta.
This is the write side: when ``settings.memory_backend == "letta"``, every ``index_memory``
write (i.e. ``remember``) is mirrored into the Letta archive **live**, so the shadow-run
stays current without waiting on the daily ``letta_delta_sync`` cron. It retires that
snapshot-drift stopgap for the create/update path.

Design invariants (mirrors the adapter — this is Luke's most silent-fail-prone component):
  * FAIL-SAFE: any error, timeout, or malformed response is swallowed with a warning. The
    sqlite write has already committed; a write-through failure must NEVER break remember().
    The daily delta-sync remains the backstop for anything this misses.
  * BEST-EFFORT UPSERT: idempotent by ``luke_id``. Old passages for the id are found via the
    search API (exact-surface query ranks the identical passage first) and deleted before the
    fresh one is written, so repeated updates don't accumulate duplicate/stale passages.
    If the find step fails, we still create (the adapter dedups by luke_id, so recall stays
    correct; at worst a duplicate lingers until the next delta-sync/compaction).
  * REST-ONLY, NO NEW DEPENDENCY: uses the same HTTP endpoints already in use. No Postgres
    driver in the core runtime.
  * DELETABLE: remove this file + the one-line call in memory.py to fully revert.

Passage shape matches ``scripts/letta_import.build_records`` exactly (text = title+content
surface for embedding parity per Phase 2.3; metadata carries luke_id + provenance) so
live-written passages are indistinguishable from bulk-loaded ones.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import structlog

from .config import settings

log = structlog.get_logger()

_SEARCH_TIMEOUT_S = 3.0
_DELETE_TIMEOUT_S = 3.0
_CREATE_TIMEOUT_S = 8.0


def _base() -> str:
    return settings.letta_base_url.rstrip("/")


def _archive() -> str:
    return settings.letta_archive_id


def _find_passage_ids(surface: str, luke_id: str) -> list[str]:
    """Existing passage ids in the archive whose metadata.luke_id == luke_id.

    Uses semantic search on the exact title+content surface: the identical passage embeds
    to (near-)identical vectors and ranks at the top, so a modest over-fetch + exact
    luke_id filter reliably catches every copy. Returns [] on any failure (create-only).
    """
    body = json.dumps(
        {"query": surface[:512], "archive_id": _archive(), "limit": 25}
    ).encode()
    req = urllib.request.Request(
        f"{_base()}/v1/passages/search",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_SEARCH_TIMEOUT_S) as resp:
            data = json.load(resp)
    except Exception as exc:
        log.warning("letta_wt_find_failed", err=str(exc)[:120], luke_id=luke_id)
        return []
    if isinstance(data, dict):
        data = data.get("results") or data.get("passages") or []
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for p in data:
        obj = p.get("passage", p) if isinstance(p, dict) else {}
        meta = obj.get("metadata") or {}
        if meta.get("luke_id") == luke_id and obj.get("id"):
            ids.append(obj["id"])
    return ids


def _delete_passage(passage_id: str) -> None:
    req = urllib.request.Request(
        f"{_base()}/v1/archives/{_archive()}/passages/{passage_id}", method="DELETE"
    )
    with urllib.request.urlopen(req, timeout=_DELETE_TIMEOUT_S):
        pass


def _create_passage(text: str, metadata: dict[str, Any]) -> None:
    body = json.dumps({"text": text, "metadata": metadata}).encode()
    req = urllib.request.Request(
        f"{_base()}/v1/archives/{_archive()}/passages",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_CREATE_TIMEOUT_S):
        pass


def letta_write_through(mem_id: str) -> None:
    """Mirror a just-committed ``index_memory`` write into the Letta archive.

    Reads the current row from sqlite (single source of truth), so the Letta passage is an
    exact mirror of what was committed. No-op unless on the letta backend with write-through
    enabled. Never raises — the sqlite write already succeeded.
    """
    if (
        settings.memory_backend != "letta"
        or not settings.letta_write_through
        or not settings.letta_archive_id
    ):
        return
    try:
        _write_through_impl(mem_id)
    except Exception as exc:  # absolute backstop — remember() must not fail on Letta
        log.warning("letta_write_through_failed", err=str(exc)[:160], luke_id=mem_id)


def _write_through_impl(mem_id: str) -> None:
    from .memory import _db  # local import avoids a circular dependency

    db = _db()
    row = db.execute(
        """SELECT m.type, f.title, f.content, m.status, m.importance,
                  m.tags_json, m.links_json, m.created, m.updated,
                  m.access_count, m.useful_count, m.taxonomy
           FROM memory_meta m JOIN memory_fts f ON m.id = f.id
           WHERE m.id = ?""",
        (mem_id,),
    ).fetchone()
    if row is None:
        return  # nothing joinable (e.g. archived → dropped from FTS); delta-sync backstop

    content = row["content"] or ""
    if not content:
        return  # tombstone / metadata-only — never a recall passage

    title = row["title"] or ""
    surface = f"{title} {content}".strip()
    try:
        tags = json.loads(row["tags_json"]) if row["tags_json"] else []
    except Exception:
        tags = []

    # Links: mirror the LIVE graph (memory_links), not links_json. links_json is only the
    # creation-time snapshot passed to remember(); the authoritative, growing graph — the
    # explicit connect() edges plus the A-MEM auto-evolution edges — lives in memory_links.
    # Reading it here is what makes the connect() write-through faithful (Phase 2.2b): after
    # link_memories() commits a new edge, re-mirroring from_id picks the edge up here.
    # Only valid (non-invalidated) outgoing edges, matching _get_neighbors_batch semantics.
    try:
        link_rows = db.execute(
            "SELECT to_id FROM memory_links "
            "WHERE from_id = ? AND (valid_until IS NULL OR valid_until = '')",
            (mem_id,),
        ).fetchall()
        links = [r["to_id"] for r in link_rows]
    except Exception:
        links = []

    # Exact build_records metadata shape so live passages match bulk-loaded ones.
    metadata = {
        "luke_id": mem_id,
        "type": row["type"],
        "title": title,
        "status": row["status"],
        "importance": row["importance"],
        "tags": tags,
        "links": links,
        "created": row["created"],
        "updated": row["updated"],
        "access_count": row["access_count"],
        "useful_count": row["useful_count"],
        "taxonomy": row["taxonomy"],
        "content_source": "live_write_through",
        "is_tombstone": False,
    }

    # Idempotent upsert: drop existing copies for this luke_id, then write the fresh one.
    stale = _find_passage_ids(surface, mem_id)
    deleted = 0
    for pid in stale:
        try:
            _delete_passage(pid)
            deleted += 1
        except Exception as exc:
            log.warning("letta_wt_delete_failed", err=str(exc)[:120], pid=pid)
    _create_passage(surface, metadata)
    log.info("letta_write_through", luke_id=mem_id, replaced=deleted)
