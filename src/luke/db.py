"""SQLite database: messages, sessions, tasks, and memory index."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import sqlite3
import struct
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import structlog
import yaml
from pydantic import BaseModel

from .config import settings

log = structlog.get_logger()

_local = threading.local()  # thread-local: conn, batch_depth, has_vec


@contextlib.contextmanager
def batch() -> Iterator[None]:
    """Suppress intermediate commits; single commit on exit. Nestable.

    Rolls back on exception at the outermost level; commits on success.
    """
    depth = getattr(_local, "batch_depth", 0)
    _local.batch_depth = depth + 1
    ok = False
    try:
        yield
        ok = True
    finally:
        _local.batch_depth -= 1
        if _local.batch_depth == 0:
            conn: sqlite3.Connection | None = getattr(_local, "conn", None)
            if conn is not None:
                if ok:
                    conn.commit()
                else:
                    conn.rollback()


def _commit(conn: sqlite3.Connection) -> None:
    """Commit unless inside a batch() context."""
    if getattr(_local, "batch_depth", 0) == 0:
        conn.commit()


# ---------------------------------------------------------------------------
# Models & TypedDicts
# ---------------------------------------------------------------------------

MEMORY_DIRS: dict[str, str] = {
    "entity": "entities",
    "episode": "episodes",
    "procedure": "procedures",
    "insight": "insights",
    "goal": "goals",
}
_DIR_TO_TYPE: dict[str, str] = {v: k for k, v in MEMORY_DIRS.items()}


class StoredMessage(BaseModel):
    id: int
    sender_name: str
    sender_id: str
    message_id: int
    content: str
    timestamp: str
    reply_to: str | None = None


class MemoryResult(TypedDict):
    id: str
    type: str
    title: str
    score: float
    relationship: NotRequired[str]
    created: NotRequired[str]


class TaskRecord(TypedDict):
    id: str
    chat_id: str
    prompt: str
    schedule_type: str
    schedule_value: str
    status: str
    last_run: str | None
    created_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is UTC-aware. Assumes naive datetimes are UTC."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def _db() -> sqlite3.Connection:
    """Return a thread-local SQLite connection (WAL handles concurrent access)."""
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        settings.store_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(settings.store_dir / "luke.db")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={settings.db_busy_timeout}")
        # Performance: 64 MB mmap for zero-copy reads, 8 MB page cache, temp tables in RAM
        conn.execute("PRAGMA mmap_size=67108864")
        conn.execute("PRAGMA cache_size=-8192")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.row_factory = sqlite3.Row
        _load_vec_extension(conn)
        _local.conn = conn
    return conn


def _load_vec_extension(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec extension (required dependency)."""
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT NOT NULL,
    sender     TEXT NOT NULL,
    sender_id  TEXT NOT NULL DEFAULT '',
    msg_id     INTEGER NOT NULL DEFAULT 0,
    content    TEXT NOT NULL,
    ts         TEXT NOT NULL,
    reply_to   TEXT,
    media_file_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id, id);

CREATE TABLE IF NOT EXISTS cursors (
    chat_id TEXT PRIMARY KEY,
    last_id INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    chat_id    TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS tasks (
    id             TEXT PRIMARY KEY,
    chat_id        TEXT NOT NULL,
    prompt         TEXT NOT NULL,
    schedule_type  TEXT NOT NULL,
    schedule_value TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'active',
    last_run       TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_logs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id   TEXT NOT NULL,
    started   TEXT NOT NULL,
    finished  TEXT,
    result    TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    id UNINDEXED, type, title, content, tags,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS memory_meta (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,
    created       TEXT NOT NULL,
    updated       TEXT NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    useful_count  INTEGER NOT NULL DEFAULT 0,
    importance    REAL NOT NULL DEFAULT 1.0,
    status        TEXT NOT NULL DEFAULT 'active',
    tags_json     TEXT NOT NULL DEFAULT '[]',
    links_json    TEXT NOT NULL DEFAULT '[]',
    is_private    INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_memory_meta_created ON memory_meta(created);
CREATE INDEX IF NOT EXISTS idx_memory_meta_status ON memory_meta(status);
CREATE INDEX IF NOT EXISTS idx_memory_meta_type_status
    ON memory_meta(type, status, updated DESC);

CREATE TABLE IF NOT EXISTS reaction_feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     TEXT NOT NULL,
    msg_id      INTEGER NOT NULL,
    sender_id   TEXT NOT NULL,
    emoji       TEXT NOT NULL,
    sentiment   TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    UNIQUE(chat_id, msg_id, sender_id, emoji)
);

CREATE TABLE IF NOT EXISTS memory_links (
    from_id      TEXT NOT NULL,
    to_id        TEXT NOT NULL,
    relationship TEXT NOT NULL,
    weight       REAL NOT NULL DEFAULT 1.0,
    created      TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id, relationship)
);
CREATE INDEX IF NOT EXISTS idx_ml_to ON memory_links(to_id);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
    embedding float[768],
    +memory_id text
);

CREATE TABLE IF NOT EXISTS behavior_state (
    name     TEXT PRIMARY KEY,
    last_run TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    mem_id         TEXT NOT NULL,
    changed_fields TEXT NOT NULL,
    timestamp      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mh_mem ON memory_history(mem_id);

CREATE TABLE IF NOT EXISTS cost_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id         TEXT NOT NULL,
    cost_usd        REAL NOT NULL,
    num_turns       INTEGER NOT NULL,
    duration_api_ms INTEGER NOT NULL,
    source          TEXT NOT NULL DEFAULT 'message',
    timestamp       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_cost_ts ON cost_log(timestamp);
"""


def _vec_rowid(mem_id: str) -> int:
    """Deterministic rowid for sqlite-vec from a memory ID (stable across restarts)."""
    digest = hashlib.sha256(mem_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def init() -> None:
    """Create tables (including sqlite-vec virtual table)."""
    db = _db()
    db.executescript(_SCHEMA)
    db.commit()


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def store_message(
    *,
    chat_id: str,
    sender_name: str,
    sender_id: str = "",
    message_id: int = 0,
    content: str,
    timestamp: str,
    reply_to: str | None = None,
    media_file_id: str | None = None,
) -> bool:
    """Store a message. Returns False if duplicate detected."""
    conn = _db()
    # Duplicate check: same chat, sender, msg_id, and content
    if sender_id and message_id:
        existing = conn.execute(
            """SELECT 1 FROM messages
               WHERE chat_id = ? AND sender_id = ? AND msg_id = ? AND content = ?
               LIMIT 1""",
            (chat_id, sender_id, message_id, content),
        ).fetchone()
        if existing:
            return False
    conn.execute(
        """INSERT INTO messages
           (chat_id, sender, sender_id, msg_id, content, ts, reply_to, media_file_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (chat_id, sender_name, sender_id, message_id, content, timestamp, reply_to, media_file_id),
    )
    _commit(conn)
    return True


def get_pending_messages(chat_id: str) -> list[StoredMessage]:
    db = _db()
    cursor_row = db.execute("SELECT last_id FROM cursors WHERE chat_id = ?", (chat_id,)).fetchone()
    last_id = cursor_row["last_id"] if cursor_row else 0

    rows = db.execute(
        """SELECT id, sender, sender_id, msg_id, content, ts, reply_to
           FROM messages WHERE chat_id = ? AND id > ? ORDER BY id""",
        (chat_id, last_id),
    ).fetchall()

    return [
        StoredMessage(
            id=r["id"],
            sender_name=r["sender"],
            sender_id=r["sender_id"],
            message_id=r["msg_id"],
            content=r["content"],
            timestamp=r["ts"],
            reply_to=r["reply_to"],
        )
        for r in rows
    ]


def get_message_by_msg_id(chat_id: str, msg_id: str) -> dict[str, str] | None:
    """Look up a message by Telegram msg_id. Returns sender + content, or None."""
    row = (
        _db()
        .execute(
            "SELECT sender, content FROM messages WHERE chat_id = ? AND msg_id = ?",
            (chat_id, int(msg_id)),
        )
        .fetchone()
    )
    return {"sender": row["sender"], "content": row["content"]} if row else None


def advance_cursor(chat_id: str, last_id: int) -> None:
    db = _db()
    db.execute(
        "INSERT OR REPLACE INTO cursors (chat_id, last_id) VALUES (?, ?)",
        (chat_id, last_id),
    )
    _commit(db)


def get_recent_messages(chat_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return the last N messages for a chat, chronological order."""
    rows = (
        _db()
        .execute(
            """SELECT sender AS sender_name, content, ts AS timestamp
           FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?""",
            (chat_id, limit),
        )
        .fetchall()
    )
    return [dict(r) for r in reversed(rows)]


# ---------------------------------------------------------------------------
# Reaction feedback
# ---------------------------------------------------------------------------

_POSITIVE_EMOJIS = frozenset(
    {"👍", "❤️", "🔥", "🎉", "💯", "⭐", "🙏", "😂", "🤣", "👏", "💪", "❤", "😍", "🥰"}
)
_NEGATIVE_EMOJIS = frozenset({"👎", "😡", "🤮", "💩", "😤", "😠", "🤦"})


def classify_reaction(emoji: str) -> str:
    """Classify emoji sentiment: positive, negative, or neutral."""
    if emoji in _POSITIVE_EMOJIS:
        return "positive"
    if emoji in _NEGATIVE_EMOJIS:
        return "negative"
    return "neutral"


def store_reaction_feedback(
    *,
    chat_id: str,
    msg_id: int,
    sender_id: str,
    emoji: str,
    timestamp: str,
) -> None:
    """Store structured reaction feedback (idempotent via UNIQUE constraint)."""
    sentiment = classify_reaction(emoji)
    conn = _db()
    conn.execute(
        """INSERT OR IGNORE INTO reaction_feedback
           (chat_id, msg_id, sender_id, emoji, sentiment, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (chat_id, msg_id, sender_id, emoji, sentiment, timestamp),
    )
    _commit(conn)


def get_message_summaries(chat_id: str, days: int = 14) -> list[dict[str, Any]]:
    """Group messages by date for the last N days. Returns date + previews."""
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    rows = (
        _db()
        .execute(
            """SELECT DATE(ts) AS date, sender AS sender_name,
                  SUBSTR(content, 1, 120) AS preview
           FROM messages WHERE chat_id = ? AND ts >= ? ORDER BY ts""",
            (chat_id, cutoff),
        )
        .fetchall()
    )
    by_date: dict[str, list[str]] = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(f"  {r['sender_name']}: {r['preview']}")
    return [{"date": d, "messages": msgs} for d, msgs in by_date.items()]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def get_session(chat_id: str) -> str | None:
    row = _db().execute("SELECT session_id FROM sessions WHERE chat_id = ?", (chat_id,)).fetchone()
    return row["session_id"] if row and row["session_id"] else None


def set_session(chat_id: str, session_id: str) -> None:
    db = _db()
    now = datetime.now(UTC).isoformat()
    db.execute(
        "INSERT OR REPLACE INTO sessions (chat_id, session_id, updated_at) VALUES (?, ?, ?)",
        (chat_id, session_id, now),
    )
    _commit(db)


def clear_sessions() -> None:
    """Clear all sessions — no Claude subprocess session survives a process restart."""
    conn = _db()
    conn.execute("DELETE FROM sessions")
    _commit(conn)


def cleanup_stale_sessions(timeout_seconds: float) -> int:
    """Remove sessions not updated within timeout_seconds. Returns count."""
    cutoff = (datetime.now(UTC) - timedelta(seconds=timeout_seconds)).isoformat()
    conn = _db()
    cur = conn.execute(
        "DELETE FROM sessions WHERE updated_at < ? AND updated_at != ''",
        (cutoff,),
    )
    _commit(conn)
    return cur.rowcount


# ---------------------------------------------------------------------------
# Tasks (scheduling)
# ---------------------------------------------------------------------------


_VALID_SCHEDULE_TYPES = {"cron", "interval", "once"}


def create_task(
    chat_id: str,
    prompt: str,
    schedule_type: str,
    schedule_value: str,
) -> str:
    if schedule_type not in _VALID_SCHEDULE_TYPES:
        msg = f"Invalid schedule_type: {schedule_type}"
        raise ValueError(msg)
    if schedule_type == "cron":
        from croniter import croniter

        if not croniter.is_valid(schedule_value):
            msg = f"Invalid cron expression: {schedule_value}"
            raise ValueError(msg)
    if schedule_type == "once":
        try:
            datetime.fromisoformat(schedule_value)
        except ValueError:
            msg = f"Invalid ISO timestamp for once schedule: {schedule_value}"
            raise ValueError(msg) from None
    if schedule_type == "interval":
        try:
            val = int(schedule_value)
        except ValueError:
            msg = f"Invalid interval (must be milliseconds): {schedule_value}"
            raise ValueError(msg) from None
        if val <= 0:
            msg = f"Interval must be positive: {schedule_value}"
            raise ValueError(msg)
    task_id = str(uuid.uuid4())[:8]
    now = datetime.now(UTC).isoformat()
    db = _db()
    db.execute(
        """INSERT INTO tasks
           (id, chat_id, prompt, schedule_type, schedule_value, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (task_id, chat_id, prompt, schedule_type, schedule_value, now),
    )
    _commit(db)
    return task_id


def get_due_tasks() -> list[TaskRecord]:
    rows = (
        _db()
        .execute(
            """SELECT * FROM tasks WHERE status = 'active'
               AND NOT (schedule_type = 'once' AND last_run IS NOT NULL)""",
        )
        .fetchall()
    )
    return cast(list[TaskRecord], [dict(r) for r in rows])


def log_task_run(
    task_id: str,
    started: str,
    finished: str | None = None,
    result: str | None = None,
) -> None:
    db = _db()
    db.execute(
        "INSERT INTO task_logs (task_id, started, finished, result) VALUES (?, ?, ?, ?)",
        (task_id, started, finished, result),
    )
    _commit(db)


def update_task_last_run(task_id: str, ts: str) -> None:
    db = _db()
    db.execute("UPDATE tasks SET last_run = ? WHERE id = ?", (ts, task_id))
    _commit(db)


def update_task_status(task_id: str, status: str) -> None:
    db = _db()
    db.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    _commit(db)


# ---------------------------------------------------------------------------
# Behavior state
# ---------------------------------------------------------------------------


def get_behavior_last_run(name: str) -> str | None:
    """Return the ISO timestamp of the last run for a behavior, or None."""
    row = _db().execute("SELECT last_run FROM behavior_state WHERE name = ?", (name,)).fetchone()
    return row["last_run"] if row else None


def set_behavior_last_run(name: str, timestamp: str) -> None:
    """Persist the last run timestamp for a behavior."""
    conn = _db()
    conn.execute(
        "INSERT OR REPLACE INTO behavior_state (name, last_run) VALUES (?, ?)",
        (name, timestamp),
    )
    _commit(conn)


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
    # vec0 KNN: auxiliary columns can be SELECTed but not filtered in WHERE
    fetch_limit = limit * 4
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
        f"""SELECT m.id, m.type, f.title
            FROM memory_meta m
            JOIN memory_fts f ON m.id = f.id
            WHERE m.id IN ({mid_placeholders}) AND {filter_clause}""",
        params,
    ).fetchall()

    scored: list[dict[str, Any]] = []
    for r in meta_rows:
        dist = mid_dist.get(r["id"], 0.0)
        similarity = 1.0 / (1.0 + dist)
        scored.append({"id": r["id"], "type": r["type"], "title": r["title"], "score": similarity})
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
    """Apply type-aware importance decay modulated by access_count. Returns updated count."""
    db = _db()
    total = 0
    for mem_type, base_rate in rates.items():
        cur = db.execute(
            """UPDATE memory_meta
               SET importance = importance * (1.0 - (1.0 - ?) / (1.0 + access_count * 0.1))
               WHERE type = ? AND status = 'active'""",
            (base_rate, mem_type),
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


# ---------------------------------------------------------------------------
# Memory index
# ---------------------------------------------------------------------------


def index_memory(
    mem_id: str,
    mem_type: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    links: list[str] | None = None,
    importance: float | None = None,
) -> None:
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

    # Upsert metadata — read existing values BEFORE INSERT OR REPLACE deletes the row
    existing = db.execute(
        "SELECT created, access_count, useful_count, importance, last_accessed"
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
    tags_json = json.dumps(tags)
    links_json = json.dumps(links)
    is_private = 1 if "private" in tags else 0
    db.execute(
        """INSERT OR REPLACE INTO memory_meta
           (id, type, created, updated,
            access_count, useful_count, importance, status,
            tags_json, links_json, is_private, last_accessed)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)""",
        (
            mem_id,
            mem_type,
            created,
            now,
            access_count,
            useful_count,
            importance,
            tags_json,
            links_json,
            is_private,
            last_accessed,
        ),
    )

    # Update links
    for link_id in links:
        db.execute(
            """INSERT OR IGNORE INTO memory_links (from_id, to_id, relationship, weight, created)
               VALUES (?, ?, 'related', 1.0, ?)""",
            (mem_id, link_id, now),
        )

    # Embed for semantic search (None on runtime errors — FTS still works)
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
) -> list[MemoryResult]:
    """Unified memory search: FTS5 + embeddings + temporal + multi-hop graph + composite scoring."""
    db = _db()
    results: dict[str, dict[str, Any]] = {}
    priv_clause = "" if include_private else " AND m.is_private = 0"

    # --- Strategy 1: FTS5 ranked search ---
    fts_ranked: list[str] = []
    fts_query = _sanitize_fts_query(query) if query else ""
    if fts_query:
        try:
            type_clause = " AND m.type = ?" if mem_type else ""
            type_params: tuple[str, ...] = (mem_type,) if mem_type else ()
            rows = db.execute(
                f"""SELECT f.id, f.type, f.title, rank
                    FROM memory_fts f
                    JOIN memory_meta m ON f.id = m.id
                    WHERE memory_fts MATCH ? AND m.status = 'active'
                    {type_clause}{priv_clause}
                    ORDER BY rank
                    LIMIT ?""",
                (fts_query, *type_params, limit),
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
            }

    # --- Strategy 2: Semantic search (embedding-based) ---
    sem_ranked: list[str] = []
    if query:
        query_embedding = _embed_query(query)
        if query_embedding is not None:
            sem_results = _semantic_search(
                query_embedding,
                mem_type=mem_type,
                limit=limit,
                include_private=include_private,
            )
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
        params: list[str] = []
        if after:
            conditions.append("m.updated >= ?")
            params.append(after)
        if before:
            conditions.append("m.updated <= ?")
            params.append(before)
        if mem_type:
            conditions.append("m.type = ?")
            params.append(mem_type)

        rows = db.execute(
            f"""SELECT m.id, m.type, f.title
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
                }

    # --- Strategy 4: Multi-hop graph traversal (BFS, depth up to graph_max_depth) ---
    if related_to:
        visited: set[str] = {related_to}
        frontier = [related_to]
        for depth in range(settings.graph_max_depth):
            hop_weight = settings.graph_decay_per_hop ** (depth + 1)
            next_frontier: list[str] = []
            for node in frontier:
                neighbors = _get_neighbors(db, node)
                for n in neighbors:
                    if n["id"] not in visited:
                        visited.add(n["id"])
                        next_frontier.append(n["id"])
                        if n["id"] not in results:
                            results[n["id"]] = {
                                "id": n["id"],
                                "type": n["type"],
                                "title": n["title"],
                                "score": n["weight"] * hop_weight,
                                "relationship": n["relationship"],
                            }
            frontier = next_frontier

    # --- Type filter (when only type is specified, no query/temporal/graph) ---
    if mem_type and not query and not after and not before and not related_to:
        rows = db.execute(
            f"""SELECT m.id, m.type, f.title
                FROM memory_meta m
                JOIN memory_fts f ON m.id = f.id
                WHERE m.type = ? AND m.status = 'active'{priv_clause}
                ORDER BY m.updated DESC
                LIMIT ?""",
            (mem_type, limit),
        ).fetchall()
        for r in rows:
            results[r["id"]] = {
                "id": r["id"],
                "type": r["type"],
                "title": r["title"],
                "score": 0.0,
            }

    # Apply type filter to combined results
    if mem_type and (query or after or before or related_to):
        results = {k: v for k, v in results.items() if v["type"] == mem_type}

    # --- Composite scoring: relevance * importance * recency * access frequency ---
    _apply_composite_scores(results)

    ranked = sorted(results.values(), key=lambda x: x.get("score", 0), reverse=True)[:limit]
    return cast(list[MemoryResult], ranked)


def _get_neighbors(db: sqlite3.Connection, node_id: str) -> list[dict[str, Any]]:
    """Get bidirectional graph neighbors of a memory node."""
    rows = db.execute(
        """SELECT ml.to_id AS id, m.type, f.title, ml.relationship, ml.weight
           FROM memory_links ml
           JOIN memory_meta m ON ml.to_id = m.id
           JOIN memory_fts f ON ml.to_id = f.id
           WHERE ml.from_id = ? AND m.status = 'active'
           UNION
           SELECT ml.from_id AS id, m.type, f.title, ml.relationship, ml.weight
           FROM memory_links ml
           JOIN memory_meta m ON ml.from_id = m.id
           JOIN memory_fts f ON ml.from_id = f.id
           WHERE ml.to_id = ? AND m.status = 'active'""",
        (node_id, node_id),
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
    """Apply composite scoring (relevance * importance * recency * access) in place."""
    if not results:
        return
    conn = _db()
    ids = list(results.keys())
    placeholders = ",".join("?" for _ in ids)
    meta_rows = conn.execute(
        f"""SELECT id, importance, access_count, useful_count, updated
            FROM memory_meta WHERE id IN ({placeholders})""",
        ids,
    ).fetchall()
    meta_map: dict[str, sqlite3.Row] = {r["id"]: r for r in meta_rows}

    # Loop-invariant constants
    context_denom = 1.0 - settings.score_weight_relevance
    log_101 = math.log(101)
    w_imp = settings.score_weight_importance
    w_rec = settings.score_weight_recency
    w_acc = settings.score_weight_access

    for mem_id, entry in results.items():
        meta = meta_map.get(mem_id)
        relevance = entry.get("score", 0)
        importance = min(meta["importance"], 1.0) if meta else 1.0
        access_count: int = meta["access_count"] if meta else 0
        useful_count: int = meta["useful_count"] if meta else 0
        updated: str = meta["updated"] if meta else ""
        recency = _recency_score(updated)
        access_score = math.log(1 + access_count) / log_101  # ~1.0 at 100 accesses
        # Utility modulation: penalize memories that get accessed but never used
        utility_rate = useful_count / max(access_count, 1)
        access_score *= settings.utility_floor + settings.utility_weight * utility_rate

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


def touch_memories(mem_ids: list[str], *, useful: bool = True) -> None:
    """Increment access_count and update last_accessed. If *useful*, also increment
    useful_count and strengthen co-access graph links (Hebbian learning)."""
    if not mem_ids:
        return
    conn = _db()
    now = datetime.now(UTC).isoformat()
    ph = ",".join("?" for _ in mem_ids)
    set_clause = (
        "access_count = access_count + 1, useful_count = useful_count + 1, last_accessed = ?"
        if useful
        else "access_count = access_count + 1, last_accessed = ?"
    )
    conn.execute(f"UPDATE memory_meta SET {set_clause} WHERE id IN ({ph})", (now, *mem_ids))
    # Hebbian co-access: strengthen existing links between co-recalled memories
    if useful and len(mem_ids) > 1:
        conn.execute(
            f"""UPDATE memory_links SET weight = MIN(weight + 0.05, 5.0)
                WHERE from_id IN ({ph}) AND to_id IN ({ph})""",
            (*mem_ids, *mem_ids),
        )
    _commit(conn)


def get_graph_neighbors(mem_ids: list[str], *, limit: int = 3) -> list[MemoryResult]:
    """Get unique 1-hop neighbors of the given memory IDs, excluding the inputs."""
    if not mem_ids:
        return []
    conn = _db()
    ph = ",".join("?" for _ in mem_ids)
    rows = conn.execute(
        f"""SELECT DISTINCT m.id, m.type, f.title, ml.relationship
            FROM memory_links ml
            JOIN memory_meta m ON ml.to_id = m.id
            JOIN memory_fts f ON ml.to_id = f.id
            WHERE ml.from_id IN ({ph}) AND m.status = 'active'
              AND ml.to_id NOT IN ({ph})
            UNION
            SELECT DISTINCT m.id, m.type, f.title, ml.relationship
            FROM memory_links ml
            JOIN memory_meta m ON ml.from_id = m.id
            JOIN memory_fts f ON ml.from_id = f.id
            WHERE ml.to_id IN ({ph}) AND m.status = 'active'
              AND ml.from_id NOT IN ({ph})
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
    conn = _db()
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        """INSERT OR IGNORE INTO memory_links (from_id, to_id, relationship, weight, created)
           VALUES (?, ?, ?, 1.0, ?)""",
        (from_id, to_id, relationship, now),
    )
    _commit(conn)
    return cur.rowcount > 0


def cleanup_archived_fts() -> None:
    """Remove archived memories from FTS5 index."""
    db = _db()
    db.execute(
        "DELETE FROM memory_fts WHERE id IN (SELECT id FROM memory_meta WHERE status = 'archived')"
    )
    _commit(db)


def cleanup_task_logs(retention_days: int = 90) -> int:
    """Delete task_logs older than retention_days. Returns count."""
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
    conn = _db()
    cur = conn.execute("DELETE FROM task_logs WHERE started < ?", (cutoff,))
    _commit(conn)
    return cur.rowcount


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
                    )
                except Exception:
                    log.warning("memory_index_skip", file=str(md_file))
                    continue


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


def log_cost(
    chat_id: str,
    cost_usd: float,
    num_turns: int,
    duration_api_ms: int,
    source: str = "message",
) -> None:
    """Record cost and usage for an agent run."""
    db = _db()
    db.execute(
        "INSERT INTO cost_log (chat_id, cost_usd, num_turns, duration_api_ms, source) "
        "VALUES (?, ?, ?, ?, ?)",
        (chat_id, cost_usd, num_turns, duration_api_ms, source),
    )
    _commit(db)


def get_cost_report(period: str = "month") -> str:
    """Aggregate cost stats for a time period. Returns formatted text."""
    db = _db()
    conditions: dict[str, str] = {
        "today": "timestamp >= date('now')",
        "week": "timestamp >= date('now', '-7 days')",
        "month": "timestamp >= date('now', '-30 days')",
        "all": "1=1",
    }
    where = conditions.get(period, conditions["month"])

    row = db.execute(
        f"SELECT COALESCE(SUM(cost_usd), 0) AS total, "
        f"COUNT(*) AS runs, "
        f"COALESCE(SUM(num_turns), 0) AS turns, "
        f"COALESCE(AVG(cost_usd), 0) AS avg_cost "
        f"FROM cost_log WHERE {where}",
    ).fetchone()
    assert row is not None

    by_source = db.execute(
        f"SELECT source, COALESCE(SUM(cost_usd), 0) AS total, COUNT(*) AS runs "
        f"FROM cost_log WHERE {where} GROUP BY source ORDER BY total DESC",
    ).fetchall()

    lines = [
        f"Cost report ({period}):",
        f"  Total: ${row['total']:.4f}",
        f"  Runs: {row['runs']}",
        f"  Turns: {row['turns']}",
        f"  Avg per run: ${row['avg_cost']:.4f}",
    ]
    if by_source:
        lines.append("  By source:")
        for s in by_source:
            lines.append(f"    {s['source']}: ${s['total']:.4f} ({s['runs']} runs)")

    by_day = db.execute(
        f"SELECT date(timestamp) AS day, COALESCE(SUM(cost_usd), 0) AS total "
        f"FROM cost_log WHERE {where} GROUP BY day ORDER BY day DESC LIMIT 7",
    ).fetchall()
    if by_day:
        lines.append("  Recent days:")
        for d in by_day:
            lines.append(f"    {d['day']}: ${d['total']:.4f}")

    return "\n".join(lines)
