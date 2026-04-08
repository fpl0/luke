"""SQLite database: messages, sessions, tasks, cost tracking."""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, TypedDict, cast

import structlog
from pydantic import BaseModel
from structlog.stdlib import BoundLogger

from .config import settings

log: BoundLogger = structlog.get_logger()

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


class StoredMessage(BaseModel):
    id: int
    sender_name: str
    sender_id: str
    message_id: int
    content: str
    timestamp: str
    reply_to: str | None = None


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
    valid_until  TEXT,
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

CREATE TABLE IF NOT EXISTS outbound_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    timestamp    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_outbound_hash ON outbound_log(chat_id, content_hash);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}',
    created    TEXT NOT NULL DEFAULT (datetime('now')),
    consumed   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type, consumed, created);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

"""

# ---------------------------------------------------------------------------
# Versioned migrations
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (
        1,
        "add consecutive_failures to tasks",
        [
            "ALTER TABLE tasks ADD COLUMN consecutive_failures INTEGER NOT NULL DEFAULT 0",
        ],
    ),
    (
        2,
        "add temporal validity to memory_links",
        [
            "ALTER TABLE memory_links ADD COLUMN valid_until TEXT",
        ],
    ),
    (
        3,
        "add events table for event-driven behaviors",
        [
            """CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload    TEXT NOT NULL DEFAULT '{}',
                created    TEXT NOT NULL DEFAULT (datetime('now')),
                consumed   INTEGER NOT NULL DEFAULT 0
            )""",
            "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type, consumed, created)",
        ],
    ),
    (
        4,
        "add consecutive_no_ops to behavior_state for smart backoff",
        [
            "ALTER TABLE behavior_state ADD COLUMN consecutive_no_ops INTEGER NOT NULL DEFAULT 0",
        ],
    ),
    (
        5,
        "add deep_work_quality table for session quality tracking",
        [
            """CREATE TABLE IF NOT EXISTS deep_work_quality (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id      TEXT NOT NULL,
                rating       INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                session_date TEXT NOT NULL DEFAULT (datetime('now')),
                created      TEXT NOT NULL DEFAULT (datetime('now'))
            )""",
            "CREATE INDEX IF NOT EXISTS idx_dwq_goal ON deep_work_quality(goal_id, created DESC)",
        ],
    ),
    (
        6,
        "add taxonomy column to memory_meta for factual/experiential/working classification",
        [
            "ALTER TABLE memory_meta ADD COLUMN taxonomy TEXT NOT NULL DEFAULT ''",
            "CREATE INDEX IF NOT EXISTS idx_memory_meta_taxonomy ON memory_meta(taxonomy)",
        ],
    ),
    (
        7,
        "repair: ensure migrations 5-6 objects exist (fixes ghost-recorded versions)",
        [
            # Re-create table/column that may have been ghost-recorded
            """CREATE TABLE IF NOT EXISTS deep_work_quality (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id      TEXT NOT NULL,
                rating       INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                session_date TEXT NOT NULL DEFAULT (datetime('now')),
                created      TEXT NOT NULL DEFAULT (datetime('now'))
            )""",
            "CREATE INDEX IF NOT EXISTS idx_dwq_goal ON deep_work_quality(goal_id, created DESC)",
            "ALTER TABLE memory_meta ADD COLUMN taxonomy TEXT NOT NULL DEFAULT ''",
            "CREATE INDEX IF NOT EXISTS idx_memory_meta_taxonomy ON memory_meta(taxonomy)",
        ],
    ),
    (
        8,
        "add skill_meta column and skill_triggers table for self-improving skill loop",
        [
            "ALTER TABLE memory_meta ADD COLUMN skill_meta TEXT",
            """CREATE TABLE IF NOT EXISTS skill_triggers (
                procedure_id  TEXT PRIMARY KEY REFERENCES memory_meta(id),
                trigger_pattern TEXT NOT NULL,
                confidence    REAL NOT NULL DEFAULT 0.5,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_applied  TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_skill_confidence ON skill_triggers(confidence DESC)",
        ],
    ),
    (
        9,
        "add clustering columns and tables for semantic memory organization",
        [
            "ALTER TABLE memory_meta ADD COLUMN cluster_id INTEGER DEFAULT NULL",
            "ALTER TABLE memory_meta ADD COLUMN cluster_updated_at TEXT DEFAULT NULL",
            "ALTER TABLE memory_meta ADD COLUMN cluster_assignment_method TEXT DEFAULT NULL",
            "CREATE INDEX IF NOT EXISTS idx_memory_meta_cluster ON memory_meta(cluster_id)",
            """CREATE TABLE IF NOT EXISTS cluster_centroids (
                cluster_id INTEGER PRIMARY KEY,
                centroid_embedding BLOB,
                member_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                method TEXT NOT NULL DEFAULT 'birch'
            )""",
            """CREATE TABLE IF NOT EXISTS cluster_summaries (
                cluster_id INTEGER PRIMARY KEY,
                summary_text TEXT NOT NULL,
                summary_embedding BLOB,
                memory_count INTEGER NOT NULL,
                type_distribution TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (cluster_id) REFERENCES cluster_centroids(cluster_id)
            )""",
        ],
    ),
    (
        10,
        "add pending_corrections table and enhance memory_history with content snapshots",
        [
            """CREATE TABLE IF NOT EXISTS pending_corrections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                mem_id          TEXT NOT NULL,
                corrected_content TEXT NOT NULL,
                confidence      REAL NOT NULL,
                source          TEXT NOT NULL DEFAULT 'auto_detection',
                status          TEXT NOT NULL DEFAULT 'pending',
                created_at      TEXT NOT NULL,
                resolved_at     TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_pc_mem ON pending_corrections(mem_id)",
            "CREATE INDEX IF NOT EXISTS idx_pc_status ON pending_corrections(status)",
            "ALTER TABLE memory_history ADD COLUMN old_content TEXT",
            "ALTER TABLE memory_history ADD COLUMN new_content TEXT",
            "ALTER TABLE memory_history ADD COLUMN change_type TEXT",
            "ALTER TABLE memory_history ADD COLUMN source TEXT",
            "ALTER TABLE memory_history ADD COLUMN confidence REAL",
        ],
    ),
    (
        10,
        "add pending_corrections table and enhance memory_history with content snapshots",
        [
            """CREATE TABLE IF NOT EXISTS pending_corrections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                mem_id          TEXT NOT NULL,
                corrected_content TEXT NOT NULL,
                confidence      REAL NOT NULL,
                source          TEXT NOT NULL DEFAULT 'auto_detection',
                status          TEXT NOT NULL DEFAULT 'pending',
                created_at      TEXT NOT NULL,
                resolved_at     TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_pc_mem ON pending_corrections(mem_id)",
            "CREATE INDEX IF NOT EXISTS idx_pc_status ON pending_corrections(status)",
            "ALTER TABLE memory_history ADD COLUMN old_content TEXT",
            "ALTER TABLE memory_history ADD COLUMN new_content TEXT",
            "ALTER TABLE memory_history ADD COLUMN change_type TEXT",
            "ALTER TABLE memory_history ADD COLUMN source TEXT",
            "ALTER TABLE memory_history ADD COLUMN confidence REAL",
        ],
    ),
    (
        11,
        "add cache token tracking to cost_log for prompt caching analysis",
        [
            "ALTER TABLE cost_log ADD COLUMN input_tokens INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE cost_log ADD COLUMN output_tokens INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE cost_log ADD COLUMN cache_create_tokens INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE cost_log ADD COLUMN cache_read_tokens INTEGER NOT NULL DEFAULT 0",
        ],
    ),
    (
        12,
        "add compression_audit table for tracking information retention during compaction",
        [
            """CREATE TABLE IF NOT EXISTS compression_audit (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                created             TEXT NOT NULL DEFAULT (datetime('now')),
                messages_compressed INTEGER NOT NULL DEFAULT 0,
                messages_kept       INTEGER NOT NULL DEFAULT 0,
                goals_expected      INTEGER NOT NULL DEFAULT 0,
                goals_preserved     INTEGER NOT NULL DEFAULT 0,
                entities_expected   INTEGER NOT NULL DEFAULT 0,
                entities_preserved  INTEGER NOT NULL DEFAULT 0,
                memory_ids_expected INTEGER NOT NULL DEFAULT 0,
                memory_ids_preserved INTEGER NOT NULL DEFAULT 0,
                retention_score     REAL NOT NULL DEFAULT 0.0,
                summary_tokens      INTEGER NOT NULL DEFAULT 0,
                identity_anchor     INTEGER NOT NULL DEFAULT 0
            )""",
            "CREATE INDEX IF NOT EXISTS idx_ca_created ON compression_audit(created DESC)",
        ],
    ),
]


def _run_migrations(db: sqlite3.Connection) -> None:
    """Run pending schema migrations. Each runs exactly once."""
    row = db.execute("SELECT MAX(version) FROM schema_version").fetchone()
    current: int = row[0] if row[0] is not None else 0
    for version, description, statements in _MIGRATIONS:
        if version <= current:
            continue
        for sql in statements:
            try:
                db.execute(sql)
            except sqlite3.OperationalError as exc:
                # Tolerate "already exists" / "duplicate column" from re-runs,
                # but let genuine failures propagate so the version isn't
                # recorded for a migration that didn't actually apply.
                msg = str(exc).lower()
                if "already exists" in msg or "duplicate column" in msg:
                    pass
                else:
                    raise
        db.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        db.commit()
        log.info("migration_applied", version=version, description=description)


def get_schema_version() -> int:
    """Return the current schema version, or 0 if unversioned."""
    row = _db().execute("SELECT MAX(version) FROM schema_version").fetchone()
    return int(row[0]) if row[0] is not None else 0


def init() -> None:
    """Create tables (including sqlite-vec virtual table) and run migrations."""
    db = _db()
    db.executescript(_SCHEMA)
    _run_migrations(db)


# ---------------------------------------------------------------------------
# Skill triggers
# ---------------------------------------------------------------------------


def upsert_skill_trigger(
    procedure_id: str,
    trigger_pattern: str,
    confidence: float = 0.5,
    success_count: int = 0,
    failure_count: int = 0,
) -> None:
    """Insert or update a skill trigger entry for a procedure memory."""
    conn = _db()
    conn.execute(
        """INSERT INTO skill_triggers
           (procedure_id, trigger_pattern, confidence, success_count, failure_count)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(procedure_id) DO UPDATE SET
             trigger_pattern = excluded.trigger_pattern,
             confidence = excluded.confidence,
             success_count = excluded.success_count,
             failure_count = excluded.failure_count""",
        (procedure_id, trigger_pattern, confidence, success_count, failure_count),
    )
    _commit(conn)


def get_matching_skill_triggers(text: str) -> list[dict[str, Any]]:
    """Return skill triggers whose regex pattern matches the given text.

    Only returns triggers with confidence >= 0.2 (above retirement threshold).
    Results sorted by confidence descending.
    """
    import re as _re

    conn = _db()
    rows = conn.execute(
        """SELECT st.procedure_id, st.trigger_pattern, st.confidence,
                  st.success_count, st.failure_count, st.last_applied
           FROM skill_triggers st
           JOIN memory_meta m ON st.procedure_id = m.id
           WHERE m.status = 'active' AND st.confidence >= 0.2
           ORDER BY st.confidence DESC""",
    ).fetchall()
    matches: list[dict[str, Any]] = []
    for r in rows:
        try:
            if _re.search(r["trigger_pattern"], text, _re.IGNORECASE):
                matches.append(dict(r))
        except _re.error:
            pass  # skip invalid regex patterns
    return matches


def get_skill_trigger(procedure_id: str) -> dict[str, Any] | None:
    """Get skill trigger for a procedure, or None."""
    row = (
        _db()
        .execute("SELECT * FROM skill_triggers WHERE procedure_id = ?", (procedure_id,))
        .fetchone()
    )
    return dict(row) if row else None


def delete_skill_trigger(procedure_id: str) -> None:
    """Remove a skill trigger entry."""
    conn = _db()
    conn.execute("DELETE FROM skill_triggers WHERE procedure_id = ?", (procedure_id,))
    _commit(conn)


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
    # Emit event for negative reactions — triggers reflection behavior
    if sentiment == "negative":
        from .bus import bus  # deferred to avoid circular import

        bus.emit("feedback_negative", {"msg_id": msg_id, "emoji": emoji})


def get_reactions(
    chat_id: str,
    *,
    msg_id: int | None = None,
    sender_id: str | None = None,
    sentiment: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Query stored reactions with optional filters. Returns newest first."""
    clauses = ["r.chat_id = ?"]
    params: list[Any] = [chat_id]
    if msg_id is not None:
        clauses.append("r.msg_id = ?")
        params.append(msg_id)
    if sender_id is not None:
        clauses.append("r.sender_id = ?")
        params.append(sender_id)
    if sentiment is not None:
        clauses.append("r.sentiment = ?")
        params.append(sentiment)
    where = " AND ".join(clauses)
    params.append(limit)
    rows = (
        _db()
        .execute(
            f"""SELECT r.msg_id, r.sender_id, r.emoji, r.sentiment, r.timestamp,
                       m.sender AS msg_sender, SUBSTR(m.content, 1, 120) AS msg_preview
                FROM reaction_feedback r
                LEFT JOIN messages m ON m.chat_id = r.chat_id AND m.msg_id = r.msg_id
                WHERE {where}
                ORDER BY r.timestamp DESC
                LIMIT ?""",
            params,
        )
        .fetchall()
    )
    return [dict(r) for r in rows]


def get_reaction_summary(chat_id: str, days: int = 7) -> dict[str, Any]:
    """Aggregate reaction stats for the given period.

    Returns sentiment counts, top emojis, and which message types
    (by sender: Luke vs user) get the most reactions.
    """
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    conn = _db()

    # Sentiment breakdown
    sentiment_rows = conn.execute(
        """SELECT sentiment, COUNT(*) AS cnt
           FROM reaction_feedback
           WHERE chat_id = ? AND timestamp >= ?
           GROUP BY sentiment""",
        (chat_id, cutoff),
    ).fetchall()
    sentiments = {r["sentiment"]: r["cnt"] for r in sentiment_rows}

    # Top emojis
    emoji_rows = conn.execute(
        """SELECT emoji, COUNT(*) AS cnt
           FROM reaction_feedback
           WHERE chat_id = ? AND timestamp >= ?
           GROUP BY emoji ORDER BY cnt DESC LIMIT 5""",
        (chat_id, cutoff),
    ).fetchall()
    top_emojis = [(r["emoji"], r["cnt"]) for r in emoji_rows]

    # Reactions on Luke's messages vs others
    sender_rows = conn.execute(
        """SELECT m.sender AS msg_sender, r.sentiment, COUNT(*) AS cnt
           FROM reaction_feedback r
           LEFT JOIN messages m ON m.chat_id = r.chat_id AND m.msg_id = r.msg_id
           WHERE r.chat_id = ? AND r.timestamp >= ?
           GROUP BY m.sender, r.sentiment""",
        (chat_id, cutoff),
    ).fetchall()
    by_sender: dict[str, dict[str, int]] = {}
    for r in sender_rows:
        sender = r["msg_sender"] or "unknown"
        by_sender.setdefault(sender, {})[r["sentiment"]] = r["cnt"]

    total = sum(sentiments.values())
    return {
        "total": total,
        "sentiments": sentiments,
        "top_emojis": top_emojis,
        "by_sender": by_sender,
        "period_days": days,
    }


def get_message_summaries(chat_id: str, days: int = 14) -> list[dict[str, Any]]:
    """Group messages by date for the last N days. Returns date + previews."""
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    rows = (
        _db()
        .execute(
            """SELECT DATE(ts) AS date, sender AS sender_name,
                  SUBSTR(content, 1, 120) AS preview
           FROM messages WHERE chat_id = ? AND ts >= ?
           ORDER BY ts DESC LIMIT 500""",
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


def cleanup_stale_sessions(timeout_seconds: float) -> list[str]:
    """Remove sessions not updated within timeout_seconds. Returns cleaned chat IDs."""
    cutoff = (datetime.now(UTC) - timedelta(seconds=timeout_seconds)).isoformat()
    conn = _db()
    rows = conn.execute(
        "DELETE FROM sessions WHERE updated_at < ? AND updated_at != '' RETURNING chat_id",
        (cutoff,),
    ).fetchall()
    _commit(conn)
    return [str(r[0]) for r in rows]


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


def list_tasks(chat_id: str) -> list[TaskRecord]:
    rows = (
        _db()
        .execute(
            "SELECT * FROM tasks WHERE chat_id = ? ORDER BY created_at DESC",
            (chat_id,),
        )
        .fetchall()
    )
    return cast(list[TaskRecord], [dict(r) for r in rows])


def delete_task(task_id: str) -> bool:
    db = _db()
    db.execute("DELETE FROM task_logs WHERE task_id = ?", (task_id,))
    cur = db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    _commit(db)
    return cur.rowcount > 0


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


def get_behavior_no_ops(name: str) -> int:
    """Return consecutive no-op count for a behavior."""
    row = (
        _db()
        .execute("SELECT consecutive_no_ops FROM behavior_state WHERE name = ?", (name,))
        .fetchone()
    )
    return int(row["consecutive_no_ops"]) if row and row["consecutive_no_ops"] else 0


def increment_behavior_no_ops(name: str) -> int:
    """Increment no-op counter. Returns new value."""
    conn = _db()
    conn.execute(
        "UPDATE behavior_state SET consecutive_no_ops = consecutive_no_ops + 1 WHERE name = ?",
        (name,),
    )
    _commit(conn)
    return get_behavior_no_ops(name)


def reset_behavior_no_ops(name: str) -> None:
    """Reset no-op counter to 0 (behavior produced useful output)."""
    conn = _db()
    conn.execute(
        "UPDATE behavior_state SET consecutive_no_ops = 0 WHERE name = ?",
        (name,),
    )
    _commit(conn)


# ---------------------------------------------------------------------------
# Events (for event-driven behavior triggers)
# ---------------------------------------------------------------------------


def emit_event(event_type: str, payload: str = "{}") -> int:
    """Emit an event. Returns the event ID."""
    conn = _db()
    cur = conn.execute(
        "INSERT INTO events (event_type, payload) VALUES (?, ?)",
        (event_type, payload),
    )
    _commit(conn)
    return cur.lastrowid or 0


def count_unconsumed_events(*event_types: str, since: str | None = None) -> int:
    """Count unconsumed events of given types, optionally since a timestamp."""
    if not event_types:
        return 0
    placeholders = ",".join("?" for _ in event_types)
    params: list[Any] = list(event_types)
    where = f"event_type IN ({placeholders}) AND consumed = 0"
    if since:
        where += " AND created > ?"
        params.append(since)
    row = _db().execute(f"SELECT COUNT(*) AS cnt FROM events WHERE {where}", params).fetchone()
    return int(row["cnt"]) if row else 0


def consume_events(*event_types: str, since: str | None = None) -> int:
    """Mark events as consumed. Returns count consumed."""
    if not event_types:
        return 0
    placeholders = ",".join("?" for _ in event_types)
    params: list[Any] = list(event_types)
    where = f"event_type IN ({placeholders}) AND consumed = 0"
    if since:
        where += " AND created > ?"
        params.append(since)
    conn = _db()
    cur = conn.execute(f"UPDATE events SET consumed = 1 WHERE {where}", params)
    _commit(conn)
    return cur.rowcount


def cleanup_events(retention_days: int = 7) -> int:
    """Remove consumed events older than retention period and stale unconsumed events.

    Consumed events are removed after `retention_days`.
    Unconsumed events are removed after 4x retention (safety net for events
    that were never consumed due to behavior gaps).
    """
    conn = _db()
    stale_days = retention_days * 4
    cur = conn.execute(
        "DELETE FROM events WHERE "
        "(consumed = 1 AND created < datetime('now', ?)) OR "
        "(consumed = 0 AND created < datetime('now', ?))",
        (f"-{retention_days} days", f"-{stale_days} days"),
    )
    _commit(conn)
    return cur.rowcount


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


def cleanup_task_logs(retention_days: int = 90) -> int:
    """Delete task_logs older than retention_days. Returns count."""
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
    conn = _db()
    cur = conn.execute("DELETE FROM task_logs WHERE started < ?", (cutoff,))
    _commit(conn)
    return cur.rowcount


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


def log_cost(
    chat_id: str,
    cost_usd: float,
    num_turns: int,
    duration_api_ms: int,
    source: str = "message",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_create_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> None:
    """Record cost and usage for an agent run."""
    db = _db()
    db.execute(
        "INSERT INTO cost_log "
        "(chat_id, cost_usd, num_turns, duration_api_ms, source, "
        " input_tokens, output_tokens, cache_create_tokens, cache_read_tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (chat_id, cost_usd, num_turns, duration_api_ms, source,
         input_tokens, output_tokens, cache_create_tokens, cache_read_tokens),
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
        f"COALESCE(AVG(cost_usd), 0) AS avg_cost, "
        f"COALESCE(SUM(input_tokens), 0) AS input_tok, "
        f"COALESCE(SUM(output_tokens), 0) AS output_tok, "
        f"COALESCE(SUM(cache_create_tokens), 0) AS cache_create, "
        f"COALESCE(SUM(cache_read_tokens), 0) AS cache_read "
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

    # Cache efficiency stats (only show if data exists)
    total_input = row["input_tok"] + row["cache_create"] + row["cache_read"]
    if total_input > 0:
        cache_hit_rate = row["cache_read"] / total_input * 100
        lines.append(f"  Tokens: {total_input:,} input ({row['output_tok']:,} output)")
        lines.append(
            f"  Cache: {row['cache_read']:,} read / {row['cache_create']:,} created "
            f"({cache_hit_rate:.0f}% hit rate)"
        )

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


# ---------------------------------------------------------------------------
# Outbound message dedup
# ---------------------------------------------------------------------------


def log_outbound(chat_id: str, content_hash: str) -> None:
    """Record an outbound message hash for dedup detection."""
    db = _db()
    db.execute(
        "INSERT INTO outbound_log (chat_id, content_hash) VALUES (?, ?)",
        (chat_id, content_hash),
    )
    _commit(db)


def is_duplicate_outbound(chat_id: str, content_hash: str, window_seconds: int = 300) -> bool:
    """Check if the same content hash was sent recently."""
    row = (
        _db()
        .execute(
            "SELECT 1 FROM outbound_log "
            "WHERE chat_id = ? AND content_hash = ? "
            "AND timestamp >= datetime('now', ?)",
            (chat_id, content_hash, f"-{window_seconds} seconds"),
        )
        .fetchone()
    )
    return row is not None


def count_recent_outbound(chat_id: str, window_seconds: int = 3600) -> int:
    """Count outbound messages sent within the given time window."""
    row = (
        _db()
        .execute(
            "SELECT COUNT(*) FROM outbound_log "
            "WHERE chat_id = ? AND timestamp >= datetime('now', ?)",
            (chat_id, f"-{window_seconds} seconds"),
        )
        .fetchone()
    )
    return row[0] if row else 0


def cleanup_outbound_log(retention_hours: int = 24) -> int:
    """Remove outbound log entries older than retention period."""
    db = _db()
    cur = db.execute(
        "DELETE FROM outbound_log WHERE timestamp < datetime('now', ?)",
        (f"-{retention_hours} hours",),
    )
    _commit(db)
    return cur.rowcount


# ---------------------------------------------------------------------------
# Task failure tracking
# ---------------------------------------------------------------------------


def increment_task_failures(task_id: str) -> int:
    """Increment consecutive failure count and return the new value."""
    conn = _db()
    row = conn.execute(
        "UPDATE tasks SET consecutive_failures = consecutive_failures + 1 "
        "WHERE id = ? RETURNING consecutive_failures",
        (task_id,),
    ).fetchone()
    _commit(conn)
    return int(row[0]) if row else 0


def reset_task_failures(task_id: str) -> None:
    """Reset consecutive failure count to 0."""
    db = _db()
    db.execute("UPDATE tasks SET consecutive_failures = 0 WHERE id = ?", (task_id,))
    _commit(db)


# ---------------------------------------------------------------------------
# Cost anomaly detection
# ---------------------------------------------------------------------------


def get_rolling_avg_cost(days: int = 7) -> float:
    """Average cost per run over the last N days."""
    row = (
        _db()
        .execute(
            "SELECT COALESCE(AVG(cost_usd), 0) AS avg_per_run "
            "FROM cost_log WHERE timestamp >= datetime('now', ?)",
            (f"-{days} days",),
        )
        .fetchone()
    )
    return float(row["avg_per_run"]) if row else 0.0


def get_daily_deep_work_cost() -> float:
    """Sum of deep work behavior costs today."""
    row = (
        _db()
        .execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total "
            "FROM cost_log WHERE source LIKE 'behavior:deep_work%' "
            "AND timestamp >= date('now')",
        )
        .fetchone()
    )
    return float(row["total"]) if row else 0.0


# ---------------------------------------------------------------------------
# Deep Work Quality Tracking
# ---------------------------------------------------------------------------


def log_deep_work_quality(goal_id: str, rating: int) -> None:
    """Record a deep work session quality rating (1-5) for a goal."""
    if not 1 <= rating <= 5:
        return
    conn = _db()
    conn.execute(
        "INSERT INTO deep_work_quality (goal_id, rating) VALUES (?, ?)",
        (goal_id, rating),
    )
    _commit(conn)


def get_recent_quality_scores(goal_id: str, n: int = 3) -> list[int]:
    """Return the last n quality ratings for a goal, newest first."""
    rows = (
        _db()
        .execute(
            "SELECT rating FROM deep_work_quality WHERE goal_id = ? ORDER BY id DESC LIMIT ?",
            (goal_id, n),
        )
        .fetchall()
    )
    return [int(r["rating"]) for r in rows]


def get_quality_blocked_goals(threshold: float = 2.0, window: int = 3) -> list[str]:
    """Return goal IDs where avg of last `window` ratings is below threshold."""
    # Get all goals with enough ratings
    rows = (
        _db()
        .execute(
            "SELECT goal_id FROM deep_work_quality GROUP BY goal_id HAVING COUNT(*) >= ?",
            (window,),
        )
        .fetchall()
    )
    blocked: list[str] = []
    for row in rows:
        scores = get_recent_quality_scores(row["goal_id"], window)
        if scores and sum(scores) / len(scores) < threshold:
            blocked.append(row["goal_id"])
    return blocked


def get_goal_quality_tier(goal_id: str, window: int = 3) -> str:
    """Return 'paused', 'reflexion', or 'normal' based on recent quality scores.

    Graduated quality gates:
    - avg < 1.5 over `window` sessions -> 'paused'
    - 1.5 <= avg < 2.5 over `window` sessions -> 'reflexion'
    - avg >= 2.5 (or insufficient data) -> 'normal'
    """
    scores = get_recent_quality_scores(goal_id, window)
    if len(scores) < window:
        return "normal"
    avg = sum(scores) / len(scores)
    if avg < 1.5:
        return "paused"
    if avg < 2.5:
        return "reflexion"
    return "normal"
