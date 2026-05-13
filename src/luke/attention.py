"""Active attention: persistent foreground commitments injected into context.

Memory is recall-on-demand. The planner is intent-on-trigger. Active
attention is the missing primitive — items that Filipe has said matter
that stay in working context permanently until explicitly cancelled.

Reads in this module are cheap (single SQL fetch) and intended to run
on every agent invocation. Writes are tool-driven (pin_attention,
unpin_attention) and are rare.
"""

from __future__ import annotations

from typing import Any

from . import db


def list_attention(chat_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return active attention items for a chat, oldest first."""
    rows = (
        db._db()
        .execute(
            "SELECT id, content, origin, created_at, related_id "
            "FROM active_attention WHERE chat_id = ? "
            "ORDER BY created_at ASC LIMIT ?",
            (chat_id, limit),
        )
        .fetchall()
    )
    return [dict(r) for r in rows]


def pin(
    chat_id: str,
    content: str,
    origin: str = "luke",
    related_id: str | None = None,
) -> int:
    """Add a new active-attention item. Returns the row id."""
    conn = db._db()
    cur = conn.execute(
        "INSERT INTO active_attention (chat_id, content, origin, related_id) "
        "VALUES (?, ?, ?, ?)",
        (chat_id, content, origin, related_id),
    )
    db._commit(conn)
    return int(cur.lastrowid or 0)


def unpin(chat_id: str, attention_id: int) -> bool:
    """Remove an active-attention item by id. Returns True if it existed."""
    conn = db._db()
    cur = conn.execute(
        "DELETE FROM active_attention WHERE id = ? AND chat_id = ?",
        (attention_id, chat_id),
    )
    db._commit(conn)
    return cur.rowcount > 0


def build_attention_block(chat_id: str) -> str | None:
    """Build the <active-attention> block for working-context injection.

    Returns None if no items.
    """
    items = list_attention(chat_id)
    if not items:
        return None
    lines: list[str] = [
        "<active-attention>",
        "# Things you said matter — keep these warm. Cancel with unpin_attention.",
        "",
    ]
    for it in items:
        marker = "•"
        origin = it.get("origin", "luke")
        prefix = f"{marker} [#{it['id']}, {origin}]"
        lines.append(f"{prefix} {it['content']}")
    lines.append("</active-attention>")
    return "\n".join(lines)
