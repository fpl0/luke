"""Block hand-written SQL against luke.db — steer to workspace/tools/q.sh.

Why this exists
---------------
Guessing luke.db's schema is my most repetitive self-inflicted failure. The
canonical column names (events.event_type/payload/created, messages.ts,
memory_meta with no title/content, table `memory_meta` not `memories`) do not
match the plausible-sounding ones, so every hand-written query is a coin flip.

q.sh was built on 2026-08-06 precisely to end this: it resolves the DB path and,
on a schema miss, prints the real columns instead of just an error. A memory
line (`ref-luke-db-schema`) then told me to always use it.

It did not work. reflect.sh's own counter for this class read **12 failures in
30h** on 2026-08-08 — the same number q.sh's header cites from the two days
*before* it existed. Four nights running the advisory was in context and the
guess happened anyway.

That is the signature of a corrective that needs enforcement rather than
another reminder (`dream-migrate-recurring-correctives-from-memory-to-hooks`).
This module is that enforcement: deterministic, no model call, and it only
fires on the narrow pattern it can offer a working replacement for.
"""

from __future__ import annotations

import re
from typing import Final

# The sanctioned wrappers. Any command that goes through one of these is fine —
# q.sh IS the replacement, and reflect.sh/other tools bake in real columns.
_SANCTIONED: Final = re.compile(r"\b(q\.sh|reflect\.sh|workspace/tools/)", re.IGNORECASE)

# Naming the DB. Covers bare `luke.db` and any absolute/relative path to it.
_TARGETS_LUKE_DB: Final = re.compile(r"\bluke\.db\b", re.IGNORECASE)

# Actually opening sqlite: the CLI, or the Python driver in a heredoc/-c.
# `cp luke.db backup`, `ls -la luke.db`, `grep luke.db` do not match — they are
# not queries and have no q.sh equivalent.
_OPENS_SQLITE: Final = re.compile(
    r"(\bsqlite3\b(?!\s*\.\w*(?:Error|Warning))|sqlite3\.connect|\bdatasette\b)",
    re.IGNORECASE,
)

REASON: Final = (
    "Don't hand-write SQL against luke.db — that is the 'no such column' loop, "
    "12 failures in the last 30h alone. Use the wrapper, which resolves the DB "
    "path and prints the REAL columns on a miss:\n"
    '  workspace/tools/q.sh "<your sql>"\n'
    "  workspace/tools/q.sh --schema            # every table + its columns\n"
    "  workspace/tools/q.sh --schema events tasks\n"
    "Run --schema FIRST if you are not certain of a column. Reciting remembered "
    "column names is what fails: events is (event_type, payload, created) not "
    "(type, data); messages.ts not .timestamp; the table is memory_meta, not "
    "memories, and it has no title/content."
)


def blocks_bash_command(command: object) -> bool:
    """True when a Bash command hand-queries luke.db outside the wrapper.

    Requires all three: it names luke.db, it opens sqlite, and it is not
    already going through a sanctioned tool. Anything else passes untouched —
    a gate that fires on unrelated commands gets waved away, which is how the
    advisory version of this rule died.
    """
    if not isinstance(command, str) or not command:
        return False
    if _SANCTIONED.search(command):
        return False
    if not _TARGETS_LUKE_DB.search(command):
        return False
    return bool(_OPENS_SQLITE.search(command))


def blocks_tool_input(tool_input: object) -> bool:
    """`blocks_bash_command` over a PreToolUse tool_input payload."""
    if not isinstance(tool_input, dict):
        return False
    return blocks_bash_command(tool_input.get("command"))
