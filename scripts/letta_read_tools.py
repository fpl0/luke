#!/usr/bin/env python3
"""TOOL-SURFACE PARITY — the read-only tool surface for the Letta agent.

Why this exists
---------------
The 5.1R reply-diff gate came back RED on 2026-08-02 (7/20, needed 18). Decomposing the
13 divergences by cause showed the single largest driver was not the memory substrate at
all: **7 of 13 were the Letta agent losing on a missing tool.** It ships with exactly
``memory_insert`` / ``memory_replace``; the production SDK arm has ~27. "Fix this failing
task", "diagnose the codebase", "show me the stored reminder", "what does the log say" —
those are capability losses, not recall losses. Per
``insight-parity-gate-must-equalize-capability-surface``, an A/B gate measures whatever
differs MOST between the arms, so until the capability surface is comparable the gate
cannot produce a meaningful memory-parity number.

Closing that gap is also a hard **Phase 6 Stage 2 precondition** in the migration plan
(the cutover ships "only with the tool surface carried over"), so this is real migration
work, not gate hygiene.

Scope: READ-ONLY, deliberately
------------------------------
Every tool here observes; none mutate. Three reasons, in priority order:

1. **The 5.1R replay is 20 live turns against Filipe's real account.** A write surface
   (send_message, schedule_task, remember, email) would fire real Telegram sends, real
   scheduled tasks and real memory writes while re-running a *test*. Read-only makes the
   re-run safe by construction rather than by the agent choosing well.
2. **It is sufficient.** All 7 tool-surface divergences were read asks — file reads, code
   search, git history, the task store, the log. None needed a write.
3. **It is separable.** The write half is a Phase 6 Stage 2 decision with its own accept
   criteria and Filipe's explicit go. Shipping the read half now does not pre-commit it.

Safety model — the sandbox is NOT a jail
----------------------------------------
Letta's local sandbox executes tool source as ``filipelm`` with full machine access
(probed 2026-08-02: cwd ``~/.letta/tool_execution_dir``, py3.12, sqlite + subprocess +
git + ripgrep all reachable). So the guards below are the ONLY thing standing between a
model-chosen argument and the filesystem. They are enforced inside each tool, not by the
runtime:

* **Path allowlist** — every path is ``realpath``'d and must land under ``~/Luke`` or
  ``~/Code/luke``. Resolving first is what defeats ``../`` escapes and symlinks.
* **Secret denylist** — ``.env``, credentials, keys and ``.git/`` internals are refused
  even inside the allowlist. The sandbox could read them regardless; the point is to stop
  the *agent* from surfacing a token into a Telegram reply.
* **No shell** — every subprocess is an argv list, never a string, so an argument can
  never become a command. Git is further restricted to a read-only subcommand allowlist.
* **sqlite is opened ``mode=ro``** via URI, so a malformed query cannot write.

Each function is fully self-contained (imports and helpers nested) because Letta uploads
the function source alone — module-level names would not exist at execution time. That
duplication is the cost of the deployment model, and it buys testability: these are
ordinary Python functions, callable and assertable in-process by
``tests/test_letta_read_tools.py`` before anything is registered.

Registration + attachment lives in ``scripts/letta_register_read_tools.py``.
"""

from __future__ import annotations

# --- shared constants, for the LOCAL tests only -----------------------------
# The uploaded tool bodies repeat these literals inline (see the self-containment note
# above); they are named here so the test-suite and the registrar can reference the same
# values without re-typing them.
RUNTIME_DIR = "/Users/filipelm/Luke"
REPO_DIR = "/Users/filipelm/Code/luke"
DB_PATH = "/Users/filipelm/Luke/luke.db"


def luke_read_file(path: str, start_line: int = 1, max_lines: int = 200) -> str:
    """Read a slice of a file from Luke's runtime directory or source repo.

    Use this to inspect source code, plans, logs, configs or any artifact on disk when
    answering a question that depends on what a file actually says. Prefer a narrow
    start_line/max_lines window over reading a whole large file.

    Args:
        path (str): Absolute path, or a path relative to the Luke source repo. Must
            resolve inside /Users/filipelm/Luke or /Users/filipelm/Code/luke.
        start_line (int): 1-indexed first line to return. Defaults to 1.
        max_lines (int): How many lines to return, capped at 400. Defaults to 200.

    Returns:
        str: The requested lines, each prefixed with its line number, or an error string
            beginning with "ERROR:" if the path is not readable or not permitted.
    """
    import os

    # Guards are inlined rather than factored into a nested helper: Letta's schema
    # generator walks the uploaded source and names the tool after the wrong def when
    # one is present (it registered a tool called "_resolve"). One outer def only.
    roots = ("/Users/filipelm/Luke", "/Users/filipelm/Code/luke")
    deny = (".env", ".credentials", "credentials.json", "id_rsa", ".ssh/", "token.json")
    p = path if os.path.isabs(path or "") else os.path.join("/Users/filipelm/Code/luke", path or "")
    rp = os.path.realpath(p)  # resolve FIRST — this is what defeats ../ and symlink escapes
    if not any(rp == r or rp.startswith(r + os.sep) for r in roots):
        return "ERROR: path outside the permitted roots (~/Luke, ~/Code/luke)"
    low = rp.lower()
    if any(d in low for d in deny) or low.endswith((".pem", ".key")) or "/.git/" in low:
        return "ERROR: refusing to read a secrets or VCS-internal path"
    if not os.path.isfile(rp):
        return f"ERROR: not a file: {rp}"
    max_lines = max(1, min(int(max_lines), 400))
    start_line = max(1, int(start_line))
    out = []
    try:
        with open(rp, "r", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if i < start_line:
                    continue
                if len(out) >= max_lines:
                    out.append(f"… (truncated at {max_lines} lines; read on from line {i})")
                    break
                out.append(f"{i}\t{line.rstrip()}")
    except Exception as e:
        return f"ERROR: {e!r}"
    if not out:
        return f"(no lines at or after {start_line} in {rp})"
    return f"{rp}\n" + "\n".join(out)


def luke_search_code(pattern: str, path: str = "", max_results: int = 40) -> str:
    """Search file contents by regular expression across Luke's repo and runtime dir.

    This is the tool for "where is X handled", "does the code do Y", "find every place
    that Z" — questions answered by grepping the codebase rather than by memory.

    Args:
        pattern (str): A regular expression (ripgrep syntax) to search for.
        path (str): Optional sub-path to restrict the search to. Empty searches the whole
            Luke source repo. Must resolve inside /Users/filipelm/Luke or
            /Users/filipelm/Code/luke.
        max_results (int): Maximum matching lines to return, capped at 120. Defaults to 40.

    Returns:
        str: Matching lines as "file:line: text", or an error string beginning with
            "ERROR:" if the path is not permitted or the search fails.
    """
    import os
    import subprocess

    # Inlined guard — see the note in luke_read_file on why there is no nested def.
    roots = ("/Users/filipelm/Luke", "/Users/filipelm/Code/luke")
    p = path or "/Users/filipelm/Code/luke"
    if not os.path.isabs(p):
        p = os.path.join("/Users/filipelm/Code/luke", p)
    rp = os.path.realpath(p)
    if not any(rp == r or rp.startswith(r + os.sep) for r in roots):
        return "ERROR: path outside the permitted roots (~/Luke, ~/Code/luke)"
    if not os.path.exists(rp):
        return f"ERROR: no such path: {rp}"
    max_results = max(1, min(int(max_results), 120))
    # argv list, never a shell string: `pattern` is model-supplied and must stay an
    # argument. --glob exclusions keep vendored deps and the venvs out of the results.
    cmd = [
        "rg", "--no-heading", "--line-number", "--color", "never",
        "--max-count", "20", "--max-filesize", "2M",
        "--glob", "!**/.git/**", "--glob", "!**/node_modules/**",
        "--glob", "!**/.venv/**", "--glob", "!**/.letta-venv/**",
        "--glob", "!**/__pycache__/**", "--glob", "!**/*.env",
        "-e", str(pattern), rp,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    except Exception as e:
        return f"ERROR: {e!r}"
    if r.returncode not in (0, 1):
        return f"ERROR: ripgrep failed: {(r.stderr or '').strip()[:300]}"
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return f"(no matches for {pattern!r} under {rp})"
    head = lines[:max_results]
    more = len(lines) - len(head)
    body = "\n".join(ln[:300] for ln in head)
    return body + (f"\n… and {more} more matches (narrow the pattern)" if more > 0 else "")


def luke_list_dir(path: str = "") -> str:
    """List the files and subdirectories at a path, with sizes and modification times.

    Use this to orient in the codebase before reading — to find out what scripts, plans
    or logs exist — rather than guessing a filename.

    Args:
        path (str): Directory to list. Empty lists the Luke source repo root. Must resolve
            inside /Users/filipelm/Luke or /Users/filipelm/Code/luke.

    Returns:
        str: One entry per line as "type  size  modified  name", or an error string
            beginning with "ERROR:".
    """
    import os
    import time

    roots = ("/Users/filipelm/Luke", "/Users/filipelm/Code/luke")
    p = path or "/Users/filipelm/Code/luke"
    if not os.path.isabs(p):
        p = os.path.join("/Users/filipelm/Code/luke", p)
    rp = os.path.realpath(p)
    if not any(rp == r or rp.startswith(r + os.sep) for r in roots):
        return "ERROR: path outside the permitted roots (~/Luke, ~/Code/luke)"
    if not os.path.isdir(rp):
        return f"ERROR: not a directory: {rp}"
    try:
        names = sorted(os.listdir(rp))
    except Exception as e:
        return f"ERROR: {e!r}"
    out = [rp]
    for n in names:
        if n in (".git", "node_modules", "__pycache__", ".venv", ".letta-venv"):
            continue
        full = os.path.join(rp, n)
        try:
            st = os.stat(full)
        except Exception:
            continue
        kind = "dir " if os.path.isdir(full) else "file"
        mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
        out.append(f"{kind}  {st.st_size:>10}  {mt}  {n}")
    return "\n".join(out) if len(out) > 1 else f"(empty: {rp})"


def luke_git(subcommand: str = "log", args: str = "") -> str:
    """Run a read-only git command against Luke's source repository.

    Use this for "what changed", "when did I ship X", "what's uncommitted", "who touched
    this line" — history questions the repo answers better than memory.

    Args:
        subcommand (str): One of log, status, show, diff, blame, branch, tag, describe,
            shortlog. Write subcommands are refused.
        args (str): Additional whitespace-separated arguments, e.g. "--oneline -10" or
            "HEAD~1 -- src/luke/agent.py". Shell metacharacters are refused.

    Returns:
        str: The command output (truncated), or an error string beginning with "ERROR:".
    """
    import subprocess

    allowed = {"log", "status", "show", "diff", "blame", "branch", "tag", "describe", "shortlog"}
    sub = (subcommand or "log").strip()
    if sub not in allowed:
        return f"ERROR: '{sub}' is not a permitted read-only subcommand ({sorted(allowed)})"
    extra = (args or "").split()
    # No shell is ever spawned, so metacharacters could not execute anyway — but refusing
    # them keeps the failure legible instead of producing a confusing git error.
    bad = [a for a in extra if any(ch in a for ch in ";|&`$><\n")]
    if bad:
        return f"ERROR: refusing arguments containing shell metacharacters: {bad}"
    if len(extra) > 12:
        return "ERROR: too many arguments"
    cmd = ["git", "-C", "/Users/filipelm/Code/luke", "--no-pager", sub] + extra
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    except Exception as e:
        return f"ERROR: {e!r}"
    out = (r.stdout or "") + (("\n[stderr] " + r.stderr) if r.stderr.strip() else "")
    out = out.strip()
    if not out:
        return f"(git {sub} produced no output)"
    return out[:6000] + ("\n… (truncated)" if len(out) > 6000 else "")


def luke_list_tasks(status: str = "", limit: int = 25, failing_only: bool = False) -> str:
    """List Luke's scheduled tasks — what is queued, what ran, what failed.

    Use this for "what reminders do I have", "is that job still scheduled", "did the
    nightly run", "show me the stored reminder". The task store is the ground truth here;
    memory is not. When the question is about something FAILING, set failing_only — the
    default listing is ordered by recency and a task that has been failing for a while can
    fall outside the window.

    Args:
        status (str): Optional filter — "active", "completed", "failed", or empty for all.
        limit (int): Maximum tasks to return, capped at 60. Defaults to 25.
        failing_only (bool): When True, return only tasks with consecutive failures,
            worst first. Defaults to False.

    Returns:
        str: One task per line with id, status, schedule, last run and a prompt excerpt,
            or an error string beginning with "ERROR:".
    """
    import sqlite3

    limit = max(1, min(int(limit), 60))
    q = ("SELECT id, status, schedule_type, schedule_value, last_run, consecutive_failures, prompt "
         "FROM tasks")
    params: list = []
    where = []
    if status:
        where.append("status = ?")
        params.append(status.strip().lower())
    if failing_only:
        where.append("consecutive_failures > 0")
    if where:
        q += " WHERE " + " AND ".join(where)
    # Failure-first when the caller asked about failures: ordering by recency alone hid a
    # currently-failing job outside the default 25-row window during verification.
    q += (" ORDER BY consecutive_failures DESC, COALESCE(last_run, created_at) DESC LIMIT ?"
          if failing_only else " ORDER BY COALESCE(last_run, created_at) DESC LIMIT ?")
    params.append(limit)
    try:
        con = sqlite3.connect("file:/Users/filipelm/Luke/luke.db?mode=ro", uri=True)
        rows = con.execute(q, params).fetchall()
        con.close()
    except Exception as e:
        return f"ERROR: {e!r}"
    if not rows:
        return ("(no tasks are currently failing)" if failing_only
                else f"(no tasks{' with status ' + status if status else ''})")
    out = []
    for tid, st, stype, sval, last, fails, prompt in rows:
        p = " ".join((prompt or "").split())[:150]
        f = f" fails={fails}" if fails else ""
        out.append(f"[{st}] {tid}  {stype}={sval}  last_run={last or '-'}{f}\n    {p}")
    return "\n".join(out)


def luke_search_messages(query: str = "", limit: int = 15, days: int = 120,
                         on_date: str = "") -> str:
    """Search the real Telegram conversation history between Filipe and Luke.

    Use this when a question refers to something that was *said* — "what did I tell you
    about X", "when did we discuss Y", "what was your answer last time". This reads the
    actual message log, so it recovers context that the memory archive summarised away or
    never captured.

    Terms are ORed and results ranked by how many of them a message contains, so a
    descriptive query still works when you do not know the exact wording. When you DO know
    the date but not the words, pass on_date with an empty query to read that whole day.

    Args:
        query (str): Words to search for, matched case-insensitively. Best-matching
            messages come first. May be empty if on_date is given.
        limit (int): Maximum messages to return, capped at 40. Defaults to 15.
        days (int): How far back to search, in days. Ignored when on_date is set.
            Defaults to 120.
        on_date (str): Optional ISO date, e.g. "2026-07-26", to restrict the search to a
            single day. Empty searches the whole window.

    Returns:
        str: Matching messages, oldest first, as "[timestamp] sender: text", or an error
            string beginning with "ERROR:".
    """
    import sqlite3
    from datetime import datetime, timedelta, timezone

    # ORed, not ANDed. ANDing every term made the tool useless on natural queries: asked
    # to find an angry message, the agent searched "angry wrong frustrated", none of which
    # appeared verbatim, and the AND returned nothing — so it reported the message did not
    # exist when it did. Ranking by match count keeps precision without that cliff.
    terms = [t.lower() for t in (query or "").split() if len(t) > 1][:8]
    on_date = (on_date or "").strip()
    if not terms and not on_date:
        return "ERROR: give a query, a date, or both"
    limit = max(1, min(int(limit), 40))

    where, params = [], []
    if terms:
        where.append("(" + " OR ".join(["LOWER(content) LIKE ?"] * len(terms)) + ")")
        params += [f"%{t}%" for t in terms]
    if on_date:
        where.append("ts >= ? AND ts < ?")
        params += [on_date, on_date + "T99"]
    else:
        where.append("ts >= ?")
        params.append((datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))).isoformat())
    # Rank by how many distinct terms a message contains, then recency; the LIMIT is
    # applied after scoring so a message matching every term is never crowded out by a
    # more recent one that matches a single common word.
    score = " + ".join(["(LOWER(content) LIKE ?)"] * len(terms)) if terms else "0"
    try:
        con = sqlite3.connect("file:/Users/filipelm/Luke/luke.db?mode=ro", uri=True)
        rows = con.execute(
            f"SELECT ts, sender, content, ({score}) AS hits FROM messages "
            f"WHERE {' AND '.join(where)} ORDER BY hits DESC, id DESC LIMIT ?",
            [f"%{t}%" for t in terms] + params + [limit],
        ).fetchall()
        con.close()
    except Exception as e:
        return f"ERROR: {e!r}"
    if not rows:
        scope = f"on {on_date}" if on_date else f"in the last {days}d"
        return (f"(no messages {scope} matching {terms or 'anything'} — try fewer or "
                f"different words, or pass on_date to read a whole day)")
    out = []
    for ts, sender, content, hits in reversed(rows):
        text = " ".join((content or "").split())[:500]
        mark = f" [{hits}/{len(terms)} terms]" if terms else ""
        out.append(f"[{(ts or '')[:16]}] {sender}{mark}: {text}")
    return "\n".join(out)


def luke_recall(query: str, limit: int = 8) -> str:
    """Search Luke's memory archive by keyword for entities, insights, procedures, episodes.

    The turn already arrives with a ranked recall injection; use this tool to go *further* —
    to chase a specific memory the injection did not surface, or to check whether something
    is stored at all before asserting it.

    Note this is the keyword (FTS5) half of Luke's retrieval, not the full
    keyword+semantic+graph fusion that produced the injection, so a miss here means "no
    keyword match", not "definitely not stored".

    Args:
        query (str): Search terms.
        limit (int): Maximum memories to return, capped at 20. Defaults to 8.

    Returns:
        str: Matching memories as id/type/title plus a body excerpt, or an error string
            beginning with "ERROR:".
    """
    import re
    import sqlite3

    q = " ".join(re.findall(r"[A-Za-z0-9_]{2,}", query or ""))
    if not q:
        return "ERROR: query is empty"
    limit = max(1, min(int(limit), 20))
    fts = " OR ".join(q.split())
    try:
        con = sqlite3.connect("file:/Users/filipelm/Luke/luke.db?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT f.id, f.title, f.content, m.type, m.importance "
                "FROM memory_fts f LEFT JOIN memory_meta m ON m.id = f.id "
                "WHERE memory_fts MATCH ? AND COALESCE(m.status,'active') = 'active' "
                "ORDER BY rank LIMIT ?",
                (fts, limit),
            ).fetchall()
        except Exception:
            # FTS5 rejects some tokenisations (bare operators, unbalanced quotes). Falling
            # back to LIKE keeps the tool useful instead of failing the whole turn.
            like = f"%{q.split()[0].lower()}%"
            rows = con.execute(
                "SELECT f.id, f.title, f.content, m.type, m.importance "
                "FROM memory_fts f LEFT JOIN memory_meta m ON m.id = f.id "
                "WHERE LOWER(f.content) LIKE ? AND COALESCE(m.status,'active') = 'active' "
                "LIMIT ?",
                (like, limit),
            ).fetchall()
        con.close()
    except Exception as e:
        return f"ERROR: {e!r}"
    if not rows:
        return f"(nothing in the archive keyword-matches {query!r})"
    out = []
    for mid, title, content, mtype, imp in rows:
        body = " ".join((content or "").split())[:600]
        out.append(f"<mem id=\"{mid}\" type=\"{mtype or '?'}\" title=\"{title or ''}\">\n{body}\n</mem>")
    return "\n\n".join(out)


def luke_tail_log(lines: int = 60, contains: str = "", log: str = "luke") -> str:
    """Read the tail of Luke's runtime log, optionally filtered to matching lines.

    Use this to answer "did that job run", "what error did it throw", "is the scanner
    alive" — questions about what the running system actually did.

    Args:
        lines (int): How many matching lines from the end to return, capped at 200.
            Defaults to 60.
        contains (str): Optional case-insensitive substring filter, e.g. "letta_search_failed".
            Empty returns the raw tail.
        log (str): Which log — "luke" (the main runtime log) or "watchdog". Defaults to "luke".

    Returns:
        str: The selected log lines, or an error string beginning with "ERROR:".
    """
    import os

    known = {
        "luke": "/Users/filipelm/Luke/luke.log",
        "watchdog": "/Users/filipelm/Luke/watchdog.log",
    }
    path = known.get((log or "luke").strip().lower())
    if not path:
        return f"ERROR: unknown log {log!r}; known: {sorted(known)}"
    if not os.path.isfile(path):
        return f"ERROR: no such log: {path}"
    lines = max(1, min(int(lines), 200))
    needle = (contains or "").lower()
    # Read the tail by seeking backwards: luke.log is >20MB and reading it whole would
    # blow the sandbox's memory and the tool's time budget for no benefit.
    try:
        size = os.path.getsize(path)
        window = min(size, 4_000_000 if needle else 400_000)
        with open(path, "rb") as f:
            f.seek(size - window)
            chunk = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"ERROR: {e!r}"
    raw = chunk.splitlines()
    if window < size and raw:
        raw = raw[1:]  # drop the partial first line from the seek
    picked = [ln for ln in raw if needle in ln.lower()] if needle else raw
    if not picked:
        return f"(no lines matching {contains!r} in the last {window} bytes of {path})"
    tail = picked[-lines:]
    return "\n".join(ln[:400] for ln in tail)


# The canonical set, in the order the registrar uploads them.
TOOLS = [
    luke_read_file,
    luke_search_code,
    luke_list_dir,
    luke_git,
    luke_list_tasks,
    luke_search_messages,
    luke_recall,
    luke_tail_log,
]
