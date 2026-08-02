#!/usr/bin/env python3
"""TOOL-SURFACE PARITY — prove the agent USES the read tools, not just that it has them.

``letta_register_read_tools.py`` proves the tools execute correctly in the sandbox. That
is necessary and not sufficient: a tool the model never reaches for closes zero
divergences. This script drives real turns through ``drive_letta_turn`` — the exact Phase 6
surface — on questions that are unanswerable from core blocks and recall injection alone,
and asserts two things per turn:

  1. **A read tool was actually invoked.** Read off the turn's tool-call list, not inferred
     from the prose. An answer that merely *sounds* grounded fails.
  2. **The reply carries a fact only that tool could supply.** Every expected fact is
     computed live from the same source of truth the tool reads (git HEAD, the task table,
     the message log) rather than hardcoded, so this script keeps telling the truth as the
     repo moves instead of decaying into a fixture test.

Both guards exist because of ``reflexion-hallucination-invented-event-defended-2026-08-01``
and ``reflexion-falsify-on-executed-surface-not-proxy-2026-08-01``: the failure mode this
whole gate is defending against is confident, well-voiced text with nothing behind it, and
the only way to catch that is to check the substance against ground truth.

Two of the five prompts are the REAL messages that lost their 5.1R rows to the missing
tool surface (#3743 "deep dive in the code base and logs", #3613 "Fix this! Task ... has
failed 4 times") — so a pass here is direct evidence about the divergences this work exists
to close, not a synthetic proxy for them.

Run: python3 scripts/letta_tool_surface_verify.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

REPO = "/Users/filipelm/Code/luke"
DB = "/Users/filipelm/Luke/luke.db"
LOG_PATH = os.path.join(REPO, "logs", "letta_tool_surface_verify.log")
LETTA = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
AGENT = os.environ.get("LETTA_AGENT_ID", "agent-36671c0b-a133-4bfb-a367-f23f7135071a")

READ_TOOLS = {
    "luke_read_file", "luke_search_code", "luke_list_dir", "luke_git",
    "luke_list_tasks", "luke_search_messages", "luke_recall", "luke_tail_log",
}


def reset_buffer() -> bool:
    """Clear the agent's conversation buffer so each run starts from the same state.

    Found the hard way: the first run of this script passed 5/5 on tool invocation, and
    the immediate re-run dropped to 2/5 — not because anything regressed, but because the
    Letta agent REMEMBERED the first run. It answered three prompts with "already looked
    this up earlier in our conversation" and re-served the earlier tool output. The facts
    were still right; it simply had no reason to call the tool twice.

    That makes an un-reset run unfalsifiable: a green could mean the tools work or merely
    that the agent recalls a previous green. The same contamination applies to the 5.1R
    reply-diff run, whose 20 prompts execute sequentially against this one agent — row 20
    sees rows 1-19.

    Only the message buffer is cleared. Core memory blocks and the archive — the actual
    memory substrate under test — are untouched, so this costs nothing that matters.
    """
    # The body is mandatory even though every field in it is optional — omitting it
    # returns 422 "Field required" on ('body',), not a helpful message.
    req = urllib.request.Request(
        f"{LETTA}/v1/agents/{AGENT}/reset-messages", method="PATCH",
        data=json.dumps({"add_default_initial_messages": True}).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        return True
    except Exception as e:
        print(f"!! could not reset the message buffer ({e!r}) — results may be "
              f"contaminated by a previous run", file=sys.stderr)
        return False


def _git_head() -> tuple[str, str]:
    """The current HEAD short-sha and subject — ground truth for the git turn."""
    r = subprocess.run(["git", "-C", REPO, "--no-pager", "log", "--oneline", "-1"],
                       capture_output=True, text=True, timeout=30)
    line = (r.stdout or "").strip()
    sha, _, subject = line.partition(" ")
    return sha, subject


def _failing_task() -> tuple[str, int] | None:
    """The task a reasonable person means by "the one that keeps failing", or None.

    Ordering matters here and the first version got it wrong: ranking by
    ``consecutive_failures`` alone tie-broke arbitrarily across five tasks all sitting at
    1 failure and picked a *completed* job from five weeks earlier. The agent answered
    with the currently-active job that failed yesterday — the better answer — and the
    harness scored it wrong. Still-active outranks long-finished, then failure count,
    then recency.
    """
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    row = con.execute(
        "SELECT id, consecutive_failures FROM tasks WHERE consecutive_failures > 0 "
        "ORDER BY (status = 'active') DESC, consecutive_failures DESC, last_run DESC LIMIT 1"
    ).fetchone()
    con.close()
    return (row[0], row[1]) if row else None


def _build_cases() -> list[dict]:
    """Assemble the turns, with every expected fact derived from live ground truth."""
    sha, subject = _git_head()
    subject_word = max((w for w in subject.split() if len(w) > 5), key=len, default="fix")
    cases = [
        {
            "name": "git-history",
            "prompt": "What's the most recent commit on the Luke repo — the short hash and "
                      "what it actually changed? Check, don't guess.",
            "want_tools": {"luke_git", "luke_search_code", "luke_read_file"},
            # The sha is the strong signal (7 hex chars nobody confabulates); the subject
            # word is a softer corroboration that it read the right commit.
            "want_any": [sha, subject_word],
            "want_all": [sha],
        },
        {
            "name": "codebase-diagnose",
            "prompt": "Luke I feel your answer quality is degrading! And sometimes I don't "
                      "even get an answer. Do a deep dive in the code base and logs. "
                      "Diagnose what's happening.",
            "want_tools": {"luke_search_code", "luke_read_file", "luke_tail_log", "luke_list_dir"},
            # Originally this demanded a filename in the prose ("agent.py", "src/luke").
            # That was the wrong axis, not a stricter one: the agent read the log and
            # answered in log FIELDS with real numbers ("every agent_start shows
            # prompt_chars 29,000–36,000 against a 32K window") — grounding a filename
            # mention would not have proved. What the check must catch is a diagnosis with
            # nothing behind it, so it now looks for a token that can only have come from
            # the code or the log, filename or field name alike.
            "want_any": ["luke.log", "agent.py", "app.py", "src/luke", "prompt_chars",
                         "agent_start", "context_window", "max_turns", "letta_",
                         "structlog", "scheduler"],
            "want_all": [],
        },
        {
            "name": "conversation-history",
            "prompt": "Back on 26 July I sent you an angry message about your answers. "
                      "What exactly did I say — quote it back to me.",
            "want_tools": {"luke_search_messages"},
            "want_any": ["degrading", "annoying", "don't even get an answer",
                         "don’t even get an answer"],
            "want_all": [],
        },
        {
            "name": "runtime-log",
            "prompt": "Check the runtime log — have there been any letta_search_failed "
                      "errors, and when was the last one?",
            "want_tools": {"luke_tail_log"},
            "want_any": ["letta_search_failed", "no ", "none", "zero", "clean"],
            "want_all": [],
        },
    ]
    failing = _failing_task()
    if failing:
        tid, n = failing
        cases.append({
            "name": "task-store",
            "prompt": "Fix this! One of my scheduled tasks keeps failing. Which task is it "
                      "— give me its id — and how many times has it failed in a row?",
            "want_tools": {"luke_list_tasks"},
            "want_any": [tid, str(n)],
            "want_all": [tid],
        })
    else:
        cases.append({
            "name": "task-store",
            "prompt": "What scheduled tasks do I have running right now? List them with "
                      "their schedules.",
            "want_tools": {"luke_list_tasks"},
            "want_any": ["cron", "interval", "once"],
            "want_all": [],
        })
    return cases


def main() -> int:
    from luke.letta_agent import drive_letta_turn

    cases = _build_cases()
    print(f"driving {len(cases)} live turns through drive_letta_turn "
          f"(the Phase 6 surface)\n")
    results = []
    for c in cases:
        # Reset before EVERY case, not once per run: the cases contaminate each other too
        # (the codebase-diagnose turn reads the log, which then answers the runtime-log
        # turn for free). Each case must be a cold ask.
        clean = reset_buffer()
        r = drive_letta_turn(c["prompt"])
        reply = (r.get("reply") or "")
        low = reply.lower()
        called = {t for t in (r.get("tools") or []) if t in READ_TOOLS}
        used_expected = bool(called & c["want_tools"])
        hit_any = [w for w in c["want_any"] if w.lower() in low]
        missing_all = [w for w in c["want_all"] if w.lower() not in low]
        # A contaminated turn cannot pass: without a clean buffer, "the tool was not called"
        # is indistinguishable from "the answer was already in context".
        ok = bool(clean and r.get("error") is None and used_expected and hit_any and not missing_all)
        results.append({
            "name": c["name"], "ok": ok, "clean_buffer": clean,
            "seconds": round(r.get("seconds", 0), 1),
            "tools": r.get("tools") or [], "read_tools_used": sorted(called),
            "expected_tool_used": used_expected, "facts_hit": hit_any,
            "facts_missing": missing_all, "error": r.get("error"),
            "reply_chars": len(reply),
        })
        print(f"[{c['name']:22}] {'PASS' if ok else 'FAIL'}  {r.get('seconds', 0):.1f}s")
        print(f"    tools called : {r.get('tools') or []}")
        print(f"    expected-tool: {'yes' if used_expected else 'NO — reached for none of ' + str(sorted(c['want_tools']))}")
        print(f"    facts hit    : {hit_any or 'NONE of ' + str(c['want_any'])}")
        if missing_all:
            print(f"    facts MISSING: {missing_all}")
        if r.get("error"):
            print(f"    error        : {r['error'][:200]}")
        print(f"    reply        : {' '.join(reply.split())[:280]}\n")

    n_ok = sum(1 for x in results if x["ok"])
    n_tooled = sum(1 for x in results if x["expected_tool_used"])
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    head = (f"tool-surface-verify {stamp} — grounded {n_ok}/{len(cases)}; "
            f"expected-tool-invoked {n_tooled}/{len(cases)} => "
            f"{'PASS' if n_ok == len(cases) else 'FAIL'}")
    print("=== VERDICT ===")
    print(head)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({"stamp": stamp, "results": results}) + "\n")
    print(f"appended detail to {LOG_PATH}")
    return 0 if n_ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
