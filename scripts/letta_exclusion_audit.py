#!/usr/bin/env python3
"""EXCLUSION AUDIT — is the 5.1R pack still fairly narrow, given the tools actually attached?

``letta_reply_diff.py`` drops prompts it judges unmeasurable, and the biggest drop class is
``tool-dependent``: asks that need a CAPABILITY rather than a memory. That filter was written
on 2026-08-01 when the Letta agent had exactly two tools (``memory_insert``/``memory_replace``),
so on any such ask the SDK arm won by having tools — which says nothing about the memory
substrate the gate exists to test.

Since then the agent gained eight read-only tools. That makes the exclusion list a **standing
liability**: every prompt it drops for "you have no tool for this" is a claim about the tool
surface, and the tool surface moved. Left alone the filter silently narrows the pack to the
questions Letta was weakest at *last week*, and the gate would keep reporting a number for a
configuration that no longer exists.

So this script does not hardcode what Letta can do. It **reads the attached tool list off the
live agent**, maps each excluded prompt to the capability DOMAIN it needs, and asserts:

    for every excluded prompt, the domain it needs has NO covering tool attached

A row whose domain is now covered is no longer unfair — it is a row the pack is missing, and
the script exits non-zero naming it. Attach a web or calendar tool tomorrow and this flags the
rows to re-admit, without anyone having to remember that the filter encodes an assumption.

Classification is regex-based and every row prints with its domain and trigger, because the
point is a reviewer can check all of them on one screen — not that a regex is authoritative.

Run: python3 scripts/letta_exclusion_audit.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

LETTA = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
AGENT = os.environ.get("LETTA_AGENT_ID", "agent-36671c0b-a133-4bfb-a367-f23f7135071a")
LOG_PATH = os.path.join(REPO, "logs", "letta_exclusion_audit.log")

# Capability domains, and the tools that would make an ask in that domain FAIR to include.
# A domain with no attached tool means the SDK arm wins on capability, so the row stays out.
# Keyed on tool NAME, matched against whatever is attached to the live agent, so this table
# describes the requirement and the agent describes reality.
DOMAIN_TOOLS: dict[str, set[str]] = {
    "email":          {"luke_read_email", "luke_search_email", "read_email"},
    "calendar":       {"luke_read_calendar", "luke_calendar", "read_calendar"},
    "web":            {"luke_web_search", "luke_browse", "web_search", "browse"},
    "write-schedule": {"luke_schedule_task", "luke_create_task", "schedule_task"},
    "write-send":     {"luke_send_message", "send_message"},
    "repo-read":      {"luke_read_file", "luke_search_code", "luke_list_dir", "luke_git", "luke_tail_log"},
    "task-read":      {"luke_list_tasks"},
    "message-read":   {"luke_search_messages", "luke_recall"},
}

# Ordered — first match wins, most specific first. A prompt that names email AND scheduling
# ("monitor my inbox and run it on a schedule") is an email ask first: that is the capability
# it fails on soonest.
DOMAIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email",    re.compile(r"\b(e-?mails?|inbox|spam|sent (mail|email)|outlook)\b", re.I)),
    ("calendar", re.compile(r"\b(calendar|find times?|availability|free slots?)\b", re.I)),
    ("web",      re.compile(r"\b(google|browse|scrape|reddit|web|online|reviews from|"
                            r"search for (other|more|some|any)|book (a |the )?(hotel|flight)|"
                            r"flights?|hotels?|https?://)\b", re.I)),
    ("write-schedule", re.compile(r"\b(remind\w* me|schedule|set (a |up )?(reminder|alarm)|"
                                  r"scheduled? tasks?)\b", re.I)),
    ("write-send",     re.compile(r"\b(send (it|this|him|her|them|an? )|reply to|forward)\b", re.I)),
    ("repo-read",      re.compile(r"\b(read the file|open the|the code ?base|the repo|the logs?|"
                                  r"git |commit)\b", re.I)),
    ("task-read",      re.compile(r"\b(tasks?|cron|jobs?)\b", re.I)),
    ("message-read",   re.compile(r"\b(messages?|conversation|chat history)\b", re.I)),
]

# An ask to DRAFT something the human will send needs no tool in either arm — it is pure
# generation off the world model, which is exactly what 5.1R measures. Checked before the
# domain patterns because "draft a message I can send her" trips ``write-send`` on the word
# the human owns, not the one Luke does.
_DRAFT_NOT_SEND = re.compile(
    r"\b(draft|write|compose)\b(?!\w).{0,60}\b(that|which|so)?\s*i (can|could|will|'ll)\s*send\b",
    re.I | re.S,
)


# Rows the domain patterns cannot classify, with the call recorded. An unclassifiable row is
# NOT automatically fair to exclude — treating "I could not tell" as "still unfair" is the
# convenient direction, and it would let the pack quietly shrink whenever a prompt is oddly
# worded. So an unknown row FAILS the audit until it is adjudicated here, in writing.
ADJUDICATED: dict[int, str] = {
    2731: "deictic complaint ('can't you just search for how to make the right change?') about "
          "the immediately-preceding code discussion — unanswerable from memory by either arm, "
          "so it measures conversation-buffer access, not the substrate. Excluded on the same "
          "grounds as the deictic penalty, not on tool surface.",
}


def attached_tools() -> set[str]:
    req = urllib.request.Request(f"{LETTA}/v1/agents/{AGENT}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        agent = json.load(r)
    return {t.get("name", "") for t in (agent.get("tools") or [])}


def domain_of(prompt: str) -> tuple[str, str]:
    """Return (domain, matched-trigger). ``draft`` short-circuits to the no-tool-needed case."""
    if _DRAFT_NOT_SEND.search(prompt):
        return "none-needed", "draft-for-human-to-send"
    for name, pat in DOMAIN_PATTERNS:
        m = pat.search(prompt)
        if m:
            return name, m.group(0).strip()
    return "unclassified", ""


def main() -> int:
    spec = importlib.util.spec_from_file_location("rd", os.path.join(REPO, "scripts", "letta_reply_diff.py"))
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)

    try:
        tools = attached_tools()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # No live agent means no ground truth about the tool surface. Refusing beats guessing:
        # a hardcoded fallback is exactly the stale assumption this script exists to kill.
        print(f"FAIL: cannot read attached tools from {LETTA} ({e}). Start the Letta server.")
        return 2
    print(f"attached tools ({len(tools)}): {', '.join(sorted(tools))}\n")

    covered = {d for d, need in DOMAIN_TOOLS.items() if need & tools}
    print(f"domains COVERED by the attached surface: {', '.join(sorted(covered)) or '(none)'}")
    print(f"domains NOT covered: {', '.join(sorted(set(DOMAIN_TOOLS) - covered))}\n")

    dbp = rd._db_path()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=rd.LOOKBACK_DAYS)).isoformat()
    con = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, sender, content, ts FROM messages WHERE ts >= ? ORDER BY id ASC", (cutoff,)
    ).fetchall()
    con.close()

    unfair, fair, unknown = [], [], []
    for mid, sender, content, ts in rows:
        if sender.startswith("Luke"):
            continue
        prompt = (content or "").strip()
        if len(prompt) < rd.MIN_PROMPT_CHARS:
            continue
        if rd._excluded(prompt) != "tool-dependent":
            continue
        domain, trigger = domain_of(prompt)
        row = (mid, ts, domain, trigger, prompt)
        if domain in covered or domain == "none-needed":
            fair.append(row)
        elif domain == "unclassified" and mid not in ADJUDICATED:
            unknown.append(row)
        else:
            unfair.append(row)

    print(f"{'':2} {'msg':>6}  {'when':16}  {'domain':15} {'trigger':26} prompt")
    for mid, ts, domain, trigger, prompt in sorted(unfair + fair + unknown, key=lambda r: r[0]):
        mark = "!!" if (mid, ts, domain, trigger, prompt) in fair else (
            "??" if (mid, ts, domain, trigger, prompt) in unknown else "  ")
        adj = " [adjudicated]" if mid in ADJUDICATED else ""
        print(f"{mark} #{mid:<5} {ts[:16]}  {domain + adj:15} {trigger[:24]:26} {prompt[:88]}")

    total = len(unfair) + len(fair) + len(unknown)
    print(f"\n{len(unfair)}/{total} excluded rows need a domain with NO attached tool "
          f"(incl. {len(ADJUDICATED)} adjudicated by hand) — still fair to exclude.")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if fair or unknown:
        if fair:
            print(f"\nFAIL: {len(fair)} excluded row(s) are now within the attached tool surface. "
                  f"The pack is narrower than the agent's actual capability — re-admit or justify:")
            for mid, _ts, domain, _trigger, prompt in fair:
                print(f"  #{mid} [{domain}] {prompt[:110]}")
        if unknown:
            print(f"\nFAIL: {len(unknown)} excluded row(s) could not be classified. An unknown "
                  f"row is not a justified exclusion — add it to ADJUDICATED with a reason:")
            for mid, _ts, _domain, _trigger, prompt in unknown:
                print(f"  #{mid} {prompt[:110]}")
        _log(f"{stamp}  FAIL  covered={','.join(str(r[0]) for r in fair) or '-'} "
             f"unclassified={','.join(str(r[0]) for r in unknown) or '-'} of {total}")
        return 1

    print("\nPASS: every tool-dependent exclusion still names a capability the Letta arm lacks. "
          "The pack is fairly narrow, not conveniently narrow.")
    _log(f"{stamp}  PASS  {total} exclusions all uncovered; tools={len(tools)}")
    return 0


def _log(line: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
