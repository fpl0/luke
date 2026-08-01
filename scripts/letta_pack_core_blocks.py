#!/usr/bin/env python3
"""Phase 2.2a — core-block packing (the actual migration win).

The bulk importer flattens every memory to an *archival* passage (recalled on demand).
That is correct for recall parity, but it throws away the one thing the Letta migration
exists for: **self-editing working memory** — always-in-context core blocks the agent reads
every turn and edits as facts change. This replaces "re-inject the constitutional blob every
turn" with "it's just in core memory."

Per docs/letta-memory-model-mapping.md §3, route by importance+type+id into named core blocks
on the migration-target agent (``luke-agent-claude``, the one on the Claude bridge):

  key-people        entities id LIKE 'person-'  imp>=1.5   (the load-bearing world model)
  key-projects      entities id LIKE 'project-' imp>=1.5
  preferences       entity user-preferences (+ any other imp>=1.5 entity)
  operating-rules   curated hard directives (read_only=True — guardrails, not world-model)
  goals             active goal memories (placeholder if none; MSc is paused)

Idempotent: a block with the label is UPDATEd in place; otherwise created + attached. Safe to
re-run — it never duplicates blocks, never deletes archival passages, never touches sqlite.
Ground-truth verified at the end by re-fetching the agent's blocks from Letta.

Run:  .letta-venv/bin/python scripts/letta_pack_core_blocks.py [--agent NAME] [--dry-run]
Revert: detach/delete the five blocks; sqlite + recall are untouched.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

from letta_client import Letta

DB_PATH = "/Users/filipelm/Luke/luke.db"
DEFAULT_AGENT = "luke-agent-claude"
BASE_URL = "http://localhost:8283"

# Curated hard directives for the read-only operating-rules block. Guardrails, not world-model —
# the agent should read but never self-edit these (native read_only enforces it). Missing ids are
# skipped gracefully (some hard rules live in the MEMORY.md file layer, not the sqlite store).
GUARDRAIL_IDS = [
    "feedback-stay-on-sqlite",
    "feedback-no-internal-text-leak",
    "feedback-git-authorship",
    "feedback-keep-agent-sdk",
    "feedback-dates-accuracy",
    "feedback-security-ask-first",
    "feedback-present-before-building",
    "feedback-own-source-code-autonomy",
    "feedback-answer-about-filipe-not-luke",
    "feedback-companion-not-tool",
    "feedback-anticipatory-outcomes-for-filipe",
]


def _rows(db, sql, args=()):
    return db.execute(sql, args).fetchall()


def _fetch(db, mem_id):
    r = db.execute(
        """SELECT m.id, m.importance, f.title, f.content
           FROM memory_meta m JOIN memory_fts f ON m.id = f.id
           WHERE m.id = ? AND m.status='active'""",
        (mem_id,),
    ).fetchone()
    return r


def _section(mem_id, importance, title, content):
    hdr = f"## {mem_id}"
    if importance is not None:
        hdr += f"  (importance {importance:.2f})"
    body = (content or "").strip()
    return f"{hdr}\n{body}"


def build_blocks(db):
    """Return list of (label, value, read_only, description) from ground-truth sqlite."""
    blocks = []

    # key-people / key-projects / preferences — entities imp>=1.5, routed by id prefix.
    ents = _rows(
        db,
        """SELECT m.id, m.importance, f.title, f.content
           FROM memory_meta m JOIN memory_fts f ON m.id=f.id
           WHERE m.type='entity' AND m.status='active' AND m.importance>=1.5
           ORDER BY m.importance DESC""",
    )
    people, projects, prefs = [], [], []
    for r in ents:
        sec = _section(r["id"], r["importance"], r["title"], r["content"])
        if r["id"].startswith("person-"):
            people.append(sec)
        elif r["id"].startswith("project-"):
            projects.append(sec)
        else:
            prefs.append(sec)
    if people:
        blocks.append(
            ("key-people", "\n\n".join(people), False,
             "Load-bearing world model: who Filipe is and the people who matter. Self-edit as facts change.")
        )
    if projects:
        blocks.append(
            ("key-projects", "\n\n".join(projects), False,
             "Active projects/life-state context. Self-edit as they evolve.")
        )
    if prefs:
        blocks.append(
            ("preferences", "\n\n".join(prefs), False,
             "How Filipe wants to be worked with. Self-edit as preferences are learned.")
        )

    # operating-rules — curated hard directives, READ-ONLY.
    rules = []
    for gid in GUARDRAIL_IDS:
        r = _fetch(db, gid)
        if r is not None:
            rules.append(_section(r["id"], None, r["title"], r["content"]))
    if rules:
        header = ("GUARDRAILS — hard directives. READ every turn; do NOT self-edit "
                  "(these are constraints, not world-model).\n\n")
        blocks.append(
            ("operating-rules", header + "\n\n".join(rules), True,
             "Hard behavioral directives. Read-only guardrails.")
        )

    # goals — active goal memories; placeholder if none (MSc paused, Letta lives as a plan file).
    goals = _rows(
        db,
        """SELECT m.id, m.importance, f.title, f.content
           FROM memory_meta m JOIN memory_fts f ON m.id=f.id
           WHERE m.type='goal' AND m.status='active' ORDER BY m.importance DESC""",
    )
    if goals:
        val = "\n\n".join(_section(r["id"], r["importance"], r["title"], r["content"]) for r in goals)
    else:
        val = ("No active goal memories right now (the MSc goal is paused). "
               "Current build focus lives as a plan file, not a goal memory. "
               "Self-edit this block when an active goal is set.")
    blocks.append(
        ("goals", val, False, "Active objectives. Self-edit progress/status as work advances.")
    )
    return blocks


def resolve_agent(c, name):
    agents = [a for a in c.agents.list(name=name)]
    if not agents:
        sys.exit(f"no agent named {name!r}")
    # newest first — pick the most recently created if duplicates exist
    agents.sort(key=lambda a: getattr(a, "created_at", "") or "", reverse=True)
    return agents[0]


def upsert(c, agent_id, existing_labels, label, value, read_only, description):
    limit = max(8000, len(value) + 2000)
    if label in existing_labels:
        c.agents.blocks.update(
            label, agent_id=agent_id, value=value, limit=limit,
            read_only=read_only, description=description,
        )
        return "updated"
    blk = c.blocks.create(
        label=label, value=value, limit=limit,
        read_only=read_only, description=description,
    )
    c.agents.blocks.attach(blk.id, agent_id=agent_id)
    return "created+attached"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=DEFAULT_AGENT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    blocks = build_blocks(db)

    print(f"built {len(blocks)} core blocks from {DB_PATH}:")
    for label, value, ro, _ in blocks:
        print(f"  {label:16} {len(value):6d} chars  read_only={ro}")
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    c = Letta(base_url=BASE_URL)
    agent = resolve_agent(c, args.agent)
    print(f"\ntarget agent: {agent.id}  ({args.agent})")
    existing = {b.label for b in c.agents.blocks.list(agent_id=agent.id)}
    print(f"existing blocks: {sorted(existing)}")

    for label, value, ro, description in blocks:
        action = upsert(c, agent.id, existing, label, value, ro, description)
        print(f"  {label:16} -> {action}")

    # Ground-truth verify: re-fetch and assert each packed label is present with content.
    print("\nverify (re-fetched from Letta):")
    final = {b.label: b for b in c.agents.blocks.list(agent_id=agent.id)}
    ok = True
    for label, value, ro, _ in blocks:
        b = final.get(label)
        if b is None or not (b.value or "").strip():
            print(f"  FAIL {label}: missing/empty")
            ok = False
        else:
            ro_actual = getattr(b, "read_only", None)
            print(f"  OK   {label:16} len={len(b.value):6d} read_only={ro_actual}")
    print("\nRESULT:", "all blocks packed + verified" if ok else "VERIFICATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
