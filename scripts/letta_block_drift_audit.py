#!/usr/bin/env python3
"""Phase 5.5 — core-block freshness / drift audit (the parity gate).

The 5 deterministic core blocks (key-people, key-projects, preferences,
operating-rules, goals) are *projections of sqlite*: ``letta_pack_core_blocks.build_blocks``
derives each one from current sqlite by id/type/importance routing. But they are packed
once and only self-edit when a turn runs THROUGH the Letta agent — the production SDK path
updates sqlite, never the blocks. So they DRIFT from sqlite reality between packs.

This audit is the gate for the 5.5 accept ("3 consecutive days where the blocks reflect the
last 48h of sqlite changes"): for each deterministic block it rebuilds the EXPECTED value from
current sqlite and diffs it against the LIVE block fetched from Letta. FRESH = live == expected;
DRIFTED otherwise (with a char-delta + a first-divergence marker).

  - persona / human are EXCLUDED: they are agent-self-edited, not sqlite projections.
  - Pre-cutover, sqlite is the master, so the blocks SHOULD track it → this audit is correct now.
    (Post-cutover the blocks become the self-editing master; a self-edit would then be a legitimate
    divergence, not drift — revisit the direction of this audit at Stage 2. See plan §5.5.)

Exit 0 iff every deterministic block is FRESH. Append a dated result line to
``logs/letta_block_freshness.log`` (the 3-consecutive-days ledger) unless --no-log.

Run:  .letta-venv/bin/python scripts/letta_block_drift_audit.py [--agent NAME] [--no-log]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone

from letta_client import Letta

# Reuse the SINGLE source of truth for what each block should contain.
from letta_pack_core_blocks import BASE_URL, DB_PATH, build_blocks, resolve_agent

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "letta_block_freshness.log")


def _norm(s: str) -> str:
    """Whitespace-insensitive compare — a trailing newline is not drift."""
    return "\n".join(line.rstrip() for line in (s or "").strip().splitlines())


def _first_divergence(a: str, b: str) -> int:
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))


def audit(agent_name: str):
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    expected = {label: value for (label, value, _ro, _desc) in build_blocks(db)}

    c = Letta(base_url=BASE_URL)
    agent = resolve_agent(c, agent_name)
    live = {b.label: (b.value or "") for b in c.agents.blocks.list(agent_id=agent.id)}

    results = []  # (label, verdict, detail)
    for label, exp in expected.items():
        got = live.get(label)
        if got is None:
            results.append((label, "MISSING", "block not attached to agent"))
            continue
        if _norm(got) == _norm(exp):
            results.append((label, "FRESH", f"{len(got)} chars"))
        else:
            div = _first_divergence(_norm(got), _norm(exp))
            results.append(
                (label, "DRIFTED",
                 f"live={len(got)} expected={len(exp)} chars; first-diff@{div}")
            )
    return agent.id, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="luke-agent-claude")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    agent_id, results = audit(args.agent)
    print(f"agent: {agent_id} ({args.agent})")
    all_fresh = all(v == "FRESH" for _, v, _ in results)
    for label, verdict, detail in results:
        print(f"  {verdict:8} {label:16} {detail}")
    verdict = "ALL FRESH" if all_fresh else "DRIFT DETECTED"
    print(f"\nRESULT: {verdict}")

    if not args.no_log:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        drifted = [l for l, v, _ in results if v != "FRESH"]
        line = (f"{stamp}  {verdict:14}  "
                + ("all 5 blocks match sqlite" if all_fresh
                   else f"drifted: {','.join(drifted)}") + "\n")
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as fh:
            fh.write(line)
        print(f"logged -> {os.path.relpath(LOG_PATH)}")

    sys.exit(0 if all_fresh else 1)


if __name__ == "__main__":
    main()
