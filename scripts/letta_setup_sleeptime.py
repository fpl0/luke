#!/usr/bin/env python3
"""Phase 3.1 — provision Letta native sleep-time (background) memory agent for luke-agent-claude.

Letta's sleep-time architecture (server 0.16.8): a **SleeptimeManager group** ties a *main* agent
(the user-facing one) to a background *sleeptime_agent* that shares the main agent's core memory
blocks and edits them off the main turn. Every `sleeptime_agent_frequency` main-agent turns, Letta
fires the sleeptime agent to consolidate/rewrite the shared blocks — this is the native version of
Luke's dream loop, and the whole point of the migration's "self-editing working memory."

THE ARCHITECTURAL CHOICE THAT MATTERS
-------------------------------------
The main agent (`luke-agent-claude`) runs on Claude via the OAuth bridge (:17596) — smart, fast,
user-facing, but it draws from the ONE scarce `CLAUDE_CODE_OAUTH_TOKEN` rate-limit pool the live
SDK-Luke session also uses (this is exactly what soft-blocks Phase 1.4). If the sleeptime agent
ALSO ran on Claude it would inherit that contention and be un-verifiable from inside any session.

So we point the **sleeptime agent at local qwen3:8b (Ollama :11434)** instead:
  - background memory consolidation does NOT need Claude-level intelligence (same division the
    current dream loop already makes — it's a summarize/prune/link job, not a reasoning job),
  - it runs OFF the main turn, so latency is irrelevant,
  - and crucially it costs ZERO OAuth quota — so **Phase 3 can be verified without the contention
    window that blocks 1.4.** Main = Claude (on-turn, scarce), sleeptime = qwen (off-turn, free).

WHAT THIS SCRIPT DOES  (all pure provisioning — no agent turn runs, no OAuth consumed)
  1. PATCH main agent  enable_sleeptime=True  -> server auto-creates `<main>-sleeptime`
     (agent_type=sleeptime_agent, sharing ALL of main's core blocks) + a SleeptimeManager group.
     Idempotent: the server only provisions when the agent has no multi_agent_group yet.
  2. PATCH the sleeptime agent's llm_config -> qwen3:8b (off the OAuth pool).
  3. Optionally set the group's sleeptime_agent_frequency.
  4. Verify from REST ground truth (group wiring, shared blocks, sleeptime model = qwen).

REVERT:  --revert  deletes the sleeptime agent, which cascades the group and resets the main
  agent's enable_sleeptime flag (agent_manager delete path). sqlite + recall untouched.

Run:   .letta-venv/bin/python scripts/letta_setup_sleeptime.py [--agent luke-agent-claude]
                                                                [--frequency 5] [--dry-run] [--revert]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Localhost must not go through the SOCKS proxy the shell exports.
for _v in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_v, None)

import httpx

BASE = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
QWEN_ENDPOINT = "http://localhost:11434/v1"

# The sleeptime agent's llm_config: local qwen via Ollama's OpenAI-compat surface — off the
# OAuth pool. Mirrors the existing luke-agent qwen config exactly (context_window 32000, etc.).
QWEN_LLM_CONFIG = {
    "model": "qwen3:8b",
    "model_endpoint_type": "openai",
    "model_endpoint": QWEN_ENDPOINT,
    "context_window": 32000,
    "put_inner_thoughts_in_kwargs": False,
    "temperature": 1.0,
    "parallel_tool_calls": False,
}


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE, timeout=30.0)


def find_agent(c: httpx.Client, name: str) -> dict | None:
    r = c.get("/v1/agents/", params={"name": name})
    r.raise_for_status()
    for a in r.json():
        if a.get("name") == name:
            return a
    # fallback: exact match over the full list
    r = c.get("/v1/agents/")
    r.raise_for_status()
    for a in r.json():
        if a.get("name") == name:
            return a
    return None


def sleeptime_group_for(c: httpx.Client, main_id: str) -> dict | None:
    """The SleeptimeManager group whose manager_agent_id is this main agent."""
    r = c.get("/v1/groups/")
    r.raise_for_status()
    for g in r.json():
        if g.get("manager_type") == "sleeptime" and g.get("manager_agent_id") == main_id:
            return g
    return None


def block_labels(c: httpx.Client, agent_id: str) -> list[tuple[str, str]]:
    r = c.get(f"/v1/agents/{agent_id}/core-memory/blocks")
    if r.status_code == 404:
        r = c.get(f"/v1/agents/{agent_id}/blocks")  # older path
    r.raise_for_status()
    return [(b["label"], b["id"]) for b in r.json()]


def revert(c: httpx.Client, main: dict) -> int:
    main_id = main["id"]
    grp = sleeptime_group_for(c, main_id)
    if not grp:
        print("Nothing to revert: no sleeptime group for", main["name"])
        return 0
    # Delete the sleeptime participant agent -> cascades group + resets enable_sleeptime.
    participants = grp.get("agent_ids") or []
    for pid in participants:
        r = c.delete(f"/v1/agents/{pid}")
        print(f"  deleted sleeptime agent {pid}: {r.status_code}")
    # Group should be gone; delete defensively if it lingers.
    if sleeptime_group_for(c, main_id):
        c.delete(f"/v1/groups/{grp['id']}")
        print(f"  deleted group {grp['id']}")
    print("Reverted. Main agent enable_sleeptime reset; sqlite + recall untouched.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="luke-agent-claude")
    ap.add_argument("--frequency", type=int, default=5,
                    help="sleeptime agent fires every N main-agent turns (Letta default 5)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    with _client() as c:
        # sanity: server up
        h = c.get("/v1/health/")
        if h.status_code != 200:
            print("Letta server not healthy:", h.status_code, file=sys.stderr)
            return 2

        main_agent = find_agent(c, args.agent)
        if not main_agent:
            print(f"Main agent '{args.agent}' not found", file=sys.stderr)
            return 2
        main_id = main_agent["id"]
        print(f"Main agent: {args.agent}  id={main_id}")

        if args.revert:
            return revert(c, main_agent)

        existing = sleeptime_group_for(c, main_id)
        if args.dry_run:
            print("DRY-RUN plan:")
            if existing:
                print(f"  group already exists ({existing['id']}, freq={existing.get('sleeptime_agent_frequency')})"
                      f" -> would ensure sleeptime model=qwen3:8b + freq={args.frequency}")
            else:
                print(f"  PATCH {args.agent} enable_sleeptime=True -> auto-create <name>-sleeptime + group")
                print(f"  PATCH sleeptime agent llm_config -> qwen3:8b @ {QWEN_ENDPOINT}")
                print(f"  set group sleeptime_agent_frequency={args.frequency}")
            return 0

        # 1. Provision (idempotent — server only creates if multi_agent_group is None).
        if not existing:
            r = c.patch(f"/v1/agents/{main_id}", json={"enable_sleeptime": True})
            r.raise_for_status()
            print("  enable_sleeptime=True -> sleeptime agent + group provisioned")
            existing = sleeptime_group_for(c, main_id)
            if not existing:
                print("ERROR: group not created after enable_sleeptime", file=sys.stderr)
                return 3
        else:
            print(f"  group already present ({existing['id']}) — idempotent, ensuring config")

        group_id = existing["id"]
        participants = existing.get("agent_ids") or []
        if len(participants) != 1:
            print(f"WARNING: expected 1 sleeptime participant, got {participants}", file=sys.stderr)
        sleeptime_id = participants[0]

        # 2. Point the sleeptime agent at local qwen (off the OAuth pool).
        r = c.get(f"/v1/agents/{sleeptime_id}")
        r.raise_for_status()
        cur_model = r.json().get("llm_config", {}).get("model")
        if cur_model != "qwen3:8b":
            r = c.patch(f"/v1/agents/{sleeptime_id}", json={"llm_config": QWEN_LLM_CONFIG})
            r.raise_for_status()
            print(f"  sleeptime agent model {cur_model} -> qwen3:8b (off OAuth pool)")
        else:
            print("  sleeptime agent already on qwen3:8b")

        # 3. Frequency.
        if existing.get("sleeptime_agent_frequency") != args.frequency:
            r = c.patch(f"/v1/groups/{group_id}",
                        json={"manager_config": {"manager_type": "sleeptime",
                                                 "manager_agent_id": main_id,
                                                 "sleeptime_agent_frequency": args.frequency}})
            if r.status_code >= 400:
                print(f"  (freq update returned {r.status_code}: {r.text[:200]})")
            else:
                print(f"  sleeptime_agent_frequency -> {args.frequency}")

        # 4. Verify from REST ground truth.
        print("\n=== VERIFY (REST ground truth) ===")
        grp = sleeptime_group_for(c, main_id)
        st = c.get(f"/v1/agents/{sleeptime_id}").json()
        main_blocks = {lbl for lbl, _ in block_labels(c, main_id)}
        st_blocks = {lbl for lbl, _ in block_labels(c, sleeptime_id)}
        shared = main_blocks & st_blocks
        ok = True
        checks = [
            ("group.manager_type == sleeptime", grp.get("manager_type") == "sleeptime"),
            ("group.manager_agent_id == main", grp.get("manager_agent_id") == main_id),
            ("sleeptime agent type", st.get("agent_type") == "sleeptime_agent"),
            ("sleeptime model == qwen3:8b (off OAuth pool)", st.get("llm_config", {}).get("model") == "qwen3:8b"),
            (f"frequency == {args.frequency}", grp.get("sleeptime_agent_frequency") == args.frequency),
            ("sleeptime shares main core blocks", main_blocks.issubset(st_blocks) and len(main_blocks) > 0),
        ]
        for name, passed in checks:
            print(f"  [{'OK' if passed else 'FAIL'}] {name}")
            ok = ok and passed
        print(f"  main blocks: {sorted(main_blocks)}")
        print(f"  shared with sleeptime: {sorted(shared)}")
        print(f"\nsleeptime agent: {st.get('name')}  id={sleeptime_id}")
        print(f"group: {group_id}  freq={grp.get('sleeptime_agent_frequency')}")
        print("\nRESULT:", "ALL PASS" if ok else "SOME FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
