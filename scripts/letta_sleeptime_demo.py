#!/usr/bin/env python3
"""Phase 3.2 — prove the qwen sleep-time agent consolidates real episodes into a shared core
block, OFF the OAuth pool.

Phase 3.1 provisioned `luke-agent-claude-sleeptime` (agent-57636380..., model qwen3:8b on
Ollama :11434) sharing all of the main agent's core blocks. This drives it DIRECTLY (a plain
REST message → qwen turn, zero Claude/OAuth quota) with a batch of recent episodes and asks it
to consolidate the current active-work state into the `goals` block (a 169-char placeholder —
improving it is a legitimate consolidation that doesn't risk the dense 2.2a world-model blocks).

Acceptance (Phase 3.2): the sleep-time agent, on qwen, fires a memory_* tool and the target
shared block's value changes — off-turn self-editing works off the contended pool that blocks 1.4.

SAFETY: snapshots every block first; restores any block OTHER than the target if qwen mutated it,
so the carefully-packed 2.2a blocks (key-people/key-projects/preferences) and the read_only
operating-rules guardrail are left exactly as they were. Fully reversible; sqlite untouched.

Run:  python3 scripts/letta_sleeptime_demo.py [--target goals]
"""
import sys, time, json, argparse, sqlite3, urllib.request

LETTA = "http://localhost:8283"
SLEEPTIME = "agent-57636380-0d33-425e-a39e-4e31518899c2"
DB = "/Users/filipelm/Luke/luke.db"


def req(method, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(LETTA + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def blocks():
    return {b["label"]: b for b in req("GET", f"/v1/agents/{SLEEPTIME}/core-memory/blocks")}


def recent_episodes(n=6):
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT f.c2, f.c3 FROM memory_meta m "
        "JOIN memory_fts_content f ON f.c0 = m.id "
        "WHERE m.type='episode' AND m.status='active' "
        "ORDER BY m.created DESC LIMIT ?", (n,)).fetchall()
    con.close()
    return [(t, (c or "")[:600]) for t, c in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="goals")
    args = ap.parse_args()
    target = args.target

    snap = blocks()
    if target not in snap:
        print("NO_TARGET_BLOCK:", target, "have:", list(snap)); sys.exit(1)
    before = {lbl: b.get("value") or "" for lbl, b in snap.items()}
    print("TARGET:", target, "read_only=", snap[target].get("read_only"))
    print("TARGET_BEFORE:", before[target][:300], flush=True)

    eps = recent_episodes()
    digest = "\n".join(f"- {t}: {c}" for t, c in eps)
    prompt = (
        "You are Luke's background memory-consolidation agent. Below are the most recent episodes "
        "from Luke's life/work. Consolidate what they say about his CURRENT active work and its "
        "state into the `" + target + "` core memory block — a concise, current snapshot of what "
        "he is actively working on and where each thread stands. Use your memory-editing tools to "
        "rewrite the `" + target + "` block, then call memory_finish_edits. Do NOT touch any other "
        "block.\n\nRECENT EPISODES:\n" + digest
    )

    t0 = time.time()
    try:
        resp = req("POST", f"/v1/agents/{SLEEPTIME}/messages",
                   {"messages": [{"role": "user", "content": prompt}]}, timeout=600)
    except Exception as e:
        print("TURN_ERROR:", repr(e)[:300], flush=True); sys.exit(1)
    dt = time.time() - t0
    print(f"TURN_SECONDS: {dt:.1f}", flush=True)

    tool_fired = []
    for m in resp.get("messages", []):
        mt = m.get("message_type", "")
        if mt == "tool_call_message":
            tc = m.get("tool_call") or {}
            tool_fired.append(tc.get("name"))
            print("TOOL_CALL:", tc.get("name"), (tc.get("arguments") or "")[:300], flush=True)
        elif mt == "tool_return_message":
            print("TOOL_RETURN:", str(m.get("tool_return", ""))[:120], flush=True)
        elif mt == "assistant_message":
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            if c:
                print("ASSISTANT:", c[:400], flush=True)

    after_blocks = blocks()
    after = {lbl: b.get("value") or "" for lbl, b in after_blocks.items()}

    target_changed = after.get(target, "") != before[target]
    print("\nTARGET_AFTER:", after.get(target, "")[:400], flush=True)
    print("TARGET_CHANGED:", target_changed, flush=True)

    # Guardrail + safety: report and restore any OTHER block qwen mutated.
    collateral = [lbl for lbl in before if lbl != target and after.get(lbl, "") != before[lbl]]
    print("COLLATERAL_EDITS:", collateral, flush=True)
    ro_label = "operating-rules"
    print("OPERATING_RULES_INTACT:", after.get(ro_label, "") == before.get(ro_label, ""), flush=True)
    for lbl in collateral:
        bid = snap[lbl]["id"]
        req("PATCH", f"/v1/blocks/{bid}", {"value": before[lbl]})
        print("RESTORED:", lbl, flush=True)

    memory_tool = any(t and t.startswith("memory_") for t in tool_fired)
    verdict = "PASS" if (target_changed and memory_tool) else "PARTIAL"
    print("MEMORY_TOOL_FIRED:", memory_tool, "tools:", tool_fired, flush=True)
    print("VERDICT:", verdict, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
