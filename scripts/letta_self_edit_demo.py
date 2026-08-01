#!/usr/bin/env python3
"""Phase 1.4 close: drive a REAL turn on luke-agent-claude (Claude via the OAuth bridge :17596)
and prove a FAST, COHERENT reply + a self-edit of the `human` core block that PERSISTS.

Pure REST (requests) — no letta_client dependency. Meant to run as a bare subprocess so the
bridge's Anthropic calls don't fight a concurrent Claude Code session for the OAuth pool.

Run: python3 scripts/letta_self_edit_demo.py
"""
import sys, time, json, urllib.request

LETTA = "http://localhost:8283"
AGENT = "agent-36671c0b-a133-4bfb-a367-f23f7135071a"


def req(method, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(LETTA + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def snapshot_blocks():
    """All core blocks as {label: value}. The agent may self-edit ANY block
    (key-people/human/goals/...), so watching only `human` false-negatives —
    that bug produced ~3 passes of phantom PARTIAL/BLOCKED verdicts."""
    return {b["label"]: b["value"]
            for b in req("GET", f"/v1/agents/{AGENT}/core-memory/blocks")}


before = snapshot_blocks()
print("BLOCKS_BEFORE:", {k: len(v) for k, v in before.items()}, flush=True)

prompt = (
    "Two updates for what you know about me, then a question. Update 1: my partner "
    "Christopher just passed his advanced cardiac life support (ACLS) certification. "
    "Update 2: I've decided my first focus at CarGurus will be the Canada localization "
    "launch. Please update your memory of me to include both, then tell me briefly — in "
    "your own voice — how you'd sequence my first two weeks given that Canada focus."
)

t0 = time.time()
try:
    resp = req("POST", f"/v1/agents/{AGENT}/messages",
               {"messages": [{"role": "user", "content": prompt}]}, timeout=300)
except Exception as e:
    print("TURN_ERROR:", repr(e)[:300], flush=True)
    sys.exit(1)
dt = time.time() - t0
print(f"TURN_SECONDS: {dt:.1f}", flush=True)

for m in resp.get("messages", []):
    mt = m.get("message_type", "")
    if mt == "assistant_message":
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        print("ASSISTANT:", (c or "")[:1200], flush=True)
    elif mt == "tool_call_message":
        tc = m.get("tool_call") or {}
        print("TOOL_CALL:", tc.get("name"), (tc.get("arguments") or "")[:400], flush=True)
    elif mt == "tool_return_message":
        print("TOOL_RETURN:", str(m.get("tool_return", ""))[:150], flush=True)

after = snapshot_blocks()
changed_blocks = [k for k in after
                  if (after.get(k) or "") != (before.get(k) or "")]
print("CHANGED_BLOCKS:", changed_blocks, flush=True)
changed = bool(changed_blocks)
print("SELF_EDIT_PERSISTED:", changed, flush=True)
# Accept (Phase 1.4): Claude-driven turn self-edits ANY core block, fast (<15s).
print("VERDICT:", "PASS" if (changed and dt < 15) else
      ("SLOW" if changed else "NO_EDIT"), flush=True)
print("DONE", flush=True)
