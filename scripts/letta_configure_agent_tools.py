#!/usr/bin/env python3
"""Phase 4.3 — configure luke-agent-claude's tool set to match the migration architecture.

Ground-truth bug found 2026-08-01 (11:00 pass): the agent shipped with Letta's native
`conversation_search` tool, which throws a Postgres `UndefinedFunctionError` in this
deployment (a message-store SQL function that isn't installed). When the agent chose to
call it mid-turn, the tool errored and the turn ended on the "let me pull some context"
preamble — a STUBBED reply. It looked like a heartbeat/loop-continuation gap; it was a
broken tool.

Architectural fix (not just a bug fix): per the plan invariant, **Letta owns the
store + self-edit surface; Luke's own recall (FTS5+semantic+RRF+graph in memory.py)
owns retrieval.** So the Letta agent should NOT use Letta's native conversation_search
at all — it retrieves via Luke's recall (adapter + write-through, Phase 2). This script
detaches conversation_search and keeps only the self-edit tools (memory_insert,
memory_replace), and clears any stray tool_rules.

Idempotent: re-running is a no-op once the agent already has exactly the target tool set.
Reversible: re-attach conversation_search by tool id if ever needed.

Run: python3 scripts/letta_configure_agent_tools.py
"""
import json, sys, urllib.request

LETTA = "http://localhost:8283"
AGENT = "agent-36671c0b-a133-4bfb-a367-f23f7135071a"  # luke-agent-claude
KEEP = {"memory_insert", "memory_replace"}
DROP = {"conversation_search"}  # broken in this deployment + wrong retrieval path for Luke


def req(method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(LETTA + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(r, timeout=timeout)
    raw = resp.read().decode()
    return json.loads(raw) if raw else {}


def current_tools():
    a = req("GET", f"/v1/agents/{AGENT}")
    return {t["name"]: t["id"] for t in a.get("tools", [])}, a.get("tool_rules", [])


tools, rules = current_tools()
print("TOOLS_BEFORE:", sorted(tools), flush=True)
print("RULES_BEFORE:", rules, flush=True)

# Clear stray tool_rules (a continue_loop rule on the broken tool made the stub worse).
if rules:
    req("PATCH", f"/v1/agents/{AGENT}", {"tool_rules": []})
    print("cleared tool_rules", flush=True)

# Detach any tool in DROP.
for name in list(tools):
    if name in DROP:
        req("PATCH", f"/v1/agents/{AGENT}/tools/detach/{tools[name]}")
        print(f"detached: {name}", flush=True)

tools_after, rules_after = current_tools()
names = set(tools_after)
print("TOOLS_AFTER:", sorted(names), flush=True)
print("RULES_AFTER:", rules_after, flush=True)

ok = KEEP.issubset(names) and not (names & DROP) and not rules_after
print("VERDICT:", "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
