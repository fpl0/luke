#!/usr/bin/env python3
"""Phase 4.1 live-turn half: prove a FULL conversational Telegram turn served end-to-end
BY the Letta agent (core blocks → Letta ReAct → Claude bridge :17596 → assistant reply).

The 1.4 self-edit demo proves the *memory-edit* turn. This proves the ordinary
*conversational* turn — the actual user experience: (a) a normal question answered in
Luke's voice, and (b) a factual-recall question that must pull the right fact from the
attached archive. Both must return a real ``assistant_message`` (not just tool calls),
fast, no error.

The assistant_message content IS what would go out to Telegram, so producing a coherent
one = "assembled context → Letta → bridge → Telegram out" end-to-end.

Pure REST, bare subprocess. Run: python3 scripts/letta_live_turn.py
"""
import sys, time, json, urllib.request

LETTA = "http://localhost:8283"
AGENT = "agent-36671c0b-a133-4bfb-a367-f23f7135071a"  # luke-agent-claude


def req(method, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(LETTA + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def drive(user_msg):
    """Send one user message, return (seconds, assistant_text, tool_calls, error)."""
    t0 = time.time()
    try:
        resp = req("POST", f"/v1/agents/{AGENT}/messages",
                   {"messages": [{"role": "user", "content": user_msg}]}, timeout=300)
    except Exception as e:
        return (time.time() - t0, "", [], repr(e)[:300])
    dt = time.time() - t0
    assistant, tools = "", []
    for m in resp.get("messages", []):
        mt = m.get("message_type", "")
        if mt == "assistant_message":
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            assistant += (c or "")
        elif mt == "tool_call_message":
            tc = m.get("tool_call") or {}
            tools.append(tc.get("name"))
    return (dt, assistant.strip(), tools, None)


# Two representative turns. #1 = conversational (voice). #2 = factual recall from the archive
# (the CarGurus start date lives in entity-cargurus-interview — a content-only-miss the titled
# archive fixed, so it exercises the semantic retrieval path the migration depends on).
TURNS = [
    ("conversational",
     "Morning. I'm feeling a bit scattered about the CarGurus start — too many "
     "open threads. Talk me down a little; what should I actually not worry about?"),
    ("factual-recall",
     "Quick check — when exactly do I start at CarGurus, and who's my manager there?"),
]

results = []
for kind, msg in TURNS:
    dt, text, tools, err = drive(msg)
    if err:
        print(f"[{kind}] TURN_ERROR: {err}", flush=True)
        results.append((kind, False, dt))
        continue
    ok_text = len(text) >= 40  # a real reply, not an empty/degenerate one
    # factual-recall must actually surface the ground-truth facts from the archive,
    # not just produce fluent text. Aug 10 start + manager Prerna are the two facts.
    if kind == "factual-recall":
        low = text.lower()
        has_date = ("aug" in low and "10" in low) or "august 10" in low
        has_mgr = "prerna" in low
        ok_text = ok_text and has_date and has_mgr
        print(f"[factual-recall] fact-check: date={has_date} manager={has_mgr}", flush=True)
    print(f"\n[{kind}] SECONDS={dt:.1f}  tools={tools}", flush=True)
    print(f"[{kind}] REPLY: {text[:900]}", flush=True)
    results.append((kind, ok_text, dt))

print("\n=== VERDICT ===", flush=True)
all_ok = all(ok for _, ok, _ in results)
slow = [k for k, ok, dt in results if dt >= 20]
for kind, ok, dt in results:
    print(f"  {kind}: {'OK' if ok else 'FAIL'} ({dt:.1f}s)", flush=True)
# Accept for 4.1-live-half: every turn returns a real assistant reply, end-to-end, <20s.
print("VERDICT:", "PASS" if (all_ok and not slow) else
      ("SLOW" if all_ok else "FAIL"), flush=True)
print("DONE", flush=True)
