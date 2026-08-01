#!/usr/bin/env python3
"""Phase 4.3 — per-turn archive injection, A/B proof.

Core blocks (build_letta_context) give the always-in-context world model. They do
NOT hold the long tail of the ~1500-passage archive. This proves the missing half:
prepend Luke's own recall() top-k into the turn so a Letta reply can ground facts
that live ONLY in the archive, not in the packed core blocks.

Test fact: the B1 US-visa interview (entity-b1-visa-application) — Fri 7 Aug 2026,
07:45, 42 Elgin Road, Ballsbridge. It is NOT in any of the 7 core blocks
(key-people / key-projects / preferences / goals / operating-rules), so a turn
answering from core blocks alone must miss or confabulate it; the SAME turn with
recall injection must surface the real date + address.

A/B, live end-to-end (core blocks → Letta ReAct → bridge :17596 → Claude → reply):
  A = baseline: user question alone.
  B = injected: build_recall_injection(question) prepended to the same question.

Pass = B surfaces the ground-truth facts AND A does materially worse (miss or
generic non-answer). That delta is the value of the injection.

Pure REST for the turn; imports Luke's real recall() for the injection. Off no
special window — the cached bridge (Phase 1.4) serves it. Run from repo root:
  .venv/bin/python scripts/letta_recall_injection.py
"""
import sys, time, json, urllib.request

sys.path.insert(0, "src")
from luke.letta_agent import build_recall_injection, compose_letta_turn_input  # noqa: E402

LETTA = "http://localhost:8283"
AGENT = "agent-36671c0b-a133-4bfb-a367-f23f7135071a"  # luke-agent-claude

QUESTION = (
    "What are the exact details of my US B1 visa interview — the date, time, and "
    "the address I need to show up at? I want to make sure I've got them right."
)
# Ground-truth tokens that only recall injection can bring into the turn.
GROUND_TRUTH = {"date": ["7 aug", "august 7", "aug 7", "7th"], "addr": ["elgin", "ballsbridge"]}


def req(method, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(LETTA + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def drive(user_msg):
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


def facts_present(text):
    low = text.lower()
    has_date = any(t in low for t in GROUND_TRUTH["date"])
    has_addr = any(t in low for t in GROUND_TRUTH["addr"])
    return has_date, has_addr


# --- Build the injection off the pool first; prove recall actually found the fact. ---
inj = build_recall_injection(QUESTION)
if not inj:
    print("INJECTION_BUILD: None — recall returned nothing (fail-safe). ABORT.", flush=True)
    sys.exit(1)
inj_has_visa = "b1-visa" in inj.lower() or "elgin" in inj.lower()
print(f"INJECTION_BUILD: {len(inj)} chars, contains-visa-entity={inj_has_visa}", flush=True)
print(f"INJECTION_PREVIEW:\n{inj[:600]}\n---", flush=True)

# --- A: baseline (no injection) ---
dtA, textA, toolsA, errA = drive(QUESTION)
dateA, addrA = facts_present(textA) if not errA else (False, False)
print(f"\n[A baseline] {dtA:.1f}s tools={toolsA} err={errA}", flush=True)
print(f"[A baseline] date={dateA} addr={addrA}", flush=True)
print(f"[A baseline] REPLY: {textA[:700]}", flush=True)

# --- B: injected via the SAME production composition seam the cutover path uses ---
injected_msg = compose_letta_turn_input(QUESTION)
dtB, textB, toolsB, errB = drive(injected_msg)
dateB, addrB = facts_present(textB) if not errB else (False, False)
print(f"\n[B injected] {dtB:.1f}s tools={toolsB} err={errB}", flush=True)
print(f"[B injected] date={dateB} addr={addrB}", flush=True)
print(f"[B injected] REPLY: {textB[:700]}", flush=True)

# --- Verdict ---
print("\n=== VERDICT ===", flush=True)
b_ok = bool(textB) and dateB and addrB
# The injection has value if B grounds the facts and B is strictly better than A on them.
a_score = int(dateA) + int(addrA)
b_score = int(dateB) + int(addrB)
delta = b_score > a_score
print(f"  B grounds facts (date+addr): {b_ok}", flush=True)
print(f"  A facts={a_score}/2  B facts={b_score}/2  B>A: {delta}", flush=True)
verdict = "PASS" if (b_ok and delta) else ("NEUTRAL" if b_ok else "FAIL")
print(f"VERDICT: {verdict}", flush=True)
print("DONE", flush=True)
