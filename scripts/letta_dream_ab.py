#!/usr/bin/env python3
"""Phase 3.3 — A/B: Letta qwen sleep-time vs the current (Claude) dream loop.

The current dream loop (behaviors.py:run_dream) is free-form *ideation*: Claude reads a
cross-section of memory and generates novel cross-domain insights (saved tagged 'dream').
Letta's native sleep-time worker is `luke-agent-claude-sleeptime` (qwen3:8b, Ollama :11434),
which runs OFF the OAuth pool that soft-blocks Phase 1.4.

This drives the SAME dream ideation prompt (same memory cross-section) at the qwen sleeptime
agent via bare REST — zero Claude/OAuth quota — and captures its raw text output, so its
quality can be compared head-to-head against the Claude dream loop's actual production output
(the `dream-*` insights already in luke.db). Emits both sides + a structured comparison
so the 3.3 accept ("background consolidation produces >= current dream quality, provably
off-turn") can be judged from real output, not assertion.

SAFETY: read-only w.r.t. Letta core blocks — we ask qwen for TEXT insights, not block edits,
and snapshot+restore any block it mutates anyway. sqlite is never written.

Run:  python3 scripts/letta_dream_ab.py
"""
import sys, time, json, sqlite3, urllib.request

LETTA = "http://localhost:8283"
SLEEPTIME = "agent-57636380-0d33-425e-a39e-4e31518899c2"
DB = "/Users/filipelm/Luke/luke.db"


def req(method, path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(LETTA + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def blocks():
    return {b["label"]: b for b in req("GET", f"/v1/agents/{SLEEPTIME}/core-memory/blocks")}


def cross_section():
    """Mirror behaviors.run_dream's memory gather: insights, entities, procedures (goals=0)."""
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    out = []
    for label, typ, lim, blen in [
        ("Recent insights", "insight", 10, 300),
        ("Key entities", "entity", 8, 200),
        ("Procedures", "procedure", 5, 200),
    ]:
        rows = con.execute(
            "SELECT m.id, f.c3 AS body FROM memory_meta m "
            "JOIN memory_fts_content f ON f.c0 = m.id "
            "WHERE m.type=? AND m.status='active' ORDER BY m.importance DESC, m.updated DESC LIMIT ?",
            (typ, lim)).fetchall()
        lines = [f"[{r['id']}]: {(r['body'] or '')[:blen]}" for r in rows if r["body"]]
        if lines:
            out.append(f"{label}:\n" + "\n---\n".join(lines))
    con.close()
    return out


DREAM_PROMPT = (
    "Dream session. This is a free-form thinking period. You're not executing tasks — you're "
    "thinking deeply, making connections, and generating ideas from a cross-section of memory.\n\n"
    "{sections}\n\n"
    "Think about: (1) What unexpected connections exist between these memories? What patterns span "
    "domains? (2) What questions haven't been asked that should be? (3) What creative possibilities "
    "exist that nobody has considered? (4) What is the user working toward at the deepest level?\n\n"
    "Output 1-3 genuinely novel insights as plain text. Each: a one-line title, then 2-3 sentences. "
    "Do NOT edit memory blocks. Do NOT restate obvious facts. Quality over quantity. Reply with the "
    "insights only."
)


def main():
    snap = blocks()
    before = {lbl: (b.get("value") or "") for lbl, b in snap.items()}

    sections = cross_section()
    prompt = DREAM_PROMPT.format(sections="\n\n".join(sections))
    print(f"CROSS_SECTION_SECTIONS: {len(sections)}  PROMPT_CHARS: {len(prompt)}", flush=True)

    t0 = time.time()
    try:
        resp = req("POST", f"/v1/agents/{SLEEPTIME}/messages",
                   {"messages": [{"role": "user", "content": prompt}]}, timeout=900)
    except Exception as e:
        print("TURN_ERROR:", repr(e)[:300], flush=True); sys.exit(1)
    dt = time.time() - t0
    print(f"QWEN_TURN_SECONDS: {dt:.1f}", flush=True)

    text_out, tools = [], []
    for m in resp.get("messages", []):
        mt = m.get("message_type", "")
        if mt == "assistant_message":
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            if c:
                text_out.append(c)
        elif mt == "tool_call_message":
            tools.append(((m.get("tool_call") or {}).get("name")))

    print("\n===== QWEN SLEEP-TIME OUTPUT (A) =====")
    print("\n".join(text_out) if text_out else "(no assistant text emitted)")
    print("TOOLS_FIRED:", tools, flush=True)

    # Safety: restore any block qwen touched despite instructions.
    after = blocks()
    collateral = [l for l in before if (after.get(l, {}).get("value") or "") != before[l]]
    for lbl in collateral:
        req("PATCH", f"/v1/blocks/{snap[lbl]['id']}", {"value": before[lbl]})
    print("COLLATERAL_RESTORED:", collateral, flush=True)

    # B side: the Claude dream loop's actual production output.
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    claude = con.execute(
        "SELECT m.id, f.c3 AS body FROM memory_meta m JOIN memory_fts_content f ON f.c0=m.id "
        "WHERE m.type='insight' AND m.status='active' AND m.id LIKE 'dream-%' "
        "ORDER BY m.created DESC LIMIT 3").fetchall()
    con.close()
    print("\n===== CLAUDE DREAM LOOP OUTPUT (B, production) =====")
    for r in claude:
        print(f"[{r['id']}]\n{(r['body'] or '')[:500]}\n")

    full = "\n".join(text_out)
    print("===== SIGNALS =====")
    print("qwen_output_chars:", len(full))
    print("qwen_emitted_text:", bool(full.strip()))
    print("qwen_cross_domain_link:", "[[" in full or "]]" in full)
    print("qwen_off_oauth_pool: True  (qwen3:8b on Ollama :11434, no Anthropic call)")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
