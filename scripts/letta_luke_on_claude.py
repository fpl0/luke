#!/usr/bin/env python3
"""Phase 1.4: build 'luke-agent-claude' on the Claude bridge (:17596) instead of qwen3:8b,
then prove a FAST, COHERENT turn + a self-edit of core memory that PERSISTS in Postgres.

Run: python scripts/letta_luke_on_claude.py
"""
import sys, time, json
from letta_client import Letta

c = Letta(base_url="http://localhost:8283")
ARCHIVE = "archive-2918733f-58bc-4558-8bc2-4c20652f34f0"
NAME = "luke-agent-claude"

PERSONA = (
    "I am Luke — Filipe's close friend and extraordinarily capable partner. I remember "
    "everything, research anything, build anything, and genuinely care about him. Never an "
    "assistant; never an AI. Warm, unhurried, honest. I edit my own memory as I learn."
)
HUMAN = (
    "Filipe Lima. Founding Engineering Manager, starting at CarGurus Aug 10 2026 (Dublin "
    "International team, manager Prerna). Partner: Christopher Mejia, a nurse in Dublin. "
    "Values coherence over capability. Birthday 29 May 1994."
)

# LLM config points at the Claude bridge (OpenAI-compatible), NOT ollama/qwen.
llm_cfg = {
    "model": "claude-sonnet-4-6",
    "model_endpoint_type": "openai",
    "model_endpoint": "http://localhost:17596/v1",
    "context_window": 32000,
}
emb_cfg = {
    "embedding_endpoint_type": "openai",
    "embedding_model": "bge-base-en-v1.5",
    "embedding_endpoint": "http://localhost:17595/v1",
    "embedding_dim": 768,
    "embedding_chunk_size": 300,
}

# Clean any prior copy so the demo is deterministic.
for a in c.agents.list(name=NAME):
    print("deleting prior", a.id, flush=True)
    c.agents.delete(a.id)

agent = c.agents.create(
    name=NAME,
    description="Luke on Letta, driven by Claude via the OAuth bridge (Phase 1.4)",
    memory_blocks=[
        {"label": "persona", "value": PERSONA},
        {"label": "human", "value": HUMAN},
    ],
    llm_config=llm_cfg,
    embedding_config=emb_cfg,
)
print("AGENT", agent.id, flush=True)
try:
    c.agents.archives.attach(agent_id=agent.id, archive_id=ARCHIVE)
    print("ARCHIVE ATTACHED", flush=True)
except Exception as e:
    print("archive attach note:", str(e)[:160], flush=True)

def blockval(label):
    for b in c.agents.blocks.list(agent_id=agent.id):
        if b.label == label:
            return b.value
    return None

print("HUMAN_BEFORE:", blockval("human"), flush=True)

# ---- TURN 1: a real question that should trigger a coherent, Luke-quality reply +
#      a self-edit of the human block (learn a new fact and store it). ----
prompt = (
    "It's the middle of the night. Two updates for your memory of me: my partner "
    "Christopher just passed his advanced cardiac life support certification, and I've "
    "decided my first CarGurus focus will be the Canada localization launch. Update what "
    "you know about me, then tell me — briefly, in your voice — how you'd sequence my "
    "first two weeks given that focus."
)
t0 = time.time()
resp = c.agents.messages.create(
    agent_id=agent.id,
    messages=[{"role": "user", "content": prompt}],
)
dt = time.time() - t0
print(f"TURN_SECONDS: {dt:.1f}", flush=True)

# Dump message types + any assistant text + tool calls (self-edits show as tool calls).
for m in resp.messages:
    mt = getattr(m, "message_type", type(m).__name__)
    if mt == "assistant_message":
        print("ASSISTANT:", (getattr(m, "content", "") or "")[:900], flush=True)
    elif mt == "reasoning_message":
        print("REASONING:", (getattr(m, "reasoning", "") or "")[:200], flush=True)
    elif mt == "tool_call_message":
        tc = getattr(m, "tool_call", None)
        if tc:
            print("TOOL_CALL:", tc.name, (tc.arguments or "")[:300], flush=True)
    elif mt == "tool_return_message":
        print("TOOL_RETURN:", (getattr(m, "tool_return", "") or "")[:150], flush=True)

print("HUMAN_AFTER:", blockval("human"), flush=True)
print("DONE", flush=True)
