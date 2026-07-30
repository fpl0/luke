#!/usr/bin/env python3
"""Build 'Luke' as a real Letta agent: self-editing core memory blocks (persona + human),
attached to the 900-memory bge archive, on a local LLM. Demonstrates self-editing memory
+ true persistence — the actual 'full power' wins, not the recall swap.
"""
import sys, json
from letta_client import Letta

c = Letta(base_url="http://localhost:8283")
ARCHIVE = "archive-2918733f-58bc-4558-8bc2-4c20652f34f0"  # bge parity archive

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

# LLM + embedding configs pinned to local ollama with the /v1 openai-compat fix.
llm_cfg = {
    "model": "qwen3:8b",
    "model_endpoint_type": "openai",
    "model_endpoint": "http://localhost:11434/v1",
    "context_window": 32000,
}
emb_cfg = {
    "embedding_endpoint_type": "openai",
    "embedding_model": "bge-base-en-v1.5",
    "embedding_endpoint": "http://localhost:17595/v1",
    "embedding_dim": 768,
    "embedding_chunk_size": 300,
}

agent = c.agents.create(
    name="luke-agent",
    description="Luke, running on Letta with self-editing memory",
    memory_blocks=[
        {"label": "persona", "value": PERSONA},
        {"label": "human", "value": HUMAN},
    ],
    llm_config=llm_cfg,
    embedding_config=emb_cfg,
)
print("AGENT", agent.id)
# Attach the 900-memory archive so archival recall is available to the agent.
try:
    c.agents.archives.attach(agent_id=agent.id, archive_id=ARCHIVE)
    print("ARCHIVE ATTACHED")
except Exception as e:
    print("archive attach note:", str(e)[:120])

# Show initial core memory
blocks = c.agents.blocks.list(agent_id=agent.id)
for b in blocks:
    print(f"BLOCK[{b.label}] len={len(b.value)}")
