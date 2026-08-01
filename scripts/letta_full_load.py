#!/usr/bin/env python3
"""Full embedded load of Luke's content-bearing memories into Letta.
Ollama local embeddings via /v1 (corrected endpoint). Writes progress to
/tmp/claude/letta_load_progress.txt so it can run detached.
"""
import sys, json, time
sys.path.insert(0, "scripts")
from letta_import import build_records
from letta_client import Letta

PROG = "/tmp/claude/letta_load_progress.txt"
def log(m):
    with open(PROG, "a") as f: f.write(f"{m}\n")

c = Letta(base_url="http://localhost:8283")
cfg = {"embedding_endpoint_type":"openai","embedding_model":"nomic-embed-text",
       "embedding_endpoint":"http://localhost:11434/v1","embedding_dim":768,
       "embedding_chunk_size":300,"batch_size":32}
arch = c.archives.create(name="luke-memories-v2", embedding_config=cfg)
log(f"ARCHIVE {arch.id}")

recs, stats = build_records("/Users/filipelm/Luke/backups/pre-letta-20260728/luke.db")
cb = [r for r in recs if r.get("text") and not r["text"].startswith("[archived tombstone")]
log(f"LOADING {len(cb)} content-bearing passages")

ok = fail = 0
t0 = time.time()
for i, r in enumerate(cb):
    try:
        meta = r.get("metadata")
        if isinstance(meta, str):
            try: meta = json.loads(meta.replace("'", '"'))
            except Exception: meta = {"raw": meta}
        c.archives.passages.create(archive_id=arch.id, text=r["text"], metadata=meta or {})
        ok += 1
    except Exception as e:
        fail += 1
        if fail <= 5: log(f"  FAIL {i}: {str(e)[:100]}")
    if (i+1) % 50 == 0:
        log(f"  {i+1}/{len(cb)} ok={ok} fail={fail} ({time.time()-t0:.0f}s)")
log(f"DONE ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s ARCHIVE={arch.id}")
