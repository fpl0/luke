#!/usr/bin/env python3
"""Re-embed Luke's 900 content-bearing passages into Letta using bge-base (parity model)."""
import sys, json, time
sys.path.insert(0, "scripts")
from letta_import import build_records
from letta_client import Letta

PROG = "/tmp/claude/letta_bge_progress.txt"
def log(m):
    with open(PROG, "a") as f: f.write(f"{m}\n")

c = Letta(base_url="http://localhost:8283")
cfg = {"embedding_endpoint_type":"openai","embedding_model":"bge-base-en-v1.5",
       "embedding_endpoint":"http://localhost:17595/v1","embedding_dim":768,
       "embedding_chunk_size":300,"batch_size":32}
arch = c.archives.create(name="luke-memories-bge", embedding_config=cfg)
log(f"ARCHIVE {arch.id}")
recs, _ = build_records("/Users/filipelm/Luke/backups/pre-letta-20260728/luke.db")
cb = [r for r in recs if r.get("text") and not r["text"].startswith("[archived tombstone")]
log(f"LOADING {len(cb)}")
ok = fail = 0; t0 = time.time()
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
    if (i+1) % 100 == 0: log(f"  {i+1}/{len(cb)} ok={ok} fail={fail} ({time.time()-t0:.0f}s)")
log(f"DONE ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s ARCHIVE={arch.id}")
