#!/usr/bin/env python3
"""Phase 2.3 — re-embed with title+content parity, then benchmark against content-only.

Luke's canonical store path (memory.py:611 index_memory) embeds ``f"{title} {content}"``.
The original bge load (letta_import.py:105) embedded ``content`` only. This closes the gap:
creates archive ``luke-memories-bge-titled`` with the same title+content surface Luke uses.

Benchmark (non-circular): ground truth = sqlite-vec semantic top-1 for each query, computed
by embedding the query and every memory's stored ``title+content`` surface via the SAME embed
server Luke uses, then argmax cosine. We then measure how often each Letta archive's top-1
matches that ground truth. Target: titled >= content-only, ideally >=5/6 realistic queries.
"""
import sys, json, time, math, urllib.request
sys.path.insert(0, "scripts")
from letta_import import build_records
from letta_client import Letta

DB = "/Users/filipelm/Luke/backups/pre-letta-20260728/luke.db"
EMBED = "http://localhost:17595/v1/embeddings"
LETTA = "http://localhost:8283"
CONTENT_ONLY_ARCHIVE = "archive-2918733f-58bc-4558-8bc2-4c20652f34f0"  # luke-memories-bge
PROG = "/tmp/claude/letta_titled_progress.txt"

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(PROG, "a") as f: f.write(line + "\n")

def embed(text):
    body = json.dumps({"model": "bge-base-en-v1.5", "input": text}).encode()
    req = urllib.request.Request(EMBED, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"][0]["embedding"]

def cos(a, b):
    d = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d / (na*nb + 1e-9)

def letta_top1(c, archive_id, query, limit=1):
    url = f"{LETTA}/v1/passages/search"
    body = json.dumps({"query": query, "archive_id": archive_id, "limit": limit}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    if isinstance(data, dict): data = data.get("results") or data.get("passages") or []
    if not data: return None
    return (data[0].get("metadata") or {}).get("luke_id")

open(PROG, "w").close()
c = Letta(base_url=LETTA)

# ---- build title+content records (content-bearing only) ----
recs, _ = build_records(DB)
cb = [r for r in recs if not r["metadata"]["is_tombstone"]]
log(f"content-bearing records: {len(cb)}")

# ---- create the titled archive ----
cfg = {"embedding_endpoint_type": "openai", "embedding_model": "bge-base-en-v1.5",
       "embedding_endpoint": EMBED.rsplit('/embeddings', 1)[0], "embedding_dim": 768,
       "embedding_chunk_size": 300, "batch_size": 32}
arch = c.archives.create(name="luke-memories-bge-titled", embedding_config=cfg)
TITLED = arch.id
log(f"TITLED ARCHIVE {TITLED}")

ok = fail = 0; t0 = time.time()
for i, r in enumerate(cb):
    m = r["metadata"]; title = m.get("title") or ""
    surface = f"{title} {r['text']}".strip()  # matches memory.py:611
    try:
        c.archives.passages.create(archive_id=TITLED, text=surface, metadata=m)
        ok += 1
    except Exception as e:
        fail += 1
        if fail <= 3: log(f"FAIL {m.get('luke_id')}: {str(e)[:100]}")
    if (i+1) % 100 == 0: log(f"loaded {i+1}/{len(cb)} ({time.time()-t0:.0f}s)")
log(f"LOAD DONE ok={ok} fail={fail} in {time.time()-t0:.0f}s")

# ---- ground-truth vectors: embed each memory's title+content surface ----
log("embedding memory surfaces for ground truth...")
surfaces = []
for r in cb:
    m = r["metadata"]; title = m.get("title") or ""
    surfaces.append((m["luke_id"], f"{title} {r['text']}".strip()))
gt_vecs = {}
for j, (lid, s) in enumerate(surfaces):
    try: gt_vecs[lid] = embed(s)
    except Exception as e: log(f"gt embed fail {lid}: {str(e)[:60]}")
    if (j+1) % 200 == 0: log(f"gt embedded {j+1}/{len(surfaces)}")
log(f"ground-truth vectors: {len(gt_vecs)}")

QUERIES = [
    "when does Filipe start his new job at CarGurus",
    "what does Christopher do for work",
    "Filipe's B1 US visa interview appointment details",
    "when is Filipe's birthday",
    "should Luke migrate its memory to postgres",
    "who is Prerna and what is her role",
    "Filipe's stress leave from Clio",
    "Luke should never send emails on Filipe's behalf",
    "the Letta full power build plan",
    "how does Luke make a PDF",
    "Filipe prefers coherence over capability",
    "Mark Anderson situation at Clio",
]

def sqlite_gt(qvec):
    best, bs = None, -2
    for lid, v in gt_vecs.items():
        s = cos(qvec, v)
        if s > bs: bs, best = s, lid
    return best

log("=== BENCHMARK ===")
titled_hits = content_hits = 0; rows = []
for q in QUERIES:
    qv = embed(q)
    gt = sqlite_gt(qv)
    t1 = letta_top1(c, TITLED, q)
    c1 = letta_top1(c, CONTENT_ONLY_ARCHIVE, q)
    th = (t1 == gt); ch = (c1 == gt)
    titled_hits += th; content_hits += ch
    rows.append({"q": q, "gt": gt, "titled": t1, "content": c1, "th": th, "ch": ch})
    log(f"{'T' if th else '.'}{'C' if ch else '.'} | gt={gt} | titled={t1} | content={c1} | {q[:40]}")

n = len(QUERIES)
log(f"RESULT titled={titled_hits}/{n}  content-only={content_hits}/{n}")
json.dump({"titled_archive": TITLED, "titled_hits": titled_hits,
           "content_hits": content_hits, "n": n, "rows": rows},
          open("/tmp/claude/letta_titled_bench.json", "w"), indent=2)
log("wrote /tmp/claude/letta_titled_bench.json")
