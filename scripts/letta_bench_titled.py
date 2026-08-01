#!/usr/bin/env python3
"""Phase 2.3 benchmark (archives already loaded). Corrected passage-unwrap parse.

Ground truth = sqlite-vec semantic top-1 (query vs each memory's title+content surface via
the same embed server). Measure how often each Letta archive top-1 matches it.

Nightly-gate hardening (2026-08-01, gate 5.4):
  * Ground-truth embeddings are CACHED (keyed by backup path+mtime+count) — the 900-surface
    embed pass cost ~200s of every run; cached runs finish in seconds.
  * The gt snapshot is frozen at the Jul-28 backup, so an archive top-1 that is NEWER than
    the snapshot cannot match gt by construction. Those are counted separately as
    "fresh-misses" (F) — a currency signal, not a retrieval failure. The gate number to
    trend is titled hits over the COMPARABLE queries.
"""
import os, sys, json, time, math, urllib.request
sys.path.insert(0, "scripts")
from letta_import import build_records

DB = "/Users/filipelm/Luke/backups/pre-letta-20260728/luke.db"
EMBED = "http://localhost:17595/v1/embeddings"
LETTA = "http://localhost:8283"
CONTENT_ONLY = "archive-2918733f-58bc-4558-8bc2-4c20652f34f0"   # luke-memories-bge (content only)
TITLED = "archive-7654f6f5-542a-47b6-bdb9-3542f1cb9eca"          # luke-memories-bge-titled

def embed(text):
    body = json.dumps({"model": "bge-base-en-v1.5", "input": text}).encode()
    req = urllib.request.Request(EMBED, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"][0]["embedding"]

def cos(a, b):
    d = sum(x*y for x, y in zip(a, b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d / (na*nb + 1e-9)

def letta_top1(archive_id, query):
    # Measure the PRODUCTION surface: skip tombstone/archived passages exactly like
    # letta_adapter does — raw archive top-1 was scoring tombstones production never sees.
    body = json.dumps({"query": query, "archive_id": archive_id, "limit": 15}).encode()
    req = urllib.request.Request(f"{LETTA}/v1/passages/search", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    if isinstance(data, dict): data = data.get("results") or data.get("passages") or []
    for item in data or []:
        obj = item.get("passage", item)   # <-- unwrap, matches letta_adapter.py
        meta = obj.get("metadata") or {}
        if meta.get("is_tombstone") or meta.get("status") == "archived":
            continue
        if meta.get("luke_id"):
            return meta["luke_id"]
    return None

recs, _ = build_records(DB)
cb = [r for r in recs if not r["metadata"]["is_tombstone"]]
surfaces = [(r["metadata"]["luke_id"], f"{(r['metadata'].get('title') or '')} {r['text']}".strip()) for r in cb]

CACHE = "/Users/filipelm/Luke/workspace/letta_bench_gt_cache.json"
cache_key = f"{DB}:{os.path.getmtime(DB)}:{len(surfaces)}"
gt_vecs = {}
try:
    cached = json.load(open(CACHE))
    if cached.get("key") == cache_key:
        gt_vecs = cached["vecs"]
        print(f"[{time.strftime('%H:%M:%S')}] gt embeddings from cache ({len(gt_vecs)})", flush=True)
except Exception:
    pass
if not gt_vecs:
    print(f"[{time.strftime('%H:%M:%S')}] embedding {len(surfaces)} ground-truth surfaces...", flush=True)
    for j, (lid, s) in enumerate(surfaces):
        try: gt_vecs[lid] = embed(s)
        except Exception as e: print(f"gt fail {lid}: {str(e)[:50]}")
        if (j+1) % 300 == 0: print(f"  gt {j+1}/{len(surfaces)}", flush=True)
    try:
        json.dump({"key": cache_key, "vecs": gt_vecs}, open(CACHE, "w"))
    except Exception as e:
        print(f"cache write failed: {str(e)[:80]}")

def sqlite_gt(qv):
    best, bs = None, -2
    for lid, v in gt_vecs.items():
        s = cos(qv, v)
        if s > bs: bs, best = s, lid
    return best

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

print(f"[{time.strftime('%H:%M:%S')}] === BENCHMARK ===", flush=True)
th_hits = ch_hits = fresh = 0; rows = []
for q in QUERIES:
    qv = embed(q); gt = sqlite_gt(qv)
    t1 = letta_top1(TITLED, q); c1 = letta_top1(CONTENT_ONLY, q)
    th = (t1 == gt); ch = (c1 == gt)
    # Archive returned a memory the frozen gt snapshot has never seen: cannot match gt by
    # construction. Count as fresh (currency), not a retrieval miss.
    fr = (not th) and t1 is not None and t1 not in gt_vecs
    th_hits += th; ch_hits += ch; fresh += fr
    rows.append({"q": q, "gt": gt, "titled": t1, "content": c1, "th": th, "ch": ch, "fresh": fr})
    mark = "T" if th else ("F" if fr else ".")
    print(f"{mark}{'C' if ch else '.'} gt={gt} | titled={t1} | content={c1} | {q[:38]}", flush=True)
n = len(QUERIES)
comparable = n - fresh
print(f"[{time.strftime('%H:%M:%S')}] RESULT titled={th_hits}/{comparable} comparable "
      f"({fresh} fresh-miss vs Jul-28 snapshot)  content-only={ch_hits}/{n}", flush=True)
os.makedirs("/tmp/claude", exist_ok=True)
json.dump({"titled": TITLED, "titled_hits": th_hits, "content_hits": ch_hits, "n": n,
           "fresh": fresh, "comparable": comparable, "rows": rows},
          open("/tmp/claude/letta_titled_bench.json", "w"), indent=2)
