#!/usr/bin/env python3
"""Idempotent delta-sync: push memories created/updated since last watermark into the
titled Letta archive, so backend=letta (shadow-run) semantic recall stays current.

The archive is a point-in-time snapshot; live memory grows daily. Without this, any memory
newer than the snapshot is invisible to the semantic-recall path (memory.py:755). This is a
STOPGAP until Phase 2.2c/4 write-through (remember() writes to Letta live). Watermark-based
so repeated cron runs don't duplicate. REST-only. Reversible; main untouched.

Watermark seeds to the original snapshot date on first run.
"""
import sys, sqlite3, json, os, urllib.request
from datetime import datetime, timezone
sys.path.insert(0, "scripts")
from letta_import import build_records

LIVE_DB = "/Users/filipelm/Luke/luke.db"
TITLED = "archive-7654f6f5-542a-47b6-bdb9-3542f1cb9eca"
LETTA = "http://localhost:8283"
WM = "/Users/filipelm/Code/luke/.letta_sync_watermark"
SEED = "2026-07-28T00:00:00"

def watermark():
    if os.path.exists(WM):
        return open(WM).read().strip()
    return SEED

def create_passage(text, metadata):
    body = json.dumps({"text": text, "metadata": metadata}).encode()
    req = urllib.request.Request(f"{LETTA}/v1/archives/{TITLED}/passages",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

wm = watermark()
run_start = datetime.now(timezone.utc).isoformat()
live = sqlite3.connect(LIVE_DB)
# new OR updated since watermark; content-bearing active only
delta_ids = {r[0] for r in live.execute(
    "SELECT id FROM memory_meta WHERE (created>? OR updated>?) AND status='active'",
    (wm, wm)).fetchall()}
print(f"[{run_start}] watermark={wm}  candidate ids={len(delta_ids)}")

recs, _ = build_records(LIVE_DB)
delta = [r for r in recs
         if r["metadata"]["luke_id"] in delta_ids and not r["metadata"]["is_tombstone"]]
ok = fail = 0
for r in delta:
    m = r["metadata"]; surface = f"{(m.get('title') or '')} {r['text']}".strip()
    try:
        create_passage(surface, m); ok += 1
    except Exception as e:
        fail += 1; print(f"FAIL {m.get('luke_id')}: {str(e)[:120]}")
print(f"DELTA SYNC ok={ok} fail={fail}")
# advance watermark only if no failures (so failed rows retry next run)
if fail == 0:
    open(WM, "w").write(run_start)
    print(f"watermark advanced -> {run_start}")
else:
    print("watermark NOT advanced (failures present; will retry next run)")
