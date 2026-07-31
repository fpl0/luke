#!/usr/bin/env python3
"""Phase 2.4 — load the 635 metadata-only tombstones into the titled archive.

Context (verified from source, 2026-07-31):
  - 635 memory_meta rows are archived AND have no recoverable text anywhere
    (not in FTS, not in memory_history) — see letta_import.py reconciliation.
  - The titled archive (luke-memories-bge-titled) holds only content-bearing
    passages (900 backup + 41 live delta = 941). Tombstones were filtered out.
  - Graph traversal (_get_neighbors_batch, memory.py:1019) reads links from
    SQLITE and returns only status='active' neighbors, so tombstones never
    affect live graph recall regardless of Letta. Their value here is *store
    completeness*: a faithful Letta mirror of the full memory ledger, needed
    for Phase 4 write-through and any future direct-Letta id resolution.

What this does:
  - Builds tombstone records from the SAME Jul-28 backup the archive was built
    from (provenance stays coherent: archive == snapshot + live deltas).
  - Loads each as a placeholder passage: text "[archived tombstone: <id> (<type>)]"
    + full metadata (is_tombstone=True, status, type, links, provenance).
  - Idempotent: skips any luke_id already present in the archive.

Verify (separate step): tombstone count in archive == 635, and the active-recall
benchmark (letta_bench_titled path) still passes — placeholders must not outrank
real content.

Usage: python scripts/letta_load_tombstones.py [--dry-run]
"""
import sys, json, time, urllib.request
sys.path.insert(0, "scripts")
from letta_import import build_records
from letta_client import Letta

DB = "/Users/filipelm/Luke/backups/pre-letta-20260728/luke.db"
LETTA = "http://localhost:8283"
ARCH = "archive-7654f6f5-542a-47b6-bdb9-3542f1cb9eca"  # luke-memories-bge-titled
PG = dict(host="localhost", port=5432, user="letta", password="letta", dbname="letta")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def existing_luke_ids():
    """Distinct luke_ids already in the archive (idempotency guard). Reads Postgres directly."""
    import psycopg2
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT DISTINCT metadata_->>'luke_id' FROM archival_passages WHERE archive_id=%s",
        (ARCH,),
    )
    ids = {r[0] for r in cur.fetchall() if r[0]}
    con.close()
    return ids


def main():
    dry = "--dry-run" in sys.argv
    recs, _ = build_records(DB)
    tombs = [r for r in recs if r["metadata"]["is_tombstone"]]
    log(f"tombstone records from backup: {len(tombs)}")
    assert len(tombs) == 635, f"expected 635 tombstones, got {len(tombs)}"

    present = existing_luke_ids()
    log(f"luke_ids already in archive: {len(present)}")
    todo = [r for r in tombs if r["metadata"]["luke_id"] not in present]
    log(f"tombstones to load (not already present): {len(todo)}")

    if dry:
        # Sanity: every tombstone has the placeholder text shape and is_tombstone flag.
        bad = [r["metadata"]["luke_id"] for r in tombs
               if not r["text"].startswith("[archived tombstone:") or not r["metadata"]["is_tombstone"]]
        log(f"DRY-RUN: malformed tombstone payloads: {len(bad)}")
        if bad:
            log(f"  e.g. {bad[:5]}")
        log("DRY-RUN done — no writes.")
        return 0 if not bad else 1

    c = Letta(base_url=LETTA)
    ok = fail = 0
    t0 = time.time()
    for i, r in enumerate(todo):
        m = r["metadata"]
        try:
            c.archives.passages.create(archive_id=ARCH, text=r["text"], metadata=m)
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                log(f"FAIL {m.get('luke_id')}: {str(e)[:100]}")
        if (i + 1) % 100 == 0:
            log(f"loaded {i+1}/{len(todo)} ({time.time()-t0:.0f}s)")
    log(f"LOAD DONE ok={ok} fail={fail} in {time.time()-t0:.0f}s")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
