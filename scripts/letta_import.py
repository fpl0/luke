#!/usr/bin/env python3
"""Letta importer for Luke's memory bank.

Reads from a BACKUP copy of luke.db (never the live db) and shapes every
memory_meta row into a Letta archival passage payload, carrying full
metadata so the graph and provenance survive the migration.

GROUND TRUTH discovered 2026-07-29 (corrects LETTA_MIGRATION_NOTES importer plan):
  memory_meta has 1535 rows: 763 active, 771 archived, 1 paused.
  Content lives in the memory_fts virtual table — but ONLY for active rows.
  Archiving a memory removes it from FTS, so archived text is NOT joinable there.
  Fallback: latest memory_history.new_content recovers 136 archived rows.
  => 899 rows have recoverable content (763 active + 136 archived).
     635 archived rows are metadata-only "tombstones" — no text exists anywhere.
  A naive "assert imported == 1535 with content" is therefore IMPOSSIBLE.
  This importer reconciles explicitly instead of silently dropping the 635.

Usage:
  python scripts/letta_import.py --dry-run           # reconcile + write manifest, no network
  python scripts/letta_import.py --load --base-url http://localhost:8283 [--archive-id ID]

The dry-run is the verification gate: it proves we can read and shape every
row and reports exactly what carries content vs. what is a tombstone.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

DEFAULT_BACKUP = "/Users/filipelm/Luke/backups/pre-letta-20260728/luke.db"
# Spot-check the two highest-value entities + the one genuinely ACTIVE goal.
# (goal-msc-cognitive-science is archived in the db despite a stale "paused" label
#  in injected context; its substance survives across active insight-msc-* rows.)
SPOT_CHECK_IDS = ["person-filipe", "user-preferences", "goal-voicebox-luke-voice"]


def latest_history_content(cur, mem_id):
    """Recover content for a memory not in FTS via its newest history snapshot."""
    row = cur.execute(
        """SELECT new_content FROM memory_history
           WHERE mem_id = ? AND new_content IS NOT NULL AND length(new_content) > 0
           ORDER BY timestamp DESC LIMIT 1""",
        (mem_id,),
    ).fetchone()
    return row[0] if row else None


def build_records(db_path):
    """Return (records, stats). Each record is a Letta-ready passage payload."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    hist_cur = con.cursor()  # separate cursor so history lookups don't reset the outer loop

    # Content map from FTS (active rows only).
    fts = {
        r[0]: {"title": r[1], "content": r[2], "tags": r[3]}
        for r in cur.execute("SELECT id, title, content, tags FROM memory_fts")
    }

    meta_rows = cur.execute(
        """SELECT id, type, created, updated, importance, status,
                  tags_json, links_json, access_count, useful_count, taxonomy
             FROM memory_meta ORDER BY id"""
    ).fetchall()

    records = []
    stats = {
        "total": 0, "from_fts": 0, "from_history": 0, "tombstone": 0,
        "by_status": {},
    }

    for row in meta_rows:
        (mid, mtype, created, updated, importance, status,
         tags_json, links_json, access_count, useful_count, taxonomy) = row
        stats["total"] += 1
        stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

        title, content, source = None, None, None
        if mid in fts:
            title = fts[mid]["title"]
            content = fts[mid]["content"]
            source = "fts"
            stats["from_fts"] += 1
        else:
            recovered = latest_history_content(hist_cur, mid)
            if recovered:
                content = recovered
                source = "history"
                stats["from_history"] += 1
            else:
                source = "tombstone"
                stats["tombstone"] += 1

        try:
            tags = json.loads(tags_json) if tags_json else []
        except json.JSONDecodeError:
            tags = []
        try:
            links = json.loads(links_json) if links_json else []
        except json.JSONDecodeError:
            links = []

        # Letta passage: text is the recall surface; metadata preserves provenance.
        passage_text = content if content else f"[archived tombstone: {mid} ({mtype})]"
        records.append({
            "text": passage_text,
            "metadata": {
                "luke_id": mid,
                "type": mtype,
                "title": title,
                "status": status,
                "importance": importance,
                "tags": tags,
                "links": links,
                "created": created,
                "updated": updated,
                "access_count": access_count,
                "useful_count": useful_count,
                "taxonomy": taxonomy,
                "content_source": source,
                "is_tombstone": content is None,
            },
        })

    con.close()
    return records, stats


def dry_run(db_path, manifest_path):
    records, stats = build_records(db_path)

    # Hard invariant: every memory_meta row is accounted for, none silently lost.
    assert stats["total"] == 1535, f"expected 1535 rows, got {stats['total']}"
    assert len(records) == stats["total"], "record count != row count"
    with_content = stats["from_fts"] + stats["from_history"]
    assert with_content + stats["tombstone"] == stats["total"], "reconciliation gap"

    by_id = {r["metadata"]["luke_id"]: r for r in records}
    spot = {}
    for sid in SPOT_CHECK_IDS:
        rec = by_id.get(sid)
        spot[sid] = {
            "found": rec is not None,
            "has_content": bool(rec and not rec["metadata"]["is_tombstone"]),
            "chars": len(rec["text"]) if rec else 0,
            "source": rec["metadata"]["content_source"] if rec else None,
        }

    manifest = {
        "source_db": db_path,
        "reconciliation": stats,
        "with_content": with_content,
        "tombstones": stats["tombstone"],
        "spot_check": spot,
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2))

    print("=== Letta import DRY-RUN — reconciliation ===")
    print(f"  total memory_meta rows : {stats['total']}")
    print(f"  by status              : {stats['by_status']}")
    print(f"  content from FTS       : {stats['from_fts']}")
    print(f"  content from history   : {stats['from_history']}")
    print(f"  metadata-only tombstone: {stats['tombstone']}")
    print(f"  => carries content     : {with_content} / {stats['total']}")
    print("  spot-check:")
    for sid, info in spot.items():
        print(f"    {sid}: {info}")
    print(f"  manifest written       : {manifest_path}")
    print("OK: every row accounted for, no silent drops.")
    return 0


def validate_payloads(db_path, out_path):
    """Materialize ALL passage payloads and enforce per-record invariants.

    dry_run() only proves reconciliation counts + 3 spot-checks. This is the
    stronger pre-flight: every one of the 1535 payloads must round-trip through
    JSON and satisfy the shape the batch endpoint requires, so a live --load is
    a single clean run with no per-row surprises. No network, no install.
    """
    records, stats = build_records(db_path)
    anomalies = []
    max_text = 0
    max_meta_keys = 0

    for i, rec in enumerate(records):
        mid = rec["metadata"].get("luke_id")
        # 1. Must serialize (catches non-ASCII / bad unicode / nested weirdness).
        try:
            blob = json.dumps(rec, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            anomalies.append(f"{mid}: not JSON-serializable ({e})")
            continue
        # 2. Round-trips back to an identical dict (no lossy coercion).
        if json.loads(blob) != rec:
            anomalies.append(f"{mid}: JSON round-trip mismatch")
        # 3. Required batch fields present and correctly typed.
        if not isinstance(rec.get("text"), str) or rec["text"] == "":
            anomalies.append(f"{mid}: empty/non-string text")
        if not isinstance(rec.get("metadata"), dict):
            anomalies.append(f"{mid}: metadata not an object")
        if not mid:
            anomalies.append(f"row {i}: missing luke_id in metadata")
        # 4. A non-tombstone claiming content must actually carry text.
        meta = rec["metadata"]
        if meta.get("is_tombstone") is False and len(rec["text"]) < 1:
            anomalies.append(f"{mid}: marked content-bearing but text empty")
        # 5. is_tombstone flag must agree with content_source.
        expect_tomb = meta.get("content_source") == "tombstone"
        if bool(meta.get("is_tombstone")) != expect_tomb:
            anomalies.append(f"{mid}: is_tombstone disagrees with content_source")
        max_text = max(max_text, len(rec["text"]))
        max_meta_keys = max(max_meta_keys, len(meta))

    # Write the full batch as NDJSON — the exact bytes a live load would send,
    # inspectable before any server exists.
    Path(out_path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records))

    print("=== Letta import PAYLOAD VALIDATION ===")
    print(f"  records materialized   : {len(records)}")
    print(f"  content-bearing        : {stats['from_fts'] + stats['from_history']}")
    print(f"  tombstones             : {stats['tombstone']}")
    print(f"  largest passage text   : {max_text} chars")
    print(f"  max metadata keys      : {max_meta_keys}")
    print(f"  anomalies              : {len(anomalies)}")
    for a in anomalies[:20]:
        print(f"    - {a}")
    print(f"  batch NDJSON written   : {out_path}")
    if anomalies:
        print(f"FAIL: {len(anomalies)} payload anomalies — fix before live load.")
        return 1
    print("OK: all payloads well-formed and invariant-clean. Load path is safe to run on go.")
    return 0


def load(db_path, base_url, archive_id):
    try:
        from letta_client import Letta
    except ImportError:
        print("letta_client not importable — activate .letta-venv", file=sys.stderr)
        return 2
    records, stats = build_records(db_path)
    client = Letta(base_url=base_url)
    # Batched load into the archive; adapter/agent wiring is a separate step.
    print(f"Loading {len(records)} passages into {base_url} archive={archive_id} ...")
    # NOTE: actual batch call wired once a test archive exists on a running server.
    raise SystemExit("load path is intentionally gated until a test server is up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", default=DEFAULT_BACKUP)
    ap.add_argument("--manifest", default="/tmp/claude/letta_import_manifest.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate-payloads", action="store_true")
    ap.add_argument("--ndjson", default="/tmp/claude/letta_batch.ndjson")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:8283")
    ap.add_argument("--archive-id", default=None)
    args = ap.parse_args()

    if not Path(args.backup).exists():
        print(f"backup not found: {args.backup}", file=sys.stderr)
        return 2
    if args.load:
        return load(args.backup, args.base_url, args.archive_id)
    if args.validate_payloads:
        return validate_payloads(args.backup, args.ndjson)
    return dry_run(args.backup, args.manifest)


if __name__ == "__main__":
    sys.exit(main())
