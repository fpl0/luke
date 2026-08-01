#!/bin/bash
# Phase 5.5 — daily core-block freshness refresh + drift audit.
# Re-packs the 5 deterministic core blocks from current sqlite (so they reflect the last 24h of
# changes, per the 5.5 accept), then runs the drift audit which appends a dated verdict line to
# logs/letta_block_freshness.log (the 3-consecutive-days ledger). Pure REST+sqlite: no Claude /
# OAuth quota consumed. Owned by launchd com.luke.lettablockfresh (daily). Idempotent + reversible.
set -euo pipefail
cd /Users/filipelm/Code/luke
PY=/Users/filipelm/Code/luke/.letta-venv/bin/python
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== letta_block_refresh $STAMP ==="
# 1. OBSERVE first: measure + log the drift accumulated since yesterday's re-pack. This is the
#    evidence the 5.5 gate actually does work (a pack-then-audit alone is trivially fresh and
#    proves nothing). Never fails the run — it's a ledger annotation, not a gate.
echo "--- observe (pre-refresh drift) ---"
"$PY" scripts/letta_block_drift_audit.py --observe
# 2. Refresh (sqlite -> blocks).
"$PY" scripts/letta_pack_core_blocks.py
# 3. Audit (verify the re-pack landed fresh + log the ledger gate line the 3-day accept counts).
echo "--- audit ---"
"$PY" scripts/letta_block_drift_audit.py
