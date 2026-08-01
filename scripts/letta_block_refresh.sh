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
# Refresh first (sqlite -> blocks), then audit (verify fresh + log the ledger line).
"$PY" scripts/letta_pack_core_blocks.py
echo "--- audit ---"
"$PY" scripts/letta_block_drift_audit.py
