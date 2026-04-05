#!/usr/bin/env bash
# guardian.sh — Crash-loop detection wrapper for Luke
#
# launchd calls this instead of `uv run luke` directly.
# Tracks rapid restarts and auto-reverts if a crash loop is detected.
#
# Flow:
#   1. Record this startup timestamp + git SHA
#   2. If N+ starts within WINDOW seconds on same SHA → crash loop
#   3. Auto-revert HEAD and reset counter
#   4. exec into `uv run luke`
#
# Luke calls _guardian_mark_healthy() after "Back online." to clear state.

set -uo pipefail

LUKE_DIR="${LUKE_DIR:-$HOME/.luke}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$LUKE_DIR/guardian_state"
ROLLBACK_LOG="$LUKE_DIR/guardian_rollbacks.log"
MAX_CRASHES=3
WINDOW=300  # 5 minutes

mkdir -p "$LUKE_DIR"

now=$(date +%s)
sha=$(cd "$REPO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Count recent crashes for this SHA
crash_count=0
if [[ -f "$STATE_FILE" ]]; then
    while IFS=' ' read -r ts commit rest; do
        [[ -z "$ts" ]] && continue
        if [[ "$commit" == "$sha" ]] && (( now - ts < WINDOW )); then
            crash_count=$(( crash_count + 1 ))
        fi
    done < "$STATE_FILE"
fi

# Record this startup
echo "$now $sha start" >> "$STATE_FILE"
crash_count=$(( crash_count + 1 ))

# Crash loop detection
if (( crash_count >= MAX_CRASHES )); then
    echo "[guardian $(date -Iseconds)] CRASH LOOP: $crash_count starts in ${WINDOW}s on $sha" >&2

    cd "$REPO_DIR" || exit 1

    # Check if we already reverted recently (prevent revert loop)
    recent_reverts=0
    if [[ -f "$ROLLBACK_LOG" ]]; then
        while IFS=' ' read -r ts rest; do
            [[ -z "$ts" ]] && continue
            if (( now - ts < 1800 )); then  # 30 min window
                recent_reverts=$(( recent_reverts + 1 ))
            fi
        done < "$ROLLBACK_LOG"
    fi

    if (( recent_reverts >= 2 )); then
        echo "[guardian $(date -Iseconds)] Too many recent rollbacks ($recent_reverts in 30min) — stopping. Manual fix needed." >&2
        # Sleep so launchd doesn't spin
        sleep 600
        exit 1
    fi

    # Try to revert
    if git revert --no-edit HEAD 2>/dev/null; then
        new_sha=$(git rev-parse --short HEAD)
        git push origin main 2>/dev/null || true
        echo "[guardian $(date -Iseconds)] Rolled back $sha → $new_sha" >&2
        echo "$now rolled_back $sha to $new_sha" >> "$ROLLBACK_LOG"
        # Reset crash state
        : > "$STATE_FILE"
    else
        echo "[guardian $(date -Iseconds)] Revert failed (conflicts?) — stopping. Manual fix needed." >&2
        echo "$now revert_failed $sha" >> "$ROLLBACK_LOG"
        sleep 600
        exit 1
    fi
fi

# Trim state file to last 20 lines (prevent growth)
if [[ -f "$STATE_FILE" ]] && (( $(wc -l < "$STATE_FILE") > 20 )); then
    tail -20 "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
fi

# Start Luke
cd "$REPO_DIR" || exit 1
exec uv run luke
