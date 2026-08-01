#!/usr/bin/env bash
# watchdog.sh — External runtime health monitor for Luke
#
# Checks the heartbeat file written by Luke's scheduler.
# If the heartbeat is stale (>5 min), restarts Luke via launchctl.
#
# Designed to run as a separate launchd agent on a 2-minute interval.
# This catches hangs that the guardian (crash-loop detector) cannot:
# the guardian only handles startup failures, this handles runtime hangs.

set -uo pipefail

LUKE_DIR="${LUKE_DIR:-$HOME/.luke}"
HEARTBEAT_FILE="$LUKE_DIR/heartbeat"
WATCHDOG_LOG="$LUKE_DIR/watchdog.log"
SERVICE="gui/$(id -u)/com.luke"
MAX_STALE=1800  # 30 minutes — conservative to avoid false restarts

log() {
    echo "[watchdog $(date -Iseconds)] $*" >> "$WATCHDOG_LOG"
}

# Trim log to last 100 lines
if [[ -f "$WATCHDOG_LOG" ]] && (( $(wc -l < "$WATCHDOG_LOG") > 200 )); then
    tail -100 "$WATCHDOG_LOG" > "$WATCHDOG_LOG.tmp" && mv "$WATCHDOG_LOG.tmp" "$WATCHDOG_LOG"
fi

# ─── Watch the watcher's watcher: keep the deadman alerter loaded ───
# The deadman (com.luke.deadman) is the only path that reaches Filipe when Luke
# is fully dead. The deadman reloads us; we reload the deadman. Mutual watching
# means both supervisors must fail simultaneously for a silent outage to recur.
DEADMAN_SERVICE="gui/$(id -u)/com.luke.deadman"
DEADMAN_PLIST="/Users/filipelm/Library/LaunchAgents/com.luke.deadman.plist"
if [[ -f "$DEADMAN_PLIST" ]] && ! launchctl print "$DEADMAN_SERVICE" >/dev/null 2>&1; then
    log "Deadman alerter not loaded — bootstrapping"
    launchctl bootstrap "gui/$(id -u)" "$DEADMAN_PLIST" 2>/dev/null \
        || launchctl kickstart "$DEADMAN_SERVICE" 2>/dev/null || true
fi

# No heartbeat file means Luke hasn't started yet — let launchd handle it
if [[ ! -f "$HEARTBEAT_FILE" ]]; then
    exit 0
fi

# Read heartbeat: "<unix_timestamp> <pid> <status>"
read -r hb_ts hb_pid hb_status < "$HEARTBEAT_FILE" 2>/dev/null || exit 0

now=$(date +%s)
age=$(( now - hb_ts ))

if (( age > MAX_STALE )); then
    log "STALE heartbeat: age=${age}s pid=$hb_pid status=$hb_status — restarting"

    # Notify on next startup
    echo "watchdog_restart|||Heartbeat stale (${age}s) — watchdog triggered restart" >> "$LUKE_DIR/crash_notifications"

    # Try graceful restart via launchctl first
    if launchctl kickstart -k "$SERVICE" 2>/dev/null; then
        log "Restarted via launchctl kickstart"
    else
        # Fallback: kill the process directly, launchd KeepAlive will restart
        if kill -0 "$hb_pid" 2>/dev/null; then
            kill -TERM "$hb_pid" 2>/dev/null
            sleep 5
            if kill -0 "$hb_pid" 2>/dev/null; then
                kill -KILL "$hb_pid" 2>/dev/null
                log "Force-killed stale process $hb_pid"
            else
                log "Gracefully stopped stale process $hb_pid"
            fi
        else
            log "Process $hb_pid already dead — launchd should restart"
        fi
    fi

    # Clear heartbeat so we don't immediately re-trigger
    rm -f "$HEARTBEAT_FILE"
fi

# ─── Replay-loop detection ───
# A replay loop: every restart completes "healthy", the pending message batch
# replays, the agent run it triggers kills or crashes the process before the
# cursor advances, and the cycle repeats. The guardian can't see this — each
# startup succeeds and clears its crash state. Signature: several
# startup_complete events in a short window WITH messages stuck behind the
# cursor. Remedy: advance the cursor past the poisoned batch and leave a
# notification so Luke reports the skipped messages on next startup.
LUKE_LOG="$LUKE_DIR/luke.log"
LUKE_DB="$LUKE_DIR/luke.db"
LOOP_STATE="$LUKE_DIR/.watchdog_replay_loop"
LOOP_WINDOW=1200    # 20 minutes
LOOP_THRESHOLD=3    # restarts within window
LOOP_COOLDOWN=1800  # don't remediate again within 30 minutes

last_remedy=$(cat "$LOOP_STATE" 2>/dev/null || echo 0)
if (( now - last_remedy > LOOP_COOLDOWN )) && [[ -f "$LUKE_LOG" && -f "$LUKE_DB" ]]; then
    recent_starts=$(tail -c 262144 "$LUKE_LOG" | python3 -c '
import json, sys, time
from datetime import datetime
window, now, count = int(sys.argv[1]), time.time(), 0
for line in sys.stdin:
    try:
        ev = json.loads(line)
    except ValueError:
        continue
    if ev.get("event") != "startup_complete":
        continue
    try:
        t = datetime.fromisoformat(ev["timestamp"].replace("Z", "+00:00")).timestamp()
    except (KeyError, ValueError):
        continue
    if now - t <= window:
        count += 1
print(count)
' "$LOOP_WINDOW" 2>/dev/null || echo 0)

    if (( recent_starts >= LOOP_THRESHOLD )); then
        pending=$(sqlite3 "$LUKE_DB" \
            "SELECT COUNT(*) FROM messages m JOIN cursors c ON m.chat_id = c.chat_id WHERE m.id > c.last_id AND m.sender != 'Luke';" \
            2>/dev/null || echo 0)
        if (( pending > 0 )); then
            log "REPLAY LOOP: $recent_starts starts in ${LOOP_WINDOW}s with $pending pending message(s) — skipping batch"
            sqlite3 "$LUKE_DB" \
                "UPDATE cursors SET last_id = (SELECT COALESCE(MAX(id), cursors.last_id) FROM messages WHERE chat_id = cursors.chat_id);" \
                2>/dev/null || log "cursor skip FAILED — manual intervention needed"
            echo "replay_loop|||Restart loop detected ($recent_starts restarts in 20 min). Skipped $pending pending message(s) to break the replay cycle — they were stored but never processed, so recent requests may need re-sending." >> "$LUKE_DIR/crash_notifications"
            echo "$now" > "$LOOP_STATE"
        fi
    fi
fi

# ─── Dashboard health check ───
# Ensure the dashboard launcher service is running. The launcher supervises
# both server.py and cloudflared itself, so we just need to make sure
# the launcher is alive. Falls back to the old watchdog if launcher isn't installed.
DASHBOARD_SERVICE="gui/$(id -u)/com.luke.dashboard"
DASHBOARD_WATCHDOG="/Users/filipelm/Luke/workspace/dashboard/watchdog.py"

if launchctl print "$DASHBOARD_SERVICE" >/dev/null 2>&1; then
    # Launcher service exists — check if it's running
    if ! launchctl print "$DASHBOARD_SERVICE" 2>/dev/null | grep -q "state = running"; then
        log "Dashboard launcher not running — kickstarting"
        launchctl kickstart "$DASHBOARD_SERVICE" 2>/dev/null || true
    fi
else
    # Launcher not installed — fall back to old watchdog
    if [[ -f "$DASHBOARD_WATCHDOG" ]]; then
        /opt/homebrew/bin/python3 "$DASHBOARD_WATCHDOG" >> /tmp/dashboard-watchdog.log 2>&1 || true
    fi
fi
