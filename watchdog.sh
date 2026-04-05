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
MAX_STALE=300  # 5 minutes — heartbeat should update every 60s

log() {
    echo "[watchdog $(date -Iseconds)] $*" >> "$WATCHDOG_LOG"
}

# Trim log to last 100 lines
if [[ -f "$WATCHDOG_LOG" ]] && (( $(wc -l < "$WATCHDOG_LOG") > 200 )); then
    tail -100 "$WATCHDOG_LOG" > "$WATCHDOG_LOG.tmp" && mv "$WATCHDOG_LOG.tmp" "$WATCHDOG_LOG"
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
