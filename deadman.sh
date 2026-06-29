#!/usr/bin/env bash
# deadman.sh — Truly external liveness alerter for Luke
#
# The guardian (crash loops) and watchdog (runtime hangs) can RESTART Luke,
# but their only notification path — the crash_notifications file — is read by
# Luke ON RESTART. When Luke is dead and CANNOT restart, Filipe is never told.
# That is exactly how the May 15 → Jun 27 six-week silence happened: the
# watchdog stopped firing and nothing reached Filipe.
#
# This script depends on NOTHING in the Luke runtime. Pure bash + curl. It runs
# as its own launchd agent (com.luke.deadman) on a short interval. If Luke's
# heartbeat stays stale past THRESHOLD — i.e. the watchdog failed to recover it —
# it messages Filipe directly via the Telegram Bot API. One alert per outage,
# plus a "back up" when the heartbeat resumes. It also makes sure the watchdog
# agent itself is loaded (watch the watcher).

set -uo pipefail

LUKE_DIR="${LUKE_DIR:-/Users/filipelm/Luke}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEARTBEAT_FILE="$LUKE_DIR/heartbeat"
STATE_FILE="$LUKE_DIR/deadman_state"
LOG="$LUKE_DIR/deadman.log"

# Stale past this many seconds = real outage. The watchdog restarts at 1800s
# (30min) and runs every 120s, so 2700s (45min) gives it a full chance to
# recover before we cry wolf. Routine restarts take seconds, not minutes.
THRESHOLD=2700

WATCHDOG_SERVICE="gui/$(id -u)/com.luke.watchdog"

log() { echo "[deadman $(date -Iseconds)] $*" >> "$LOG"; }

# Trim log
if [[ -f "$LOG" ]] && (( $(wc -l < "$LOG") > 200 )); then
    tail -100 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

# ─── Load Telegram credentials from the repo .env (no Luke runtime involved) ───
BOT_TOKEN=""
CHAT_ID=""
if [[ -f "$REPO_DIR/.env" ]]; then
    BOT_TOKEN="$(grep -E '^TELEGRAM_BOT_TOKEN=' "$REPO_DIR/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"'[:space:]')"
    CHAT_ID="$(grep -E '^CHAT_ID=' "$REPO_DIR/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"'[:space:]')"
fi

send_telegram() {
    local text="$1"
    if [[ -z "$BOT_TOKEN" || -z "$CHAT_ID" ]]; then
        log "Cannot send — missing BOT_TOKEN or CHAT_ID"
        return 1
    fi
    curl -sS --max-time 20 \
        "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d chat_id="$CHAT_ID" \
        -d parse_mode="HTML" \
        --data-urlencode text="$text" >/dev/null 2>&1
}

# ─── Watch the watcher: make sure the watchdog agent is loaded ───
if ! launchctl print "$WATCHDOG_SERVICE" >/dev/null 2>&1; then
    log "Watchdog agent not loaded — bootstrapping"
    launchctl bootstrap "gui/$(id -u)" \
        "/Users/filipelm/Library/LaunchAgents/com.luke.watchdog.plist" 2>/dev/null \
        || launchctl kickstart "$WATCHDOG_SERVICE" 2>/dev/null || true
fi

# ─── Read prior state: "<state> <last_good_ts>"  (state = ok|dark) ───
prev_state="ok"
last_good_ts=0
if [[ -f "$STATE_FILE" ]]; then
    read -r prev_state last_good_ts < "$STATE_FILE" 2>/dev/null || true
    [[ -z "$prev_state" ]] && prev_state="ok"
    [[ -z "$last_good_ts" ]] && last_good_ts=0
fi

now=$(date +%s)

# ─── Determine current liveness from the heartbeat ───
alive=false
if [[ -f "$HEARTBEAT_FILE" ]]; then
    read -r hb_ts hb_pid hb_status < "$HEARTBEAT_FILE" 2>/dev/null || true
    if [[ -n "${hb_ts:-}" ]] && (( now - hb_ts <= THRESHOLD )); then
        alive=true
    fi
fi

if [[ "$alive" == "true" ]]; then
    # Heartbeat fresh. If we'd alerted an outage, announce recovery.
    if [[ "$prev_state" == "dark" ]]; then
        send_telegram "💚 <b>I'm back.</b> Heartbeat resumed — the outage is over. (deadman recovery)"
        log "Recovery: heartbeat fresh again, sent back-up alert"
    fi
    echo "ok $now" > "$STATE_FILE"
else
    # Heartbeat stale or missing. Anchor outage start at last known-good time.
    (( last_good_ts == 0 )) && last_good_ts=$now
    dark_for=$(( now - last_good_ts ))
    if (( dark_for > THRESHOLD )) && [[ "$prev_state" != "dark" ]]; then
        mins=$(( dark_for / 60 ))
        send_telegram "🔴 <b>Luke is down.</b> My heartbeat has been stale for ~${mins} min and the watchdog hasn't recovered me. Something needs a look — I can't fix this from inside because I'm not running. (deadman alert)"
        log "OUTAGE: stale ${dark_for}s, watchdog failed to recover — alerted Filipe"
        echo "dark $last_good_ts" > "$STATE_FILE"
    else
        # Still within grace, or already alerted — keep anchor.
        echo "$prev_state $last_good_ts" > "$STATE_FILE"
    fi
fi
