#!/usr/bin/env bash
# letta_stack_health.sh — External fail-loud monitor for the Letta memory stack.
#
# Phase 5.3 of goal-letta-full-power. Luke's whole silent-fail history (the
# May 15 → Jun 27 six-week silence) says a load-bearing dependency that dies
# quietly is the worst failure mode. Once MEMORY_BACKEND=letta, Luke's semantic
# recall is sourced from the Letta stack — four daemons that can each drop
# silently with NO user-visible symptom except subtly worse memory:
#   1. Postgres 16      (:5432)  — Letta's store + pgvector
#   2. Letta server     (:8283)  — /v1/health/
#   3. Claude bridge    (:17596) — /v1/models (OpenAI-compat proxy → Claude)
#   4. Embed (Ollama)   (:11434) — /api/tags (bge embeddings for recall)
# Plus the delta-sync freshness watermark (the shadow archive drifts if the
# daily sync dies — a live-recall correctness gap, not a crash).
#
# Modelled on deadman.sh: depends on NOTHING in the Luke runtime — pure bash +
# curl. Reads Telegram creds straight from the repo .env and messages Filipe
# directly. One alert per outage, one recovery alert when the stack comes back.
# Critical alerts are gated on MEMORY_BACKEND=letta (a down bridge while Luke is
# still on sqlite is not load-bearing — logged, not alerted) and on a grace
# window so a momentary blip during a daemon restart doesn't cry wolf.
#
# Runs as its own launchd agent (com.luke.lettahealth) on a short interval.
# Reversible; touches no live Luke state.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$REPO_DIR/letta_stack_health.log"
STATE_FILE="$REPO_DIR/.letta_health_state"
WM_FILE="$REPO_DIR/.letta_sync_watermark"

# A daemon must stay down longer than this before we alert (covers routine
# launchd restarts / KeepAlive respawns, which take seconds).
DOWN_GRACE=300          # 5 min
# Delta-sync watermark older than this = the daily 04:30 sync has silently died.
SYNC_STALE=172800       # 48h

log() { echo "[lettahealth $(date -Iseconds)] $*" >> "$LOG"; }

# Trim log
if [[ -f "$LOG" ]] && (( $(wc -l < "$LOG") > 200 )); then
    tail -100 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

# ─── Telegram creds from repo .env (no Luke runtime involved) ───
BOT_TOKEN=""; CHAT_ID=""; MEMORY_BACKEND="sqlite"
if [[ -f "$REPO_DIR/.env" ]]; then
    BOT_TOKEN="$(grep -E '^TELEGRAM_BOT_TOKEN=' "$REPO_DIR/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"'[:space:]')"
    CHAT_ID="$(grep -E '^CHAT_ID=' "$REPO_DIR/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"'[:space:]')"
    MEMORY_BACKEND="$(grep -E '^MEMORY_BACKEND=' "$REPO_DIR/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"'[:space:]')"
    [[ -z "$MEMORY_BACKEND" ]] && MEMORY_BACKEND="sqlite"
fi

send_telegram() {
    local text="$1"
    if [[ -z "$BOT_TOKEN" || -z "$CHAT_ID" ]]; then
        log "Cannot send — missing BOT_TOKEN or CHAT_ID"; return 1
    fi
    curl -sS --max-time 20 \
        "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d chat_id="$CHAT_ID" -d parse_mode="HTML" \
        --data-urlencode text="$text" >/dev/null 2>&1
}

# ─── Probe each daemon (short timeouts; a hang counts as down) ───
down=()

pg_isready -h localhost -p 5432 >/dev/null 2>&1 || down+=("Postgres:5432")

code=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://localhost:8283/v1/health/ 2>/dev/null)
[[ "$code" == "200" ]] || down+=("Letta:8283($code)")

code=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://localhost:17596/v1/models 2>/dev/null)
[[ "$code" == "200" ]] || down+=("bridge:17596($code)")

code=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://localhost:11434/api/tags 2>/dev/null)
[[ "$code" == "200" ]] || down+=("embed:11434($code)")

# ─── Delta-sync freshness (only meaningful when shadow-run is live) ───
sync_stale=""
if [[ "$MEMORY_BACKEND" == "letta" && -f "$WM_FILE" ]]; then
    wm="$(cat "$WM_FILE" 2>/dev/null)"
    # watermark is an ISO ts; convert to epoch (BSD date)
    wm_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%S" "${wm%%.*}" +%s 2>/dev/null || echo 0)
    now=$(date +%s)
    if (( wm_epoch > 0 )) && (( now - wm_epoch > SYNC_STALE )); then
        hrs=$(( (now - wm_epoch) / 3600 ))
        sync_stale="delta-sync watermark ${hrs}h old"
    fi
fi

now=$(date +%s)
# Build a stable signature of the current problem set so we alert once per distinct outage.
problem=""
(( ${#down[@]} > 0 )) && problem="${down[*]}"
[[ -n "$sync_stale" ]] && problem="${problem}${problem:+ | }${sync_stale}"

# ─── State: "<sig>\t<first_seen_epoch>\t<alerted:0|1>" ───
prev_sig=""; first_seen=0; alerted=0
if [[ -f "$STATE_FILE" ]]; then
    IFS=$'\t' read -r prev_sig first_seen alerted < "$STATE_FILE" 2>/dev/null || true
    [[ -z "$first_seen" ]] && first_seen=0
    [[ -z "$alerted" ]] && alerted=0
fi

# Backend gate: a down bridge/Letta while on sqlite is not load-bearing.
critical=false
if [[ "$MEMORY_BACKEND" == "letta" && -n "$problem" ]]; then
    critical=true
fi

if [[ -z "$problem" ]]; then
    # Healthy. If we had alerted an outage, announce recovery.
    if (( alerted == 1 )); then
        send_telegram "💚 <b>Letta stack recovered.</b> All memory daemons are back (Postgres, Letta, bridge, embed). (lettahealth)"
        log "RECOVERY: stack healthy again after outage — sent all-clear"
    fi
    : > "$STATE_FILE"   # clear
    log "OK backend=$MEMORY_BACKEND — all daemons healthy"
else
    # A problem exists. Anchor first-seen if this is a new signature.
    if [[ "$problem" != "$prev_sig" ]]; then
        first_seen=$now; alerted=0
    fi
    down_for=$(( now - first_seen ))
    if [[ "$critical" == "true" ]] && (( down_for >= DOWN_GRACE )) && (( alerted == 0 )); then
        send_telegram "🔴 <b>Letta memory stack degraded.</b> backend=letta is live and: <b>${problem}</b>. Semantic recall is sourced from Letta right now, so this silently worsens my memory until it's fixed. (lettahealth)"
        log "OUTAGE ALERT: $problem (down ${down_for}s, backend=letta)"
        alerted=1
    else
        log "PROBLEM (no alert): $problem | critical=$critical down_for=${down_for}s alerted=$alerted backend=$MEMORY_BACKEND"
    fi
    printf '%s\t%s\t%s\n' "$problem" "$first_seen" "$alerted" > "$STATE_FILE"
fi
