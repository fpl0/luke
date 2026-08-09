#!/usr/bin/env bash
# deploy.sh — Safe deployment pipeline for Luke
#
# Usage:
#   ./deploy.sh                  # Deploy current branch (must be on main)
#   ./deploy.sh <feature-branch> # Merge feature-branch into main, then deploy
#
# Pipeline:
#   1. PRE-DEPLOY: Run full test suite — abort if any fail
#   2. DEPLOY:     Merge feature branch → main + push
#   3. RESTART:    Wait for Luke to go idle, then graceful SIGTERM via kickstart -k
#   4. HEALTH:     Watch logs for the startup_complete event within 90s
#   5. ROLLBACK:   On health failure → git revert HEAD + push + restart
#
# Flags:
#   --no-drain       Restart immediately even if a turn is in flight
#   --drain-auto     Also wait out autonomous runs (crons/deep work), not just user turns
#
# TWO FAILURE MODES THIS SCRIPT DEFENDS AGAINST, both found in the 2026-08-01 audit:
#
#  (a) SELF-KILL.  `launchctl kickstart -k` restarts the whole com.luke job, and an
#      autonomous Luke session runs this script *inside* that job — so the restart
#      takes deploy.sh down with it and steps 4–5 never execute.  A broken deploy
#      then stays live with no health check and no rollback.  Fixed by re-execing
#      detached as a throwaway launchd job (see below); a separate job is the only
#      reliable escape, since launchd reaps the descendants of the one it restarts.
#
#  (b) KILLING A LIVE TURN.  app.py drains its background tasks on SIGTERM, but
#      launchd's ExitTimeOut is what bounds that drain, and an agent turn runs
#      5–15 minutes.  Waiting for idle before the kickstart is the actual fix;
#      the raised ExitTimeOut in com.luke.plist only widens the safety margin.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve LUKE_DIR the way the service does: environment, then .env, then
# default. Watching the wrong log here makes the health check fail on a good
# deploy and roll it back (happened 2026-08-01).
if [[ -z "${LUKE_DIR:-}" ]] && [[ -f "$REPO_DIR/.env" ]]; then
    LUKE_DIR="$(grep -E '^LUKE_DIR=' "$REPO_DIR/.env" | tail -1 | cut -d= -f2- | tr -d '"' | tr -d "'")"
fi
LUKE_DIR="${LUKE_DIR:-$HOME/.luke}"
LUKE_LOG="$LUKE_DIR/luke.log"
LAUNCHD_LABEL="com.luke"
HEALTH_TIMEOUT=90   # seconds to wait for startup_complete in the log
# How long to wait for an in-flight turn to finish before restarting anyway.
# 900s tracks agent_timeout: past that the run is being killed by its own timeout,
# so there is nothing left to protect.
DRAIN_TIMEOUT="${DRAIN_TIMEOUT:-900}"
HEARTBEAT_STALE=300 # heartbeat older than this ⇒ Luke is dead or hung, don't wait on it

DRAIN=1
DRAIN_AUTONOMOUS=0
FEATURE_BRANCH=""
for arg in "$@"; do
    case "$arg" in
        --no-drain)    DRAIN=0 ;;
        --drain-auto)  DRAIN_AUTONOMOUS=1 ;;
        -*)            echo "unknown flag: $arg" >&2; exit 2 ;;
        *)             FEATURE_BRANCH="$arg" ;;
    esac
done

# ─── Colours ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[deploy]${NC} $*"; }
err()  { echo -e "${RED}[deploy]${NC} $*" >&2; }
die()  { err "$*"; exit 1; }

# ─── Live-process introspection ──────────────────────────────────────────────
# `<unix_ts> <pid> <status>` — the scheduler rewrites it every tick.
# `|| true` is load-bearing: with no heartbeat file awk exits 2, and under
# `set -e` that aborted the caller with status 2 instead of returning "not live".
heartbeat_field() { awk -v n="$1" 'NR==1 {print $n}' "$LUKE_DIR/heartbeat" 2>/dev/null || true; }

# Is the service alive and reporting recently? Used to decide whether the
# in-flight marker can be trusted at all.
luke_is_live() {
    local ts pid
    ts="$(heartbeat_field 1)"; pid="$(heartbeat_field 2)"
    [[ "$ts" =~ ^[0-9]+$ ]] && [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    (( $(date +%s) - ts < HEARTBEAT_STALE )) || return 1
    kill -0 "$pid" 2>/dev/null
}

# Walk our own ancestry looking for the live Luke pid. If we find it, this
# script is a descendant of the very job we are about to restart, and a
# kickstart would kill us before the health check ever runs.
inside_luke_tree() {
    local luke_pid pid guard=0
    luke_pid="$(heartbeat_field 2)"
    [[ "$luke_pid" =~ ^[0-9]+$ ]] || return 1
    pid="$$"
    while [[ "$pid" =~ ^[0-9]+$ ]] && (( pid > 1 && guard < 64 )); do
        [[ "$pid" == "$luke_pid" ]] && return 0
        pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
        guard=$(( guard + 1 ))
    done
    return 1
}

# ─── Detached runner ─────────────────────────────────────────────────────────
# `launchctl submit` creates a job with KeepAlive ON, so a submitted command that
# exits — for ANY reason, including "file not found" — is relaunched forever. The
# first live run of this guard spun 43 times in under a minute before it was killed
# by hand. So the submitted command is never deploy.sh itself; it is this wrapper,
# whose last act is to remove its own label whatever happened.
#
# It also takes an ABSOLUTE path to deploy.sh. The submitted job does not inherit
# our working directory, and `${BASH_SOURCE[0]}` is whatever the caller typed — the
# 43 failures above were all `./deploy.sh: No such file or directory`.
build_detached_runner() {
    local label="$1" log="$2" runner="$3"; shift 3
    local quoted_args="" a
    for a in "$@"; do quoted_args+=" $(printf '%q' "$a")"; done
    cat > "$runner" <<RUNNER
#!/bin/bash
# Throwaway runner generated by deploy.sh. Kept after the run, next to its log,
# so a failed detached deploy can be read back rather than guessed at.
export LUKE_DEPLOY_DETACHED=1
export LUKE_DIR=$(printf '%q' "$LUKE_DIR")
# launchd gives a submitted job a bare PATH, so the caller's is captured here.
# Without it step 1 dies on "uv not found" — which it did, on the second live run.
export PATH=$(printf '%q' "$PATH")
export HOME=$(printf '%q' "$HOME")
/bin/bash $(printf '%q' "$REPO_DIR/deploy.sh")$quoted_args
rc=\$?
echo "[runner] deploy.sh exited \$rc — removing launchd label $label"
launchctl remove $(printf '%q' "$label") 2>/dev/null || true
exit \$rc
RUNNER
    chmod +x "$runner"
    echo "$log" > /dev/null  # log path is launchd's business, not the runner's
}

# ─── Drain: wait for in-flight agent runs to finish ──────────────────────────
# `<unix_ts> <pid> <user_runs> <auto_runs>` — see src/luke/inflight.py. The pid
# check is the load-bearing part: a process killed mid-run leaves a non-zero
# count behind forever, so a marker from a dead pid means nothing.
inflight_busy() {
    local hb_pid f_pid users autos
    hb_pid="$(heartbeat_field 2)"
    [[ -f "$LUKE_DIR/inflight" ]] || return 1
    read -r _ f_pid users autos _ < "$LUKE_DIR/inflight" 2>/dev/null || return 1
    [[ "$f_pid" == "$hb_pid" ]] || return 1
    [[ "$users" =~ ^[0-9]+$ ]] && [[ "$autos" =~ ^[0-9]+$ ]] || return 1
    (( users > 0 )) && { DRAIN_REASON="$users user turn(s)"; return 0; }
    if (( DRAIN_AUTONOMOUS )) && (( autos > 0 )); then
        DRAIN_REASON="$autos autonomous run(s)"; return 0
    fi
    return 1
}

wait_for_idle() {
    (( DRAIN )) || { warn "--no-drain: restarting without waiting for in-flight turns."; return; }
    if ! luke_is_live; then
        warn "No fresh heartbeat — Luke is down or hung. Restarting without a drain."
        return
    fi
    local waited=0
    DRAIN_REASON=""
    inflight_busy || { info "Luke is idle — safe to restart."; return; }
    info "Waiting for Luke to go idle (${DRAIN_REASON}) — up to ${DRAIN_TIMEOUT}s…"
    while (( waited < DRAIN_TIMEOUT )); do
        sleep 3; waited=$(( waited + 3 ))
        if ! inflight_busy; then
            info "Idle after ${waited}s — restarting."
            return
        fi
        (( waited % 60 == 0 )) && info "  …still busy (${DRAIN_REASON}) at ${waited}s"
    done
    warn "Still busy after ${DRAIN_TIMEOUT}s (${DRAIN_REASON}) — restarting anyway."
    warn "An in-flight turn will be lost. This is the old behaviour, now at least visible."
}

# ─── Rollback ────────────────────────────────────────────────────────────────
# Called if health check fails. Reverts HEAD, pushes, restarts.
do_rollback() {
    local deploy_sha="$1"
    err "Step 5/5 — ROLLBACK: reverting $deploy_sha…"
    git revert --no-edit HEAD
    git push origin main

    local rollback_sha
    rollback_sha="$(git rev-parse HEAD)"
    warn "Rolled back to $rollback_sha — restarting Luke…"

    # Deliberately NOT draining here. We only reach this path when the health
    # check already failed, so anything "in flight" is running on code that did
    # not come up cleanly — getting back to a known-good SHA beats protecting it.
    launchctl kickstart -k "gui/$(id -u)/$LAUNCHD_LABEL" 2>/dev/null || true

    # Poll for successful restart of the rollback
    local offset elapsed found=0
    offset="$(wc -c < "$LUKE_LOG" 2>/dev/null || echo 0)"
    sleep 5
    elapsed=5
    while (( elapsed < 60 )); do
        sleep 2; elapsed=$(( elapsed + 2 ))
        if tail -c "+$((offset + 1))" "$LUKE_LOG" 2>/dev/null | grep -q '"event": "startup_complete"'; then
            found=1; break
        fi
    done

    if (( found )); then
        warn "Rollback OK — Luke online on $rollback_sha"
    else
        err "Rollback health check also failed. Check logs: tail -f $LUKE_LOG"
    fi
    die "Deploy failed — rolled back to $rollback_sha"
}

# Everything above is definitions. tests/test_deploy_drain.py sources this file
# with DEPLOY_SH_SOURCE_ONLY=1 to exercise the guards above against synthetic
# heartbeat/inflight files — the alternative is shipping restart logic that has
# only ever been read, not run.
if [[ -n "${DEPLOY_SH_SOURCE_ONLY:-}" ]]; then return 0; fi

# ─── Self-kill guard: re-exec as a separate launchd job ──────────────────────
# `nohup`/`disown` are NOT enough — launchd reaps the descendants of the job it
# restarts regardless of session or SIGHUP disposition. `launchctl submit` starts
# a genuinely independent job, which is what survives.
if [[ -z "${LUKE_DEPLOY_DETACHED:-}" ]] && inside_luke_tree; then
    detached_stamp="$(date +%Y%m%dT%H%M%S)"
    detached_label="com.luke.deploy.$$"
    detached_log="$LUKE_DIR/deploy-$detached_stamp.log"
    detached_runner="$LUKE_DIR/deploy-$detached_stamp.runner.sh"
    warn "Running inside the com.luke process tree — the restart would kill this script"
    warn "before its health check. Re-execing as detached job '$detached_label'."
    build_detached_runner "$detached_label" "$detached_log" "$detached_runner" "$@"
    if launchctl submit -l "$detached_label" -o "$detached_log" -e "$detached_log" \
        -- /bin/bash "$detached_runner" 2>/dev/null
    then
        info "Deploy continues detached. Follow it with: tail -f $detached_log"
        warn "This shell will be restarted mid-deploy — READ THAT LOG for the outcome."
        exit 0
    fi
    launchctl remove "$detached_label" 2>/dev/null || true
    err "launchctl submit failed — refusing to run attached, because the restart"
    die "would kill the health check and leave a bad deploy live. Run this from a shell outside Luke."
fi

# ─── Guards ──────────────────────────────────────────────────────────────────
cd "$REPO_DIR"

command -v uv        >/dev/null 2>&1 || die "uv not found"
command -v git       >/dev/null 2>&1 || die "git not found"
command -v launchctl >/dev/null 2>&1 || die "launchctl not found (macOS only)"
[[ -f "$LUKE_LOG" ]] || die "Luke log not found at $LUKE_LOG — wrong LUKE_DIR? Health check would always fail."

# ─── Step 1: PRE-DEPLOY — full test suite ────────────────────────────────────
info "Step 1/5 — Running test suite…"
if ! uv run pytest --tb=short -q; then
    die "Tests FAILED — deploy aborted. Fix failures before deploying."
fi
info "All tests passed."

# ─── Step 2: DEPLOY — merge + push ───────────────────────────────────────────
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [[ -n "$FEATURE_BRANCH" ]]; then
    info "Step 2/5 — Merging '$FEATURE_BRANCH' into main…"
    [[ "$CURRENT_BRANCH" != "main" ]] && git checkout main
    git pull --ff-only origin main
    git merge --no-ff "$FEATURE_BRANCH" -m "deploy: merge $FEATURE_BRANCH into main"
elif [[ "$CURRENT_BRANCH" == "main" ]]; then
    info "Step 2/5 — On main — pulling latest…"
    git pull --ff-only origin main
else
    die "On branch '$CURRENT_BRANCH' with no feature branch argument. Pass the branch name or checkout main first."
fi

DEPLOY_SHA="$(git rev-parse HEAD)"
info "Deploying commit: $DEPLOY_SHA"
git push origin main

# ─── Step 3: GRACEFUL RESTART ────────────────────────────────────────────────
info "Step 3/5 — Graceful restart via launchctl kickstart -k…"
wait_for_idle
# Capture log offset *before* restart so health check only scans new output
LOG_OFFSET="$(wc -c < "$LUKE_LOG" 2>/dev/null || echo 0)"

launchctl kickstart -k "gui/$(id -u)/$LAUNCHD_LABEL" 2>/dev/null || {
    warn "kickstart returned non-zero — attempting load + start…"
    launchctl load "$HOME/Library/LaunchAgents/$LAUNCHD_LABEL.plist" 2>/dev/null || true
    launchctl start "$LAUNCHD_LABEL" 2>/dev/null || true
}

# ─── Step 4: HEALTH CHECK ────────────────────────────────────────────────────
info "Step 4/5 — Waiting up to ${HEALTH_TIMEOUT}s for startup_complete in logs…"

# "Back online." is a Telegram message, not a reliable log line — outbound
# dedup suppresses it on back-to-back restarts, which made healthy deploys
# roll back. startup_complete is written to the log on every clean boot.
ELAPSED=0
FOUND=0
while (( ELAPSED < HEALTH_TIMEOUT )); do
    sleep 2; ELAPSED=$(( ELAPSED + 2 ))
    if tail -c "+$((LOG_OFFSET + 1))" "$LUKE_LOG" 2>/dev/null | grep -q '"event": "startup_complete"'; then
        FOUND=1; break
    fi
done

if (( FOUND )); then
    info "Health check PASSED (${ELAPSED}s). Luke is online."
    info "Step 5/5 — Deploy complete. Commit: $DEPLOY_SHA"
    exit 0
else
    err "Health check FAILED — startup_complete not seen after ${HEALTH_TIMEOUT}s."
    do_rollback "$DEPLOY_SHA"
fi
