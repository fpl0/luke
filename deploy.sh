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
#   3. RESTART:    Graceful SIGTERM via launchctl kickstart -k (60s drain)
#   4. HEALTH:     Watch logs for "Back online." within 90s
#   5. ROLLBACK:   On health failure → git revert HEAD + push + restart

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUKE_DIR="${LUKE_DIR:-$HOME/.luke}"
LUKE_LOG="$LUKE_DIR/luke.log"
LAUNCHD_LABEL="com.luke"
HEALTH_TIMEOUT=90   # seconds to wait for "Back online."
FEATURE_BRANCH="${1:-}"

# ─── Colours ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[deploy]${NC} $*"; }
err()  { echo -e "${RED}[deploy]${NC} $*" >&2; }
die()  { err "$*"; exit 1; }

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

    launchctl kickstart -k "gui/$(id -u)/$LAUNCHD_LABEL" 2>/dev/null || true

    # Poll for successful restart of the rollback
    local offset elapsed found=0
    offset="$(wc -c < "$LUKE_LOG" 2>/dev/null || echo 0)"
    sleep 5
    elapsed=5
    while (( elapsed < 60 )); do
        sleep 2; elapsed=$(( elapsed + 2 ))
        if tail -c "+$((offset + 1))" "$LUKE_LOG" 2>/dev/null | grep -q 'Back online\.'; then
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

# ─── Guards ──────────────────────────────────────────────────────────────────
cd "$REPO_DIR"

command -v uv        >/dev/null 2>&1 || die "uv not found"
command -v git       >/dev/null 2>&1 || die "git not found"
command -v launchctl >/dev/null 2>&1 || die "launchctl not found (macOS only)"

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
# Capture log offset *before* restart so health check only scans new output
LOG_OFFSET="$(wc -c < "$LUKE_LOG" 2>/dev/null || echo 0)"

launchctl kickstart -k "gui/$(id -u)/$LAUNCHD_LABEL" 2>/dev/null || {
    warn "kickstart returned non-zero — attempting load + start…"
    launchctl load "$HOME/Library/LaunchAgents/$LAUNCHD_LABEL.plist" 2>/dev/null || true
    launchctl start "$LAUNCHD_LABEL" 2>/dev/null || true
}

# ─── Step 4: HEALTH CHECK ────────────────────────────────────────────────────
info "Step 4/5 — Waiting up to ${HEALTH_TIMEOUT}s for \"Back online.\" in logs…"

ELAPSED=0
FOUND=0
while (( ELAPSED < HEALTH_TIMEOUT )); do
    sleep 2; ELAPSED=$(( ELAPSED + 2 ))
    if tail -c "+$((LOG_OFFSET + 1))" "$LUKE_LOG" 2>/dev/null | grep -q 'Back online\.'; then
        FOUND=1; break
    fi
done

if (( FOUND )); then
    info "Health check PASSED (${ELAPSED}s). Luke is online."
    info "Step 5/5 — Deploy complete. Commit: $DEPLOY_SHA"
    exit 0
else
    err "Health check FAILED — \"Back online.\" not seen after ${HEALTH_TIMEOUT}s."
    do_rollback "$DEPLOY_SHA"
fi
