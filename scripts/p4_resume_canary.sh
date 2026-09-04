#!/bin/bash
# Weekly / on-SDK-change caller for the P4 resume canary.
#
# Runs OUTSIDE the daemon on purpose (com.luke.p4canary, launchd). The thing this watches
# is a routing decision inside app.py; an alarm living in the same process it is watching
# is the failure recorded in insight-the-run-that-notices-things-was-the-one-dying-unnoticed.
#
# --if-due is a no-op on most days: it runs the full matrix only when the claude-agent-sdk
# version has moved since the last recorded verdict, or when a week has passed. Fires
# daily so a version bump is caught within 24h rather than up to seven days later.
#
# Exit 0 silent / 1 FAIL (the script itself touches $LUKE_DIR/cheap_resume.off) /
# 2 !!RESUME_CANARY_BLIND!!.

set -uo pipefail

REPO="/Users/filipelm/Code/luke"
LOG="$REPO/logs/p4_resume_canary.log"
mkdir -p "$REPO/logs"

cd "$REPO" || exit 2

# The whole story of the 14 Aug false result: a subprocess that did not source .env got a
# complete, plausible, four-cell table saying the opposite of the truth.
if [ -f "$REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO/.env"
    set +a
fi

out=$("$REPO/.venv/bin/python" "$REPO/scripts/p4_resume_canary.py" --if-due 2>&1)
code=$?

case "$code" in
    0) [ -n "$out" ] && echo "$(date -u +%FT%TZ) $out" >>"$LOG" ;;
    1) echo "$(date -u +%FT%TZ) FAIL $out" >>"$LOG" ;;
    *) echo "$(date -u +%FT%TZ) !!RESUME_CANARY_BLIND!! $out" >>"$LOG" ;;
esac

exit "$code"
