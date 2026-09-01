"""Live check for the stall-alert downtime credit (marker STALL-DOWNTIME-DEPLOY-20260901).

Mirrors enforce_plan_momentum's arithmetic without sending anything: for every
in_progress plan, print wall-clock staleness, credited downtime, and the
observed hours the alert actually gates on. Run against the deployed venv.
"""

import re
from datetime import UTC, datetime
from pathlib import Path

from luke import db
from luke.behaviors import _STALL_ALERT_HOURS, _STALL_NUDGE_HOURS, _plan_last_updated
from luke.config import settings

now = datetime.now(UTC)
plans_dir = settings.workspace_dir / "plans"
rows = []
for path in sorted(Path(plans_dir).glob("*.md")):
    head = path.read_text(encoding="utf-8")[:2000]
    m = re.search(r"\*\*Status:\*\*\s*(\S+)", head)
    if not m or m.group(1).strip().lower() != "in_progress":
        continue
    updated = _plan_last_updated(path)
    if updated is None:
        rows.append((path.stem, None, None, None, "NO Last updated header"))
        continue
    stale_h = (now - updated).total_seconds() / 3600
    down_h = db.downtime_hours_since(updated)
    observed_h = stale_h - down_h
    if stale_h < _STALL_NUDGE_HOURS:
        verdict = "quiet (under 48h nudge)"
    elif stale_h < _STALL_ALERT_HOURS:
        verdict = "nudge only (under 96h wall-clock)"
    elif observed_h < _STALL_ALERT_HOURS:
        verdict = "SUPPRESSED by downtime credit"
    else:
        verdict = "*** WOULD ALERT ***"
    rows.append((path.stem, stale_h, down_h, observed_h, verdict))

print(f"{'plan':<44} {'stale':>8} {'down':>8} {'observed':>9}  verdict")
print("-" * 100)
for stem, stale_h, down_h, observed_h, verdict in rows:
    if stale_h is None:
        print(f"{stem:<44} {'-':>8} {'-':>8} {'-':>9}  {verdict}")
        continue
    print(f"{stem:<44} {stale_h:>8.1f} {down_h:>8.1f} {observed_h:>9.1f}  {verdict}")

would = [r for r in rows if r[4] == "*** WOULD ALERT ***"]
print(f"\n{len(rows)} in_progress plans; {len(would)} would alert Filipe right now.")
