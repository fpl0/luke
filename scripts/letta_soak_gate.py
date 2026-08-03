#!/usr/bin/env python3
"""Gate 5.6 — the soak — measured from artefacts instead of asserted by whoever woke up.

5.6 is an ACCEPT gate: "7 consecutive green days by Aug 10, any red day restarts the
counter." Three of the four gates deciding the cutover are closed; this is the only one
still accumulating, and until now the whole of it lived in the 03:40 cron's PROMPT — an
agent grepped a log, eyeballed two files, and hand-wrote `soak day k/7 GREEN` into the
plan. Same family as the work claim that was advisory prose (d9c8d74): the rule was right
and nothing executed it.

Read the prompt against the machine and it does not measure what it says:

1. **The log it greps is not there.** Step 1 `cd /Users/filipelm/Code/luke`; step 3 greps
   `luke.log`, which lives at `/Users/filipelm/Luke/luke.log`. In that cwd the file does
   not exist, so the count is empty and the day reports GREEN *by absence of evidence* —
   the one direction a gate must never fail.
2. **The pattern matches the gate's own commands.** `grep letta_search_failed luke.log`
   also hits `{"event": "tool_use", ...}` lines echoing previous nights' greps: 34 raw
   matches against 28 real events on 2026-08-03, and the gap grows by one every time the
   gate runs. A measurement that writes into the thing it measures. The real events are
   `"event": "letta_search_failed"`; the last genuine one was 2026-08-01T09:32:17Z.
3. **It names an artefact that does not exist.** `lettablockfresh.log` is the launchd
   stdout sink; the freshness verdict is in `logs/letta_block_freshness.log`.
4. **A missing night is indistinguishable from a green one.** The counter is incremented
   by hand from the previous line. If a night never runs there is no line, and nothing
   downstream can tell "we did not measure" from "we measured and it was fine". On Aug 10
   the claim "7 consecutive green days" would rest on seven agent judgements and no record
   of the days between them.

So this file keeps a ledger instead. One row per UTC day, written from the artefacts, with
the streak computed by walking it — never incremented. The distinctions that matter:

  GREEN        every piece of evidence present, fresh, and passing
  RED          evidence present and failing — a real regression
  INSUFFICIENT evidence absent or stale. NOT green. Breaks the streak.
  MISSING      no run happened that day at all. Inserted automatically for every calendar
               day between the last row and today, so a gap can never be stepped over.

INSUFFICIENT and MISSING both reset the counter, and that is deliberate: 5.6 claims seven
*measured* green days, and a day nobody measured is not one of them. Being strict costs
nothing here — the streak restarting on 2026-08-03 still reaches day 7 on 2026-08-09, a
day before the deadline.

Commands:
  record [--force]   measure today, insert any missed days, append, print the gate line
  status             the streak and every day it rests on

Exit: 0 GREEN · 1 RED · 2 INSUFFICIENT · 3 usage/internal
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(REPO, "logs", "letta_soak.jsonl")
BLOCKFRESH = os.path.join(REPO, "logs", "letta_block_freshness.log")
BENCH = "/tmp/claude/letta_titled_bench.json"

# Evidence older than this is not evidence about today, whatever it says — and the age has
# to come from WHO PRODUCES IT, not from a round number. The bench is written by the same
# 03:40 run a minute earlier, so anything hours old means step 1 did not run. The 5.5
# refresh is a separate launchd job at 04:10Z, i.e. THIRTY MINUTES AFTER this gate, so its
# newest verdict is always ~23h30m old on a perfectly healthy night (the ordering flagged
# in the plan on 2026-08-03). A flat 24h would sit ~30 minutes from tipping every night and
# reset the streak on the first late start — a spurious RED with a real cost, seven days
# from the deadline. One full cycle plus jitter.
BENCH_MAX_AGE = timedelta(hours=6)
BLOCKFRESH_MAX_AGE = timedelta(hours=26)
# Falling back to a 24h window when the ledger is empty; a longer window only ever counts
# MORE failures, so widening it after a gap is the safe direction.
DEFAULT_WINDOW = timedelta(hours=24)

# A cheap prefilter so a 24MB log is not JSON-parsed line by line — deliberately LOOSE.
# The first version of this was `'"event": "letta_search_failed"'`, which reads as the
# careful choice and is the same defect in a better disguise: it encodes an assumption
# about the renderer's whitespace, and a logger emitting compact JSON would make it match
# nothing at all — zero failures, forever, green by absence. The event FIELD decides.
SEARCH_FAILED_MARKER = "letta_search_failed"


# ---------------------------------------------------------------- environment

LOG_ROOTS = ["/Users/filipelm/Luke", REPO]


def resolve_luke_log() -> tuple[str | None, str]:
    """Find the live luke.log by LOOKING, not by encoding a claim about where it lives.

    The cron's bug is exactly the failure this returns None for: a path that is wrong is
    a path that reads as zero failures. Candidates in order of authority; the reason
    string is carried into the ledger so a later reader knows which file was counted.
    """
    candidates = []
    env = os.environ.get("LUKE_HOME")
    if env:
        candidates.append(os.path.join(env, "luke.log"))
    candidates += [os.path.join(root, "luke.log") for root in LOG_ROOTS]
    for path in candidates:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path, f"resolved {path}"
    return None, "luke.log not found at any of: " + ", ".join(candidates)


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- evidence

def count_search_failures(log_path: str, since: datetime) -> tuple[int, str | None]:
    """Real `letta_search_failed` events in the window, and the newest one's timestamp.

    Decided on the event FIELD, after parsing. The bare substring also matches `tool_use`
    records echoing the gate's own grep, which is why the naive count grows every night the
    gate runs — the measurement writes into the thing it measures.
    """
    count = 0
    newest: str | None = None
    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            if SEARCH_FAILED_MARKER not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # A tool_use echo carries the marker inside `input`, never as its own event.
            if rec.get("event") != "letta_search_failed":
                continue
            ts = _parse_ts(rec.get("timestamp"))
            if ts is None:
                # Undatable event: count it. Failing toward RED is the safe direction.
                count += 1
                continue
            if ts >= since:
                count += 1
            if newest is None or rec.get("timestamp", "") > newest:
                newest = rec.get("timestamp")
    return count, newest


def read_blockfresh(now: datetime) -> tuple[str, str | None, str]:
    """Last verdict in the 5.5 freshness log, and whether it is recent enough to count."""
    if not os.path.isfile(BLOCKFRESH):
        return "MISSING", None, f"{BLOCKFRESH} does not exist"
    last = None
    with open(BLOCKFRESH, "r", errors="replace") as fh:
        for line in fh:
            if line.strip():
                last = line.strip()
    if not last:
        return "MISSING", None, "freshness log is empty"
    parts = last.split(None, 1)
    ts = _parse_ts(parts[0]) if parts else None
    body = parts[1] if len(parts) > 1 else ""
    if ts is None:
        return "MISSING", None, f"unparseable freshness line: {last[:60]}"
    if now - ts > BLOCKFRESH_MAX_AGE:
        return "STALE", parts[0], f"last refresh {parts[0]} is older than one 5.5 cycle"
    if body.startswith("ALL FRESH"):
        return "FRESH", parts[0], "all 5 blocks match sqlite"
    return "DRIFT", parts[0], body[:80]


def read_bench(now: datetime) -> tuple[str, dict | None, str]:
    """The 5.4 top-1 result, scored on the COMPARABLE basis the script itself reports.

    The cron's alert clause says `top-1 < 12/12`, but `letta_bench_titled` reports N of
    `comparable` because one ground-truth row is a frozen Jul-28 snapshot that scores
    freshness as a miss — established as the correct basis by the Aug-1 re-baseline. On a
    literal reading that clause has been tripped three nights running by a row working as
    designed, which is an alert condition only satisfiable by reading the plan.
    """
    if not os.path.isfile(BENCH):
        return "MISSING", None, f"{BENCH} does not exist — 5.4 did not run"
    age = now - datetime.fromtimestamp(os.path.getmtime(BENCH), timezone.utc)
    if age > BENCH_MAX_AGE:
        return "STALE", None, (f"bench result is {age.days}d{age.seconds // 3600}h old — "
                               "5.4 did not run in this pass")
    try:
        data = json.load(open(BENCH))
    except (json.JSONDecodeError, OSError) as exc:
        return "MISSING", None, f"unreadable bench result: {exc}"
    hits, comparable = data.get("titled_hits"), data.get("comparable")
    if not isinstance(hits, int) or not isinstance(comparable, int) or comparable <= 0:
        return "MISSING", None, "bench result has no comparable score"
    summary = {"hits": hits, "comparable": comparable, "fresh_miss": data.get("fresh", 0)}
    if hits < comparable:
        return "MISS", summary, f"top-1 {hits}/{comparable} comparable"
    return "PASS", summary, f"top-1 {hits}/{comparable} comparable"


# ---------------------------------------------------------------- ledger

def load_ledger() -> list[dict]:
    if not os.path.isfile(LEDGER):
        return []
    rows = []
    with open(LEDGER, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn line is a gap in the record, not something to guess at.
                rows.append({"date": None, "verdict": "MISSING",
                             "reasons": ["unparseable ledger line"]})
    return rows


def append_rows(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def streak(rows: list[dict]) -> int:
    """Consecutive GREEN days ending at the newest row. Walked, never incremented."""
    n = 0
    for row in reversed(rows):
        if row.get("verdict") == "GREEN":
            n += 1
        else:
            break
    return n


def missed_days(last_date: str | None, today: str) -> list[str]:
    """Every calendar day strictly between the last row and today, so a gap is recorded."""
    if not last_date:
        return []
    try:
        cur = datetime.strptime(last_date, "%Y-%m-%d").date()
        end = datetime.strptime(today, "%Y-%m-%d").date()
    except ValueError:
        return []
    out = []
    cur += timedelta(days=1)
    while cur < end:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


# ---------------------------------------------------------------- commands

def cmd_record(args: argparse.Namespace) -> int:
    now = _now()
    today = now.date().isoformat()
    rows = load_ledger()
    dated = [r for r in rows if r.get("date")]

    if dated and dated[-1]["date"] == today and not args.force:
        print(f"already recorded for {today}: {dated[-1]['verdict']} "
              f"(streak {streak(rows)}/7). Use --force to re-measure.")
        return 3

    # The window for counting failures runs from the last measurement, so nothing that
    # happened between two runs falls between two stools.
    since = _parse_ts(dated[-1].get("measured_at")) if dated else None
    if since is None:
        since = now - DEFAULT_WINDOW

    reasons: list[str] = []
    verdict = "GREEN"

    # RED (a real regression, and something to act on) outranks INSUFFICIENT (we could
    # not tell). Neither is green, but they mean different things to whoever reads it.
    rank = {"GREEN": 0, "INSUFFICIENT": 1, "RED": 2}

    def demote(level: str, why: str) -> None:
        nonlocal verdict
        reasons.append(why)
        if rank[level] > rank[verdict]:
            verdict = level

    log_path, log_why = resolve_luke_log()
    failures, last_failure = 0, None
    if log_path is None:
        demote("INSUFFICIENT", log_why)
    else:
        failures, last_failure = count_search_failures(log_path, since)
        if failures:
            demote("RED", f"{failures} letta_search_failed since {since.isoformat()}")

    bf_state, bf_at, bf_why = read_blockfresh(now)
    if bf_state == "DRIFT":
        demote("RED", f"core blocks drifted: {bf_why}")
    elif bf_state != "FRESH":
        demote("INSUFFICIENT", f"blockfresh {bf_state}: {bf_why}")

    bench_state, bench, bench_why = read_bench(now)
    if bench_state == "MISS":
        demote("RED", f"5.4 {bench_why}")
    elif bench_state != "PASS":
        demote("INSUFFICIENT", f"bench {bench_state}: {bench_why}")

    gap_rows = [
        {"date": d, "verdict": "MISSING", "measured_at": None,
         "reasons": ["no gate run recorded for this day"], "streak_after": 0}
        for d in missed_days(dated[-1]["date"] if dated else None, today)
    ]

    row = {
        "date": today,
        "verdict": verdict,
        "measured_at": now.isoformat(),
        "window_from": since.isoformat(),
        "search_failed": failures,
        "search_failed_last": last_failure,
        "luke_log": log_why,
        "blockfresh": bf_state,
        "blockfresh_at": bf_at,
        "bench": bench_state,
        "bench_detail": bench,
        "reasons": reasons or ["all evidence present, fresh and passing"],
    }
    new = gap_rows + [row]
    row["streak_after"] = streak(rows + new)

    if not args.dry_run:
        append_rows(new)

    for gap in gap_rows:
        print(f"  gap: {gap['date']} MISSING — no gate run recorded")
    print(f"{today} {verdict} — streak {row['streak_after']}/7")
    for why in row["reasons"]:
        print(f"  · {why}")
    print()
    print("gate-log line:")
    bench_txt = (f"{bench['hits']}/{bench['comparable']} comparable" if bench else bench_state)
    print(f"- {today}: 5.4 top-1 {bench_txt}, blockfresh {bf_state}, "
          f"search_failed {failures} — soak day {row['streak_after']}/7 {verdict}")

    return {"GREEN": 0, "RED": 1}.get(verdict, 2)


def cmd_status(_args: argparse.Namespace) -> int:
    rows = load_ledger()
    if not rows:
        print("no soak ledger yet — run `record`")
        return 2
    n = streak(rows)
    print(f"soak streak: {n}/7\n")
    for row in rows[-10:]:
        mark = {"GREEN": "✓", "RED": "✗", "INSUFFICIENT": "?", "MISSING": "—"}.get(
            row.get("verdict", ""), "?")
        why = "; ".join(row.get("reasons", []))[:90]
        print(f"  {mark} {row.get('date')} {row.get('verdict'):12s} {why}")
    print()
    if n >= 7:
        print("5.6 ACCEPT MET — 7 consecutive measured green days")
        return 0
    need = 7 - n
    print(f"{need} more consecutive green day(s) needed; earliest close "
          f"{(_now().date() + timedelta(days=need)).isoformat()}")
    return 0 if n else 2


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="measure today and append to the ledger")
    rec.add_argument("--force", action="store_true", help="re-measure a day already recorded")
    rec.add_argument("--dry-run", action="store_true", help="measure but do not write")
    rec.set_defaults(fn=cmd_record)
    st = sub.add_parser("status", help="current streak and the days it rests on")
    st.set_defaults(fn=cmd_status)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
