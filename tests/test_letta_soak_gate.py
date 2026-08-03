"""The soak gate has to be unable to report a green day it cannot substantiate.

5.6 is an accept gate for an irreversible substrate swap, and every one of its failure modes
found on 2026-08-03 pointed the same way — toward a green: a log path that did not exist in
the cron's cwd (absence of evidence read as zero failures), a grep pattern that matched the
gate's own previous commands, an artefact named that was never written, and a hand-incremented
counter that could step over a night nobody measured.

So these tests are almost all about the *absence* cases. A gate tested only on the day
everything works is indistinguishable from `return "GREEN"`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def sg():
    spec = importlib.util.spec_from_file_location(
        "letta_soak_gate", os.path.join(REPO, "scripts", "letta_soak_gate.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["letta_soak_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _event(ts: str) -> str:
    return json.dumps({"err": "HTTP Error 500", "event": "letta_search_failed",
                       "level": "warning", "timestamp": ts})


def _echo(ts: str) -> str:
    """A tool_use record carrying the marker inside the command it echoes."""
    return json.dumps({"tool": "Bash",
                       "input": '{\'command\': \'grep -c "letta_search_failed" luke.log\'}',
                       "event": "tool_use", "level": "info", "timestamp": ts})


def _write_log(tmp_path, lines):
    path = tmp_path / "luke.log"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# ------------------------------------------------------- the self-referential grep

def test_tool_use_echoes_are_not_failures(sg, tmp_path):
    """The naive `grep letta_search_failed` count grows every night the gate runs.

    Each run's own command is logged as a tool_use record containing the marker, so the
    measurement writes into the thing it measures. On the live log that was 34 raw matches
    against 28 real events.
    """
    since = datetime(2026, 8, 1, tzinfo=timezone.utc)
    log = _write_log(tmp_path, [_echo("2026-08-02T03:41:53Z"), _echo("2026-08-03T03:42:03Z")])
    assert sg.count_search_failures(log, since) == (0, None)

    raw = sum(1 for line in open(log) if "letta_search_failed" in line)
    assert raw == 2, "the substring the cron greps for really is present on both echo lines"


def test_compact_json_is_still_counted(sg, tmp_path):
    """The prefilter must not encode the renderer's whitespace.

    Narrowing it to `'"event": "letta_search_failed"'` reads as the careful choice and is
    the green-by-absence bug again: a logger emitting compact JSON would match nothing, and
    the gate would report zero failures forever without anything erroring.
    """
    line = '{"event":"letta_search_failed","timestamp":"2026-08-03T01:00:00Z"}'
    log = _write_log(tmp_path, [line])
    assert sg.count_search_failures(log, datetime(2026, 8, 3, tzinfo=timezone.utc))[0] == 1


def test_real_events_counted_and_windowed(sg, tmp_path):
    since = datetime(2026, 8, 2, tzinfo=timezone.utc)
    log = _write_log(tmp_path, [
        _event("2026-08-01T09:32:17Z"),   # before the window
        _event("2026-08-02T10:00:00Z"),   # in
        _event("2026-08-03T01:00:00Z"),   # in
        _echo("2026-08-03T03:42:03Z"),
    ])
    count, newest = sg.count_search_failures(log, since)
    assert count == 2
    assert newest == "2026-08-03T01:00:00Z"


def test_undatable_event_counts_toward_red(sg, tmp_path):
    """An event we cannot place in time is counted. Failing toward RED is the safe side."""
    log = _write_log(tmp_path, [json.dumps({"event": "letta_search_failed", "err": "x"})])
    count, _ = sg.count_search_failures(log, datetime(2026, 8, 3, tzinfo=timezone.utc))
    assert count == 1


# ------------------------------------------------------- absence of evidence

def test_missing_log_is_insufficient_not_green(sg, tmp_path, monkeypatch):
    """The cron's actual bug: cwd is the repo, luke.log lives elsewhere, count reads zero."""
    monkeypatch.delenv("LUKE_HOME", raising=False)
    monkeypatch.setattr(sg, "LOG_ROOTS", [str(tmp_path)])
    path, why = sg.resolve_luke_log()
    assert path is None
    assert "not found" in why


def test_empty_log_is_not_accepted(sg, tmp_path, monkeypatch):
    """A zero-byte luke.log is a broken pipeline, not a quiet night."""
    (tmp_path / "luke.log").write_text("")
    monkeypatch.delenv("LUKE_HOME", raising=False)
    monkeypatch.setattr(sg, "LOG_ROOTS", [str(tmp_path)])
    assert sg.resolve_luke_log()[0] is None


def test_env_root_wins_and_is_reported(sg, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "luke.log").write_text(_event("2026-08-03T01:00:00Z") + "\n")
    monkeypatch.setenv("LUKE_HOME", str(home))
    monkeypatch.setattr(sg, "LOG_ROOTS", [])
    path, why = sg.resolve_luke_log()
    assert path == str(home / "luke.log")
    assert str(home) in why, "the ledger records which file was actually counted"


# ------------------------------------------------------- staleness

def test_blockfresh_stale_is_not_fresh(sg, tmp_path, monkeypatch):
    """A FRESH verdict from two days ago is not evidence about today."""
    now = datetime(2026, 8, 3, 4, tzinfo=timezone.utc)
    log = tmp_path / "bf.log"
    log.write_text("2026-08-01T04:10:06Z  ALL FRESH       all 5 blocks match sqlite\n")
    monkeypatch.setattr(sg, "BLOCKFRESH", str(log))
    state, _, why = sg.read_blockfresh(now)
    assert state == "STALE"
    assert "cycle" in why


def test_the_healthy_ordering_is_not_read_as_stale(sg, tmp_path, monkeypatch):
    """5.5 refreshes at 04:10Z, THIS gate runs at 03:40Z — the newest verdict is always
    ~23h30m old on a perfectly healthy night. A flat 24h window sits half an hour from
    resetting the streak every night, and the first late start fires a spurious RED."""
    log = tmp_path / "bf.log"
    log.write_text("2026-08-02T04:10:06Z  ALL FRESH       all 5 blocks match sqlite\n")
    monkeypatch.setattr(sg, "BLOCKFRESH", str(log))
    assert sg.read_blockfresh(datetime(2026, 8, 3, 3, 40, tzinfo=timezone.utc))[0] == "FRESH"
    # An hour late on both sides is still one cycle; a whole missed cycle is not.
    assert sg.read_blockfresh(datetime(2026, 8, 3, 5, 30, tzinfo=timezone.utc))[0] == "FRESH"
    assert sg.read_blockfresh(datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc))[0] == "STALE"


def test_blockfresh_drift_is_red(sg, tmp_path, monkeypatch):
    now = datetime(2026, 8, 3, 4, 30, tzinfo=timezone.utc)
    log = tmp_path / "bf.log"
    log.write_text("2026-08-03T04:10:05Z  OBSERVED DRIFT  drifted: goals\n")
    monkeypatch.setattr(sg, "BLOCKFRESH", str(log))
    assert sg.read_blockfresh(now)[0] == "DRIFT"


def test_blockfresh_reads_the_last_line_not_the_first(sg, tmp_path, monkeypatch):
    """The 5.5 job writes DRIFT then re-packs and writes ALL FRESH; only the last counts."""
    now = datetime(2026, 8, 3, 4, 30, tzinfo=timezone.utc)
    log = tmp_path / "bf.log"
    log.write_text("2026-08-03T04:10:05Z  OBSERVED DRIFT  drifted: goals\n"
                   "2026-08-03T04:10:06Z  ALL FRESH       all 5 blocks match sqlite\n")
    monkeypatch.setattr(sg, "BLOCKFRESH", str(log))
    assert sg.read_blockfresh(now)[0] == "FRESH"


def test_bench_stale_by_mtime(sg, tmp_path, monkeypatch):
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"titled_hits": 11, "comparable": 11, "fresh": 1}))
    old = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()
    os.utime(bench, (old, old))
    monkeypatch.setattr(sg, "BENCH", str(bench))
    assert sg.read_bench(datetime.now(timezone.utc))[0] == "STALE"


def test_bench_scored_on_comparable_not_twelve(sg, tmp_path, monkeypatch):
    """The cron's alert clause says `top-1 < 12/12`; the script reports 11/11 comparable.

    Read literally that clause has been tripped three nights running by a frozen-snapshot
    row working as designed — an alert only satisfiable by having read the plan.
    """
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"titled_hits": 11, "comparable": 11, "n": 12, "fresh": 1}))
    monkeypatch.setattr(sg, "BENCH", str(bench))
    state, summary, _ = sg.read_bench(datetime.now(timezone.utc))
    assert state == "PASS"
    assert summary["fresh_miss"] == 1


def test_bench_real_miss_is_red(sg, tmp_path, monkeypatch):
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"titled_hits": 9, "comparable": 11, "fresh": 1}))
    monkeypatch.setattr(sg, "BENCH", str(bench))
    assert sg.read_bench(datetime.now(timezone.utc))[0] == "MISS"


def test_absent_bench_is_insufficient(sg, tmp_path, monkeypatch):
    monkeypatch.setattr(sg, "BENCH", str(tmp_path / "nope.json"))
    assert sg.read_bench(datetime.now(timezone.utc))[0] == "MISSING"


# ------------------------------------------------------- the ledger

def test_gap_days_are_recorded_not_stepped_over(sg):
    assert sg.missed_days("2026-08-03", "2026-08-07") == [
        "2026-08-04", "2026-08-05", "2026-08-06"]
    assert sg.missed_days("2026-08-03", "2026-08-04") == []
    assert sg.missed_days(None, "2026-08-04") == []


@pytest.mark.parametrize("breaker", ["RED", "INSUFFICIENT", "MISSING"])
def test_every_non_green_breaks_the_streak(sg, breaker):
    rows = [{"verdict": "GREEN"}] * 4 + [{"verdict": breaker}] + [{"verdict": "GREEN"}] * 2
    assert sg.streak(rows) == 2, "a day we could not measure is not a green day"


def test_streak_is_walked_from_the_newest_row(sg):
    assert sg.streak([{"verdict": "GREEN"}] * 7) == 7
    assert sg.streak([]) == 0
    assert sg.streak([{"verdict": "GREEN"}, {"verdict": "RED"}]) == 0


def test_torn_ledger_line_is_a_gap_not_a_guess(sg, tmp_path, monkeypatch):
    ledger = tmp_path / "soak.jsonl"
    ledger.write_text(json.dumps({"date": "2026-08-01", "verdict": "GREEN"}) + "\n"
                      + '{"date": "2026-08-02", "verdict": "GRE\n'
                      + json.dumps({"date": "2026-08-03", "verdict": "GREEN"}) + "\n")
    monkeypatch.setattr(sg, "LEDGER", str(ledger))
    rows = sg.load_ledger()
    assert [r["verdict"] for r in rows] == ["GREEN", "MISSING", "GREEN"]
    assert sg.streak(rows) == 1


def test_record_refuses_a_second_row_for_the_same_day(sg, tmp_path, monkeypatch, capsys):
    today = datetime.now(timezone.utc).date().isoformat()
    ledger = tmp_path / "soak.jsonl"
    ledger.write_text(json.dumps({"date": today, "verdict": "GREEN",
                                  "measured_at": datetime.now(timezone.utc).isoformat()}) + "\n")
    monkeypatch.setattr(sg, "LEDGER", str(ledger))
    assert sg.main(["record"]) == 3
    assert "already recorded" in capsys.readouterr().out
    assert len(ledger.read_text().strip().splitlines()) == 1


# ------------------------------------------------------- end to end

def _stage(sg, tmp_path, monkeypatch, *, failures=0, fresh=True, hits=11, comparable=11):
    now = datetime.now(timezone.utc)
    lines = [_echo(now.isoformat())]
    lines += [_event(now.isoformat())] * failures
    (tmp_path / "luke.log").write_text("\n".join(lines) + "\n")
    monkeypatch.delenv("LUKE_HOME", raising=False)
    monkeypatch.setattr(sg, "LOG_ROOTS", [str(tmp_path)])

    bf = tmp_path / "bf.log"
    state = "ALL FRESH       all 5 blocks match sqlite" if fresh else "OBSERVED DRIFT  drifted: goals"
    bf.write_text(f"{now.isoformat()}  {state}\n")
    monkeypatch.setattr(sg, "BLOCKFRESH", str(bf))

    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"titled_hits": hits, "comparable": comparable, "fresh": 1}))
    monkeypatch.setattr(sg, "BENCH", str(bench))
    monkeypatch.setattr(sg, "LEDGER", str(tmp_path / "soak.jsonl"))


def test_all_evidence_present_and_passing_is_green(sg, tmp_path, monkeypatch):
    _stage(sg, tmp_path, monkeypatch)
    assert sg.main(["record"]) == 0
    row = json.loads(open(tmp_path / "soak.jsonl").read().strip().splitlines()[-1])
    assert row["verdict"] == "GREEN" and row["streak_after"] == 1


def test_a_real_failure_is_red(sg, tmp_path, monkeypatch):
    _stage(sg, tmp_path, monkeypatch, failures=1)
    assert sg.main(["record"]) == 1


def test_red_outranks_insufficient(sg, tmp_path, monkeypatch):
    """Both are non-green, but only one is something to act on — don't lose the signal."""
    _stage(sg, tmp_path, monkeypatch, failures=1)
    monkeypatch.setattr(sg, "BENCH", str(tmp_path / "gone.json"))
    assert sg.main(["record"]) == 1
    row = json.loads(open(tmp_path / "soak.jsonl").read().strip().splitlines()[-1])
    assert row["verdict"] == "RED"
    assert any("letta_search_failed" in r for r in row["reasons"])
    assert any("bench" in r for r in row["reasons"]), "the insufficiency is still recorded"


def test_missing_evidence_alone_is_insufficient(sg, tmp_path, monkeypatch):
    _stage(sg, tmp_path, monkeypatch)
    monkeypatch.setattr(sg, "BENCH", str(tmp_path / "gone.json"))
    assert sg.main(["record"]) == 2


def test_a_skipped_night_lands_in_the_ledger_as_missing(sg, tmp_path, monkeypatch):
    """The failure the hand-incremented counter could not see: nothing ran, nothing said so."""
    _stage(sg, tmp_path, monkeypatch)
    today = datetime.now(timezone.utc).date()
    old = (today - timedelta(days=3)).isoformat()
    (tmp_path / "soak.jsonl").write_text(json.dumps({
        "date": old, "verdict": "GREEN",
        "measured_at": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}) + "\n")

    assert sg.main(["record"]) == 0
    rows = [json.loads(x) for x in open(tmp_path / "soak.jsonl").read().strip().splitlines()]
    assert [r["date"] for r in rows] == [
        old, (today - timedelta(days=2)).isoformat(),
        (today - timedelta(days=1)).isoformat(), today.isoformat()]
    assert [r["verdict"] for r in rows[1:3]] == ["MISSING", "MISSING"]
    assert rows[-1]["streak_after"] == 1, "the streak restarts at the gap, it does not jump it"
