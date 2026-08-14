"""deploy.sh's drain + self-kill guards, exercised for real.

These are the two guards that only run during a deploy, which is exactly when
nobody is watching and a mistake is expensive: a wrong "idle" reading kills a
live turn, a wrong "busy" reading hangs the deploy for DRAIN_TIMEOUT.  Reading
the script is not evidence that it works, so each case sources deploy.sh with
``DEPLOY_SH_SOURCE_ONLY=1`` and runs the functions against synthetic
heartbeat/inflight files.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

DEPLOY_SH = Path(__file__).resolve().parents[1] / "deploy.sh"


def run_snippet(luke_dir: Path, snippet: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    """Source deploy.sh's definitions, then run `snippet`. Returns (rc, output)."""
    script = f'DEPLOY_SH_SOURCE_ONLY=1 source "{DEPLOY_SH}"\n{snippet}\n'
    full_env = {**os.environ, "LUKE_DIR": str(luke_dir), **(env or {})}
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=full_env,
        timeout=60,
    )
    return proc.returncode, proc.stdout + proc.stderr


def write_state(
    luke_dir: Path,
    *,
    hb_pid: int | None = None,
    hb_age: int = 0,
    hb_status: str = "tick",
    inflight: tuple[int, int, int] | None = None,
) -> None:
    """inflight is (pid, user_runs, auto_runs); None writes no marker at all."""
    pid = os.getpid() if hb_pid is None else hb_pid
    (luke_dir / "heartbeat").write_text(f"{int(time.time()) - hb_age} {pid} {hb_status}\n")
    if inflight is not None:
        f_pid, users, autos = inflight
        (luke_dir / "inflight").write_text(f"{int(time.time())} {f_pid} {users} {autos}\n")


@pytest.fixture
def luke_dir(tmp_path: Path) -> Path:
    (tmp_path / "luke.log").write_text("")
    return tmp_path


def test_source_only_mode_defines_the_guards_without_deploying(luke_dir):
    """If sourcing ran the pipeline, these tests would deploy the repo."""
    rc, out = run_snippet(luke_dir, 'declare -F wait_for_idle inflight_busy inside_luke_tree luke_is_live')
    assert rc == 0
    for fn in ("wait_for_idle", "inflight_busy", "inside_luke_tree", "luke_is_live"):
        assert fn in out
    assert "Step 1/5" not in out


# ─── luke_is_live ────────────────────────────────────────────────────────────


def test_live_when_heartbeat_is_fresh_and_pid_exists(luke_dir):
    write_state(luke_dir)
    rc, _ = run_snippet(luke_dir, "luke_is_live")
    assert rc == 0


def test_not_live_when_heartbeat_is_stale(luke_dir):
    write_state(luke_dir, hb_age=600)
    rc, _ = run_snippet(luke_dir, "luke_is_live")
    assert rc == 1


def test_not_live_when_the_pid_is_gone(luke_dir):
    write_state(luke_dir, hb_pid=999_999)
    rc, _ = run_snippet(luke_dir, "luke_is_live")
    assert rc == 1


def test_not_live_when_heartbeat_is_missing_or_junk(luke_dir):
    rc, _ = run_snippet(luke_dir, "luke_is_live")
    assert rc == 1
    (luke_dir / "heartbeat").write_text("garbage\n")
    rc, _ = run_snippet(luke_dir, "luke_is_live")
    assert rc == 1


# ─── inflight_busy ───────────────────────────────────────────────────────────


def test_busy_with_a_user_turn_open(luke_dir):
    write_state(luke_dir, inflight=(os.getpid(), 1, 0))
    rc, _ = run_snippet(luke_dir, "inflight_busy")
    assert rc == 0


def test_idle_with_no_runs(luke_dir):
    write_state(luke_dir, inflight=(os.getpid(), 0, 0))
    rc, _ = run_snippet(luke_dir, "inflight_busy")
    assert rc == 1


def test_autonomous_runs_are_drained_by_default(luke_dir):
    """THE regression. This used to assert the opposite, and it cost three nights.

    ``f580ac19`` — the daily self-reflection cron, Filipe's standing directive —
    starts at 00:00 and runs 4-15 minutes.  On 11, 13 and 14 Aug 2026 a deploy
    launched around 00:05 read this marker, saw ``users=0``, logged "Luke is idle
    and the conversation is quiet — safe to restart" and killed it: three of four
    nights, 37-47 tool turns each, the last one 23,361 output tokens in with five
    files already edited.  The old default was reasoned from a 30-second mail
    scan and generalised to everything the scheduler runs.
    """
    write_state(luke_dir, inflight=(os.getpid(), 0, 3))
    rc, _ = run_snippet(luke_dir, "inflight_busy")
    assert rc == 0, "an autonomous run in flight must block a restart"


def test_autonomous_draining_can_still_be_waived_explicitly(luke_dir):
    """The escape hatch has to exist, or an urgent deploy has no way through."""
    write_state(luke_dir, inflight=(os.getpid(), 0, 3))
    rc, _ = run_snippet(luke_dir, "DRAIN_AUTONOMOUS=0; inflight_busy")
    assert rc == 1


def test_a_genuinely_idle_luke_is_still_not_waited_on(luke_dir):
    """The new default must not make every deploy wait DRAIN_TIMEOUT."""
    write_state(luke_dir, inflight=(os.getpid(), 0, 0))
    rc, _ = run_snippet(luke_dir, "inflight_busy")
    assert rc == 1


def test_the_autonomous_reason_names_the_count(luke_dir):
    """DRAIN_REASON is what lands in the deploy log; "busy" alone is not a clue."""
    write_state(luke_dir, inflight=(os.getpid(), 0, 2))
    rc, out = run_snippet(luke_dir, 'inflight_busy && echo "REASON=$DRAIN_REASON"')
    assert rc == 0
    assert "REASON=2 autonomous run(s)" in out


def test_a_user_turn_still_outranks_the_autonomous_reason(luke_dir):
    """With both open the message should name the user turn — it is the costlier."""
    write_state(luke_dir, inflight=(os.getpid(), 1, 2))
    rc, out = run_snippet(luke_dir, 'inflight_busy && echo "REASON=$DRAIN_REASON"')
    assert rc == 0
    assert "REASON=1 user turn(s)" in out


def test_dead_process_marker_is_ignored_for_autonomous_runs_too(luke_dir):
    """The pid gate is what stops the new default hanging every deploy forever.

    A process SIGKILLed mid-cron leaves ``autos=1`` on disk permanently. Under
    the old default that was harmless because autonomous counts were ignored;
    now it would wait out DRAIN_TIMEOUT on every single deploy.
    """
    write_state(luke_dir, hb_pid=os.getpid(), inflight=(os.getpid() + 12345, 0, 1))
    rc, _ = run_snippet(luke_dir, "inflight_busy")
    assert rc == 1


def test_marker_from_a_dead_process_is_ignored(luke_dir):
    """THE case this pid check exists for: a process SIGKILLed mid-turn leaves
    `users=1` on disk forever. Without the pid gate every future deploy would
    wait out the full DRAIN_TIMEOUT for a turn that died days ago."""
    write_state(luke_dir, hb_pid=os.getpid(), inflight=(os.getpid() + 12345, 1, 0))
    rc, _ = run_snippet(luke_dir, "inflight_busy")
    assert rc == 1


def test_missing_or_malformed_marker_reads_as_idle(luke_dir):
    """Fail open: an old build with no inflight.py must still be deployable."""
    write_state(luke_dir)
    rc, _ = run_snippet(luke_dir, "inflight_busy")
    assert rc == 1
    (luke_dir / "inflight").write_text("not a marker\n")
    rc, _ = run_snippet(luke_dir, "inflight_busy")
    assert rc == 1


# ─── wait_for_idle ───────────────────────────────────────────────────────────


def test_wait_returns_immediately_when_idle(luke_dir):
    write_state(luke_dir, inflight=(os.getpid(), 0, 0))
    started = time.monotonic()
    rc, out = run_snippet(luke_dir, "wait_for_idle")
    assert rc == 0
    assert "idle" in out
    assert time.monotonic() - started < 5


def test_wait_gives_up_after_the_timeout_and_says_so(luke_dir):
    """Refusing to restart would be worse than a lost turn — the new code would
    never ship. But it must be loud, not silent like the old behaviour."""
    write_state(luke_dir, inflight=(os.getpid(), 1, 0))
    rc, out = run_snippet(luke_dir, "wait_for_idle", env={"DRAIN_TIMEOUT": "3"})
    assert rc == 0
    assert "restarting anyway" in out
    assert "will be lost" in out


def test_wait_notices_the_turn_finishing(luke_dir):
    write_state(luke_dir, inflight=(os.getpid(), 1, 0))
    snippet = (
        f'( sleep 4; echo "$(date +%s) {os.getpid()} 0 0" > "{luke_dir}/inflight" ) &\n'
        "wait_for_idle"
    )
    rc, out = run_snippet(luke_dir, snippet, env={"DRAIN_TIMEOUT": "60"})
    assert rc == 0
    assert "Idle after" in out
    assert "restarting anyway" not in out


def test_no_drain_flag_skips_the_wait(luke_dir):
    write_state(luke_dir, inflight=(os.getpid(), 1, 0))
    rc, out = run_snippet(luke_dir, "DRAIN=0; wait_for_idle", env={"DRAIN_TIMEOUT": "600"})
    assert rc == 0
    assert "without waiting" in out


def test_a_dead_luke_is_not_waited_on(luke_dir):
    """If the service is hung, the marker is meaningless and the restart is the
    whole point. Blocking here would turn a crash into an outage."""
    write_state(luke_dir, hb_age=600, inflight=(os.getpid(), 1, 0))
    rc, out = run_snippet(luke_dir, "wait_for_idle", env={"DRAIN_TIMEOUT": "600"})
    assert rc == 0
    assert "down or hung" in out


# ─── inside_luke_tree ────────────────────────────────────────────────────────


def test_detects_being_a_descendant_of_the_live_process(luke_dir):
    """The heartbeat pid is our own parent here, which is exactly the shape of an
    autonomous session shelling out to deploy.sh."""
    write_state(luke_dir, hb_pid=os.getpid())
    rc, _ = run_snippet(luke_dir, "inside_luke_tree")
    assert rc == 0


def test_not_inside_when_the_pid_is_unrelated(luke_dir):
    write_state(luke_dir, hb_pid=1)
    rc, _ = run_snippet(luke_dir, "inside_luke_tree")
    assert rc == 1


def test_not_inside_when_there_is_no_heartbeat(luke_dir):
    rc, _ = run_snippet(luke_dir, "inside_luke_tree")
    assert rc == 1


# ─── flag parsing ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "expect"),
    [
        ([], "branch= drain=1 auto=1"),
        (["my-feature"], "branch=my-feature drain=1 auto=1"),
        (["--no-drain"], "branch= drain=0 auto=1"),
        (["my-feature", "--no-drain"], "branch=my-feature drain=0 auto=1"),
        (["--drain-auto", "my-feature"], "branch=my-feature drain=1 auto=1"),
        (["--no-drain-auto"], "branch= drain=1 auto=0"),
        (["--no-drain-auto", "my-feature"], "branch=my-feature drain=1 auto=0"),
        # Both spellings present: last one wins, same as every other flag here.
        (["--drain-auto", "--no-drain-auto"], "branch= drain=1 auto=0"),
    ],
)
def test_flags_and_branch_parse_in_any_order(luke_dir, argv, expect):
    """The branch used to be `$1`, so `deploy.sh --no-drain` would have tried to
    merge a branch named '--no-drain'."""
    args = " ".join(f'"{a}"' for a in argv)
    script = (
        f'set -- {args}\n'
        f'DEPLOY_SH_SOURCE_ONLY=1 source "{DEPLOY_SH}"\n'
        'echo "branch=$FEATURE_BRANCH drain=$DRAIN auto=$DRAIN_AUTONOMOUS"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "LUKE_DIR": str(luke_dir)},
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert expect in proc.stdout


def test_unknown_flag_is_rejected(luke_dir):
    script = f'set -- "--bogus"\nDEPLOY_SH_SOURCE_ONLY=1 source "{DEPLOY_SH}"\n'
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "LUKE_DIR": str(luke_dir)},
        timeout=60,
    )
    assert proc.returncode == 2
    assert "unknown flag" in proc.stderr


# ─── the detached runner ─────────────────────────────────────────────────────
#
# Every case here comes from the guard's first live run, which spun 43 times in
# under a minute: `launchctl submit` sets KeepAlive, and the submitted command
# died instantly on a relative path it could not resolve from launchd's cwd.


def build_runner(luke_dir: Path, *args: str) -> str:
    runner = luke_dir / "runner.sh"
    quoted = " ".join(f'"{a}"' for a in args)
    rc, out = run_snippet(
        luke_dir,
        f'build_detached_runner "com.luke.deploy.777" "{luke_dir}/x.log" "{runner}" {quoted}',
    )
    assert rc == 0, out
    return runner.read_text()


def test_runner_removes_its_own_label(luke_dir):
    """THE bug. Without this line launchd relaunches the job forever, because
    `launchctl submit` creates it with KeepAlive on."""
    body = build_runner(luke_dir)
    assert "launchctl remove com.luke.deploy.777" in body
    # …and unconditionally: it must run after a FAILED deploy too, which is the
    # case that actually loops.
    assert body.index("rc=$?") < body.index("launchctl remove")


def test_runner_invokes_deploy_by_absolute_path(luke_dir):
    """launchd does not inherit our cwd. The 43 failures were all
    `./deploy.sh: No such file or directory`."""
    body = build_runner(luke_dir)
    assert str(DEPLOY_SH) in body
    assert "./deploy.sh" not in body


def test_runner_marks_itself_detached_to_stop_infinite_recursion(luke_dir):
    """Without LUKE_DEPLOY_DETACHED the re-exec'd copy would detect the tree
    again and submit another job, and so on."""
    assert "export LUKE_DEPLOY_DETACHED=1" in build_runner(luke_dir)


def test_runner_carries_the_callers_path(luke_dir):
    """launchd hands a submitted job a bare PATH. The second live run of this
    guard died on `uv not found` in step 1 for exactly this reason."""
    body = build_runner(luke_dir)
    assert "export PATH=" in body
    assert "uv" in subprocess.run(
        ["bash", "-c", f'{[l for l in body.splitlines() if l.startswith("export PATH=")][0]}; command -v uv'],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout


def test_runner_forwards_arguments(luke_dir):
    body = build_runner(luke_dir, "my-feature", "--no-drain")
    assert "my-feature" in body
    assert "--no-drain" in body


def test_runner_quotes_hostile_arguments(luke_dir):
    """A branch name is attacker-adjacent input at best and a typo at worst;
    it is interpolated into a generated shell script."""
    body = build_runner(luke_dir, "; touch /tmp/pwned #")
    assert "; touch /tmp/pwned #\n" not in body
    rc, _ = run_snippet(luke_dir, f'bash -n "{luke_dir}/runner.sh"')
    assert rc == 0


def test_runner_preserves_the_deploy_exit_code(luke_dir):
    body = build_runner(luke_dir)
    assert "exit $rc" in body


def test_runner_is_executable_and_valid_bash(luke_dir):
    build_runner(luke_dir)
    runner = luke_dir / "runner.sh"
    assert os.access(runner, os.X_OK)
    assert subprocess.run(["bash", "-n", str(runner)], timeout=30).returncode == 0


def test_runner_runs_end_to_end_against_a_stub_deploy(luke_dir, monkeypatch):
    """Execute the generated runner for real with a stubbed deploy.sh and a
    stubbed launchctl, and assert both the exit code and the label removal."""
    stub_bin = luke_dir / "bin"
    stub_bin.mkdir()
    (stub_bin / "launchctl").write_text(
        f'#!/bin/bash\necho "launchctl $*" >> "{luke_dir}/launchctl.calls"\n'
    )
    (stub_bin / "launchctl").chmod(0o755)

    body = build_runner(luke_dir).replace(str(DEPLOY_SH), str(luke_dir / "fake-deploy.sh"))
    # The runner pins PATH deliberately, so the stub launchctl has to go in front
    # of it rather than into the environment we hand the subprocess.
    body = "\n".join(
        f'export PATH="{stub_bin}:$PATH"' if line.startswith("export PATH=") else line
        for line in body.splitlines()
    )
    (luke_dir / "fake-deploy.sh").write_text('#!/bin/bash\necho "deploy ran: $*"\nexit 7\n')
    (luke_dir / "runner.sh").write_text(body)
    (luke_dir / "runner.sh").chmod(0o755)

    proc = subprocess.run(
        ["bash", str(luke_dir / "runner.sh")],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{stub_bin}:{os.environ['PATH']}"},
        timeout=60,
    )
    assert proc.returncode == 7, proc.stdout + proc.stderr
    assert "deploy ran" in proc.stdout
    assert "remove com.luke.deploy.777" in (luke_dir / "launchctl.calls").read_text()


def test_exit_timeout_is_set_in_the_plist():
    """launchd's 20s default SIGKILLs the process mid-drain. deploy.sh waits for
    idle, but every other restart path (watchdog, manual kickstart) relies on this."""
    plist = (Path(__file__).resolve().parents[1] / "com.luke.plist").read_text()
    assert "<key>ExitTimeOut</key>" in plist


# ─── conversation_is_live ────────────────────────────────────────────────────
# Failure mode (c), 2026-08-11. inflight_busy only answers "is a turn EXECUTING
# right now?", and the pause between his message and the reply is idle by that
# measure. So the drain announced "safe to restart" and killed the exchange
# between turns; the restart wiped the session table and the next message
# arrived with no transcript. Five times on 10 Aug, the last at 01:06.


def write_messages_db(luke_dir: Path, ages_minutes: list[float] | None) -> None:
    """Build a synthetic luke.db. `None` writes no database at all.

    Column is `ts`, not `timestamp` — the real schema, confirmed against
    `q.sh --schema messages`. A test that invents the column would pass while
    the gate silently returned "cannot tell" forever.
    """
    if ages_minutes is None:
        return
    import sqlite3
    from datetime import UTC, datetime, timedelta

    conn = sqlite3.connect(luke_dir / "luke.db")
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, sender TEXT, content TEXT, ts TEXT)")
    now = datetime.now(UTC)
    for i, age in enumerate(ages_minutes):
        ts = (now - timedelta(minutes=age)).isoformat()
        conn.execute("INSERT INTO messages (sender, content, ts) VALUES (?,?,?)", ("Filipe", "x", ts))
    conn.commit()
    conn.close()


@pytest.mark.parametrize(
    "ages,quiet_min,expect_live",
    [
        ([1.0], 10, True),        # mid-exchange — the case that broke
        ([3.0], 10, True),        # still inside the window
        ([30.0], 10, False),      # long quiet — deploy freely
        ([1.0], 0, False),        # explicitly disabled
        ([], 10, False),          # empty table ⇒ cannot tell ⇒ must not block
        (None, 10, False),        # no db at all ⇒ cannot tell ⇒ must not block
        ([-120.0], 10, False),    # clock skew into the future ⇒ must not block
        ([45.0, 2.0], 10, True),  # MAX(ts) wins, not row order
    ],
)
def test_conversation_is_live(luke_dir, ages, quiet_min, expect_live):
    write_messages_db(luke_dir, ages)
    rc, out = run_snippet(
        luke_dir,
        'conversation_is_live && echo LIVE || echo QUIET',
        env={"CONVERSATION_QUIET_MIN": str(quiet_min)},
    )
    assert rc == 0, out
    assert ("LIVE" if expect_live else "QUIET") in out, out


def test_drain_blocks_between_turns_even_with_nothing_in_flight(luke_dir):
    """THE REGRESSION. No turn executing, but he sent something a minute ago.

    Old behaviour: inflight_busy false ⇒ "Luke is idle — safe to restart".
    """
    write_state(luke_dir, inflight=(os.getpid(), 0, 0))
    write_messages_db(luke_dir, [1.0])
    rc, out = run_snippet(luke_dir, 'drain_blocked && echo BLOCKED || echo FREE')
    assert rc == 0, out
    assert "BLOCKED" in out, out
    # and the old, narrower signal genuinely reads idle — proving the gap was real
    rc2, out2 = run_snippet(luke_dir, 'inflight_busy && echo BUSY || echo IDLE')
    assert "IDLE" in out2, out2


def test_drain_frees_once_the_conversation_goes_quiet(luke_dir):
    write_state(luke_dir, inflight=(os.getpid(), 0, 0))
    write_messages_db(luke_dir, [45.0])
    rc, out = run_snippet(luke_dir, 'drain_blocked && echo BLOCKED || echo FREE')
    assert rc == 0, out
    assert "FREE" in out, out


def test_check_drain_flag_is_read_only_and_reports(luke_dir):
    """--check-drain must never deploy: no tests, no git, no restart."""
    write_state(luke_dir, inflight=(os.getpid(), 0, 0))
    write_messages_db(luke_dir, [1.0])
    proc = subprocess.run(
        ["bash", str(DEPLOY_SH), "--check-drain"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "LUKE_DIR": str(luke_dir)},
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 1, combined      # 1 = would wait
    assert "CONVERSATION OPEN" in combined, combined
    assert "Step 1/5" not in combined, combined


def test_timeout_defers_rather_than_wiping_a_live_conversation(luke_dir):
    """The hole in the first version of the fix: DRAIN_TIMEOUT expiring used to
    restart 'anyway'. For a live conversation that is just the original bug on a
    timer — talk for 15 minutes and the transcript dies regardless."""
    write_state(luke_dir, inflight=(os.getpid(), 0, 0))
    write_messages_db(luke_dir, [1.0])
    rc, out = run_snippet(
        luke_dir,
        'DEPLOY_DEFERRED=0\nwait_for_idle\necho "DEFERRED=$DEPLOY_DEFERRED"',
        env={"DRAIN_TIMEOUT": "6"},
    )
    assert rc == 0, out
    assert "DEFERRED=1" in out, out
    assert "DEFERRING the restart" in out, out
    assert "restarting anyway" not in out, out


def test_timeout_still_forces_through_a_stuck_in_flight_turn(luke_dir):
    """The deferral must NOT leak into the in-flight case: past DRAIN_TIMEOUT the
    run is being killed by its own agent timeout, so there is nothing to protect
    and a deploy that never restarts is its own failure."""
    write_state(luke_dir, inflight=(os.getpid(), 1, 0))
    write_messages_db(luke_dir, [90.0])
    rc, out = run_snippet(
        luke_dir,
        'DEPLOY_DEFERRED=0\nwait_for_idle\necho "DEFERRED=$DEPLOY_DEFERRED"',
        env={"DRAIN_TIMEOUT": "6"},
    )
    assert rc == 0, out
    assert "DEFERRED=0" in out, out
    assert "restarting anyway" in out, out
