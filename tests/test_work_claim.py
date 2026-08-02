"""Goal-level work claims.

Two properties carry the whole design and both are falsified here rather than asserted:
a claim must DENY a peer that is genuinely still working, and it must NEVER deny when
anything goes wrong (fail-open — see the module docstring on the Aug 10 deadline).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import pytest

from luke import work_claim


@pytest.fixture()
def store(tmp_settings: Any) -> Any:
    tmp_settings.store_dir.mkdir(parents=True, exist_ok=True)
    return tmp_settings


def test_first_claim_granted_second_denied(store: Any) -> None:
    """The actual 2026-08-02 collision: two sessions, same goal, one must yield."""
    first = work_claim.claim("goal-letta-parity", holder="02:00 build")
    assert first is not None
    second = work_claim.claim("goal-letta-parity", holder="03:03 deep work")
    assert second is None, "a second live session must be denied the same goal"


def test_release_frees_the_goal(store: Any) -> None:
    first = work_claim.claim("goal-x", holder="a")
    assert first is not None
    assert work_claim.release("goal-x", first.token) is True
    second = work_claim.claim("goal-x", holder="b")
    assert second is not None, "goal must be claimable again after release"


def test_release_requires_the_holders_token(store: Any) -> None:
    first = work_claim.claim("goal-x", holder="a")
    assert first is not None
    assert work_claim.release("goal-x", "not-the-token") is False
    assert work_claim.current("goal-x") is not None, "claim must survive a bogus release"


def test_different_goals_do_not_block_each_other(store: Any) -> None:
    assert work_claim.claim("goal-a") is not None
    assert work_claim.claim("goal-b") is not None


def test_expired_claim_is_reclaimable(store: Any) -> None:
    """A crashed session must not hold a deadline-bearing goal forever."""
    first = work_claim.claim("goal-x", ttl_seconds=60, holder="crashed")
    assert first is not None
    path = work_claim._claim_path("goal-x")
    rec = json.loads(open(path).read())
    rec["expires_at"] = time.time() - 1
    open(path, "w").write(json.dumps(rec))

    assert work_claim.current("goal-x") is None, "an expired claim is not a live claim"
    assert work_claim.claim("goal-x", holder="next") is not None


def test_dead_holder_is_reclaimed_before_ttl(store: Any) -> None:
    """Don't make the next session wait out a 45-minute TTL for a process that is gone."""
    first = work_claim.claim("goal-x", ttl_seconds=99999, holder="dead")
    assert first is not None
    path = work_claim._claim_path("goal-x")
    rec = json.loads(open(path).read())
    # A pid that cannot be alive; host already matches (recorded from this machine).
    rec["pid"] = 2**31 - 1
    open(path, "w").write(json.dumps(rec))

    assert work_claim.claim("goal-x", holder="next") is not None


def test_live_holder_on_another_host_is_not_reclaimed_early(store: Any) -> None:
    """pid liveness is meaningless across hosts — only the TTL may free that claim."""
    first = work_claim.claim("goal-x", ttl_seconds=99999, holder="remote")
    assert first is not None
    path = work_claim._claim_path("goal-x")
    rec = json.loads(open(path).read())
    rec["pid"] = 2**31 - 1
    rec["host"] = "some-other-machine"
    open(path, "w").write(json.dumps(rec))

    assert work_claim.claim("goal-x", holder="next") is None


def test_corrupt_claim_file_does_not_block(store: Any) -> None:
    os.makedirs(work_claim._claims_dir(), exist_ok=True)
    open(work_claim._claim_path("goal-x"), "w").write("{not json")
    assert work_claim.claim("goal-x", holder="next") is not None


def test_fails_open_when_the_store_is_broken(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property that matters most: a bug in here must never stall a goal."""
    monkeypatch.setattr(
        work_claim.os, "makedirs", lambda *a, **k: (_ for _ in ()).throw(OSError("disk gone"))
    )
    granted = work_claim.claim("goal-letta-parity", holder="deadline work")
    assert granted is not None, "an internal error must GRANT, never deny"
    assert granted.token == "", "a fail-open grant holds no releasable token"


def test_goal_id_cannot_escape_the_claims_dir(store: Any) -> None:
    path = work_claim._claim_path("../../etc/passwd")
    assert os.path.dirname(os.path.abspath(path)) == os.path.abspath(work_claim._claims_dir())


def test_context_manager_releases_on_exception(store: Any) -> None:
    with pytest.raises(RuntimeError):
        with work_claim.claimed("goal-x", holder="a") as c:
            assert c is not None
            raise RuntimeError("session died mid-build")
    assert work_claim.current("goal-x") is None
    assert work_claim.claim("goal-x", holder="b") is not None


def test_cli_acquire_then_denies_with_exit_3(store: Any, capsys: Any) -> None:
    assert work_claim._main(["work_claim", "acquire", "goal-x", "--holder", "first"]) == 0
    # Callers take the LAST stdout line — structlog shares this stream (see module docstring).
    token = capsys.readouterr().out.strip().splitlines()[-1].strip()
    assert token and " " not in token

    assert work_claim._main(["work_claim", "acquire", "goal-x", "--holder", "second"]) == 3
    assert "HELD" in capsys.readouterr().out

    assert work_claim._main(["work_claim", "release", "goal-x", token]) == 0
    assert work_claim._main(["work_claim", "acquire", "goal-x", "--holder", "third"]) == 0


def test_cli_denies_across_separate_processes(tmp_path: Any) -> None:
    """The regression that in-process tests cannot see.

    The first cut recorded ``os.getpid()`` even from the CLI. That pid is dead the instant
    the CLI exits, so the next caller read "holder is gone", reclaimed, and the claim
    granted to everyone — a lock that locked nothing. Both claims sharing one live pytest
    process hid it completely; only two real processes expose it.
    """
    import subprocess
    import sys

    env = {
        **os.environ,
        "LUKE_DIR": str(tmp_path / "luke"),
        "PYTHONPATH": "src",
        "TELEGRAM_BOT_TOKEN": "0000000000:AAHfakeTestTokenForUnitTesting1234",
    }

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "luke.work_claim", *args],
            capture_output=True,
            text=True,
            env=env,
        )

    first = run("acquire", "goal-letta-parity", "--holder", "build cron")
    assert first.returncode == 0, first.stderr
    token = first.stdout.strip().splitlines()[-1].strip()

    second = run("acquire", "goal-letta-parity", "--holder", "deep work tick")
    assert second.returncode == 3, (
        "a separate process must be DENIED a goal another session holds; "
        f"got exit {second.returncode}: {second.stdout}"
    )

    assert run("release", "goal-letta-parity", token).returncode == 0
    assert run("acquire", "goal-letta-parity", "--holder", "next").returncode == 0


def test_cli_status_reports_free_and_held(store: Any, capsys: Any) -> None:
    assert work_claim._main(["work_claim", "status", "goal-x"]) == 0
    assert "FREE" in capsys.readouterr().out
    work_claim.claim("goal-x", holder="someone")
    assert work_claim._main(["work_claim", "status", "goal-x"]) == 0
    assert "someone" in capsys.readouterr().out


def test_ttl_outlives_the_session_it_protects(store: Any) -> None:
    """The TTL must cover the whole session ceiling.

    The first cut hardcoded 2700s from "a session runs ~30 min". That assumption
    expired inside a day (deep_work_timeout went to 5400) and the constant did not
    move — so every claim quietly stopped covering the back half of its own session:
    at minute 45 the goal reads FREE and a peer walks into the running session.
    """
    from luke.config import settings as live_settings

    assert work_claim.default_ttl_seconds() > live_settings.deep_work_timeout, (
        "a claim that expires before the session it protects re-opens the exact "
        "collision this module exists to prevent"
    )
    c = work_claim.claim("goal-x", holder="deep work")
    assert c is not None
    assert c.seconds_left > live_settings.deep_work_timeout


def test_ttl_tracks_a_raised_session_timeout(store: Any, monkeypatch: Any) -> None:
    """Negative control: raise the session ceiling and the TTL follows it."""
    from luke.config import settings as live_settings

    monkeypatch.setattr(live_settings, "deep_work_timeout", 9000.0)
    assert work_claim.default_ttl_seconds() > 9000


def test_release_with_no_token_leaves_a_peers_claim_intact(store: Any) -> None:
    """An empty token is a fail-open holder, not a master key.

    release() used to read a falsy token as "force" and unlink whatever claim file
    was present — so a session that got its claim from the fail-open path could
    delete a LIVE peer's claim on the way out, turning the defensive fallback into
    the collision the module prevents.
    """
    peer = work_claim.claim("goal-x", holder="real holder")
    assert peer is not None
    assert work_claim.release("goal-x", "") is False
    still = work_claim.current("goal-x")
    assert still is not None and still["holder"] == "real holder"
    # The real holder can still release normally.
    assert work_claim.release("goal-x", peer.token) is True
