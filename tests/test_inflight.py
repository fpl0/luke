"""In-flight marker — the signal deploy.sh drains on.

The contract these tests defend: the file must never lie in the direction that
matters.  Reading "idle" while a turn is open costs Filipe a lost answer; reading
"busy" forever after a crash costs a deploy that hangs.  Both are covered here.
"""

from __future__ import annotations

import os

import pytest

from luke import inflight


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setattr(inflight.settings, "store_dir", tmp_path)
    inflight.reset()
    return tmp_path


def _read(store) -> list[str]:
    return (store / "inflight").read_text().split()


def test_reset_writes_a_zeroed_marker(_store):
    ts, pid, users, autos = _read(_store)
    assert int(ts) > 0
    assert int(pid) == os.getpid()
    assert (users, autos) == ("0", "0")


def test_begin_and_end_round_trip(_store):
    inflight.begin(autonomous=False)
    assert _read(_store)[2:] == ["1", "0"]
    inflight.end(autonomous=False)
    assert _read(_store)[2:] == ["0", "0"]


def test_user_and_autonomous_runs_are_counted_separately(_store):
    """deploy.sh always waits out a user turn but may skip a cron — so the two
    cannot share a counter."""
    inflight.begin(autonomous=False)
    inflight.begin(autonomous=True)
    inflight.begin(autonomous=True)
    assert _read(_store)[2:] == ["1", "2"]
    inflight.end(autonomous=True)
    assert _read(_store)[2:] == ["1", "1"]


def test_counts_never_go_negative(_store):
    """An unbalanced end() (a run whose begin predates a reset) must not push the
    counter below zero — a -1 would read as idle while a real turn is open."""
    inflight.end(autonomous=False)
    inflight.end(autonomous=True)
    assert _read(_store)[2:] == ["0", "0"]
    inflight.begin(autonomous=False)
    assert _read(_store)[2:] == ["1", "0"]


def test_marker_records_the_writing_pid(_store):
    """The pid is what lets a reader discard a marker left by a killed process."""
    inflight.begin(autonomous=False)
    assert int(_read(_store)[1]) == os.getpid()


@pytest.mark.asyncio
async def test_tracked_counts_for_the_life_of_the_run(_store):
    seen: list[tuple[int, int]] = []

    @inflight.tracked
    async def fake_run(*, chat_id: str, autonomous: bool = False):
        seen.append(inflight.counts())
        return "done"

    assert await fake_run(chat_id="1") == "done"
    assert seen == [(1, 0)]
    assert inflight.counts() == (0, 0)


@pytest.mark.asyncio
async def test_tracked_reads_the_autonomous_kwarg(_store):
    seen: list[tuple[int, int]] = []

    @inflight.tracked
    async def fake_run(*, chat_id: str, autonomous: bool = False):
        seen.append(inflight.counts())

    await fake_run(chat_id="1", autonomous=True)
    assert seen == [(0, 1)]


@pytest.mark.asyncio
async def test_tracked_decrements_when_the_run_raises(_store):
    """A crashed or timed-out turn must release the marker, or the next deploy
    waits out the full DRAIN_TIMEOUT for a run that ended minutes ago."""

    @inflight.tracked
    async def boom(*, chat_id: str, autonomous: bool = False):
        raise RuntimeError("agent blew up")

    with pytest.raises(RuntimeError):
        await boom(chat_id="1")
    assert inflight.counts() == (0, 0)
    assert _read(_store)[2:] == ["0", "0"]


@pytest.mark.asyncio
async def test_tracked_decrements_on_cancellation(_store):
    """asyncio.wait_for cancels the run on agent_timeout — the most common way a
    turn ends badly. finally: must still fire."""
    import asyncio

    @inflight.tracked
    async def slow(*, chat_id: str, autonomous: bool = False):
        await asyncio.sleep(10)

    task = asyncio.create_task(slow(chat_id="1"))
    await asyncio.sleep(0)
    assert inflight.counts() == (1, 0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert inflight.counts() == (0, 0)


def test_write_failure_never_raises(_store, monkeypatch):
    """Best effort by design: losing the marker degrades to today's behaviour
    (a deploy that might interrupt a turn). Killing the turn to protect the
    marker would be strictly worse."""
    monkeypatch.setattr(inflight, "_path", lambda: (_ for _ in ()).throw(AttributeError("no store_dir")))
    inflight.begin(autonomous=False)
    inflight.end(autonomous=False)


def test_run_agent_is_wrapped():
    """The decorator has to be on the real function, not just available."""
    from luke import agent

    assert getattr(agent.run_agent, "__wrapped__", None) is not None
    assert agent.run_agent.__name__ == "run_agent"
