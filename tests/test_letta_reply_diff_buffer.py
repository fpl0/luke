"""The 5.1R replay ran with an uncontrolled conversation buffer, and nothing could show it.

A Letta agent keeps an in-context message buffer across turns, and ``message_buffer_autoclear``
is False on ``luke-agent-claude`` (read off the live agent, 2026-08-02). ``cmd_run`` never
touched it. Because the 20-prompt set is frozen on purpose, every re-run replayed prompts the
arm had already answered with its own earlier answers still in context — contamination in the
direction that flatters Letta, and invisible downstream: the runs file records replies, not the
state they were produced in.

Measured, not argued: at 22:29Z the live buffer held two full passes of the set plus three
probes, and a re-drive of #3436 produced the correct OAuth-token facts while calling ZERO
tools, opening with "Checked the code before answering" when it had checked nothing that turn.

These are decision tests, not happy-path ones — the same standard the journal and provenance
guards are held to in this suite:

  * the reset happens BEFORE the first turn (resetting after turn 1 clears nothing that
    mattered) and only on ``--fresh``;
  * a resumed run does NOT reset, because there the buffer holds this run's own earlier rows,
    which is the intended within-run condition;
  * a reset that FAILS aborts the run with no runs file. Fail-open is wrong here specifically:
    a replay that continues on a dirty buffer yields a number nobody can tell from a clean one;
  * the depth the run started at is written into the artefact, so the variable is readable by
    someone who was not watching when it ran.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SET_BUILT = "2026-08-02T07:12:47.622126+00:00"


@pytest.fixture()
def rd(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "letta_reply_diff", os.path.join(REPO, "scripts", "letta_reply_diff.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["letta_reply_diff"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "LOGS", str(tmp_path))
    for name in ("SET_PATH", "RUNS_PATH", "PACK_PATH", "KEY_PATH", "PARTIAL_PATH", "VERDICT_LOG"):
        monkeypatch.setattr(mod, name, str(tmp_path / os.path.basename(getattr(mod, name))))
    return mod


@pytest.fixture()
def wired(rd, monkeypatch):
    """A 2-prompt set and a stubbed turn driver, with buffer calls recorded in order.

    The order log is the point: it is the only way to assert the reset landed before the
    first turn rather than merely at some point during the run.
    """
    from luke import letta_agent

    events: list[str] = []
    state = {"depth": 41, "reset_result": 1}

    def fake_depth(*a, **kw):
        events.append("depth")
        return state["depth"]

    def fake_reset(*a, **kw):
        events.append("reset")
        return state["reset_result"]

    def fake_turn(body, inject_recall=True):
        events.append("turn")
        return {"reply": "a real answer " * 8, "seconds": 1.0, "tools": []}

    monkeypatch.setattr(letta_agent, "agent_buffer_depth", fake_depth)
    monkeypatch.setattr(letta_agent, "reset_agent_messages", fake_reset)
    monkeypatch.setattr(letta_agent, "drive_letta_turn", fake_turn)
    monkeypatch.setattr(letta_agent, "compose_letta_turn_input",
                        lambda prompt, as_of=None: f"<mem id=1>ctx</mem>\n{prompt}")

    prompts = [{"msg_id": i, "ts": "2026-07-01T10:00:00+00:00", "prompt": f"p{i}",
                "bucket": "b", "sdk_reply": "sdk answer " * 5, "score": 1,
                "prior_context": []} for i in range(2)]
    with open(rd.SET_PATH, "w") as f:
        json.dump({"built": SET_BUILT, "prompts": prompts}, f)
    return rd, events, state


def test_fresh_run_clears_the_buffer_before_the_first_turn(wired):
    """Order is the assertion. A reset that lands after turn 1 has cleared nothing."""
    rd, events, _ = wired
    rd.cmd_run(fresh=True)

    assert "reset" in events, "a --fresh replay ran without clearing the conversation buffer"
    assert events.index("reset") < events.index("turn"), (
        f"reset must precede the first turn, got {events}"
    )


def test_a_resumed_run_does_not_clear_the_buffer(wired):
    """Mid-run the buffer holds this run's own earlier rows — that is the intended state."""
    rd, events, _ = wired
    rd.cmd_run(fresh=False)

    assert "reset" not in events, "a resumed run cleared the buffer and lost its own earlier turns"
    assert "turn" in events


def test_a_failed_reset_aborts_the_run_and_writes_no_artefact(wired):
    """Fail-open here would produce a dirty measurement that reads exactly like a clean one."""
    rd, events, state = wired
    state["reset_result"] = None

    with pytest.raises(SystemExit) as e:
        rd.cmd_run(fresh=True)

    assert e.value.code == 1
    assert "turn" not in events, "turns were driven after the buffer reset failed"
    assert not os.path.exists(rd.RUNS_PATH), "a run with an uncontrolled buffer left an artefact"


def test_the_runs_file_records_the_buffer_depth_it_started_at(wired):
    """Without this the contamination is unreadable after the fact — which is how it survived."""
    rd, _, state = wired
    state["reset_result"] = 1
    rd.cmd_run(fresh=True)

    runs = json.load(open(rd.RUNS_PATH))
    assert "buffer_at_start" in runs, "the run's buffer state is absent from its own artefact"
    assert runs["buffer_at_start"] == 1

    # And a dirty (resumed) run records the dirt rather than hiding it.
    os.remove(rd.RUNS_PATH)
    state["depth"] = 87
    rd.cmd_run(fresh=False)
    assert json.load(open(rd.RUNS_PATH))["buffer_at_start"] == 87


# ---------------------------------------------------------------------------
# The transport half — asserted against the wire, not against a mock of ourselves
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_buffer_depth_reads_the_in_context_ids_not_the_message_history(monkeypatch):
    """``/messages`` pages the whole persisted history — it returned 214 rows for an agent
    whose live context held 1. Reading that number would have made every reset look inert."""
    from luke import letta_agent

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return _Resp({"message_ids": ["m1", "m2", "m3"]})

    monkeypatch.setattr(letta_agent.urllib.request, "urlopen", fake_urlopen)
    assert letta_agent.agent_buffer_depth("agent-x") == 3
    assert seen["url"].endswith("/v1/agents/agent-x")
    assert "/messages" not in seen["url"]


def test_reset_patches_reset_messages_without_repopulating_defaults(monkeypatch):
    """``add_default_initial_messages=True`` would refill the buffer with seed turns — a
    reset that leaves messages behind, which is worse than not resetting because it looks done."""
    from luke import letta_agent

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["body"] = json.loads(req.data.decode())
        return _Resp({"message_ids": ["system"]})

    monkeypatch.setattr(letta_agent.urllib.request, "urlopen", fake_urlopen)
    assert letta_agent.reset_agent_messages("agent-x") == 1
    assert seen["url"].endswith("/v1/agents/agent-x/reset-messages")
    assert seen["method"] == "PATCH"
    assert seen["body"] == {"add_default_initial_messages": False}


def test_reset_returns_none_on_transport_failure_rather_than_raising(monkeypatch):
    """The caller decides what a failed reset means; here it means abort, but a scripted
    caller must be able to see the failure instead of catching an exception it didn't expect."""
    from luke import letta_agent

    def boom(req, timeout=None):
        raise OSError("bridge down")

    monkeypatch.setattr(letta_agent.urllib.request, "urlopen", boom)
    assert letta_agent.reset_agent_messages("agent-x") is None
    assert letta_agent.agent_buffer_depth("agent-x") is None
