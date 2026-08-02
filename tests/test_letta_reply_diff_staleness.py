"""A frozen set makes every ``set_built`` guard inert — these are the guards that replace them.

The 5.1R pipeline has three provenance checks, and all three answer the same question: which
HARNESS produced these replies? They read ``set_built``. That worked while the set was being
rebuilt. Then the plan froze the set on purpose, so a fix shipped against a red gate could be
re-measured against the same 20 prompts — and at that moment ``set_built`` stopped
distinguishing anything. The 13:07Z replies, the 13:31Z replies and tonight's replies all carry
one identical stamp.

Concretely, on 2026-08-02 the pre-guardrail 13:07Z runs file sat on disk and passed both guards
as current. What stood between it and a judged verdict was a sentence of prose in a cron prompt
saying "the runs file on disk is pre-guardrail; --fresh is correct here". Prose is not a guard.

``run_id`` — a content hash of the replies — asks the question ``set_built`` no longer can:
which RUN is this? The three failure modes it closes, all silent, all reporting a stale number
in a shape that looks completely current:

  * `pack` re-judging replies that already produced a verdict;
  * `score` unblinding with a key packed from a different replay;
  * `score` reading a judgments file written about an earlier replay — the last hop, and until
    now the only one with no check at all.

Every guard is negative-controlled: the test asserts it fires, and that the same input without
the stale condition does not fire it. Same reason as the cause tests — a green happy path over
an inert guard is what ``work_claim`` already taught, where thirteen unit tests passed while
the lock granted itself to everybody.
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


def _rows(reply="letta answer"):
    return [{"msg_id": i, "invalid": None, "sdk_reply": "sdk answer", "letta_reply": reply,
             "bucket": "factual", "ts": "2026-07-20T10:00:00+00:00", "prompt": f"prompt {i}"}
            for i in range(1, 21)]


def _write_set(rd):
    with open(rd.SET_PATH, "w") as f:
        json.dump({"built": SET_BUILT,
                   "prompts": [{"msg_id": i} for i in range(1, 21)]}, f)


def _write_runs(rd, rows):
    with open(rd.RUNS_PATH, "w") as f:
        json.dump({"ran": "2026-08-02T19:00:00+00:00", "set_built": SET_BUILT,
                   "run_id": rd._run_id(SET_BUILT, rows), "rows": rows}, f)
    return rd._run_id(SET_BUILT, rows)


# --- run_id is an identity of the replies, not of the set ------------------------------------

def test_run_id_separates_two_replays_of_the_same_frozen_set(rd):
    """The whole premise: same set, different answers, and set_built cannot see the difference."""
    a, b = _rows("letta answer"), _rows("letta answer, post-guardrail")
    assert rd._run_id(SET_BUILT, a) != rd._run_id(SET_BUILT, b)
    # ...while the stamp the existing guards compare is byte-identical across both.
    assert SET_BUILT == SET_BUILT


def test_run_id_is_stable_for_identical_replies(rd):
    """Scoring judgments against byte-identical answers is harmless, so it must not be blocked."""
    assert rd._run_id(SET_BUILT, _rows()) == rd._run_id(SET_BUILT, list(reversed(_rows())))


# --- pack refuses to re-judge a run that already has a verdict -------------------------------

def test_pack_refuses_a_run_that_was_already_scored(rd, monkeypatch, capsys):
    _write_set(rd)
    run_id = _write_runs(rd, _rows())
    with open(rd.VERDICT_LOG, "w") as f:
        f.write(f"5.1R 2026-08-02T13:07:00+00:00 run={run_id} — ... => FAIL\n")
    monkeypatch.setattr(sys, "argv", ["letta_reply_diff.py", "pack"])
    with pytest.raises(SystemExit) as e:
        rd.cmd_pack()
    assert e.value.code == 1
    assert "already been scored" in capsys.readouterr().out


def test_pack_allows_a_run_with_no_verdict_on_record(rd, monkeypatch):
    """Negative control — the same call, with the only stale condition removed."""
    _write_set(rd)
    _write_runs(rd, _rows())
    with open(rd.VERDICT_LOG, "w") as f:
        f.write("5.1R 2026-08-02T13:07:00+00:00 run=000000000000 — ... => FAIL\n")
    monkeypatch.setattr(sys, "argv", ["letta_reply_diff.py", "pack"])
    rd.cmd_pack()
    assert os.path.exists(rd.PACK_PATH)


def test_rejudge_flag_is_an_escape_hatch_that_says_so(rd, monkeypatch, capsys):
    _write_set(rd)
    run_id = _write_runs(rd, _rows())
    with open(rd.VERDICT_LOG, "w") as f:
        f.write(f"5.1R 2026-08-02T13:07:00+00:00 run={run_id} — ... => FAIL\n")
    monkeypatch.setattr(sys, "argv", ["letta_reply_diff.py", "pack", "--rejudge"])
    rd.cmd_pack()
    assert "NOT a new measurement" in capsys.readouterr().out


def test_pack_stamps_the_key_with_the_run_it_packed(rd, monkeypatch):
    _write_set(rd)
    run_id = _write_runs(rd, _rows())
    monkeypatch.setattr(sys, "argv", ["letta_reply_diff.py", "pack"])
    rd.cmd_pack()
    with open(rd.KEY_PATH) as f:
        assert json.load(f)["run_id"] == run_id


# --- score refuses a key or judgments belonging to a different replay ------------------------

def _judgments(rd, path, run_id=None):
    body = [{"msg_id": i, "material_divergence": False} for i in range(1, 21)]
    with open(path, "w") as f:
        json.dump({"run_id": run_id, "judgments": body} if run_id else {"judgments": body}, f)
    return path


def test_score_refuses_a_key_packed_from_a_different_run(rd, capsys):
    _write_runs(rd, _rows("post-guardrail"))
    with open(rd.KEY_PATH, "w") as f:
        json.dump({"set_built": SET_BUILT, "run_id": rd._run_id(SET_BUILT, _rows("pre")),
                   "arms": {str(i): {"A": "letta", "B": "sdk", "invalid": None}
                            for i in range(1, 21)}}, f)
    j = _judgments(rd, os.path.join(rd.LOGS, "j.json"))
    with pytest.raises(SystemExit) as e:
        rd.cmd_score(j)
    assert e.value.code == 1
    assert "belongs to a different replay" in capsys.readouterr().out


def test_score_refuses_judgments_stamped_with_another_run(rd, capsys):
    """The last hop, and the one that had no check at all: same 20 msg_ids, earlier replay."""
    run_id = _write_runs(rd, _rows("post-guardrail"))
    with open(rd.KEY_PATH, "w") as f:
        json.dump({"set_built": SET_BUILT, "run_id": run_id,
                   "arms": {str(i): {"A": "letta", "B": "sdk", "invalid": None}
                            for i in range(1, 21)}}, f)
    j = _judgments(rd, os.path.join(rd.LOGS, "j.json"),
                   run_id=rd._run_id(SET_BUILT, _rows("pre-guardrail")))
    with pytest.raises(SystemExit) as e:
        rd.cmd_score(j)
    assert e.value.code == 1
    assert "a different replay" in capsys.readouterr().out


def test_score_accepts_judgments_stamped_with_the_current_run(rd, capsys):
    """Negative control for both score guards — identical shape, correct stamp, scores through."""
    run_id = _write_runs(rd, _rows("post-guardrail"))
    with open(rd.KEY_PATH, "w") as f:
        json.dump({"set_built": SET_BUILT, "run_id": run_id,
                   "arms": {str(i): {"A": "letta", "B": "sdk", "invalid": None}
                            for i in range(1, 21)}}, f)
    j = _judgments(rd, os.path.join(rd.LOGS, "j.json"), run_id=run_id)
    with pytest.raises(SystemExit) as e:
        rd.cmd_score(j)
    assert e.value.code == 0
    assert "20/20 measured" in capsys.readouterr().out


def test_score_refuses_unstamped_judgments_older_than_the_pack(rd, capsys):
    """The stamp needs cooperation from whoever writes the judgments; the file clock does not."""
    run_id = _write_runs(rd, _rows())
    j = _judgments(rd, os.path.join(rd.LOGS, "j.json"))
    with open(rd.KEY_PATH, "w") as f:
        json.dump({"set_built": SET_BUILT, "run_id": run_id,
                   "arms": {str(i): {"A": "letta", "B": "sdk", "invalid": None}
                            for i in range(1, 21)}}, f)
    os.utime(j, (0, 0))  # judged before the pack it would be scored against
    with pytest.raises(SystemExit) as e:
        rd.cmd_score(j)
    assert e.value.code == 1
    assert "older than" in capsys.readouterr().out


def test_score_accepts_unstamped_judgments_written_after_the_pack(rd):
    """Negative control — the file clock must not punish a judge that simply omits the stamp."""
    run_id = _write_runs(rd, _rows())
    with open(rd.KEY_PATH, "w") as f:
        json.dump({"set_built": SET_BUILT, "run_id": run_id,
                   "arms": {str(i): {"A": "letta", "B": "sdk", "invalid": None}
                            for i in range(1, 21)}}, f)
    j = _judgments(rd, os.path.join(rd.LOGS, "j.json"))
    with pytest.raises(SystemExit) as e:
        rd.cmd_score(j)
    assert e.value.code == 0


def test_legacy_unstamped_runs_and_key_still_score(rd):
    """Backward compatibility is deliberate: tonight's cron must not be broken by this change."""
    rows = [{"msg_id": i, "invalid": None} for i in range(1, 21)]
    with open(rd.RUNS_PATH, "w") as f:
        json.dump({"set_built": SET_BUILT, "rows": rows}, f)
    with open(rd.KEY_PATH, "w") as f:
        json.dump({"set_built": SET_BUILT,
                   "arms": {str(i): {"A": "letta", "B": "sdk", "invalid": None}
                            for i in range(1, 21)}}, f)
    j = _judgments(rd, os.path.join(rd.LOGS, "j.json"))
    with pytest.raises(SystemExit) as e:
        rd.cmd_score(j)
    assert e.value.code == 0


def test_verdict_line_carries_the_run_id_or_pack_can_never_read_it_back(rd, capsys):
    """The pack guard reads run ids out of the verdict log — the loop has to actually close."""
    run_id = _write_runs(rd, _rows())
    with open(rd.KEY_PATH, "w") as f:
        json.dump({"set_built": SET_BUILT, "run_id": run_id,
                   "arms": {str(i): {"A": "letta", "B": "sdk", "invalid": None}
                            for i in range(1, 21)}}, f)
    j = _judgments(rd, os.path.join(rd.LOGS, "j.json"), run_id=run_id)
    with pytest.raises(SystemExit):
        rd.cmd_score(j)
    assert f"run={run_id}" in capsys.readouterr().out
    assert run_id in rd._scored_run_ids()
