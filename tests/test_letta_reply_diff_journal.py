"""The 5.1R replay must survive being killed, and must never pack a partial run as a gate.

``cmd_run`` drives 20 live Letta turns. It used to hold every row in memory and write them
only after the last one, so a session that died at turn 19 — a timeout, a bridge hiccup, the
30m ceiling that actually killed the 2026-08-02 01:03 session — lost all twenty. The journal
fixes that, and these tests cover the three decisions it makes rather than its happy path:

  * a row from a PREVIOUS set build is discarded, not resumed. Rows replayed under a different
    harness (different ``as_of`` anchoring, conversation depth, tool surface) are exactly the
    artefact class that made the Aug-01 numbers meaningless; mixing them into one pack would
    be undetectable downstream.
  * a torn final line — the literal signature of a hard kill mid-write — is dropped without
    taking the rest of the journal with it.
  * an incomplete run leaves NO runs file, and ``pack`` refuses a short one. The accept clause
    is ">=18/20"; a pack built from 12 rows reads identically to one built from 20.

The lesson from ``work_claim`` applies here and is why these are decision tests: that module's
unit tests were green while the lock granted itself to everyone. Green tests on the happy path
prove nothing about a guard.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SET_BUILT = "2026-08-02T07:12:47.622126+00:00"
OLD_BUILT = "2026-07-30T00:00:00+00:00"


@pytest.fixture()
def rd(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "letta_reply_diff", os.path.join(REPO, "scripts", "letta_reply_diff.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["letta_reply_diff"] = mod
    spec.loader.exec_module(mod)
    # Redirect every artefact into tmp so the real logs/ are never touched.
    monkeypatch.setattr(mod, "LOGS", str(tmp_path))
    for name in ("SET_PATH", "RUNS_PATH", "PACK_PATH", "KEY_PATH", "PARTIAL_PATH", "VERDICT_LOG"):
        monkeypatch.setattr(mod, name, str(tmp_path / os.path.basename(getattr(mod, name))))
    return mod


def _journal(rd, rows):
    with open(rd.PARTIAL_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(msg_id, built=SET_BUILT):
    return {"msg_id": msg_id, "set_built": built, "letta_reply": "x" * 80, "invalid": None}


def test_resumes_rows_from_the_current_set_build(rd):
    _journal(rd, [_row(1), _row(2)])
    assert [r["msg_id"] for r in rd._load_partial(SET_BUILT, {1, 2, 3})] == [1, 2]


def test_discards_rows_from_a_previous_set_build(rd, capsys):
    """A row replayed under an older harness must be re-run, not inherited."""
    _journal(rd, [_row(1, built=OLD_BUILT), _row(2)])
    kept = rd._load_partial(SET_BUILT, {1, 2, 3})
    assert [r["msg_id"] for r in kept] == [2]
    assert "dropped 1" in capsys.readouterr().out


def test_discards_rows_no_longer_in_the_set(rd):
    """`build` can reshuffle the top 20; a dropped prompt must not linger in the pack."""
    _journal(rd, [_row(1), _row(99)])
    assert [r["msg_id"] for r in rd._load_partial(SET_BUILT, {1, 2, 3})] == [1]


def test_torn_final_line_costs_one_row_not_the_journal(rd):
    """The signature of a hard kill: last write half-flushed."""
    with open(rd.PARTIAL_PATH, "w") as f:
        f.write(json.dumps(_row(1)) + "\n")
        f.write(json.dumps(_row(2)) + "\n")
        f.write('{"msg_id": 3, "letta_re')
    assert [r["msg_id"] for r in rd._load_partial(SET_BUILT, {1, 2, 3})] == [1, 2]


def test_missing_journal_is_a_clean_start_not_an_error(rd):
    assert rd._load_partial(SET_BUILT, {1, 2}) == []


def test_pack_refuses_a_short_runs_file(rd, capsys):
    """A 12-row pack and a 20-row pack are indistinguishable once judged."""
    with open(rd.SET_PATH, "w") as f:
        json.dump({"built": SET_BUILT, "prompts": [{"msg_id": i} for i in range(20)]}, f)
    with open(rd.RUNS_PATH, "w") as f:
        json.dump({"rows": [_row(i) for i in range(12)]}, f)

    with pytest.raises(SystemExit) as e:
        rd.cmd_pack()
    assert e.value.code == 1
    assert "REFUSING" in capsys.readouterr().out
    assert not os.path.exists(rd.PACK_PATH), "a refused pack must not leave an artefact"


# --- provenance: right row count, wrong harness -----------------------------------------
#
# The count check above cannot see this class at all. On 2026-08-02 the runs file on disk
# held 20 rows replayed on Aug-01 21:12Z under the two-tool, unanchored, 2-message-buffer
# agent, while the set had been rebuilt that morning under the fixed harness. `pack` ran
# happily and wrote a 49k judge pack with a fresh key: 20 == 20. Judging it would have
# produced a verdict stamped today that measured the exact harness the re-run exists to
# replace — and the plan's standing warning ("do NOT re-judge the Aug-01 pack") was the only
# thing standing between that pack and a number.


def _packable(msg_id, sdk="sdk answer " * 5, letta="letta answer " * 5):
    return {"msg_id": msg_id, "ts": "2026-07-01T10:00:00+00:00", "prompt": f"p{msg_id}",
            "bucket": "b", "sdk_reply": sdk, "letta_reply": letta, "invalid": None}


def _stage(rd, n=3, set_built=SET_BUILT, runs_built=SET_BUILT, stamped=True):
    with open(rd.SET_PATH, "w") as f:
        json.dump({"built": set_built, "prompts": [{"msg_id": i} for i in range(n)]}, f)
    runs = {"ran": "2026-08-02T12:00:00+00:00", "rows": [_packable(i) for i in range(n)]}
    if stamped:
        runs["set_built"] = runs_built
    with open(rd.RUNS_PATH, "w") as f:
        json.dump(runs, f)


def test_pack_refuses_runs_replayed_against_a_previous_set_build(rd, capsys):
    _stage(rd, runs_built=OLD_BUILT)
    with pytest.raises(SystemExit) as e:
        rd.cmd_pack()
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "REFUSING" in out and OLD_BUILT in out
    assert not os.path.exists(rd.PACK_PATH), "a refused pack must not leave an artefact"


def test_pack_refuses_an_unstamped_runs_file(rd, capsys):
    """Every runs file written before this guard existed is, by definition, from an old harness."""
    _stage(rd, stamped=False)
    with pytest.raises(SystemExit) as e:
        rd.cmd_pack()
    assert e.value.code == 1
    assert "(unstamped)" in capsys.readouterr().out
    assert not os.path.exists(rd.PACK_PATH)


def test_pack_writes_a_stamped_key_when_provenance_matches(rd):
    _stage(rd)
    rd.cmd_pack()
    with open(rd.KEY_PATH) as f:
        key = json.load(f)
    assert key["set_built"] == SET_BUILT
    assert set(key["arms"]) == {"0", "1", "2"}


def test_run_stamps_the_set_build_and_pack_then_accepts(rd, monkeypatch):
    """The write half of the guard, exercised rather than read.

    A guard that refuses everything is not a guard, and the two refusal tests above pass
    just as happily if `run` never writes the stamp at all — the whole pipeline would simply
    be dead. Drive `cmd_run` for real against a stubbed turn driver (no OAuth, no live
    agent) and check the artefact it produces is one `pack` accepts.
    """
    from luke import letta_agent

    monkeypatch.setattr(letta_agent, "compose_letta_turn_input",
                        lambda prompt, as_of=None: f"<mem id=1>ctx</mem>\n{prompt}")
    monkeypatch.setattr(letta_agent, "drive_letta_turn",
                        lambda body, inject_recall=True: {"reply": "a real answer " * 8,
                                                          "seconds": 1.0, "tools": []})
    prompts = [{"msg_id": i, "ts": "2026-07-01T10:00:00+00:00", "prompt": f"p{i}",
                "bucket": "b", "sdk_reply": "sdk answer " * 5, "score": 1,
                "prior_context": []} for i in range(2)]
    with open(rd.SET_PATH, "w") as f:
        json.dump({"built": SET_BUILT, "prompts": prompts}, f)

    rd.cmd_run()
    with open(rd.RUNS_PATH) as f:
        assert json.load(f)["set_built"] == SET_BUILT

    rd.cmd_pack()  # must NOT raise SystemExit
    assert os.path.exists(rd.PACK_PATH)


def test_score_refuses_a_key_packed_from_different_runs(rd, capsys):
    """Re-run, forget to re-pack, judge the pack still sitting on disk."""
    _stage(rd)
    with open(rd.KEY_PATH, "w") as f:
        json.dump({"set_built": OLD_BUILT, "arms": {"0": {"A": "sdk", "B": "letta"}}}, f)
    judgments = str(rd.KEY_PATH) + ".judgments"
    with open(judgments, "w") as f:
        json.dump([{"msg_id": 0, "material_divergence": False}], f)

    with pytest.raises(SystemExit) as e:
        rd.cmd_score(judgments)
    assert e.value.code == 1
    assert "REFUSING" in capsys.readouterr().out
    assert not os.path.exists(rd.VERDICT_LOG), "a refused score must not append a verdict"


def test_score_refuses_a_legacy_unstamped_key(rd, capsys):
    _stage(rd)
    with open(rd.KEY_PATH, "w") as f:
        json.dump({"0": {"A": "sdk", "B": "letta", "invalid": None}}, f)
    judgments = str(rd.KEY_PATH) + ".judgments"
    with open(judgments, "w") as f:
        json.dump([{"msg_id": 0, "material_divergence": False}], f)

    with pytest.raises(SystemExit) as e:
        rd.cmd_score(judgments)
    assert e.value.code == 1
    assert "no provenance stamp" in capsys.readouterr().out
    assert not os.path.exists(rd.VERDICT_LOG)
