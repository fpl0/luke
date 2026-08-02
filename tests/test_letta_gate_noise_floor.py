"""The noise-floor control must refuse the ways it can silently lie.

``letta_gate_noise_floor.py`` exists to answer one question — how often does the SAME arm
materially disagree with ITSELF? — and the answer is only worth anything if the two passes
being compared really are the same arm under one configuration. The first attempt at this
measurement was confounded exactly there: the self-verification guardrail was packed into
the agent's operating-rules block BETWEEN pass 1 and pass 2, so two of the four divergences
were a config change wearing the costume of noise. Nothing errored. The number looked fine.

So these tests drive the refusal paths, and each one is negative-controlled: a guard that
cannot be shown failing is indistinguishable from no guard.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUILT = "2026-08-02T07:12:47+00:00"
PROMPTS = [
    {"msg_id": 1, "prompt": "how much am I paying?", "ts": "2026-07-01T10:00:00+00:00",
     "bucket": "conversational"},
    {"msg_id": 2, "prompt": "who is on the team?", "ts": "2026-07-02T10:00:00+00:00",
     "bucket": "factual"},
]


@pytest.fixture()
def nf(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "letta_gate_noise_floor", os.path.join(REPO, "scripts", "letta_gate_noise_floor.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["letta_gate_noise_floor"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "LOGS", str(tmp_path))
    monkeypatch.setattr(mod, "SET_PATH", str(tmp_path / "set.json"))
    monkeypatch.setattr(mod, "PASS1_PATH", str(tmp_path / "p1.json"))
    monkeypatch.setattr(mod, "KEY_PATH", str(tmp_path / "key.json"))
    monkeypatch.setattr(mod, "PACK_PATH", str(tmp_path / "pack.md"))
    monkeypatch.setattr(mod, "VERDICT_LOG", str(tmp_path / "nf.log"))
    return mod


def _write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def _seed(nf, *, set_built=BUILT, p1_built=BUILT, p1_rows=None):
    _write(nf.SET_PATH, {"built": set_built, "prompts": PROMPTS})
    rows = p1_rows if p1_rows is not None else [
        {"msg_id": p["msg_id"], "letta_reply": f"pass one answer {p['msg_id']}", "invalid": None}
        for p in PROMPTS
    ]
    _write(nf.PASS1_PATH, {"ran": "2026-08-02T13:07:45+00:00", "set_built": p1_built, "rows": rows})


def _seed_slot(nf, n, replies):
    _write(nf._slot(n), {"ran": "2026-08-02T13:31:03+00:00", "set_built": BUILT,
                         "rows": [{"msg_id": mid, "letta_reply_p2": text, "invalid": None}
                                  for mid, text in replies.items()]})


def test_refuses_a_pass_one_from_a_different_harness(nf):
    """A noise floor across two harnesses measures the harness, wearing the noise's name."""
    _seed(nf, p1_built="2026-08-01T21:12:00+00:00")
    with pytest.raises(SystemExit) as e:
        nf._load_inputs()
    assert "REFUSING" in str(e.value)

    _seed(nf)  # negative control: same build passes
    spec, p1 = nf._load_inputs()
    assert len(p1["rows"]) == 2 and spec["built"] == BUILT


def test_refuses_a_short_pass_one(nf):
    _seed(nf, p1_rows=[{"msg_id": 1, "letta_reply": "only one", "invalid": None}])
    with pytest.raises(SystemExit) as e:
        nf._load_inputs()
    assert "REFUSING" in str(e.value)


def test_pack_refuses_a_missing_slot(nf, capsys):
    _seed(nf)
    with pytest.raises(SystemExit) as e:
        nf.cmd_pack(1, 3)
    assert "run --slot 3" in str(e.value)


def test_pack_reads_either_slot_shape(nf):
    """Pass 1 is a 5.1R row (`letta_reply`); later passes are this script's (`letta_reply_p2`).

    Assuming one shape would make a cross-slot pack silently render '(no answer produced)'
    for every item on one side — which reads, to a judge, as a total divergence.
    """
    _seed(nf)
    _seed_slot(nf, 2, {1: "pass two answer 1", 2: "pass two answer 2"})
    nf.cmd_pack(1, 2)
    pack = open(nf.PACK_PATH).read()
    assert "pass one answer 1" in pack and "pass two answer 1" in pack
    assert "(no answer produced)" not in pack


def test_pack_records_which_slots_it_compared(nf):
    """Without this the key cannot tell a confounded pairing from a clean one after the fact."""
    _seed(nf)
    _seed_slot(nf, 2, {1: "a", 2: "b"})
    _seed_slot(nf, 3, {1: "c", 2: "d"})
    nf.cmd_pack(2, 3)
    key = json.load(open(nf.KEY_PATH))
    assert key["slots"] == [2, 3]
    assert all(v["slots"] == {"p1": 2, "p2": 3} for v in key["arms"].values())


def test_blinding_salt_differs_from_the_5_1r_shuffle(nf):
    """The person judging this pack already judged the 5.1R pack and has seen its key.

    Reusing ``_arm_order`` would hand them this pack's assignment for free, so blindness
    would be nominal. Over the real 20 msg_ids the two shuffles must genuinely disagree.
    """
    rd_spec = importlib.util.spec_from_file_location(
        "rd_for_salt", os.path.join(REPO, "scripts", "letta_reply_diff.py"))
    rd = importlib.util.module_from_spec(rd_spec)
    rd_spec.loader.exec_module(rd)
    real_ids = [2814, 2835, 2935, 2973, 3406, 3436, 3505, 3528, 3560, 3566,
                3613, 3632, 3682, 3691, 3710, 3743, 3789, 3883, 3940, 3950]
    disagree = sum(1 for m in real_ids
                   if (rd._arm_order(m)[0] == "sdk") != (nf._pq_order(m)[0] == "p1"))
    assert 0 < disagree < len(real_ids), f"salts are aliased: {disagree}/{len(real_ids)}"


def test_score_refuses_a_key_from_another_set_build(nf, tmp_path):
    _seed(nf)
    _write(nf.KEY_PATH, {"set_built": "2026-08-01T00:00:00+00:00", "arms": {}})
    j = tmp_path / "j.json"
    _write(j, {"judgments": []})
    with pytest.raises(SystemExit) as e:
        nf.cmd_score(str(j))
    assert "REFUSING" in str(e.value)


def test_unjudged_row_counts_as_divergence_not_as_stable(nf, tmp_path, capsys):
    """A row nobody judged must not quietly improve the floor by leaving the numerator."""
    _seed(nf)
    _write(nf.KEY_PATH, {"set_built": BUILT, "slots": [2, 3],
                         "arms": {"1": {"P": "p2", "Q": "p3", "invalid": None},
                                  "2": {"P": "p3", "Q": "p2", "invalid": None}}})
    j = tmp_path / "j.json"
    _write(j, {"judgments": [{"msg_id": 1, "material_divergence": False}]})
    nf.cmd_score(str(j))
    out = capsys.readouterr().out
    assert "self-divergence 1/2" in out
    assert "no judgment emitted" in out


@pytest.mark.parametrize("k,expect", [(0, "REACHABLE"), (3, "UNREACHABLE")])
def test_verdict_wording_tracks_the_arithmetic(nf, tmp_path, capsys, k, expect):
    """20 rows, bar 18: two divergences is the whole budget, three blows it."""
    _seed(nf)
    ids = [str(i) for i in range(1, 21)]
    _write(nf.KEY_PATH, {"set_built": BUILT, "slots": [2, 3],
                         "arms": {i: {"P": "p2", "Q": "p3", "invalid": None} for i in ids}})
    j = tmp_path / "j.json"
    _write(j, {"judgments": [{"msg_id": int(i), "material_divergence": n < k}
                             for n, i in enumerate(ids)]})
    nf.cmd_score(str(j))
    assert expect in capsys.readouterr().out
