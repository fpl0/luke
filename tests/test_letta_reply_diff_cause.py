"""The 5.1R gate must charge Letta for forgetting, and not for the harness's own asymmetries.

Two judged runs (7/20 on Aug-01, 11/20 on Aug-02) reported a memory result that was mostly not
about memory: rows where one arm had a pasted email the other never saw, rows anchored to
different moments in time, rows needing a tool rather than a recollection. Those are properties
of the comparison. Scoring them as forgetting made the number unactionable in both directions —
it could not fall for a real reason, and it could not rise for one either.

``divergence_cause`` fixes that, and every test here is about the fence around it rather than the
feature. The exclusion path is the only way this gate can get easier, so what needs proving is
that it CANNOT be walked without evidence, cannot be walked far, and does not blunt the veto on
the rows that still count:

  * attribution fails closed — missing, unknown, or unevidenced causes are scored as recall;
  * exclusions are capped — past the cap the run is VOID, never PASS;
  * the denominator shrinks with the exclusions, so a rump cannot clear a percentage bar;
  * the veto still fires at full strength on every non-excluded row.

Each guard is negative-controlled: the test asserts the guard fires, and that the same input
without the guarded condition does not fire it. A green happy path here would prove nothing —
that was the ``work_claim`` lesson, where the unit tests passed while the lock granted itself to
everybody.
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


def _setup(rd, judgments, n=20):
    """Write a runs file + key for `n` valid rows, and the given judgments. Returns exit code."""
    rows = [{"msg_id": i, "invalid": None} for i in range(1, n + 1)]
    with open(rd.RUNS_PATH, "w") as f:
        json.dump({"set_built": SET_BUILT, "rows": rows}, f)
    with open(rd.KEY_PATH, "w") as f:
        arms = {str(i): {"A": "letta", "B": "sdk", "invalid": None} for i in range(1, n + 1)}
        json.dump({"set_built": SET_BUILT, "arms": arms}, f)
    jpath = os.path.join(rd.LOGS, "judgments.json")
    with open(jpath, "w") as f:
        json.dump(judgments, f)
    return jpath


def _ok(msg_id):
    return {"msg_id": msg_id, "material_divergence": False}


def _diverge(msg_id, cause=None, evidence="quotes an email absent from the other", **kw):
    j = {"msg_id": msg_id, "material_divergence": True, "worse_arm": "A", "note": "gap"}
    if cause:
        j["divergence_cause"] = cause
        j["cause_evidence"] = evidence
    j.update(kw)
    return j


def _run(rd, jpath):
    with pytest.raises(SystemExit) as e:
        rd.cmd_score(jpath)
    return e.value.code


# --- attribution fails closed ------------------------------------------------------------

def test_unlabelled_divergence_is_charged_as_recall(rd):
    """The pre-change judgments format has no cause field. It must not become a free pass."""
    assert rd._attributed_cause({"material_divergence": True}) == "recall"


def test_unknown_cause_is_charged_as_recall(rd):
    """A judge inventing 'harness_artifact' does not get to define the gate's vocabulary."""
    assert rd._attributed_cause({"divergence_cause": "harness_artifact"}) == "recall"


def test_harness_cause_without_evidence_is_charged_as_recall(rd):
    """An excuse with nothing quoted behind it is not an excuse."""
    assert rd._attributed_cause({"divergence_cause": "clock", "cause_evidence": "  "}) == "recall"
    # Negative control: the SAME cause, with evidence, is honoured.
    assert rd._attributed_cause(
        {"divergence_cause": "clock", "cause_evidence": "one says 3 days left, other says 5"}
    ) == "clock"


def test_memory_causes_are_never_excluded(rd):
    for cause in ("recall", "self_knowledge", "quality"):
        assert rd._attributed_cause({"divergence_cause": cause, "cause_evidence": "x"}) == cause
        assert cause not in rd.HARNESS_CAUSES


# --- the exclusion is capped -------------------------------------------------------------

def test_too_many_exclusions_is_void_not_pass(rd, capsys):
    """The failure mode this gate must have is 'fix the harness', not 'the survivors agreed'."""
    judgments = [_ok(i) for i in range(1, 14)] + [
        _diverge(i, cause="context_asymmetry") for i in range(14, 21)
    ]
    code = _run(rd, _setup(rd, judgments))
    out = capsys.readouterr().out
    assert code == 1 and "VOID" in out and "PASS" not in out


def test_exclusions_under_the_cap_still_pass(rd, capsys):
    """Negative control for the cap: same shape, fewer exclusions, and the gate works normally."""
    judgments = [_ok(i) for i in range(1, 17)] + [
        _diverge(i, cause="context_asymmetry") for i in range(17, 21)
    ]
    code = _run(rd, _setup(rd, judgments))
    out = capsys.readouterr().out
    assert code == 0 and "PASS" in out
    assert "harness-excluded 4/20" in out


def test_denominator_shrinks_with_exclusions(rd, capsys):
    """16/16 must not be reported as 16/20 — the excluded rows leave the measurement entirely."""
    judgments = [_ok(i) for i in range(1, 17)] + [
        _diverge(i, cause="clock", evidence="different 'today'") for i in range(17, 21)
    ]
    _run(rd, _setup(rd, judgments))
    assert "16/16 measured" in capsys.readouterr().out


def test_a_rump_cannot_clear_the_bar(rd, capsys):
    """13 comparable rows agreeing is a small sample, not a parity result."""
    judgments = [_ok(i) for i in range(1, 14)] + [
        _diverge(i, cause="tooling", evidence="one sends the message") for i in range(14, 21)
    ]
    code = _run(rd, _setup(rd, judgments))
    assert code == 1 and "VOID" in capsys.readouterr().out


# --- the veto keeps its teeth ------------------------------------------------------------

def test_veto_still_fails_an_otherwise_perfect_run(rd, capsys):
    """One recall row where letta is wrong and sdk is right sinks the run, 19 clean or not."""
    judgments = [_ok(i) for i in range(1, 20)] + [
        _diverge(20, cause="recall", factually_wrong_arm="A")
    ]
    code = _run(rd, _setup(rd, judgments))
    assert code == 1 and "letta-wrong-where-sdk-right 1" in capsys.readouterr().out


def test_veto_does_not_fire_on_material_the_arm_never_had(rd, capsys):
    """Being 'wrong' about a pasted email you were never shown is not a memory failure.

    This is the one place the change deliberately blunts the veto, so it is stated outright:
    the row is excluded, and the veto is counted over the rows that remain comparable.
    """
    judgments = [_ok(i) for i in range(1, 20)] + [
        _diverge(20, cause="context_asymmetry", factually_wrong_arm="A")
    ]
    code = _run(rd, _setup(rd, judgments))
    assert code == 0 and "letta-wrong-where-sdk-right 0" in capsys.readouterr().out


def test_invalid_rows_are_never_excusable(rd, capsys):
    """A crashed Letta turn stays a failure — you cannot pass this gate by not answering.

    The label is the attack here: an invalid row carrying ``context_asymmetry`` would, if the
    cause were read before the validity check, leave the denominator and take its own failure
    with it. Three invalid rows would then read as a clean 17/17. The check has to come first.
    """
    rows = [{"msg_id": i, "invalid": None} for i in range(1, 18)]
    rows += [{"msg_id": i, "invalid": "empty reply"} for i in range(18, 21)]
    jpath = _setup(rd, [_ok(i) for i in range(1, 18)]
                   + [_diverge(i, cause="context_asymmetry") for i in range(18, 21)])
    with open(rd.RUNS_PATH, "w") as f:
        json.dump({"set_built": SET_BUILT, "rows": rows}, f)
    code = _run(rd, jpath)
    out = capsys.readouterr().out
    # Charged, not excluded: the denominator keeps all 20 and the run fails on 17/20.
    assert code == 1
    assert "invalid letta turn" in out
    assert "17/20 measured" in out
    assert "harness-excluded 0/20" in out
