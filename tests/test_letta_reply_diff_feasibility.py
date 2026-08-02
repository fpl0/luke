"""The 5.1R accept region is the go/no-go criterion, so it is pinned here in numbers.

Two problems this file exists for, both found on 2026-08-02.

**The bar moved without the person whose decision it is.** ``divergence_cause`` (68aa400)
changed what PASS means eight days before the cutover call, and the change was correct but
invisible: four constants and one formula, edited in a normal-looking commit. Nothing anywhere
stated the old bar, so nothing could show it had changed. The region is asserted here as literal
numbers, each negative-controlled by moving the constant that produces it — so the next widening
is a test diff that says exactly how much easier the gate got, and Filipe can be told.

**The verdict was decided before the run started, and only announced after it.** Past
``HARNESS_CAP`` the result is VOID whatever Letta remembered — but ``score`` can only say so
once twenty live turns and a blind judge have been spent. The pack's own documented asymmetry
is six rows (two clock, four context), one over the cap. ``feasibility`` asks the same
arithmetic up front.

The arithmetic lives in ``accept`` and is called by both, deliberately: a preflight that
reimplements the bar is a second bar, and this file has produced that failure four times already
(the qwen ``context_window``, ``recall(before=...)``, the work-claim TTL, the ``set_built``
guards — each correct when written, each silently wrong once its surroundings moved).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def _exit(fn, *a, **kw):
    with pytest.raises(SystemExit) as e:
        fn(*a, **kw)
    return e.value.code


# --- the region, in numbers ---------------------------------------------------------------

def test_the_accept_region_is_exactly_this(rd):
    """If this test changes, the go/no-go bar changed. That is the entire point of it."""
    assert [rd.memory_budget(20, e) for e in range(0, 8)] == [2, 2, 2, 2, 2, 1, -1, -1]


def test_past_the_cap_pass_is_unreachable_at_zero_memory_divergences(rd):
    """A perfect Letta cannot pass a pack that is mostly not a comparison — by design."""
    assert rd.accept(20, 14, 6, 0)[0] == "VOID"
    # Negative control: one fewer excluded row and the same perfect arm passes.
    assert rd.accept(20, 15, 5, 0)[0] == "PASS"


def test_exclusions_tighten_the_absolute_memory_budget(rd):
    """The bar is a percentage of a shrinking denominator, so excusing rows costs slack.

    Worth pinning because it reads the other way at a glance: exclusions look like they can
    only help Letta, and between E=4 and E=5 they take a divergence off its budget.
    """
    assert rd.memory_budget(20, 4) == 2
    assert rd.memory_budget(20, 5) == 1


def test_a_veto_fails_the_run_anywhere_in_the_region(rd):
    for e in range(0, rd.HARNESS_CAP + 1):
        assert rd.accept(20, 20 - e, e, 1)[0] != "PASS"


@pytest.mark.parametrize("const,value", [("HARNESS_CAP", 6), ("MIN_AGREEMENT", 0.8)])
def test_each_constant_is_load_bearing(rd, monkeypatch, const, value):
    """Negative control for the region: move a constant, the region must move.

    A pinned region proves nothing if some of what pins it is inert — that was the work_claim
    lesson, where thirteen green tests sat over a lock that granted itself to everyone.
    """
    before = [rd.memory_budget(20, e) for e in range(0, 8)]
    monkeypatch.setattr(rd, const, value)
    assert [rd.memory_budget(20, e) for e in range(0, 8)] != before, f"{const} is inert"


def test_min_measured_is_dominated_by_the_cap_and_currently_does_nothing(rd, monkeypatch):
    """MIN_MEASURED is the one fence of the three that cannot fire at n=20.

    The module docstring describes the exclusion as "fenced on three sides". Two of them bite:
    the cap and the agreement bar over a shrinking denominator. The third cannot — HARNESS_CAP=5
    already guarantees measured >= 15, so a floor of 14 is unreachable from above. Asserted
    rather than deleted because it is a live fence at other constants (raise the cap and it
    starts binding immediately, second case below), and because a guard that quietly does
    nothing is exactly the class this file keeps finding.
    """
    before = [rd.memory_budget(20, e) for e in range(0, 8)]
    monkeypatch.setattr(rd, "MIN_MEASURED", 15)
    assert [rd.memory_budget(20, e) for e in range(0, 8)] == before
    # It is not dead code, only dominated: widen the cap and it takes over as the binding fence.
    monkeypatch.setattr(rd, "HARNESS_CAP", 10)
    assert rd.accept(20, 14, 6, 0)[0] == "VOID"


def test_score_uses_the_extracted_arithmetic_rather_than_its_own_copy(rd, monkeypatch, capsys):
    """cmd_score must ask `accept`, not re-derive the bar beside it."""
    monkeypatch.setattr(rd, "accept", lambda *a, **kw: ("SENTINEL", 99, 98))
    rows = [{"msg_id": i, "invalid": None} for i in range(1, 21)]
    json.dump({"set_built": "s", "run_id": "r", "rows": rows}, open(rd.RUNS_PATH, "w"))
    json.dump({"set_built": "s", "run_id": "r",
               "arms": {str(i): {"A": "letta", "B": "sdk"} for i in range(1, 21)}},
              open(rd.KEY_PATH, "w"))
    jp = os.path.join(rd.LOGS, "j.json")
    json.dump({"run_id": "r",
               "judgments": [{"msg_id": i, "material_divergence": False} for i in range(1, 21)]},
              open(jp, "w"))
    _exit(rd.cmd_score, jp)
    assert "SENTINEL" in capsys.readouterr().out


# --- the preflight ------------------------------------------------------------------------

def _conclusion(out: str) -> str:
    """The preflight's verdict line, not the region table above it.

    Read as the last non-empty line on purpose: the table itself contains the word
    UNREACHABLE on every row past the cap, so a bare substring check over the whole
    output passes whatever the preflight concluded.
    """
    return [ln for ln in out.splitlines() if ln.strip()][-1]


def test_feasibility_refuses_a_pack_that_cannot_pass(rd, capsys):
    code = _exit(rd.cmd_feasibility, asymmetric=6)
    assert code == 1 and _conclusion(capsys.readouterr().out).startswith("UNREACHABLE:")


def test_feasibility_clears_a_pack_that_can(rd, capsys):
    """Negative control: one row fewer and the preflight gets out of the way."""
    code = _exit(rd.cmd_feasibility, asymmetric=5)
    assert code == 0 and _conclusion(capsys.readouterr().out).startswith("REACHABLE:")


def test_feasibility_never_invents_the_number_it_checks(rd, capsys):
    """With no scored run and no argument, it reports the region and declines to predict.

    Guessing here would be a prediction wearing a preflight's exit code — the failure this
    goal already produced once, when a control I confounded myself returned the dramatic
    answer I expected.
    """
    code = _exit(rd.cmd_feasibility)
    conclusion = _conclusion(capsys.readouterr().out)
    assert code == 0 and "unknown" in conclusion
    assert not conclusion.startswith(("REACHABLE:", "UNREACHABLE:"))


def test_feasibility_reads_the_last_scored_run(rd, capsys):
    """Grounded in an artefact on disk rather than a number someone typed."""
    with open(rd.VERDICT_LOG, "w") as f:
        f.write("5.1R 2026-08-01 — no-material-divergence 11/20 measured; "
                "harness-excluded 2/20 (cap 5) => FAIL\n")
        f.write("5.1R 2026-08-02 — no-material-divergence 14/14 measured; "
                "harness-excluded 6/20 (cap 5) => VOID\n")
    code = _exit(rd.cmd_feasibility)
    out = capsys.readouterr().out
    # Reads the LAST line, not the first — a stale earlier run must not clear a current pack.
    assert code == 1 and "6 of 20" in _conclusion(out)
