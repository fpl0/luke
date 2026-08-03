"""The comparability classifier has to be falsifiable in the direction that costs something.

`letta_comparability_audit.py` was written after seeing the 20 rows it is measured against, so
its recall is fitted and proves nothing. What these tests pin instead are the properties that
were NOT free: that it fails toward COMPARABLE when it does not recognise a prompt, that the
attachment-dominance rule is load-bearing rather than decoration, and that `breadth` refuses a
filter broad enough to starve the pack.

Each guard is negative-controlled — reverted to the weaker version it replaced, and asserted to
produce the failure it was written to stop. A classifier whose tests only feed it the prompts it
already matches is indistinguishable from one that returns a constant.
"""

from __future__ import annotations

import importlib.util
import io
import contextlib
import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def ca():
    spec = importlib.util.spec_from_file_location(
        "ca", os.path.join(REPO, "scripts", "letta_comparability_audit.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the soft edge points toward COMPARABLE ------------------------------------------------


def test_unrecognised_prompt_is_comparable(ca):
    """An unknown prompt is DRAWN, not dropped.

    This is the whole safety direction. If an unrecognised prompt were filtered out, the pool
    would shrink toward whatever these regexes happen to like and nobody would see it happen —
    the exact failure the exclusion audit's `unclassified` handling exists to prevent, arrived
    at from the other side.
    """
    klass, where = ca.classify("What did Prerna say about the Dublin team charter?", [])
    assert klass is None and where == ""


def test_durable_world_fact_is_comparable(ca):
    """The class must not swallow the prompts the gate exists to measure."""
    for prompt in (
        "Remind me what Christopher's shift pattern is",
        "Who is on the Dublin International team?",
        "What was the reason I turned Cloudbeds down?",
    ):
        assert ca.classify(prompt, [])[0] is None, prompt


# --- the three classes fire on their own axis ----------------------------------------------


def test_mutable_state_fires_on_system_surface(ca):
    assert ca.classify("Fix this! ⚠️ Task 'Evening check-in' has failed 4 times", [])[0] == \
        "mutable-state"


def test_write_action_fires_on_perform_ask(ca):
    assert ca.classify("Can we test this in a branch? Don't change main. Implement letta", [])[0] \
        == "write-action"


def test_artifact_ref_fires_on_url(ca):
    assert ca.classify("Updated contract: https://apps.docusign.com/api/esign/x", [])[0] == \
        "artifact-ref"


def test_dissatisfaction_alone_is_not_mutable_state(ca):
    """A complaint about Luke is not automatically clock-dependent.

    The narrowness here is deliberate and was paid for: one judging pass attributed
    "your answer quality is degrading" to `clock` and another to `quality`. Matching every
    complaint would make the filter an opinion about tone.
    """
    assert ca.classify("I don't think this is good enough, honestly", [])[0] is None


# --- attachment DOMINANCE, negative-controlled ---------------------------------------------


SELF_CONTAINED = "Rosa, Samuel, Javier, Conor, Daniel, and Jaime. That's the team."
ONE_ATTACHMENT = [
    "Filipe Lima: \n[Document saved: /x/chat.zip]",
    "Luke: Got it, unpacked.",
    "Filipe Lima: so who reported to whom?",
    "Luke: Martin was the boss.",
]
ALL_ATTACHMENTS = [f"Filipe Lima: \n[Document saved: /x/chat{i}.zip]" for i in range(7)] + [
    "Luke: All seven unpacked."
]


def test_incidental_attachment_does_not_flag(ca):
    """One attachment in the backscroll does not make a self-contained prompt incomparable."""
    assert ca.classify(SELF_CONTAINED, ONE_ATTACHMENT)[0] is None


def test_dominant_attachments_flag(ca):
    klass, where = ca.classify("Ultimately CarGurus called Alysson, Pat, and Mary", ALL_ATTACHMENTS)
    assert klass == "artifact-ref"
    assert where.startswith("prior(7/8)")


def test_dominance_rule_is_load_bearing(ca, monkeypatch):
    """Revert dominance to "any attachment" and the incidental case must start flagging.

    Without this the dominance threshold is a number nobody can tell from any other number.
    """
    original = ca.classify

    def weakened(prompt, prior_context=None):
        prior = prior_context or []
        if ca.MUTABLE_STATE.search(prompt):
            return "mutable-state", "prompt"
        if ca.WRITE_ACTION.search(prompt):
            return "write-action", "prompt"
        if ca.ARTIFACT_REF.search(prompt):
            return "artifact-ref", "prompt"
        if any(ca._ATTACHMENT_MARKER.search(t) for t in prior):   # the rule it replaced
            return "artifact-ref", "prior"
        return None, ""

    monkeypatch.setattr(ca, "classify", weakened)
    assert ca.classify(SELF_CONTAINED, ONE_ATTACHMENT)[0] == "artifact-ref"  # the defect
    monkeypatch.undo()
    assert original(SELF_CONTAINED, ONE_ATTACHMENT)[0] is None               # the fix


# --- deictic inheritance, negative-controlled ----------------------------------------------


MUTABLE_PARENT = ["Filipe Lima: Show me what is in your content right now",
                  "Luke: Here's my working context — key entities, goals..."]


def test_deictic_reply_inherits_parent_class(ca):
    klass, where = ca.classify("hmmm, I'm not sure this is the correct thing.", MUTABLE_PARENT)
    assert (klass, where) == ("mutable-state", "prior")


def test_self_contained_reply_does_not_inherit(ca):
    """Inheritance is gated on the prompt LEANING on the parent, not on the parent alone."""
    assert ca.classify("What is Prerna's email address?", MUTABLE_PARENT)[0] is None


# --- breadth refuses a filter that starves the pack ----------------------------------------


def test_breadth_fails_when_filter_leaves_too_few(ca, monkeypatch):
    """A runaway class must be caught by the audit, not discovered when `build` returns 11 rows.

    Driven by making the filter match everything — the only way to know the guard fires is to
    make the thing it guards against actually happen.
    """
    monkeypatch.setattr(ca, "MUTABLE_STATE", re.compile(r".", re.S))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ca.cmd_breadth()
    assert rc == 1
    assert "too broad" in buf.getvalue()


def test_breadth_passes_on_the_real_classes(ca):
    """...and the same command must pass on the shipped patterns, or the guard is just a fail."""
    with contextlib.redirect_stdout(io.StringIO()):
        assert ca.cmd_breadth() == 0


# --- the branch arithmetic asks the REAL accept() ------------------------------------------


def test_branches_uses_the_gates_own_accept(ca):
    """A branch table that reimplements the bar is a second bar.

    Same reasoning as `feasibility` in `letta_reply_diff` — asserted here because the failure
    mode is silent: a local copy stays green forever while the real gate moves.
    """
    src = open(os.path.join(REPO, "scripts", "letta_comparability_audit.py")).read()
    body = src.split("def cmd_branches")[1].split("\ndef ")[0]
    assert "rd.accept(" in body
    assert "rd.memory_budget(" in body
    # no locally reimplemented threshold
    assert "0.9" not in body and "MIN_AGREEMENT =" not in body
