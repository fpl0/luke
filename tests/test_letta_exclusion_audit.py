"""The exclusion audit must FAIL when the tool surface grows past the pack's assumptions.

``letta_exclusion_audit.py`` exists to stop the 5.1R pack from staying narrow after the reason
for its narrowness goes away. That is only worth anything if the failing path actually fires,
so these tests drive it against synthetic tool surfaces: attach a web tool and the six web-
dependent rows must be named; attach nothing new and it must pass. Without the negative
controls the audit is indistinguishable from a script that always prints PASS.

The prompt corpus is a fixture, not the live database — the real archive would make these
tests drift with the message history.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import io
import os
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The real surface as of 2026-08-02: eight read-only tools plus the two memory editors.
REAL_SURFACE = {
    "luke_git", "luke_list_dir", "luke_list_tasks", "luke_read_file", "luke_recall",
    "luke_search_code", "luke_search_messages", "luke_tail_log",
    "memory_insert", "memory_replace",
}

# One prompt per capability domain, phrased the way Filipe actually phrases them.
FIXTURE_PROMPTS = [
    (1, "Check my email, they want to offer.", "email"),
    (2, "Actually, also check my calendar or that hourly check-in.", "calendar"),
    (3, "Pull the reviews from Cloudbeds. What is it like to work there?", "web"),
    (4, "Ok, remind me Wednesday to get a ticket. I want to watch this.", "write-schedule"),
]


@pytest.fixture()
def audit(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "letta_exclusion_audit", os.path.join(REPO, "scripts", "letta_exclusion_audit.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["letta_exclusion_audit"] = mod
    spec.loader.exec_module(mod)

    # Stub the reply-diff module the audit loads, so the corpus is the fixture above rather
    # than the live message archive.
    rd = types.SimpleNamespace(
        _db_path=lambda: ":memory:",
        LOOKBACK_DAYS=45,
        MIN_PROMPT_CHARS=25,
        _excluded=lambda p: "tool-dependent",
    )

    def _fake_spec(_name, _path):
        loader = types.SimpleNamespace(exec_module=lambda m: None)
        return types.SimpleNamespace(loader=loader)

    monkeypatch.setattr(mod.importlib.util, "spec_from_file_location", _fake_spec)
    monkeypatch.setattr(mod.importlib.util, "module_from_spec", lambda _s: rd)
    monkeypatch.setattr(mod.sqlite3, "connect", lambda *_a, **_k: _FakeCon())
    monkeypatch.setattr(mod, "_log", lambda _line: None)
    # The inclusion half reads the frozen 5.1R set. Point it at nothing by default so the
    # exclusion tests stay about exclusions; the inclusion tests write their own set.
    monkeypatch.setattr(mod, "SET_PATH", "/nonexistent/letta_reply_diff_set.json")
    # Start from an empty adjudication table so these tests assert the MECHANISM rather
    # than today's two real entries — which are a decision, and will change.
    monkeypatch.setattr(mod, "ADJUDICATED_INCLUSIONS", {})
    return mod


def _write_set(tmp_path, prompts):
    """A minimal frozen set — only msg_id/prompt/ts/bucket are read by the audit."""
    path = tmp_path / "set.json"
    path.write_text(json.dumps({
        "built": "2026-08-02T07:12:47+00:00",
        "prompts": [{"msg_id": mid, "prompt": text, "ts": "2026-07-01T10:00:00+00:00",
                     "bucket": "conversational"} for mid, text in prompts],
    }))
    return str(path)


class _FakeCon:
    def execute(self, _sql, _params):
        return types.SimpleNamespace(
            fetchall=lambda: [(mid, "Filipe", text, "2026-07-01T10:00:00+00:00")
                              for mid, text, _d in FIXTURE_PROMPTS]
        )

    def close(self):
        pass


def _run(mod, tools):
    mod.attached_tools = lambda: set(tools)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main()
    return rc, buf.getvalue()


def test_passes_on_the_real_surface(audit):
    """Every fixture row needs email/calendar/web/scheduling — none of which is attached."""
    rc, out = _run(audit, REAL_SURFACE)
    assert rc == 0, out
    assert "PASS" in out


def test_attaching_a_web_tool_flags_the_web_rows(audit):
    rc, out = _run(audit, REAL_SURFACE | {"luke_web_search"})
    assert rc == 1
    assert "#3 [web]" in out
    assert "#1 [email]" not in out  # unrelated domains stay excluded


@pytest.mark.parametrize(
    "tool,expect_id",
    [("luke_read_calendar", "#2 [calendar]"), ("luke_schedule_task", "#4 [write-schedule]")],
)
def test_each_domain_unlocks_only_its_own_rows(audit, tool, expect_id):
    rc, out = _run(audit, REAL_SURFACE | {tool})
    assert rc == 1
    assert expect_id in out


def test_unclassifiable_row_fails_until_adjudicated(audit, monkeypatch):
    """'I could not classify it' must never read as 'still fair to exclude'."""
    monkeypatch.setattr(
        _FakeCon, "execute",
        lambda self, _s, _p: types.SimpleNamespace(
            fetchall=lambda: [(99, "Filipe", "Oh my god can you just do the thing properly?",
                               "2026-07-01T10:00:00+00:00")]
        ),
    )
    rc, out = _run(audit, REAL_SURFACE)
    assert rc == 1
    assert "could not be classified" in out

    audit.ADJUDICATED[99] = "deictic complaint, measures the buffer not the substrate"
    rc, out = _run(audit, REAL_SURFACE)
    assert rc == 0, out
    assert "adjudicated" in out


def test_no_live_agent_refuses_rather_than_guessing(audit):
    """A hardcoded fallback surface is the exact stale assumption this script kills."""
    def _boom():
        raise OSError("connection refused")

    audit.attached_tools = _boom
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = audit.main()
    assert rc == 2
    assert "cannot read attached tools" in buf.getvalue()


def test_draft_for_human_is_not_a_send(audit):
    """The one filter bug the audit found: the send belongs to Filipe, so no tool is needed."""
    domain, _trigger = audit.domain_of("Now draft a message that I can send her")
    assert domain == "none-needed"
    assert audit.domain_of("Send her the message we discussed")[0] == "write-send"


# --- the inclusion half: the rows the filter KEPT -------------------------------------


def test_kept_row_needing_an_uncovered_domain_fails_until_adjudicated(audit, monkeypatch, tmp_path):
    """The whole point. A filter audited in one direction is wrong in one direction."""
    monkeypatch.setattr(audit, "SET_PATH", _write_set(tmp_path, [
        (2935, "Updated contract: https://apps.docusign.com/api/esign/Signing/Envelope"),
        (2973, "Look at our exchange as Karl Jung would. What is your analysis?"),
    ]))
    rc, out = _run(audit, REAL_SURFACE)
    assert rc == 1, out
    assert "#2935 [web]" in out
    assert "#2973" not in out.split("inclusion audit")[1]  # unclassified is not flagged

    audit.ADJUDICATED_INCLUSIONS[2935] = (audit.INCLUDE_PENDING, "no arm can open the URL")
    rc, out = _run(audit, REAL_SURFACE)
    assert rc == audit.INCLUDE_PENDING_EXIT, out
    assert "awaiting a written decision" in out


def test_attaching_the_covering_tool_clears_the_kept_row(audit, monkeypatch, tmp_path):
    """Negative control: the flag tracks the live tool surface, not a hardcoded row list."""
    monkeypatch.setattr(audit, "SET_PATH", _write_set(tmp_path, [
        (2935, "Updated contract: https://apps.docusign.com/api/esign/Signing/Envelope"),
    ]))
    # The exclusion half fails on a web tool (its fixture has a web row), so drive the
    # inclusion half directly — it is the assertion under test.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = audit.audit_included({"web", "repo-read"}, "stamp")
    assert rc == 0
    assert "no kept row needs an uncovered capability" in buf.getvalue()


def test_pending_exit_is_distinct_from_the_narrow_pack_failure(audit, monkeypatch, tmp_path):
    """A caller must be able to tell 'do not run' from 'run, but the bar is undecided'."""
    assert audit.INCLUDE_PENDING_EXIT != 1
    monkeypatch.setattr(audit, "SET_PATH", _write_set(tmp_path, [
        (3528, "The email draft is completely off. Why would I send back a summary?"),
    ]))
    audit.ADJUDICATED_INCLUSIONS[3528] = (audit.INCLUDE_PENDING, "no arm can read the body")
    rc, _out = _run(audit, REAL_SURFACE)
    assert rc == audit.INCLUDE_PENDING_EXIT
    # ...and the same corpus with a web tool attached fails on the OTHER half first,
    # proving the two codes are not interchangeable.
    rc2, _ = _run(audit, REAL_SURFACE | {"luke_web_search"})
    assert rc2 == 1


def test_missing_set_skips_rather_than_silently_passing(audit):
    """No frozen set means no claim — say so on screen instead of printing a green PASS."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = audit.audit_included({"repo-read"}, "stamp")
    assert rc == 0
    assert "INCLUSION AUDIT SKIPPED" in buf.getvalue()
