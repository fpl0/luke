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
    return mod


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
