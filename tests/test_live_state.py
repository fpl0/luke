"""Tests for the injected live-state block.

The anchoring case is 2026-08-07 16:03, when Filipe reported a 7/10 headache
nineteen hours into a fast and I answered it flat. That moment is replayed
directly: if the block does not carry the fast at that timestamp, the fix does
not fix the thing it was built for.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from luke import live_state


@pytest.fixture
def anchor(tmp_path, monkeypatch):
    """Point the provider at a temp state file and return a writer for it."""
    health = tmp_path / "workspace" / "health"
    health.mkdir(parents=True)
    monkeypatch.setattr(
        type(live_state.settings),
        "workspace_dir",
        property(lambda _self: tmp_path / "workspace"),
    )

    def write(**payload):
        (health / "fast_state.json").write_text(json.dumps(payload))

    return write


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


# --- the real incident -------------------------------------------------


def test_headache_moment_carries_the_fast(anchor):
    """7 Aug 16:03 — the message was 'Why am I having headache?'"""
    anchor(started_at="2026-08-06T21:00:00+01:00", broke_at=None)
    out = live_state.render(now=_dt("2026-08-07T16:03:00+01:00"))
    assert "FASTING" in out
    assert "hour 19" in out
    assert "headache" in out.lower()


def test_after_the_break_it_says_not_fasting(anchor):
    """8 Aug 19:30 — the video shelf called him mid-fast. It should not have."""
    anchor(started_at="2026-08-06T21:00:00+01:00", broke_at="2026-08-08T13:52:00+01:00")
    out = live_state.render(now=_dt("2026-08-08T19:30:00+01:00"))
    assert "NOT FASTING" in out
    assert "hour 41" in out
    assert "refeeding" in out.lower()


def test_break_is_reported_not_the_elapsed_since_start(anchor):
    """The hour count after a break is the fast's LENGTH, never a running total."""
    anchor(started_at="2026-08-06T21:00:00+01:00", broke_at="2026-08-08T13:52:00+01:00")
    out = live_state.render(now=_dt("2026-08-08T21:00:00+01:00"))
    assert "hour 41" in out
    assert "hour 48" not in out


# --- silence, which is most of the time --------------------------------


def test_no_state_file_renders_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        type(live_state.settings), "workspace_dir", property(lambda _self: tmp_path / "nope")
    )
    assert live_state.render() == ""


def test_completed_fast_expires_and_stops_being_announced(anchor):
    anchor(started_at="2026-08-06T21:00:00+01:00", broke_at="2026-08-08T13:52:00+01:00")
    assert live_state.render(now=_dt("2026-08-10T13:00:00+01:00")) == ""


def test_stale_anchor_with_no_break_is_not_asserted_as_a_fast(anchor):
    """An eight-day-old anchor is a file nobody closed, not an eight-day fast."""
    anchor(started_at="2026-08-01T21:00:00+01:00", broke_at=None)
    assert live_state.render(now=_dt("2026-08-09T21:00:00+01:00")) == ""


def test_anchor_in_the_future_is_ignored(anchor):
    anchor(started_at="2026-08-20T21:00:00+01:00", broke_at=None)
    assert live_state.render(now=_dt("2026-08-09T21:00:00+01:00")) == ""


def test_break_before_start_is_ignored_as_incoherent(anchor):
    anchor(started_at="2026-08-08T21:00:00+01:00", broke_at="2026-08-06T13:52:00+01:00")
    out = live_state.render(now=_dt("2026-08-09T02:00:00+01:00"))
    assert "NOT FASTING" not in out


# --- it must never break a turn ----------------------------------------


def test_corrupt_json_is_silent(anchor, tmp_path):
    (tmp_path / "workspace" / "health" / "fast_state.json").write_text("{not json")
    assert live_state.render() == ""


def test_unparseable_timestamp_is_silent(anchor):
    anchor(started_at="sometime last Thursday", broke_at=None)
    assert live_state.render() == ""


def test_json_that_is_not_an_object_is_silent(anchor, tmp_path):
    (tmp_path / "workspace" / "health" / "fast_state.json").write_text('["nope"]')
    assert live_state.render() == ""


def test_a_raising_provider_does_not_take_down_the_block():
    def boom(_now):
        raise RuntimeError("provider is broken")

    def fine(_now):
        return "STILL HERE"

    out = live_state.render(providers=(boom, fine))
    assert "STILL HERE" in out


def test_all_providers_silent_means_no_header():
    assert live_state.render(providers=((lambda _now: None),)) == ""


def test_header_is_present_when_anything_is_live(anchor):
    anchor(started_at="2026-08-06T21:00:00+01:00", broke_at=None)
    out = live_state.render(now=_dt("2026-08-07T16:03:00+01:00"))
    assert out.startswith("[LIVE STATE")
    assert out.endswith("\n\n")
