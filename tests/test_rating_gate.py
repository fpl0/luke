"""Tests for the deep-work rating gate.

The BLOCKED cases are the three real ratings from 2026-09-02 that this gate was
built for — sessions rated 4 with no send anywhere in their window. The PASSES
are the neighbours that must stay untouched: a 3 on a quiet session is the
honest rating the protocol asks for, not something to interrupt, and the
one-shot override has to work or the gate becomes a cap on the scale instead of
a prompt to decide.
"""

from __future__ import annotations

import pytest

from luke.rating_gate import REASON, TOOL_NAME, blocks

# The real rows behind this gate: rated 4, nothing shipped in the session.
UNLANDED_FOURS = [
    {"goal_id": "goal-boston-trip-report", "rating": 4},
    {"goal_id": "goal-irish-citizenship", "rating": 4},
    {"goal_id": "goal-director-of-engineering-2028", "rating": 4},
    {"goal_id": "weekly-review-2026-09-01", "rating": 5},
]


@pytest.mark.parametrize("tool_input", UNLANDED_FOURS)
def test_blocks_unlanded_high_rating(tool_input):
    assert blocks(TOOL_NAME, tool_input, shipped=False, already_fired=False)


@pytest.mark.parametrize("tool_input", UNLANDED_FOURS)
def test_same_rating_passes_once_something_shipped(tool_input):
    assert not blocks(TOOL_NAME, tool_input, shipped=True, already_fired=False)


@pytest.mark.parametrize("tool_input", UNLANDED_FOURS)
def test_one_shot_override_lets_the_recall_through(tool_input):
    """The whole design: it costs one considered decision, never the rating."""
    assert not blocks(TOOL_NAME, tool_input, shipped=False, already_fired=True)


@pytest.mark.parametrize("rating", [1, 2, 3])
def test_low_ratings_are_never_gated(rating):
    """A 3 on a quiet session IS the fix. Interrupting it would invert the gate."""
    assert not blocks(
        TOOL_NAME, {"goal_id": "g", "rating": rating}, shipped=False, already_fired=False
    )


def test_other_tools_untouched():
    assert not blocks("mcp__luke__send_message", {"rating": 5}, shipped=False, already_fired=False)


@pytest.mark.parametrize("bad", [None, "not-a-dict", 42, []])
def test_non_dict_input_is_not_a_block(bad):
    assert not blocks(TOOL_NAME, bad, shipped=False, already_fired=False)


@pytest.mark.parametrize("bad", [{}, {"rating": None}, {"rating": "four"}, {"goal_id": "g"}])
def test_unreadable_rating_never_blocks(bad):
    """Fail open on a malformed call — this gate must not be the thing that
    stops a session from recording anything at all."""
    assert not blocks(TOOL_NAME, bad, shipped=False, already_fired=False)


def test_string_rating_still_gated():
    """MCP integer coercion is not guaranteed upstream; '4' is still a 4."""
    assert blocks(TOOL_NAME, {"goal_id": "g", "rating": "4"}, shipped=False, already_fired=False)


def test_reason_carries_the_protocol_sentence_and_the_way_out():
    assert "not a 4" in REASON
    assert "re-call with the same rating" in REASON
    assert "fires once per run" in REASON
