"""Tests for the send-time state reconciliation gate.

Every case here is a real message from 2026-08-08, the night three automated
messages in a row told Filipe he was mid-fast after he had broken it at 13:52
and said so. The freshness gate could not catch these: its window is 15
minutes and the break was 7h29m old by the 21:00 check-in.
"""

from datetime import UTC, datetime

import pytest

from luke.state_reconcile import RULES, block_reason, reconcile

NOW = datetime(2026, 8, 8, 20, 0, tzinfo=UTC)  # 21:00 Dublin, when it fired


def msg(content: str, hh: int, mm: int = 0) -> dict:
    return {"content": content, "timestamp": datetime(2026, 8, 8, hh, mm, tzinfo=UTC)}


BREAK = msg("I just broke the fasfastingting", 12, 52)  # his actual typo


class TestTheEightAugustMiss:
    def test_evening_checkin_hour_count_is_blocked(self):
        v = reconcile(
            "Coming up on hour 47 of the fast, if my arithmetic's holding.", [BREAK], now=NOW
        )
        assert v.blocked
        assert v.rule == "fast-already-broken"

    def test_video_shelf_midfast_framing_is_blocked(self):
        v = reconcile(
            "You're mid-fast with Monday coming, so nothing that feels like homework.",
            [BREAK],
            now=NOW,
        )
        assert v.blocked

    def test_block_reason_quotes_him_and_names_the_tool(self):
        r = block_reason(reconcile("hour 47 of the fast", [BREAK], now=NOW))
        assert "broke the fas" in r
        assert "fast_state.py" in r
        assert "Rewrite" in r

    def test_seven_hours_stale_still_bites(self):
        """The whole point: recency must be irrelevant."""
        v = reconcile("still fasting, hour 47", [BREAK], now=NOW)
        assert v.blocked
        age = (NOW - BREAK["timestamp"]).total_seconds() / 3600
        assert age > 7, "this case is only meaningful if it is far outside the 15-min window"


class TestDoesNotOverblock:
    def test_silent_when_draft_says_nothing_about_fasting(self):
        assert not reconcile("How was the rest of the day?", [BREAK], now=NOW).blocked

    def test_past_tense_discussion_of_the_broken_fast_is_fine(self):
        v = reconcile("41 hours is your longest yet, and you called it yourself.", [BREAK], now=NOW)
        assert not v.blocked

    def test_no_revocation_means_an_hour_count_is_allowed(self):
        v = reconcile(
            "Coming up on hour 20 of the fast.", [msg("Started at 21:00 last night", 8)], now=NOW
        )
        assert not v.blocked

    @pytest.mark.parametrize(
        "q",
        [
            "Will black tea break my fast?",
            "Is it too late to break the fast tomorrow at 21:00?",
            "How much time have I fasted?",
        ],
    )
    def test_questions_never_revoke(self, q):
        assert not reconcile("hour 47 of the fast", [msg(q, 11)], now=NOW).blocked

    def test_yesterdays_break_does_not_kill_todays_fast(self):
        y = {
            "content": "I just broke the fast",
            "timestamp": datetime(2026, 8, 7, 18, 0, tzinfo=UTC),
        }
        assert not reconcile("hour 12 of the fast", [y], now=NOW).blocked

    def test_empty_draft_and_no_history_are_silent(self):
        assert not reconcile("", [BREAK], now=NOW).blocked
        assert not reconcile("hour 47 of the fast", [], now=NOW).blocked


class TestFailsOpen:
    def test_unparseable_timestamps_do_not_block(self):
        bad = [{"content": "I just broke the fast", "timestamp": "not-a-date"}]
        assert not reconcile("hour 47 of the fast", bad, now=NOW).blocked

    def test_missing_keys_do_not_raise(self):
        assert not reconcile("hour 47 of the fast", [{}, {"content": None}], now=NOW).blocked

    def test_internal_error_fails_open_with_a_recorded_reason(self):
        class Exploding:
            def search(self, _):
                raise RuntimeError("boom")

        from luke.state_reconcile import Rule

        bad = (Rule("x", Exploding(), Exploding(), "g"),)
        v = reconcile("anything", [BREAK], now=NOW, rules=bad)
        assert not v.blocked and "state-reconcile-error" in v.error


class TestOtherRules:
    def test_already_called_him_blocks_a_reminder_to_call(self):
        v = reconcile(
            "Don't forget to call Christopher tonight.",
            [msg("I just called him. It all good.", 18, 35)],
            now=NOW,
        )
        assert v.blocked and v.rule == "already-done-it"

    def test_asking_how_the_call_went_is_still_fine(self):
        v = reconcile(
            "How did the call with Christopher go?",
            [msg("I just called him. It all good.", 18, 35)],
            now=NOW,
        )
        assert not v.blocked

    def test_every_rule_has_guidance(self):
        assert all(r.guidance for r in RULES)


class TestFoundByReplayingTheRealLog:
    """Both fixed after running the gate over the actual 8 Aug message log —
    synthetic cases passed while these two were wrong."""

    def test_cites_the_earliest_revocation_not_the_angry_repeat(self):
        msgs = [BREAK, msg("LUKE I BROKE THE FAST ALREADY!", 20, 14)]
        v = reconcile("Coming up on hour 47 of the fast.", msgs, now=NOW)
        assert v.blocked
        assert "just broke the fas" in v.quote, "must quote 13:52, not the 21:14 shout"
        assert "12:52" in block_reason(v) or "13:52" in block_reason(v)

    def test_my_own_correction_is_not_blocked(self):
        """'You broke it at 13:52, hour 41' is the RIGHT message to send."""
        v = reconcile(
            "Fair hit. You broke it at 13:52 today, hour 41 — and the shelf still went out.",
            [BREAK],
            now=NOW,
        )
        assert not v.blocked

    def test_explaining_the_stale_message_is_not_blocked(self):
        """Quoting the bad phrase while explaining the bug must not trip the gate."""
        v = reconcile(
            "The cron wrote 'mid-fast' without checking the clock. That's the bug.",
            [BREAK],
            now=NOW,
        )
        assert not v.blocked

    def test_present_tense_assertion_still_blocked(self):
        for draft in [
            "You're still fasting, so go gently.",
            "Coming up on hour 47.",
            "mid-fast with Monday coming",
        ]:
            assert reconcile(draft, [BREAK], now=NOW).blocked, draft

    def test_quoted_stale_phrase_while_explaining_the_bug_is_not_blocked(self):
        v = reconcile(
            'It wrote the "mid-fast" line without checking the clock. That is the bug.',
            [BREAK],
            now=NOW,
        )
        assert not v.blocked

    def test_code_fenced_stale_phrase_is_not_blocked(self):
        v = reconcile(
            "The cron said <code>hour 47 of the fast</code> and that was wrong.", [BREAK], now=NOW
        )
        assert not v.blocked

    def test_but_an_unquoted_assertion_beside_a_quote_still_blocks(self):
        v = reconcile(
            'It wrote the "mid-fast" line. Anyway, you\'re still fasting so go gently.',
            [BREAK],
            now=NOW,
        )
        assert v.blocked
