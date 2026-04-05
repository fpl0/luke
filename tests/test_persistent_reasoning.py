"""Tests for persistent reasoning features."""

from __future__ import annotations

from unittest.mock import MagicMock

from luke.app import _extract_pending_actions, _extract_topics


class TestExtractPendingActions:
    def test_empty_input(self) -> None:
        assert _extract_pending_actions([]) == []

    def test_no_actions(self) -> None:
        assert _extract_pending_actions(["Hello, how are you?"]) == []

    def test_ill_pattern(self) -> None:
        result = _extract_pending_actions(["I'll work on the dashboard next"])
        assert len(result) >= 1
        assert "work on the dashboard" in result[0].lower()

    def test_next_step_pattern(self) -> None:
        result = _extract_pending_actions(["Next step: implement the event system"])
        assert len(result) >= 1

    def test_working_on_pattern(self) -> None:
        result = _extract_pending_actions(["Working on persistent reasoning now"])
        assert len(result) >= 1

    def test_max_five_actions(self) -> None:
        text = ". ".join(f"I'll do task number {i} which is important" for i in range(10))
        result = _extract_pending_actions([text])
        assert len(result) <= 5

    def test_dedup(self) -> None:
        result = _extract_pending_actions(["I'll fix the dashboard. I'll fix the dashboard again."])
        # Should not have duplicates
        assert len(result) == len(set(result))

    def test_multiple_texts(self) -> None:
        result = _extract_pending_actions(
            [
                "I'll handle the first thing.",
                "Next step: handle the second thing.",
            ]
        )
        assert len(result) >= 2


class TestExtractTopics:
    def test_empty_messages(self) -> None:
        msgs = [MagicMock(content="the the the")]
        result = _extract_topics(msgs, [])
        assert isinstance(result, list)

    def test_extracts_frequent_words(self) -> None:
        msgs = [MagicMock(content="dashboard dashboard dashboard fixes")]
        result = _extract_topics(msgs, [])
        assert "dashboard" in result

    def test_filters_stopwords(self) -> None:
        msgs = [MagicMock(content="the the the and and and")]
        result = _extract_topics(msgs, [])
        assert "the" not in result
        assert "and" not in result
