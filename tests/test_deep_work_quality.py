"""Tests for deep work quality tracking (db + behaviors)."""

from __future__ import annotations

from typing import Any


class TestLogDeepWorkQuality:
    def test_log_valid_rating(self, test_db: Any) -> None:
        test_db.log_deep_work_quality("goal-test", 4)
        scores = test_db.get_recent_quality_scores("goal-test")
        assert scores == [4]

    def test_log_multiple_ratings(self, test_db: Any) -> None:
        test_db.log_deep_work_quality("goal-test", 3)
        test_db.log_deep_work_quality("goal-test", 5)
        test_db.log_deep_work_quality("goal-test", 2)
        scores = test_db.get_recent_quality_scores("goal-test")
        assert scores == [2, 5, 3]  # newest first

    def test_log_invalid_rating_ignored(self, test_db: Any) -> None:
        test_db.log_deep_work_quality("goal-test", 0)
        test_db.log_deep_work_quality("goal-test", 6)
        scores = test_db.get_recent_quality_scores("goal-test")
        assert scores == []

    def test_log_boundary_ratings(self, test_db: Any) -> None:
        test_db.log_deep_work_quality("goal-test", 1)
        test_db.log_deep_work_quality("goal-test", 5)
        scores = test_db.get_recent_quality_scores("goal-test")
        assert scores == [5, 1]

    def test_separate_goals(self, test_db: Any) -> None:
        test_db.log_deep_work_quality("goal-a", 4)
        test_db.log_deep_work_quality("goal-b", 2)
        assert test_db.get_recent_quality_scores("goal-a") == [4]
        assert test_db.get_recent_quality_scores("goal-b") == [2]


class TestGetRecentQualityScores:
    def test_empty(self, test_db: Any) -> None:
        assert test_db.get_recent_quality_scores("nonexistent") == []

    def test_limit(self, test_db: Any) -> None:
        for i in range(1, 6):
            test_db.log_deep_work_quality("goal-test", i)
        scores = test_db.get_recent_quality_scores("goal-test", 3)
        assert len(scores) == 3
        assert scores == [5, 4, 3]  # newest first

    def test_fewer_than_requested(self, test_db: Any) -> None:
        test_db.log_deep_work_quality("goal-test", 3)
        scores = test_db.get_recent_quality_scores("goal-test", 5)
        assert scores == [3]


class TestGetQualityBlockedGoals:
    def test_no_data(self, test_db: Any) -> None:
        assert test_db.get_quality_blocked_goals() == []

    def test_high_quality_not_blocked(self, test_db: Any) -> None:
        for _ in range(3):
            test_db.log_deep_work_quality("goal-good", 4)
        assert test_db.get_quality_blocked_goals() == []

    def test_low_quality_blocked(self, test_db: Any) -> None:
        for _ in range(3):
            test_db.log_deep_work_quality("goal-bad", 1)
        blocked = test_db.get_quality_blocked_goals()
        assert "goal-bad" in blocked

    def test_borderline_not_blocked(self, test_db: Any) -> None:
        # avg exactly 2.0 should NOT be blocked (threshold is strictly <)
        test_db.log_deep_work_quality("goal-border", 2)
        test_db.log_deep_work_quality("goal-border", 2)
        test_db.log_deep_work_quality("goal-border", 2)
        assert test_db.get_quality_blocked_goals() == []

    def test_not_enough_data_not_blocked(self, test_db: Any) -> None:
        # Only 2 ratings, need 3
        test_db.log_deep_work_quality("goal-new", 1)
        test_db.log_deep_work_quality("goal-new", 1)
        assert test_db.get_quality_blocked_goals() == []

    def test_mixed_goals(self, test_db: Any) -> None:
        for _ in range(3):
            test_db.log_deep_work_quality("goal-good", 5)
            test_db.log_deep_work_quality("goal-bad", 1)
        blocked = test_db.get_quality_blocked_goals()
        assert "goal-bad" in blocked
        assert "goal-good" not in blocked

    def test_custom_threshold(self, test_db: Any) -> None:
        for _ in range(3):
            test_db.log_deep_work_quality("goal-mid", 3)
        # Not blocked at default threshold (2.0)
        assert test_db.get_quality_blocked_goals() == []
        # Blocked at higher threshold (3.5)
        blocked = test_db.get_quality_blocked_goals(threshold=3.5)
        assert "goal-mid" in blocked


class TestParsePlanStatus:
    def test_in_progress(self, tmp_settings: Any) -> None:
        from luke.behaviors import _parse_plan_status

        plans_dir = tmp_settings.workspace_dir / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / "goal-test.md").write_text(
            "# Test Plan\n\n**Status:** in_progress\n**Created:** 2026-04-01\n"
        )
        assert _parse_plan_status("goal-test") == "in_progress"

    def test_paused(self, tmp_settings: Any) -> None:
        from luke.behaviors import _parse_plan_status

        plans_dir = tmp_settings.workspace_dir / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / "goal-test.md").write_text("# Test Plan\n\n**Status:** paused\n")
        assert _parse_plan_status("goal-test") == "paused"

    def test_completed(self, tmp_settings: Any) -> None:
        from luke.behaviors import _parse_plan_status

        plans_dir = tmp_settings.workspace_dir / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / "goal-test.md").write_text("# Test Plan\n\n**Status:** completed\n")
        assert _parse_plan_status("goal-test") == "completed"

    def test_no_plan_file(self, tmp_settings: Any) -> None:
        from luke.behaviors import _parse_plan_status

        assert _parse_plan_status("goal-nonexistent") is None

    def test_no_status_in_file(self, tmp_settings: Any) -> None:
        from luke.behaviors import _parse_plan_status

        plans_dir = tmp_settings.workspace_dir / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / "goal-test.md").write_text("# Just a title\nNo status here.\n")
        assert _parse_plan_status("goal-test") is None
