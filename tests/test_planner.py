"""Unit tests for luke.planner — intent generation, priority, backoff, budget gating."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from luke.db import ensure_utc as _real_ensure_utc
from luke.planner import (
    BEHAVIOR_EVENTS,
    MAINTENANCE_BEHAVIORS,
    Intent,
    _deep_work_intents,
    _maintenance_intents,
    effective_interval,
    generate_intents,
    plan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_db(
    *,
    elapsed: dict[str, float] | None = None,
    unconsumed: dict[str, int] | None = None,
    no_ops: dict[str, int] | None = None,
    daily_cost: float = 0.0,
) -> MagicMock:
    """Build a mock db with configurable per-behavior elapsed times and events."""
    elapsed = elapsed or {}
    unconsumed = unconsumed or {}
    no_ops = no_ops or {}

    def _get_last_run(name: str) -> str | None:
        if name in elapsed:
            ts = datetime.now(UTC) - timedelta(seconds=elapsed[name])
            return ts.isoformat()
        return None  # never ran → infinite elapsed

    def _count_events(*event_types: str) -> int:
        return sum(unconsumed.get(e, 0) for e in event_types)

    m = MagicMock()
    m.get_behavior_last_run.side_effect = _get_last_run
    m.get_behavior_no_ops.side_effect = lambda name: no_ops.get(name, 0)
    m.count_unconsumed_events.side_effect = _count_events
    m.get_daily_deep_work_cost.return_value = daily_cost
    m.ensure_utc = _real_ensure_utc
    return m


def _mock_settings(**overrides: float) -> MagicMock:
    """Build mock settings with sane defaults. Override any field via kwargs."""
    defaults = {
        "episode_consolidation_interval": 21600.0,
        "consolidation_min_cluster": 2,
        "proactive_scan_interval": 10800.0,
        "reflection_interval": 604800.0,
        "insight_consolidation_interval": 604800.0,
        "feedback_consolidation_interval": 2592000.0,
        "lifecycle_review_interval": 2592000.0,
        "skill_extraction_interval": 21600.0,
        "dream_interval": 21600.0,
        "dream_max_budget_usd": 2.0,
        "deep_work_interval": 7200.0,
        "deep_work_max_budget_usd": 10.0,
        "daily_deep_work_budget_usd": 120.0,
        "behavior_max_budget_usd": 1.5,
    }
    defaults.update(overrides)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


# ---------------------------------------------------------------------------
# Intent model
# ---------------------------------------------------------------------------


class TestIntent:
    def test_frozen(self) -> None:
        intent = Intent("deep_work", 0.85, "goal", 10.0)
        with pytest.raises(AttributeError):
            intent.priority = 0.5  # type: ignore[misc]

    def test_default_context(self) -> None:
        intent = Intent("consolidation", 0.5, "event", 1.5)
        assert intent.context == {}

    def test_context_preserved(self) -> None:
        ctx = {"daily_remaining_usd": 50.0}
        intent = Intent("deep_work", 0.85, "goal", 10.0, context=ctx)
        assert intent.context["daily_remaining_usd"] == 50.0


# ---------------------------------------------------------------------------
# effective_interval (backoff)
# ---------------------------------------------------------------------------


class TestEffectiveInterval:
    def test_no_backoff(self) -> None:
        db = _mock_db(no_ops={"consolidation": 0})
        with patch("luke.planner.db", new=db):
            assert effective_interval("consolidation", 100.0) == 100.0

    def test_exponential_backoff(self) -> None:
        db = _mock_db(no_ops={"consolidation": 2})
        with patch("luke.planner.db", new=db):
            # 100 * 2^2 = 400
            assert effective_interval("consolidation", 100.0) == 400.0

    def test_cap_limits_backoff(self) -> None:
        # proactive_scan cap = 1 → max 2x
        db = _mock_db(no_ops={"proactive_scan": 10})
        with patch("luke.planner.db", new=db):
            assert effective_interval("proactive_scan", 100.0) == 200.0

    def test_default_cap_is_4(self) -> None:
        # Unknown behavior uses default cap 4 → max 16x
        db = _mock_db(no_ops={"skill_extraction": 99})
        with patch("luke.planner.db", new=db):
            assert effective_interval("skill_extraction", 100.0) == 1600.0

    def test_consolidation_cap_is_2(self) -> None:
        db = _mock_db(no_ops={"consolidation": 99})
        with patch("luke.planner.db", new=db):
            # cap=2 → 100 * 2^2 = 400
            assert effective_interval("consolidation", 100.0) == 400.0


# ---------------------------------------------------------------------------
# Deep work intents
# ---------------------------------------------------------------------------


class TestDeepWorkIntents:
    def test_not_due_yet(self) -> None:
        """No intent if deep_work ran recently."""
        db = _mock_db(elapsed={"deep_work": 3600.0})  # 1h < 2h interval
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            assert _deep_work_intents() == []

    def test_event_driven_high_priority(self) -> None:
        """Goal events → priority 0.85."""
        db = _mock_db(
            elapsed={"deep_work": 8000.0},
            unconsumed={"goal_updated": 1},
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            intents = _deep_work_intents()
            assert len(intents) == 1
            assert intents[0].priority == 0.85
            assert intents[0].source == "goal"

    def test_time_fallback_lower_priority(self) -> None:
        """No events but 2x overdue → priority 0.75."""
        db = _mock_db(
            elapsed={"deep_work": 15000.0},  # > 2 * 7200
            unconsumed={},
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            intents = _deep_work_intents()
            assert len(intents) == 1
            assert intents[0].priority == 0.75
            assert intents[0].source == "time"

    def test_no_intent_between_1x_and_2x_without_events(self) -> None:
        """Between interval and 2x interval, no events → no intent."""
        db = _mock_db(
            elapsed={"deep_work": 10000.0},  # between 7200 and 14400
            unconsumed={},
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            assert _deep_work_intents() == []

    def test_budget_exhausted(self) -> None:
        """Daily budget spent → no intent even if overdue."""
        db = _mock_db(
            elapsed={"deep_work": 100000.0},
            unconsumed={"goal_updated": 5},
            daily_cost=120.0,
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            assert _deep_work_intents() == []

    def test_budget_caps_intent_cost(self) -> None:
        """Intent budget_usd is min(max_budget, remaining)."""
        db = _mock_db(
            elapsed={"deep_work": 100000.0},
            unconsumed={"goal_updated": 1},
            daily_cost=115.0,  # only $5 remaining
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            intents = _deep_work_intents()
            assert len(intents) == 1
            assert intents[0].budget_usd == 5.0  # min(10.0, 5.0)

    def test_never_ran_triggers_intent(self) -> None:
        """If deep_work never ran (elapsed=inf), intent fires."""
        db = _mock_db(elapsed={}, unconsumed={"goal_updated": 1})  # no entry → inf
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            intents = _deep_work_intents()
            assert len(intents) == 1


# ---------------------------------------------------------------------------
# Maintenance intents
# ---------------------------------------------------------------------------


class TestMaintenanceIntents:
    def test_consolidation_with_events(self) -> None:
        """Consolidation fires at event priority when episodes exist."""
        db = _mock_db(
            elapsed={"consolidation": 25000.0},  # > 21600
            unconsumed={"new_episode": 3},
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            intents = _maintenance_intents()
            consolidation = [i for i in intents if i.kind == "consolidation"]
            assert len(consolidation) == 1
            assert consolidation[0].priority == 0.50
            assert consolidation[0].source == "event"

    def test_consolidation_time_fallback(self) -> None:
        """Consolidation fires at lower priority after 2x interval without events."""
        db = _mock_db(
            elapsed={"consolidation": 50000.0},  # > 21600 * 2
            unconsumed={},
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            intents = _maintenance_intents()
            consolidation = [i for i in intents if i.kind == "consolidation"]
            assert len(consolidation) == 1
            assert consolidation[0].priority == 0.35
            assert consolidation[0].source == "time"

    def test_consolidation_below_min_cluster(self) -> None:
        """Not enough episodes and not overdue enough → no intent."""
        db = _mock_db(
            elapsed={"consolidation": 25000.0},  # > interval but < 2x
            unconsumed={"new_episode": 1},  # < min_cluster (2)
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            intents = _maintenance_intents()
            assert not [i for i in intents if i.kind == "consolidation"]

    def test_proactive_scan_event_driven(self) -> None:
        db = _mock_db(
            elapsed={"proactive_scan": 12000.0},
            unconsumed={"goal_updated": 1},
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            intents = _maintenance_intents()
            scan = [i for i in intents if i.kind == "proactive_scan"]
            assert len(scan) == 1
            assert scan[0].priority == 0.55

    def test_proactive_scan_time_fallback_at_3x(self) -> None:
        """Proactive scan time fallback requires 3x interval."""
        db = _mock_db(
            elapsed={"proactive_scan": 35000.0},  # > 10800 * 3
            unconsumed={},
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            intents = _maintenance_intents()
            scan = [i for i in intents if i.kind == "proactive_scan"]
            assert len(scan) == 1
            assert scan[0].source == "time"

    def test_dream_uses_dream_budget(self) -> None:
        """Dream intent uses dream_max_budget_usd, not behavior_max_budget_usd."""
        db = _mock_db(
            elapsed={"dream": 25000.0},
            unconsumed={"new_insight": 1},
        )
        s = _mock_settings(dream_max_budget_usd=2.0, behavior_max_budget_usd=1.5)
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            intents = _maintenance_intents()
            dream = [i for i in intents if i.kind == "dream"]
            assert len(dream) == 1
            assert dream[0].budget_usd == 2.0

    def test_nothing_due(self) -> None:
        """Nothing is due → empty list."""
        db = _mock_db(
            elapsed={
                "consolidation": 100.0,
                "proactive_scan": 100.0,
                "reflection": 100.0,
                "insight_consolidation": 100.0,
                "feedback_consolidation": 100.0,
                "lifecycle_review": 100.0,
                "skill_extraction": 100.0,
                "dream": 100.0,
            },
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            assert _maintenance_intents() == []

    def test_multiple_behaviors_due(self) -> None:
        """Multiple behaviors can fire in one tick."""
        db = _mock_db(
            elapsed={
                "consolidation": 50000.0,
                "proactive_scan": 35000.0,
            },
            unconsumed={"new_episode": 5, "goal_updated": 2},
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            intents = _maintenance_intents()
            kinds = {i.kind for i in intents}
            assert "consolidation" in kinds
            assert "proactive_scan" in kinds


# ---------------------------------------------------------------------------
# generate_intents + plan
# ---------------------------------------------------------------------------


class TestGenerateAndPlan:
    def test_generate_combines_sources(self) -> None:
        """generate_intents merges deep_work and maintenance."""
        db = _mock_db(
            elapsed={"deep_work": 100000.0, "consolidation": 50000.0},
            unconsumed={"goal_updated": 1, "new_episode": 5},
        )
        s = _mock_settings()
        with patch("luke.planner.db", new=db), patch("luke.planner.settings", new=s):
            intents = generate_intents()
            kinds = {i.kind for i in intents}
            assert "deep_work" in kinds
            assert "consolidation" in kinds

    def test_plan_partitions_correctly(self) -> None:
        """plan() separates maintenance from deep_work."""
        intents = [
            Intent("consolidation", 0.50, "event", 1.5),
            Intent("proactive_scan", 0.55, "event", 1.5),
            Intent("deep_work", 0.85, "goal", 10.0),
        ]
        maint, dw = plan(intents)
        assert len(maint) == 2
        assert dw is not None
        assert dw.kind == "deep_work"

    def test_plan_no_deep_work(self) -> None:
        """plan() returns None for deep_work when none present."""
        intents = [
            Intent("consolidation", 0.50, "event", 1.5),
        ]
        maint, dw = plan(intents)
        assert len(maint) == 1
        assert dw is None

    def test_plan_maintenance_sorted_by_priority(self) -> None:
        """Maintenance intents come out highest-priority first."""
        intents = [
            Intent("dream", 0.20, "time", 2.0),
            Intent("proactive_scan", 0.55, "event", 1.5),
            Intent("consolidation", 0.35, "time", 1.5),
        ]
        maint, _ = plan(intents)
        priorities = [i.priority for i in maint]
        assert priorities == sorted(priorities, reverse=True)

    def test_plan_empty(self) -> None:
        """Empty intents → empty results."""
        maint, dw = plan([])
        assert maint == []
        assert dw is None


# ---------------------------------------------------------------------------
# Behavior events mapping completeness
# ---------------------------------------------------------------------------


class TestBehaviorEventsMapping:
    def test_all_maintenance_behaviors_have_events_entry(self) -> None:
        """Every maintenance behavior must appear in BEHAVIOR_EVENTS."""
        for b in MAINTENANCE_BEHAVIORS:
            assert b in BEHAVIOR_EVENTS, f"{b} missing from BEHAVIOR_EVENTS"

    def test_no_extra_behaviors_in_events_map(self) -> None:
        """BEHAVIOR_EVENTS shouldn't list behaviors that aren't maintenance."""
        for b in BEHAVIOR_EVENTS:
            assert b in MAINTENANCE_BEHAVIORS, f"{b} in BEHAVIOR_EVENTS but not MAINTENANCE_BEHAVIORS"
