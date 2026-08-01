"""Tests for luke.behaviors — consolidation, reflection, proactive scan, goal execution."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

_SEM = asyncio.Semaphore(5)

# ---------------------------------------------------------------------------
# run_consolidation
# ---------------------------------------------------------------------------


class TestRunConsolidation:
    async def test_no_chat_id(self) -> None:
        from luke.behaviors import run_consolidation

        with patch("luke.behaviors.settings") as mock_settings:
            mock_settings.chat_id = ""
            await run_consolidation(AsyncMock(), _SEM)

    async def test_no_clusters(self) -> None:
        from luke.behaviors import run_consolidation

        with (
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.consolidation_min_cluster = 3
            mock_memory.recluster_offline.return_value = {"n_clusters": 0}
            mock_memory.get_consolidation_candidates.return_value = []
            await run_consolidation(AsyncMock(), _SEM)

    async def test_with_clusters(self, tmp_settings: Any) -> None:
        from luke.behaviors import run_consolidation

        mem_dir = tmp_settings.memory_dir / "episodes"
        mem_dir.mkdir(parents=True, exist_ok=True)
        # Create episode files
        for i in range(3):
            (mem_dir / f"ep{i}.md").write_text(
                f"---\nid: ep{i}\ntype: episode\n---\n\n# Episode {i}\n\nContent {i}"
            )

        cluster = [
            {"id": f"ep{i}", "tags": {"a", "b", "c"}, "links": set(), "created": "", "updated": ""}
            for i in range(3)
        ]

        with (
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.run_agent", new_callable=AsyncMock) as mock_agent,
        ):
            mock_memory.recluster_offline.return_value = {"n_clusters": 0}
            mock_memory.get_consolidation_candidates.return_value = [cluster]
            mock_agent.return_value = MagicMock(texts=[])
            await run_consolidation(AsyncMock(), _SEM)

        mock_agent.assert_called_once()

    async def test_agent_exception_handled(self) -> None:
        from luke.behaviors import run_consolidation

        cluster = [
            {"id": f"ep{i}", "tags": set(), "links": set(), "created": "", "updated": ""}
            for i in range(3)
        ]

        with (
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.run_agent", side_effect=RuntimeError("agent error")),
            patch("luke.behaviors.read_memory_body", return_value="content"),
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.consolidation_min_cluster = 3
            mock_settings.max_consolidation_clusters = 3
            mock_settings.agent_timeout = 10
            mock_memory.recluster_offline.return_value = {"n_clusters": 0}
            mock_memory.get_consolidation_candidates.return_value = [cluster]
            # Should not raise
            await run_consolidation(AsyncMock(), _SEM)

    async def test_empty_contents_skips(self) -> None:
        from luke.behaviors import run_consolidation

        cluster = [
            {"id": f"ep{i}", "tags": set(), "links": set(), "created": "", "updated": ""}
            for i in range(3)
        ]

        with (
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.read_memory_body", return_value=""),
            patch("luke.behaviors.settings") as mock_settings,
            patch("luke.behaviors.run_agent", new_callable=AsyncMock) as mock_agent,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.consolidation_min_cluster = 3
            mock_settings.max_consolidation_clusters = 3
            mock_memory.recluster_offline.return_value = {"n_clusters": 0}
            mock_memory.get_consolidation_candidates.return_value = [cluster]
            await run_consolidation(AsyncMock(), _SEM)

        mock_agent.assert_not_called()


# ---------------------------------------------------------------------------
# run_reflection
# ---------------------------------------------------------------------------


class TestRunReflection:
    async def test_no_chat_id(self) -> None:
        from luke.behaviors import run_reflection

        with patch("luke.behaviors.settings") as mock_settings:
            mock_settings.chat_id = ""
            await run_reflection(AsyncMock(), _SEM)

    async def test_no_recent_memories(self) -> None:
        from luke.behaviors import run_reflection

        with (
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_memory.recall_by_time_window.return_value = []
            await run_reflection(AsyncMock(), _SEM)

    async def test_with_memories(self, tmp_settings: Any) -> None:
        from luke.behaviors import run_reflection

        mem_dir = tmp_settings.memory_dir / "episodes"
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / "ep1.md").write_text("---\nid: ep1\ntype: episode\n---\n\n# Ep1\n\nContent")

        memories = [{"id": "ep1", "type": "episode", "title": "Ep1", "score": 1.0}]

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.run_agent", new_callable=AsyncMock) as mock_agent,
        ):
            mock_memory.recall_by_time_window.return_value = memories
            mock_db.get_recent_messages.return_value = [
                {"sender_name": "User", "timestamp": "2024-01-01", "content": "Hello"}
            ]
            mock_db.get_reactions.return_value = []
            mock_db.get_reaction_summary.return_value = {
                "total": 0,
                "sentiments": {},
                "top_emojis": [],
                "by_sender": {},
                "period_days": 7,
            }
            mock_agent.return_value = MagicMock(texts=[])
            await run_reflection(AsyncMock(), _SEM)

        mock_agent.assert_called_once()

    async def test_agent_exception_handled(self) -> None:
        from luke.behaviors import run_reflection

        memories = [{"id": "ep1", "type": "episode", "title": "Ep1", "score": 1.0}]

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.run_agent", side_effect=RuntimeError("error")),
            patch("luke.behaviors.read_memory_body", return_value="content"),
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.agent_timeout = 10
            mock_settings.assistant_name = "Luke"
            mock_memory.recall_by_time_window.return_value = memories
            mock_db.get_recent_messages.return_value = []
            mock_db.get_reactions.return_value = []
            mock_db.get_reaction_summary.return_value = {
                "total": 0,
                "sentiments": {},
                "top_emojis": [],
                "by_sender": {},
                "period_days": 7,
            }
            await run_reflection(AsyncMock(), _SEM)


# ---------------------------------------------------------------------------
# run_proactive_scan
# ---------------------------------------------------------------------------


class TestRunProactiveScan:
    async def test_no_chat_id(self) -> None:
        from luke.behaviors import run_proactive_scan

        with patch("luke.behaviors.settings") as mock_settings:
            mock_settings.chat_id = ""
            await run_proactive_scan(AsyncMock(), _SEM)

    async def test_no_sections(self) -> None:
        from luke.behaviors import run_proactive_scan

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_memory.recall.return_value = []
            mock_db.get_message_summaries.return_value = []
            await run_proactive_scan(AsyncMock(), _SEM)

    async def test_with_goals_and_insights(self, tmp_settings: Any) -> None:
        from luke.behaviors import run_proactive_scan

        (tmp_settings.memory_dir / "goals").mkdir(parents=True, exist_ok=True)
        (tmp_settings.memory_dir / "goals" / "g1.md").write_text(
            "---\nid: g1\ntype: goal\n---\n\n# Goal 1\n\nFinish project"
        )
        (tmp_settings.memory_dir / "insights").mkdir(parents=True, exist_ok=True)
        (tmp_settings.memory_dir / "insights" / "i1.md").write_text(
            "---\nid: i1\ntype: insight\n---\n\n# Insight 1\n\nPattern"
        )

        goals = [{"id": "g1", "type": "goal", "title": "Goal 1", "score": 1.0}]
        insights = [{"id": "i1", "type": "insight", "title": "Insight 1", "score": 1.0}]

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.run_agent", new_callable=AsyncMock) as mock_agent,
        ):
            mock_memory.recall.side_effect = [goals, insights, []]
            mock_db.get_message_summaries.return_value = [
                {"date": "2024-01-01", "messages": ["User: hi"]}
            ]
            mock_agent.return_value = MagicMock(texts=[])
            await run_proactive_scan(AsyncMock(), _SEM)

        mock_agent.assert_called_once()

    async def test_proactive_scan_passes_urgent_true(self, tmp_settings: Any) -> None:
        """Proactive scan must pass urgent=True so it can draw from the attention reserve."""
        from luke.behaviors import run_proactive_scan

        goals = [{"id": "g1", "type": "goal", "title": "Goal 1", "score": 1.0}]

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.read_memory_body", return_value="Goal content"),
            patch("luke.behaviors.run_agent", new_callable=AsyncMock) as mock_agent,
        ):
            mock_memory.recall.side_effect = [goals, [], []]
            mock_db.get_message_summaries.return_value = []
            mock_agent.return_value = MagicMock(texts=[])
            await run_proactive_scan(AsyncMock(), _SEM)

        call_kwargs = mock_agent.call_args.kwargs
        assert call_kwargs.get("urgent") is True

    async def test_agent_exception_handled(self) -> None:
        from luke.behaviors import run_proactive_scan

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.run_agent", side_effect=RuntimeError("err")),
            patch("luke.behaviors.read_memory_body", return_value="content"),
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.agent_timeout = 10
            mock_memory.recall.return_value = [
                {"id": "g1", "type": "goal", "title": "G1", "score": 1.0}
            ]
            mock_db.get_message_summaries.return_value = []
            await run_proactive_scan(AsyncMock(), _SEM)


# ---------------------------------------------------------------------------
# run_reflexion
# ---------------------------------------------------------------------------


class TestRunReflexion:
    async def test_no_chat_id(self) -> None:
        from luke.behaviors import run_reflexion

        with patch("luke.behaviors.settings") as mock_settings:
            mock_settings.chat_id = ""
            await run_reflexion(AsyncMock(), _SEM)

    async def _capture_prompt(self, payload: dict[str, Any]) -> str:
        from luke.behaviors import run_reflexion

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.bus"),
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.read_memory_body", return_value=""),
            patch("luke.behaviors._run_behavior", new_callable=AsyncMock) as mock_run_behavior,
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.consolidation_model = "sonnet"
            mock_db.get_recent_quality_scores.return_value = []
            mock_db.count_events_matching.return_value = 0  # not saturated
            mock_memory.recall.return_value = []
            await run_reflexion(AsyncMock(), _SEM, event_kind="quality_low", event_payload=payload)

        mock_run_behavior.assert_called_once()
        prompt = mock_run_behavior.call_args.args[1]
        assert isinstance(prompt, str)
        return prompt

    async def test_prompt_contains_counterfactual_question(self) -> None:
        prompt = await self._capture_prompt({"goal_id": "g1", "reason": "low quality"})
        # Counterfactual question must be present with "smallest change" phrasing.
        assert "Counterfactual" in prompt
        assert "SMALLEST change" in prompt
        assert "flipped" in prompt

    async def test_prompt_instructs_counterfactual_tags_and_link(self) -> None:
        prompt = await self._capture_prompt({"goal_id": "g1", "reason": "low quality"})
        # Counterfactual save instructions must reference the right tags and link
        # back to the parent reflexion insight.
        assert "'counterfactual', 'g1'" in prompt
        assert "counterfactual-g1-" in prompt
        assert "reflexion-g1-" in prompt
        assert "links=" in prompt
        assert "importance=1.3" in prompt

    async def test_prompt_keeps_existing_reflexion_save(self) -> None:
        prompt = await self._capture_prompt({"goal_id": "g1", "reason": "low quality"})
        # The original reflexion-save instruction must still be present.
        assert "'reflexion', 'g1'" in prompt
        assert "Root cause" in prompt

    # --- Gate 1: empty-payload circuit-breaker -----------------------------
    async def test_empty_payload_skips_analysis(self) -> None:
        """`continuation_failure {}` carries no diagnostic signal — file once,
        do not spawn the analysis (insight-reflexion-loop-structural-failure)."""
        from luke.behaviors import run_reflexion

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.bus") as mock_bus,
            patch("luke.behaviors._run_behavior", new_callable=AsyncMock) as mock_run_behavior,
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_db.count_events_matching.return_value = 0  # not yet filed
            await run_reflexion(
                AsyncMock(), _SEM, event_kind="continuation_failure", event_payload={}
            )

        mock_run_behavior.assert_not_called()
        mock_bus.emit.assert_called_once()
        assert mock_bus.emit.call_args.args[0] == "reflexion_empty_payload"

    async def test_empty_payload_files_only_once(self) -> None:
        from luke.behaviors import run_reflexion

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.bus") as mock_bus,
            patch("luke.behaviors._run_behavior", new_callable=AsyncMock) as mock_run_behavior,
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_db.count_events_matching.return_value = 1  # already filed
            await run_reflexion(
                AsyncMock(), _SEM, event_kind="continuation_failure", event_payload={}
            )

        mock_run_behavior.assert_not_called()
        mock_bus.emit.assert_not_called()  # de-duped

    # --- Gate 2: saturation circuit-breaker --------------------------------
    async def test_saturation_skips_analysis_and_escalates_once(self) -> None:
        from luke.behaviors import run_reflexion

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.bus") as mock_bus,
            patch("luke.behaviors._run_behavior", new_callable=AsyncMock) as mock_run_behavior,
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"

            # 3 prior fires of this signature, 0 prior saturation markers
            def _counts(event_type: str, like: str | None = None, **_: Any) -> int:
                if event_type == "reflexion_fired":
                    return 3
                return 0

            mock_db.count_events_matching.side_effect = _counts
            await run_reflexion(
                AsyncMock(),
                _SEM,
                event_kind="deep_work_skipped",
                event_payload={"reason": "all_goals_filtered"},
            )

        mock_run_behavior.assert_not_called()
        emitted = [c.args[0] for c in mock_bus.emit.call_args_list]
        assert "reflexion_saturated" in emitted
        assert "reflexion_fired" not in emitted  # did not record a 4th fire

    async def test_below_saturation_records_fire_and_proceeds(self) -> None:
        from luke.behaviors import run_reflexion

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.bus") as mock_bus,
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.read_memory_body", return_value=""),
            patch("luke.behaviors._run_behavior", new_callable=AsyncMock) as mock_run_behavior,
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.consolidation_model = "sonnet"
            mock_db.get_recent_quality_scores.return_value = []
            mock_db.count_events_matching.return_value = 2  # below threshold
            mock_memory.recall.return_value = []
            await run_reflexion(
                AsyncMock(),
                _SEM,
                event_kind="deep_work_skipped",
                event_payload={"reason": "all_goals_filtered"},
            )

        mock_run_behavior.assert_called_once()
        emitted = [c.args[0] for c in mock_bus.emit.call_args_list]
        assert "reflexion_fired" in emitted  # recorded the fire for the counter


# ---------------------------------------------------------------------------
# run_skill_extraction
# ---------------------------------------------------------------------------


class TestRunSkillExtraction:
    async def test_no_chat_id(self) -> None:
        from luke.behaviors import run_skill_extraction

        with patch("luke.behaviors.settings") as mock_settings:
            mock_settings.chat_id = ""
            await run_skill_extraction(AsyncMock(), _SEM)

    async def test_skips_when_too_few_episode_bodies(self) -> None:
        from luke.behaviors import run_skill_extraction

        with (
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.read_memory_body", side_effect=["episode one", ""]),
            patch("luke.behaviors._run_behavior", new_callable=AsyncMock) as mock_run_behavior,
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_memory.recall.return_value = [
                {"id": "ep1", "type": "episode", "title": "Ep1", "score": 1.0},
                {"id": "ep2", "type": "episode", "title": "Ep2", "score": 1.0},
            ]
            await run_skill_extraction(AsyncMock(), _SEM)

        mock_run_behavior.assert_not_called()

    async def test_runs_with_recent_episodes(self) -> None:
        from luke.behaviors import run_skill_extraction

        with (
            patch("luke.behaviors.memory") as mock_memory,
            patch(
                "luke.behaviors.read_memory_body",
                side_effect=["episode one", "episode two", "existing procedure"],
            ),
            patch("luke.behaviors._run_behavior", new_callable=AsyncMock) as mock_run_behavior,
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.consolidation_model = "sonnet"
            mock_memory.recall.side_effect = [
                [
                    {"id": "ep1", "type": "episode", "title": "Ep1", "score": 1.0},
                    {"id": "ep2", "type": "episode", "title": "Ep2", "score": 1.0},
                ],
                [
                    {"id": "proc1", "type": "procedure", "title": "Proc1", "score": 1.0},
                ],
            ]
            await run_skill_extraction(AsyncMock(), _SEM)

        mock_run_behavior.assert_called_once()
        assert mock_run_behavior.call_args.args[0] == "skill_extraction"


# ---------------------------------------------------------------------------
# run_deep_work
# ---------------------------------------------------------------------------


class TestRunDeepWork:
    async def test_no_chat_id(self) -> None:
        from luke.behaviors import run_deep_work

        with patch("luke.behaviors.settings") as mock_settings:
            mock_settings.chat_id = ""
            await run_deep_work(AsyncMock(), _SEM)

    async def test_no_goals(self) -> None:
        from luke.behaviors import run_deep_work

        with (
            patch("luke.behaviors.db"),
            patch("luke.behaviors.bus"),
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.attention") as mock_attention,
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_memory.recall.return_value = []
            mock_attention.list_attention.return_value = []
            await run_deep_work(AsyncMock(), _SEM)

    async def test_with_goals(self, tmp_settings: Any) -> None:
        from luke.behaviors import run_deep_work

        (tmp_settings.memory_dir / "goals").mkdir(parents=True, exist_ok=True)
        (tmp_settings.memory_dir / "goals" / "g1.md").write_text(
            "---\nid: g1\ntype: goal\n---\n\n# Goal 1\n\nLearn Rust"
        )

        goals = [{"id": "g1", "type": "goal", "title": "Goal 1", "score": 1.0}]

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.bus"),
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("luke.behaviors.send_long_message", new_callable=AsyncMock),
        ):
            mock_db.get_quality_blocked_goals.return_value = []
            mock_db.get_recent_quality_scores.return_value = []
            mock_memory.recall.return_value = goals
            mock_agent.return_value = MagicMock(texts=[])
            await run_deep_work(AsyncMock(), _SEM)

        mock_agent.assert_called_once()
        call_kwargs = mock_agent.call_args.kwargs
        assert call_kwargs["max_turns"] == tmp_settings.deep_work_max_turns
        assert call_kwargs["max_sends"] == 1
        # Deep work is non-urgent — must not draw from the attention reserve
        assert not call_kwargs.get("urgent", False)

    async def test_agent_exception_handled(self) -> None:
        from luke.behaviors import run_deep_work

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.bus"),
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.run_agent", side_effect=RuntimeError("err")),
            patch("luke.behaviors.read_memory_body", return_value="content"),
            patch("luke.behaviors.settings") as mock_settings,
            patch("luke.behaviors.send_long_message", new_callable=AsyncMock),
        ):
            mock_settings.chat_id = "12345"
            mock_settings.agent_timeout = 10
            mock_settings.deep_work_model = "opus"
            mock_settings.deep_work_max_turns = 300
            mock_settings.workspace_dir = Path("/tmp/test_workspace")
            mock_db.get_quality_blocked_goals.return_value = []
            mock_db.get_recent_quality_scores.return_value = []
            mock_memory.recall.return_value = [
                {"id": "g1", "type": "goal", "title": "G1", "score": 1.0}
            ]
            await run_deep_work(AsyncMock(), _SEM)

    async def test_empty_goal_bodies_skips(self) -> None:
        from luke.behaviors import run_deep_work

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.bus"),
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.attention") as mock_attention,
            patch("luke.behaviors.read_memory_body", return_value=""),
            patch("luke.behaviors.settings") as mock_settings,
            patch("luke.behaviors.run_agent", new_callable=AsyncMock) as mock_agent,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.agent_timeout = 10
            mock_settings.workspace_dir = Path("/tmp/test_workspace")
            mock_db.get_quality_blocked_goals.return_value = []
            mock_db.get_recent_quality_scores.return_value = []
            mock_memory.recall.return_value = [
                {"id": "g1", "type": "goal", "title": "G1", "score": 1.0}
            ]
            mock_attention.list_attention.return_value = []
            await run_deep_work(AsyncMock(), _SEM)

        mock_agent.assert_not_called()


# ---------------------------------------------------------------------------
# Deep work lifecycle notifications (start / outcome / completion)
# ---------------------------------------------------------------------------


class TestDeepWorkLifecycleNotifications:
    def _goal_fixture(self, tmp_settings: Any) -> list[dict[str, Any]]:
        (tmp_settings.memory_dir / "goals").mkdir(parents=True, exist_ok=True)
        (tmp_settings.memory_dir / "goals" / "g1.md").write_text(
            "---\nid: g1\ntype: goal\n---\n\n# Goal 1\n\nShip the thing"
        )
        return [{"id": "g1", "type": "goal", "title": "Goal 1", "score": 1.0}]

    async def test_start_and_outcome_sent(self, tmp_settings: Any) -> None:
        """Filipe is informed of session start and outcome deterministically —
        never dependent on the agent choosing to speak."""
        from luke.behaviors import run_deep_work

        goals = self._goal_fixture(tmp_settings)
        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.bus"),
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.run_agent", new_callable=AsyncMock) as mock_agent,
            patch("luke.behaviors.send_long_message", new_callable=AsyncMock) as mock_send,
        ):
            mock_db.get_quality_blocked_goals.return_value = []
            mock_db.get_recent_quality_scores.return_value = []
            mock_memory.recall.return_value = goals
            mock_agent.return_value = MagicMock(texts=[])
            await run_deep_work(AsyncMock(), _SEM)

        texts = [c.args[2] for c in mock_send.call_args_list]
        assert any("Deep work session starting" in t and "g1" in t for t in texts)
        assert any("Deep work session done" in t for t in texts)

    async def test_failed_session_notifies(self, tmp_settings: Any) -> None:
        """A dead session is announced, not silently swallowed."""
        from luke.behaviors import run_deep_work

        goals = self._goal_fixture(tmp_settings)
        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.bus"),
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.run_agent", side_effect=RuntimeError("boom")),
            patch("luke.behaviors.send_long_message", new_callable=AsyncMock) as mock_send,
        ):
            mock_db.get_quality_blocked_goals.return_value = []
            mock_db.get_recent_quality_scores.return_value = []
            mock_memory.recall.return_value = goals
            await run_deep_work(AsyncMock(), _SEM)

        texts = [c.args[2] for c in mock_send.call_args_list]
        assert any("session failed" in t for t in texts)

    async def test_notify_failure_never_raises(self, tmp_settings: Any) -> None:
        """A notification must not kill the session it narrates."""
        from luke.behaviors import _notify

        with patch(
            "luke.behaviors.send_long_message",
            new_callable=AsyncMock,
            side_effect=RuntimeError("telegram down"),
        ):
            await _notify(AsyncMock(), "hello")  # must not raise


# ---------------------------------------------------------------------------
# Attention-fallback deep work runs on the cheap tier
# ---------------------------------------------------------------------------


class TestAttentionFallbackTier:
    async def test_fallback_uses_cheap_model(self, monkeypatch: Any) -> None:
        """Goalless sessions free-ran opus producing nothing (23 sessions /
        4 responses, Jul 27-Aug 1). The fallback path must run on the
        consolidation tier."""
        from luke.behaviors import _run_attention_deep_work
        from luke.config import settings

        monkeypatch.setattr(settings, "chat_id", "12345")

        pins = [{"id": 1, "origin": "filipe", "content": "a pinned thing"}]
        signals = {
            "replies": 0,
            "positive_reactions": 0,
            "negative_reactions": 0,
            "corrections": 0,
        }
        with (
            patch("luke.behaviors.attention") as mock_attention,
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.bus"),
            patch("luke.behaviors._run_behavior", new=AsyncMock()) as mock_run,
        ):
            mock_attention.list_attention.return_value = pins
            mock_db.get_behavior_last_run.return_value = None
            mock_db.get_engagement_signals.return_value = signals

            ran = await _run_attention_deep_work(AsyncMock(), _SEM, reason="no_goals")

        assert ran is True
        kwargs = mock_run.call_args.kwargs
        assert kwargs["model"] == settings.consolidation_model


# ---------------------------------------------------------------------------
# Plan momentum enforcement — stalled plans get nudged, deep stalls page Filipe
# ---------------------------------------------------------------------------


class TestEnforcePlanMomentum:
    def _write_plan(
        self,
        tmp_settings: Any,
        name: str,
        *,
        status: str = "in_progress",
        updated_hours_ago: float = 0.0,
    ) -> None:
        from datetime import UTC, datetime, timedelta

        plans = tmp_settings.workspace_dir / "plans"
        plans.mkdir(parents=True, exist_ok=True)
        updated = (datetime.now(UTC) - timedelta(hours=updated_hours_ago)).isoformat()
        (plans / f"{name}.md").write_text(
            f"# {name}\n\n**Status:** {status}\n**Last updated:** {updated}\n"
            "**Steps completed:** 1/3\n\n## Steps\n- [x] done thing\n- [ ] ship the next artifact\n"
        )

    async def test_fresh_plan_untouched(self, test_db: Any, tmp_settings: Any) -> None:
        from luke.behaviors import enforce_plan_momentum

        self._write_plan(tmp_settings, "goal-fresh", updated_hours_ago=2)
        with patch("luke.behaviors.send_long_message", new_callable=AsyncMock) as mock_send:
            assert await enforce_plan_momentum(AsyncMock()) == 0
        mock_send.assert_not_called()

    async def test_stalled_plan_emits_goal_updated(self, test_db: Any, tmp_settings: Any) -> None:
        """48h+ stall → goal_updated event, which drives the deep-work intent."""
        from luke import db as luke_db
        from luke.behaviors import enforce_plan_momentum

        self._write_plan(tmp_settings, "goal-stalled", updated_hours_ago=72)
        before = luke_db.count_unconsumed_events("goal_updated")
        with patch("luke.behaviors.send_long_message", new_callable=AsyncMock) as mock_send:
            assert await enforce_plan_momentum(AsyncMock()) == 1
        assert luke_db.count_unconsumed_events("goal_updated") == before + 1
        mock_send.assert_not_called()  # 72h < alert threshold: nudge silently

    async def test_deep_stall_alerts_filipe_with_next_step(
        self, test_db: Any, tmp_settings: Any
    ) -> None:
        from luke.behaviors import enforce_plan_momentum

        self._write_plan(tmp_settings, "goal-abandoned-feeling", updated_hours_ago=120)
        with patch("luke.behaviors.send_long_message", new_callable=AsyncMock) as mock_send:
            assert await enforce_plan_momentum(AsyncMock()) == 1
        mock_send.assert_called_once()
        text = mock_send.call_args.args[2]
        assert "No progress" in text
        assert "ship the next artifact" in text

    async def test_renudge_suppressed_within_a_day(self, test_db: Any, tmp_settings: Any) -> None:
        from luke.behaviors import enforce_plan_momentum

        self._write_plan(tmp_settings, "goal-stalled", updated_hours_ago=72)
        with patch("luke.behaviors.send_long_message", new_callable=AsyncMock):
            assert await enforce_plan_momentum(AsyncMock()) == 1
            assert await enforce_plan_momentum(AsyncMock()) == 0  # daily rate limit

    async def test_terminal_plans_ignored(self, test_db: Any, tmp_settings: Any) -> None:
        from luke.behaviors import enforce_plan_momentum

        self._write_plan(tmp_settings, "goal-done", status="completed", updated_hours_ago=500)
        self._write_plan(tmp_settings, "goal-paused", status="paused", updated_hours_ago=500)
        with patch("luke.behaviors.send_long_message", new_callable=AsyncMock):
            assert await enforce_plan_momentum(AsyncMock()) == 0
