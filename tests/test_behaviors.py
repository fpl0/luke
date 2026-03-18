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
            mock_memory.recall_by_time_window.return_value = memories
            mock_db.get_recent_messages.return_value = []
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
            mock_memory.recall.side_effect = [goals, insights]
            mock_db.get_message_summaries.return_value = [
                {"date": "2024-01-01", "messages": ["User: hi"]}
            ]
            mock_agent.return_value = MagicMock(texts=[])
            await run_proactive_scan(AsyncMock(), _SEM)

        mock_agent.assert_called_once()

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
# run_deep_work
# ---------------------------------------------------------------------------


class TestRunDeepWork:
    async def test_no_chat_id(self) -> None:
        from luke.behaviors import run_deep_work

        with patch("luke.behaviors.settings") as mock_settings:
            mock_settings.chat_id = ""
            await run_deep_work(AsyncMock(), _SEM)

    async def test_budget_exhausted(self) -> None:
        from luke.behaviors import run_deep_work

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.settings") as mock_settings,
            patch("luke.behaviors.run_agent", new_callable=AsyncMock) as mock_agent,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.daily_deep_work_budget_usd = 10.0
            mock_db.get_daily_deep_work_cost.return_value = 15.0
            await run_deep_work(AsyncMock(), _SEM)

        mock_agent.assert_not_called()

    async def test_no_goals(self) -> None:
        from luke.behaviors import run_deep_work

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.daily_deep_work_budget_usd = 60.0
            mock_db.get_daily_deep_work_cost.return_value = 0.0
            mock_memory.recall.return_value = []
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
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.run_agent", new_callable=AsyncMock) as mock_agent,
        ):
            mock_db.get_daily_deep_work_cost.return_value = 0.0
            mock_memory.recall.return_value = goals
            mock_agent.return_value = MagicMock(texts=[])
            await run_deep_work(AsyncMock(), _SEM)

        mock_agent.assert_called_once()
        # Verify budget overrides (model routing disabled — no model param)
        call_kwargs = mock_agent.call_args.kwargs
        assert call_kwargs["max_turns"] == tmp_settings.deep_work_max_turns
        assert call_kwargs["max_sends"] == 1

    async def test_agent_exception_handled(self) -> None:
        from luke.behaviors import run_deep_work

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.run_agent", side_effect=RuntimeError("err")),
            patch("luke.behaviors.read_memory_body", return_value="content"),
            patch("luke.behaviors.settings") as mock_settings,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.agent_timeout = 10
            mock_settings.daily_deep_work_budget_usd = 60.0
            mock_settings.deep_work_model = "opus"
            mock_settings.deep_work_max_turns = 300
            mock_settings.deep_work_max_budget_usd = 3.0
            mock_settings.workspace_dir = Path("/tmp/test_workspace")
            mock_db.get_daily_deep_work_cost.return_value = 0.0
            mock_memory.recall.return_value = [
                {"id": "g1", "type": "goal", "title": "G1", "score": 1.0}
            ]
            await run_deep_work(AsyncMock(), _SEM)

    async def test_empty_goal_bodies_skips(self) -> None:
        from luke.behaviors import run_deep_work

        with (
            patch("luke.behaviors.db") as mock_db,
            patch("luke.behaviors.memory") as mock_memory,
            patch("luke.behaviors.read_memory_body", return_value=""),
            patch("luke.behaviors.settings") as mock_settings,
            patch("luke.behaviors.run_agent", new_callable=AsyncMock) as mock_agent,
        ):
            mock_settings.chat_id = "12345"
            mock_settings.agent_timeout = 10
            mock_settings.daily_deep_work_budget_usd = 60.0
            mock_settings.workspace_dir = Path("/tmp/test_workspace")
            mock_db.get_daily_deep_work_cost.return_value = 0.0
            mock_memory.recall.return_value = [
                {"id": "g1", "type": "goal", "title": "G1", "score": 1.0}
            ]
            await run_deep_work(AsyncMock(), _SEM)

        mock_agent.assert_not_called()
