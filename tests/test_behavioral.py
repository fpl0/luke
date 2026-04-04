"""Behavioral tests: personality consistency, emotional response, anti-regression, memory quality.

These tests verify that Luke maintains its character, routes correctly based on user emotional
state, does not drift from persona, and retrieves relevant memories for sample queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Persona fixture — loaded once for the whole module
# ---------------------------------------------------------------------------

_PERSONA_PATH = Path(__file__).resolve().parent.parent / "src" / "luke" / "templates" / "LUKE.md"


@pytest.fixture(scope="module")
def persona() -> str:
    """Load the canonical Luke persona template."""
    return _PERSONA_PATH.read_text()


# ---------------------------------------------------------------------------
# 1. Personality Consistency Tests
# ---------------------------------------------------------------------------


class TestPersonalityConsistency:
    """Verify Luke's persona is well-formed and contains required character markers."""

    def test_persona_file_exists(self) -> None:
        assert _PERSONA_PATH.exists(), "LUKE.md template must exist at src/luke/templates/LUKE.md"

    def test_persona_identifies_as_luke(self, persona: str) -> None:
        """Luke must define its own name explicitly."""
        assert "You are Luke" in persona or "Luke" in persona[:500], (
            "Persona must establish the Luke identity near the top"
        )

    def test_persona_defines_personal_voice(self, persona: str) -> None:
        """Persona must describe the voice/tone style."""
        voice_markers = ["voice", "warm", "tone", "unhurried", "wry"]
        assert any(m in persona.lower() for m in voice_markers), (
            "Persona must describe Luke's voice/tone"
        )

    def test_persona_has_memory_integration_rule(self, persona: str) -> None:
        """Persona must instruct using memory tools, not just claiming to remember."""
        assert "remember" in persona.lower(), (
            "Persona must reference the remember tool or memory behavior"
        )

    def test_persona_has_no_markdown_formatting_rule(self, persona: str) -> None:
        """Luke must never use markdown — Telegram renders HTML."""
        assert "markdown" in persona.lower(), (
            "Persona must include the no-markdown rule for Telegram"
        )

    def test_persona_references_telegram_format(self, persona: str) -> None:
        """Persona must describe Telegram message format so Luke handles msg IDs."""
        assert "msg:" in persona or "msg_id" in persona or "Telegram" in persona, (
            "Persona must reference Telegram message format"
        )

    def test_assistant_name_is_luke(self) -> None:
        """Default assistant_name in settings must be 'Luke'."""
        from luke.config import settings

        assert settings.assistant_name == "Luke"

    def test_persona_contains_opinion_stance(self, persona: str) -> None:
        """Luke must have opinions — not just 'here are the pros and cons'."""
        opinion_markers = ["opinion", "disagree", "wrong", "warmly", "conviction", "values"]
        assert any(m in persona.lower() for m in opinion_markers), (
            "Persona must express that Luke has genuine opinions"
        )


# ---------------------------------------------------------------------------
# 2. Emotional Response Tests (model routing)
# ---------------------------------------------------------------------------


class TestEmotionalResponse:
    """_classify_effort routes different user emotional states to the right model tier."""

    def test_frustrated_short_message_routes_to_haiku(self) -> None:
        """Short frustrated acks ('ok', 'fine', 'whatever') use the lightweight model."""
        from luke.app import _classify_effort
        from luke.config import settings

        _, _, model = _classify_effort("ok")
        assert model == settings.model_low, (
            f"Short ack should route to {settings.model_low}, got {model}"
        )

    def test_excited_complex_query_routes_to_high_tier(self) -> None:
        """Excited users with complex questions get the high-tier model."""
        from luke.app import _classify_effort
        from luke.config import settings

        prompt = (
            "I'm so excited — can you help me research, analyze, and compare the best ways "
            "to implement this? I want to design the whole thing from scratch!"
        )
        _, _, model = _classify_effort(prompt)
        assert model == settings.model_high, (
            f"Complex excited query should route to {settings.model_high}, got {model}"
        )

    def test_confused_multi_question_routes_to_high_tier(self) -> None:
        """Confused users sending many questions get more reasoning power."""
        from luke.app import _classify_effort
        from luke.config import settings

        prompt = "I don't understand? Why is this happening? What should I do? Where do I start?"
        _, _, model = _classify_effort(prompt)
        assert model == settings.model_high, (
            f"Multi-question confusion should route to {settings.model_high}, got {model}"
        )

    def test_sad_simple_expression_routes_to_haiku(self) -> None:
        """Very short emotional messages use the lightweight model (brevity = haiku)."""
        from luke.app import _classify_effort
        from luke.config import settings

        _, _, model = _classify_effort("thanks")
        assert model == settings.model_low

    def test_technical_frustration_with_code_routes_to_opus(self) -> None:
        """Frustrated users with code errors get the most capable model."""
        from luke.app import _classify_effort
        from luke.config import settings

        prompt = "This is so annoying — I keep getting this traceback and I can't fix the bug"
        _, _, model = _classify_effort(prompt)
        assert model == settings.model_high, (
            f"Code/error message should route to {settings.model_high}, got {model}"
        )

    def test_casual_conversation_routes_to_medium(self) -> None:
        """Normal casual conversation (15+ words, no question, no code) routes to medium tier."""
        from luke.app import _classify_effort
        from luke.config import settings

        # Must be 15+ words to clear the haiku threshold, no "?" and no code keywords
        prompt = (
            "I had a really interesting day at work today and have been seriously "
            "thinking about whether to stay in my current role or move on to something new"
        )
        _, _, model = _classify_effort(prompt)
        assert model == settings.model_medium, (
            f"Casual conversation should route to {settings.model_medium}, got {model}"
        )

    def test_image_message_routes_to_high_tier(self) -> None:
        """Messages with images are treated as complex and get the high-tier model."""
        from luke.app import _classify_effort
        from luke.config import settings

        prompt = [{"type": "image", "source": "base64"}, {"type": "text", "text": "what is this"}]
        _, _, model = _classify_effort(prompt)
        assert model == settings.model_high, (
            f"Image message should route to {settings.model_high}, got {model}"
        )

    def test_trivial_greeting_skips_recall(self) -> None:
        """Pure greetings should not trigger a memory recall round-trip."""
        from luke.app import _needs_recall

        assert not _needs_recall("hey"), "greeting 'hey' should not trigger recall"
        assert not _needs_recall("hi"), "greeting 'hi' should not trigger recall"
        assert not _needs_recall("ok"), "'ok' should not trigger recall"
        assert not _needs_recall("thanks"), "'thanks' should not trigger recall"

    def test_substantive_message_triggers_recall(self) -> None:
        """Non-trivial messages should trigger memory recall."""
        from luke.app import _needs_recall

        assert _needs_recall("What were we working on last week with the Python project?")
        assert _needs_recall("I've been thinking about the trip to Portugal")
        assert _needs_recall("Can you remind me what I said about the meeting?")


# ---------------------------------------------------------------------------
# 3. Anti-Regression Tests (character drift prevention)
# ---------------------------------------------------------------------------


class TestAntiRegression:
    """These tests catch personality drift — forbidden phrases must never appear in persona."""

    def test_persona_does_not_say_i_am_an_ai(self, persona: str) -> None:
        """Persona must prohibit Luke from identifying as an AI, not encourage it."""
        # "as an AI" appears only in the prohibition rule — verify it's a negation
        prohibited_first_person = [
            "I'm an AI",
            "I am an AI",
            "I'm a language model",
            "I am a language model",
        ]
        for phrase in prohibited_first_person:
            assert phrase.lower() not in persona.lower(), (
                f"Persona must not contain first-person AI identification: '{phrase}'"
            )
        # Verify the prohibition is present (the phrase exists only in the rule against it)
        prohibited = (
            "never refer to yourself as an AI" in persona
            or "You never refer to yourself" in persona
        )
        assert prohibited, "Persona must explicitly prohibit AI self-identification"

    def test_persona_does_not_say_i_am_an_assistant(self, persona: str) -> None:
        """Luke must not use the word 'assistant' in first person."""
        # 'assistant' may appear in 'not an assistant' — check the positive framing
        assert "I am an assistant" not in persona and "I'm an assistant" not in persona, (
            "Persona must not describe Luke as an assistant"
        )

    def test_persona_bans_absolutely(self, persona: str) -> None:
        """'Absolutely!' is explicitly called out as hollow filler."""
        assert "Absolutely" in persona or "absolutely" in persona, (
            "Persona must explicitly ban 'Absolutely!' as hollow filler"
        )

    def test_persona_bans_great_question(self, persona: str) -> None:
        """'Great question!' is sycophantic and must be banned."""
        assert "Great question" in persona, "Persona must explicitly disallow 'Great question!'"

    def test_persona_bans_the_user_phrasing(self, persona: str) -> None:
        """Luke must say 'you', never 'the user'."""
        assert "the user" in persona, "Persona must include the rule to say 'you' not 'the user'"

    def test_persona_bans_would_you_like_me_to(self, persona: str) -> None:
        """Luke does not ask permission — just does the work."""
        assert "would you like me to" in persona.lower() or "don't ask" in persona.lower(), (
            "Persona must address not asking 'would you like me to...'"
        )

    def test_haiku_tools_do_not_include_destructive_ops(self) -> None:
        """Haiku (lightweight) must not have access to destructive tools."""
        from luke.agent import _ALLOWED_HAIKU

        destructive = {
            "mcp__luke__delete_message",
            "mcp__luke__forget",
            "mcp__luke__schedule_task",
            "mcp__luke__bulk_memory",
        }
        overlap = destructive & set(_ALLOWED_HAIKU)
        assert not overlap, f"Haiku tier must not include destructive tools: {overlap}"

    def test_opus_tools_include_all_mcp_tools(self) -> None:
        """Opus (max capability) must have access to the full MCP wildcard."""
        from luke.agent import _ALLOWED_OPUS

        assert "mcp__luke__*" in _ALLOWED_OPUS, "Opus must use wildcard to access all MCP tools"

    def test_sonnet_excludes_bulk_memory_and_schedule(self) -> None:
        """Sonnet tier excludes bulk operations — only Opus gets those."""
        from luke.agent import _ALLOWED_SONNET

        assert "mcp__luke__schedule_task" not in _ALLOWED_SONNET, (
            "Sonnet must not include schedule_task"
        )
        assert "mcp__luke__bulk_memory" not in _ALLOWED_SONNET, (
            "Sonnet must not include bulk_memory"
        )

    def test_model_tier_ordering_is_correct(self) -> None:
        """The model rank map must rank haiku < sonnet < opus."""
        from luke.app import _MODEL_RANK

        assert _MODEL_RANK["haiku"] < _MODEL_RANK["sonnet"] < _MODEL_RANK["opus"], (
            "Model rank must be haiku < sonnet < opus"
        )

    def test_trivial_words_includes_core_greetings(self) -> None:
        """Trivial word set must cover common greetings that should not trigger recall."""
        from luke.app import _TRIVIAL_WORDS

        required = {"hi", "hey", "hello", "ok", "okay", "thanks", "yes", "no"}
        missing = required - _TRIVIAL_WORDS
        assert not missing, f"Trivial words set is missing: {missing}"


# ---------------------------------------------------------------------------
# 4. Memory Retrieval Quality Tests
# ---------------------------------------------------------------------------


class TestMemoryRetrievalQuality:
    """Verify recall returns relevant results above irrelevant ones for sample queries."""

    def test_fts_returns_relevant_result(self, test_db: Any) -> None:
        """A query matching indexed content should return that memory."""
        from luke import memory

        memory.index_memory(
            "proj-alpha",
            "episode",
            "Project Alpha kickoff",
            "Started the Alpha project with the backend team to build a new payment system",
        )
        results = memory.recall(query="Alpha project payment backend")
        ids = [r["id"] for r in results]
        assert "proj-alpha" in ids, "FTS must return 'proj-alpha' for a direct content match"

    def test_relevant_beats_irrelevant(self, test_db: Any) -> None:
        """A memory closely matching the query must score higher than an unrelated one."""
        from luke import memory

        memory.index_memory(
            "flight-rome",
            "episode",
            "Booked flight to Rome",
            "Booked an ITA Airways flight from Lisbon to Rome Fiumicino for next Tuesday",
        )
        memory.index_memory(
            "recipe-pasta",
            "episode",
            "Pasta recipe",
            "Learned to make spaghetti carbonara with guanciale and pecorino",
        )
        results = memory.recall(query="Rome flight booking ITA Airways")
        assert results, "Should return at least one result"
        top_id = results[0]["id"]
        assert top_id == "flight-rome", (
            f"'flight-rome' should rank first for a flight query, got {top_id!r}"
        )

    def test_type_filter_restricts_results(self, test_db: Any) -> None:
        """Recall with mem_type filter must only return that type."""
        from luke import memory

        memory.index_memory("person-ana", "entity", "Ana", "Ana is a close friend from university")
        memory.index_memory("meeting-ana", "episode", "Met Ana", "Had coffee with Ana to catch up")
        results = memory.recall(query="Ana", mem_type="entity")
        types = {r["type"] for r in results}
        assert types == {"entity"}, (
            f"Type filter 'entity' must exclude episodes, got types: {types}"
        )

    def test_recall_empty_query_returns_all_active(self, test_db: Any) -> None:
        """An empty query with no filters returns active memories (type-scan path)."""
        from luke import memory

        memory.index_memory("fact-a", "insight", "Insight A", "Key insight about productivity")
        memory.index_memory("fact-b", "procedure", "Proc B", "How to set up the dev environment")
        results = memory.recall(mem_type="insight")
        ids = [r["id"] for r in results]
        assert "fact-a" in ids, "Empty query with type filter should return indexed insight"

    def test_malformed_fts_query_does_not_crash(self, test_db: Any) -> None:
        """FTS5 malformed queries must be caught gracefully — no exception raised."""
        from luke import memory

        # Parentheses imbalance or bare operators can crash FTS5
        bad_queries = ["AND OR", "((unclosed", "NOT NOT NOT", "* wildcard only"]
        for q in bad_queries:
            results = memory.recall(query=q)
            assert isinstance(results, list), f"recall({q!r}) must return a list, not raise"

    def test_goal_type_recall(self, test_db: Any) -> None:
        """Goals must be indexable and retrievable by type."""
        from luke import memory

        memory.index_memory(
            "goal-fitness",
            "goal",
            "Get fit by summer",
            "Run a 5k by July, go to the gym 4x per week, track calories",
        )
        results = memory.recall(query="fitness gym running", mem_type="goal")
        ids = [r["id"] for r in results]
        assert "goal-fitness" in ids, "Goal memory must be retrievable by relevant query"

    def test_procedure_recall_for_how_to_query(self, test_db: Any) -> None:
        """Procedure memories should surface for how-to queries."""
        from luke import memory

        memory.index_memory(
            "proc-deploy",
            "procedure",
            "Deploy to production",
            "Run uv sync, then git push, then launchctl kickstart, check logs after",
        )
        results = memory.recall(query="how to deploy production launchctl")
        ids = [r["id"] for r in results]
        assert "proc-deploy" in ids, "Procedure must be returned for a deployment how-to query"
