"""Tests for luke.config — Settings validation and cached properties."""

from __future__ import annotations

import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "0000000000:AAHfakeTestTokenForUnitTesting1234")

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from luke.config import Settings


def _make(**overrides: object) -> Settings:
    """Create a Settings instance with defaults suitable for testing."""
    defaults: dict[str, object] = {
        "telegram_bot_token": SecretStr("test-token"),
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


class TestValidation:
    def test_valid_defaults(self) -> None:
        s = _make()
        assert s.assistant_name == "Luke"
        assert s.max_concurrent == 8

    def test_missing_token_raises(self) -> None:
        with pytest.raises(ValidationError, match="missing"):
            _make(telegram_bot_token=SecretStr(""))

    def test_decay_rate_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="decay rate"):
            _make(decay_rate_entity=0.0)

    def test_decay_rate_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="decay rate"):
            _make(decay_rate_entity=1.0)

    def test_decay_rate_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="decay rate"):
            _make(decay_rate_entity=-0.5)

    def test_decay_rate_valid(self) -> None:
        s = _make(decay_rate_entity=0.99)
        assert s.decay_rate_entity == 0.99

    def test_score_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"sum to 1\.0"):
            _make(score_weight_relevance=0.5, score_weight_importance=0.5)

    def test_score_weights_valid(self) -> None:
        s = _make(
            score_weight_relevance=0.25,
            score_weight_importance=0.25,
            score_weight_recency=0.25,
            score_weight_access=0.25,
        )
        assert s.score_weight_relevance == 0.25


class TestCachedProperties:
    def test_decay_rates_property(self) -> None:
        s = _make()
        rates = s.decay_rates
        assert set(rates.keys()) == {"entity", "episode", "procedure", "insight", "goal"}
        assert rates["entity"] == s.decay_rate_entity

    def test_luke_dir_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Bypass .env file and process env to test the field default
        monkeypatch.delenv("LUKE_DIR", raising=False)
        s = _make(_env_file=None)
        assert s.luke_dir == Path.home() / ".luke"

    def test_luke_dir_override(self) -> None:
        s = _make(luke_dir=Path("/tmp/custom-luke"))
        assert s.luke_dir == Path("/tmp/custom-luke")

    def test_workspace_dir(self) -> None:
        s = _make(luke_dir=Path("/tmp/test"))
        assert s.workspace_dir == Path("/tmp/test/workspace")

    def test_memory_dir(self) -> None:
        s = _make(luke_dir=Path("/tmp/test"))
        assert s.memory_dir == Path("/tmp/test/memory")

    def test_store_dir(self) -> None:
        s = _make(luke_dir=Path("/tmp/test"))
        assert s.store_dir == Path("/tmp/test")

    def test_recalibrated_decay_rates(self) -> None:
        """Verify decay rates are gentler than original values."""
        s = _make()
        assert s.decay_rate_entity == 0.9998
        assert s.decay_rate_episode == 0.999
        assert s.decay_rate_procedure == 0.9999
        assert s.decay_rate_insight == 0.9995
        assert s.decay_rate_goal == 0.9997

    def test_consolidation_min_cluster(self) -> None:
        s = _make()
        assert s.consolidation_min_cluster == 2
