"""Shared test fixtures for Luke test suite."""

from __future__ import annotations

# Must set env BEFORE any luke imports — Settings() runs at import time
import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "0000000000:AAHfakeTestTokenForUnitTesting1234")

from pathlib import Path
from typing import Any, cast

import pytest

from luke import db
from luke.config import settings


def _clear_cached_properties() -> None:
    """Clear cached_property values so they recompute from a new luke_dir."""
    obj_dict = cast(dict[str, Any], settings.__dict__)
    for prop in ("workspace_dir", "memory_dir", "store_dir", "decay_rates"):
        obj_dict.pop(prop, None)


@pytest.fixture()
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Redirect settings to use tmp_path for all data dirs."""
    monkeypatch.setattr(settings, "luke_dir", tmp_path / "luke")
    _clear_cached_properties()
    yield settings
    _clear_cached_properties()


@pytest.fixture()
def test_db(tmp_settings: Any) -> Any:
    """Provide a fresh SQLite database per test."""
    # Clear any cached thread-local connection
    db._local.__dict__.pop("conn", None)
    # Create dirs and schema
    tmp_settings.store_dir.mkdir(parents=True, exist_ok=True)
    # Create memory subdirs
    tmp_settings.memory_dir.mkdir(parents=True, exist_ok=True)
    for subdir in db.MEMORY_DIRS.values():
        (tmp_settings.memory_dir / subdir).mkdir(exist_ok=True)
    # Set chat_id for tests
    tmp_settings.chat_id = "12345"
    db.init()
    yield db
    conn = getattr(db._local, "conn", None)
    if conn is not None:
        conn.close()
    db._local.__dict__.pop("conn", None)
    db._local.__dict__.pop("batch_depth", None)
