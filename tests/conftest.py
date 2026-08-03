"""Shared test fixtures for Luke test suite."""

from __future__ import annotations

# Must set env BEFORE any luke imports — Settings() runs at import time
import os
import tempfile

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "0000000000:AAHfakeTestTokenForUnitTesting1234")

# Point LUKE_DIR at a throwaway directory for the whole session. Without this
# it falls through to .env — the developer's LIVE data dir — and any test that
# doesn't take `tmp_settings` writes straight into it. That is not theoretical:
# `_save_conv_state` is dispatched as a fire-and-forget to_thread task
# (app.py), so it can outlive the fixture teardown that restores
# `settings.luke_dir`, and land test fixture content in the real
# memory/episodes/conversation-state-latest.md — corrupting the continuity
# anchor the agent reads on every turn. Observed 2026-08-03.
#
# Assignment, not setdefault: an exported LUKE_DIR pointing at live data is
# exactly the case this must override.
os.environ["LUKE_DIR"] = tempfile.mkdtemp(prefix="luke-test-session-")

from pathlib import Path
from typing import Any

import pytest

from luke import db
from luke.config import settings
from luke.memory import MEMORY_DIRS


def _clear_cached_properties() -> None:
    """Clear cached_property values so they recompute from a new luke_dir."""
    obj_dict = vars(settings)
    for prop in ("workspace_dir", "memory_dir", "store_dir", "decay_rates"):
        obj_dict.pop(prop, None)


@pytest.fixture(autouse=True)
def fake_embed_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the HTTP embed seam with a deterministic in-process fake.

    memory.py embeds via the local bge embed server (:17595); tests must never
    depend on that daemon. The fake hashes tokens into a fixed-dim bag-of-words
    vector: identical texts embed identically, overlapping texts stay
    cosine-similar — enough for the duplicate/similarity code paths without
    model weights or network.
    """
    import hashlib as _hashlib
    import math as _math

    def _fake(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * 768
            for token in text.lower().split():
                idx = int(_hashlib.md5(token.encode()).hexdigest(), 16) % 768
                vec[idx] += 1.0
            norm = _math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out

    monkeypatch.setattr("luke.memory._embed_via_server", _fake)


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
    for subdir in MEMORY_DIRS.values():
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
