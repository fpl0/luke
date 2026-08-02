"""The operating-rules block is only as strong as the memories it is packed from.

``letta_pack_core_blocks._fetch`` skips a missing id "gracefully" — which is right for the
packer (one archived memory should not abort the nightly re-pack) and wrong for the guard:
archive or rename a memory in ``GUARDRAIL_IDS`` and a hard directive quietly stops reaching
the agent, with a successful "all blocks packed + verified" printed over the top of it. The
block just gets shorter, and nothing reads block length.

That matters most for the newest entry. ``feedback-verify-self-claims-with-tools`` is the
fix for both genuine 5.1R vetoes; if it evaporates the gate regresses and the run output
still looks green.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "letta_pack_core_blocks.py"


def _load_packer():
    """Load the packer for its constants and ``_fetch`` only.

    The script runs under .letta-venv and imports ``letta_client``, which the main test venv
    does not have. Stubbing it keeps this guard runnable in the normal suite — the pieces
    under test are a list of ids and one sqlite SELECT, neither of which touches the client.
    """
    if "letta_client" not in sys.modules:
        stub = types.ModuleType("letta_client")
        stub.Letta = object
        sys.modules["letta_client"] = stub
    spec = importlib.util.spec_from_file_location("letta_pack_core_blocks", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


packer = _load_packer()


@pytest.fixture(scope="module")
def db():
    if not Path(packer.DB_PATH).exists():
        pytest.skip(f"{packer.DB_PATH} not present")
    conn = sqlite3.connect(f"file:{packer.DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.mark.parametrize("mem_id", packer.GUARDRAIL_IDS)
def test_every_guardrail_id_resolves_to_a_live_memory(db, mem_id):
    """A guardrail id that no longer resolves is a directive that silently stopped applying."""
    assert packer._fetch(db, mem_id) is not None, (
        f"{mem_id} is in GUARDRAIL_IDS but has no active memory — the packer would skip it "
        f"and still report success, dropping a hard rule from operating-rules"
    )


@pytest.mark.parametrize("mem_id", packer.ANCHOR_IDS)
def test_every_anchor_id_resolves_to_a_live_memory(db, mem_id):
    """Same failure mode for the world-model anchors the importance cut must never evict."""
    assert packer._fetch(db, mem_id) is not None, (
        f"{mem_id} is in ANCHOR_IDS but has no active memory — the core blocks would lose a "
        f"load-bearing entity without the pack reporting anything"
    )


def test_self_verification_guardrail_is_present_and_names_its_tools(db):
    """The rule only works if it tells the agent WHICH tools to reach for.

    Stated as a bare "check before you answer" it is the same advisory shape as the as_of
    anchor's "say so if you don't know", which was already in the turn prompt and provably
    did not fire. Naming the tools is what turned it into behaviour.
    """
    mem_id = "feedback-verify-self-claims-with-tools"
    assert mem_id in packer.GUARDRAIL_IDS, "the self-verification rule left GUARDRAIL_IDS"
    row = packer._fetch(db, mem_id)
    assert row is not None, f"{mem_id} is not an active memory"
    body = (row["content"] or "").lower()
    for tool in ("luke_search_code", "luke_recall"):
        assert tool in body, f"{mem_id} no longer names {tool} as the thing to reach for"
