"""Phase 4.1 — Letta agent-loop backend (context assembly half).

When ``settings.agent_backend == "letta"`` the always-in-context persona / human /
world-model for a turn is sourced from the ``luke-agent-claude`` core memory blocks
(packed in Phase 2.2a) instead of re-injecting the ``build_working_context`` blob into
the system prompt every turn. Sourcing the world-model from self-editing core blocks —
rather than rebuilding it from sqlite on each turn — is the whole point of the migration.

This module owns the **bridge-independent** half of 4.1: reading the core blocks over
Letta's REST API and assembling them into the exact ``system_prompt.append`` string shape
the SDK turn already consumes (agent.py). It makes **no Claude turn** — it only reads block
state — so it is fully verifiable off the OAuth pool. The live end-to-end turn (running the
assembled context *through* the Letta agent → bridge → Telegram) is gated on a low-contention
window, the same blocker as Phase 1.4, and is wired but not exercised here.

Fail-safe by design: any REST error returns ``None`` so the caller falls back to the sqlite
``build_working_context`` blob. Flipping ``agent_backend`` back to ``"sdk"`` fully reverts.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import structlog

from .config import settings

log = structlog.get_logger()

# Canonical order for assembling blocks into the system-prompt append. persona + human are
# the classic MemGPT identity blocks and lead; the read-only operating-rules guardrail is last
# so it reads as the final, non-negotiable word (mirrors how the injected <constitutional>
# block anchors the SDK persona). Any block present on the agent but absent here is appended
# after, in server order, so a newly-added block is never silently dropped.
_BLOCK_ORDER = (
    "persona",
    "human",
    "key-people",
    "key-projects",
    "preferences",
    "goals",
    "operating-rules",
)


def _get_core_blocks(agent_id: str, timeout: float = 10.0) -> list[dict[str, Any]] | None:
    """Fetch the agent's core-memory blocks over REST. None on any failure (fail-safe).

    Uses the ``/core-memory/blocks`` endpoint, not the agent-retrieve envelope: the client
    flattens ``agent.memory.blocks`` to ``[]`` in some responses, but this endpoint returns
    the real block set (verified in Phase 2.2a / 3.1).
    """
    url = f"{settings.letta_base_url}/v1/agents/{agent_id}/core-memory/blocks"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (localhost)
            blocks = json.load(resp)
        if not isinstance(blocks, list) or not blocks:
            log.warning("letta_core_blocks_empty", agent_id=agent_id)
            return None
        return blocks
    except Exception as e:  # fail safe — caller falls back to the sqlite blob
        log.warning("letta_core_blocks_fetch_failed", agent_id=agent_id, error=str(e))
        return None


def _render_block(block: dict[str, Any]) -> str:
    """Render one core block in Letta's own labelled-block format."""
    label = block.get("label", "block")
    value = (block.get("value") or "").strip()
    ro = " read_only=\"true\"" if block.get("read_only") else ""
    return f'<block label="{label}"{ro}>\n{value}\n</block>'


def build_letta_context(agent_id: str | None = None) -> str | None:
    """Assemble the turn's always-in-context world-model from Letta core blocks.

    Returns the ``system_prompt.append`` string sourced from the ``luke-agent-claude`` core
    blocks — the Phase 4.1 replacement for ``context.build_working_context``. Returns ``None``
    on any error so the caller keeps the sqlite blob (fail-safe, reversible).
    """
    agent_id = agent_id or settings.letta_agent_id
    blocks = _get_core_blocks(agent_id)
    if not blocks:
        return None

    by_label = {b.get("label"): b for b in blocks}
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label in _BLOCK_ORDER:
        if label in by_label:
            ordered.append(by_label[label])
            seen.add(label)
    # Any block on the agent not named in _BLOCK_ORDER is appended (never silently dropped).
    for b in blocks:
        if b.get("label") not in seen:
            ordered.append(b)

    rendered = "\n\n".join(_render_block(b) for b in ordered)
    total_chars = sum(len((b.get("value") or "")) for b in ordered)
    log.info(
        "letta_context_assembled",
        agent_id=agent_id,
        block_count=len(ordered),
        chars=total_chars,
    )
    return (
        "<letta-core-memory>\n"
        "# Always-in-context world model — sourced live from your Letta core memory blocks.\n"
        "# These self-edit across turns; they are the ground truth, not a per-turn re-injection.\n\n"
        f"{rendered}\n"
        "</letta-core-memory>"
    )
