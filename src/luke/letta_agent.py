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
import time
import urllib.request
from typing import Any

import structlog

from .config import settings

log = structlog.get_logger()
_clock = time.perf_counter

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


# ---------------------------------------------------------------------------
# Phase 4.3 — per-turn archive injection
# ---------------------------------------------------------------------------
#
# The 7 core blocks (build_letta_context) are the ALWAYS-in-context world model: the
# people/projects/preferences/rules that apply to every turn. They deliberately do NOT
# hold the long tail — the ~1500 archive passages of one-off entities, procedures,
# episodes, and dated facts (a visa appointment, a specific procedure, last month's
# episode). A Letta turn that answers only from core blocks therefore can't ground a
# question whose answer lives only in the archive; it either misses or confabulates.
#
# This is the missing retrieval half. Before a turn, we run Luke's OWN recall() — FTS5 +
# semantic + graph + composite scoring, the exact production ranker — over the user's
# message, take the top-k, and render their bodies as a compact retrieval-context block
# to prepend to the turn. This preserves the plan invariant: **Letta owns store + self
# edit; Luke's recall owns retrieval/ranking.** We do NOT use Letta's native
# archival/conversation search (detached in 4.3 tool-config) — Luke's fused recall is the
# retrieval surface. Fail-safe: any error returns None and the turn proceeds on core
# blocks alone (same degradation posture as build_letta_context).


def _passage_body(mem_id: str, cap: int) -> str | None:
    """Read a memory's stored body from the FTS content column, truncated to *cap* chars.

    recall() returns id/type/title/score but not the body; the body lives in memory_fts.
    None if the row is missing (fail-safe — that passage is simply skipped).
    """
    from . import memory as _memory

    try:
        row = _memory._db().execute(
            "SELECT content FROM memory_fts WHERE id = ?", (mem_id,)
        ).fetchone()
    except Exception as e:
        log.warning("letta_recall_body_failed", mem_id=mem_id, error=str(e))
        return None
    if not row or row["content"] is None:
        return None
    body = str(row["content"]).strip()
    if len(body) > cap:
        body = body[:cap].rstrip() + " …"
    return body


def build_recall_injection(
    query: str, *, k: int = 6, per_passage_cap: int = 700, total_char_budget: int = 4200
) -> str | None:
    """Retrieve the turn's top-k memories via Luke's recall() and render them for injection.

    Returns a ``<retrieved-memories>`` block to prepend to a Letta turn, or ``None`` when
    the query is empty, recall returns nothing, or any error occurs (fail-safe — the turn
    then runs on core blocks alone). ``total_char_budget`` caps the whole block so a turn's
    input stays well under the bridge's cached-context ceiling proven in Phase 1.4.
    """
    query = (query or "").strip()
    if not query:
        return None

    try:
        from . import memory as _memory

        hits = _memory.recall(query=query, limit=k)
    except Exception as e:
        log.warning("letta_recall_injection_failed", error=str(e))
        return None

    if not hits:
        return None

    lines: list[str] = []
    used = 0
    rendered_count = 0
    for h in hits:
        body = _passage_body(h["id"], per_passage_cap)
        if not body:
            continue
        entry = f'<mem id="{h["id"]}" type="{h["type"]}" title="{h["title"]}">\n{body}\n</mem>'
        if used + len(entry) > total_char_budget and rendered_count > 0:
            break  # keep the highest-ranked passages that fit; never blow the budget
        lines.append(entry)
        used += len(entry)
        rendered_count += 1

    if not lines:
        return None

    log.info(
        "letta_recall_injection",
        query_chars=len(query),
        hits=len(hits),
        rendered=rendered_count,
        chars=used,
    )
    body = "\n\n".join(lines)
    return (
        "<retrieved-memories>\n"
        "# Retrieved from your memory for THIS turn (Luke's own recall — FTS5 + semantic +\n"
        "# graph, ranked). These are turn-specific facts from your archive, beyond the\n"
        "# always-in-context core blocks. Ground your reply in them; cite specifics when relevant.\n\n"
        f"{body}\n"
        "</retrieved-memories>"
    )


# ---------------------------------------------------------------------------
# Canonical Letta turn-driver (the single place injection is applied)
# ---------------------------------------------------------------------------
#
# Every demo script (letta_live_turn, letta_self_edit_demo, letta_recall_injection…)
# hand-rolled its own drive() copy, and each would have to remember to prepend the
# recall injection. Centralising it here makes archive-injection the DEFAULT for any
# Letta turn — the production agent-loop path (when backend=letta) and the scripts both
# call this one function, so a turn can never silently ship without its retrieval half.
# Pure REST over the bridge daemon; ``inject_recall`` is fail-safe (a None injection just
# runs the turn on core blocks alone). Returns a dict, never raises on a turn error.


def drive_letta_turn(
    user_msg: str,
    agent_id: str | None = None,
    *,
    inject_recall: bool = True,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Drive one Letta agent turn, prepending Luke's recall() context by default.

    Returns ``{seconds, reply, tools, injected, error}``. ``injected`` is True when a
    recall block was actually prepended (recall found something), False when the turn ran
    on core blocks alone. ``error`` is a short string on a transport/turn failure, else None.
    """
    agent_id = agent_id or settings.letta_agent_id
    injected = False
    msg = user_msg
    if inject_recall:
        inj = build_recall_injection(user_msg)
        if inj:
            msg = f"{inj}\n\n[Answer using the retrieved facts above where relevant.]\n{user_msg}"
            injected = True

    url = f"{settings.letta_base_url}/v1/agents/{agent_id}/messages"
    data = json.dumps({"messages": [{"role": "user", "content": msg}]}).encode()
    t0 = _clock()
    try:
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (localhost)
            payload = json.load(resp)
    except Exception as e:
        return {"seconds": _clock() - t0, "reply": "", "tools": [],
                "injected": injected, "error": repr(e)[:300]}

    reply, tools = "", []
    for m in payload.get("messages", []):
        mt = m.get("message_type", "")
        if mt == "assistant_message":
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            reply += (c or "")
        elif mt == "tool_call_message":
            tools.append((m.get("tool_call") or {}).get("name"))
    log.info("letta_turn", agent_id=agent_id, seconds=round(_clock() - t0, 2),
             injected=injected, tools=tools)
    return {"seconds": _clock() - t0, "reply": reply.strip(), "tools": tools,
            "injected": injected, "error": None}


def compose_letta_turn_input(user_msg: str, *, k: int = 6) -> str:
    """Build the message body to send to the Letta agent for one turn.

    Prepends the recall injection (turn-specific archive facts, ranked by Luke's own
    recall) to the user's message, so the Letta agent — which otherwise sees only its 7
    self-editing core blocks — can ground a reply in the long-tail archive. When recall
    yields nothing (empty query, no hits, or any error) the user message passes through
    unchanged (fail-safe: the turn still runs on core blocks alone).

    This is the single composition seam the Phase 6 live turn-driver and the demo scripts
    both call, so the retrieval half of the migration lives in one tested place rather than
    being reconstructed per caller.
    """
    inj = build_recall_injection(user_msg, k=k)
    if not inj:
        return user_msg
    return f"{inj}\n\n[Answer using the retrieved facts above where relevant.]\n{user_msg}"
