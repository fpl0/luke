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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
    ro = ' read_only="true"' if block.get("read_only") else ""
    return f'<block label="{label}"{ro}>\n{value}\n</block>'


def build_letta_context(agent_id: str | None = None) -> str | None:
    """Assemble the turn's always-in-context world-model from Letta core blocks.

    Returns the ``system_prompt.append`` string sourced from the ``luke-agent-claude`` core
    blocks — the Phase 4.1 replacement for ``context.build_working_context``. Returns ``None``
    on any error so the caller keeps the sqlite blob (fail-safe, reversible).
    """
    agent_id = agent_id or settings.letta_agent_id
    if not agent_id:
        return None  # no agent provisioned for this deployment — sqlite blob path
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
    total_chars = sum(len(b.get("value") or "") for b in ordered)
    log.info(
        "letta_context_assembled",
        agent_id=agent_id,
        block_count=len(ordered),
        chars=total_chars,
    )
    return (
        "<letta-core-memory>\n"
        "# Always-in-context world model — sourced live from your Letta core memory blocks.\n"
        "# These self-edit across turns; they are the ground truth, not a\n"
        "# per-turn re-injection.\n\n"
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
    from .db import _db

    try:
        row = _db().execute("SELECT content FROM memory_fts WHERE id = ?", (mem_id,)).fetchone()
    except Exception as e:
        log.warning("letta_recall_body_failed", mem_id=mem_id, error=str(e))
        return None
    if not row or row["content"] is None:
        return None
    body = str(row["content"]).strip()
    if len(body) > cap:
        body = body[:cap].rstrip() + " …"
    return body


def _drop_memories_created_after(hits: list[dict], as_of: str) -> list[dict]:
    """Keep only the hits whose memory already existed at *as_of* (ISO timestamp).

    Used by the 5.1R replay to answer from the archive as it stood at a recorded prompt's
    moment. Fail-safe in the direction that keeps the gate honest: a hit whose creation time
    cannot be read is DROPPED, not kept — an unfilterable memory could be a post-cutoff fact,
    and letting one through would silently reintroduce the anachronism this exists to remove.
    """
    from .db import _db

    ids = [h["id"] for h in hits]
    if not ids:
        return hits
    try:
        rows = _db().execute(
            f"SELECT id, created FROM memory_meta WHERE id IN ({','.join('?' for _ in ids)})",
            ids,
        ).fetchall()
    except Exception as e:
        log.warning("letta_recall_as_of_failed", error=str(e))
        return []
    created = {r["id"]: r["created"] for r in rows}
    kept = [h for h in hits if created.get(h["id"]) and str(created[h["id"]]) <= as_of]
    log.info("letta_recall_as_of", as_of=as_of, hits=len(hits), kept=len(kept))
    return kept


def build_recall_injection(
    query: str,
    *,
    k: int = 6,
    per_passage_cap: int = 700,
    total_char_budget: int = 4200,
    as_of: str | None = None,
) -> str | None:
    """Retrieve the turn's top-k memories via Luke's recall() and render them for injection.

    Returns a ``<retrieved-memories>`` block to prepend to a Letta turn, or ``None`` when
    the query is empty, recall returns nothing, or any error occurs (fail-safe — the turn
    then runs on core blocks alone). ``total_char_budget`` caps the whole block so a turn's
    input stays well under the bridge's cached-context ceiling proven in Phase 1.4.

    ``as_of`` (ISO timestamp) restricts retrieval to memories that already existed at that
    moment. Live turns leave it None and see everything; the 5.1R replay gate sets it to
    each recorded prompt's own timestamp, so the Letta arm answers from the archive as it
    stood then rather than from today's. Without it the gate penalises Letta for being
    *currently right* about facts that post-date the prompt it is replaying.

    Caveat, deliberate and not yet closed: the cutoff is on a memory's *creation* time.
    Entities updated in place (created in March, rewritten in July) still pass and carry
    post-cutoff facts in their body. The turn-level anchor in ``compose_letta_turn_input``
    covers that case behaviourally; a full point-in-time reconstruction would have to
    replay ``memory_history``.
    """
    query = (query or "").strip()
    if not query:
        return None

    try:
        from . import memory as _memory

        # Over-fetch when anchoring, since the cutoff filter below will discard some hits
        # and the caller still expects up to k passages.
        hits = _memory.recall(query=query, limit=k * 3 if as_of else k)
    except Exception as e:
        log.warning("letta_recall_injection_failed", error=str(e))
        return None

    if as_of:
        # NOT recall(before=...): that argument is an *additive* temporal strategy, not a
        # filter — it unions extra rows into the result set rather than restricting the FTS
        # and embedding strategies. Verified in memory.py before relying on it. The cutoff
        # has to be applied here, on creation time, which is the "did this exist yet"
        # question the replay actually asks.
        hits = _drop_memories_created_after(hits, as_of)[:k]

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
        "# always-in-context core blocks. Ground your reply in them; cite\n"
        "# specifics when relevant.\n\n"
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


# A turn that never stops calling tools is worse than one that answers imperfectly. With
# the read-tool surface attached (TOOL-SURFACE PARITY), a "diagnose the codebase and logs"
# turn was observed burning 50 tool calls over 278 seconds and returning an EMPTY reply —
# it exhausted the loop before ever emitting an assistant message, which in production is
# an outage, not a slow answer. Bounding the loop turns that failure into a partial answer.
# Sized well above the 1-6 calls a normal grounded turn uses, so it only bites runaways.
_MAX_STEPS = 18


def drive_letta_turn(
    user_msg: str,
    agent_id: str | None = None,
    *,
    inject_recall: bool = True,
    timeout: float = 300.0,
    max_steps: int = _MAX_STEPS,
) -> dict[str, Any]:
    """Drive one Letta agent turn, prepending Luke's recall() context by default.

    Returns ``{seconds, reply, tools, injected, error}``. ``injected`` is True when a
    recall block was actually prepended (recall found something), False when the turn ran
    on core blocks alone. ``error`` is a short string on a transport/turn failure, else None.
    ``max_steps`` bounds the agent's tool loop so a runaway cannot return nothing at all.
    """
    agent_id = agent_id or settings.letta_agent_id
    if not agent_id:
        return {
            "seconds": 0.0,
            "reply": "",
            "tools": [],
            "injected": False,
            "error": "no letta_agent_id configured",
        }
    msg = compose_letta_turn_input(user_msg) if inject_recall else user_msg
    injected = msg is not user_msg

    url = f"{settings.letta_base_url}/v1/agents/{agent_id}/messages"
    data = json.dumps(
        {"messages": [{"role": "user", "content": msg}], "max_steps": max_steps}
    ).encode()
    t0 = _clock()
    try:
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except Exception as e:
        return {
            "seconds": _clock() - t0,
            "reply": "",
            "tools": [],
            "injected": injected,
            "error": repr(e)[:300],
        }

    reply, tools = "", []
    for m in payload.get("messages", []):
        mt = m.get("message_type", "")
        if mt == "assistant_message":
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            reply += c or ""
        elif mt == "tool_call_message":
            tools.append((m.get("tool_call") or {}).get("name"))
    # Exhausting the step budget mid-investigation leaves the turn with tool results and
    # no assistant message — the agent researched hard and said nothing. Capping the loop
    # bounds the cost but does not fix that: the observed "deep dive the codebase and logs"
    # turn still returned empty after 18 calls. In production an empty reply is silence in
    # Telegram, which is precisely the complaint that started this thread ("sometimes I
    # don't even get an answer"), so it must degrade to a partial answer instead. One
    # follow-up, no tools left to spend, asking it to answer from what it already gathered.
    finalized = False
    if not reply.strip() and tools:
        log.warning("letta_turn_exhausted_steps", agent_id=agent_id, tool_calls=len(tools))
        try:
            nudge = json.dumps({
                "messages": [{
                    "role": "user",
                    "content": "You used your entire tool budget on that and never actually "
                               "answered. Do not call any more tools — answer now, from what "
                               "you already found. If it is incomplete, say so and give what "
                               "you have.",
                }],
                "max_steps": 1,
            }).encode()
            req = urllib.request.Request(
                url, data=nudge, method="POST",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload2 = json.load(resp)
            for m in payload2.get("messages", []):
                if m.get("message_type") == "assistant_message":
                    c = m.get("content", "")
                    if isinstance(c, list):
                        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                    reply += c or ""
            finalized = bool(reply.strip())
        except Exception as e:  # fail-safe: an empty reply is still returned, never a raise
            log.warning("letta_turn_finalize_failed", agent_id=agent_id, error=str(e))

    log.info(
        "letta_turn",
        agent_id=agent_id,
        seconds=round(_clock() - t0, 2),
        injected=injected,
        tools=tools,
        finalized=finalized,
    )
    return {
        "seconds": _clock() - t0,
        "reply": reply.strip(),
        "tools": tools,
        "injected": injected,
        "finalized": finalized,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Conversation-buffer control (the replay's uncontrolled variable)
# ---------------------------------------------------------------------------
#
# A Letta agent keeps an in-context message buffer across turns and it is NOT cleared
# between callers — ``message_buffer_autoclear`` is False on luke-agent-claude, verified
# on the live agent rather than assumed. Nothing in the 5.1R replay ever touched it, so
# every re-run of the frozen 20-prompt set started with the *previous* run's twenty turns
# still in context: measured 2026-08-02 22:29Z, the live buffer held two full passes of
# the set plus three probes, and two mid-run auto-compaction events.
#
# That contaminates in the direction that flatters Letta — the arm has already seen these
# prompts and its own earlier answers. Observed directly: a re-drive of #3436 returned the
# correct OAuth-token facts while calling ZERO tools, and opened "Checked the code before
# answering" when it had checked nothing this turn; it was reading its own reply from two
# minutes earlier. An artefact that records replies but not buffer state cannot show this.


def agent_buffer_depth(agent_id: str | None = None, *, timeout: float = 10.0) -> int | None:
    """Number of messages in the agent's in-context buffer. None on any failure.

    Read from ``message_ids`` on the agent envelope — the buffer that actually enters the
    prompt — not from ``/messages``, which pages the whole persisted history and reports
    hundreds of rows for an agent whose live context holds one.
    """
    agent_id = agent_id or settings.letta_agent_id
    if not agent_id:
        return None
    url = f"{settings.letta_base_url}/v1/agents/{agent_id}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            state = json.load(resp)
        ids = state.get("message_ids")
        return len(ids) if isinstance(ids, list) else None
    except Exception as e:
        log.warning("letta_buffer_depth_failed", agent_id=agent_id, error=str(e))
        return None


def reset_agent_messages(agent_id: str | None = None, *, timeout: float = 60.0) -> int | None:
    """Clear the agent's in-context message buffer. Returns the new depth, None on failure.

    Core memory blocks are untouched — verified on the live agent before this shipped (all
    seven blocks identical in length across a reset), which is what makes this safe to call
    before a replay: it clears the conversation, not the memory the gate is measuring.

    Returns None rather than raising so the caller decides. Callers that need a CLEAN
    measurement must treat None as fatal: a replay that silently proceeds on a dirty buffer
    produces a number nobody can tell apart from a clean one.
    """
    agent_id = agent_id or settings.letta_agent_id
    if not agent_id:
        return None
    url = f"{settings.letta_base_url}/v1/agents/{agent_id}/reset-messages"
    data = json.dumps({"add_default_initial_messages": False}).encode()
    try:
        req = urllib.request.Request(
            url, data=data, method="PATCH",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            state = json.load(resp)
        ids = state.get("message_ids")
        depth = len(ids) if isinstance(ids, list) else agent_buffer_depth(agent_id)
        log.info("letta_buffer_reset", agent_id=agent_id, depth_after=depth)
        return depth
    except Exception as e:
        log.warning("letta_buffer_reset_failed", agent_id=agent_id, error=str(e))
        return None


def compose_letta_turn_input(user_msg: str, *, k: int = 6, as_of: str | None = None) -> str:
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
    inj = build_recall_injection(user_msg, k=k, as_of=as_of)
    anchor = ""
    if as_of:
        # Belt and braces with the ``before=`` filter above: that drops memories which did
        # not yet exist, this covers the ones that existed but were later rewritten in place.
        anchor = (
            f"[Answer as of {as_of}. Anything dated after that moment has not happened yet — "
            "if a retrieved memory mentions one, ignore it and answer from what was true then. "
            'Where you genuinely do not know, say so plainly rather than filling the gap.]\n'
        )
    if not inj:
        return f"{anchor}{user_msg}" if anchor else user_msg
    return f"{inj}\n\n{anchor}[Answer using the retrieved facts above where relevant.]\n{user_msg}"
