"""Stop a 4-or-5 deep-work rating in a session that shipped nothing.

Why this exists
---------------
`proc-deep-work-session-protocol` already says it in terms: *"Rate against
whether the work landed with them, not against how much you produced. Zero
replies and zero use of what you shipped is not a 4."*

On 2026-08-16 the Sunday review measured what that instruction was actually
worth — 79 sessions, 99% of ratings inside the 3-4 band, silent-day premium
-0.14 — and built `deep_work_calibration.py` to grade the grader. The lesson
was saved as `insight-a-self-issued-rating-scale-collapses-to-a-receipt`.

Eighteen days later, on 2026-09-03, the same reading: **14 sessions, 100% in
the 3-4 band, silent-day premium -0.07.** Nothing moved. The reason is where
the rule lives. `deep_work_calibration.py` prints at midnight, *after* the
ratings are already in the table, and the procedure memory is advisory. The
one surface that runs at the moment a rating is issued — the deep-work session
itself — carried none of it.

That is the signature of a corrective that needs enforcement rather than
another reminder (`proc-recurring-failure-to-guardrail`:
three same-cause reflexions become an executed gate, not a fourth note).

What it deliberately does NOT do
--------------------------------
It does not manufacture variance, which `proc-deep-work-calibration` warns is
Goodhart with extra steps. It fires ONLY on the exact sentence the protocol
already contains — a 4 or a 5 when nothing left the building this run — and it
is **one-shot per run**: the immediate re-call goes through at whatever rating
is chosen. A deliberately silent goal (bloods, citizenship — both wait-shaped
by design) can still be rated a 4; the gate costs one considered decision, not
the rating.

Threshold picked from data, not taste. Over the 14 rated sessions of
2026-09-01/02, nine shipped nothing in-session; gating at >=3 would have fired
on all nine and trained a reflex click-through. Gating at >=4 fires on three.
"""

from __future__ import annotations

from typing import Final

TOOL_NAME: Final = "mcp__luke__log_deep_work_quality"

# The run's shipped fact, published so the WRITER can persist it.
#
# 2026-09-06. Until then this module was handed `shipped` on every rating
# and dropped it the moment it decided not to block. `deep_work_calibration.py`
# then reconstructed reach from the messages table by DATE and printed the
# caveat in its own output — "nothing links a send to the session that produced
# it". The fact was computed at the point of truth and guessed at one layer
# down.
#
# 2026-09-07, marker SHIPPED-NEVER-CROSSED-THE-TASK-BOUNDARY. The 6 Sep fix used
# a ContextVar, reasoning that a module global would race because two Luke
# sessions on one trigger is a documented event (2026-09-03). The concurrency
# concern was real and the remedy defeated the whole mechanism: the PreToolUse
# hook and the MCP tool handler run in DIFFERENT asyncio tasks, and a Task
# copies its context at creation, so a value set in the hook is structurally
# invisible to the writer. Every one of the 514 rows was NULL — including rows
# written after the code went live — and nothing caught it, because
# `deep_work_calibration.py` reports an unpopulated column as "predates the
# column", which is indistinguishable from "the writer is broken".
#
# So: a module-level map, which DOES cross task boundaries, keyed by goal_id to
# keep the isolation the ContextVar was chosen for. Two concurrent sessions
# rating DIFFERENT goals are now cleanly separated; the residual race is two
# sessions rating the SAME goal in the same instant, which swaps one boolean.
# That is strictly better than a mechanism that loses the value every time.
#
# None means "no gate hook ran for this call" — honest unknown, never False.
_PENDING_SHIPPED: dict[str, bool] = {}

# The hook fires on every rating attempt; the write only follows the ones that
# are not blocked. Blocked attempts would otherwise leak an entry each, so the
# map is bounded and evicts oldest-first (dicts preserve insertion order).
_PENDING_MAX: Final = 64


def publish_shipped(goal_id: str, shipped: bool) -> None:
    """Record the run's shipped fact for the write that is about to happen."""
    if not goal_id:
        return
    _PENDING_SHIPPED.pop(goal_id, None)  # re-insert so eviction order is fresh
    _PENDING_SHIPPED[goal_id] = shipped
    while len(_PENDING_SHIPPED) > _PENDING_MAX:
        _PENDING_SHIPPED.pop(next(iter(_PENDING_SHIPPED)))


def consume_shipped(goal_id: str) -> bool | None:
    """Take the published fact for this goal, or None if no hook published one.

    Read-and-clear: a stale entry must never attach itself to a later rating
    that ran without a hook.
    """
    return _PENDING_SHIPPED.pop(goal_id, None)


# Below this, the gate stays out of the way. 3 is the honest rating for a
# session that did real work nobody has seen yet; 4 is a claim about landing.
_MIN_GATED_RATING: Final = 4

REASON: Final = (
    "A 4 means the work LANDED, and nothing left the building this session — "
    "no message, no document, no artifact delivered. The deep-work protocol is "
    "explicit: 'Rate against whether the work landed with them, not against how "
    "much you produced. Zero replies and zero use of what you shipped is not a "
    "4.'\n"
    "Measured 2026-09-03: 14 sessions, 100% of ratings in the 3-4 band, "
    "silent-day premium -0.07. The scale is a receipt for having run, and the "
    "advisory version of this rule has been in memory since 16 Aug without "
    "moving the histogram.\n"
    "So decide, once: if this session produced work he has not seen and may "
    "never use, it is a 3 — or a 2 if nothing of his came back either. If it "
    "genuinely landed (a parked send with a dated owner, an artifact he asked "
    "for, a correction that changed a live surface), re-call with the same "
    "rating and it will go through. This gate fires once per run."
)


def blocks(tool_name: str, tool_input: object, *, shipped: bool, already_fired: bool) -> bool:
    """True when this rating claims landing that the run's own record denies.

    `shipped` is the caller's fact — did any send/artifact tool succeed in this
    run. `already_fired` keeps it one-shot so an override is always available.
    """
    if already_fired or shipped or tool_name != TOOL_NAME:
        return False
    if not isinstance(tool_input, dict):
        return False
    try:
        rating = int(tool_input.get("rating"))  # type: ignore[arg-type]
    except TypeError, ValueError:
        return False
    return rating >= _MIN_GATED_RATING
