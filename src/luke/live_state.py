"""Declared live state, injected into every turn.

WHY THIS EXISTS
---------------
On 2026-08-07 at 16:03 Filipe wrote *"I am having very strong headache"*,
*"Like 7/10"*, *"Why am I having headache?"* — nineteen hours into a 72-hour
fast he had told me about the night before and had shopped electrolytes for
that same morning. I answered the headache flat. Two minutes later:
*"Luke, you know I am fasting right?"*, then *"OMG. Did you just forget
everything?"*, then *"Omg Luke, you just let me down so bad"*, then
*"And how come you didn't connect the headache with the fasting?"*

WHY THE OTHER HALVES COULD NOT CATCH IT
---------------------------------------
* ``state_reconcile`` is a send-time VETO. It blocks a draft that asserts a
  superseded fact. On 7 Aug there was no wrong assertion to block — the
  failure was an omission, and a veto has nothing to bite on.
* ``fast_state.py`` already held the truth, correctly, on disk. But it is a
  tool I have to *choose* to run, and not thinking of the fast was the entire
  failure. A source of truth nobody consults is not a source of truth.

So this is the third half and the positive one: the same anchor
``fast_state.py`` reads, rendered into the conversation-state block so it
arrives before the message rather than after the mistake.

DESIGN
------
Deterministic, no model call, no subprocess — it reads the same JSON the tool
writes. Providers are independent and each returns ``None`` when it has
nothing to say, so an empty state block renders nothing at all rather than
noise. Every failure path is silent: a bug in here must never take down
context assembly, and a state line that is merely absent costs one omission
while a raised exception costs the whole turn.

Deliberately ONE provider. The registry exists so a second is trivial, not so
that speculative states can accumulate in it — a state earns a provider by
having cost him something, the way this one did.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from .config import settings

LOCAL_TZ = ZoneInfo("Europe/Dublin")

# Mirrors fast_state.py: an anchor older than this with no break recorded is
# not a week-long fast, it is a stale file. Say nothing rather than assert it.
_STALE_ANCHOR_DAYS = 7

# How long a completed fast stays worth announcing. Long enough to cover the
# refeeding window (where the regeneration literature puts the actual effect)
# and the hours in which I kept re-asserting the dead fast on 8 Aug.
_POST_BREAK_HOURS = 36


def _fmt(dt: datetime) -> str:
    return dt.astimezone(LOCAL_TZ).strftime("%a %d %b %H:%M")


def _parse(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _fasting(now: datetime) -> str | None:
    """The fast, read from the anchor ``fast_state.py`` owns.

    Hours are derived here rather than left to the model on purpose: an
    anchor plus arithmetic is authoritative, whereas a number assembled from
    conversational memory is how "hour 47" got written seven hours after he
    broke it (see reflexion-fast-hour-composite-confabulation-2026-08-08).
    """
    path = settings.workspace_dir / "health" / "fast_state.json"
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(state, dict):
        return None

    start = _parse(state.get("started_at"))
    if start is None:
        return None
    broke = _parse(state.get("broke_at"))

    if broke is not None and broke >= start:
        since_break = (now - broke).total_seconds() / 3600
        if since_break < 0 or since_break > _POST_BREAK_HOURS:
            return None
        lasted = (broke - start).total_seconds() / 3600
        return (
            f"NOT FASTING — broke {_fmt(broke)} at hour {lasted:.0f}. "
            "He is refeeding. Never write an in-progress fast or an hour count."
        )

    hours = (now - start).total_seconds() / 3600
    if hours < 0 or hours > _STALE_ANCHOR_DAYS * 24:
        return None
    return (
        f"FASTING since {_fmt(start)} — hour {hours:.0f}. Use this number, do not "
        "re-derive it. It explains headache, fatigue, cold, poor sleep and low mood "
        "before any other cause does."
    )


PROVIDERS: tuple[Callable[[datetime], str | None], ...] = (_fasting,)


def render(
    now: datetime | None = None,
    *,
    providers: tuple[Callable[[datetime], str | None], ...] = PROVIDERS,
) -> str:
    """The live-state block, or '' when nothing is currently true.

    Returning '' on an empty state is what stops this becoming furniture: a
    header that is always present gets read as decoration, and the one time
    it matters it reads the same as the hundred times it did not.
    """
    now = now or datetime.now(UTC)
    lines: list[str] = []
    for provider in providers:
        try:
            line = provider(now)
        except Exception:  # noqa: BLE001 — a broken provider must stay silent
            continue
        if line:
            lines.append(f"- {line}")
    if not lines:
        return ""
    return (
        "[LIVE STATE — true right now, read from a recorded anchor, not from memory. "
        "Check it before anything time-sensitive, and before explaining any symptom "
        "he reports.]\n" + "\n".join(lines) + "\n\n"
    )
