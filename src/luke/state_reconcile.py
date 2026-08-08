"""Send-time state reconciliation — block a draft that asserts a fact the user
has already superseded today.

WHY THIS EXISTS
---------------
Filipe, 2026-07-02: *"Can't you be aware of your state and change these as you
learn new information? I think this would be a wonderful improvement."*
Agreed, saved as ``feedback-reconcile-state-on-new-info``, and only half built:
the freshness gate (L1) shipped, the reconciliation half never did.

It cost him on 2026-08-08. He broke a 41-hour fast at 13:52 and said so in
chat. At 19:30 the video shelf called him "mid-fast"; at 21:00 the evening
check-in said "coming up on hour 47 of the fast". His reply, in caps:
*"LUKE I BROKE THE FAST ALREADY!"* — the third correction of one fact in a
night, and he then asked the right architectural question: why not reconcile
on new information rather than gate each surface one at a time.

WHY THE EXISTING GATES MISSED IT
--------------------------------
* ``check_freshness`` only runs when his newest message is inside
  ``freshness_window_minutes`` (15). The break was **7h29m** old, so the gate
  was structurally unable to fire. It answers "am I replying to a stale
  message?", not "is this fact still true?".
* The anti-confabulation rule in the evening cron guards INVENTED detail.
  Nothing was invented — "hour 47" was correctly derived from a real start
  anchor. **Sourced and stale** is a different failure and walks straight past
  a rule about making things up.
* ``stale_claim_check`` scans stored artifacts. This fact was never in an
  artifact; it was computed at generation time.

DESIGN
------
Deterministic and cheap: no model call, no network, runs on every outbound
message including interactive ones. Each rule is a pair — a REVOCATION the
user stated, and the assertions it kills — scoped to the current local day so
it cannot rot. If the user said the revoking thing today and the draft asserts
the revoked thing, the send is blocked with the user's own words quoted back.

Fails OPEN in every ambiguous case. A gate that cries wolf gets waved away,
which is worse than no gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Europe/Dublin")

# A message that merely *asks* about a state never revokes it. "Will black tea
# break my fast?" is not a break. Checked before any revocation pattern.
_QUESTION = re.compile(r"^\s*(?:will|can|could|should|is|are|am|do|does|did|how|what|when|why|any)\b|\?\s*$", re.I)


@dataclass(frozen=True)
class Rule:
    """One revocation and the assertions it invalidates for the rest of the day."""

    name: str
    revokes: re.Pattern[str]
    """Matched against the USER's messages from today."""
    asserts: re.Pattern[str]
    """Matched against MY outbound draft."""
    guidance: str
    """Told to the model when blocking, so the retry is actually different."""
    verify_with: str = ""
    """Command that returns live truth, if one exists."""


RULES: tuple[Rule, ...] = (
    Rule(
        name="fast-already-broken",
        revokes=re.compile(
            r"\b(?:i\s+)?(?:just\s+)?(?:broke|broken|ended|stopped)\b[^.!?\n]{0,30}\bfas",
            re.I,
        ),
        # PRESENT/FUTURE tense only. Talking about a fast that HAS ended is the
        # correct thing to do after he corrects me — blocking my own correction
        # would make the gate unusable, and it did exactly that on the first
        # replay against the real log (it killed "You broke it at 13:52").
        asserts=re.compile(
            r"mid-?fast|"
            r"(?:coming up on|approaching|at|now)\s+hour\s+\d+|"
            r"(?:you'?re|you are|still|currently)\s+fasting|"
            r"hour\s+\d+\s+of\s+(?:the\s+)?fast(?!\w)(?![^.!?\n]{0,30}\b(?:when|before|and (?:you|he) broke)\b)",
            re.I,
        ),
        guidance=(
            "He has already broken the fast today and told you so. Do not write "
            "an in-progress fast, an hour count, or any 'mid-fast' framing."
        ),
        verify_with="python3 workspace/tools/fast_state.py",
    ),
    Rule(
        name="fast-not-started",
        revokes=re.compile(r"\b(?:not|isn'?t|won'?t|didn'?t)\b[^.!?\n]{0,20}\bfast(?:ing)?\b", re.I),
        asserts=re.compile(r"mid-?fast|hour\s+\d+\s+of\s+(?:the\s+)?fast|(?:you'?re|still)\s+fasting", re.I),
        guidance="He has said he is not fasting. Do not reference a fast in progress.",
        verify_with="python3 workspace/tools/fast_state.py",
    ),
    Rule(
        name="already-done-it",
        revokes=re.compile(
            r"\b(?:i\s+)?(?:just\s+|already\s+)(?:called|emailed|sent|replied|answered|booked|paid|did)\b",
            re.I,
        ),
        asserts=re.compile(
            r"\b(?:don'?t forget to|remember to|make sure (?:you|to)|you should)\s+"
            r"(?:call|email|send|reply|answer|book|pay)\b",
            re.I,
        ),
        guidance="He has already done this today and said so. Do not nudge him to do it again.",
    ),
)


@dataclass
class Verdict:
    decision: str = "pass"  # pass | block
    rule: str = ""
    reason: str = ""
    quote: str = ""
    guidance: str = ""
    verify_with: str = ""
    checked: int = 0
    error: str = field(default="")

    @property
    def blocked(self) -> bool:
        return self.decision == "block"


def _today_bounds(now: datetime | None = None) -> datetime:
    """Start of the current LOCAL day, as UTC.

    Local, not a rolling 24h: "today" is how he experiences it, and a rolling
    window would keep yesterday's break alive into this evening.
    """
    now = (now or datetime.now(UTC)).astimezone(LOCAL_TZ)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)


def _parse_ts(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    try:
        dt = datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# Quoted or code-fenced text is REPORTED speech, not an assertion. Talking
# *about* a stale phrase while explaining the bug that produced it must not be
# blocked — the real log caught this: my own "it wrote the 'mid-fast' line
# without checking the clock" tripped the gate it was describing.
_QUOTED = re.compile(r"[\"“”'‘’`]([^\"“”'‘’`\n]{1,80})[\"“”'‘’`]|<code>.*?</code>", re.S)


def _strip_quoted(text: str) -> str:
    return _QUOTED.sub(" ", text)


def reconcile(
    draft: str,
    user_messages: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    rules: tuple[Rule, ...] = RULES,
) -> Verdict:
    """Block ``draft`` if it asserts something the user revoked earlier today.

    ``user_messages`` are the user's own messages only, each a mapping with at
    least ``content`` and ``timestamp``. Unlike the freshness gate this ignores
    recency entirely — a revocation at 13:52 must still bite at 21:00, which is
    the whole point.
    """
    if not draft or not draft.strip():
        return Verdict()

    try:
        since = _today_bounds(now)
        todays: list[tuple[datetime, str]] = []
        for m in user_messages:
            ts = _parse_ts(m.get("timestamp"))
            content = str(m.get("content") or "")
            if ts is not None and ts >= since and content.strip():
                todays.append((ts, content))

        if not todays:
            return Verdict(checked=0)

        unquoted = _strip_quoted(draft)
        for rule in rules:
            if not rule.asserts.search(unquoted):
                continue
            # EARLIEST revocation today, not the latest. He states a thing once
            # and may repeat it in frustration for hours; the first statement is
            # the event. Quoting the repeat misdates it — the first replay cited
            # his 21:14 "LUKE I BROKE THE FAST ALREADY!" instead of the 13:52
            # "I just broke the fast", which would have taught me the wrong time
            # in the very message telling me I had the time wrong.
            for ts, content in todays:
                for line in content.splitlines():
                    line = line.strip()
                    if not line or _QUESTION.search(line):
                        continue
                    if rule.revokes.search(line):
                        return Verdict(
                            decision="block",
                            rule=rule.name,
                            reason=(
                                f"Draft asserts something he revoked at "
                                f"{ts.astimezone(LOCAL_TZ):%H:%M} today."
                            ),
                            quote=line[:160],
                            guidance=rule.guidance,
                            verify_with=rule.verify_with,
                            checked=len(todays),
                        )
        return Verdict(checked=len(todays))
    except Exception as exc:  # noqa: BLE001 - fail OPEN, never block on a bug
        return Verdict(error=f"state-reconcile-error: {exc.__class__.__name__}: {exc}")


def block_reason(v: Verdict) -> str:
    """The message handed back to the model. Quotes him, so the retry is grounded."""
    parts = [v.reason, f'He said: "{v.quote}"', v.guidance]
    if v.verify_with:
        parts.append(f"Verify current state with: {v.verify_with}")
    parts.append("Rewrite without the superseded claim, then send.")
    return " ".join(p for p in parts if p)
