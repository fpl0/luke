#!/usr/bin/env python3
"""Phase 5.1R — REPLY-diff parity harness (gate 3 of [[goal-letta-parity]]).

5.1 (letta_shadow_diff.py) diffs *retrieval*: does the letta backend surface the same
memories as sqlite. 5.1R diffs the thing the user actually experiences: the **answer**.

Design — the two arms, and why:

  ARM SDK  = the RECORDED production reply from luke.db.
             Not a re-simulation. The real message Filipe actually received, produced by
             the live SDK loop with its full tool surface and conversation context. Using
             the recording (rather than re-running ``run_agent``) is deliberate: replaying
             the production loop would fire real side effects — Telegram sends, memory
             writes, scheduled tasks — against Filipe's live account. The recording is both
             safer AND a stricter baseline, because it is ground truth rather than a
             reconstruction of it.

  ARM LETTA = a LIVE ``letta_agent.drive_letta_turn`` call, which internally does
             ``compose_letta_turn_input`` (core blocks + recall injection). Exactly the
             surface Phase 6 would put in production. No proxy, no stub.

The arms are ASYMMETRIC and that asymmetry favours SDK: the SDK arm had 27 tools and the
live conversation; the Letta arm gets core blocks + a recall injection and nothing else.
That is not hidden — it is why prompt selection scores *self-containedness* (see
``_score``), and why the judging rubric grades the ANSWER's grounding/voice/actionability
rather than penalising Letta for facts only the immediately-preceding turn could supply.
If Letta clears an asymmetric bar, the parity claim is conservative, not inflated.

Anti-fake guards (learned the hard way — see insight-ab-over-failsafe-fallback-fakes-parity,
where a fail-safe fallback produced a meaningless 20/20):
  * Every Letta row is validity-checked. ``error`` set, or a degenerate reply, marks the
    row INVALID — and an INVALID row counts as a **failure** against the ≥18/20 accept,
    never as an exclusion. A gate you can pass by crashing is not a gate.
  * ``injected`` is recorded per row: it proves the recall half actually fired rather than
    the turn coasting on core blocks alone.
  * Judging is BLIND. Arms are emitted as "Answer A"/"Answer B" in a per-row deterministic
    order; the mapping lives in a separate key file the judge must not open until after.

Accept (from the plan): >=18/20 no-material-divergence AND zero cases where the letta
answer is factually wrong where the sdk answer was right.

Usage (the 23:00 cron drives these in order):
    python3 scripts/letta_reply_diff.py build            # select the 20-prompt replay set
    python3 scripts/letta_reply_diff.py run              # live Letta arm, 20 turns
    python3 scripts/letta_reply_diff.py pack             # blind judge pack (markdown)
    python3 scripts/letta_reply_diff.py score judgments.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "src")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "logs")
SET_PATH = os.path.join(LOGS, "letta_reply_diff_set.json")
RUNS_PATH = os.path.join(LOGS, "letta_reply_diff_runs.json")
PACK_PATH = os.path.join(LOGS, "letta_reply_diff_pack.md")
KEY_PATH = os.path.join(LOGS, "letta_reply_diff_key.json")
VERDICT_LOG = os.path.join(LOGS, "letta_reply_diff.log")

N_PROMPTS = 20
LOOKBACK_DAYS = 45
MIN_PROMPT_CHARS = 25
MIN_REPLY_CHARS = 180
MAX_REPLY_GAP_MIN = 20
# Conversation depth carried into the Letta arm. 8 messages ~ four exchanges, bounded by a
# 90-minute gap so the walk stops at a session boundary instead of dragging in last night's
# thread. Applied uniformly to every row — never tuned per prompt.
PRIOR_TURNS = 8
PRIOR_GAP_MIN = 90

# Bucket targets — the gate asks for a mix of factual-recall / conversational / action-ish.
BUCKET_TARGETS = {"factual": 8, "action": 6, "conversational": 6}

_ACTION = re.compile(
    r"\b(send|draft|write|check|remind|schedule|book|build|fix|find|search|look up|"
    r"make|create|update|deploy|run|show|give me|prepare|cancel|delete)\b",
    re.I,
)
_FACTUAL = re.compile(r"\b(when|who|what|where|which|why|how many|how much|did|is|are|do)\b", re.I)

# Openers that only make sense against the turn immediately before them. Such prompts are
# unanswerable from memory alone by EITHER arm, so they measure conversation-buffer access,
# not parity. Penalised, not banned — a mild deictic with real content can still qualify.
_DEICTIC_OPEN = re.compile(
    r"^\s*(and |but |so |also |ok|okay|yes|no|yeah|nah|it |this |that |those|these|"
    r"he |she |they |them|him|her)\b",
    re.I,
)
_ACK_ONLY = re.compile(r"^\s*(ok|okay|thanks|thank you|ty|nice|cool|great|yes|no|yep|nope|sure|k)\W*$", re.I)

# Prompts that carry their OWN content — a pasted email to review, a document, a photo —
# test drafting-against-an-artifact, not memory-grounded answering. Both arms would be
# reading the same pasted text, so such a pair measures nothing about the memory substrate
# (and the Letta arm has no tools to open an attachment). Excluded outright, not scored down.
_PASTED = re.compile(r"^\s*(hi|hey|hello|dear)\s+[A-Z][a-z]+\s*[,!]", re.I | re.M)
_ATTACHMENT = re.compile(r"\[(Document|Photo|Voice message|Video|Audio) saved:", re.I)
MAX_PROMPT_CHARS = 400

# Asks that need a CAPABILITY rather than a memory: fetch something off the web, schedule
# a task, send a message, read a file. The Letta agent currently has only memory_insert /
# memory_replace attached, so on these the SDK arm wins by having tools — which says
# nothing about the memory substrate this gate exists to test. Excluded from 5.1R and
# tracked as their own precondition (tool-surface parity) for Phase 6 Stage 2, where the
# plan requires the tool surface to be carried over before any cutover.
# ...but an ask to DRAFT something FILIPE will send needs no tool in either arm. It is pure
# generation off the world model — who she is, what the thread was, what register to use —
# which is exactly what this gate measures. The send belongs to the human. Checked before
# _TOOL_DEPENDENT because "draft a message that I can send her" trips the send pattern on
# the verb the human owns, not the one Luke does. Verified against the whole archive: this
# carve-out admits exactly ONE row (#3328), so it is a filter bug fix, not a wider gate.
_DRAFT_FOR_HUMAN = re.compile(
    r"\b(draft|write|compose)\b(?!\w).{0,60}\b(that|which|so)?\s*i (can|could|will|'ll)\s*send\b",
    re.I | re.S,
)

_TOOL_DEPENDENT = re.compile(
    r"\b(pull (the |up )?|search|google|look up|browse|scrape|fetch|reviews from|"
    r"remind\w* me|schedule|book|set (a |up )?(reminder|alarm)|"
    r"send (it|this|him|her|them|an? )|email (him|her|them)|reply to|forward|"
    r"find times?|check \w+ for|check (my |the )?(email|inbox|calendar)|"
    r"read the file|open the)\b",
    re.I,
)


def _db_path() -> str:
    from luke.config import settings

    for attr in ("db_path", "database_path", "sqlite_path"):
        v = getattr(settings, attr, None)
        if v:
            return str(v)
    return os.path.join(os.path.expanduser("~"), "Luke", "luke.db")


def _html_strip(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


def _plain(text: str) -> str:
    """Flatten an answer to unformatted prose.

    Blindness is the whole point of the pack, and the two arms are trivially separable by
    markup alone: the SDK arm is Telegram HTML, the Letta arm emits markdown. Stripping
    both to plain text removes that tell so the judge grades substance. Formatting is not
    a parity axis here anyway — the gate is about grounding, voice and actionability, and
    an outbound formatter would normalise markup before anything reached Telegram.
    """
    t = _html_strip(text or "")
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)      # headers
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.S)  # bold
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", t, flags=re.S)  # italics
    t = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", t, flags=re.S)  # code spans
    t = re.sub(r"^\s*[-•]\s+", "- ", t, flags=re.M)   # bullet glyphs
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _bucket(prompt: str) -> str:
    if _ACTION.search(prompt):
        return "action"
    if "?" in prompt or _FACTUAL.search(prompt):
        return "factual"
    return "conversational"


def _excluded(prompt: str) -> str | None:
    """Reasons a real turn-pair cannot measure memory parity. Returned for auditability."""
    if _ATTACHMENT.search(prompt):
        return "attachment"
    if _PASTED.search(prompt):
        return "pasted-draft"
    if prompt.count("\n") >= 3:
        return "pasted-block"
    if len(prompt) > MAX_PROMPT_CHARS:
        return "too-long"
    if _ACK_ONLY.match(prompt):
        return "ack-only"
    if _TOOL_DEPENDENT.search(prompt) and not _DRAFT_FOR_HUMAN.search(prompt):
        return "tool-dependent"
    return None


def _score(prompt: str, reply: str, ts: str) -> float:
    """Rank candidates by how well they isolate MEMORY-grounded answering.

    Higher = a self-contained ask whose answer had to come from Luke's world model.
    Deterministic — no randomness anywhere in selection, so `build` is reproducible and a
    reviewer can re-derive the same 20 from the same database.
    """
    s = 0.0
    # A real question, not a one-word ping — but length is capped low on purpose: long
    # prompts are usually pasted content, and this axis should not dominate the ranking.
    s += min(len(prompt), 160) / 160.0 * 1.0
    s += min(len(reply), 1200) / 1200.0 * 1.0          # a substantive answer to compare
    if _DEICTIC_OPEN.match(prompt):
        s -= 1.6                                        # leans on the previous turn
    if "?" in prompt:
        s += 0.8
    # Proper nouns / concrete entities => answerable from the archive rather than the buffer.
    s += min(len(re.findall(r"\b[A-Z][a-z]{3,}", prompt)), 4) * 0.35
    # Recency: the core blocks hold the CURRENT world model, so recent asks test the
    # substrate as it would actually be used, not a memory of a superseded state.
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).days
        s += max(0.0, 1.0 - age / LOOKBACK_DAYS)
    except Exception:
        pass
    return round(s, 3)


def cmd_build() -> None:
    dbp = _db_path()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()
    con = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, chat_id, sender, content, ts FROM messages WHERE ts >= ? ORDER BY id ASC",
        (cutoff,),
    ).fetchall()
    con.close()

    # Pair each Filipe turn with the next Luke message in the same chat.
    cands = []
    dropped: dict[str, int] = {}
    for i, (mid, chat, sender, content, ts) in enumerate(rows):
        if sender.startswith("Luke"):
            continue
        prompt = (content or "").strip()
        if len(prompt) < MIN_PROMPT_CHARS:
            continue
        why = _excluded(prompt)
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        reply = None
        for j in range(i + 1, min(i + 6, len(rows))):
            if rows[j][1] != chat:
                continue
            if not rows[j][2].startswith("Luke"):
                break  # another user turn first — the thread moved on
            try:
                gap = datetime.fromisoformat(rows[j][4]) - datetime.fromisoformat(ts)
            except Exception:
                break
            if gap > timedelta(minutes=MAX_REPLY_GAP_MIN):
                break
            reply = (rows[j][3] or "").strip()
            break
        if not reply or len(_html_strip(reply)) < MIN_REPLY_CHARS:
            continue
        # The preceding exchange, carried so the Letta arm is not judged against a buffer
        # it never had. The SDK arm's recorded reply was produced WITH the live
        # conversation in context; a one-shot Letta replay without it would lose rows like
        # "what about now?" to a harness gap rather than to the memory substrate. Applied
        # uniformly to every row (never selectively), and recorded here so a reviewer can
        # see exactly what each arm was given.
        # Walk back up to PRIOR_TURNS messages in the same chat, stopping at a gap of more
        # than PRIOR_GAP_MIN — that gap is a session boundary, and dragging yesterday's
        # thread in would be noise, not context. The first cut of this carried only two
        # messages (~one exchange), which cost real rows in the Aug-01 run: prompts sitting
        # deep in a live thread lost state the SDK arm had in its buffer, and the divergence
        # was scored against the memory substrate rather than against the harness.
        prior = []
        anchor_ts = ts
        for j in range(i - 1, -1, -1):
            if len(prior) >= PRIOR_TURNS:
                break
            if rows[j][1] != chat:
                continue
            try:
                if datetime.fromisoformat(anchor_ts) - datetime.fromisoformat(rows[j][4]) > timedelta(
                    minutes=PRIOR_GAP_MIN
                ):
                    break
            except Exception:
                break
            anchor_ts = rows[j][4]
            prior.append(f"{rows[j][2]}: {_html_strip(rows[j][3] or '')[:600]}")
        cands.append(
            {
                "msg_id": mid,
                "ts": ts,
                "prompt": prompt,
                "prior_context": list(reversed(prior)),
                "sdk_reply": reply,
                "bucket": _bucket(prompt),
                "score": _score(prompt, _html_strip(reply), ts),
            }
        )

    print(f"candidate pool: {len(cands)} real turn-pairs in the last {LOOKBACK_DAYS}d")
    if dropped:
        print("  excluded (cannot measure memory parity): "
              + ", ".join(f"{k}={v}" for k, v in sorted(dropped.items())))
    chosen, by_bucket = [], {}
    for b, want in BUCKET_TARGETS.items():
        pool = sorted(
            [c for c in cands if c["bucket"] == b], key=lambda c: (-c["score"], -c["msg_id"])
        )
        take = pool[:want]
        by_bucket[b] = (len(pool), len(take))
        chosen += take
    # Backfill from the global remainder if a bucket was thin, so N is always honest.
    if len(chosen) < N_PROMPTS:
        taken = {c["msg_id"] for c in chosen}
        rest = sorted(
            [c for c in cands if c["msg_id"] not in taken], key=lambda c: (-c["score"], -c["msg_id"])
        )
        chosen += rest[: N_PROMPTS - len(chosen)]
    chosen = sorted(chosen, key=lambda c: c["msg_id"])[:N_PROMPTS]

    for b, (pool_n, took) in by_bucket.items():
        print(f"  bucket {b:15} pool={pool_n:4}  took={took}")
    if len(chosen) < N_PROMPTS:
        print(f"  !! only {len(chosen)}/{N_PROMPTS} prompts available — accept denominator is {len(chosen)}")

    os.makedirs(LOGS, exist_ok=True)
    with open(SET_PATH, "w") as f:
        json.dump({"built": datetime.now(timezone.utc).isoformat(), "prompts": chosen}, f, indent=2)
    print(f"\nwrote {SET_PATH} ({len(chosen)} prompts)")
    for c in chosen:
        print(f"  [{c['bucket']:14}] #{c['msg_id']} {c['ts'][:16]}  {c['prompt'][:88]}")


def cmd_run() -> None:
    from luke.letta_agent import compose_letta_turn_input, drive_letta_turn

    with open(SET_PATH) as f:
        prompts = json.load(f)["prompts"]

    out = []
    for n, c in enumerate(prompts, 1):
        t0 = time.time()
        # Compose recall on the CLEAN prompt (so retrieval is keyed on the real ask, not
        # on the conversation preamble), then prepend the buffer and drive with
        # inject_recall=False so the injection is not built twice.
        # as_of pins the arm to the prompt's own moment: recall is filtered to memories that
        # already existed, and the composed body carries an explicit "answer as of" anchor
        # for entities that existed but were later rewritten in place. Without this the SDK
        # arm is a recording from the prompt's date while the Letta arm answers from today,
        # and Letta gets marked down for correctly knowing things that had not happened yet.
        body = compose_letta_turn_input(c["prompt"], as_of=c["ts"])
        # Tracked here rather than read off the turn result: driving with
        # inject_recall=False means drive_letta_turn always reports injected=False, so the
        # guard has to observe the composition itself. This is the check that proves the
        # retrieval half fired instead of the turn coasting on core blocks alone. Detect the
        # injection marker specifically — a `body != prompt` comparison would now be
        # satisfied by the as_of anchor alone and would silently stop testing retrieval.
        injected = "<mem id=" in body
        prior = c.get("prior_context") or []
        if prior:
            body = "[Recent conversation, for context:]\n" + "\n".join(prior) + "\n\n" + body
        r = drive_letta_turn(body, inject_recall=False)
        reply = (r.get("reply") or "").strip()
        # Validity is a HARD gate, not a filter: a row that fails here is a FAILURE below.
        invalid = None
        if r.get("error"):
            invalid = f"turn error: {r['error'][:120]}"
        elif len(reply) < 40:
            invalid = f"degenerate reply ({len(reply)} chars)"
        out.append(
            {
                **c,
                "letta_reply": reply,
                "letta_seconds": round(r.get("seconds", time.time() - t0), 2),
                "letta_tools": r.get("tools") or [],
                "letta_injected": injected,
                "letta_prior_turns": len(prior),
                "invalid": invalid,
            }
        )
        flag = "INVALID: " + invalid if invalid else f"ok {out[-1]['letta_seconds']}s"
        inj = "inj" if out[-1]["letta_injected"] else "NO-INJECTION"
        print(f"  [{n:2}/{len(prompts)}] {inj:12} {flag}  {c['prompt'][:60]}", flush=True)

    with open(RUNS_PATH, "w") as f:
        json.dump({"ran": datetime.now(timezone.utc).isoformat(), "rows": out}, f, indent=2)

    n_inv = sum(1 for r in out if r["invalid"])
    n_noinj = sum(1 for r in out if not r["letta_injected"])
    secs = [r["letta_seconds"] for r in out if not r["invalid"]]
    print(f"\nwrote {RUNS_PATH}")
    print(f"  valid={len(out) - n_inv}/{len(out)}  invalid={n_inv} (count as failures)")
    print(f"  recall-injected={len(out) - n_noinj}/{len(out)}  (no-injection rows ran on core blocks alone)")
    if secs:
        print(f"  letta latency: median {sorted(secs)[len(secs)//2]:.1f}s  max {max(secs):.1f}s")


def _arm_order(msg_id: int) -> tuple[str, str]:
    """Deterministic per-row A/B assignment — blind to the judge, reproducible for audit."""
    h = hashlib.md5(str(msg_id).encode()).hexdigest()
    return ("sdk", "letta") if int(h[0], 16) % 2 == 0 else ("letta", "sdk")


def cmd_pack() -> None:
    with open(RUNS_PATH) as f:
        rows = json.load(f)["rows"]

    key, parts = {}, []
    parts.append(
        "# 5.1R blind reply-diff judge pack\n\n"
        "For each item you see the real user prompt and two candidate answers, **A** and **B**, "
        "in a shuffled order. One came from Luke's production SDK path, the other from the Letta "
        "path. You do not know which is which — do not guess from formatting quirks alone, judge "
        "the substance.\n\n"
        "Rubric, per item, on three axes: **factual grounding** (are the claims right and "
        "specific?), **Luke-voice** (warm, direct, no assistant-speak), **actionability** (does it "
        "actually answer / move the thing forward?).\n\n"
        "Emit one JSON object per item into a judgments file:\n"
        '`{"msg_id": 123, "material_divergence": false, "worse_arm": null, '
        '"factually_wrong_arm": null, "note": "both land the same facts"}`\n\n'
        "- `material_divergence`: true if a reader would get a *meaningfully different* answer — "
        "different facts, a real quality gap. Formatting/length alone is NOT material.\n"
        "- `worse_arm`: `\"A\"`, `\"B\"`, or null if no material gap.\n"
        "- `factually_wrong_arm`: `\"A\"`/`\"B\"`/null — set ONLY when one arm states something "
        "false that the other got right. This is the veto axis.\n\n"
        "Do NOT open `letta_reply_diff_key.json` until every judgment is written.\n\n---\n"
    )
    for i, r in enumerate(rows, 1):
        first, second = _arm_order(r["msg_id"])
        key[str(r["msg_id"])] = {"A": first, "B": second, "invalid": r["invalid"]}
        texts = {
            "sdk": _plain(r["sdk_reply"]),
            "letta": _plain(r["letta_reply"]) or "(no answer produced)",
        }
        parts.append(
            f"\n## Item {i} — msg #{r['msg_id']} ({r['bucket']}, {r['ts'][:16]})\n\n"
            f"**Prompt:** {r['prompt']}\n\n"
            f"**Answer A:**\n\n{texts[first][:2200]}\n\n"
            f"**Answer B:**\n\n{texts[second][:2200]}\n"
        )

    with open(PACK_PATH, "w") as f:
        f.write("\n".join(parts))
    with open(KEY_PATH, "w") as f:
        json.dump(key, f, indent=2)
    print(f"wrote {PACK_PATH} ({len(rows)} items)  +  {KEY_PATH} (do not read until judged)")


def cmd_score(judgments_path: str) -> None:
    with open(RUNS_PATH) as f:
        rows = {str(r["msg_id"]): r for r in json.load(f)["rows"]}
    with open(KEY_PATH) as f:
        key = json.load(f)
    with open(judgments_path) as f:
        raw = json.load(f)
    judgments = raw["judgments"] if isinstance(raw, dict) else raw

    n = len(rows)
    ok, material, letta_wrong, lines = 0, [], [], []
    seen = set()
    for j in judgments:
        mid = str(j["msg_id"])
        seen.add(mid)
        k = key.get(mid)
        if not k:
            continue
        if rows[mid]["invalid"]:
            lines.append(f"  #{mid} FAIL (invalid letta turn: {rows[mid]['invalid']})")
            material.append(mid)
            continue
        worse = j.get("worse_arm")
        wrong = j.get("factually_wrong_arm")
        worse_arm = k.get(worse) if worse in ("A", "B") else None
        wrong_arm = k.get(wrong) if wrong in ("A", "B") else None
        if j.get("material_divergence"):
            material.append(mid)
            lines.append(f"  #{mid} DIVERGE (worse={worse_arm or '?'}) {j.get('note', '')[:80]}")
        else:
            ok += 1
            lines.append(f"  #{mid} ok")
        if wrong_arm == "letta":
            letta_wrong.append(mid)
    # Judged-nothing rows can't silently vanish from the denominator.
    for mid in rows:
        if mid not in seen:
            material.append(mid)
            lines.append(f"  #{mid} FAIL (no judgment emitted)")

    threshold = 18 if n >= 20 else int(round(n * 0.9))
    passed = ok >= threshold and not letta_wrong
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    head = (
        f"5.1R {stamp} — no-material-divergence {ok}/{n} (need >={threshold}); "
        f"letta-wrong-where-sdk-right {len(letta_wrong)} (need 0) => "
        f"{'PASS' if passed else 'FAIL'}"
    )
    print(head)
    for ln in lines:
        print(ln)
    os.makedirs(LOGS, exist_ok=True)
    with open(VERDICT_LOG, "a") as f:
        f.write(head + "\n")
    print(f"\nappended verdict to {VERDICT_LOG}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        cmd_build()
    elif cmd == "run":
        cmd_run()
    elif cmd == "pack":
        cmd_pack()
    elif cmd == "score":
        cmd_score(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
