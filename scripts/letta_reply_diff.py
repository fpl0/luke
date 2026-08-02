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

Accept: >=90% no-material-divergence over the MEASURED rows, AND zero cases where the letta
answer is factually wrong where the sdk answer was right.

"Measured" is the change made 2026-08-02, after two runs (7/20, then 11/20) whose failures were
mostly not about memory. The judge now attributes a CAUSE to every divergence, still blind, and
rows caused by the comparison itself — context_asymmetry (one arm was handed material the other
never saw), clock (the arms answer from different moments), tooling (the ask needed a capability,
not a memory) — are excluded from the denominator instead of being charged to Letta. The gate was
supposed to answer "does Letta remember as well", and was in fact answering "were the two arms
given the same situation", which is a question about the harness.

That exclusion is the only soft edge in the gate, so it is fenced on three sides: attribution
fails CLOSED (no label, unknown label, or no evidence quoted => counted as a memory failure);
exclusions are capped (>5 of 20, or fewer than 14 comparable rows left, reports VOID rather than
PASS); and the veto axis still applies at full strength to every non-excluded row. The intended
failure mode is a harness that gets fixed, never a gate that gets easier.

Usage (the 23:00 cron drives these in order):
    python3 scripts/letta_reply_diff.py build            # select the 20-prompt replay set
    python3 scripts/letta_reply_diff.py run              # live Letta arm, 20 turns
    #   resumable: each turn is journalled, so re-running `run` continues where a killed
    #   session stopped. --fresh discards the journal; --limit N replays only N turns
    #   (leaves the run incomplete on purpose, so it can never be packed as a gate).
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
# Turn-by-turn journal. ``run`` appends each completed row here immediately, so a session
# that dies mid-replay (timeout, bridge hiccup) loses one turn instead of twenty — the
# failure that already cost this goal a session on 2026-08-02. Deleted on a full run.
PARTIAL_PATH = os.path.join(LOGS, "letta_reply_diff_partial.jsonl")

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


def _load_partial(set_built: str, valid_ids: set[int]) -> list[dict]:
    """Rows already completed for THIS set, from the journal.

    Resume is keyed on the set's ``built`` stamp, not just msg_id. A row replayed against a
    previous build of the pack was produced under a different harness (different as_of
    anchoring, different conversation depth, different tool surface), and silently mixing
    those into one pack is precisely the harness-artefact class that made the Aug-01 numbers
    meaningless. Stale rows are dropped loudly, never reused.
    """
    if not os.path.exists(PARTIAL_PATH):
        return []
    kept, stale = [], 0
    with open(PARTIAL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stale += 1  # torn final line from a hard kill — discard, replay that turn
                continue
            if row.get("set_built") == set_built and row.get("msg_id") in valid_ids:
                kept.append(row)
            else:
                stale += 1
    if stale:
        print(f"  journal: dropped {stale} row(s) from a previous set build / torn write")
    return kept


def cmd_run(limit: int | None = None, fresh: bool = False) -> None:
    from luke.letta_agent import compose_letta_turn_input, drive_letta_turn

    with open(SET_PATH) as f:
        spec = json.load(f)
    prompts, set_built = spec["prompts"], spec["built"]

    if fresh and os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)
        print("  journal: cleared (--fresh)")
    done = {r["msg_id"]: r for r in _load_partial(set_built, {p["msg_id"] for p in prompts})}
    if done:
        # Age matters, and is reported rather than silently tolerated. Resuming minutes after
        # a crash is what the journal is for. Resuming many hours later splits one gate across
        # two different memory states (core blocks re-pack at 05:10, live write-through lands
        # new memories all day), so the run stops being a single measurement. Judgement call
        # for the operator — but it can only be made if the number is on screen.
        age_h = (time.time() - os.path.getmtime(PARTIAL_PATH)) / 3600
        note = "  <-- >4h old: consider --fresh, this splits the run across memory states" if age_h > 4 else ""
        print(f"  journal: resuming with {len(done)}/{len(prompts)} turn(s) already replayed"
              f" (last write {age_h:.1f}h ago){note}")

    todo = [c for c in prompts if c["msg_id"] not in done]
    if limit is not None:
        todo = todo[:limit]
        print(f"  --limit {limit}: replaying {len(todo)} of {len(prompts) - len(done)} remaining")

    out = []
    for n, c in enumerate(todo, 1):
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
        row = {
            **c,
            "letta_reply": reply,
            "letta_seconds": round(r.get("seconds", time.time() - t0), 2),
            "letta_tools": r.get("tools") or [],
            "letta_injected": injected,
            "letta_prior_turns": len(prior),
            "invalid": invalid,
        }
        out.append(row)
        # Journal BEFORE the next turn starts: everything up to here survives a kill.
        with open(PARTIAL_PATH, "a") as f:
            f.write(json.dumps({**row, "set_built": set_built}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        flag = "INVALID: " + invalid if invalid else f"ok {row['letta_seconds']}s"
        inj = "inj" if row["letta_injected"] else "NO-INJECTION"
        print(f"  [{n:2}/{len(todo)}] {inj:12} {flag}  {c['prompt'][:60]}", flush=True)

    # Order by the set, not by completion, so a resumed run packs identically to a clean one.
    have = {**done, **{r["msg_id"]: r for r in out}}
    rows = [{k: v for k, v in have[c["msg_id"]].items() if k != "set_built"}
            for c in prompts if c["msg_id"] in have]

    n_inv = sum(1 for r in rows if r["invalid"])
    n_noinj = sum(1 for r in rows if not r["letta_injected"])
    secs = [r["letta_seconds"] for r in rows if not r["invalid"]]
    if secs:
        print(f"\n  letta latency: median {sorted(secs)[len(secs)//2]:.1f}s  max {max(secs):.1f}s"
              f"  mean {sum(secs)/len(secs):.1f}s")

    if len(rows) < len(prompts):
        # Deliberately do NOT write RUNS_PATH. A partial replay must never leave behind an
        # artefact that `pack` would happily turn into a judgeable pack — a gate scored on
        # 12 of 20 rows reads exactly like a gate scored on 20.
        print(f"\nINCOMPLETE — {len(rows)}/{len(prompts)} replayed, {RUNS_PATH} NOT written.")
        print(f"  progress is safe in {PARTIAL_PATH}; re-run `run` to continue where it stopped.")
        return

    with open(RUNS_PATH, "w") as f:
        # set_built is the provenance link back to the harness these replies were produced
        # under. Without it the runs file is undatable: `pack` could only count rows, and 20
        # rows replayed in August under a two-tool, unanchored, 2-message-buffer agent look
        # exactly like 20 rows replayed today. That is not hypothetical — it was the state on
        # disk when this was written.
        json.dump({"ran": datetime.now(timezone.utc).isoformat(),
                   "set_built": set_built, "run_id": _run_id(set_built, rows),
                   "rows": rows}, f, indent=2)
    os.remove(PARTIAL_PATH)
    print(f"\nwrote {RUNS_PATH}  run={_run_id(set_built, rows)}")
    print(f"  valid={len(rows) - n_inv}/{len(rows)}  invalid={n_inv} (count as failures)")
    print(f"  recall-injected={len(rows) - n_noinj}/{len(rows)}  (no-injection rows ran on core blocks alone)")


def _run_id(set_built: str, rows: list[dict]) -> str:
    """Identity of one RUN, not of the set.

    Every provenance guard in this file keys on ``set_built`` — and the plan then froze the
    set, on purpose, so that a fix shipped against a red gate can be re-measured against the
    same 20 prompts. The moment it froze, ``set_built`` stopped distinguishing anything: the
    13:07Z replies, the 13:31Z replies and tonight's replies all carry the identical stamp, so
    `pack` and `score` wave through a runs file from an arbitrary earlier measurement while
    reporting it as current. That is not hypothetical either — the 13:07Z runs file was sitting
    on disk passing both guards when this was written, and the only thing standing between it
    and a judged pack was a sentence of prose in a cron prompt.

    Hash the replies instead. Same replies -> same id, which is the right equivalence class:
    scoring judgments against byte-identical answers is harmless. Any reply moves -> new id.
    """
    h = hashlib.sha256(set_built.encode())
    for r in sorted(rows, key=lambda r: r["msg_id"]):
        h.update(str(r["msg_id"]).encode())
        for arm in ("sdk_reply", "letta_reply"):
            h.update(json.dumps(r.get(arm), sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def _scored_run_ids() -> set[str]:
    """Run ids that have already produced a verdict, read back out of the verdict log."""
    if not os.path.exists(VERDICT_LOG):
        return set()
    with open(VERDICT_LOG) as f:
        return set(re.findall(r"\brun=([0-9a-f]{12})\b", f.read()))


def _arm_order(msg_id: int) -> tuple[str, str]:
    """Deterministic per-row A/B assignment — blind to the judge, reproducible for audit."""
    h = hashlib.md5(str(msg_id).encode()).hexdigest()
    return ("sdk", "letta") if int(h[0], 16) % 2 == 0 else ("letta", "sdk")


def cmd_pack() -> None:
    if not os.path.exists(RUNS_PATH):
        # Normal state, not an error condition: `run` deletes nothing but writes RUNS_PATH
        # only on a complete replay, and stale files from an older lineage are quarantined
        # rather than left lying around. Say so instead of raising a traceback.
        print(f"No runs file at {RUNS_PATH} — run `run` to completion first.")
        sys.exit(1)
    with open(RUNS_PATH) as f:
        runs = json.load(f)
    rows = runs["rows"]

    with open(SET_PATH) as f:
        spec = json.load(f)
    n_set, set_built = len(spec["prompts"]), spec["built"]

    # Second half of the partial-run guard. `run` refuses to write RUNS_PATH short, but an
    # older or hand-edited runs file could still be short, and the accept clause (>=18/20) is
    # only meaningful against a full 20. Refuse rather than silently rescale the denominator.
    if len(rows) != n_set:
        print(f"REFUSING: runs has {len(rows)} row(s), set has {n_set}. "
              f"The accept clause is defined over the whole set — re-run `run` to completion.")
        sys.exit(1)

    # Third guard, and the one the count check cannot cover: rows from the RIGHT set size but
    # the WRONG harness. The journal already refuses to resume such rows mid-run; a completed
    # runs file from an earlier build was still packable, because 20 == 20. The failure is
    # silent and total — the pack, the key and the verdict all look current, and the gate
    # reports on the very harness it was rebuilt to replace.
    runs_built = runs.get("set_built")
    if runs_built != set_built:
        print(f"REFUSING: runs were replayed against set build {runs_built or '(unstamped)'}, "
              f"the set on disk is {set_built}. These replies predate the current harness — "
              f"re-run `run` (add --fresh if a journal from the old build is present).")
        sys.exit(1)

    # Fourth guard, and the only one that survives the set being frozen. The three above all
    # ask "which HARNESS produced these replies?" and answer it from `set_built`, which is now
    # constant by design — so none of them can tell one measurement from the next against the
    # same set. This one asks "have these exact replies already been judged?" and reads the
    # answer out of the verdict log. Re-judging a scored run is how a stale number gets
    # reported as current, and until now nothing but a sentence in a cron prompt prevented it.
    run_id = runs.get("run_id")
    if run_id and run_id in _scored_run_ids() and "--rejudge" not in sys.argv:
        print(f"REFUSING: run {run_id} has already been scored (see {VERDICT_LOG}). Packing it "
              f"again re-judges a measurement that is already on the record — re-run `run "
              f"--fresh` for a current number, or pass --rejudge if you deliberately mean to "
              f"re-judge these same replies.")
        sys.exit(1)
    if "--rejudge" in sys.argv:
        print(f"!! --rejudge: re-packing already-scored run {run_id or '(unstamped)'} — the "
              f"verdict this produces is NOT a new measurement.")

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
        '"factually_wrong_arm": null, "divergence_cause": null, "cause_evidence": null, '
        '"note": "both land the same facts"}`\n\n'
        "- `material_divergence`: true if a reader would get a *meaningfully different* answer — "
        "different facts, a real quality gap. Formatting/length alone is NOT material.\n"
        "- `worse_arm`: `\"A\"`, `\"B\"`, or null if no material gap.\n"
        "- `factually_wrong_arm`: `\"A\"`/`\"B\"`/null — set ONLY when one arm states something "
        "false that the other got right. This is the veto axis.\n\n"
        "When `material_divergence` is true you MUST also set `divergence_cause` and quote your "
        "evidence in `cause_evidence`. The cause is judged BLIND, from the text alone — you are "
        "not being asked which backend is which, only what KIND of gap this is:\n\n"
        "- `\"recall\"` — the gap is about remembered facts. One answer knows something about "
        "Filipe's history, projects or past conversations that the other has lost, garbled or "
        "invented. **This is the one the gate is actually measuring.**\n"
        "- `\"self_knowledge\"` — the question is about Luke himself (his own code, data, costs, "
        "capabilities) and one answer states something confidently false about it. Also counts "
        "against the gate: the answer was available and was not looked up.\n"
        "- `\"context_asymmetry\"` — one answer visibly works from material that never appears in "
        "the other at all: a pasted email, an attached document, the immediately-preceding turn. "
        "Not a memory gap — the two answers were given different inputs.\n"
        "- `\"clock\"` — the answers are consistent with each other but anchored to different "
        "moments in time (different 'today', different countdown, one knows about an event the "
        "other is too early to know about).\n"
        "- `\"tooling\"` — the ask needs a capability, not a memory: send this, schedule that, "
        "open this file. One answer performs it, the other cannot.\n"
        "- `\"quality\"` — same facts, genuinely worse answer: hedged, evasive, assistant-speak, "
        "doesn't answer the question.\n\n"
        "`cause_evidence` must be a short QUOTE or concrete pointer from the answers that "
        "justifies the cause — not a restatement of it. If you cannot point at evidence, the "
        "cause is `\"recall\"`.\n\n"
        "Do NOT open `letta_reply_diff_key.json` until every judgment is written.\n\n"
        f"Write your judgments as `{{\"run_id\": \"{run_id or ''}\", \"judgments\": [...]}}`. The "
        "`run_id` identifies which replay you judged; `score` refuses judgments belonging to a "
        "different one. It says nothing about which arm is which.\n\n---\n"
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
        # The key carries the same stamp so `score` can prove the judgments it is unblinding
        # belong to the runs on disk — a re-run that is judged against the PREVIOUS pack's key
        # is the same staleness bug one step further down the pipeline.
        json.dump({"set_built": set_built, "run_id": run_id, "arms": key}, f, indent=2)
    print(f"wrote {PACK_PATH} ({len(rows)} items)  +  {KEY_PATH} (do not read until judged)")
    print(f"  run={run_id or '(unstamped)'} — put this in your judgments file as "
          f"\"run_id\": \"{run_id or ''}\"")


# Causes that describe a defect in the COMPARISON rather than in memory. A row attributed to
# one of these was never a like-for-like test, so it is excluded from the accept clause — and
# because that exclusion is the only soft edge in the gate, it is capped (HARNESS_CAP) and the
# attribution has to be earned (_attributed_cause fails closed).
HARNESS_CAUSES = ("context_asymmetry", "clock", "tooling")
MEMORY_CAUSES = ("recall", "self_knowledge", "quality")
# Above this many exclusions the run is VOID, not PASS: a harness that produces a quarter of its
# rows as non-comparisons is not measuring memory, whatever the surviving rows say.
HARNESS_CAP = 5
# ...and never score a gate on a rump. Below this many genuinely-comparable rows there is no
# result, only a small sample that happened to agree.
MIN_MEASURED = 14


def _attributed_cause(j: dict) -> str:
    """Cause for a divergence, defaulting to the answer that costs Letta the gate.

    Every soft edge here is pointed the same way: an unlabelled row, an unknown label, or an
    excuse with no evidence quoted behind it all resolve to `recall`. Excluding a row has to be
    argued for; counting it against Letta is what happens by default.
    """
    cause = (j.get("divergence_cause") or "").strip().lower()
    if cause not in HARNESS_CAUSES + MEMORY_CAUSES:
        return "recall"
    if cause in HARNESS_CAUSES and not (j.get("cause_evidence") or "").strip():
        return "recall"
    return cause


def cmd_score(judgments_path: str) -> None:
    with open(RUNS_PATH) as f:
        runs = json.load(f)
    rows = {str(r["msg_id"]): r for r in runs["rows"]}
    with open(KEY_PATH) as f:
        keyfile = json.load(f)
    # A flat key predates the provenance stamp, so by construction it was written by an older
    # pack — refuse rather than fall back to reading it, which is the convenient direction.
    if not isinstance(keyfile, dict) or "arms" not in keyfile:
        print("REFUSING: key file has no provenance stamp — it was written by an older `pack`. "
              "Re-run `pack` against the current runs before scoring.")
        sys.exit(1)
    if keyfile.get("set_built") != runs.get("set_built"):
        print(f"REFUSING: key was packed from set build {keyfile.get('set_built')}, runs are "
              f"from {runs.get('set_built')}. The judgments belong to a different pack — "
              f"re-run `pack` and re-judge.")
        sys.exit(1)
    # set_built cannot tell two runs of the same frozen set apart; run_id can.
    if keyfile.get("run_id") != runs.get("run_id"):
        print(f"REFUSING: key was packed from run {keyfile.get('run_id') or '(unstamped)'}, "
              f"runs on disk are {runs.get('run_id') or '(unstamped)'}. The A/B key belongs to "
              f"a different replay — re-run `pack`.")
        sys.exit(1)
    key = keyfile["arms"]
    with open(judgments_path) as f:
        raw = json.load(f)
    judgments = raw["judgments"] if isinstance(raw, dict) else raw
    # The last unguarded hop. The set is frozen, so a judgments file from an earlier replay has
    # the same 20 msg_ids as this one: every id resolves in the key, the coverage backstop below
    # is satisfied, and a previous measurement is reported as the current one with nothing in
    # the output looking wrong. Prefer the explicit stamp; fall back to the file clock, which
    # needs no cooperation from whoever wrote the judgments and cannot false-positive on a file
    # written after the pack it was judged from.
    stamped = raw.get("run_id") if isinstance(raw, dict) else None
    if stamped and runs.get("run_id") and stamped != runs["run_id"]:
        print(f"REFUSING: judgments are stamped run {stamped}, runs on disk are "
              f"{runs['run_id']}. These judgments were written about a different replay.")
        sys.exit(1)
    if not stamped and os.path.exists(KEY_PATH) and (
            os.path.getmtime(judgments_path) < os.path.getmtime(KEY_PATH)):
        print(f"REFUSING: {judgments_path} is older than {KEY_PATH} — it was written before the "
              f"pack it would be scored against, so it judges an earlier replay. Re-judge the "
              f"current pack (and stamp it with the run_id `pack` printed).")
        sys.exit(1)

    n = len(rows)
    ok, memory_diverge, letta_wrong, lines = 0, [], [], []
    harness = {c: [] for c in HARNESS_CAUSES}
    seen = set()
    for j in judgments:
        mid = str(j["msg_id"])
        seen.add(mid)
        k = key.get(mid)
        if not k:
            continue
        if rows[mid]["invalid"]:
            lines.append(f"  #{mid} FAIL (invalid letta turn: {rows[mid]['invalid']})")
            memory_diverge.append(mid)
            continue
        worse = j.get("worse_arm")
        wrong = j.get("factually_wrong_arm")
        worse_arm = k.get(worse) if worse in ("A", "B") else None
        wrong_arm = k.get(wrong) if wrong in ("A", "B") else None
        cause = _attributed_cause(j)
        if j.get("material_divergence"):
            if cause in HARNESS_CAUSES:
                # The two arms were not asked the same question in the same conditions. That is a
                # defect in the harness, not evidence about memory — it is excluded from the
                # accept clause and surfaced by name below, where it is capped.
                harness[cause].append(mid)
                lines.append(
                    f"  #{mid} EXCLUDED[{cause}] (worse={worse_arm or '?'}) "
                    f"{(j.get('cause_evidence') or '')[:70]}"
                )
            else:
                memory_diverge.append(mid)
                lines.append(
                    f"  #{mid} DIVERGE[{cause}] (worse={worse_arm or '?'}) "
                    f"{j.get('note', '')[:70]}"
                )
        else:
            ok += 1
            lines.append(f"  #{mid} ok")
        # The veto is about memory, so an arm that was never given the material cannot trip it.
        if wrong_arm == "letta" and cause not in HARNESS_CAUSES:
            letta_wrong.append(mid)
    # Judged-nothing rows can't silently vanish from the denominator.
    for mid in rows:
        if mid not in seen:
            memory_diverge.append(mid)
            lines.append(f"  #{mid} FAIL (no judgment emitted)")

    excluded = sorted(m for ms in harness.values() for m in ms)
    measured = n - len(excluded)
    # Excluding harness artefacts is only honest while there are few of them. Past the cap the
    # run stops being a measurement of anything: too much of the set was two different questions.
    # It reports VOID rather than PASS — excuses can never carry this gate, they can only kill it
    # and force the harness to be fixed.
    void = len(excluded) > HARNESS_CAP or measured < MIN_MEASURED
    threshold = max(1, int(round(measured * 0.9)))
    passed = (not void) and ok >= threshold and not letta_wrong
    verdict = "VOID" if void else ("PASS" if passed else "FAIL")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    head = (
        f"5.1R {stamp} run={runs.get('run_id') or 'unstamped'} — "
        f"no-material-divergence {ok}/{measured} measured "
        f"(need >={threshold}); letta-wrong-where-sdk-right {len(letta_wrong)} (need 0); "
        f"harness-excluded {len(excluded)}/{n} (cap {HARNESS_CAP}) => {verdict}"
    )
    print(head)
    for ln in lines:
        print(ln)
    if excluded:
        detail = "; ".join(f"{c}={len(m)}" for c, m in sorted(harness.items()) if m)
        note = f"5.1R   excluded as harness artefact: {detail}"
        print(note)
        if void:
            print("5.1R   VOID — too much of the set was not a like-for-like comparison. "
                  "Fix the harness and re-run; this is not a memory result either way.")
    os.makedirs(LOGS, exist_ok=True)
    with open(VERDICT_LOG, "a") as f:
        f.write(head + "\n")
        if excluded:
            f.write(note + "\n")
    print(f"\nappended verdict to {VERDICT_LOG}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        cmd_build()
    elif cmd == "run":
        # --limit is for cheap harness checks (measure per-turn cost without burning 20
        # OAuth turns); it always leaves the run INCOMPLETE, so it cannot produce a pack.
        argv = sys.argv[2:]
        lim = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
        cmd_run(limit=lim, fresh="--fresh" in argv)
    elif cmd == "pack":
        # --rejudge is read inside cmd_pack; it is an escape hatch, not a routine flag.
        cmd_pack()
    elif cmd == "score":
        cmd_score(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
