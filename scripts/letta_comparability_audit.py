#!/usr/bin/env python3
"""Comparability as a property of the PROMPT, asked before the run instead of after it.

5.1R excludes non-comparable rows at JUDGE time: the blind judge attributes a cause, harness
causes leave the denominator, and the exclusions are capped at ``HARNESS_CAP`` because an
uncapped excuse is not a gate. That design is right, and on 2026-08-03 it produced VOID —
7 of 20 rows were not like-for-like, two over the cap, and no attribution available reaches
PASS. The gate could not return a verdict at all.

The thing nobody had checked is that **exclusion was never the only lever**. A row that is
not comparable is not comparable *before it is drawn*, and the candidate pool is 322 eligible
turn-pairs for a 20-row pack. Filtering a class out at set-build time costs nothing in sample
size and leaves the accept bar exactly where it is — unlike excusing rows after the fact,
which shrinks the denominator and (measured, see `letta_reply_diff.memory_budget`) makes the
gate HARDER at five exclusions and impossible at six.

So this script asks the same question the judge asks, from the prompt alone:

  mutable-state  the correct answer depends on the state of Luke's own running system at that
                 moment — task health, code, logs, what it had shipped, how it was behaving.
                 Replayed today the arm holds two clocks and cannot tell them apart. (The
                 class the plan reserved for Filipe on 2026-08-02, before this run.)
  write-action   the ask is to PERFORM a change. The archived SDK reply reports a commit; the
                 Letta arm's ten tools are all reads, so it answers with a plan and a blind
                 judge correctly scores the plan as weaker. (Plan note, 2026-08-02 20:15.)
  artifact-ref   the ask is about something that is not text in the message store — a URL, an
                 attachment, an email body, a draft produced in a prior turn.

WHAT THIS SCRIPT DOES NOT DO: it does not filter the pack. The pack is frozen on purpose and
removing rows from it after seeing a red number is the move that makes a gate meaningless.
`build --comparable` exists in this file as a *measurement* of what a filtered pool looks
like; wiring it into `letta_reply_diff.cmd_build` is Filipe's call, and the numbers he needs
to make it are what `agreement`, `breadth` and `branches` print.

Honest statement of method, because it decides how much the agreement number is worth: the
three CLASSES were named in writing before the 2026-08-03 run (the clock class in the plan's
step list, the write-repo class in its notes, the artifact class in `letta_exclusion_audit`'s
inclusion half, commit 10f0b39). The PATTERNS in this file were written afterwards, against
those 20 rows. So recall against that run is fitted and is not evidence; FN=0 should be read
as "it was tuned until it was", nothing more. The two numbers that carry information are the
false positives (rows the judge measured and this flags anyway) and the breadth over the
302 rows it has never seen.

Commands:
  agreement [run_id]  detector vs the blind judge's independent attribution, row by row
  breadth             how much of the eligible pool each class removes, and what remains
  branches [run_id]   what the real `accept()` returns under each adjudication of the run
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

SET_PATH = os.path.join(REPO, "logs", "letta_reply_diff_set.json")
LOG_PATH = os.path.join(REPO, "logs", "letta_comparability_audit.log")


# --- the three classes -------------------------------------------------------------------
#
# Ordered, first match wins. A prompt that is both a write-action and an artifact-ref fails
# on the mutable-state axis first if it names one, because that is the axis a replay cannot
# repair at all: the other two are capability gaps that could in principle be closed by
# attaching a tool, and the clock cannot be.

# The state of Luke's own running system. Deliberately narrow — a complaint ABOUT Luke is not
# automatically clock-dependent (#3743 "your answer quality is degrading" was attributed to
# `quality`, a memory cause, by one judging pass and to `clock` by another), so this looks for
# prompts that name a system surface or a moment-in-time behaviour, not for dissatisfaction.
MUTABLE_STATE = re.compile(
    r"(has failed \d+ times|⚠️|task '|the (cron|scheduler|task) |luke\.log|"
    r"\byour (answer|response|reply|output)s? (quality|are|is|got)|"
    r"\b(you|luke)('re| are|'ve| have)? (answering|getting|being) (a lot )?"
    r"(quicker|faster|slower|worse|better)|"
    r"\bself-improve\b|\bwon'?t you do anything|\bare you (still )?(working|running|doing)|"
    r"\bpart of the fix\b|\bthe fix (was|is)\b|\bdid you (fix|ship|deploy|commit)\b|"
    r"\b(what('| i)?s|show me what.{0,20})\s+(is\s+)?in your (content|context|memory|head)\b)",
    re.I,
)

# An ask to change code/config/infra. The read-only tool surface is deliberate (the replay
# drives live turns against a real account), so this class is a permanent property of the
# arms, not a gap waiting on a tool.
WRITE_ACTION = re.compile(
    r"\b(implement|deploy|restart|refactor|rewrite|migrate|"
    r"fix (this|it|that|the)|build (it|this|the)|"
    r"(test|do) (this|it) in a branch|don'?t change main|push (it|this)|commit (it|this))\b",
    re.I,
)

# Something that is not text in the message store.
ARTIFACT_REF = re.compile(
    r"(https?://|\.(zip|pdf|docx?|png|jpe?g|csv)\b|"
    r"\bthe (email|draft|document|attachment|file|video|recording)\b)",
    re.I,
)

_ATTACHMENT_MARKER = re.compile(r"\[(Document|Photo|Voice message|Video) saved:", re.I)

# A prompt that opens by leaning on the turn before it inherits that turn's comparability.
# Non-comparability propagates down a thread: #3883 ("hmmm, I'm not sure this is the correct
# thing") is a reply to "show me what is in your content right now", and answering it needs
# the same live state the parent needed.
_DEICTIC_OPEN = re.compile(
    r"^\s*(hmm+|hm+|ok|okay|but|and|so|no|yes|yeah|i'?m not sure|this|that|it)\b", re.I
)

CLASSES = ("mutable-state", "write-action", "artifact-ref")


def classify(prompt: str, prior_context: list[str] | None = None) -> tuple[str | None, str]:
    """Return (class, where-the-signal-was). None means comparable as far as this can tell.

    Fails toward COMPARABLE — an unrecognised prompt is drawn, judged, and (if it turns out
    not to be like-for-like) caught by the judge and the cap, which is the existing safety
    net. Pointing the soft edge the other way would let the pool quietly shrink toward
    whatever these regexes happen to like.
    """
    prior = prior_context or []
    joined = "\n".join(prior)

    if MUTABLE_STATE.search(prompt):
        return "mutable-state", "prompt"
    if WRITE_ACTION.search(prompt):
        return "write-action", "prompt"
    if ARTIFACT_REF.search(prompt):
        return "artifact-ref", "prompt"

    # An attachment somewhere in the preceding eight messages does NOT make a self-contained
    # prompt non-comparable — requiring attachments to DOMINATE the prior context is what
    # separates #3505 (7 of 8 prior turns are WhatsApp .zip drops the Letta arm sees only as
    # path strings) from #2835 and #3406, which the unqualified rule flagged and the judge
    # measured cleanly.
    n_att = sum(1 for t in prior if _ATTACHMENT_MARKER.search(t))
    if prior and n_att * 2 >= len(prior):
        return "artifact-ref", f"prior({n_att}/{len(prior)})"

    if _DEICTIC_OPEN.match(prompt) and MUTABLE_STATE.search(joined):
        return "mutable-state", "prior"

    return None, ""


# --- plumbing ----------------------------------------------------------------------------


def _rd():
    spec = importlib.util.spec_from_file_location(
        "rd", os.path.join(REPO, "scripts", "letta_reply_diff.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _judgments_path(run_id: str | None) -> str:
    logs = os.path.join(REPO, "logs")
    if run_id:
        for name in (f"51r_judgments_{run_id}.json", f"letta_reply_diff_judgments_{run_id}.json"):
            p = os.path.join(logs, name)
            if os.path.exists(p):
                return p
        raise SystemExit(f"FAIL: no judgments file for run {run_id} in {logs}")
    cands = [
        os.path.join(logs, f)
        for f in os.listdir(logs)
        if re.match(r"(51r_judgments_|letta_reply_diff_judgments_)[0-9a-f]{12}\.json$", f)
    ]
    if not cands:
        raise SystemExit(f"FAIL: no judgments files in {logs}. Score a run first.")
    return max(cands, key=os.path.getmtime)


def _load_judgments(run_id: str | None) -> tuple[str, dict[int, dict]]:
    path = _judgments_path(run_id)
    blob = json.load(open(path))
    rows = blob["judgments"] if isinstance(blob, dict) else blob
    rid = blob.get("run_id", "?") if isinstance(blob, dict) else "?"
    return rid, {r["msg_id"]: r for r in rows}


def _eligible_pool(rd) -> list[tuple[int, str, str]]:
    """(msg_id, prompt, ts) for every turn the 5.1R filter would let into the candidate pool."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=rd.LOOKBACK_DAYS)).isoformat()
    con = sqlite3.connect(f"file:{rd._db_path()}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, sender, content, ts FROM messages WHERE ts >= ? ORDER BY id ASC", (cutoff,)
    ).fetchall()
    con.close()
    out = []
    for mid, sender, content, ts in rows:
        if sender.startswith("Luke"):
            continue
        prompt = (content or "").strip()
        if len(prompt) < rd.MIN_PROMPT_CHARS or rd._excluded(prompt):
            continue
        out.append((mid, prompt, ts))
    return out


# --- commands ----------------------------------------------------------------------------


def cmd_agreement(run_id: str | None) -> int:
    """Detector vs the blind judge, row by row, on a pack that has already been scored."""
    rd = _rd()
    rid, judged = _load_judgments(run_id)
    prompts = json.load(open(SET_PATH))["prompts"]

    print(f"comparability detector vs blind judge — run {rid}, n={len(prompts)}\n")
    tp = fn = fp = tn = 0
    for p in prompts:
        mid = p["msg_id"]
        cause = (judged.get(mid) or {}).get("divergence_cause")
        judged_harness = cause in rd.HARNESS_CAUSES
        klass, where = classify(p["prompt"], p.get("prior_context"))
        if judged_harness and klass:
            tp += 1
            mark = "AGREE  "
        elif judged_harness:
            fn += 1
            mark = "MISSED "
        elif klass:
            fp += 1
            mark = "EXTRA  "
        else:
            tn += 1
            mark = ""
        if mark:
            head = " ".join(p["prompt"].split())[:58]
            print(f"  {mark} #{mid}  judge={str(cause):18} detector={str(klass):14}"
                  f"[{where}]  | {head}")

    print(f"\n  agree={tp}  missed={fn}  extra={fp}  both-clean={tn}")
    print(
        "\n  Read this the right way. `missed` is FITTED — the patterns were written against\n"
        "  these rows, so driving it to 0 was always available and means nothing on its own.\n"
        "  `extra` is the number that costs something, and `breadth` is the real check."
    )
    return 0


def cmd_breadth() -> int:
    """What each class removes from the pool the pack is drawn from."""
    rd = _rd()
    pool = _eligible_pool(rd)
    counts = {c: 0 for c in CLASSES}
    comparable = 0
    for _mid, prompt, _ts in pool:
        klass, _ = classify(prompt)
        if klass:
            counts[klass] += 1
        else:
            comparable += 1

    n = len(pool)
    print(f"eligible candidate pool (last {rd.LOOKBACK_DAYS}d, after the existing 5.1R "
          f"filters): {n}\n")
    for c in CLASSES:
        print(f"  {c:16} {counts[c]:4}  ({100 * counts[c] / n:.0f}%)")
    print(f"  {'COMPARABLE':16} {comparable:4}  ({100 * comparable / n:.0f}%)")
    print(f"\n  a 20-row pack drawn from the comparable remainder has {comparable} candidates "
          f"to choose from\n  ({comparable / 20:.0f}x the pack size). Filtering this class at "
          f"BUILD time is not sample-constrained —\n  which is the whole reason it is a better "
          f"lever than excusing rows after the fact.")
    # Breadth is only meaningful next to how much of the pool a runaway filter WOULD take.
    if comparable < 20:
        print("\n  FAIL: fewer than 20 comparable candidates. The filter is too broad to use.")
        return 1
    return 0


def cmd_branches(run_id: str | None) -> int:
    """What the gate returns under each way the open adjudications could go."""
    rd = _rd()
    rid, judged = _load_judgments(run_id)
    n = len(judged)

    excluded, memory_div, clean = [], [], []
    for mid, j in judged.items():
        if not j.get("material_divergence"):
            clean.append(mid)
            continue
        cause = rd._attributed_cause(j)
        (excluded if cause in rd.HARNESS_CAUSES else memory_div).append((mid, cause))

    by_cause: dict[str, list[int]] = {}
    for mid, cause in excluded:
        by_cause.setdefault(cause, []).append(mid)

    print(f"run {rid} — n={n}, clean={len(clean)}, memory divergences={len(memory_div)}, "
          f"harness-excluded={len(excluded)}")
    for cause, mids in sorted(by_cause.items()):
        print(f"    {cause:18} {', '.join('#%d' % m for m in sorted(mids))}")
    print()

    ok = len(clean)
    branches: list[tuple[str, int]] = [("as judged", len(excluded))]
    for cause, mids in sorted(by_cause.items()):
        branches.append((f"'{cause}' ruled IN (charged to memory)", len(excluded) - len(mids)))
    branches.append(("all harness rows ruled IN", 0))

    print("  BRANCH — re-attributing this run's rows (the pack stays frozen):")
    reachable = False
    for name, e in branches:
        verdict, measured, thr = rd.accept(n, ok, e, 0)
        reachable = reachable or verdict == "PASS"
        print(f"    {name:44} E={e}  ok={ok}/{measured} need>={thr}  => {verdict}")
    print(f"\n    PASS reachable by re-attribution alone: {'YES' if reachable else 'NO'}"
          f" — which is what keeps an exclusion call made by\n    the same agent that judged "
          f"the rows from being a route to a green.\n")

    # The branch nobody had computed: filter the class at build time instead, and the
    # denominator stays at n. This is the honest one, because it is the only branch where the
    # gate can return a verdict — and on this run's own divergence rate, that verdict is a
    # FAIL, not a PASS. Making a gate measurable is not the same as making it pass.
    print("  BRANCH — filter the class at BUILD time and re-run on a fresh comparable pack:")
    rate = len(memory_div) / max(1, n - len(excluded))
    projected = round(rate * n)
    for m in sorted({2, 3, projected, len(memory_div)}):
        if m > n:
            continue
        verdict, measured, thr = rd.accept(n, n - m, 0, 0)
        tag = "  <- projected at this run's memory-divergence rate" if m == projected else ""
        print(f"    {m} memory divergences of {n} comparable rows"
              f"{'':>{max(0, 12 - len(str(m)))}} ok={n - m}/{measured} need>={thr}"
              f"  => {verdict}{tag}")
    print(f"\n    memory-divergence rate on the measured rows: {len(memory_div)}/"
          f"{n - len(excluded)} = {100 * rate:.0f}%  =>  ~{projected} of {n}.")
    print(f"    budget at E=0 is {rd.memory_budget(n, 0)}. So a comparable pack is expected to "
          f"FAIL, not pass —\n    this lever buys a VERDICT, not a green. Worth saying plainly "
          f"before anyone rules on it.")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "agreement"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "agreement":
        return cmd_agreement(arg)
    if cmd == "breadth":
        return cmd_breadth()
    if cmd == "branches":
        return cmd_branches(arg)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
