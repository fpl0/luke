#!/usr/bin/env python3
"""Did the Letta arm ever HAVE the material that lost it the row?

The blind judge attributes a `divergence_cause` from the two replies alone. It cannot
see either arm's input, so it cannot tell these two apart:

  * **utilization** — the winning fact was sitting in the Letta arm's own composed input
    and the arm did not use it. That is the substrate's failure and belongs in the
    denominator.
  * **supply** — the winning fact was in neither the injected conversation window nor the
    recall injection. The arm was never given it, and no amount of memory would have
    produced it. That is the comparison's asymmetry, not Letta's.

Both get labelled `recall` today, which points the next fix at the wrong subsystem: one
is answered by retrieval work, the other only by changing what the replay supplies.

This script answers the question mechanically, from artefacts already on disk plus a
recomputed recall injection at each prompt's own ``as_of``. It reads only. It does NOT
re-attribute anything, does not touch the pack, the key, or the accept bar — a row's
cause of record stays whatever the blind judge wrote. Output is evidence for the
adjudication, not a change to it.

Direction of the fences, stated up front because both directions are self-serving here:

  * Ambiguity resolves to **PRESENT** (charged to the substrate), matching the existing
    fail-closed convention in ``letta_reply_diff.py`` where unevidenced attribution
    scores as ``recall``. A row with no extractable distinctive token is reported
    UNDECIDABLE and counted as PRESENT.
  * Only rows the **Letta arm lost** are audited. The SDK arm's input is an archived
    live thread that no longer exists as an artefact, so the same check cannot be run
    against it and pretending otherwise would manufacture symmetry.

Usage:
    python3 scripts/letta_input_availability_audit.py \
        [--runs logs/letta_reply_diff_runs.json] \
        [--key logs/letta_reply_diff_key.json] \
        --judgments logs/51r_judgments_<run_id>.json

Exit codes:
    0  every audited row is PRESENT — the divergences are the substrate's
    2  at least one row is ABSENT — that row's material was never supplied
    1  usage / provenance error (mismatched run_id, missing file)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Capitalised words and bare numbers carry almost all of a reply's distinctive content;
# these are the ones that say nothing about *which* reply you are reading.
_STOP = {
    "the", "this", "that", "there", "here", "then", "than", "they", "them", "their",
    "what", "when", "where", "which", "while", "with", "without", "your", "yours",
    "you", "and", "but", "not", "both", "one", "two", "three", "also", "yes", "for",
    "from", "into", "its", "his", "her", "she", "him", "our", "are", "was", "were",
    "has", "have", "had", "can", "could", "would", "should", "will", "may", "might",
    "luke", "filipe", "lima", "hi", "hey", "subject", "thanks", "thank", "best",
    "honest", "honestly", "something", "someone", "anything", "everything", "nothing",
    "want", "need", "just", "only", "still", "even", "actually", "really", "fine",
    "good", "great", "sure", "okay", "send", "sent", "read", "reply", "email",
}

# A capital letter after sentence-terminal punctuation, a bullet or a line start says
# nothing about content — "Fair", "Scrap", "Done" are openers, not facts. Requiring a
# preceding word on the same line keeps proper nouns and drops sentence-initial capitals,
# which is what separated the load-bearing tokens from the noise on the 4b29d3843358 run.
_WORD = re.compile(r"(?<=[a-z,;)\]] )([A-Z][A-Za-z'’]{2,})")
_NUM = re.compile(r"\b\d[\d,.]*(?:[KkMm%]|\b)")
_QUOTED = re.compile(r"[\"“”']([^\"“”'\n]{3,40})[\"“”']")
# Pronoun contractions survive the not-sentence-initial rule ("and I'm", "so That's") and
# carry no content.
_CONTRACTION = re.compile(r"^(I|You|He|She|We|They|That|This|It|There|Here)('|’)")


def distinctive_tokens(winner: str, loser: str, prompt: str) -> list[str]:
    """Tokens that carry the winning reply's specificity and are absent from the loser's.

    A deliberately narrow screen: mid-sentence capitalised words (proper nouns), numbers,
    and short quoted fragments. Tokens already in the prompt are dropped — both arms had
    those, so they cannot distinguish the replies. This is evidence to read row by row,
    not a verdict; the per-row lists are printed for exactly that reason.
    """
    cand: list[str] = []
    cand += _WORD.findall(winner)
    cand += [m for m in _NUM.findall(winner) if len(m.strip()) >= 2]
    cand += [m.strip() for m in _QUOTED.findall(winner)]
    cand = [c for c in cand if not _CONTRACTION.match(c)]

    loser_l, prompt_l = loser.lower(), prompt.lower()
    out, seen = [], set()
    for t in cand:
        tl = t.lower().strip(".,;:!?")
        if len(tl) < 3 or tl in _STOP or tl in seen:
            continue
        if tl in loser_l or tl in prompt_l:
            continue
        seen.add(tl)
        out.append(t)
    return out


def letta_input_text(row: dict, *, recompute_recall: bool = True) -> tuple[str, bool]:
    """Everything the Letta arm was given for this turn: prompt + window + recall block.

    Returns ``(text, recall_ok)``. ``recall_ok`` is False when the injection could not be
    recomputed — the caller then has an under-estimate of what was available, so every
    row it touches must fail toward PRESENT rather than ABSENT.
    """
    parts = [row.get("prompt", "")]
    for p in row.get("prior_context") or []:
        parts.append(p if isinstance(p, str) else json.dumps(p))

    recall_ok = True
    if recompute_recall and row.get("letta_injected"):
        try:
            from luke.letta_agent import build_recall_injection

            inj = build_recall_injection(row["prompt"], as_of=row.get("ts"))
            if inj:
                parts.append(inj)
            else:
                recall_ok = False
        except Exception:
            recall_ok = False
    elif row.get("letta_injected"):
        recall_ok = False

    return "\n".join(parts), recall_ok


def audit(runs_path: Path, key_path: Path, judgments_path: Path) -> int:
    runs = json.loads(runs_path.read_text())
    key = json.loads(key_path.read_text())
    judg = json.loads(judgments_path.read_text())

    run_id = runs.get("run_id")
    for name, obj in (("key", key), ("judgments", judg)):
        if obj.get("run_id") != run_id:
            print(
                f"PROVENANCE FAIL: {name} run_id={obj.get('run_id')!r} != runs run_id={run_id!r}",
                file=sys.stderr,
            )
            return 1

    rows = {r["msg_id"]: r for r in runs["rows"]}
    arms = key["arms"]

    print(f"input-availability audit — run {run_id}")
    print(f"  runs: {runs_path}\n  judgments: {judgments_path}\n")

    present, absent, undecidable, skipped = [], [], [], []

    for j in judg["judgments"]:
        mid = j["msg_id"]
        if not j.get("material_divergence") or not j.get("worse_arm"):
            continue
        arm_map = arms.get(str(mid), {})
        if arm_map.get(j["worse_arm"]) != "letta":
            skipped.append(mid)  # the SDK arm lost; its input is not an artefact
            continue

        row = rows[mid]
        winner_arm = "A" if j["worse_arm"] == "B" else "B"
        winner = row["sdk_reply"] if arm_map.get(winner_arm) == "sdk" else row["letta_reply"]
        loser = row["letta_reply"]

        toks = distinctive_tokens(winner, loser, row.get("prompt", ""))
        supplied, ok = letta_input_text(row)
        supplied_l = supplied.lower()
        avail = [t for t in toks if t.lower().strip(".,;:!?") in supplied_l]

        cause = j.get("divergence_cause")
        if not toks:
            undecidable.append(mid)
            verdict = "UNDECIDABLE->PRESENT"
        elif avail:
            present.append(mid)
            verdict = "PRESENT"
        elif not ok:
            present.append(mid)
            verdict = "RECALL-UNAVAILABLE->PRESENT"
        else:
            absent.append(mid)
            verdict = "ABSENT"

        print(f"  #{mid}  cause={cause:<18} {verdict}")
        print(f"     distinctive tokens ({len(toks)}): {', '.join(toks[:12]) or '—'}")
        print(f"     available to letta ({len(avail)}): {', '.join(avail[:12]) or '—'}")

    print()
    print(f"  PRESENT (utilization — the substrate's): {len(present)}  {present}")
    print(f"  ABSENT  (supply — never given to letta): {len(absent)}  {absent}")
    print(f"  of which undecidable, failed toward PRESENT: {len(undecidable)}  {undecidable}")
    print(f"  not audited (SDK arm lost, input not an artefact): {len(skipped)}  {skipped}")
    print()
    if absent:
        print("RESULT: some divergences charged to letta were never supplied to it — "
              "these are supply gaps, and the row-by-row evidence above is what an "
              "adjudication needs. NOT re-attributed here.")
        return 2
    print("RESULT: every divergence letta lost had its winning material in letta's own "
          "input. These are utilization failures and belong to the substrate.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="logs/letta_reply_diff_runs.json")
    ap.add_argument("--key", default="logs/letta_reply_diff_key.json")
    ap.add_argument("--judgments", required=True)
    a = ap.parse_args()
    return audit(REPO / a.runs, REPO / a.key, REPO / a.judgments)


if __name__ == "__main__":
    raise SystemExit(main())
