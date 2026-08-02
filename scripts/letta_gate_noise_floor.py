#!/usr/bin/env python3
"""NOISE FLOOR — how often does the SAME arm materially disagree with ITSELF?

5.1R compares one SDK reply against one Letta reply per prompt and accepts at
``>=18/20 no-material-divergence``. That bar was written down as a *memory-parity*
threshold. It is only that if two answers to the same open-ended conversational prompt,
from systems with identical memory, would essentially always agree — i.e. if the
generation process contributes ~zero material divergence of its own.

Nobody has ever checked that. This does.

Method: replay the identical 20 prompts through the identical Letta arm a SECOND time,
with byte-identical composition (same ``as_of`` anchor, same conversation buffer, same
``inject_recall=False``), then judge pass-1 vs pass-2 with the SAME blind rubric 5.1R
uses. Every substrate variable is held constant — same agent, same memory, same core
blocks, same tools, same prompt. Whatever divergence survives is the **generation noise
floor**: rows where "material divergence" is measuring sampling, not substrate.

Why the number bounds the gate. Two samples from the same distribution diverge at some
rate k/20. Two samples from *different* distributions (SDK vs Letta) cannot be expected
to agree more often than that — a substrate difference can add divergence, not cancel
it. So ``20 - k`` is an upper bound in expectation on any cross-arm score, and if
``20 - k < 18`` the accept clause is unreachable by ANY substrate, including the
incumbent SDK path compared against itself.

Two honest caveats, both recorded because they cut in opposite directions:

  1. **k is an UNDERestimate.** Pass 2 runs against the same live agent, so pass 1's
     turns sit in its conversation buffer — the agent can see what it said and drift
     toward self-consistency. Real independent-sample noise is >= k. This is the
     conservative direction: it makes "the bar is unreachable" harder to prove, not
     easier, so a k large enough to break the bar despite this bias is solid.
  2. **k is not the whole story.** Same-arm and cross-arm divergence are not the same
     distribution. k bounds the bar; it does not by itself excuse any particular row
     Letta lost. The genuine defects stay genuine.

This script does NOT touch any 5.1R artefact and does NOT change the pack or the bar.
It produces a number. What to do about the bar is a decision, made elsewhere, in writing.

    python3 scripts/letta_gate_noise_floor.py run    # 20 turns, ~4-5 min, crash-durable
    python3 scripts/letta_gate_noise_floor.py pack   # blind P/Q pack, salted independently
    python3 scripts/letta_gate_noise_floor.py score  <judgments.json>
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
LOGS = os.path.join(REPO, "logs")

SET_PATH = os.path.join(LOGS, "letta_reply_diff_set.json")
PASS1_PATH = os.path.join(LOGS, "letta_reply_diff_runs.json")
PASS2_PATH = os.path.join(LOGS, "noise_floor_pass2.json")
PARTIAL_PATH = os.path.join(LOGS, "noise_floor_partial.jsonl")


def _slot(n: int) -> str:
    """Pass N's artefact. Pass 1 is the 5.1R replay itself; 2+ are this script's.

    Slots exist because the first attempt at this measurement was confounded and the
    confound was invisible until the key was opened: the ``feedback-verify-self-claims``
    guardrail was packed into the agent's operating-rules block BETWEEN pass 1 (13:07Z)
    and pass 2 (13:31Z). Two of the four divergences were that change, not noise. A noise
    floor has to compare two passes taken under one configuration, so the config has to be
    addressable — otherwise "same arm twice" quietly means "two different arms".
    """
    return PASS1_PATH if n == 1 else os.path.join(LOGS, f"noise_floor_pass{n}.json")
PACK_PATH = os.path.join(LOGS, "noise_floor_pack.md")
KEY_PATH = os.path.join(LOGS, "noise_floor_key.json")
VERDICT_LOG = os.path.join(LOGS, "noise_floor.log")


def _rd():
    """The 5.1R module, loaded by path — its _plain/_arm_order are the shared surface."""
    spec = importlib.util.spec_from_file_location(
        "rd", os.path.join(REPO, "scripts", "letta_reply_diff.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_inputs() -> tuple[dict, dict]:
    """Set + pass-1 runs, with the same provenance refusal 5.1R's `pack` enforces.

    A noise floor measured against a pass-1 from a different harness is not a noise
    floor — it is the harness diff, wearing the same number. Refuse rather than rescale.
    """
    if not os.path.exists(PASS1_PATH):
        sys.exit(f"No pass-1 runs at {PASS1_PATH} — run the 5.1R replay to completion first.")
    with open(SET_PATH) as f:
        spec = json.load(f)
    with open(PASS1_PATH) as f:
        pass1 = json.load(f)
    if pass1.get("set_built") != spec["built"]:
        sys.exit(
            f"REFUSING: pass-1 was replayed against set build "
            f"{pass1.get('set_built') or '(unstamped)'}, the set on disk is {spec['built']}. "
            f"A noise floor across two harnesses measures the harness, not the noise."
        )
    if len(pass1["rows"]) != len(spec["prompts"]):
        sys.exit(f"REFUSING: pass-1 has {len(pass1['rows'])} rows, set has {len(spec['prompts'])}.")
    return spec, pass1


def _load_partial(set_built: str, valid: set[int]) -> list[dict]:
    """Resume journal. Same keying as 5.1R: a row from another build is dropped LOUDLY."""
    if not os.path.exists(PARTIAL_PATH):
        return []
    rows, dropped = [], 0
    with open(PARTIAL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1  # torn final line from a kill mid-write
                continue
            if r.get("set_built") != set_built or r.get("msg_id") not in valid:
                dropped += 1
                continue
            rows.append(r)
    if dropped:
        print(f"  journal: dropped {dropped} row(s) from a different build / torn write")
    seen, out = set(), []
    for r in rows:  # last write per msg_id wins
        out = [x for x in out if x["msg_id"] != r["msg_id"]]
        out.append(r)
        seen.add(r["msg_id"])
    return out


def cmd_run(fresh: bool = False, limit: int | None = None, slot: int = 2) -> None:
    from luke.letta_agent import compose_letta_turn_input, drive_letta_turn

    spec, _pass1 = _load_inputs()
    prompts, set_built = spec["prompts"], spec["built"]

    if fresh and os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)
        print("  journal: cleared (--fresh)")
    done = {r["msg_id"]: r for r in _load_partial(set_built, {p["msg_id"] for p in prompts})}
    if done:
        print(f"  journal: resuming with {len(done)}/{len(prompts)} turn(s) already replayed")

    todo = [c for c in prompts if c["msg_id"] not in done]
    if limit is not None:
        todo = todo[:limit]

    out = []
    for n, c in enumerate(todo, 1):
        t0 = time.time()
        # Byte-identical composition to cmd_run in letta_reply_diff. If this drifts, the
        # measurement stops being "the same input twice" and the number means nothing.
        body = compose_letta_turn_input(c["prompt"], as_of=c["ts"])
        injected = "<mem id=" in body
        prior = c.get("prior_context") or []
        if prior:
            body = "[Recent conversation, for context:]\n" + "\n".join(prior) + "\n\n" + body
        r = drive_letta_turn(body, inject_recall=False)
        reply = (r.get("reply") or "").strip()
        invalid = None
        if r.get("error"):
            invalid = f"turn error: {r['error'][:120]}"
        elif len(reply) < 40:
            invalid = f"degenerate reply ({len(reply)} chars)"
        row = {
            "msg_id": c["msg_id"],
            "letta_reply_p2": reply,
            "seconds": round(r.get("seconds", time.time() - t0), 2),
            "tools": r.get("tools") or [],
            "injected": injected,
            "invalid": invalid,
        }
        out.append(row)
        with open(PARTIAL_PATH, "a") as f:
            f.write(json.dumps({**row, "set_built": set_built}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        flag = "INVALID: " + invalid if invalid else f"ok {row['seconds']}s"
        print(f"  [{n:2}/{len(todo)}] {'inj' if injected else 'NO-INJ':7} {flag}  "
              f"{c['prompt'][:56]}", flush=True)

    have = {**done, **{r["msg_id"]: r for r in out}}
    rows = [{k: v for k, v in have[c["msg_id"]].items() if k != "set_built"}
            for c in prompts if c["msg_id"] in have]
    if len(rows) < len(prompts):
        # Same anti-fake guard as 5.1R: a partial pass must not leave a packable artefact.
        print(f"\nINCOMPLETE — {len(rows)}/{len(prompts)} replayed, {_slot(slot)} NOT written.")
        return
    with open(_slot(slot), "w") as f:
        json.dump({"ran": datetime.now(timezone.utc).isoformat(),
                   "set_built": set_built, "rows": rows}, f, indent=2)
    os.remove(PARTIAL_PATH)
    n_inv = sum(1 for r in rows if r["invalid"])
    print(f"\nwrote {_slot(slot)}  valid={len(rows) - n_inv}/{len(rows)}")


def _pq_order(msg_id: int) -> tuple[str, str]:
    """Independent salt from 5.1R's ``_arm_order``.

    Reusing that shuffle would let anyone who has already seen the 5.1R key infer this
    pack's assignment for free — and the person judging this pack is the person who
    judged that one. Salted separately so blindness is real, not nominal.
    """
    h = hashlib.md5(f"noisefloor:{msg_id}".encode()).hexdigest()
    return ("p1", "p2") if int(h[0], 16) % 2 == 0 else ("p2", "p1")


def cmd_pack(a: int = 1, b: int = 2) -> None:
    rd = _rd()
    spec, pass1 = _load_inputs()
    for n in (a, b):
        if not os.path.exists(_slot(n)):
            sys.exit(f"No pass-{n} at {_slot(n)} — run `run --slot {n}` to completion first.")
    with open(_slot(a)) as f:
        passa = json.load(f)
    with open(_slot(b)) as f:
        pass2 = json.load(f)
    pass1 = passa
    if pass2.get("set_built") != spec["built"]:
        sys.exit("REFUSING: pass-2 belongs to a different set build.")
    def _reply(r):
        # Pass 1 is a 5.1R runs row (`letta_reply`); later passes are this script's
        # (`letta_reply_p2`). Read whichever is present rather than assuming a slot's shape.
        return r.get("letta_reply") if "letta_reply" in r else r.get("letta_reply_p2", "")

    p1 = {r["msg_id"]: r for r in pass1["rows"]}
    p2 = {r["msg_id"]: r for r in pass2["rows"]}
    if set(p1) != set(p2):
        sys.exit(f"REFUSING: pass-1 covers {len(p1)} rows, pass-2 covers {len(p2)}.")

    key, parts = {}, []
    parts.append(
        "# Noise-floor blind judge pack (same arm, twice)\n\n"
        "Each item shows a real user prompt and two candidate answers, **P** and **Q**, in a "
        "shuffled order. Judge them on exactly the rubric 5.1R uses — **factual grounding**, "
        "**Luke-voice**, **actionability** — and emit one JSON object per item:\n\n"
        '`{"msg_id": 123, "material_divergence": false, "worse_arm": null, '
        '"factually_wrong_arm": null, "note": "..."}`\n\n'
        "- `material_divergence`: true if a reader would get a *meaningfully different* answer "
        "— different facts, a real quality gap. Formatting/length alone is NOT material.\n"
        "- `worse_arm`: `\"P\"`, `\"Q\"`, or null if no material gap.\n"
        "- `factually_wrong_arm`: `\"P\"`/`\"Q\"`/null — set ONLY when one states something "
        "false the other got right.\n\n"
        "Apply the SAME threshold for 'material' you applied in 5.1R. If you are stricter or "
        "looser here the comparison is void — the whole point is that one rubric is being held "
        "constant across two packs.\n\n"
        "Do NOT open `noise_floor_key.json` until every judgment is written.\n\n---\n"
    )
    for i, c in enumerate(spec["prompts"], 1):
        mid = c["msg_id"]
        first, second = _pq_order(mid)
        key[str(mid)] = {"P": first, "Q": second, "slots": {"p1": a, "p2": b},
                         "invalid": p1[mid]["invalid"] or p2[mid]["invalid"]}
        texts = {
            "p1": rd._plain(_reply(p1[mid])) or "(no answer produced)",
            "p2": rd._plain(_reply(p2[mid])) or "(no answer produced)",
        }
        parts.append(
            f"\n## Item {i} — msg #{mid} ({c['bucket']}, {c['ts'][:16]})\n\n"
            f"**Prompt:** {c['prompt']}\n\n"
            f"**Answer P:**\n\n{texts[first][:2200]}\n\n"
            f"**Answer Q:**\n\n{texts[second][:2200]}\n"
        )
    with open(PACK_PATH, "w") as f:
        f.write("\n".join(parts))
    with open(KEY_PATH, "w") as f:
        json.dump({"set_built": spec["built"], "slots": [a, b], "arms": key}, f, indent=2)
    print(f"wrote {PACK_PATH} ({len(spec['prompts'])} items)  +  {KEY_PATH} (do not read yet)")


def cmd_score(judgments_path: str) -> None:
    spec, _ = _load_inputs()
    with open(KEY_PATH) as f:
        keyfile = json.load(f)
    if keyfile.get("set_built") != spec["built"]:
        sys.exit("REFUSING: key belongs to a different set build — re-pack and re-judge.")
    key = keyfile["arms"]
    with open(judgments_path) as f:
        raw = json.load(f)
    judgments = raw["judgments"] if isinstance(raw, dict) else raw

    n = len(key)
    diverged, wrong, lines, seen = [], [], [], set()
    for j in judgments:
        mid = str(j["msg_id"])
        seen.add(mid)
        if mid not in key:
            continue
        if j.get("material_divergence"):
            diverged.append(mid)
            lines.append(f"  #{mid} SELF-DIVERGE  {j.get('note', '')[:88]}")
        else:
            lines.append(f"  #{mid} stable")
        if j.get("factually_wrong_arm") in ("P", "Q"):
            wrong.append(mid)
    for mid in key:
        if mid not in seen:
            lines.append(f"  #{mid} FAIL (no judgment emitted)")
            diverged.append(mid)

    k = len(diverged)
    ceiling = n - k
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    head = (
        f"NOISE-FLOOR {stamp} — same arm twice: self-divergence {k}/{n}; "
        f"one arm factually wrong where the other was right on {len(wrong)}/{n}. "
        f"=> expected cross-arm ceiling <= {ceiling}/{n}; 5.1R bar is 18/20 "
        f"({'REACHABLE' if ceiling >= 18 else 'UNREACHABLE — the bar exceeds the noise floor'})"
    )
    print(head)
    for ln in lines:
        print(ln)
    if wrong:
        print(f"\n  factual-veto rows (same arm contradicting itself): {', '.join(wrong)}")
    os.makedirs(LOGS, exist_ok=True)
    with open(VERDICT_LOG, "a") as f:
        f.write(head + "\n")
    print(f"\nappended to {VERDICT_LOG}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        cmd_run(fresh="--fresh" in sys.argv,
                limit=int(sys.argv[sys.argv.index("--limit") + 1])
                if "--limit" in sys.argv else None,
                slot=int(sys.argv[sys.argv.index("--slot") + 1]) if "--slot" in sys.argv else 2)
    elif cmd == "pack":
        slots = [int(x) for x in sys.argv[2:4]] if len(sys.argv) > 3 else [1, 2]
        cmd_pack(*slots)
    elif cmd == "score":
        cmd_score(sys.argv[2])
    else:
        sys.exit(__doc__)
