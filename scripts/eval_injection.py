#!/usr/bin/env python3
"""Offline replay harness for memory-injection ranking changes.

Replays real queries against a read-only snapshot of the live corpus and diffs
the current ranker against a frozen copy of the pre-change formula. Run it once
before a ranking change to capture a baseline, once after, and compare.

    uv run python scripts/eval_injection.py --source log --out baseline.jsonl
    uv run python scripts/eval_injection.py --source log --out after.jsonl
    uv run python scripts/eval_injection.py --compare baseline.jsonl after.jsonl

Why a frozen scorer instead of two checkouts: both rankings come from ONE
retrieval pass, so the candidate sets are identical by construction and each
query is embedded once. Run on unmodified code, the frozen copy should agree
with the library exactly — tau 1.0, zero slot deltas. That agreement is the
harness's own self-check; if the baseline run does NOT show it, the frozen copy
has drifted from the library and every later number is worthless.

Nothing here writes to $LUKE_DIR. The snapshot is taken with sqlite3's online
backup API, never a file copy: the live DB is WAL with a live writer, and a
plain copy can capture a torn page set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

LIVE_DIR = Path(os.environ.get("LUKE_DIR", str(Path.home() / "Luke")))
LIVE_DB = LIVE_DIR / "luke.db"
LIVE_LOG = LIVE_DIR / "luke.log"
EMBED_URL = "http://127.0.0.1:17595/v1/embeddings"

# Pairing window: process() drains every pending message for a chat into one
# combined_text, so the queries behind a recall_done at T are the msg_in events
# for that chat in (T - WINDOW, T].
_PAIR_WINDOW = timedelta(seconds=90)

# Body cap used for the turn-layer token estimate. Deliberately a constant in
# THIS file rather than settings.recall_content_limit: the point is to compare
# what SELECTION costs across runs, holding the cap fixed. A run that also
# changed the cap would otherwise conflate the two effects.
_EVAL_BODY_CHARS = 3000


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def snapshot(dest: Path) -> Path:
    """Consistent read-only copy of the live corpus. Never mutates $LUKE_DIR."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    src = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    dst = sqlite3.connect(dest / "luke.db")
    try:
        with dst:
            src.backup(dst)
    finally:
        src.close()
        dst.close()

    # Memory bodies are read from disk by read_memory_body(); symlink rather
    # than copy — the harness only ever reads them.
    (dest / "memory").symlink_to(LIVE_DIR / "memory")
    return dest


def assert_embed_server() -> None:
    """Abort unless the bge server answers.

    _embed_via_server returns None on failure, which silently degrades recall to
    FTS-only. Measuring that instead of the real system is the single easiest
    way to draw a confident wrong conclusion from this harness.
    """
    req = urllib.request.Request(
        EMBED_URL,
        data=json.dumps({"input": ["ping"]}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
    except (urllib.error.URLError, OSError, RuntimeError) as exc:
        sys.exit(
            f"bge embed server unreachable at {EMBED_URL} ({exc}).\n"
            "Recall would silently degrade to FTS-only and the results would be "
            "measuring a different system. Start it and re-run."
        )


# ---------------------------------------------------------------------------
# Query sources
# ---------------------------------------------------------------------------


def _iter_log_events(log_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _ts(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError, AttributeError:
        return None


def queries_from_log(log_path: Path) -> list[dict[str, Any]]:
    """Pair each recall_done with the msg_in events that produced it.

    recall_done carries the injected ids but not the query; msg_in carries the
    text but not the recall. Pairing them is what makes replay FIDELITY
    measurable — without ground-truth ids there is no way to tell whether the
    replay reproduces production at all.
    """
    events = _iter_log_events(log_path)
    inbound: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for e in events:
        if e.get("event") != "msg_in":
            continue
        ts, text = _ts(e.get("timestamp", "")), e.get("text")
        if ts and text:
            inbound[str(e.get("chat"))].append((ts, text))
    for msgs in inbound.values():
        msgs.sort(key=lambda p: p[0])

    cases: list[dict[str, Any]] = []
    for e in events:
        if e.get("event") != "recall_done":
            continue
        ts = _ts(e.get("timestamp", ""))
        if ts is None:
            continue
        chat = str(e.get("chat_id"))
        texts = [t for (mt, t) in inbound.get(chat, []) if ts - _PAIR_WINDOW < mt <= ts]
        if not texts:
            continue
        cases.append(
            {
                "query": " ".join(texts),
                "logged_ids": list(e.get("ids") or []),
                "timestamp": e.get("timestamp"),
            }
        )
    return cases


def queries_from_messages(db_path: Path) -> list[dict[str, Any]]:
    """Wider corpus: every inbound user message. No ground-truth ids."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT content, timestamp FROM messages "
            "WHERE sender_name != ? AND content != '' ORDER BY timestamp",
            ("Luke",),
        ).fetchall()
    finally:
        conn.close()
    return [{"query": r["content"], "logged_ids": [], "timestamp": r["timestamp"]} for r in rows]


# ---------------------------------------------------------------------------
# The frozen pre-change scorer
# ---------------------------------------------------------------------------


def legacy_scores(
    pool: dict[str, dict[str, Any]],
    *,
    recency_score: Any,
    weights: tuple[float, float, float, float],
    utility_floor: float,
    utility_weight: float,
) -> dict[str, float]:
    """Frozen snapshot of memory._apply_composite_scores as of 1388ba2.

    Lives here, not in the library, so the comparison needs no second checkout
    and no runtime switch in production code. Keep byte-faithful to the original
    — the baseline self-check (tau 1.0) is what proves it still is. Delete this
    function once the change has been accepted and the baseline retired.
    """
    w_relevance, base_w_imp, base_w_rec, base_w_acc = weights
    context_denom = 1.0 - w_relevance
    log_101 = math.log(101)
    taxonomy_weights: dict[str, tuple[float, float, float]] = {
        "factual": (0.40, 0.10, 0.10),
        "experiential": (0.15, 0.35, 0.10),
        "working": (0.05, 0.45, 0.10),
    }

    out: dict[str, float] = {}
    for mem_id, entry in pool.items():
        relevance = entry.get("score", 0)
        importance = min(entry["importance"], 1.0)  # the clamp being replaced
        access_count = entry["access_count"]
        useful_count = entry["useful_count"]
        recency = recency_score(entry["updated"])
        access_score = math.log(1 + access_count) / log_101
        utility_rate = useful_count / max(access_count, 1)  # unclamped, as it was
        access_score *= utility_floor + utility_weight * utility_rate

        w_imp, w_rec, w_acc = taxonomy_weights.get(
            entry.get("taxonomy", ""), (base_w_imp, base_w_rec, base_w_acc)
        )
        context = w_imp * importance + w_rec * min(recency, 1.0) + w_acc * min(access_score, 1.0)
        context_norm = context / context_denom if context_denom > 0 else context
        out[mem_id] = relevance * context_norm if relevance > 0 else context_norm * 0.3
    return out


# ---------------------------------------------------------------------------
# Rank comparison
# ---------------------------------------------------------------------------


def kendall_tau(old: list[str], new: list[str]) -> float:
    """Kendall tau-b over the union of two rankings, absent items ranked last.

    1.0 = identical ordering, 0.0 = uncorrelated, -1.0 = reversed.
    """
    union = list(dict.fromkeys([*old, *new]))
    if len(union) < 2:
        return 1.0
    tail = len(union)
    r_old = {m: (old.index(m) if m in old else tail) for m in union}
    r_new = {m: (new.index(m) if m in new else tail) for m in union}

    concordant = discordant = 0
    for i in range(len(union)):
        for j in range(i + 1, len(union)):
            a, b = union[i], union[j]
            do = r_old[a] - r_old[b]
            dn = r_new[a] - r_new[b]
            if do == 0 or dn == 0:
                continue  # tied in one ranking — excluded from tau-b
            if (do > 0) == (dn > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def reconstruct_injected(query: str, recall_ids: list[str]) -> set[str]:
    """Rebuild the full set _auto_recall would inject, using library calls only.

    Fidelity has to compare like with like. The logged `ids` are not recall's
    top-k — they are recall PLUS trigger-matched skills (which displace recall
    hits; recent turns log 7 skills against a limit of 8) PLUS up to 3 graph
    neighbours. Comparing raw recall output against that set understates
    fidelity badly enough to look like a broken harness.

    Built from library functions rather than a frozen copy so it tracks the code
    as it changes. Consequence: once the assembler deliberately changes what
    gets injected, fidelity moves too. Interpret it on the BASELINE run, where
    it answers the only question it exists to answer — is this replay
    reproducing production well enough to trust the other metrics?

    Takes the caller's already-computed recall ids rather than re-running the
    query — recall is the expensive half and it has just been done.
    """
    from luke import memory

    ids = set(recall_ids)
    ids |= {s["id"] for s in memory.get_trigger_matched_skills(query)}
    if ids:
        ids |= {n["id"] for n in memory.get_graph_neighbors(list(ids), limit=3)}
    return ids


def run(cases: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    from luke import db, memory
    from luke.config import settings

    # Migrate the SNAPSHOT (never the live DB — LUKE_DIR was repointed before
    # this import). Without this the replay measures the new scorer against
    # un-migrated data, which is precisely the state the deploy never occupies
    # and the one where the change looks worst: unclamping importance before
    # the ceiling-valued procedures are renormalized makes them rank higher,
    # not lower.
    db.init()
    version = db.get_schema_version()
    print(f"snapshot migrated to schema v{version}")

    orig_apply = memory._apply_composite_scores
    pool: dict[str, dict[str, Any]] = {}

    def capture(results: dict[str, dict[str, Any]]) -> None:
        pool.clear()
        pool.update({k: dict(v) for k, v in results.items()})
        orig_apply(results)

    memory._apply_composite_scores = capture

    weights = (
        settings.score_weight_relevance,
        settings.score_weight_importance,
        settings.score_weight_recency,
        settings.score_weight_access,
    )
    # utility_weight is deleted by the change under test; fall back to the
    # pre-change default so the frozen scorer stays faithful after it goes.
    utility_weight = getattr(settings, "utility_weight", 0.3)
    # utility_floor is retuned by the change; the frozen scorer must keep the
    # value it had when the baseline formula was written.
    legacy_floor = 0.7

    k = settings.auto_recall_limit
    rows: list[dict[str, Any]] = []
    meta_cache: dict[str, dict[str, Any]] = {}

    try:
        # Most RECENT queries, not the oldest: the snapshot is today's corpus,
        # and an April query replayed against it is measuring four months of
        # archival, not the ranker.
        for case in cases[-limit:]:
            pool.clear()
            new_hits = memory.recall(query=case["query"], limit=k)
            if not pool:
                continue
            new_ids = [h["id"] for h in new_hits]

            old_scored = legacy_scores(
                pool,
                recency_score=memory._recency_score,
                weights=weights,
                utility_floor=legacy_floor,
                utility_weight=utility_weight,
            )
            old_ids = [
                m for m, _ in sorted(old_scored.items(), key=lambda p: p[1], reverse=True)[:k]
            ]

            for mid in set(new_ids) | set(old_ids):
                if mid not in meta_cache and mid in pool:
                    e = pool[mid]
                    meta_cache[mid] = {
                        "type": e["type"],
                        "importance": e["importance"],
                        "access_count": e["access_count"],
                        "useful_count": e["useful_count"],
                    }

            logged = case.get("logged_ids") or []
            rows.append(
                {
                    "query": case["query"][:200],
                    "timestamp": case.get("timestamp"),
                    "old_ids": old_ids,
                    "new_ids": new_ids,
                    "logged_ids": logged,
                    "injected_ids": sorted(reconstruct_injected(case["query"], new_ids))
                    if logged
                    else [],
                    "tau": round(kendall_tau(old_ids, new_ids), 4),
                    "pool_size": len(pool),
                }
            )
    finally:
        memory._apply_composite_scores = orig_apply

    return {"rows": rows, "meta": meta_cache}


def _utility_factor_for(meta: dict[str, Any]) -> float:
    """Utility factor of a memory, using the library's helper once it exists.

    Before the change lands there is no helper, so fall back to the raw rate —
    the quantity the new gate is built from anyway. getattr rather than a try/
    ImportError because the name is genuinely absent at this commit and a static
    import would not type-check.
    """
    from luke import memory

    helper = getattr(memory, "utility_factor", None)
    if helper is not None:
        return float(helper(meta["access_count"], meta["useful_count"]))
    return float(meta["useful_count"]) / max(int(meta["access_count"]), 1)


def report(result: dict[str, Any], out_path: Path | None) -> None:
    from luke.memory import read_memory_body

    rows: list[dict[str, Any]] = result["rows"]
    meta: dict[str, dict[str, Any]] = result["meta"]
    if not rows:
        sys.exit("No replayable queries produced candidates — nothing to report.")

    # (6) Replay fidelity FIRST — a validity check, not a quality metric.
    with_truth = [r for r in rows if r["logged_ids"] and r["injected_ids"]]
    print(f"\n{'=' * 68}\nREPLAY FIDELITY (validity — read this first)\n{'=' * 68}")
    if with_truth:
        jaccard = []
        recovered = []
        for r in with_truth:
            mine, logged = set(r["injected_ids"]), set(r["logged_ids"])
            jaccard.append(len(mine & logged) / len(mine | logged))
            recovered.append(len(mine & logged) / len(logged))
        print(f"  queries with ground truth   : {len(with_truth)} / {len(rows)}")
        print(f"  Jaccard(replayed, logged)   : {sum(jaccard) / len(jaccard):.1%}")
        print(f"  logged ids recovered        : {sum(recovered) / len(recovered):.1%}")
        # Rows are chronological, so the tail is genuinely the recent past.
        # The gap between these two lines IS the drift measurement: recent
        # events score far higher because their memories still exist.
        for n in (500, 200, 50):
            if len(recovered) > n:
                tail = recovered[-n:]
                print(f"    ...most recent {n:<4}      : {sum(tail) / len(tail):.1%}")
        print("  Compares the FULL injected set (recall + trigger skills +")
        print("  neighbours), not raw recall — the logged ids are that union.")
        print("  Drift is expected: memories archived since an event ran cannot")
        print("  come back. Read this on the baseline run; it is meaningless as")
        print("  a quality signal once the assembler changes what gets injected.")
    else:
        print("  no ground truth in this source (--source messages)")

    # (1) Type distribution shift
    old_types: Counter[str] = Counter()
    new_types: Counter[str] = Counter()
    for r in rows:
        for mid in r["old_ids"]:
            old_types[meta.get(mid, {}).get("type", "?")] += 1
        for mid in r["new_ids"]:
            new_types[meta.get(mid, {}).get("type", "?")] += 1
    old_total = sum(old_types.values()) or 1
    new_total = sum(new_types.values()) or 1

    print(f"\n{'=' * 68}\nTYPE DISTRIBUTION (union of all top-k)\n{'=' * 68}")
    print(f"  {'type':<12} {'old':>8} {'share':>8} {'new':>8} {'share':>8} {'delta':>8}")
    for t in sorted(set(old_types) | set(new_types), key=lambda x: -new_types[x]):
        o, n = old_types[t], new_types[t]
        print(
            f"  {t:<12} {o:>8} {o / old_total:>7.1%} {n:>8} {n / new_total:>7.1%} "
            f"{(n / new_total) - (o / old_total):>+7.1%}"
        )

    # (2) Per-memory slot delta
    slots: Counter[str] = Counter()
    for r in rows:
        for mid in r["new_ids"]:
            slots[mid] += 1
        for mid in r["old_ids"]:
            slots[mid] -= 1
    moved = [(m, d) for m, d in slots.items() if d]
    moved.sort(key=lambda p: p[1])

    def _fmt(mid: str, delta: int) -> str:
        m = meta.get(mid, {})
        util = m["useful_count"] / max(m["access_count"], 1) if m else 0.0
        return (
            f"  {delta:>+6}  {mid[:46]:<46} {m.get('type', '?'):<10} "
            f"imp {m.get('importance', 0):.2f}  util {util:.2f}"
        )

    print(f"\n{'=' * 68}\nSLOT DELTA (new appearances - old)\n{'=' * 68}")
    if moved:
        print("  biggest losers:")
        for mid, d in moved[:10]:
            print(_fmt(mid, d))
        print("  biggest gainers:")
        for mid, d in reversed(moved[-10:]):
            print(_fmt(mid, d))
    else:
        print("  no movement — rankings identical (expected on a baseline run)")

    # (3) Rank correlation
    taus = sorted(r["tau"] for r in rows)
    mean_tau = sum(taus) / len(taus)
    print(f"\n{'=' * 68}\nRANK CORRELATION (Kendall tau, old vs new)\n{'=' * 68}")
    print(
        f"  mean {mean_tau:.4f}   p10 {taus[len(taus) // 10]:.4f}   "
        f"median {taus[len(taus) // 2]:.4f}"
    )
    if mean_tau > 0.999:
        print("  ~1.0 → the frozen scorer matches the library. Baseline is valid.")

    # (4) Turn-layer token estimate, at a FIXED body cap (see _EVAL_BODY_CHARS)
    def _tokens(ids: list[str]) -> int:
        total = 0
        for mid in ids:
            m = meta.get(mid)
            if not m:
                continue
            body = read_memory_body(m["type"], mid, _EVAL_BODY_CHARS)
            total += max(1, len(f"[{mid}] ({m['type']}) {body}") // 4)
        return total

    old_tok = [_tokens(r["old_ids"]) for r in rows]
    new_tok = [_tokens(r["new_ids"]) for r in rows]
    print(f"\n{'=' * 68}")
    print(f"TURN-LAYER TOKENS (selection only, body cap held at {_EVAL_BODY_CHARS})")
    print("=" * 68)
    for label, toks in (("old", old_tok), ("new", new_tok)):
        p95 = sorted(toks)[int(len(toks) * 0.95)]
        print(f"  {label}  mean {sum(toks) / len(toks):>7.0f}   p95 {p95:>7}")

    # (5) Utility-weighted precision — the standing watch metric
    def _mean_util(key: str) -> float:
        vals = [_utility_factor_for(meta[mid]) for r in rows for mid in r[key] if mid in meta]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n{'=' * 68}\nUTILITY OF SELECTED SET (watch metric — should rise)\n{'=' * 68}")
    print(f"  old {_mean_util('old_ids'):.4f}   new {_mean_util('new_ids'):.4f}")

    print(f"\n  queries replayed: {len(rows)}\n")

    if out_path:
        with out_path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"  wrote {out_path}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("log", "messages"), default="log")
    ap.add_argument("--out", type=Path, default=None, help="write per-query JSONL here")
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp")) / "luke-eval-snapshot",
    )
    args = ap.parse_args()

    if not LIVE_DB.exists():
        sys.exit(f"live DB not found at {LIVE_DB}")
    assert_embed_server()

    dest = snapshot(args.snapshot_dir)
    print(f"snapshot: {dest}")

    # MUST precede any luke import — settings.store_dir is a cached_property,
    # resolved on first access and never recomputed.
    os.environ["LUKE_DIR"] = str(dest)

    if args.source == "log":
        if not LIVE_LOG.exists():
            sys.exit(f"log not found at {LIVE_LOG}")
        cases = queries_from_log(LIVE_LOG)
    else:
        cases = queries_from_messages(dest / "luke.db")

    from luke.app import _needs_recall  # moves to context.py in the assembler commit

    cases = [c for c in cases if _needs_recall(c["query"])]
    print(f"queries: {len(cases)} ({args.source} source, after _needs_recall)")

    report(run(cases, args.limit), args.out)


if __name__ == "__main__":
    main()
