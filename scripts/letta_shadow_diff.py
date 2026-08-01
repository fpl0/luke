#!/usr/bin/env python3
"""Phase 5.1 — shadow-run recall divergence harness.

Runs the REAL production `recall()` (FTS5 + semantic + RRF + graph + composite
scoring) twice per query — once with settings.memory_backend='sqlite', once with
='letta' — and diffs the fused top-k. This is the production-representative
divergence (what a user would actually experience), NOT the archive-top-1
benchmark (letta_bench_titled.py) — recall fuses FTS (always sqlite) with the
semantic candidates the backend supplies, so the final ranking can agree even
when the raw archive top-1 differs, and vice-versa.

Off the OAuth pool: recall only touches the embed server + Letta search API +
sqlite. No Claude turn. Safe to run inside a live session.

Accept (Phase 5): "zero UNHANDLED divergences" — a divergence is handled if the
letta result is an equal-or-better answer (tie / better-match), unhandled if it's
a genuine regression (letta drops the right memory the sqlite path surfaces).
This harness classifies; a human eyeballs the DIVERGE rows.
"""
import sys, time, json

sys.path.insert(0, "src")
from luke.config import settings  # noqa: E402
from luke import memory as M  # noqa: E402

TOPK = 5

# Representative of real recall usage — current life context + the known-hard
# semantic queries from the 2.3/2.4 benchmarks. Mix of easy (exact-title-ish),
# hard (semantic-only), and ambiguous (multiple plausible answers) so the diff
# exercises the fusion path, not just clean hits.
QUERIES = [
    "when does Filipe start his new job at CarGurus",
    "what does Christopher do for work",
    "Filipe's B1 US visa interview appointment details",
    "when is Filipe's birthday",
    "should Luke migrate its memory to postgres",
    "the Letta full-power build plan",
    "who is Prerna",
    "how do I make a PDF",
    "what are Filipe's communication preferences",
    "the CarGurus founding EM role and team charter",
    "Mark Anderson situation at Clio",
    "how does the deep work session protocol work",
    "what happened during the six week silence",
    "Theo the parallel AI agent Filipe is building",
    "voice cloning in Filipe's voice",
    "never send emails on Filipe's behalf",
    "stress leave and return to work",
    "the visa interview is on August 7",
    "how to verify a production commit is live",
    "what is the observability cliff and why hold cutover",
]


# TIER 2 — adversarial paraphrases with ~zero keyword overlap with the target's
# title, so FTS5 contributes little and the SEMANTIC backend actually drives the
# ranking. This is where a letta-vs-sqlite divergence can surface; the tier-1
# real-usage queries are FTS-anchored and wash it out. An unhandled divergence =
# sqlite surfaces the correct memory in the top-k and letta drops it.
ADVERSARIAL = [
    "which employer is Filipe joining next month",
    "the person Filipe lives with and their profession",
    "the appointment at the American embassy in Dublin",
    "swapping the storage layer under the memory system",
    "the manager who loves tea on the Boston team",
    "the strange quiet stretch before the new job begins",
    "why not cut the memory system over the week the job starts",
]


def _rid(r):
    return r.id if hasattr(r, "id") else r["id"]


def _rscore(r):
    s = r.score if hasattr(r, "score") else r.get("score", 0.0)
    return round(s or 0.0, 4)


def _letta_live_count(queries):
    """How many queries actually sourced candidates FROM Letta (non-None).
    The recall() letta path fails safe to sqlite on any miss (cold server,
    timeout, empty) — if this is < len(queries) the 'letta' pass is partly or
    wholly sqlite-vs-sqlite and any 'identical' verdict is meaningless. This is
    the guard that caught the first run's false 20/20 (cold-start fallback)."""
    from luke.letta_adapter import letta_semantic_search
    live = 0
    for q in queries:
        try:
            if letta_semantic_search(q, limit=TOPK) is not None:
                live += 1
        except Exception:
            pass
    return live


def run(backend, queries):
    settings.memory_backend = backend
    out = {}
    for q in queries:
        try:
            res = M.recall(query=q, limit=TOPK)
            out[q] = [(_rid(r), _rscore(r)) for r in res]
        except Exception as e:
            out[q] = [("__ERROR__:" + str(e)[:60], 0.0)]
    return out


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def tier(name, queries):
    print(f"\n[{time.strftime('%H:%M:%S')}] === TIER {name}: {len(queries)} queries x2 backends ===", flush=True)
    live = _letta_live_count(queries)
    guard = "OK" if live == len(queries) else "!! DEGRADED — letta partly fell back to sqlite; diff is unreliable"
    print(f"  LETTA-SOURCED GUARD: {live}/{len(queries)} queries sourced from Letta [{guard}]", flush=True)
    sq = run("sqlite", queries)
    le = run("letta", queries)
    settings.memory_backend = "sqlite"

    top1_match = 0
    diverge = []
    jac_sum = 0.0
    rows = []
    for q in queries:
        s_ids = [i for i, _ in sq[q]]
        l_ids = [i for i, _ in le[q]]
        s1 = s_ids[0] if s_ids else None
        l1 = l_ids[0] if l_ids else None
        j = jaccard(s_ids, l_ids)
        jac_sum += j
        match = s1 == l1
        if match:
            top1_match += 1
            cls = "IDENTICAL" if s_ids == l_ids else "TOP1-MATCH/REORDER"
        else:
            cls = "TOP1-DIVERGE"
            diverge.append((q, s_ids, l_ids))
        rows.append((cls, round(j, 2), q, s1, l1))

    n = len(queries)
    print(f"  TOP-1 AGREEMENT: {top1_match}/{n}   MEAN TOP-{TOPK} JACCARD: {jac_sum/n:.2f}")
    for cls, j, q, s1, l1 in rows:
        print(f"    [{cls:18}] jac={j:.2f}  {q}")
        if cls == "TOP1-DIVERGE":
            print(f"        sqlite#1: {s1}")
            print(f"        letta #1: {l1}")
    if diverge:
        print(f"  DIVERGENCES TO ADJUDICATE ({len(diverge)}) — is letta's #1 a tie/better, or a regression?")
        for q, s_ids, l_ids in diverge:
            print(f"    Q: {q}")
            print(f"      sqlite top{TOPK}: {s_ids}")
            print(f"      letta  top{TOPK}: {l_ids}")
    return {"tier": name, "queries": n, "top1_agreement": top1_match,
            "mean_jaccard": round(jac_sum / n, 3),
            "divergences": [{"q": q, "sqlite": s, "letta": l} for q, s, l in diverge]}


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Phase 5.1 shadow-run recall divergence — real recall() x2 backends", flush=True)
    r1 = tier("1 REAL-USAGE (FTS-anchored)", QUERIES)
    r2 = tier("2 ADVERSARIAL (semantic-driven)", ADVERSARIAL)
    print("\n" + "=" * 78)
    print("JSON_RESULT=" + json.dumps({"tiers": [r1, r2]}))


if __name__ == "__main__":
    main()
