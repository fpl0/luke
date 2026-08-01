# Letta passage-search 500 — root cause, fix applied, remaining lead

**Date:** 2026-08-01 (~14:00 pass). **Repo:** letta-integration branch. **Symptom flagged by:**
the 4.3 archive-injection pass — `recall()`'s Letta semantic source throws
`letta_search_failed HTTP 500` on every call and fails over to FTS5, so with
`MEMORY_BACKEND=letta` live, semantic recall silently degrades to lexical-only.

This is **two independent faults** in Letta's archival passage path. One is fixed; one
is isolated but not yet fixed.

## Ground truth (all verified live, sandbox-disabled)

- `POST /v1/passages/search` and `POST /v1/archives/{id}/passages` (insert) both return
  `{"detail":"An unknown error occurred"}` / HTTP 500 — for query AND no-query list mode.
- Letta server: PID launched `.letta-venv/bin/letta server --port 8283`, orphaned (ppid 1),
  **no stderr captured** (`~/.letta/logs` empty) — the generic handler masks the traceback,
  so both faults were isolated by reproducing Letta's steps against Postgres directly.
- PG conn: `postgresql://letta:letta@localhost:5432/letta`; table `archival_passages`
  (shared across 5 archives), no HNSW/IVFFlat vector index (btree only → brute-force scan).

## FAULT 1 — embedding dimension mismatch  **[FIXED this pass]**

- Stored `embedding` column was **`vector(4096)`**, but every non-null embedding was a real
  **768-dim** vector **zero-padded to 4096** by the original bulk loader.
  Proof: `vector_dims = 4096` for all; `subvector(embedding,769,3328) <#> itself = 0` for
  **all 3400 non-null rows across every archive** (no energy past dim 768); all 4 non-null
  archives declare 768-dim models (`bge-base-en-v1.5`, `nomic-embed-text`).
- Letta operates natively at **768** (query embedding from the configured endpoint :17595 is
  768-dim; `embedding_config.embedding_dim = 768`). So every `embedding <=> query` compared
  768 vs 4096 → pgvector "different vector dimensions 768 and 4096" → 500. Insert 500'd the
  same way (Letta produced 768, column demanded 4096).
- **Fix (lossless):**
  ```sql
  ALTER TABLE archival_passages
    ALTER COLUMN embedding TYPE vector(768)
    USING CASE WHEN embedding IS NULL THEN NULL ELSE subvector(embedding,1,768) END;
  ```
  The padding is zeros, so truncation loses nothing; it aligns the DB with what Letta
  natively produces. Safe on the shared column because **no** archive has real >768 data
  (proven above). 1536 NULL-embedding rows (archive-edcc7646) preserved as NULL.
- **Verified after fix:** column is `vector(768)`; raw PG search
  `embedding <=> <768-query>` returns the correct top-1 (visa query → `entity-b1-visa-
  application`, dist 0.277). The **database side of recall is now correct.**

## FAULT 2 — Letta's text-embedding call step  **[ISOLATED, not fixed]**

- After Fault 1, raw PG vector search works, but Letta REST search **and** insert **still 500**.
- Both operations share exactly one non-DB step: **embedding the text via the configured
  endpoint `http://localhost:17595/v1`** (type `openai`). The DB is proven clean, so the
  remaining fault is in that call or its response handling.
- **Lead (not proven):** :17595 **ignores `encoding_format`** — a request with
  `{"encoding_format":"base64"}` still returns a **float array**, not a base64 string.
  Recent `openai`-python clients request base64 by default; if Letta's embedding client does,
  it would try to base64-decode a float list and throw → 500. The proxy process on :17595
  was not identifiable via `lsof` (likely inside another process/container).
- **Next-pass steps to close it:**
  1. Get the real traceback: restart the Letta server with stderr captured
     (`letta server --port 8283 2>~/.letta/logs/server.err`) in a low-contention window, then
     hit `/v1/passages/search` once and read the exception. This removes all guesswork.
  2. If it is the base64 issue: either make :17595 honor `encoding_format` (return base64 when
     asked), or pin Letta's embedding client to `encoding_format="float"`, or point
     `embedding_endpoint` at the :11434 Ollama embed server everything else uses (confirm dim
     768 parity first).
  3. Re-verify: `scripts/letta_verify_passage_search.py` (added this pass) must print
     `SEARCH_OK` with the visa entity as top-1 through the REST API, and Luke's `recall()`
     with `memory_backend=letta` must source semantic candidates from Letta (not the FTS5
     fallback) — check the `_letta_live_count` guard from `scripts/letta_shadow_diff.py`.

## Impact / posture

- Live recall **fails safe to FTS5** throughout (never broken for the user), so this is a
  shadow-run quality gap, not an outage. FTS5 lexical recall handles most queries well; the
  loss is semantic-only / paraphrase queries.
- Fault 1's fix is a strict improvement and a prerequisite for Fault 2's fix to land recall
  parity. Until Fault 2 is closed, `MEMORY_BACKEND=letta` semantic recall is lexical-only.
- Phase 6 cutover stays held post-Aug-10 regardless.
