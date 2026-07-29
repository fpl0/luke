# Letta Migration — grounded working notes (2026-07-28)

Status: safety foundation DONE + committed (e1d9a48). Letta installing into `.letta-venv` (py3.12).
This file is the durable handoff so the build resumes precisely without re-deriving anything.

## Verified source memory schema (from backup luke.db)
- `memory_meta` — metadata. Columns: `id, type, created, updated, access_count, useful_count,
  importance, status, tags_json, links_json, is_private, last_accessed, taxonomy, skill_meta, cluster_id`.
  NOTE: type column is `type` (not `mem_type`). Status values incl. `active`, `archived`.
- Active memories (763): insight 506, procedure 204, entity 26, episode 26, goal 1. Total rows 1535 (incl archived).
- Content: `memory_fts` (FTS5 virtual table); shadow `memory_fts_content` has c0=id, c1=type, c2..=text fields.
- Vectors: `memory_vec` (+ sqlite-vec sidecars). Links: `memory_links` (6912 rows). History: `memory_history`.

## Importer plan (import from BACKUP copy only, never live db)
1. Read active+archived rows from `memory_meta` joined to content from `memory_fts`.
2. For each: create a Letta archival-memory passage (or memory block for high-importance entities/goals),
   carrying id, type, importance, tags, created/updated as metadata.
3. Preserve links_json as Letta relations (or metadata) so the graph survives.
4. VERIFY: count imported == count source (assert 1535). Spot-check 5 known ids incl. `person-filipe`,
   `user-preferences`, the active goal.

## API contract VERIFIED against openapi_letta.json (2026-07-28 15:30, accountability loop)
Every operation the importer + adapter need exists in the real API — model is feasible, no gap:
- Self-editing working memory → `GET/PATCH /v1/agents/{id}/core-memory/blocks/{label}` + attach/detach.
  This is the whole reason for the migration (native self-editing memory blocks). Present.
- Recall over the memory bank → `GET /v1/agents/{id}/archival-memory/search` (+ `POST /v1/passages/search`).
  Maps 1:1 to Luke's `recall`. Present.
- Bulk migration of the 1535 rows → `POST /v1/archives/{archive_id}/passages/batch`. Present —
  so import is a batched load, not 1535 serial POSTs.
- Per-memory CRUD → archival-memory GET/POST/DELETE by memory_id. Present.
CONCLUSION: foundation is genuinely ready; the swap is a build decision (Filipe's call), not a feasibility risk.

## Adapter plan (gated, reversible)
- Env flag `LUKE_MEMORY_BACKEND` (default `sqlite` = current behavior; `letta` = new path).
- Letta server runs locally on a NON-default port; test Luke instance points at it.
- Do NOT hot-swap the live process until the test instance answers + recalls a known memory.

## Rollback: `git checkout main` (+ guardian.sh auto-revert). Backups in
`/Users/filipelm/Luke/backups/pre-letta-20260728/`. See LETTA_ROLLBACK.md.

## CORRECTION to importer plan — content reconciliation (2026-07-29)
The step-1 assumption ("read active+archived rows joined to content from memory_fts")
is WRONG: `memory_fts` only holds ACTIVE rows. Archiving removes the FTS entry, so
archived text is not joinable there. Verified ground truth on the backup:
- 1535 memory_meta rows: 763 active, 771 archived, 1 paused.
- Content recoverable: 764 from FTS + 136 from latest `memory_history.new_content` = **900**.
- **635 archived rows are metadata-only tombstones** — no text exists anywhere in the db.
So "assert imported == 1535 WITH content" is impossible. `scripts/letta_import.py`
reconciles explicitly instead: every row becomes a passage (tombstones carry a
placeholder + full metadata), nothing is silently dropped, and the 900 content-bearing
memories — which include the entire ACTIVE recall path — migrate intact.
NOTE: `goal-msc-cognitive-science` is archived in the db (a stale "paused" label lingers
in injected context); its substance survives across the active `insight-msc-*` rows. The
one genuinely active goal is `goal-voicebox-luke-voice` (spot-checked, content present).

## Next steps when resuming
1. DONE — `.letta-venv/bin/letta` (letta 0.16.8) installed; backup verified.
2. DONE — `scripts/letta_import.py` written; `--dry-run` reconciles all 1535 rows,
   asserts totals, spot-checks person-filipe + user-preferences + the active goal. PASSES.
3. TODO — `letta server` boot smoke test on a spare port (server currently DOWN on 8283,
   pg DOWN on 5432; letta can run on its bundled sqlite for a local test instance).
4. TODO — wire `--load` batch call against a test archive; prove recall of a known id.
5. GATED ON FILIPE — the live hot-swap (irreversible). Everything above is reversible.
