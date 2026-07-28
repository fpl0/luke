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

## Next steps when resuming
1. Confirm `.letta-venv/bin/letta` installed OK (`tail /tmp/claude/letta_install.log`).
2. `letta server` boot smoke test on a spare port.
3. Write `scripts/letta_import.py` per importer plan; run against backup; assert counts.
4. Write the gated adapter + a test-instance boot script; prove recall; THEN discuss cutover with Filipe.
