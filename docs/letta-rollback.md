# Letta Integration — Safety & Rollback

Branch: `letta-integration` (created off clean `main` @ 6557e92 on 2026-07-28).
Goal: test a Letta-backed memory core WITHOUT touching `main` and WITHOUT losing any memory.

## Backups taken before any work (verified)
- `/Users/filipelm/Luke/backups/pre-letta-20260728/luke.db` — full memory/events DB, `PRAGMA integrity_check = ok`, 1535 memories (memory_meta), 11962 events. 29 MB.
- `/Users/filipelm/Luke/backups/pre-letta-20260728/file-memory/` — 25 files incl. MEMORY.md (20448 bytes).

## Hard safety rules for this branch
1. NEVER commit to `main`. All work stays on `letta-integration`.
2. NEVER read/write the LIVE `/Users/filipelm/Luke/luke.db` for migration — import from the backup copy only.
3. Letta runs as a SEPARATE local server + a SEPARATE test Luke instance on a non-default port. Do NOT hot-swap the live process until the branch is proven on the test instance.
4. Letta backend is gated behind env `LUKE_MEMORY_BACKEND=letta` (default stays the current SQLite memory, so main behavior is unchanged unless explicitly opted in).

## Rollback (any failure)
```bash
cd /Users/filipelm/Code/luke
git checkout main            # instantly back to the shipped brain
./guardian.sh                # (auto-reverts bad deploys on its own too)
# restore memory if ever needed:
cp /Users/filipelm/Luke/backups/pre-letta-20260728/luke.db /Users/filipelm/Luke/luke.db
```
The live process runs on `main`; switching branches + restart returns it exactly to today's state.
