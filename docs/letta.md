# Letta Backend

Luke 2.0 introduces [Letta](https://docs.letta.com) as an optional backend behind two
narrow, independently reversible seams. Sqlite remains the source of truth for every
memory row; Letta adds a server-side vector store and self-editing core memory on top.

## The two seams

**`MEMORY_BACKEND`** (`sqlite` | `letta`) — where `recall()` sources its *semantic
candidate set*. On `letta`, the adapter queries the Letta archive's vector store and
re-joins candidate ids onto Luke's own `memory_meta`/`memory_fts` rows, so FTS5, RRF
fusion, graph traversal, and composite scoring are unchanged. Any error, timeout, or
empty response falls back to the in-process sqlite-vec index — Letta being down must
never break recall.

**`AGENT_BACKEND`** (`sdk` | `letta`) — where a turn's always-in-context world model
comes from. On `letta`, the system-prompt append is assembled from the agent's
self-editing core memory blocks instead of re-building the `build_working_context`
blob from sqlite each turn. Fails back to the sqlite blob.

Write path: when `MEMORY_BACKEND=letta`, every `remember`/`connect`/`forget`/`restore`
is mirrored live into the Letta archive (idempotent upsert by `luke_id`, both edge
endpoints re-mirrored on link expiry). Write-through failures never break the sqlite
write; the daily delta-sync cron is the backstop.

## Source modules

| Module | Seam |
|--------|------|
| `src/luke/letta_adapter.py` | recall candidate sourcing (read) |
| `src/luke/letta_writer.py` | live write-through mirror (write) |
| `src/luke/letta_agent.py` | core-block context assembly + turn driver |

All three are REST-only over `urllib` — no Letta SDK dependency in the core runtime.
Deleting the three files and the call sites in `memory.py`/`agent.py` fully reverts.

## The stack (launchd)

| Job | Runs | Port |
|-----|------|------|
| (manual/venv) | Letta server (Postgres-backed) | 8283 |
| `com.luke.bridge` | `scripts/claude_letta_bridge.py` — OpenAI→Anthropic proxy so Letta drives Claude off the OAuth token | 17596 |
| `com.luke.bgeembed` | `scripts/bge_embed_server.py` — bge-base-en-v1.5 embeddings (same model as sqlite-vec, parity by construction) | 17595 |
| `com.luke.lettasync` | `scripts/letta_delta_sync.py` — daily watermark delta-sync (04:30) | — |
| `com.luke.lettahealth` | `letta_stack_health.sh` — fail-loud Telegram monitor, every 5 min | — |

Postgres serves the Letta server from `$LUKE_DIR/letta-data/pgdata` on 5432.

## Configuration

Set in `.env` (see `.env.example`). `LETTA_ARCHIVE_ID` and `LETTA_AGENT_ID` are
deployment artifacts printed by the provisioning scripts
(`scripts/letta_load_bge_titled.py`, `scripts/letta_luke_on_claude.py`); an empty id
disables that letta path even when the backend flag says `letta`.

## Scripts

`scripts/letta_*.py` are experiment/provisioning tooling, not runtime code — they run
from the separate `.letta-venv` (which carries `letta-client`). The exceptions wired
into launchd are listed above. `scripts/probe_*.py` are live-stack verification probes;
they are named `probe_`, not `test_`, precisely so pytest can never collect them
(`testpaths = ["tests"]` is the second lock on that door).

## Deep-dive docs

- [letta-memory-model-mapping.md](letta-memory-model-mapping.md) — which memory type maps to core block vs archival passage
- [letta-migration-notes.md](letta-migration-notes.md) — verified source schema + API contract findings
- [letta-recall-500-rootcause.md](letta-recall-500-rootcause.md) — the two recall-500 faults (dimension mismatch; embed server ignoring `encoding_format`)
- [letta-rollback.md](letta-rollback.md) — rollback contract and backup locations
