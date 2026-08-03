# Storage Unification

Luke's state is spread across ten substrates. This document inventories them,
shows what that split is costing (including one live, silent failure), sets the
rule for what belongs in SQLite, and lays out the migration.

## 1. What exists today

Measured against the live instance at `~/Luke` on 2026-08-03.

| # | Substrate | Size | Holds | Verdict |
|---|-----------|------|-------|---------|
| 1 | `luke.db` | 31 MB, 30 tables | messages, tasks, events, intents, cost, memory metadata/links/history/vectors/FTS, clusters, corrections, skill triggers, behavior state | **canonical** |
| 2 | `memory/**/*.md` | 1568 files, 6.3 MB | memory *content* (frontmatter + body) | duplicated into FTS |
| 3 | `workspace/plans/*.md` | 13 files | plan status, last-updated, step checklists | **regex-parsed state** |
| 4 | goal memory bodies | 22 files | `<b>Status:</b>`, `<b>Progress:</b>`, `<b>Deadline:</b>` inside CDATA | **regex-parsed state** |
| 5 | `claims/*.json` | 0 live | advisory work locks (O_EXCL + TTL + pid liveness) | hand-rolled txn |
| 6 | `heartbeat`, `deadman_state`, `guardian_state`, `luke.lock`, `luke.sock`, `known_good_commit`, `restart_status.txt` | 7 files | process liveness / supervision | **stays on disk** |
| 7 | `LUKE.md`, `constitutional.yaml`, `context.yaml` | 3 files | persona, invariants, user context | **stays on disk** |
| 8 | `workspace/activity_log.jsonl`, `scorecard_data.json`, `crashes/*.json`, `workspace/mirror/*.html` | ad-hoc | logs and artifacts | fold into `events` |
| 9 | `workspace/dashboard/dashboard.db` | **0 bytes** | nothing | delete |
| 10 | `media/`, `workspace/` blobs, `.txt` transcripts beside `.ogg` | 3.9 GB | binaries and build output | **stays on disk** |

## 2. What it costs

### 2.1 A live failure: every goal is invisible

All 22 goal memories carry `memory_meta.status = 'archived'`. `recall()` filters
to `status = 'active'`, so:

```
memory.recall(mem_type='goal', limit=10)  ->  0 results
```

Three call sites depend on that returning something —
[behaviors.py:451](src/luke/behaviors.py:451) (proactive scan),
[behaviors.py:743](src/luke/behaviors.py:743) (deep work) and
[behaviors.py:1307](src/luke/behaviors.py:1307) (reflection). All three have been
running against an empty goal list. Nothing errors; the loops just find nothing
to pursue.

The root cause is a conflated field. `memory_meta.status` answers *"should
retrieval surface this document?"*. It is also being used to answer *"is Filipe
still pursuing this objective?"*. Those are different questions, and archiving a
finished goal's *document* silently retired the whole *goal system*.

### 2.2 The same fact stored three times, disagreeing

Goal status lives in `memory_meta.status`, in the memory body as HTML-in-CDATA,
and in `workspace/plans/{goal_id}.md` as `**Status:**`. They do not agree:

| goal | `memory_meta` | memory body | plan file |
|------|---------------|-------------|-----------|
| `goal-new-job` | archived | paused | **completed** |
| `goal-msc-cognitive-science` | archived | paused | **abandoned** |
| `goal-luke-v16-close-the-loops` | archived | blocked | **abandoned** |
| `goal-modern-java-mastery` | archived | abandoned | abandoned |

There is no tiebreak rule, so the answer depends on which module you ask.

Meanwhile `workspace/plans/cargurus-ramp.md` is `**Status:** in_progress` and its
own text calls it *"Filipe's stated top priority"* — but it has no goal memory at
all, so no goal-driven code path can ever reach it. Six of the 13 plan files have
no corresponding goal.

### 2.3 Control flow driven by regex over prose

Plan status is extracted by regex in four separate places — `_parse_plan_status`
([behaviors.py:552](src/luke/behaviors.py:552)), `_plan_last_updated`
([behaviors.py:583](src/luke/behaviors.py:583)), `enforce_plan_momentum`
([behaviors.py:609](src/luke/behaviors.py:609)), `reconcile_stale_plans`
([memory.py:1381](src/luke/memory.py:1381)) — plus a fifth in the dashboard
(`api_plans`, server.py:994). Each has its own pattern. A plan that writes
`Status: paused` instead of `**Status:** paused` is invisible to some of them and
not others. `reconcile_stale_plans` then *rewrites the markdown in place* to
correct status — code editing prose to store a state transition.

### 2.4 Silent index drift on hand-edited files

`sync_memory_index()` ([memory.py:2122](src/luke/memory.py:2122)) only indexes
files whose id is **not already** in `memory_meta`. Any edit to an
already-indexed file — by hand, or by the agent's own `Edit`/`Write` tools —
never reaches FTS or the vector index. Currently 15 files are newer on disk than
their last `index_memory()` call. Twelve are goals, i.e. exactly the memories
that get mutated in place.

### 2.5 Content stored twice

`memory_fts` is a normal FTS5 table with an indexed `content` column, so every
memory body exists as a markdown file *and* as a row copy. Two writers, no
constraint tying them together.

### 2.6 What is *not* broken

Worth stating, because it bounds the work. The file↔DB sync for ordinary
memories is sound: 1568 files vs 1569 `memory_meta` rows, exactly 1 orphan
(`insight-self-modification-crash-loop`, meta row with no file). Zero active
memories are missing from FTS or from the vector index. The 791/1569 FTS gap is
*intentional* — the 778 archived memories are deliberately dropped from FTS by
`cleanup_archived_fts()`. Episodes, insights, procedures and entities are
append-mostly and stay consistent.

**The problem is not the dual store. It is mutable state living in prose.**

## 3. The rule

> **SQLite owns state that is queried, mutated, or must stay consistent.
> The filesystem owns bytes that are authored, read whole, and edited by a human.**
>
> Corollary: no control-flow decision may depend on a regex over prose.

This is what "unify on SQLite" should mean here — not "put every byte in the
database". Three categories deliberately stay on disk:

- **Supervision primitives** (`heartbeat`, `deadman_state`, `luke.lock`,
  `luke.sock`). These are the out-of-band liveness channel. A watchdog that
  queries SQLite to decide whether the SQLite-holding process is wedged deadlocks
  in exactly the scenario it exists to catch. They stay files.
- **Config and persona** (`LUKE.md`, `constitutional.yaml`, `context.yaml`).
  Read once at startup, hand-edited, diffable. A database row is strictly worse
  to edit and buys nothing.
- **Blobs and build output** (`media/`, `workspace/`). The filesystem is the
  right database for 3.9 GB of files.

## 4. Correctness is the constraint

Memory correctness outranks unification. If a phase cannot be made provably
safe, it does not ship — an elegant schema that loses one insight is a failure.
Three consequences shape everything below.

**The markdown files are permanent, not transitional.** Phase 2 makes the DB
canonical for *querying*, but the 1568 `.md` files keep being written on every
index, forever. They are the durable, greppable, `git`-able, human-readable
backstop: if `luke.db` is lost or a migration goes wrong, every memory body is
still on disk and `sync_memory_index()` rebuilds the database from it. Deleting
them would trade a real safety net for disk space Luke does not need.

**These invariants must hold at all times**, and are what the harness checks:

| # | Invariant | Today |
|---|-----------|-------|
| I1 | every `.md` file has a `memory_meta` row | ✅ 1568/1568 |
| I2 | every `memory_meta` row has a `.md` file | ❌ 1 orphan |
| I3 | every active memory has an FTS row | ✅ 791/791 |
| I4 | every active memory has a vector | ✅ 791/791 |
| I5 | no archived memory is in FTS | ✅ 0 |
| I6 | file body sha == indexed body sha | ❌ 15 stale |
| I7 | a goal's status has exactly one answer | ❌ 3 answers, disagreeing |
| I8 | `recall(mem_type=T)` is non-empty for every type with active members | ❌ goals return 0 |

**Nothing is ever deleted to complete a migration.** Old tables are renamed
`_v1` and kept for a full release cycle. Old files are never removed. Every
phase's rollback is a config flip, not a restore.

### What could corrupt memory, and what stops it

| Risk | Guard |
|------|-------|
| backfill misparses a body and truncates it | sha every body pre- and post-migration; counts and hashes must match exactly, or abort |
| external-content FTS rebuild loses documents | build the new index alongside the old, diff ranked ids over a fixed query set, promote only on parity |
| write-through export fails silently, files drift from rows | I6 checked hourly; a mismatch re-indexes from disk and logs |
| migration runs against the live DB mid-write | migrations run on a snapshot copy first; live run only after the copy verifies clean |
| a phase lands and degrades recall in a way no invariant catches | freeze a 50-query golden set with expected top-5 ids *before* Phase 1; regressions fail the gate |

## 5. Target schema

### Phase 1 — goals and plans

```sql
-- Lifecycle of the OBJECTIVE, distinct from retrievability of its document.
CREATE TABLE goals (
    id        TEXT PRIMARY KEY REFERENCES memory_meta(id),
    lifecycle TEXT NOT NULL DEFAULT 'active',  -- active|paused|blocked|completed|abandoned
    progress  REAL NOT NULL DEFAULT 0.0,
    deadline  TEXT,
    updated   TEXT NOT NULL
);
CREATE INDEX idx_goals_lifecycle ON goals(lifecycle);

CREATE TABLE plans (
    goal_id      TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'in_progress',
    body         TEXT NOT NULL DEFAULT '',   -- narrative: landscape, blockers, build log
    created      TEXT NOT NULL,
    last_updated TEXT NOT NULL
);

CREATE TABLE plan_steps (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id      TEXT NOT NULL REFERENCES plans(goal_id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    description  TEXT NOT NULL,
    done         INTEGER NOT NULL DEFAULT 0,
    completed_at TEXT
);
CREATE INDEX idx_plan_steps_goal ON plan_steps(goal_id, seq);
```

`plans.goal_id` is intentionally **not** a foreign key to `goals` — six current
plans have no goal, and that is legitimate (a plan can precede its goal). The
reconciler reports the gap instead of the schema forbidding it.

### Phase 2 — memory content

```sql
ALTER TABLE memory_meta ADD COLUMN title    TEXT NOT NULL DEFAULT '';
ALTER TABLE memory_meta ADD COLUMN content  TEXT NOT NULL DEFAULT '';
ALTER TABLE memory_meta ADD COLUMN body_sha TEXT NOT NULL DEFAULT '';
```

Then rebuild FTS as an external-content table so the body exists **once** in the
database:

```sql
CREATE VIRTUAL TABLE memory_fts USING fts5(
    title, content, tags,
    content='memory_meta', content_rowid='rowid',
    tokenize='porter unicode61'
);
```

> **Hazard.** External-content FTS keys on `rowid`, but `index_memory()` uses
> `INSERT OR REPLACE INTO memory_meta` ([memory.py:566](src/luke/memory.py:566)),
> which deletes and reinserts the row — assigning a **new rowid** every update
> and silently orphaning that memory's FTS entry. This must be converted to
> `INSERT … ON CONFLICT(id) DO UPDATE` *before* the FTS rebuild, or Phase 2
> quietly corrupts search on every subsequent memory update. Add it to I3 so the
> harness would catch a regression.

Markdown becomes a **write-through export**, not the source of truth: every
`index_memory()` writes the row, then renders the file. `body_sha` is the
reconciliation key — on startup, a file whose sha differs from the row was edited
out of band, so **disk wins** and the row is re-indexed. That preserves the "open
a memory in my editor" affordance and closes the §2.4 drift, in both directions.
Per §4 the files are kept permanently, so `luke.db` remains reconstructible from
disk at any point.

### Phase 3 — claims

```sql
CREATE TABLE work_claims (
    goal_id    TEXT PRIMARY KEY,
    token      TEXT NOT NULL,
    holder     TEXT NOT NULL,
    pid        INTEGER,
    host       TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    expires_at REAL NOT NULL
);
```

The whole `O_EXCL` + read + staleness + unlink dance collapses to one atomic
statement:

```sql
INSERT INTO work_claims (...) VALUES (...)
ON CONFLICT(goal_id) DO UPDATE SET
    token=excluded.token, holder=excluded.holder, pid=excluded.pid,
    host=excluded.host, claimed_at=excluded.claimed_at, expires_at=excluded.expires_at
WHERE work_claims.expires_at < :now
RETURNING token;
```

`RETURNING` empty means a live peer holds it. **The fail-open contract must
survive the port**: `claim()` today grants on every error path, because a false
denial before a deadline is worse than a double-grant on an advisory lock. A
`sqlite3.OperationalError` (locked DB) must therefore still grant, not raise.
Keep pid-liveness reclaim as an extra `OR (host = :host AND pid = :dead_pid)`
predicate.

### Phase 4 — fold in the strays

`crashes/*.json`, `workspace/activity_log.jsonl` and `scorecard_data.json` all
become `events` rows (the table already exists, 14565 rows, with `event_type` +
JSON `payload`). Delete the 0-byte `dashboard.db`.

## 6. Migration plan

Each phase is one migration entry in `_MIGRATIONS` ([db.py:297](src/luke/db.py:297)),
next version **15**, plus a backfill and a parity check. Ship them one at a time;
each is independently revertable.

### Phase 0 — stop the bleeding (data fix, no schema change)

The goal system is dead *right now*, and it stays dead for as long as the
migration takes. Fix the data first:

1. Re-activate the goal memories that are genuinely still live —
   `goal-msc-cognitive-science`, `goal-voicebox-luke-voice`, `goal-new-job` and
   `goal-luke-v16-close-the-loops` are `paused`/`blocked` in their bodies, not
   finished.
2. Create the missing goal memory for `cargurus-ramp`, or retire the plan.
3. Confirm `recall(mem_type='goal')` is non-empty before anything else lands.

This is a judgment call about which objectives are real, so it needs Filipe's
input — not a script.

### Phase 0.5 — ship the invariant harness *before* any schema change

Nothing structural moves until drift is detectable. This phase writes no
migration; it makes I1–I8 continuously enforced, so every later phase lands
against a system that reports its own corruption.

1. `src/luke/integrity.py` — one function per invariant, each returning
   offending ids. The three audit scripts written while investigating this
   (file↔meta↔FTS↔vec parity, goal three-way status agreement, disk-newer-than-
   index) are the starting point.
2. Wire it into the hourly maintenance pass in `scheduler.py`. I1–I6 auto-heal
   where healing is unambiguous (re-index from disk, drop a stale FTS row);
   anything ambiguous pages Filipe rather than guessing.
3. `tests/test_integrity.py` — each invariant gets a test that corrupts a
   fixture DB and asserts detection.
4. Freeze the 50-query golden recall set with expected top-5 ids. This is the
   regression baseline for Phases 1–2 and has to exist before they start.

Standalone value: even if the rest is never built, memory drift stops being
something discovered by accident months later.

### Phase 1 — goals and plans (the payload)

1. Migration 15 creates `goals`, `plans`, `plan_steps`.
2. Backfill: parse each goal body's `<b>Status:</b>`/`<b>Progress:</b>` and each
   `plans/*.md` `**Status:**`/`**Last updated:**`/`- [ ]` **once**, into rows.
   Where the three sources disagree (§2.2), the *plan file* wins — it is the most
   recently touched — and every conflict is logged for review.
3. Replace all five regex parsers with row reads. Delete `_parse_plan_status`,
   `_plan_last_updated`, `_plan_next_step`; rewrite `enforce_plan_momentum` and
   `reconcile_stale_plans` as SQL.
4. Add MCP tools `create_plan` / `update_plan_step` / `set_goal_lifecycle` so the
   agent writes rows instead of markdown. Update `LUKE.md` §207 and §248, which
   currently instruct it to write `workspace/plans/{goal_id}.md` by hand.
5. `plans/*.md` becomes a rendered export for the dashboard and for reading.
6. Decouple lifecycle from retrievability: `recall(mem_type='goal')` joins
   `goals.lifecycle NOT IN ('completed','abandoned')` rather than depending on
   `memory_meta.status`.

### Phase 2 — memory content

1. Migration 16 adds `title`/`content`/`body_sha`; backfill from the 1568 files.
2. Rebuild `memory_fts` as external-content, reindex, verify hit-parity on a
   fixed query set **before** dropping the old table.
3. `index_memory()` writes row → renders file. `sync_memory_index()` becomes a
   sha-based two-way reconcile.
4. Fix the one orphan (`insight-self-modification-crash-loop`) and the 15 stale
   files as part of the backfill.

### Phase 3 — claims

Port `work_claim.py` to the table. Keep the module's public API (`claim()`,
`release()`, `Claim`) byte-identical so callers do not change. Its existing
tests are the contract — they must pass unmodified.

### Phase 4 — strays and dashboard

Fold `crashes/`, `activity_log.jsonl`, `scorecard_data.json` into `events`.
Point `server.py` at `luke.db` only; delete `dashboard.db`, `DB_PATH`'s
markdown-scanning siblings, and the `api_plans` regex block.

## 7. Per-phase gate

No phase is done until all five hold. Any failure reverts the phase.

1. **Snapshot taken** — `backups/luke-pre-phaseN.db`, and the migration ran
   clean on that copy before touching the live DB.
2. **Invariants green** — I1–I8 pass after the phase, and no invariant that was
   green before is red after.
3. **Golden set unchanged** — the 50 frozen recall queries return the same top-5
   ids as the Phase 0.5 baseline. Ranking may shift; membership may not.
4. **Body hashes identical** — for any phase touching content, all 1568 body
   shas match pre-migration values. Mismatch aborts.
5. **Old store intact** — `_v1` tables retained, markdown files untouched,
   rollback verified by actually running it once on the snapshot.

## 8. Expected end state

- Ten substrates → four: `luke.db`, config files, supervision files, blobs.
- One canonical copy of every memory body, with markdown as a derived view.
- Zero regexes over prose in control flow.
- Goal lifecycle and document retrievability as separate, independently queryable
  facts — and a goal system that is actually running.
