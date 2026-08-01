# Phase 2.1 — Luke memory model → Letta mapping

**Branch:** `letta-integration` · **Status:** design doc (Phase 2.1 of `goal-letta-full-power`)
**Written:** 2026-07-31, from ground truth (`src/luke/memory.py`, `docs/letta-migration-notes.md`,
`scripts/letta_import.py`, `openapi_letta.json`). No narration — every field below is a real
column or a real Letta primitive.

## Purpose
The importer (`letta_import.py`) already flattens **every** memory to an archival passage +
metadata. That is correct for *bulk load and recall parity* (Phase 0/1) but it is NOT the full
model: it throws away the one thing the migration exists for — **self-editing working memory**
(core blocks) — and it does not yet give `remember/forget/connect` a Letta equivalent (Phase 2.2).
This doc fixes the target: which memory lands where, and why. It is the contract Phase 2.2/2.3
build against.

---

## 1. Luke's source model (verified)

`memory_meta` columns (from `src/luke/memory.py` + migration notes):

| field | type | meaning | recall relevance |
|---|---|---|---|
| `id` | text, kebab | stable identity (`person-filipe`) | primary key across stores |
| `type` | enum | entity / episode / insight / procedure / goal | routes to target primitive |
| `content` | text (FTS) | the memory body | embedded → recall surface |
| `title` | text (FTS) | short label | embedded WITH content (see 2.3) |
| `importance` | float 0.1–2.0 | core-fact ↔ ephemeral | selects core-block vs archival |
| `tags_json` | json[] | 2–5 searchable tags; `private`→`is_private` | metadata filter |
| `links_json` | json[] | outbound graph edges (ids) | graph traversal |
| `taxonomy` | enum | factual / experiential / working | recall decay rate |
| `status` | enum | active / archived / paused | 900 active-content vs 635 tombstones |
| `is_private` | 0/1 | derived from `"private" in tags` | recall gating |
| `access_count`, `useful_count` | int | usage signal | ranking / decay modulation |
| `cluster_id` | int | insight-consolidation cluster | dream loop grouping |
| `created`, `updated`, `last_accessed` | ts | provenance / recency | recency scoring |

**Type distribution (active):** insight 506, procedure 204, entity 26, episode 26, goal 1.
Total rows 1535 → 900 content-bearing + 635 metadata-only tombstones.

**Graph (`memory_links`, 6912 rows):** labeled, directed, weighted, time-bounded (`valid_until`).
Labels split into **causal** (`caused, derived_from, supersedes, contradicts, supports,
blocked_by, enables`) — prioritized in "why" traversal — and **contextual** (`related, involves,
contributes_to, uses, about, informed_by`). Default label is chosen by `(from_type, to_type)`
pair via `_DEFAULT_RELATIONSHIP`.

**Taxonomy defaults per type:** entity/procedure/insight → `factual`; episode → `experiential`;
goal → `working`. Taxonomy drives recall decay: factual is durable, working decays fastest.

---

## 2. Letta target primitives (verified against `openapi_letta.json`)

- **Core-memory blocks** — `GET/PATCH /v1/agents/{id}/core-memory/blocks/{label}`, attach/detach.
  Always-in-context, agent-self-editable. Char-limited (`limit` field, default ~5000/block);
  total in-context budget is small. **This is the whole reason for the migration.**
- **Archival memory** — `GET /v1/agents/{id}/archival-memory/search`, `POST /v1/passages/search`,
  per-memory CRUD, `POST /v1/archives/{archive_id}/passages/batch`. Unbounded, embedding-searched.
  Maps 1:1 to Luke's `recall`.
- **Passage metadata** — arbitrary JSON per passage. Carries everything core-block/archival
  structure can't (importance, tags, links, taxonomy, counts, provenance).

Letta has **no native typed graph edges** between passages. The 6912 links must live as passage
metadata (`links: [{to, label, weight, valid_until}]`) and be re-traversed application-side, OR
be re-derived at recall time. This is the one real modeling gap — addressed in §4.

---

## 3. The mapping (type → primitive)

Routing rule is **importance + type + status**, not type alone:

| source | target primitive | rationale |
|---|---|---|
| **goal** (active) | **core block** `label=goals` (packed) | always relevant; agent must self-edit progress/status; only 1 active — fits one block trivially |
| **entity**, importance ≥ 1.5 | **core block** `label=key-people` / `key-projects` | the load-bearing world-model (person-filipe, user-preferences, cargurus). Self-edited as facts change — the native win over the injected blob |
| **entity**, importance < 1.5 | archival + metadata | recalled on demand, not always in-context |
| **insight** | archival + metadata | 506 of them; too many for core; surfaced by recall/similarity. High-importance behavioral insights (the constitutional/feedback ones) → candidate for a small `operating-rules` core block (see §3.1) |
| **procedure** | archival + metadata | 204; retrieved when task matches `trigger_pattern` (skill_meta) |
| **episode** | archival + metadata | experiential log; recall + dream-loop consolidation source |
| **tombstone** (635, no content) | archival passage, placeholder text, full metadata, `is_tombstone=true` | provenance survives; excluded from active recall |

**Budget check:** core blocks would hold 1 goal + ~5 importance-≥1.5 entities + 1 rules block ≈
6–7 blocks. person-filipe and user-preferences are the largest (multi-KB); each may exceed a
single 5000-char block → split into `key-people/filipe-core` + `key-people/filipe-detail`, or
raise that block's `limit`. **Action for 2.2:** measure the 3 heaviest core candidates against the
block limit before load; don't assume they fit.

### 3.1 The `operating-rules` core block (design decision)
Luke's behavior is governed by a set of hard directives that currently live as high-importance
`feedback-*`/`insight-*` memories AND are duplicated in the injected blob (constitutional layer,
`feedback-stay-on-sqlite`, `feedback-never-send-emails`, etc.). In the Letta model these should be
a **single always-in-context core block** the agent may read but should NOT freely self-edit
(they're guardrails, not world-model). Letta blocks don't have per-block write-locks, so enforce
this by convention + a Phase 5 shadow check that diffs the block against a canonical copy.
This replaces "re-inject the constitutional layer every turn" with "it's just in core memory."

---

## 4. The graph problem (links → Letta)

6912 labeled edges, no native Letta edge type. Three options, ranked:

1. **Metadata + app-side traversal (chosen).** Each passage carries
   `links: [{to_id, label, weight, valid_until}]` (importer already stores `links`). Luke's
   existing `_get_neighbors_batch` / causal-filter traversal logic (already written in
   `memory.py`) runs unchanged against a metadata index instead of `memory_links`. Keeps the
   causal-vs-contextual distinction and time-bounding. **No semantic loss.**
2. Encode edges as their own passages (`edge:{from}:{label}:{to}`) — searchable but pollutes
   recall and doubles passage count. Rejected.
3. Drop to `related`-only flat links — loses causal traversal ("why" queries). Rejected; the
   causal set is load-bearing for dream/reflexion reasoning.

**Consequence:** the Letta backend keeps a thin `links` metadata index (or reads it from passage
metadata at traversal time). `connect()` (Phase 2.2) writes to that index; recall's causal
traversal reads it. The graph is preserved, just relocated from a SQL table to passage metadata.

---

## 5. Semantics preservation (`remember` / `forget` / `connect` → Phase 2.2 tool contract)

| Luke op | current SQLite behavior | Letta equivalent |
|---|---|---|
| `remember(entity/goal, imp≥1.5)` | upsert `memory_meta` + FTS + vec | PATCH the target **core block** (agent self-edit) OR create/update passage; route by §3 |
| `remember(insight/procedure/episode)` | upsert + auto-link by type pair | `POST` archival passage + write `links` metadata via `_DEFAULT_RELATIONSHIP` |
| `forget` | set `status=archived` (reversible; keeps row) | set passage metadata `status=archived` + `is_tombstone` semantics; DO NOT delete (restore must work) |
| `restore` | `status=active` | flip metadata back |
| `connect(from,to,label,weight)` | insert `memory_links` row | append to `from` passage's `links` metadata (or core block's link list) |
| `recall` | FTS + vec + graph rank | `archival-memory/search` (already routed via adapter) + core blocks always present + app-side graph rank |
| importance/taxonomy decay | ranking model in `recall` | preserved as metadata; ranking stays app-side (Letta doesn't model Luke's decay curve) |

**Key invariant:** Letta replaces the *store + self-edit surface*, NOT Luke's ranking/decay/graph
logic. That logic (`recall`, `_semantic_search`, `_get_neighbors_batch`, decay rates) stays in
`memory.py` and reads from Letta instead of SQLite. This is why the adapter pattern (Phase 0) is
the right seam — recall already fails-safe between backends.

---

## 6. What Phase 2.2 / 2.3 must build (this doc's output)

- [ ] **2.2a** Core-block packing: write the 1 goal + high-importance entities + `operating-rules`
      into named core blocks at load time (not flat passages). Measure the 3 heaviest vs block
      `limit`; split or raise limit as needed (§3).
- [ ] **2.2b** `links` metadata index + port `_get_neighbors_batch` traversal to read it; wire
      `connect()` to write it (§4).
- [ ] **2.2c** Map `remember/forget/restore` onto the routing table (§5) as Letta memory-edit
      tools; keep ranking/decay app-side.
- [ ] **2.3** Re-embed with **title+content** (importer currently embeds content-only per migration
      notes; Luke embeds title+content) → re-benchmark, target ≥5/6 top-1 (was 4/6).

**Accept (Phase 2):** every MCP memory op has a Letta equivalent; recall parity ≥ sqlite; graph +
causal traversal intact; no memory semantics lost. This doc is the design half; 2.2/2.3 are the build.

---

## 7. Open risk carried to the go/no-go
The Postgres/Docker runtime dependency (migration notes, "SERVER-BOOT FINDING") is unchanged by
this doc and still sits in tension with `feedback-stay-on-sqlite`. Nothing here commits the live
process — it defines the *target shape* so that when Filipe green-lights the cutover (post-Aug-10),
the model is fully specified and the only remaining work is mechanical load + shadow verify.
