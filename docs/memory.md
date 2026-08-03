# Memory

Hybrid knowledge graph with temporal decay. Memories are markdown files on disk, indexed in SQLite for retrieval, and automatically injected into agent prompts.

## Five Types

| Type | Directory | Lifecycle |
|------|-----------|-----------|
| **entity** | `entities/` | Evolves over time — people, projects, concepts |
| **episode** | `episodes/` | Accumulates, then fades or consolidates — events, decisions |
| **procedure** | `procedures/` | Stable — how-to knowledge, workflows |
| **insight** | `insights/` | Distilled from patterns — preferences, rules |
| **goal** | `goals/` | Active — objectives with deadlines and progress |

## Dual Storage

Each memory exists as a **markdown file** (YAML frontmatter + body) on disk and is indexed across **four SQLite structures**:

- **memory_fts** — FTS5 virtual table for lexical search (Porter stemming, Unicode tokenization)
- **memory_meta** — metadata table tracking type, timestamps, access count, useful count, importance, status, tags, links, privacy flag, and last accessed timestamp
- **memory_vec** — sqlite-vec table storing 768-dim bge embeddings for semantic search
- **memory_links** — relationship graph with weighted, labeled edges between memories

Files are human-readable and agent-editable. SQLite provides retrieval.

## Retrieval Pipeline

`recall()` combines multiple strategies and merges results:

1. **FTS5 lexical search** — queries are sanitized (operators stripped) and joined with OR semantics so longer queries return MORE results, ranked by BM25. This prevents the implicit-AND paradox where more words = fewer matches.
2. **Semantic search** — the local bge embed server (:17595, `BAAI/bge-base-en-v1.5`) encodes the raw query over localhost HTTP, sqlite-vec finds nearest neighbors via KNN. One canonical model copy serves every consumer; if the server is unreachable, recall degrades to FTS-only and the hourly `backfill_missing_embeddings()` pass heals missing vectors.
3. **Reciprocal Rank Fusion** (k=60) — merges FTS5 and semantic rankings without requiring score normalization. Memories ranked highly by both methods get the highest combined scores.
4. **Temporal filter** — optional date range on `updated` timestamp
5. **Graph traversal** — optional BFS from a related memory (depth 2, exponential weight decay per hop)

## Composite Scoring

After retrieval, each memory is scored on four weighted dimensions:

| Factor | Weight | Source |
|--------|--------|--------|
| **Relevance** | 0.4 | FTS/semantic/RRF match quality |
| **Importance** | 0.25 | Agent-set 0.1–2.0, normalized to [0,1] by `importance_score()`, decayed hourly |
| **Recency** | 0.2 | Exponential decay, 30-day time constant |
| **Access** | 0.15 | Logarithmic function of access count |

Relevance gates context quality — a low-relevance memory scores low regardless of importance. Non-query results (temporal, graph) are dampened to 0.3× context score. Weights must sum to 1.0 (validated at startup).

Importance is stored on a 0.1–2.0 scale and divided by that ceiling for scoring, so the whole range discriminates.

It used to be *clamped* at 1.0 instead. That made it an identical constant for 85% of the corpus — and its weight is highest (0.40) under the `factual` taxonomy that covers 84.5% of memories — so the factor nominally worth a quarter to two-fifths of the score did no ranking work at all. Both rankers now call the same `memory.importance_score()`; before, `context.py` divided by 2.0 while `memory.py` clamped, so the same memory carried two different importances depending on which layer surfaced it.

Percentile normalization was considered and rejected: the top fifth of the corpus was one minting cohort whose internal spread is decay noise correlated with access count, so rank-normalizing it would have re-encoded the very popularity signal the utility gate exists to damp.

### Taxonomy

`memory_meta.taxonomy` (`factual` / `experiential` / `working`, defaulted per type) selects the weight split — `factual` is importance-heavy (0.40/0.10/0.10), `experiential` recency-heavy, `working` recency-dominant. It also modulates decay: working memories decay 3× faster, factual at half rate.

## Utility Tracking

The system distinguishes two kinds of memory access:

- **Intentional access** — agent explicitly calls `recall`, `connect`, or `remember` with links. Increments both `access_count` and `useful_count`.
- **Speculative access** — auto-injection surfaces memories based on message text. Increments `access_count` only.

The **utility gate** (`memory.utility_factor()`) multiplies the *final* composite score, from `utility_floor` (0.5) up to 1.0. Utility can demote, never promote — promotion is what built the rich-get-richer loop in the first place.

The observed rate is shrunk toward a prior (0.6) with a pseudocount of 20 accesses, so a brand-new memory scores exactly 1.0 and nothing is punished for lacking evidence. Only sustained, high-volume non-use demotes: `proc-trust-repair-commitment-delivery` (260 accesses, 15 uses) lands at 0.58.

It used to modulate `access_score` instead, which carries weight 0.10–0.15 — so the most it could move a final score was under 3%. That memory kept ranking into every injection despite a 6% hit rate. The gate now has real authority.

**Known bias:** `useful_count` is driven mainly by explicit `recall` calls, since retroactive credit needs the literal memory id to appear in the reply. The rate therefore reads closer to "deliberately re-fetched" than "was useful". The separation it produces matches human judgement on the real corpus, so it earns its place — `scripts/eval_injection.py` reports mean utility of the selected set as a standing watch metric.

**Response-aware upgrade:** After the agent responds, any injected memory whose id appears in the response text is credited with `useful_only=True` — raising `useful_count` without raising `access_count`. The scan covers *both* layers: a fact used from standing context earned nothing before, even though the ranker it feeds scores by exactly that signal. Speculative exposure touches stay scoped to the turn layer, because crediting exposure for having been injected is the loop the gate exists to break.

**Retrieval miss logging:** When the agent explicitly recalls memories not found in auto-injection, the query and missed IDs are logged to `recall_misses` for future analysis.

## Adaptive Forgetting

Hourly, each memory's importance decays at a type-specific rate:

| Type | Rate | 30 days (0 acc) | 90 days (0 acc) |
|------|------|-----------------|-----------------|
| entity | 0.9998 | 98.5% | 85.0% |
| episode | 0.999 | 48.7% | 11.5% |
| procedure | 0.9999 | 99.3% | 80.5% |
| insight | 0.9995 | 69.7% | 33.9% |
| goal | 0.9997 | 80.4% | 52.3% |

Decay is modulated by access count — implementing spaced repetition. At 0 accesses, the full rate applies. At 10 accesses, decay halves. At 100 accesses, decay is reduced ~90%. Higher access counts reduce effective decay rate.

## Graph

Memories connect via `memory_links` with labeled, weighted edges. Traversal is bidirectional BFS up to depth 2, with exponential weight decay per hop (default: 0.5× per hop). Recalling one memory surfaces its neighbors.

**Hebbian co-access strengthening:** when multiple linked memories are recalled together (intentional access), the link weight between them increases by 0.05 (capped at 5.0). Over time, frequently co-recalled memories develop stronger associations, improving graph-based retrieval quality. Link weights are preserved when re-linking — `INSERT OR IGNORE` semantics prevent resetting accumulated weight.

**Temporal Validity:** Links have `valid_from` and `valid_until` columns. `valid_from` is set when a link is created. `valid_until` is set when a link is invalidated (via `invalidate_link()` or the `connect` tool's `supersedes_rel` parameter). Graph traversal filters out expired links by default. Expired links are preserved for history — never deleted.

**Causal Relationships:** Standard labels: `related`, `involves`, `contributes_to`, `derived_from`, `uses`, `about`, `informed_by`, `supports`, `caused`, `supersedes`, `contradicts`, `blocked_by`, `enables`. Default labels are assigned based on type pairs (episode→entity = "involves", insight→entity = "about", etc.). Causal labels (`caused`, `derived_from`, `supersedes`, `contradicts`, `supports`, `blocked_by`, `enables`) are prioritized in graph traversal when the query implies causal intent ("why", "because", "reason").

## Injection

`context.assemble_context()` is the single decision point, called once per run
from `run_agent`. Everything below happens in one `asyncio.to_thread` hop, so a
slow corpus or a hung embed server cannot stall the event loop.

It emits **two blocks**, because they have different lifetimes:

| Block | Content | Placement | Lifetime |
|---|---|---|---|
| `system_block` | conversation-state, active attention, recent outputs, standing memory | system prompt | replaced every run |
| `turn_block` | recall hits, trigger-matched skills, ranked graph neighbours, repeat-question flag | user prompt | accumulates in the transcript, correctly |

Both blocks are framed as knowledge rather than voice, and both frames are
charged to the budget that renders them. The turn block's frame matters more
than it looks: it lands closer to the user's words than the persona does, so
unframed it was setting the register. See [persona.md](persona.md#where-the-register-is-actually-set).

Order of spend, with one `seen` set threaded through so nothing appears twice:

1. **Pinned** — conversation-state (trimmed to keep the *newest* exchange), attention, recent outputs. Not budgeted: continuity is not optional context.
2. **Turn evidence** — capped at 60% of budget so a run of long recall hits can never starve standing context. Procedures capped at 3; trigger-matched skills are exempt from the check but still counted, so a chosen procedure is never blocked while total share stays bounded. Each entry renders `[id] (type, 4 months ago) body` — the ranker decays by recency, and the age label is what lets the model see it.
3. **Standing memory** — spends the remainder under `_BACKGROUND_SPEC`, which is one table defining both what renders and what it costs. Eight insight slots are reserved for feedback insights: they are durable behavioural rules with old timestamps, so they lose the recency competition (measured 1 of 25 on the live corpus) despite being the encoded "what Filipe cares about".

One thing in the turn block is not a memory. `_repeat_note` compares the query
against the last 20 messages and flags a question Filipe is asking again within
20 minutes at ≥0.68 similarity. It is deliberately unbudgeted — one line, and
the turn it matters on is exactly the turn a budget squeeze would drop it. The
detection is fuzzy because the failure it catches was fuzzy: three rewordings of
one question in 80 seconds defeated exact matching, and Luke answered all three
by rewording the answer.

Only what actually **rendered** counts as injected — retrieval routinely produces
more candidates than fit (42 for one real query), and a candidate the budget
rejected must not earn an exposure touch or be excluded from the standing layer
for a slot it never occupied.

Before this was unified, the two layers were built in different modules and
neither could see the other: `app.process` built the turn prefix, `run_agent`
built the system block, and by then the prefix was already in the prompt. So
nothing deduped them, nothing shared a budget, and conversation-state was
injected twice on 61% of turns while also consuming one of only 8 recall slots.

Assembly never raises. Memory is an enhancement, and losing it must never cost
the caller its prompt — the old `asyncio.gather` in `process()` ran prompt
building beside recall without `return_exceptions`, so a transient embed timeout
discarded images, transcripts and the `msg:{id}` handles the reply tool needs.

Effort classification runs *before* assembly so injected context doesn't inflate
word count. Budgets are true rendered tokens: 2,500 / 4,000 / 6,000 / 8,000 by
effort tier, 4,000 for autonomous runs.

## Conflict Detection & Version History

When updating an entity, `detect_changes()` compares old and new content and reports what changed back to the agent. Changes are recorded in the `memory_history` table, creating a timeline of how entities evolved. The `memory_history` tool lets the agent query "when did X change?"

**Overlap Detection:** When saving an insight or entity, `find_similar()` checks for semantically similar existing memories of the same type. Results surfaced to the agent in the `remember` tool response. Agent decides: merge, archive old + supersedes link, or keep both.

## Consolidation

Daily behavior clusters related episodes (≥2 shared tags or ≥2 shared links) and asks the agent to synthesize insights. See [autonomous behaviors](autonomous-behaviors.md).

**Insight Consolidation:** Weekly behavior detects clusters of semantically similar non-feedback insights via KNN (sqlite-vec). Synthesizes clusters into authoritative consolidated insights with `derived_from` links. Archives fragments.

**Feedback Consolidation:** Monthly behavior targets `feedback-*` insights specifically. Consolidates into a structured user preferences entity organized by category. Archives individual feedback fragments.

## Self-Healing

`sync_memory_index()` runs on startup, scanning `memory/` for unindexed files and indexing them with embeddings.

## Archiving

- **`forget`** tool → sets `status = 'archived'` (file stays on disk, excluded from queries)
- **`restore`** tool → reverses archiving
- **Weekly auto-prune** → archives episodes older than 5 years with importance below 0.1
- **Hourly cleanup** → removes archived entries from FTS index

## Lifecycle Review

Monthly behavior cross-references stale memories with recent activity. Flags: entities not updated in 90 days (with episode mention count), procedures not accessed in 60 days, completed goals still active. Agent reviews and takes action (update, archive, extract lessons).
