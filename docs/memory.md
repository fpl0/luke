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
- **memory_vec** — sqlite-vec table storing 768-dim fastembed embeddings for semantic search
- **memory_links** — relationship graph with weighted, labeled edges between memories

Files are human-readable and agent-editable. SQLite provides retrieval.

## Retrieval Pipeline

`recall()` combines multiple strategies and merges results:

1. **FTS5 lexical search** — queries are sanitized (operators stripped) and joined with OR semantics so longer queries return MORE results, ranked by BM25. This prevents the implicit-AND paradox where more words = fewer matches.
2. **Semantic search** — fastembed (`BAAI/bge-base-en-v1.5`) encodes the raw query, sqlite-vec finds nearest neighbors via KNN. Uses asymmetric retrieval (`query_embed` vs `passage_embed`).
3. **Reciprocal Rank Fusion** (k=60) — merges FTS5 and semantic rankings without requiring score normalization. Memories ranked highly by both methods get the highest combined scores.
4. **Temporal filter** — optional date range on `updated` timestamp
5. **Graph traversal** — optional BFS from a related memory (depth 2, exponential weight decay per hop)

## Composite Scoring

After retrieval, each memory is scored on four weighted dimensions:

| Factor | Weight | Source |
|--------|--------|--------|
| **Relevance** | 0.4 | FTS/semantic/RRF match quality |
| **Importance** | 0.25 | Agent-set value (clamped to [0,1] in scoring), decayed hourly |
| **Recency** | 0.2 | Exponential decay, 30-day time constant |
| **Access** | 0.15 | Logarithmic function of access count, modulated by utility rate |

Relevance gates context quality — a low-relevance memory scores low regardless of importance. Non-query results (temporal, graph) are dampened to 0.3× context score. Weights must sum to 1.0 (validated at startup).

Importance can be stored above 1.0 (up to 2.0) to give important memories more decay runway, but is clamped to 1.0 during scoring so it doesn't dominate the other factors.

## Utility Tracking

The system distinguishes two kinds of memory access:

- **Intentional access** — agent explicitly calls `recall`, `connect`, or `remember` with links. Increments both `access_count` and `useful_count`.
- **Speculative access** — auto-injection surfaces memories based on message text. Increments `access_count` only.

The **utility rate** (`useful_count / access_count`) modulates the access score: memories with high utility (frequently used by the agent) get full credit, while memories that are frequently auto-injected but rarely used explicitly get a mild penalty (30% reduction). This closes the feedback loop — the scoring system learns which memories are actually useful.

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

## Auto-Injection

On every `process()` call:

1. Combine pending message text
2. Skip trivial messages (< 3 chars, common filler words)
3. Run hybrid recall in a thread (CPU-bound embedding)
4. Expand 1-hop graph neighbors from results
5. Track speculative access for all surfaced memories
6. Format as XML context block, prepend to prompt
7. Cache for 5 minutes with query-aware hashing (different queries bypass cache)

Effort classification runs *before* injection so injected context doesn't inflate word count.

## Conflict Detection & Version History

When updating an entity, `detect_changes()` compares old and new content and reports what changed back to the agent. Changes are recorded in the `memory_history` table, creating a timeline of how entities evolved. The `memory_history` tool lets the agent query "when did X change?"

## Consolidation

Daily behavior clusters related episodes (≥2 shared tags or ≥2 shared links) and asks the agent to synthesize insights. See [autonomous behaviors](autonomous-behaviors.md).

## Self-Healing

`sync_memory_index()` runs on startup, scanning `memory/` for unindexed files and indexing them with embeddings.

## Archiving

- **`forget`** tool → sets `status = 'archived'` (file stays on disk, excluded from queries)
- **`restore`** tool → reverses archiving
- **Weekly auto-prune** → archives episodes older than 5 years with importance below 0.1
- **Hourly cleanup** → removes archived entries from FTS index
