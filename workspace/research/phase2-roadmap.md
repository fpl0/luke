# Phase 2 Cognitive Architecture Roadmap

## Executive Summary

Luke has all the machinery for intelligence but none of the machinery for learning. Phase 1 built the foundation: memory system (FTS5 + semantic + graph + composite scoring + adaptive forgetting), autonomous behaviors, model routing, constitutional constraints, and Paperclip governance. Phase 2 must close the learning loop.

**The single highest-leverage investment**: Implement the reflexion system. The spec is written (workspace/research/self-improvement-spec.md), the infrastructure exists, and the gap is purely implementation. This transforms Luke from a capable assistant into a genuinely improving one.

---

## 1. Gap Analysis: Luke Today vs. Irreplaceable

### Current Capability Scorecard (updated from FPL-29)

| Dimension | Score | Status | Bottleneck |
|-----------|-------|--------|------------|
| Memory depth/recall | 6/10 | Foundation built | Flat retrieval degrades at scale; no clustering |
| Personality consistency | 7/10 | Constitutional anchors | No drift detection or correction |
| Proactive behavior | 5/10 | 8 behaviors exist | 60s polling latency; no event-driven triggers |
| Multi-modal | 6/10 | Photo/voice/video | No multi-modal embeddings; browser limited |
| Tool use breadth | 8/10 | 27 MCP tools | Good coverage; sub-agent scoping works |
| Context window management | 4/10 | Naive injection | No gating, no position-aware injection, no utilization tracking |
| Learning from feedback | 2/10 | Spec only | **Critical gap** - no reflexion, no closed loop |
| Emotional intelligence | 5/10 | Constitutional layer | No emotional memory tagging; no valence scoring |
| Task completion reliability | 4/10 | Single-process | FPL-37 addressing; no self-healing yet |
| Speed/latency | 6/10 | Model routing works | 60s scheduler tick; no async event bus |
| Privacy/data ownership | 9/10 | Local-first | SQLite, local files, no cloud dependency |
| Multi-platform reach | 4/10 | Telegram only | No web, no voice call, no email integration |
| Self-improvement | 1/10 | **Not implemented** | **Critical gap** - all pieces exist, none connected |
| Integration ecosystem | 3/10 | MCP tools only | No external API integrations beyond Telegram/Claude |
| User delight | 5/10 | Autonomous behaviors | Dream/deep work exist but unpredictable quality |

**Overall: 5.1/10** - Strong foundation, critical learning gap

### The Irreplaceability Threshold

For Luke to be truly irreplaceable (the system that runs someone's life), it must:
1. **Never make the same mistake twice** (reflexion)
2. **Know what it knows and does not know** (metacognition)
3. **Get better at its job without being told** (self-improvement loop)
4. **Remember what matters and forget what does not** (memory reconsolidation)
5. **Anticipate needs before they are expressed** (prospective memory + pattern prediction)

Luke currently achieves #4 partially. #1, #2, #3, and #5 are not implemented.

---

## 2. Phase 2 Priority Stack

### P0: Reflexion System (Self-Improvement Layer 1)

**Why first**: This is the missing link between capability and intelligence. Every other improvement compounds on top of it.

**What to build**:
1. **Post-action reflection**: After every significant tool use sequence, log what happened, what went wrong, what to do differently
2. **Error pattern detection**: Weekly behavior that scans reflection memories for recurring patterns
3. **Self-correction injection**: Feed top error patterns into the agent system prompt as "things to watch for"
4. **Execution trace logging**: Structured JSON traces of every agent run (tool calls, decisions, outcomes, costs)

**Spec reference**: `workspace/research/self-improvement-spec.md` (already written, needs implementation)

**Expected impact**: 30% reduction in repeated errors within 30 days. Luke starts learning from mistakes.

**Effort**: Medium. Memory tools exist. Behavior pattern exists. Spec is detailed. Missing piece: the wiring.

**Owner**: Engineer with Research oversight

---

### P1: Context Engineering - Retrieval Gating + Position-Aware Injection

**Why second**: Current context injection is wasteful and imprecise. Every message triggers recall regardless of need. This wastes tokens and introduces noise.

**What to build**:
1. **Retrieval gating**: Classify message effort before recalling. Skip recall for trivial greetings, acknowledgments, or when the agent parametric knowledge is sufficient
2. **Position-aware injection**: Place high-signal memories at the beginning and end of the prompt. Research (Liu et al., 2024) shows LLMs underweight middle context by 20-40%
3. **Context utilization tracking**: Log which injected memories appear in the agent response. Use this to tune retrieval weights
4. **Context compression**: For long conversations, summarize earlier context instead of truncating

**Expected impact**: 30% reduction in context token usage with equal or better task completion.

**Effort**: Low-Medium. Auto-injection code exists in `app.py`; needs gating logic and position-aware formatting.

**Owner**: Engineer

---

### P2: Semantic Clustering for Memory Organization

**Why third**: Flat retrieval will not scale. As memories grow from dozens to hundreds to thousands, the system needs structure.

**What to build**:
1. **Online clustering**: Assign new memories to nearest cluster at creation time (no full re-scan)
2. **Offline re-clustering**: During consolidation runs, re-cluster using HDBSCAN or similar
3. **Cluster-aware retrieval**: Search relevant clusters first, not the entire corpus
4. **Cluster summaries**: Each cluster has a summary insight that can be retrieved independently

**Spec reference**: `workspace/research/memory-consolidation-spec.md` (already written)

**Expected impact**: 40% faster retrieval for factual queries. Precision from ~60% to >80%.

**Effort**: Medium-High. New data structures in SQLite, recall pipeline changes, consolidation integration.

**Owner**: Memory Systems Engineer

---

### P3: Memory Reconsolidation

**Why fourth**: Memories become stale. When Luke corrects a fact during conversation, the correction should update the stored memory automatically.

**What to build**:
1. **Correction detection**: Monitor conversation for "that is not right", "actually", self-corrections
2. **Auto-update**: Trigger memory update with corrected content
3. **Re-embedding**: Update the vector store entry
4. **Change log**: Track memory corrections over time (already partially exists via memory_history)

**Expected impact**: Memories stay current. No accumulated contradictions.

**Effort**: Low-Medium. The `remember` tool already supports entity updates; needs automatic trigger.

**Owner**: Memory Systems Engineer

---

### P4: Event-Driven Architecture (Replace Polling)

**Why fifth**: The 60-second scheduler tick is a fundamental latency bottleneck.

**What to build**:
1. **Async event bus**: asyncio.Queue-based pub/sub
2. **Behavior subscriptions**: Behaviors subscribe to event types instead of polling
3. **Real-time triggering**: Immediate response with debounce/coalescing
4. **Event prioritization**: Critical events (user_message) bypass queue

**Expected impact**: Luke responds immediately instead of up to 60 seconds later. Enables true reactive behaviors.

**Effort**: Medium. Requires scheduler loop refactoring. Event infrastructure partially exists in `db.py`.

**Owner**: Engineer

---

### P5: Metacognition Layer

**Why sixth**: Luke needs to know what it knows and does not know. This is the foundation for reliable autonomous behavior.

**What to build**:
1. **Confidence scoring**: Agent rates its confidence in each response (1-5)
2. **Knowledge gap detection**: When confidence is low, trigger research sub-agent or ask user
3. **Capability self-assessment**: Before accepting a task, assess whether it has the tools/knowledge to complete it
4. **Uncertainty communication**: Express uncertainty to user instead of hallucinating

**Expected impact**: Fewer hallucinations. Better task routing. User trusts Luke more because it admits uncertainty.

**Effort**: Medium. Requires prompt engineering, new hooks, and confidence tracking in memory.

**Owner**: Head of AI Research (design) + Engineer (implementation)

---

## 3. Self-Improvement Loop Design

### The Learning Pipeline

```
Interaction -> Trace -> Reflect -> Detect Pattern -> Update Behavior -> Measure Impact
     |          |         |          |               |                |
     v          v         v          v               v                v
  User msg   JSON log   Reflection  Pattern     System prompt    Error rate
  Tool use   Tool call  memory      memory      Constitutional   Context quality
  Outcome    Cost       What failed Recurring   rule update      Task completion
             Decision              mistakes      Tool scoping     User satisfaction
```

### Weekly Learning Cycle

1. **Monday**: Trace analysis behavior scans last 7 days of execution traces
2. **Tuesday**: Reflection consolidation clusters reflection memories into patterns
3. **Wednesday**: Pattern-to-rule conversion: top 3 error patterns become system prompt additions
4. **Thursday**: Impact measurement: compare error rates before/after rule changes
5. **Friday**: Rule pruning: remove rules that no longer reduce errors

### Monthly Learning Cycle

1. **Capability audit**: What can Luke do now that it could not do last month?
2. **Memory quality review**: Are memories actually improving retrieval quality?
3. **Constitutional review**: Are constitutional rules still valid? Any need updating?
4. **User satisfaction**: Correlate behavioral changes with user feedback signals

---

## 4. Cross-Agent Learning: Paperclip -> Luke

The Paperclip org is building organizational intelligence. Luke needs individual intelligence. These should compound:

### Knowledge Flow

| Paperclip Output | Luke Input | Mechanism |
|-----------------|-----------|----------|
| Architecture decisions (FPL-37, FPL-45) | System configuration | Issue documents -> memory import |
| Failure patterns (reliability audit) | Error pattern database | Comments -> reflection memories |
| Best practices (test coverage) | Behavioral standards | Skills -> constitutional rules |
| Research findings (FPL-22, FPL-26, FPL-56, FPL-59, FPL-61) | Cognitive architecture updates | Research docs -> system prompt |
| World model facts | Entity memories | World model -> Luke memory sync |

### Implementation

Create a Paperclip routine (weekly) that:
1. Scans completed issues for architectural decisions and failure patterns
2. Converts them into Luke-compatible memory entries
3. Imports them into Luke memory system via the `remember` tool
4. Logs what was imported for audit trail

---

## 5. Concrete Engineering Specs

### Spec 1: Reflexion System (P0)

**Files to modify**:
- `src/luke/agent.py`: Add trace logging in `run_agent()`, add reflection hook after tool sequences
- `src/luke/memory.py`: Add `store_reflection()` function, add `get_error_patterns()` query
- `src/luke/behaviors.py`: Add `analyze_traces()` behavior (weekly)
- `src/luke/app.py`: Inject error patterns into system prompt before each `process()` call

**Data model**:
```yaml
# Reflection memory format
id: reflection-{timestamp}
type: insight
taxonomy: experiential
tags: ["reflexion", "error-pattern", "{tool_name}"]
error_type: hallucination | wrong_tool | missed_memory | timeout | other
context: "What was the agent trying to do?"
what_happened: "What actually occurred?"
what_should_have_happened: "What was the correct behavior?"
root_cause: "Why did this happen?"
fix: "What should the agent do differently next time?"
confidence: 0.8  # Agent confidence in its own analysis
```

**System prompt injection**:
```
<recent-error-patterns>
{Top 3 error patterns from last 7 days, formatted as actionable guidance}
</recent-error-patterns>
```

### Spec 2: Context Engineering (P1)

**Files to modify**:
- `src/luke/app.py`: Add `_classify_effort()` refinement for retrieval gating, modify `_auto_recall()` to accept skip flag
- `src/luke/agent.py`: Add context utilization tracking in response analysis

**Retrieval gating logic**:
```python
def should_recall(messages):
    combined = " ".join(m.content for m in messages)
    if len(combined) < 50:  # Short message
        return False
    if any(greeting in combined.lower() for greeting in ["thanks", "ok", "yes", "no"]):
        return False
    if _has_recent_context(chat_id, within_minutes=5):  # Context still fresh
        return False
    return True
```

**Position-aware injection**:
```python
def build_prompt_with_context(base_prompt, memories):
    # Split memories by score
    critical = [m for m in memories if m.score > 0.8]
    relevant = [m for m in memories if 0.5 < m.score <= 0.8]
    ambient = [m for m in memories if m.score <= 0.5]
    
    # Critical at beginning (highest attention)
    # Relevant in middle (standard)
    # Ambient at end (recency effect)
    return f"""
<critical-context>
{format_memories(critical)}
</critical-context>

{base_prompt}

<additional-context>
{format_memories(relevant)}
{format_memories(ambient)}
</additional-context>
"""
```

### Spec 3: Semantic Clustering (P2)

**Files to modify**:
- `src/luke/memory.py`: Add `_cluster_memories()`, `_get_cluster_for_memory()`, modify `recall()` to use cluster-aware retrieval
- `src/luke/db.py`: Add `memory_clusters` table

**New table**:
```sql
CREATE TABLE memory_clusters (
    cluster_id TEXT PRIMARY KEY,
    centroid_embedding BLOB,
    member_count INTEGER DEFAULT 0,
    summary TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE memory_cluster_members (
    cluster_id TEXT REFERENCES memory_clusters(cluster_id),
    memory_id TEXT REFERENCES memory_meta(id),
    membership_score REAL,
    PRIMARY KEY (cluster_id, memory_id)
);
```

---

## 6. What NOT to Build in Phase 2

- **Multi-agent governance**: Interesting but premature. Luke is single-agent; Paperclip handles org coordination.
- **Prospective memory**: Deferred to Phase 3. Current scheduler is sufficient for now.
- **Emotional memory tagging**: Deferred. Constitutional layer handles emotional intelligence adequately.
- **Cross-session context sharing**: Deferred for privacy. Each session should remain isolated.
- **Custom storage backend**: SQLite is sufficient. Do not over-engineer storage.
- **Graph visualization**: Nice to have but does not improve Luke intelligence.

---

## 7. Success Metrics

| Metric | Current | Phase 2 Target | Measurement |
|--------|---------|---------------|-------------|
| Repeated error rate | Unknown | <10% of previous week | Trace analysis |
| Context token efficiency | ~100% of recall injected | 70% of recall injected (30% gated) | Token count per message |
| Retrieval precision | ~60% | >80% | User feedback + utilization tracking |
| Memory freshness | Unknown | <5% stale memories | Lifecycle review |
| Task completion rate | Unknown | >90% | Scheduler task_logs |
| User satisfaction | Unknown | Improving trend | Reaction feedback + proactive scan |

---

## 8. Recommendations for FPL-43 (Decide Next Steps)

1. **Finish Phase 1 Paperclip** (FPL-47, FPL-48, FPL-49) - this is the infrastructure gate
2. **Merge FPL-37 reliability fixes** - uptime is the foundation for everything else
3. **Start P0: Reflexion System** - highest leverage cognitive architecture improvement
4. **Start P1: Context Engineering** - immediate efficiency gains
5. **Parallel: P2 Memory Clustering** - Memory Systems Engineer can start independently
6. **Defer: P3-P5** until P0-P2 are proven

The team should sequence: Infrastructure (Phase 1 Paperclip + FPL-37) -> Learning (Reflexion) -> Efficiency (Context Engineering) -> Scale (Clustering).

This is the path from capable assistant to genuinely intelligent system.
