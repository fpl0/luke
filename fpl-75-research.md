# Memory Reconsolidation: Automatic Correction Detection and Memory Update Pipeline

## Executive Summary

Luke currently handles memory updates through explicit agent action only — the `remember` tool supports entity updates with basic string-diff change detection, but there is no automatic correction detection during conversation processing. This research delivers a complete design for automatic memory reconsolidation that detects when Luke or the user corrects a fact and updates stored memories without manual intervention.

**Key finding**: The most effective approach combines three mechanisms already partially present in Luke: (1) post-response analysis of recalled memory references to detect corrections, (2) semantic contradiction detection extending the existing `find_similar()` pipeline, and (3) typed conflict resolution (independent/extendable/contradictory) with version-chain preservation. The implementation effort is Low-Medium and can be delivered in 3 phases.

---

## 1. Current State Analysis

### What Exists

| Component | Location | Capability | Gap |
|-----------|----------|-----------|-----|
| `detect_changes()` | `memory.py:279` | String diff for title/content on entity updates | No semantic contradiction detection |
| `find_similar()` | `memory.py:1346` | Semantic overlap detection for insights/entities | Surfaces candidates but does not auto-resolve |
| `remember` tool | `agent.py:516-659` | Entity updates with conflict detection + history logging | Only for entities, agent-initiated only |
| `memory_history` table | `db.py:229-235` | Change log with JSON change descriptions | Stores descriptions, not full content snapshots |
| `touch_memories()` | `memory.py:1071` | Tracks access_count, useful_count, Hebbian links | No correction-aware scoring |
| Consolidation behaviors | `behaviors.py:87-563` | Weekly dedup, feedback synthesis, lifecycle review | No contradiction scanning pass |
| `get_factual_duplicate_candidates()` | `memory.py:423` | Finds semantically similar factual memories | Dedup only, not contradiction detection |

### What Does Not Exist

- **Real-time correction detection** during conversation processing
- **Semantic contradiction detection** — only exact string comparison exists
- **Automatic entity updating** when the agent states a correction
- **Correction-aware scoring** — the spec mentions a feedback bonus (F_i) but it is not implemented
- **Typed conflict resolution** — no classification of relationships as independent/extendable/contradictory

### Insertion Points in Current Architecture

The ideal insertion points for correction detection, in priority order:

1. **Post-agent response analysis** (`app.py:461-467`): After the agent responds, the code already identifies which recalled memory IDs were referenced. This is the ideal spot to also analyze whether the agent corrected any recalled information.

2. **Within the `remember` tool** (`agent.py:530`): Already has `detect_changes()` for entities — could be extended to detect corrections in any memory type with semantic contradiction detection.

3. **Stop hook** (`agent.py:938`): Already instructs the agent to update entities that changed — could add explicit correction detection instructions.

4. **`_save_conv_state()`** (`app.py:557`): Runs after each conversation, extracts topics and pending actions — could also extract corrections.

---

## 2. Correction Detection Design

### 2.1 Detection Mechanisms (Three-Layer Approach)

#### Layer 1: Explicit Correction Detection (Low Effort, Immediate)

Detect explicit correction patterns in the agent response after memory recall:

```python
# Insert in app.py after line 467 (touch_memories call)
def detect_corrections(recalled_memory_ids, agent_response, conversation_messages):
    """Detect when the agent corrects information from recalled memories."""
    corrections = []
    recalled_memories = memory.get_memories_by_ids(recalled_memory_ids)
    
    for mem in recalled_memories:
        # Check if agent response contradicts this memory
        is_contradiction = llm_classify_relationship(
            mem.content, agent_response, "contradiction"
        )
        if is_contradiction:
            # Extract the corrected content from the response
            corrected_content = extract_correction(agent_response, mem)
            corrections.append({
                "memory_id": mem.id,
                "original": mem.content,
                "corrected": corrected_content,
                "confidence": is_contradiction.confidence,
                "source": "agent_response"
            })
    
    return corrections
```

**Correction signal patterns to detect**:
- Explicit: "that is not right", "actually", "I was wrong", "correction", "let me correct"
- Implicit: Agent provides different factual information than what was recalled
- User-initiated: User says "no, X is actually Y", "that is outdated", "I changed"

#### Layer 2: Semantic Contradiction Detection (Medium Effort)

Extend `find_similar()` to classify relationships, not just surface candidates:

```python
# Extend memory.py find_similar() or add new function
def detect_contradictions(new_content, existing_memories, threshold=0.4):
    """Classify relationship between new content and existing memories.
    
    Returns typed relationships:
    - independent: No meaningful overlap (similarity < 0.3)
    - extendable: Complementary information (0.3 <= similarity <= 0.7)
    - contradictory: Conflicting information (similarity > 0.7 + LLM verification)
    """
    results = []
    for existing in existing_memories:
        semantic_sim = cosine_similarity(
            embed(new_content), existing.embedding
        )
        
        if semantic_sim < 0.3:
            relationship = "independent"
        elif semantic_sim > 0.7:
            # High overlap — use LLM to verify contradiction
            relationship = llm_verify_contradiction(
                existing.content, new_content
            )
        else:
            relationship = "extendable"
        
        results.append({
            "memory_id": existing.id,
            "similarity": semantic_sim,
            "relationship": relationship,
            "existing": existing.content,
            "new": new_content
        })
    
    return results
```

#### Layer 3: Retrieval-Failure-Triggered Reconsolidation (Future)

Inspired by HiMem architecture: when recall returns low-confidence results for a query the agent believes should have an answer, trigger a deeper search and use findings to update Note Memory.

**Defer to Phase 3** — requires episodic memory layer that Luke does not yet have.

### 2.2 Correction Confidence Scoring

Not all corrections should trigger automatic updates. Use a confidence threshold:

```python
def compute_correction_confidence(correction):
    """Score how confident we are that this is a genuine correction."""
    signals = {
        "explicit_language": 0.3,    # "actually", "correction", etc.
        "source_reliability": 0.3,   # user_direct=1.0, agent_self=0.8, inferred=0.5
        "semantic_strength": 0.2,    # How strong is the contradiction signal
        "temporal_recency": 0.2,     # Newer information is more likely correct
    }
    
    score = sum(signals[k] * correction.get(k, 0) for k in signals)
    return score

# Thresholds:
# >= 0.7: Auto-update (with history logging)
# 0.5-0.7: Flag for agent review in next conversation
# < 0.5: Log but take no action
```

---

## 3. Auto-Update Pipeline Design

### 3.1 Update Flow

```
Conversation → Agent Response → Correction Detection
                                        ↓
                              Confidence Scoring
                                        ↓
                         ┌──────────────┼──────────────┐
                         ↓              ↓              ↓
                   Auto-Update    Flag for Review   Log Only
                   (>= 0.7)       (0.5-0.7)        (< 0.5)
                         ↓              ↓              ↓
                   ┌─────┴─────┐        │              │
                   ↓           ↓        │              │
              Update DB    Add to     Store in      Store in
              Re-embed     review     pending       pending
              Log history  queue      corrections   corrections
                   ↓           ↓        ↓              ↓
              Notify agent  Next conv  Next conv     Weekly
              if critical   triggers   triggers      consolidation
```

### 3.2 Code Changes: `app.py`

Insert correction detection after the existing `touch_memories()` call:

```python
# app.py — after line 467 (existing touch_memories call)
# NEW: Correction detection pipeline
if recalled_memory_ids and response_text:
    corrections = detect_corrections(recalled_memory_ids, response_text, messages)
    for correction in corrections:
        confidence = compute_correction_confidence(correction)
        if confidence >= 0.7:
            # Auto-update
            memory.apply_correction(
                correction["memory_id"],
                correction["corrected"],
                confidence=confidence,
                source="auto_detection"
            )
        elif confidence >= 0.5:
            # Flag for review
            memory.flag_for_review(
                correction["memory_id"],
                correction["corrected"],
                confidence=confidence
            )
```

### 3.3 Code Changes: `memory.py`

New functions to add:

```python
def apply_correction(mem_id, corrected_content, confidence, source="auto_detection"):
    """Apply a detected correction to a memory.
    
    1. Fetch current memory content
    2. Classify relationship (extendable vs contradictory)
    3. If extendable: merge content
    4. If contradictory: supersede with version chain
    5. Re-embed and re-index
    6. Log to memory_history
    """
    existing = get_memory_fts(mem_id)
    relationship = classify_relationship(existing.content, corrected_content)
    
    if relationship == "extendable":
        # Merge: append new information
        new_content = merge_content(existing.content, corrected_content)
    elif relationship == "contradictory":
        # Supersede: replace with new, link to old
        new_content = corrected_content
        # Create superseded link
        link_memories(mem_id, mem_id, "supersedes", weight=1.0)
    
    # Update FTS
    conn.execute("UPDATE memory_fts SET content = ? WHERE id = ?", 
                 (new_content, mem_id))
    
    # Re-embed
    embedding = embed(new_content)
    conn.execute("UPDATE memory_vec SET embedding = ? WHERE rowid = (
        SELECT rowid FROM memory_vec WHERE memory_id = ?)", 
        (embedding, mem_id))
    
    # Update metadata
    conn.execute("UPDATE memory_meta SET updated_at = ? WHERE id = ?",
                 (datetime.utcnow().isoformat(), mem_id))
    
    # Log to history
    record_memory_change(mem_id, [
        f"Auto-correction applied (confidence={confidence:.2f}, source={source})",
        f"Content: '{existing.content[:50]}...' -> '{new_content[:50]}...'"
    ])


def flag_for_review(mem_id, corrected_content, confidence):
    """Flag a potential correction for agent review."""
    # Store in a pending_corrections table (new)
    conn.execute(
        "INSERT INTO pending_corrections (mem_id, corrected_content, confidence, created_at) VALUES (?, ?, ?, ?)",
        (mem_id, corrected_content, confidence, datetime.utcnow().isoformat())
    )


def get_pending_corrections(mem_id=None, limit=10):
    """Get pending corrections for agent review."""
    if mem_id:
        return conn.execute(
            "SELECT * FROM pending_corrections WHERE mem_id = ? ORDER BY created_at DESC LIMIT ?",
            (mem_id, limit)
        ).fetchall()
    return conn.execute(
        "SELECT * FROM pending_corrections ORDER BY created_at DESC LIMIT ?",
        (limit,)\n    ).fetchall()


def classify_relationship(existing_content, new_content):
    """Classify relationship between existing and new content.
    
    Uses semantic similarity + LLM verification for high-overlap cases.
    Returns: 'independent', 'extendable', or 'contradictory'
    """
    existing_emb = embed(existing_content)
    new_emb = embed(new_content)
    sim = cosine_similarity(existing_emb, new_emb)
    
    if sim < 0.3:
        return "independent"
    elif sim > 0.7:
        # LLM verification for high-overlap cases
        return llm_classify_conflict(existing_content, new_content)
    else:
        return "extendable"
```

### 3.4 Code Changes: `agent.py`

Extend the `remember` tool stop hook to include correction detection:

```python
# agent.py — in the stop hook (line 938 area)
# Add to the existing "Did any entity change?" instruction:
"""
Did any recalled information get corrected during this conversation?
If so, use the remember tool to update the entity with the corrected content.
Corrections include: factual updates, changed preferences, outdated information.
"""
```

Also add a new `review_corrections` tool that surfaces pending corrections:

```python
# New MCP tool
def review_corrections(chat_id):
    """Review pending memory corrections detected automatically.
    
    Returns pending corrections with original content, proposed correction,
    and confidence score. Agent can approve, reject, or modify.
    """
    pending = memory.get_pending_corrections(limit=5)
    if not pending:
        return "No pending corrections to review."
    
    results = []
    for p in pending:
        results.append({
            "memory_id": p["mem_id"],
            "original": get_memory_content(p["mem_id"]),
            "proposed": p["corrected_content"],
            "confidence": p["confidence"],
            "detected_at": p["created_at"]
        })
    return json.dumps(results, indent=2)
```

### 3.5 New Database Schema

```sql
-- Pending corrections queue
CREATE TABLE IF NOT EXISTS pending_corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mem_id          TEXT NOT NULL,
    corrected_content TEXT NOT NULL,
    confidence      REAL NOT NULL,
    source          TEXT DEFAULT 'auto_detection',
    status          TEXT DEFAULT 'pending',  -- pending, approved, rejected, applied
    created_at      TEXT NOT NULL,
    resolved_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_pc_mem ON pending_corrections(mem_id);
CREATE INDEX IF NOT EXISTS idx_pc_status ON pending_corrections(status);

-- Enhancement to memory_history: store full content snapshots
-- (migration, not new table)
ALTER TABLE memory_history ADD COLUMN old_content TEXT;
ALTER TABLE memory_history ADD COLUMN new_content TEXT;
```

---

## 4. Conflict Resolution Design

### 4.1 Typed Conflict Resolution Framework

Based on belief revision theory (AGM framework) and HiMem research:

```
1. DETECT: On memory creation/update, run find_similar() with threshold
2. CLASSIFY: LLM classifies relationship (independent/extendable/contradictory)
3. RESOLVE:
   - Independent → ADD normally, no action needed
   - Extendable → UPDATE with merged content, keep history
   - Contradictory → 
     a. If new source is more reliable → supersede old, link with supersedes
     b. If uncertain → mark both with contradiction link, flag for review
     c. If temporal → newer supersedes older (default)
4. RECORD: All actions logged to memory_history with reasoning
```

### 4.2 Contradiction Handling Matrix

| Scenario | Detection | Resolution | Audit |
|----------|-----------|------------|-------|
| User corrects Luke directly | Explicit language + high confidence | Auto-update + supersede link | Full history log |
| Luke self-corrects | Agent response differs from recalled | Auto-update + supersede link | History + reflection |
| Conflicting memories discovered | Weekly consolidation scan | Flag for agent review | Contradiction link |
| Temporal update (outdated info) | `valid_until` expiry or newer info | Supersede with version chain | History log |
| Ambiguous contradiction | Medium confidence semantic overlap | Pending correction queue | Flag for review |

### 4.3 Source Reliability Scoring

```python
SOURCE_RELIABILITY = {
    "user_direct": 1.0,      # User explicitly states a fact
    "agent_self_correction": 0.85,  # Luke corrects its own mistake
    "agent_inferred": 0.6,   # Luke infers from context
    "external_tool": 0.5,    # Information from MCP tools
    "system_detected": 0.4,  # Automatic detection without explicit signal
}
```

---

## 5. Change Tracking Enhancement

### 5.1 Memory History Improvements

Current `memory_history` stores JSON change descriptions. Enhance to store:

```python
{
    "mem_id": "entity-filipe",
    "changed_fields": ["Title: 'Filipe location' -> 'Filipe current location'"],
    "timestamp": "2026-04-05T06:00:00",
    "old_content": "Filipe is in São Paulo",  # NEW
    "new_content": "Filipe moved to Lisbon",   # NEW
    "change_type": "correction",               # NEW: correction, update, extension
    "source": "auto_detection",                # NEW: auto_detection, agent_explicit, consolidation
    "confidence": 0.85,                        # NEW
    "supersedes_id": null                      # NEW: if this created a superseded link
}
```

### 5.2 Version Chain Visualization

The `memory_history` tool (already an MCP tool) should be enhanced to show the version chain:

```
entity-filipe-location:
  v1 (2026-03-01): "Filipe is in São Paulo" [archived]
    ↓ superseded by v2
  v2 (2026-04-05): "Filipe moved to Lisbon" [active]
    
  Change history:
    - 2026-04-05: Auto-correction detected (confidence=0.92, source=user_direct)
      Content: "Filipe is in São Paulo" -> "Filipe moved to Lisbon"
```

---

## 6. Implementation Plan

### Phase 1: Foundation (1-2 days)

**Goal**: Basic correction detection with explicit signals

1. Add `detect_corrections()` to `app.py` post-response pipeline
2. Add `classify_relationship()` to `memory.py`
3. Add `apply_correction()` to `memory.py` with auto-update logic
4. Create `pending_corrections` table (migration #10)
5. Extend `memory_history` with content snapshots (migration #10)
6. Enhance `remember` tool stop hook with correction instructions

**Files modified**: `app.py`, `memory.py`, `agent.py`, `db.py`

### Phase 2: Confidence and Review (1-2 days)

**Goal**: Confidence scoring and pending correction review

1. Implement `compute_correction_confidence()` with multi-signal scoring
2. Add `review_corrections` MCP tool
3. Add `flag_for_review()` and `get_pending_corrections()` to `memory.py`
4. Wire pending corrections into conversation context injection

**Files modified**: `memory.py`, `agent.py`, `app.py`

### Phase 3: Consolidation Integration (1 day)

**Goal**: Weekly consolidation includes contradiction scanning

1. Add contradiction scan pass to `run_consolidation()` in `behaviors.py`
2. Use `get_factual_duplicate_candidates()` extended with contradiction classification
3. Generate contradiction report for agent review

**Files modified**: `behaviors.py`, `memory.py`

---

## 7. Comparison with Existing Systems

| System | Correction Detection | Auto-Update | Conflict Resolution | Version History |
|--------|--------------------|-------------|-------------------|----------------|
| **Luke (current)** | None | Manual only | String diff only | Basic history log |
| **Luke (proposed)** | 3-layer detection | Confidence-gated | Typed (AGM framework) | Enhanced with snapshots |
| Letta/MemGPT | Agent-initiated only | Agent tool calls | None | None for archival |
| Mem0 | Automatic fact extraction | Automatic | Semantic dedup | Graph-based |
| HiMem (research) | Retrieval-failure triggered | Typed ADD/UPDATE/DELETE | AGM-based | Episodic immutable |
| Selective Memory | Write-time gating | Salience-scored | Version chains | Full version history |

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| False positive corrections corrupting memories | High | Confidence threshold (0.7) + version chains (reversible) |
| Performance impact on conversation latency | Medium | Async correction processing, not blocking |
| Contradiction detection false negatives | Low | Weekly consolidation catch-up scan |
| Memory history bloat | Low | Prune old snapshots (>30 days) during lifecycle review |
| Agent confusion from pending corrections | Medium | Surface max 5 pending per conversation |

---

## 9. Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Memory freshness (stale %) | Unknown | <5% stale | Lifecycle review |
| Correction detection rate | 0% | >80% of explicit corrections | Manual audit of conversations |
| False positive rate | N/A | <10% | Agent review of auto-updates |
| Memory contradiction count | Unknown | <2% of total memories | Weekly consolidation scan |
| Time to memory update | Manual (days) | Automatic (seconds) | memory_history timestamps |
