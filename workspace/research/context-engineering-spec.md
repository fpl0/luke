# Context Engineering Spec for Luke

## Overview

This spec defines the context engineering system for Luke — the mechanisms that determine what information reaches the LLM, when, and in what form. Context engineering is the highest-leverage improvement area for Luke because it directly affects every single interaction: better context means better responses, fewer hallucinations, and more efficient token usage.

**Design principle:** Context is a scarce resource. Every token injected into the prompt costs money, reduces available space for reasoning, and adds noise. The system must be ruthless about signal-to-noise ratio.

---

## 1. Context Tier System

### Architecture

Luke's context is organized into three tiers, each with different inclusion rules, retrieval strategies, and token budgets.

```
┌─────────────────────────────────────────┐
│  TIER 1: CRITICAL (Always Included)    │  ~10-15% of context budget
│  - User identity & core preferences     │
│  - Active task context                  │
│  - Immediate conversation history       │
│  - Safety constraints                   │
├─────────────────────────────────────────┤
│  TIER 2: RELEVANT (Retrieved + Scored) │  ~60-70% of context budget
│  - Related memories (semantic search)   │
│  - Recent episodic memories             │
│  - Domain-specific knowledge            │
│  - Tool definitions                     │
├─────────────────────────────────────────┤
│  TIER 3: AMBIENT (On-Demand)           │  ~15-25% of context budget
│  - Historical patterns                  │
│  - Long-term preferences                │
│  - Reference documentation              │
│  - Fallback knowledge                   │
└─────────────────────────────────────────┘
```

### Tier 1: Critical (Always Included)

**Token budget:** 500-1000 tokens (fixed, never compressed)

**Contents:**
- User name, timezone, communication preferences
- Current active task/issue identifier and goal
- Last 3-5 conversation turns (full fidelity)
- Safety constraints and behavioral rules
- Current date/time and temporal context

**Injection rules:**
- ALWAYS included, regardless of context window size
- NEVER compressed or summarized
- Placed at the very beginning of the prompt (position bias optimization)

**Implementation:**
```python
# Pseudocode: context_tier_1()
def build_critical_context(user, conversation, active_task):
    return Template(
        user_identity=user.profile.summary,          # ~100 tokens
        active_task=active_task.brief,               # ~200 tokens
        conversation_history=conversation.last(5),   # ~400 tokens
        safety_rules=SAFETY_CONSTRAINTS,             # ~150 tokens
        temporal_context=current_datetime_info(),    # ~50 tokens
    )
```

### Tier 2: Relevant (Retrieved + Scored)

**Token budget:** Dynamic, 60-70% of remaining context after Tier 1

**Contents:**
- Memories retrieved via semantic search (top-k by relevance)
- Recent episodic memories (last 24-48 hours)
- Domain-specific knowledge relevant to current task
- Tool definitions needed for current action

**Retrieval strategy:**
1. Generate search query from current conversation + active task
2. Search memory store (hybrid: BM25 + semantic)
3. Score each result by:
   - Semantic similarity to query (0-1)
   - Recency bonus: `exp(-age_days / decay_half_life)`
   - Importance multiplier: `memory.importance_score`
   - Recency-access bonus: recently accessed memories get +0.1
4. Sort by composite score, fill budget greedily
5. Compress each item to token-efficient form

**Injection rules:**
- Included only if composite score > threshold (default 0.3)
- Ordered by score (highest first)
- Compressed using context compression pipeline (see Section 5)
- Placed after Tier 1, before Tier 3

**Implementation:**
```python
# Pseudocode: context_tier_2()
def build_relevant_context(query, budget_tokens):
    candidates = memory.search(
        query=query,
        limit=50,  # Over-fetch, then filter
        hybrid=True  # BM25 + semantic
    )
    
    scored = []
    for mem in candidates:
        score = (
            0.4 * mem.semantic_similarity +
            0.2 * recency_bonus(mem.created_at) +
            0.3 * mem.importance_score +
            0.1 * mem.access_recency_bonus
        )
        if score > 0.3:
            scored.append((score, mem))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    context = []
    used_tokens = 0
    for score, mem in scored:
        compressed = compress_memory(mem)
        token_cost = count_tokens(compressed)
        if used_tokens + token_cost <= budget_tokens:
            context.append(compressed)
            used_tokens += token_cost
        else:
            break
    
    return context, used_tokens
```

### Tier 3: Ambient (On-Demand)

**Token budget:** 15-25% of remaining context

**Contents:**
- Long-term behavioral patterns (e.g., "user prefers concise responses")
- Historical precedents for similar tasks
- Reference documentation (tool usage patterns, common workflows)
- Fallback knowledge when Tier 2 returns insufficient results

**Retrieval strategy:**
- Only populated if Tier 2 uses < 50% of its budget
- Retrieved using broader, less specific queries
- Lower score threshold (0.15 vs 0.3 for Tier 2)

**Injection rules:**
- Included only when Tier 2 is under-utilized
- Summarized heavily (1-2 sentences per item)
- Placed at the end of the context block

---

## 2. Position-Aware Context Injection

### Problem

Research (Liu et al., "Lost in the Middle") shows that LLMs pay disproportionate attention to content at the beginning and end of long contexts, with a significant drop-off in the middle. Naive context injection ignores this bias.

### Solution

Inject context using a **U-shaped attention pattern**:

```
Prompt Structure:
┌──────────────────────────────────────┐
│ [SYSTEM] Core instructions           │  ← High attention
│ [TIER 1] Critical context            │  ← High attention
│ [USER] Current query                 │  ← High attention
├──────────────────────────────────────┤
│ [TIER 2] Most relevant memories      │  ← Medium attention
│ [TIER 2] Moderately relevant         │  ← Lower attention
│ [TIER 2] Less relevant               │  ← Lowest attention (danger zone)
├──────────────────────────────────────┤
│ [TIER 3] Ambient patterns            │  ← Low attention
│ [TIER 1] Reinforced key facts        │  ← High attention (recency boost)
│ [INSTRUCTION] Response guidelines    │  ← High attention
└──────────────────────────────────────┘
```

### Implementation Rules

1. **Tier 1 at top**: User identity, active task, safety rules — always first
2. **Tier 2 by score**: Highest relevance first, declining
3. **Key facts repeated**: Critical facts from Tier 1 that are relevant to the current query are repeated at the end (recency boost)
4. **Instructions last**: Response formatting and behavioral instructions always at the very end
5. **No middle dumping**: If Tier 2 has many items, split them — top half before ambient, bottom half after

### Token Allocation Algorithm

```python
def allocate_context_positions(tier1, tier2, tier3, instructions, total_budget):
    # Reserve fixed allocations
    tier1_tokens = count_tokens(tier1)  # Fixed, never compressed
    instruction_tokens = count_tokens(instructions)  # Fixed
    
    remaining = total_budget - tier1_tokens - instruction_tokens
    
    # Tier 2 gets 70% of remaining, Tier 3 gets 30%
    tier2_budget = int(remaining * 0.7)
    tier3_budget = remaining - tier2_budget
    
    # Build context in position-aware order
    context = []
    context.extend(tier1)  # Position 1: high attention
    
    tier2_items = fill_budget(tier2, tier2_budget)
    context.extend(tier2_items)  # Position 2: declining attention
    
    if tier3_items := fill_budget(tier3, tier3_budget):
        context.extend(tier3_items)  # Position 3: low attention
    
    # Repeat critical facts relevant to query
    critical_facts = extract_relevant_facts(tier1, current_query)
    context.extend(critical_facts)  # Position 4: recency boost
    
    context.extend(instructions)  # Position 5: highest attention
    
    return context
```

---

## 3. Context Utilization Measurement

### Goal

Measure which injected context tokens actually contribute to responses. Without this measurement, we're flying blind on context quality.

### Method: Context Ablation Testing

**Procedure:**
1. For each response, log the full context that was injected
2. After the response, run an ablation analysis:
   - Remove each context item individually and re-generate the response
   - If the response changes significantly (semantic similarity < 0.9), that context item was used
   - Track which items are used vs. unused
3. Aggregate usage rates over time to identify:
   - High-value context sources (always used)
   - Low-value context sources (rarely used)
   - Optimal retrieval parameters (k, threshold, etc.)

### Metrics

| Metric | Formula | Target |
|---|---|---|
| Context utilization rate | `used_items / total_injected_items` | >80% |
| Token efficiency | `tokens_in_used_items / total_context_tokens` | >70% |
| Retrieval precision | `relevant_items / retrieved_items` | >85% |
| Context waste rate | `1 - utilization_rate` | <20% |

### Implementation

```python
# Pseudocode: context utilization tracking
class ContextUtilizationTracker:
    def __init__(self):
        self.injection_log = []  # (response_id, context_items)
        self.ablation_results = []  # (response_id, item_id, was_used)
    
    def log_injection(self, response_id, context_items):
        self.injection_log.append({
            "response_id": response_id,
            "items": [{"id": item.id, "tier": item.tier, "tokens": item.token_count} 
                      for item in context_items]
        })
    
    async def run_ablation(self, response_id, original_response, context_items):
        results = []
        for item in context_items:
            reduced_context = [i for i in context_items if i.id != item.id]
            ablated_response = await generate_with_context(reduced_context)
            similarity = semantic_similarity(original_response, ablated_response)
            was_used = similarity < 0.9
            results.append({"item_id": item.id, "was_used": was_used, "similarity": similarity})
        
        self.ablation_results.extend(results)
        return results
    
    def get_utilization_stats(self, window_days=7):
        recent = filter_by_date(self.ablation_results, window_days)
        total = len(recent)
        used = sum(1 for r in recent if r["was_used"])
        return {
            "utilization_rate": used / total if total > 0 else 0,
            "total_items": total,
            "used_items": used
        }
```

### Sampling Strategy

Full ablation is expensive (requires N+1 regenerations per response). Use sampling:
- Ablate 10% of responses randomly
- Always ablate responses with >20 context items (likely over-retrieval)
- Never ablate responses to time-sensitive queries (user waiting)

---

## 4. Retrieval Gating

### Problem

Not every query needs external context. LLMs have substantial parametric knowledge. Unnecessary retrieval adds noise, costs tokens, and increases latency.

### Solution

Implement a **retrieval gate** — a lightweight classifier that decides whether to retrieve context or rely on the model's internal knowledge.

### Gate Design

**Input features:**
- Query type classification (factual, creative, procedural, personal, analytical)
- Query specificity (number of named entities, dates, specific references)
- Conversation history length (longer conversations may need more context)
- Time since last retrieval (avoid rapid-fire retrievals)
- User's known preference for detail level

**Decision rules:**
```
IF query_type == "personal" AND references_user_history:
    RETRIEVE (high confidence)
ELIF query_type == "factual" AND is_common_knowledge:
    SKIP RETRIEVAL (rely on parametric knowledge)
ELIF query_type == "creative":
    SKIP RETRIEVAL (creativity benefits from less constraint)
ELIF query_type == "procedural" AND has_tool_definitions:
    RETRIEVE tool definitions only
ELIF conversation_turns_since_last_retrieval > 5:
    RETRIEVE (periodic context refresh)
ELSE:
    LIGHT RETRIEVAL (top-5 instead of top-20)
```

### Implementation

```python
class RetrievalGate:
    def __init__(self):
        self.common_knowledge_cache = set()  # Cached factual queries
        self.retrieval_history = []  # Track recent retrievals
    
    def decide(self, query, conversation_context):
        features = self.extract_features(query, conversation_context)
        
        # Fast-path: check cache
        if query.normalized in self.common_knowledge_cache:
            return GateDecision.SKIP
        
        # Rule-based decision
        decision = self.apply_rules(features)
        
        # Log for learning
        self.log_decision(query, features, decision)
        
        return decision
    
    def apply_rules(self, features):
        score = 0.0
        
        # Personal references strongly trigger retrieval
        if features.has_personal_reference:
            score += 0.5
        
        # Specific entities trigger retrieval
        score += min(0.3, features.named_entity_count * 0.1)
        
        # Common knowledge reduces retrieval need
        if features.is_common_knowledge:
            score -= 0.4
        
        # Recent retrieval reduces need
        if features.seconds_since_last_retrieval < 60:
            score -= 0.2
        
        if score > 0.4:
            return GateDecision.RETRIEVE(top_k=20)
        elif score > 0.2:
            return GateDecision.RETRIEVE(top_k=5)
        else:
            return GateDecision.SKIP
```

### Learning from Feedback

The gate's decisions are logged and periodically evaluated:
- When retrieval was skipped but the response was wrong → gate should have retrieved
- When retrieval happened but no context items were used → gate should have skipped
- Adjust thresholds based on these signals

---

## 5. Context Compression Pipeline

### Architecture

```
Raw Memory Items
       │
       ▼
┌─────────────┐
│  Tokenizer  │  Count tokens, identify budget constraints
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Importance      │  Score each item by relevance to current query
│ Scorer          │  (semantic similarity + recency + importance)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Selective       │  Remove low-value tokens within each item
│ Token Pruning   │  (keep named entities, dates, facts; remove filler)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Summarizer      │  For items still over budget, generate dense summary
│ (LLM-lite)      │  using a small, fast model
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Quality         │  Verify compressed item preserves key information
│ Checker         │  (fact preservation check)
└──────┬──────────┘
       │
       ▼
Compressed Context Items
```

### Compression Strategies by Item Type

| Item Type | Strategy | Compression Ratio |
|---|---|---|
| Factual memory | Extract key-value pairs, remove prose | 3-5x |
| Episodic memory | Summarize event, preserve date and outcome | 5-10x |
| Pattern/insight | Keep as-is (already dense) | 1x |
| Tool definition | Keep signature, compress description | 2-3x |
| Conversation history | Summarize turns, keep key exchanges | 4-6x |
| User preference | Keep as-is (short by nature) | 1x |

### Implementation

```python
class ContextCompressor:
    def __init__(self, small_model="fast-model"):
        self.small_model = small_model
    
    def compress(self, items, target_tokens):
        total_tokens = sum(count_tokens(item) for item in items)
        
        if total_tokens <= target_tokens:
            return items  # No compression needed
        
        compression_ratio = total_tokens / target_tokens
        
        compressed = []
        for item in items:
            strategy = self.select_strategy(item, compression_ratio)
            compressed_item = strategy.apply(item)
            compressed.append(compressed_item)
        
        # If still over budget, apply aggressive summarization
        while count_tokens(compressed) > target_tokens:
            compressed = self.aggressive_summarize(compressed, target_tokens)
        
        return compressed
    
    def select_strategy(self, item, compression_ratio):
        if item.type == "factual":
            return KeyValueExtractor(ratio=compression_ratio)
        elif item.type == "episodic":
            return EventSummarizer(ratio=compression_ratio)
        elif item.type == "pattern":
            return Passthrough()  # Already dense
        elif item.type == "tool_definition":
            return SignaturePreserver(ratio=compression_ratio)
        elif item.type == "conversation":
            return ConversationSummarizer(ratio=compression_ratio)
        else:
            return GenericSummarizer(ratio=compression_ratio)
```

### Quality Preservation

After compression, verify that key information is preserved:
- Named entities present in original must be present in compressed
- Dates and numbers must be preserved exactly
- Causal relationships must be maintained
- If quality check fails, fall back to less aggressive compression

---

## 6. Integration Plan

### Phase 1: Context Tiers (Week 1-2)
- Implement tier system in `context.py`
- Add tier-aware retrieval in `memory.py`
- Update prompt assembly in `agent.py`
- **Owner:** Memory Systems Engineer + Engineer
- **Research support:** This spec

### Phase 2: Position-Aware Injection (Week 2-3)
- Implement U-shaped context ordering
- Add critical fact repetition at end of prompt
- **Owner:** Engineer
- **Research support:** Context utilization measurement setup

### Phase 3: Retrieval Gating (Week 3-4)
- Implement retrieval gate with rule-based decisions
- Add decision logging for future learning
- **Owner:** Engineer
- **Research support:** Gate decision analysis

### Phase 4: Compression Pipeline (Week 4-6)
- Implement compression strategies by item type
- Add quality preservation checks
- **Owner:** Memory Systems Engineer
- **Research support:** Compression strategy evaluation

### Phase 5: Measurement & Optimization (Week 6-8)
- Deploy context utilization tracking
- Run ablation studies on production traffic
- Optimize retrieval parameters based on data
- **Owner:** QA Engineer + Memory Systems Engineer
- **Research support:** Ablation analysis

---

## 7. Success Criteria

| Criterion | Measurement | Target |
|---|---|---|
| Context utilization rate | Ablation testing | >80% |
| Token efficiency | Used tokens / total context tokens | >70% |
| Response quality | QA behavioral tests | No regression, ideally improvement |
| Latency | P50 response time | No increase (compression offsets retrieval cost) |
| Cost per response | Token cost | -20% (via compression + gating) |

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Compression loses important information | Quality checker with fact preservation verification |
| Retrieval gate skips when it shouldn't | Conservative default (retrieve on uncertainty) |
| Context tiers add complexity | Gradual rollout, feature flags per tier |
| Ablation testing is expensive | Sampling strategy (10% of responses) |
| Position bias varies by model | Model-specific tuning, A/B test per model |

---

## 9. Appendix: Paper Summaries

### Lost in the Middle (Liu et al.)
**Finding:** LLMs perform significantly worse when relevant information is in the middle of long contexts vs. at the beginning or end. The effect is consistent across model sizes and context lengths.
**Implication for Luke:** Never dump retrieved memories in the middle of the prompt. Use U-shaped ordering with critical info at both ends.

### LongLLMLingua
**Finding:** Token-level importance scoring can compress prompts 2-4x with minimal quality loss. Question-aware compression (scoring tokens by relevance to the query) outperforms question-agnostic compression.
**Implication for Luke:** Implement question-aware token pruning as the primary compression strategy.

### RECOMP (Xu et al.)
**Finding:** Training a compressor model to produce dense summaries of retrieved documents improves RAG quality while reducing context length. Abstractive compression outperforms extractive compression.
**Implication for Luke:** Use a small LLM for abstractive compression of episodic memories, extractive compression for factual memories.

### Anthropic: Building Effective Agents
**Finding:** Effective agent systems use context deliberately — they don't dump everything into the prompt. They use structured context, explicit instructions, and careful information flow design.
**Implication for Luke:** The tier system aligns with Anthropic's recommendation of structured, purposeful context injection.
