# Memory Consolidation Spec for Luke

## Overview

This spec defines the memory consolidation system for Luke — mechanisms for organizing, pruning, and structuring memories over time so that retrieval quality improves as the memory corpus grows, rather than degrading.

**Current state:** Luke stores memories as files with embeddings for semantic search. Memories accumulate without pruning, clustering, or hierarchical organization. Retrieval is flat: query → search → top-k.

**Target state:** Luke has a structured memory system with importance-based decay, semantic clustering, episodic/semantic separation, and scheduled consolidation — modeled on human memory organization principles.

**Design principle:** Memory systems should feel human. Not just technically correct retrieval, but intuitively right — remembering what matters, forgetting what doesn't, and organizing knowledge the way a trusted assistant would.

---

## 1. Importance Decay Model

### Problem

All memories currently persist indefinitely. This causes:
- Retrieval noise: irrelevant old memories compete with relevant recent ones
- Storage growth: unbounded accumulation without pruning
- Stale context: outdated information injected into prompts

### Solution: Exponential Decay with Reinforcement

Each memory has an **activation score** that decays over time but is reinforced by access and importance signals.

### Activation Formula

Based on Anderson's ACT-R theory:

```
A_i(t) = ln(Σ_{j=1}^{n} (t - t_j)^(-d)) + I_i + F_i + E_i

Where:
  A_i(t)     = activation of memory i at time t
  t_j        = time of j-th access of memory i
  n          = number of times memory i has been accessed
  d          = decay parameter (default 0.5, tunable)
  I_i        = intrinsic importance score (0-2, set at creation)
  F_i        = feedback bonus (from user corrections, -1 to +1)
  E_i        = emotional valence bonus (-0.5 to +0.5)
```

**Simplified implementation:**

```python
def compute_activation(memory, current_time):
    # Base decay: exponential with half-life
    age_days = (current_time - memory.created_at).days
    base_activation = math.exp(-age_days / memory.half_life_days)
    
    # Access reinforcement: each access boosts activation
    access_boost = 0
    for access_time in memory.access_history[-10:]:  # Last 10 accesses
        access_age = (current_time - access_time).days
        access_boost += 0.3 * math.exp(-access_age / 7)  # 7-day half-life for access boost
    
    # Intrinsic importance (set at creation, never decays)
    importance = memory.importance_score  # 0.0 - 1.0
    
    # Feedback bonus (user corrections)
    feedback = memory.feedback_score  # -1.0 to +1.0
    
    # Emotional valence (detected from context)
    emotion = memory.emotional_valence  # -0.5 to +0.5
    
    activation = base_activation * 0.4 + access_boost * 0.3 + importance * 0.2 + feedback * 0.05 + emotion * 0.05
    
    return max(0.0, min(1.0, activation))
```

### Decay Parameters

| Memory Type | Half-Life | Notes |
|---|---|---|
| Factual (names, preferences) | 180 days | Long half-life — facts persist |
| Episodic (events, conversations) | 30 days | Medium half-life — events fade |
| Pattern/insight | 365 days | Long half-life — insights are valuable |
| Tool usage log | 7 days | Short half-life — logs expire quickly |
| Temporary context | 1 day | Very short — cleanup daily |

### Spaced Repetition Signals

When a memory's activation drops below a threshold, flag it for review:

```python
REVIEW_THRESHOLDS = {
    "review_pending": 0.15,    # Flag for review
    "review_urgent": 0.05,     # Flag for urgent review or prune
    "prune_threshold": 0.02,   # Auto-prune if no review response
}

def check_review_status(memory, current_activation):
    if current_activation < REVIEW_THRESHOLDS["prune_threshold"]:
        return MemoryStatus.PRUNE
    elif current_activation < REVIEW_THRESHOLDS["review_urgent"]:
        return MemoryStatus.REVIEW_URGENT
    elif current_activation < REVIEW_THRESHOLDS["review_pending"]:
        return MemoryStatus.REVIEW_PENDING
    else:
        return MemoryStatus.ACTIVE
```

**Review process:**
1. During consolidation, collect all memories below review threshold
2. Generate a review summary: "These N memories are fading. Keep or prune?"
3. Present to Luke agent for decision during next heartbeat
4. If "keep" → reset activation to 0.5, extend half-life by 50%
5. If "prune" → archive to cold storage, remove from active index
6. If no response after 3 consolidation cycles → auto-archive

### Importance Scoring at Creation

When a memory is created, assign intrinsic importance:

```python
def assign_importance(memory, context):
    score = 0.5  # Default
    
    # User explicitly marked as important
    if context.user_marked_important:
        score += 0.3
    
    # Referenced by multiple other memories
    score += min(0.2, memory.incoming_links * 0.05)
    
    # Contains named entities (people, places, organizations)
    score += min(0.1, len(context.named_entities) * 0.02)
    
    # Emotional intensity detected
    score += abs(context.emotional_intensity) * 0.1
    
    # Task outcome (success/failure memories are more important)
    if context.is_task_outcome:
        score += 0.15
    
    return min(1.0, score)
```

---

## 2. Semantic Clustering System

### Problem

Memories are stored flat — no grouping, no hierarchy, no relationships beyond individual embeddings. This makes retrieval inefficient and prevents Luke from "understanding" the structure of what it knows.

### Solution: Automatic Clustering with Hierarchical Organization

### Clustering Algorithm

Use a two-phase approach:

**Phase 1: Online clustering (incremental)**
- As each new memory is created, assign it to the nearest existing cluster
- If distance to all clusters > threshold, create new cluster
- Use existing embeddings — no re-clustering needed

**Phase 2: Offline consolidation (scheduled)**
- During consolidation runs, re-cluster all memories with updated embeddings
- Use HDBSCAN or similar density-based clustering for better cluster quality
- Merge clusters that have converged, split clusters that have diverged

```python
class MemoryClusterer:
    def __init__(self, embedding_model, min_cluster_size=5):
        self.embedding_model = embedding_model
        self.min_cluster_size = min_cluster_size
        self.clusters = []  # List of Cluster objects
    
    def assign_online(self, new_memory):
        """Assign new memory to nearest cluster (online)"""
        if not self.clusters:
            self.clusters.append(Cluster(members=[new_memory]))
            return self.clusters[0]
        
        # Compute similarity to each cluster centroid
        similarities = [
            cosine_similarity(new_memory.embedding, cluster.centroid)
            for cluster in self.clusters
        ]
        
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        if best_sim > 0.7:  # Threshold for cluster membership
            self.clusters[best_idx].add(new_memory)
            self.clusters[best_idx].update_centroid()
            return self.clusters[best_idx]
        else:
            # Create new cluster
            new_cluster = Cluster(members=[new_memory])
            self.clusters.append(new_cluster)
            return new_cluster
    
    def consolidate_offline(self, all_memories):
        """Re-cluster all memories (offline, during consolidation)"""
        embeddings = [m.embedding for m in all_memories]
        
        # Use HDBSCAN for density-based clustering
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='cosine',
            cluster_selection_epsilon=0.3
        )
        labels = clusterer.fit_predict(embeddings)
        
        # Build new clusters
        new_clusters = []
        for label in set(labels):
            if label == -1:  # Noise points
                continue
            members = [m for m, l in zip(all_memories, labels) if l == label]
            if len(members) >= self.min_cluster_size:
                new_clusters.append(Cluster(members=members, label=label))
        
        self.clusters = new_clusters
        return self.clusters
```

### Hierarchical Memory Structure

Organize memories at four levels of abstraction:

```
Level 1: Facts (Concrete)
  - "Filipe's timezone is UTC-3"
  - "Luke runs on macOS with launchd"
  - "The Telegram bot token is stored in .env"
  
Level 2: Events (Episodic)
  - "On 2026-04-01, Filipe asked about context engineering research"
  - "On 2026-03-28, the reliability agent detected a launchd crash"
  
Level 3: Patterns (Generalizations)
  - "Filipe prefers research deliverables as markdown files"
  - "Agent crashes usually happen after model API changes"
  
Level 4: Principles (Abstract)
  - "Context quality matters more than context quantity"
  - "Self-improvement requires measurement before optimization"
```

**Implementation:**

```python
class MemoryHierarchy:
    def __init__(self):
        self.facts = MemoryStore(type="fact")
        self.events = MemoryStore(type="event")
        self.patterns = MemoryStore(type="pattern")
        self.principles = MemoryStore(type="principle")
    
    def promote(self, memory, new_level):
        """Promote memory to higher abstraction level"""
        current_store = self.get_store(memory.level)
        current_store.remove(memory)
        
        memory.level = new_level
        new_store = self.get_store(new_level)
        new_store.add(memory)
    
    def extract_pattern(self, events):
        """Extract a pattern from multiple events"""
        # Find commonalities across events
        common_entities = intersect_entities(events)
        common_context = intersect_context(events)
        
        if len(events) >= 3 and common_entities:
            pattern = Memory(
                content=f"Pattern: {generate_pattern_description(events)}",
                level="pattern",
                source_events=[e.id for e in events],
                importance=min(1.0, len(events) * 0.2)
            )
            self.patterns.add(pattern)
            return pattern
        return None
```

### Cross-Linking

Automatically create bidirectional links between related memories:

```python
def build_cross_links(memory, all_memories, threshold=0.6):
    """Find and create links to related memories"""
    links = []
    for other in all_memories:
        if other.id == memory.id:
            continue
        
        similarity = cosine_similarity(memory.embedding, other.embedding)
        if similarity > threshold:
            links.append(MemoryLink(
                source=memory.id,
                target=other.id,
                strength=similarity,
                link_type="semantic"
            ))
    
    # Also link by shared entities
    shared_entities = set(memory.entities) & set(other.entities)
    if shared_entities:
        links.append(MemoryLink(
            source=memory.id,
            target=other.id,
            strength=len(shared_entities) * 0.1,
            link_type="entity",
            entities=shared_entities
        ))
    
    return links
```

### Cluster Retrieval

When retrieving memories, use cluster-aware retrieval:

```python
def cluster_aware_retrieval(query, clusters, top_k=20):
    """Retrieve from clusters, not flat memory store"""
    # Find relevant clusters
    query_embedding = embed(query)
    cluster_scores = [
        (cluster, cosine_similarity(query_embedding, cluster.centroid))
        for cluster in clusters
    ]
    
    # Select top clusters
    top_clusters = sorted(cluster_scores, key=lambda x: x[1], reverse=True)[:5]
    
    # Retrieve within selected clusters
    results = []
    for cluster, score in top_clusters:
        cluster_results = cluster.search(query, top_k=max(3, top_k // 5))
        results.extend(cluster_results)
    
    # Sort by relevance, return top_k
    results.sort(key=lambda x: x.similarity, reverse=True)
    return results[:top_k]
```

---

## 3. Episodic vs. Semantic Memory Separation

### Problem

Luke currently stores all memories in a single flat store. This conflates:
- **Episodic memories**: specific events with timestamps ("Filipe asked about X on date Y")
- **Semantic memories**: general knowledge ("Filipe prefers X", "The system does Y")

These require different storage, retrieval, and consolidation strategies.

### Solution: Two Memory Stores with Different Characteristics

### Episodic Memory Store

| Property | Value |
|---|---|
| Content | Time-stamped events, conversations, interactions |
| Structure | Chronological + semantic index |
| Retrieval | Temporal + semantic query (e.g., "what happened last week about X?") |
| Decay | Faster (30-day half-life) |
| Compression | Summarize events, preserve date and outcome |
| Storage | Files organized by date: `memory/episodes/2026/04/04/` |

```python
class EpisodicStore:
    def __init__(self, base_path):
        self.base_path = base_path
        self.index = SemanticIndex()  # For semantic search within episodes
    
    def store(self, episode):
        """Store an episodic memory"""
        date_path = episode.timestamp.strftime("%Y/%m/%d")
        file_path = os.path.join(self.base_path, date_path, f"{episode.id}.yaml")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        data = {
            "id": episode.id,
            "timestamp": episode.timestamp.isoformat(),
            "content": episode.content,
            "entities": episode.entities,
            "outcome": episode.outcome,
            "participants": episode.participants,
            "embedding": episode.embedding.tolist()
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(data, f)
        
        self.index.add(episode.id, episode.embedding, episode.content)
    
    def retrieve(self, query, time_range=None, top_k=20):
        """Retrieve episodic memories with optional time filter"""
        results = self.index.search(query, top_k=top_k * 2)  # Over-fetch
        
        if time_range:
            results = [r for r in results if time_range.start <= r.timestamp <= time_range.end]
        
        return results[:top_k]
```

### Semantic Memory Store

| Property | Value |
|---|---|
| Content | Facts, preferences, patterns, principles |
| Structure | Clustered by topic, hierarchical |
| Retrieval | Pure semantic query (e.g., "what does Filipe prefer?") |
| Decay | Slower (180-365 day half-life) |
| Compression | Keep as-is (already dense) |
| Storage | Files organized by cluster: `memory/semantic/clusters/` |

```python
class SemanticStore:
    def __init__(self, base_path):
        self.base_path = base_path
        self.clusters = {}  # cluster_id -> Cluster
        self.index = SemanticIndex()
    
    def store(self, fact):
        """Store a semantic memory"""
        cluster = self.assign_to_cluster(fact)
        cluster.add(fact)
        self.index.add(fact.id, fact.embedding, fact.content)
    
    def retrieve(self, query, top_k=20):
        """Retrieve semantic memories"""
        return self.index.search(query, top_k=top_k)
    
    def assign_to_cluster(self, fact):
        """Assign fact to appropriate cluster"""
        # Use online clustering (see Section 2)
        pass
```

### Episodic-to-Semantic Consolidation Pipeline

Periodically extract generalizable facts from episodic memories:

```python
class EpisodicToSemanticConsolidator:
    def __init__(self, episodic_store, semantic_store, llm):
        self.episodic = episodic_store
        self.semantic = semantic_store
        self.llm = llm
    
    def consolidate(self, time_window_days=7):
        """Extract semantic facts from recent episodic memories"""
        cutoff = datetime.now() - timedelta(days=time_window_days)
        recent_episodes = self.episodic.retrieve_all(after=cutoff)
        
        # Group episodes by topic
        topic_groups = self.group_by_topic(recent_episodes)
        
        new_facts = []
        for topic, episodes in topic_groups.items():
            if len(episodes) >= 2:  # Need multiple episodes to extract pattern
                fact = self.extract_fact(episodes)
                if fact:
                    self.semantic.store(fact)
                    new_facts.append(fact)
                    
                    # Link fact to source episodes
                    for episode in episodes:
                        episode.link_to_fact(fact.id)
        
        return new_facts
    
    def extract_fact(self, episodes):
        """Extract a generalizable fact from multiple episodes"""
        prompt = f"""
Given these episodes, extract any generalizable facts or patterns:

{format_episodes(episodes)}

Return a single concise fact statement, or NONE if no pattern emerges.
"""
        response = self.llm.generate(prompt)
        
        if response.strip().upper() == "NONE":
            return None
        
        return SemanticFact(
            content=response.strip(),
            source_episodes=[e.id for e in episodes],
            confidence=len(episodes) / 5.0,  # Higher with more episodes
            created_at=datetime.now()
        )
```

### Different Retrieval Strategies

| Query Type | Store | Strategy |
|---|---|---|
| "What did Filipe say about X?" | Episodic | Temporal + semantic search |
| "What does Filipe prefer?" | Semantic | Pure semantic search |
| "What happened last Tuesday?" | Episodic | Temporal range query |
| "How does the memory system work?" | Semantic | Cluster-based retrieval |
| "Remind me about our conversation on Y" | Episodic | Exact date lookup |
| "What have we learned about Z?" | Both | Semantic for facts, episodic for examples |

```python
class MemoryRetriever:
    def __init__(self, episodic_store, semantic_store):
        self.episodic = episodic_store
        self.semantic = semantic_store
    
    def retrieve(self, query, context):
        """Intelligent retrieval based on query type"""
        query_type = self.classify_query(query, context)
        
        if query_type == QueryType.EPISODIC:
            return self.episodic.retrieve(query, context.time_range)
        elif query_type == QueryType.SEMANTIC:
            return self.semantic.retrieve(query)
        elif query_type == QueryType.MIXED:
            semantic_results = self.semantic.retrieve(query, top_k=10)
            episodic_results = self.episodic.retrieve(query, top_k=10)
            return self.merge_and_rank(semantic_results, episodic_results)
        else:
            # Default: semantic first, episodic as fallback
            results = self.semantic.retrieve(query, top_k=15)
            if len(results) < 5:
                results.extend(self.episodic.retrieve(query, top_k=10))
            return results
```

---

## 4. Sleep-Based Consolidation

### Problem

Without periodic consolidation, memories become disorganized, duplicates accumulate, and the system degrades over time.

### Solution: Scheduled Consolidation Runs During Low-Activity Periods

### Consolidation Schedule

```python
CONSOLIDATION_SCHEDULE = {
    "daily_light": {
        "cron": "0 4 * * *",  # 4 AM daily
        "scope": "recent_only",  # Last 24 hours
        "operations": ["decay_update", "cross_link", "duplicate_check"]
    },
    "weekly_full": {
        "cron": "0 3 * * 0",  # 3 AM Sunday
        "scope": "all_memories",
        "operations": ["decay_update", "recluster", "cross_link", "duplicate_check", "pattern_extraction", "review_flag"]
    },
    "monthly_deep": {
        "cron": "0 2 1 * *",  # 2 AM 1st of month
        "scope": "all_memories",
        "operations": ["decay_update", "recluster", "cross_link", "duplicate_check", "pattern_extraction", "review_flag", "archive_pruned", "hierarchy_promotion"]
    }
}
```

### Consolidation Operations

```python
class ConsolidationEngine:
    def __init__(self, episodic_store, semantic_store, clusterer, consolidator, llm):
        self.episodic = episodic_store
        self.semantic = semantic_store
        self.clusterer = clusterer
        self.consolidator = consolidator
        self.llm = llm
    
    def run_consolidation(self, scope, operations):
        """Run a consolidation cycle"""
        report = ConsolidationReport()
        start_time = time.time()
        
        # Load memories based on scope
        if scope == "recent_only":
            cutoff = datetime.now() - timedelta(hours=24)
            memories = self.episodic.retrieve_all(after=cutoff)
        else:
            memories = self.episodic.retrieve_all()
        
        for op in operations:
            op_start = time.time()
            result = getattr(self, f"_op_{op}")(memories)
            report.add_operation(op, time.time() - op_start, result)
        
        report.total_time = time.time() - start_time
        report.memories_processed = len(memories)
        
        # Save report
        self.save_report(report)
        
        return report
    
    def _op_decay_update(self, memories):
        """Update activation scores for all memories"""
        updated = 0
        for memory in memories:
            old_activation = memory.activation
            memory.activation = compute_activation(memory, datetime.now())
            memory.status = check_review_status(memory, memory.activation)
            if memory.activation != old_activation:
                updated += 1
        return {"updated": updated, "total": len(memories)}
    
    def _op_recluster(self, memories):
        """Re-cluster all memories offline"""
        old_clusters = len(self.clusterer.clusters)
        self.clusterer.consolidate_offline(memories)
        new_clusters = len(self.clusterer.clusters)
        return {"old_clusters": old_clusters, "new_clusters": new_clusters}
    
    def _op_cross_link(self, memories):
        """Build cross-links between related memories"""
        new_links = 0
        for memory in memories:
            links = build_cross_links(memory, memories)
            new_links += len(links)
            memory.links.extend(links)
        return {"new_links": new_links}
    
    def _op_duplicate_check(self, memories):
        """Find and merge duplicate memories"""
        duplicates_found = 0
        duplicates_merged = 0
        
        # Find near-duplicates (very high similarity)
        for i, m1 in enumerate(memories):
            for m2 in memories[i+1:]:
                sim = cosine_similarity(m1.embedding, m2.embedding)
                if sim > 0.95 and self.are_semantically_equivalent(m1, m2):
                    duplicates_found += 1
                    merged = self.merge_memories(m1, m2)
                    duplicates_merged += 1
        
        return {"found": duplicates_found, "merged": duplicates_merged}
    
    def _op_pattern_extraction(self, memories):
        """Extract patterns from episodic memories"""
        new_patterns = self.consolidator.consolidate(time_window_days=7)
        return {"patterns_extracted": len(new_patterns)}
    
    def _op_review_flag(self, memories):
        """Flag memories needing review"""
        flagged = [m for m in memories if m.status in [MemoryStatus.REVIEW_PENDING, MemoryStatus.REVIEW_URGENT]]
        return {"flagged": len(flagged)}
    
    def _op_archive_pruned(self, memories):
        """Archive memories below prune threshold"""
        archived = 0
        for memory in memories:
            if memory.status == MemoryStatus.PRUNE:
                self.archive_memory(memory)
                archived += 1
        return {"archived": archived}
    
    def _op_hierarchy_promotion(self, memories):
        """Promote memories to higher abstraction levels"""
        promoted = 0
        # Find patterns that have been reinforced enough to become principles
        for memory in memories:
            if memory.level == "pattern" and self.should_promote_to_principle(memory):
                memory.level = "principle"
                promoted += 1
        return {"promoted": promoted}
```

### Memory Reconsolidation

When a memory is retrieved and corrected, update the stored version:

```python
def reconsolidate_memory(memory, corrected_content, context):
    """Update memory based on correction during retrieval"""
    memory.content = corrected_content
    memory.last_corrected = datetime.now()
    memory.correction_count += 1
    
    # Update embedding
    memory.embedding = embed(corrected_content)
    
    # Boost activation (corrected memories are important)
    memory.activation = min(1.0, memory.activation + 0.2)
    
    # Update cross-links (embedding changed)
    memory.links = build_cross_links(memory, all_memories)
    
    return memory
```

### Consolidation Reports

After each consolidation run, generate a report:

```yaml
# Example consolidation report
consolidation_report:
  run_id: "consol-2026-04-04-04-00"
  type: "daily_light"
  timestamp: "2026-04-04T04:00:00Z"
  duration_seconds: 45
  memories_processed: 234
  
  operations:
    decay_update:
      duration_seconds: 5
      result: {updated: 187, total: 234}
    cross_link:
      duration_seconds: 20
      result: {new_links: 45}
    duplicate_check:
      duration_seconds: 15
      result: {found: 3, merged: 2}
  
  summary:
    memories_active: 232
    memories_review_pending: 12
    memories_review_urgent: 3
    memories_archived: 0
    total_clusters: 28
    new_patterns_extracted: 1
```

---

## 5. Integration Plan

### Phase 1: Activation & Decay (Week 1-2)
- Add activation score field to existing memory schema
- Implement decay computation in `memory.py`
- Add importance scoring at memory creation
- **Owner:** Memory Systems Engineer
- **Research support:** This spec

### Phase 2: Clustering (Week 2-4)
- Implement online clustering for new memories
- Add offline re-clustering to consolidation pipeline
- Build cluster-aware retrieval
- **Owner:** Memory Systems Engineer
- **Research support:** Clustering algorithm evaluation

### Phase 3: Episodic/Semantic Separation (Week 4-6)
- Create separate stores in `db.py`
- Implement episodic-to-semantic consolidation pipeline
- Update retrieval logic in `memory.py` to use dual stores
- **Owner:** Memory Systems Engineer
- **Research support:** Retrieval strategy evaluation

### Phase 4: Consolidation Engine (Week 6-8)
- Implement scheduled consolidation runs
- Add all consolidation operations
- Build consolidation reporting
- **Owner:** Memory Systems Engineer + Reliability Engineer (scheduling)
- **Research support:** Consolidation frequency optimization

### Phase 5: Measurement & Tuning (Week 8-10)
- Deploy retrieval quality measurement
- Tune decay parameters based on data
- Optimize cluster count and size
- **Owner:** QA Engineer + Memory Systems Engineer
- **Research support:** Parameter tuning analysis

---

## 6. Success Criteria

| Criterion | Measurement | Target |
|---|---|---|
| Retrieval precision | Relevant items / retrieved items | >85% |
| Memory storage growth rate | New memories per week after pruning | -50% vs. baseline |
| Cluster coherence | Silhouette score on embedding space | >0.7 |
| Factual query latency | P50 retrieval time for semantic queries | <200ms |
| Duplicate rate | Duplicates found / total memories | <2% |
| Review response rate | Memories reviewed / memories flagged | >80% |

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Decay prunes important memories | Conservative thresholds, review-before-prune process |
| Clustering creates bad groups | Offline re-clustering with quality metrics, manual override |
| Episodic/semantic split breaks existing retrieval | Gradual migration, dual-read during transition |
| Consolidation runs are slow | Scope-limited daily runs, full runs during maintenance window |
| Memory reconsolidation loses original data | Keep original in archive, store correction history |

---

## 8. Appendix: Paper Summaries

### ACT-R Theory (Anderson)
**Finding:** Human memory activation follows a power-law decay function, reinforced by each access. The activation equation predicts retrieval latency and accuracy with high precision.
**Implication for Luke:** Use the ACT-R activation formula as the foundation for memory scoring. It's well-validated and computationally efficient.

### Tulving's Episodic/Semantic Distinction
**Finding:** Episodic and semantic memory are functionally and neurologically distinct systems with different retrieval mechanisms, decay rates, and consolidation processes.
**Implication for Luke:** Separate stores are justified — they serve different purposes and should be optimized independently.

### Sleep and Memory Consolidation (Walker & Stickgold)
**Finding:** Sleep-dependent memory consolidation reorganizes memories, strengthens important ones, and integrates new memories with existing knowledge. The process is selective, not uniform.
**Implication for Luke:** Scheduled consolidation runs should mimic sleep consolidation — selective, importance-weighted, and integrative. Not just cleanup, but active reorganization.

### Memory Reconsolidation
**Finding:** When a memory is retrieved, it becomes labile and can be updated before being re-stored. This is how memories stay current rather than stale.
**Implication for Luke:** Implement reconsolidation — when Luke corrects a memory during use, update the stored version, not just add a new one.
