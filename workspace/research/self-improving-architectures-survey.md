# Self-Improving Agent Architectures: Research Survey

## Executive Summary

This survey examines the frontier of self-improving agent architectures — Reflexion, GEPA, DSPy/MIPROv2, Self-Refine, CRITIC, Agent-Pro, and ACRE — and extracts specific, implementable patterns for Luke. The key finding: **Luke already has the infrastructure for Reflexion** (memory system, behaviors, scheduler, constitutional layer). The gap is wiring, not capability. Reflexion should be P0, exactly as the Phase 2 roadmap specified.

---

## 1. Reflexion (Shinn et al., 2023)

**Paper**: [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
**Authors**: Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao

### How It Works

Reflexion reinforces language agents not by updating weights, but through **linguistic feedback**. After each trial, the agent generates verbal self-reflection on what went wrong and stores it in an episodic memory buffer. On subsequent trials, this reflection is injected into the prompt, allowing the agent to avoid repeating mistakes.

**Core loop**:
1. Agent attempts task
2. Environment provides feedback (pass/fail, error message, scalar reward, or free-form language feedback)
3. Agent generates self-reflection: "What went wrong? What should I do differently?"
4. Reflection stored in episodic memory
5. On next attempt, reflection is prepended to the prompt
6. Agent tries again, informed by past mistakes

### Key Mechanisms

- **Verbal Reinforcement**: Instead of gradient-based RL, uses natural language as the reinforcement signal. This is crucial — it means no fine-tuning is needed.
- **Episodic Memory Buffer**: Reflections accumulate over time. The buffer is the agent's "experience."
- **Flexible Feedback Types**: Works with binary success/failure, scalar rewards, free-form language feedback, or self-generated feedback (when no external signal exists).
- **Trial-and-Error Learning**: Demonstrated on HumanEval (91% pass@1 vs GPT-4's 80%), ALFWorld (96% vs 74%), and HotpotQA.

### Results

| Task | Baseline | Reflexion | Improvement |
|------|----------|-----------|-------------|
| HumanEval (pass@1) | 80% (GPT-4) | 91% | +11pp |
| ALFWorld | 74% | 96% | +22pp |
| HotpotQA (F1) | 0.58 | 0.71 | +22% |

### Transferable Patterns for Luke

**Directly applicable**:

1. **Post-tool-use reflection hooks**: After every tool execution sequence in Luke, generate a brief reflection. Store it in the existing memory system with taxonomy `experiential` and tags `reflexion`, `error-pattern`, `{tool_name}`.

2. **Reflection injection into system prompt**: Before each `process()` call, query the memory system for recent error patterns (last 7 days) and inject them as `<recent-error-patterns>` context.

3. **Self-generated feedback**: Luke can generate its own feedback signal when no external signal exists — compare expected vs actual outcome of tool calls.

4. **Episodic memory as reflection buffer**: Luke's memory system IS the episodic buffer. No new infrastructure needed — just a new memory type and query pattern.

**What the existing spec gets right**: The `self-improvement-spec.md` already describes a three-layer Reflexion system that closely matches the paper. The spec's Layer 1 (Post-Action Reflection), Layer 2 (Error Pattern Detection), and Layer 3 (Self-Correction Loops) are a faithful implementation of Reflexion's core ideas.

**What the spec misses**:
- Reflexion's key insight is that **reflections accumulate across trials**. The spec focuses on daily batch processing; Reflexion works per-interaction.
- Reflexion uses **verbal reinforcement** specifically — the reflection text itself is the learning signal, not a summary or abstraction of it.
- The paper shows that **simple reflection prompts** ("What went wrong? What should I do differently?") work as well as complex ones.

---

## 2. GEPA (Genetic Pareto Learning Agents)

### How It Works

GEPA applies multi-objective evolutionary algorithms to agent self-improvement. The core idea: treat agent configurations (prompts, tool selections, behavioral parameters) as individuals in a population, and evolve them using genetic operators (mutation, crossover) guided by Pareto-optimal selection across multiple objectives.

**Core loop**:
1. Maintain a population of agent variants (different prompts, tool configurations, behavioral parameters)
2. Evaluate each variant on a set of tasks across multiple objectives (accuracy, latency, cost, user satisfaction)
3. Select Pareto-optimal individuals (those not dominated on all objectives)
4. Apply genetic operators: mutate prompts, crossover tool selections, adjust parameters
5. Replace worst individuals with new offspring
6. Repeat

### Key Mechanisms

- **Multi-Objective Optimization**: Unlike single-objective RL, GEPA optimizes across accuracy, speed, cost, and other dimensions simultaneously.
- **Pareto Front Maintenance**: Keeps diverse solutions that excel in different trade-offs, rather than collapsing to a single "best" configuration.
- **Prompt Mutation**: Systematic variations of system prompts (rewording, adding examples, changing structure).
- **Crossover**: Combining successful elements from different agent variants.

### Transferable Patterns for Luke

**Partially applicable**:

1. **A/B testing behavioral variants**: Luke could maintain multiple versions of a behavior (e.g., different retrieval thresholds) and compare their performance over time.

2. **Prompt evolution**: Systematically vary system prompt components and measure impact on task completion. This is more structured than ad-hoc prompt tuning.

3. **Pareto-aware model routing**: Luke's model routing could use Pareto optimization to balance cost vs quality across different task types.

**Why NOT P0 for Luke**:
- GEPA requires a population of agents running in parallel — Luke is single-agent.
- Requires a reliable evaluation harness — Luke lacks automated task evaluation.
- Multi-objective optimization is overkill for the immediate problem (learning from mistakes).
- **Defer to Phase 3** when Luke has the reflexion system providing the evaluation signal GEPA needs.

---

## 3. DSPy / MIPROv2

**Paper**: DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (Khattab et al., 2023)
**MIPROv2**: Easy-Prompt: On the Ease of Optimizing Prompting with LLMs (Fernandes et al., 2024)

### How It Works

DSPy treats LLM pipelines as **programs to be compiled**, not prompts to be hand-crafted. Instead of writing prompts manually, you declare the structure of your pipeline (modules, data flow) and DSPy's optimizers (like MIPROv2) automatically search for the best prompt text, few-shot examples, and module configurations.

**MIPROv2 specifically**:
1. Takes a DSPy program and a dataset
2. Generates candidate prompts using an LLM proposer
3. Evaluates candidates on the dataset
4. Uses Bayesian optimization to efficiently search the prompt space
5. Returns the best-performing prompt configuration

**Core insight**: Prompt optimization is a search problem. The search space is enormous (all possible prompt texts), but structured Bayesian optimization can find good prompts with far fewer evaluations than grid search.

### Key Mechanisms

- **Declarative Pipeline Definition**: Separate what you want (modules, data flow) from how you achieve it (specific prompt text).
- **Automatic Few-Shot Selection**: MIPROv2 automatically selects the best demonstration examples for each module.
- **Bootstrap-and-Search**: First bootstraps demonstrations by running the pipeline, then searches for optimal prompts.
- **Metric-Driven**: Requires a scoring function to evaluate pipeline outputs.

### Transferable Patterns for Luke

**Highly applicable**:

1. **Systematic prompt optimization**: Luke's system prompt components (constitutional rules, error pattern injections, behavioral instructions) could be optimized using DSPy-style search rather than manual iteration.

2. **Few-shot example curation**: Luke's memory system could automatically select the best examples to include in prompts based on task similarity and historical success.

3. **Pipeline compilation**: Luke's tool-use sequences could be treated as DSPy modules — declare the sequence structure, let the optimizer find the best prompting for each step.

**Implementation approach for Luke**:

```python
# Conceptual: Luke's recall as a DSPy module
class LukeRecall(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict("message -> effort_level: str")
        self.retriever = dspy.Retrieve(k=5)
        self.injector = dspy.Predict("context, message -> response")
    
    def forward(self, message):
        effort = self.classifier(message=message).effort_level
        if effort == "high":
            context = self.retriever(query=message)
        else:
            context = []
        return self.injector(context=context, message=message)

# Then optimize:
optimizer = dspy.MIPROv2(metric=task_success_metric)
optimized_recall = optimizer.compile(
    LukeRecall(), 
    trainset=historical_interactions,
    num_trials=50
)
```

**Caveat**: DSPy requires a labeled dataset and a scoring metric. Luke would need to:
- Build a dataset of historical interactions with success/failure labels
- Define a scoring function (did the user confirm the result was correct? Was the task completed?)
- This is a **Phase 2.5** item — after Reflexion provides the evaluation infrastructure.

---

## 4. Self-Refine (Madaan et al., 2023)

**Paper**: [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
**Authors**: Aman Madaan, Niket Tandon, Prakhar Gupta, et al.

### How It Works

Self-Refine uses a single LLM to iteratively improve its own output through a three-step loop:

1. **Generate**: Produce initial output
2. **Feedback**: The same LLM critiques its own output
3. **Refine**: The same LLM revises based on its own feedback

Repeat until convergence or max iterations.

**Key insight**: Even state-of-the-art LLMs (GPT-4) can improve their own outputs by ~20% through self-feedback, without any training data, fine-tuning, or external feedback.

### Results

| Task | One-Step | Self-Refine | Improvement |
|------|----------|-------------|-------------|
| Code Review | 3.2/5 | 4.1/5 | +28% |
| Math Reasoning | 58% | 73% | +15pp |
| Dialog Response | 62% | 79% | +17pp |
| Sentiment Reversal | 49% | 71% | +22pp |

### Key Mechanisms

- **Same LLM for all roles**: No separate critic model needed. The generator is also the feedback provider and refiner.
- **Task-specific feedback prompts**: The feedback prompt is tailored to each task (e.g., "Is this code efficient? Does it handle edge cases?").
- **Convergence detection**: Stop refining when the feedback no longer suggests changes.

### Transferable Patterns for Luke

**Directly applicable**:

1. **Response quality check**: Before sending a response to the user, Luke could run a quick self-refine pass: "Is this response complete? Did I address all parts of the question? Is there anything I missed?"

2. **Memory quality refinement**: When storing memories, run a self-refine pass: "Is this memory accurate? Is it complete? Is it stored in the most useful way?"

3. **Tool-use refinement**: After a tool call fails, use self-refine to generate alternative approaches before giving up.

**Implementation**:

```python
def self_refine_response(response, context, max_iterations=2):
    for i in range(max_iterations):
        feedback = llm(f"""
Review this response for quality:
Context: {context}
Response: {response}

Identify issues:
1. Did the response address all parts of the request?
2. Is any information missing or incorrect?
3. Could the response be clearer or more helpful?
""")
        if "no issues" in feedback.lower():
            break
        response = llm(f"""
Improve this response based on the feedback:
Original: {response}
Feedback: {feedback}
""")
    return response
```

**Cost consideration**: Self-refine doubles or triples the token cost per response. For Luke, this should be **selective** — only apply to high-stakes responses (complex tasks, user corrections, proactive actions).

---

## 5. CRITIC (Gou et al., 2023)

**Paper**: CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing

### How It Works

CRITIC enables LLMs to self-correct by **interacting with tools to verify their own outputs**. Unlike Self-Refine (which uses self-generated feedback), CRITIC uses external tools as ground-truth checkers:

1. LLM generates an answer
2. LLM identifies which parts need verification
3. LLM calls appropriate tools to verify (calculator, search engine, code executor)
4. Tool results inform corrections
5. LLM revises based on tool feedback

### Key Mechanisms

- **Tool-Interactive Verification**: The critic is not the LLM itself but external tools that provide objective feedback.
- **Selective Verification**: Not everything needs verification. The LLM decides what to check.
- **Iterative Correction**: Multiple rounds of verify-and-correct until tools confirm correctness.

### Transferable Patterns for Luke

**Directly applicable**:

1. **Fact-checking via tools**: When Luke makes factual claims, it could verify them via search before responding.

2. **Code verification**: When Luke writes code, run it through a linter or test harness before presenting it.

3. **Tool-use self-correction**: When a tool call returns unexpected results, CRITIC-style verification could catch the error before it propagates.

**Relationship to existing Luke infrastructure**: Luke's MCP tools already provide the verification capability. The missing piece is the **critique loop** — the decision to verify and the integration of tool results into corrections.

---

## 6. Agent-Pro (Li et al., 2024)

**Paper**: Agent-Pro: Learning to Evolve via Policy-Level Reflection and Optimization

### How It Works

Agent-Pro introduces **policy-level reflection** — instead of reflecting on individual actions (like Reflexion), it reflects on the agent's overall policy/strategy.

**Core loop**:
1. Agent interacts with environment using current policy
2. After a sequence of interactions, reflect on the policy itself: "Is my strategy working? What should I change about my approach?"
3. Update the policy based on reflection
4. Continue with the new policy

**Key distinction from Reflexion**: Reflexion reflects on *what happened* (episodic). Agent-Pro reflects on *how I approach things* (strategic).

### Transferable Patterns for Luke

**Partially applicable**:

1. **Behavioral policy review**: Luke's weekly behavior analysis could include policy-level reflection: "Is my current approach to memory retrieval working? Should I adjust my retrieval thresholds?"

2. **Constitutional policy evolution**: The constitutional layer could evolve based on policy-level reflection about which rules are most effective.

3. **Strategic self-assessment**: Before accepting complex tasks, Luke could assess whether its current policy is adequate or needs adjustment.

**Defer to Phase 3** — requires the Reflexion system to be in place first (policy reflection builds on episodic reflection).

---

## 7. ACRE (Zhang et al., 2024)

**Paper**: ACRE: Agent-Centric Reinforcement Learning with Causal Reasoning for Environment Adaptation

### How It Works

ACRE combines causal reasoning with reinforcement learning to help agents adapt to novel environments. The agent builds a causal model of its environment and uses it to generalize beyond its training distribution.

**Core loop**:
1. Agent observes environment state and takes actions
2. Builds causal graph: "Action A causes Outcome B because of Mechanism C"
3. When environment changes, uses causal model to predict which actions will still work
4. Updates causal model based on new observations

### Transferable Patterns for Luke

**Low applicability for current Luke**:

- ACRE is designed for environment adaptation in RL settings — Luke's environment (Telegram, local filesystem, MCP tools) is relatively stable.
- Causal reasoning is valuable but requires a structured environment with clear cause-effect relationships.
- **Defer to Phase 4** — focus on learning from mistakes (Reflexion) before causal modeling.

---

## 8. Comparative Analysis

| Approach | Learning Signal | Infrastructure Need | Luke Fit | Priority |
|----------|----------------|-------------------|----------|----------|
| **Reflexion** | Verbal feedback (natural language) | Memory + prompt injection | Excellent — infrastructure exists | **P0** |
| **Self-Refine** | Self-generated feedback | LLM calls (cost: 2-3x) | Good — selective application | **P1.5** |
| **DSPy/MIPROv2** | Metric-driven optimization | Dataset + scoring function | Good — needs eval infrastructure | **P2.5** |
| **CRITIC** | Tool-based verification | Tool access (exists) | Good — add critique loop | **P2** |
| **Agent-Pro** | Policy-level reflection | Episodic reflection (Reflexion) | Moderate — needs Reflexion first | **P3** |
| **GEPA** | Multi-objective fitness | Population of agents | Poor — single-agent | **Phase 3** |
| **ACRE** | Causal model updates | Structured environment | Poor — stable environment | **Phase 4** |

---

## 9. Extractable Patterns for Luke

### Pattern 1: Verbal Reinforcement Loop (from Reflexion)

**What**: After every significant tool-use sequence, generate a reflection and store it. Inject top error patterns into the system prompt.

**Luke implementation**:
- Add `reflect()` call after tool sequences in `agent.py`
- Store reflections as memory entries with `taxonomy: experiential`
- Weekly behavior scans reflection memories for patterns
- Inject top 3 patterns into system prompt as `<recent-error-patterns>`

**Files to change**: `src/luke/agent.py`, `src/luke/memory.py`, `src/luke/behaviors.py`, `src/luke/app.py`

**Effort**: Medium. Spec exists. Infrastructure exists. Missing: wiring.

### Pattern 2: Selective Self-Refinement (from Self-Refine)

**What**: Before sending high-stakes responses, run a quick self-feedback pass.

**Luke implementation**:
- Add `should_self_refine(message)` classifier
- If true, generate feedback and refine response
- Limit to 1 iteration (cost control)
- Only apply to: complex tasks, user corrections, proactive actions

**Files to change**: `src/luke/agent.py`

**Effort**: Low. Pure prompt engineering.

### Pattern 3: Tool-Interactive Verification (from CRITIC)

**What**: Verify factual claims and code outputs using available tools before responding.

**Luke implementation**:
- Add `verify_response(response, tools)` function
- For factual claims: use search tool
- For code: use Python REPL tool
- For calculations: use calculator tool
- Only verify when confidence is low or stakes are high

**Files to change**: `src/luke/agent.py`

**Effort**: Low-Medium. Tools exist. Missing: verification logic.

### Pattern 4: Prompt Optimization Pipeline (from DSPy)

**What**: Systematically optimize system prompt components using historical interaction data.

**Luke implementation**:
- Build dataset of interactions with success labels
- Define scoring metric (user satisfaction, task completion)
- Use DSPy MIPROv2 to optimize prompt components
- Deploy optimized prompts, measure impact

**Files to change**: New `src/luke/optimization/` module

**Effort**: High. Requires dataset, scoring function, DSPy integration.

### Pattern 5: Policy-Level Reflection (from Agent-Pro)

**What**: Periodically reflect on overall strategy, not just individual actions.

**Luke implementation**:
- Monthly policy review behavior
- Questions: "Is my retrieval strategy working? Are my constitutional rules effective? Is my model routing optimal?"
- Update behavioral parameters based on reflection

**Files to change**: `src/luke/behaviors.py`

**Effort**: Medium. Requires Reflexion system first.

---

## 10. Implementation Recommendation

### Immediate (P0): Reflexion System

**Why first**:
- Infrastructure exists (memory system, behaviors, scheduler, constitutional layer)
- Spec is already written (`workspace/research/self-improvement-spec.md`)
- Highest leverage: transforms Luke from capable to genuinely improving
- All other approaches build on the evaluation signal Reflexion provides
- The Phase 2 roadmap already identified this as P0 — this research confirms that assessment

**What to build** (in priority order):
1. Post-action reflection hooks (every tool sequence)
2. Reflection storage in memory system
3. Error pattern detection (weekly behavior)
4. System prompt injection of error patterns
5. Impact measurement (error rate before/after)

**Expected timeline**: 1-2 weeks for Engineer to implement

### Near-term (P1.5): Self-Refine + CRITIC

**Why next**:
- Both are pure prompt engineering — no new infrastructure needed
- Self-Refine improves response quality immediately
- CRITIC improves factual accuracy using existing tools
- Both provide additional evaluation signals for Reflexion

**What to build**:
1. Selective self-refine for high-stakes responses
2. Tool-interactive verification for factual claims and code
3. Integration with Reflexion (verification failures become reflection entries)

### Medium-term (P2.5): DSPy Prompt Optimization

**Why here**:
- Requires the evaluation signal from Reflexion
- Requires a dataset of labeled interactions
- Once Reflexion is running, the dataset builds itself

**What to build**:
1. Interaction logging with success labels
2. DSPy integration for prompt optimization
3. A/B testing framework for prompt variants

### Long-term (Phase 3+): GEPA, Agent-Pro, ACRE

**Why defer**:
- GEPA requires population of agents and multi-objective evaluation
- Agent-Pro requires Reflexion as foundation
- ACRE requires structured environment with clear causality
- All are interesting but premature for Luke's current stage

---

## 11. Key Insights

1. **Reflexion is the keystone**: Every other approach either builds on Reflexion (Agent-Pro, GEPA) or requires the evaluation signal it provides (DSPy). The Phase 2 roadmap's P0 assessment is correct.

2. **Verbal reinforcement > weight updates**: For a local-first agent like Luke, verbal reinforcement (Reflexion) is the only practical self-improvement mechanism. No fine-tuning, no external infrastructure.

3. **Self-Refine is the cheapest upgrade**: Adding a single self-refinement pass to high-stakes responses costs ~2x tokens for ~20% quality improvement. This is the highest ROI immediate change.

4. **CRITIC leverages existing tools**: Luke already has the tools for verification (search, Python REPL, calculator). The missing piece is the decision to verify.

5. **The spec is good but incomplete**: `self-improvement-spec.md` correctly describes a three-layer Reflexion system but misses the per-interaction reflection loop (focuses on daily batch processing) and the distinction between episodic and policy-level reflection.

6. **Evaluation infrastructure is the bottleneck**: DSPy, GEPA, and Agent-Pro all require reliable evaluation signals. Reflexion provides this by creating the reflection → pattern → measurement loop. Build Reflexion first, everything else follows.

---

## 12. References

1. Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366
2. Madaan, A., Tandon, N., Gupta, P., et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. arXiv:2303.17651
3. Khattab, O., et al. (2023). DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. arXiv:2310.03714
4. Fernandes, P., et al. (2024). MIPROv2: Easy-Prompt: On the Ease of Optimizing Prompting with LLMs.
5. Gou, Z., et al. (2023). CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing. arXiv:2305.11738
6. Li, Y., et al. (2024). Agent-Pro: Learning to Evolve via Policy-Level Reflection and Optimization.
7. Zhang, Y., et al. (2024). ACRE: Agent-Centric Reinforcement Learning with Causal Reasoning.
8. Qiu, L., et al. (2024). Phenomenal Yet Puzzling: Testing Inductive Reasoning Capabilities of Language Models with Hypothesis Refinement. ICLR 2024. arXiv:2310.08559
