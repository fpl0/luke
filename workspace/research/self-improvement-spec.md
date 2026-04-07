# Self-Improvement Loops Spec for Luke

## Research Validation (FPL-66)

This spec was validated against the comprehensive research survey in `self-improving-architectures-survey.md` (FPL-66). Key updates from research findings:

1. **Per-interaction reflection > daily batch**: Reflexion (Shinn et al.) shows reflections work best when generated immediately after each action, not batched daily. Layer 1 should trigger per tool-use sequence, not on a schedule.
2. **Verbal reinforcement is the learning signal**: The reflection text itself — not summaries or abstractions — is what drives improvement. Store raw reflections, not processed summaries.
3. **Simple reflection prompts work best**: "What went wrong? What should I do differently?" performs as well as complex prompts. Keep it simple.
4. **Episodic vs policy distinction**: This spec covers episodic reflection (Layer 1-2). Policy-level reflection (Agent-Pro) is a future addition requiring this foundation.

## Overview

This spec defines the self-improvement system for Luke — mechanisms for learning from experience, optimizing behavior, and evolving capabilities over time without human intervention.

**Current state:** Luke operates without systematic self-improvement. Each interaction is independent; there is no mechanism for learning from mistakes, optimizing prompts based on outcomes, or evolving skills over time.

**Target state:** Luke has a closed-loop self-improvement system that measures performance, identifies improvement opportunities, tests changes, and promotes successful variants — all automatically.

**Design principle:** Self-improvement must be safe. Every change is measured, tested, and reversible. No autonomous self-modification of core behavior — only optimization within defined boundaries.

---

## 1. Reflexion System

### Problem

Luke makes mistakes but doesn't systematically learn from them. The same errors can recur because there's no mechanism for post-action analysis and behavioral adjustment.

### Solution: Three-Layer Reflexion

```
Layer 1: Post-Action Reflection (Every interaction)
  ↓
Layer 2: Error Pattern Detection (Daily)
  ↓
Layer 3: Self-Correction Loops (Real-time)
```

### Layer 1: Post-Action Reflection

After every significant action, generate a brief reflection:

```python
class ReflexionEngine:
    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory = memory_store
        self.reflection_log = []
    
    def reflect(self, action, outcome, context):
        """Generate post-action reflection"""
        prompt = f"""
Action: {action.description}
Expected outcome: {action.expected_outcome}
Actual outcome: {outcome.description}
Context: {context.summary}

Reflect briefly:
1. What went well?
2. What could have been better?
3. What should I do differently next time?
4. Rate success (1-5)
"""
        reflection = self.llm.generate(prompt, max_tokens=200)
        
        reflection_data = {
            "action_id": action.id,
            "timestamp": datetime.now(),
            "reflection": reflection,
            "success_rating": extract_rating(reflection),
            "improvement_suggestions": extract_suggestions(reflection),
            "error_detected": outcome.success == False
        }
        
        self.reflection_log.append(reflection_data)
        self.memory.store_reflection(reflection_data)
        
        return reflection_data
```

**When to reflect:**
- After every tool execution (success/failure)
- After every multi-step task completion
- After every user correction ("that's not what I meant")
- NOT after simple conversational turns

**Reflection storage:**
```yaml
# Reflection memory entry
reflection:
  action_id: "tool_call_123"
  action_type: "file_read"
  timestamp: "2026-04-04T19:00:00Z"
  expected: "Read config file and parse settings"
  actual: "File not found at expected path"
  reflection: "I assumed the config file location without checking. Next time, I should search for the file or ask the user."
  success_rating: 2
  improvement: "Verify file existence before reading; use glob search as fallback"
  error_type: "assumption_error"
```

### Layer 2: Error Pattern Detection

Daily analysis of reflection logs to identify recurring patterns:

```python
class ErrorPatternDetector:
    def __init__(self, reflection_log, llm):
        self.log = reflection_log
        self.llm = llm
        self.known_patterns = []  # Persisted pattern definitions
    
    def detect_patterns(self, window_days=7):
        """Find recurring error patterns in recent reflections"""
        recent = [r for r in self.log if r["timestamp"] > datetime.now() - timedelta(days=window_days)]
        errors = [r for r in recent if r.get("error_detected")]
        
        if len(errors) < 3:
            return []  # Not enough data
        
        # Group by error type
        by_type = defaultdict(list)
        for error in errors:
            by_type[error.get("error_type", "unknown")].append(error)
        
        # Identify patterns (3+ occurrences of same type)
        patterns = []
        for error_type, instances in by_type.items():
            if len(instances) >= 3:
                pattern = self.analyze_pattern(error_type, instances)
                patterns.append(pattern)
        
        # Also use LLM to find non-obvious patterns
        llm_patterns = self.llm_find_patterns(errors)
        patterns.extend(llm_patterns)
        
        return patterns
    
    def analyze_pattern(self, error_type, instances):
        """Analyze a specific error pattern"""
        return {
            "type": error_type,
            "frequency": len(instances),
            "first_seen": min(i["timestamp"] for i in instances),
            "last_seen": max(i["timestamp"] for i in instances),
            "contexts": [i.get("context_summary") for i in instances],
            "suggestions": [i.get("improvement_suggestions") for i in instances],
            "severity": self.assess_severity(instances)
        }
    
    def assess_severity(self, instances):
        """Assess pattern severity based on frequency and impact"""
        avg_rating = np.mean([i.get("success_rating", 3) for i in instances])
        frequency = len(instances)
        
        if frequency >= 10 or avg_rating <= 1.5:
            return "critical"
        elif frequency >= 5 or avg_rating <= 2.0:
            return "high"
        elif frequency >= 3:
            return "medium"
        else:
            return "low"
```

### Layer 3: Self-Correction Loops

Real-time self-correction when the agent detects a potential error:

```python
class SelfCorrectionLoop:
    def __init__(self, llm, error_patterns):
        self.llm = llm
        self.error_patterns = error_patterns  # Known patterns to check against
    
    def check_before_respond(self, action_plan, context):
        """Pre-flight check: will this action likely fail?"""
        for pattern in self.error_patterns:
            if self.matches_pattern(action_plan, pattern):
                correction = self.generate_correction(action_plan, pattern, context)
                return Correction(
                    needed=True,
                    reason=f"Matches known error pattern: {pattern.type}",
                    suggestion=correction
                )
        return Correction(needed=False)
    
    def check_after_respond(self, response, context):
        """Post-flight check: does this response look correct?"""
        prompt = f"""
Review this response for potential issues:

Response: {response.text}
Context: {context.summary}

Check for:
1. Factual accuracy (any claims that might be wrong?)
2. Completeness (does it fully address the request?)
3. Appropriateness (is the tone and format correct?)
4. Safety (any concerning content?)

Flag any issues found.
"""
        review = self.llm.generate(prompt, max_tokens=150)
        issues = parse_review(review)
        
        if issues:
            return Correction(
                needed=True,
                reason=f"Self-review found {len(issues)} issues",
                issues=issues
            )
        return Correction(needed=False)
    
    def matches_pattern(self, action_plan, pattern):
        """Check if action plan matches a known error pattern"""
        # Simple heuristic: check if action type and context match
        if action_plan.type in pattern.action_types:
            context_similarity = self.compute_context_similarity(
                action_plan.context, pattern.contexts
            )
            return context_similarity > 0.6
        return False
```

---

## 2. Execution-Trace Analysis

### Problem

Luke's behavior is opaque. There's no systematic way to understand what the agent does, why it makes certain decisions, or how to improve its behavior over time.

### Solution: Full Execution Trace Logging with Analysis Pipeline

### Trace Logging

Record every step of every action:

```python
class ExecutionTracer:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.current_trace = None
    
    def start_trace(self, task_id, task_description):
        """Begin a new execution trace"""
        self.current_trace = Trace(
            id=generate_id(),
            task_id=task_id,
            task_description=task_description,
            start_time=datetime.now(),
            steps=[],
            outcome=None
        )
        return self.current_trace.id
    
    def record_step(self, step_type, input_data, output_data, metadata=None):
        """Record a single step in the execution"""
        if not self.current_trace:
            return
        
        step = TraceStep(
            step_number=len(self.current_trace.steps) + 1,
            type=step_type,
            input=input_data,
            output=output_data,
            timestamp=datetime.now(),
            metadata=metadata or {},
            duration_ms=None  # Set on next step or end
        )
        
        if self.current_trace.steps:
            prev = self.current_trace.steps[-1]
            prev.duration_ms = (step.timestamp - prev.timestamp).total_seconds() * 1000
        
        self.current_trace.steps.append(step)
    
    def end_trace(self, outcome, metrics=None):
        """Complete the execution trace"""
        if not self.current_trace:
            return
        
        self.current_trace.end_time = datetime.now()
        self.current_trace.outcome = outcome
        self.current_trace.metrics = metrics or {}
        self.current_trace.total_duration_ms = (
            self.current_trace.end_time - self.current_trace.start_time
        ).total_seconds() * 1000
        
        # Save trace
        self.save_trace(self.current_trace)
        trace_id = self.current_trace.id
        self.current_trace = None
        
        return trace_id
    
    def save_trace(self, trace):
        """Save trace to storage"""
        date_path = trace.start_time.strftime("%Y/%m/%d")
        file_path = os.path.join(self.storage_path, date_path, f"{trace.id}.json")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(trace.to_dict(), f, indent=2, default=str)
```

### Trace Data Model

```json
{
  "id": "trace_20260404_001",
  "task_id": "FPL-57",
  "task_description": "Produce context engineering spec",
  "start_time": "2026-04-04T18:59:20Z",
  "end_time": "2026-04-04T19:01:09Z",
  "total_duration_ms": 109000,
  "outcome": {
    "success": true,
    "deliverable": "workspace/research/context-engineering-spec.md",
    "quality_score": null
  },
  "steps": [
    {
      "step_number": 1,
      "type": "tool_call",
      "tool": "checkout",
      "input": {"issue_id": "FPL-57"},
      "output": {"status": "checked_out"},
      "timestamp": "2026-04-04T18:59:20Z",
      "duration_ms": 500
    },
    {
      "step_number": 2,
      "type": "llm_call",
      "model": "qwen3.6-plus:free",
      "input_tokens": 2500,
      "output_tokens": 8000,
      "prompt_summary": "Research context engineering spec...",
      "timestamp": "2026-04-04T18:59:25Z",
      "duration_ms": 45000
    },
    {
      "step_number": 3,
      "type": "file_write",
      "file": "workspace/research/context-engineering-spec.md",
      "size_bytes": 15000,
      "timestamp": "2026-04-04T19:00:10Z",
      "duration_ms": 50
    }
  ],
  "metrics": {
    "total_steps": 3,
    "tool_calls": 1,
    "llm_calls": 1,
    "file_operations": 1,
    "total_input_tokens": 2500,
    "total_output_tokens": 8000,
    "total_cost_cents": 0
  }
}
```

### Weekly Analysis Pipeline

```python
class TraceAnalyzer:
    def __init__(self, trace_storage, llm):
        self.storage = trace_storage
        self.llm = llm
    
    def analyze_week(self, week_start):
        """Analyze all traces from a given week"""
        traces = self.storage.get_traces(week_start, week_start + timedelta(days=7))
        
        if not traces:
            return None
        
        analysis = {
            "week": week_start.isoformat(),
            "total_traces": len(traces),
            "success_rate": self.compute_success_rate(traces),
            "avg_duration_ms": np.mean([t.total_duration_ms for t in traces]),
            "avg_steps": np.mean([len(t.steps) for t in traces]),
            "tool_usage": self.analyze_tool_usage(traces),
            "common_patterns": self.extract_common_patterns(traces),
            "successful_patterns": self.extract_successful_patterns(traces),
            "failure_patterns": self.extract_failure_patterns(traces),
            "improvement_opportunities": self.identify_improvements(traces)
        }
        
        # Generate behavioral templates from successful traces
        templates = self.generate_templates(traces)
        analysis["new_templates"] = templates
        
        return analysis
    
    def extract_common_patterns(self, traces):
        """Find common sequences of steps across traces"""
        sequences = [self.extract_step_sequence(t) for t in traces]
        
        # Find frequent subsequences
        patterns = mine_frequent_subsequences(sequences, min_support=0.3)
        
        return [
            {
                "pattern": p.sequence,
                "frequency": p.support,
                "avg_outcome": p.avg_outcome
            }
            for p in patterns
        ]
    
    def generate_templates(self, traces):
        """Generate behavioral templates from successful traces"""
        successful = [t for t in traces if t.outcome.success]
        
        templates = []
        for trace in successful:
            template = self.trace_to_template(trace)
            if template.is_generalizable():
                templates.append(template)
        
        return templates
    
    def trace_to_template(self, trace):
        """Convert a successful trace into a reusable template"""
        prompt = f"""
Convert this successful execution trace into a reusable behavioral template:

Task: {trace.task_description}
Steps: {format_steps(trace.steps)}
Outcome: {trace.outcome.description}

Create a template that captures the general approach, not the specific details.
Include:
1. When to use this template (preconditions)
2. Step-by-step procedure
3. Common pitfalls to avoid
4. Success criteria
"""
        template_text = self.llm.generate(prompt, max_tokens=500)
        
        return BehavioralTemplate(
            name=f"Template: {trace.task_description[:50]}",
            preconditions=self.extract_preconditions(trace),
            steps=self.extract_generalized_steps(trace),
            pitfalls=self.extract_pitfalls(trace),
            success_criteria=trace.outcome.description,
            source_trace_id=trace.id
        )
```

### Behavioral Template Storage

```yaml
# Example behavioral template
template:
  id: "tmpl_research_spec_001"
  name: "Research Spec Generation"
  preconditions:
    - "Task requires producing a detailed technical specification"
    - "Research on a topic is needed before writing"
  steps:
    - "Identify key research areas and reference papers"
    - "Survey relevant literature and extract key findings"
    - "Design architecture based on research findings"
    - "Write spec with implementation details, pseudocode, and integration plan"
    - "Include success criteria, risks, and mitigation strategies"
    - "Mark deliverable as ready for review"
  pitfalls:
    - "Don't write pure academic content — always include implementation specs"
    - "Don't skip the research phase — specs without research are guesses"
    - "Don't forget success criteria — engineering needs measurable targets"
  success_criteria: "Spec is actionable by engineering team with clear deliverables"
  source_traces: ["trace_20260404_001", "trace_20260404_002"]
  usage_count: 0
  success_rate: null
```

---

## 3. Offline Optimization (MIPROv2/GEPA)

### Problem

Luke's prompts and behavioral instructions are static. They don't improve based on performance data, even though systematic prompt optimization can yield significant improvements.

### Solution: Automated Prompt Optimization Pipeline

### Prompt Performance Tracking

```python
class PromptPerformanceTracker:
    def __init__(self, storage):
        self.storage = storage
        self.prompt_scores = defaultdict(list)  # prompt_id -> [scores]
    
    def record_execution(self, prompt_id, prompt_version, input_data, output_data, outcome):
        """Record a prompt execution with outcome"""
        score = self.score_outcome(outcome)
        
        record = {
            "prompt_id": prompt_id,
            "prompt_version": prompt_version,
            "timestamp": datetime.now(),
            "input_hash": hash(input_data),
            "output_length": len(output_data),
            "score": score,
            "outcome_type": outcome.type
        }
        
        self.storage.save(record)
        self.prompt_scores[prompt_id].append(score)
    
    def score_outcome(self, outcome):
        """Score an outcome on a 0-1 scale"""
        if outcome.success:
            base = 0.8
        else:
            base = 0.2
        
        # Adjust based on quality indicators
        if outcome.user_satisfied:
            base += 0.1
        if outcome.completed_on_time:
            base += 0.05
        if outcome.no_corrections_needed:
            base += 0.05
        
        return min(1.0, base)
    
    def get_performance(self, prompt_id, window_days=30):
        """Get performance stats for a prompt"""
        scores = [s for s in self.prompt_scores[prompt_id] 
                  if s["timestamp"] > datetime.now() - timedelta(days=window_days)]
        
        if not scores:
            return None
        
        return {
            "prompt_id": prompt_id,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "sample_size": len(scores),
            "trend": self.compute_trend(scores)
        }
```

### Optimization Pipeline

Weekly batch optimization of underperforming prompts:

```python
class PromptOptimizer:
    def __init__(self, tracker, llm, storage):
        self.tracker = tracker
        self.llm = llm
        self.storage = storage
    
    def optimize_weekly(self):
        """Weekly optimization run"""
        # Find underperforming prompts
        all_prompts = self.storage.list_prompts()
        performances = {
            p.id: self.tracker.get_performance(p.id)
            for p in all_prompts
        }
        
        # Select prompts for optimization
        candidates = []
        for prompt_id, perf in performances.items():
            if perf and perf["mean_score"] < 0.7 and perf["sample_size"] >= 5:
                candidates.append((prompt_id, perf))
        
        # Sort by improvement potential (low score, high variance = most room for improvement)
        candidates.sort(key=lambda x: x[1]["mean_score"] + x[1]["std_score"])
        
        optimized = []
        for prompt_id, perf in candidates[:5]:  # Optimize top 5 worst performers
            result = self.optimize_prompt(prompt_id, perf)
            optimized.append(result)
        
        return optimized
    
    def optimize_prompt(self, prompt_id, performance):
        """Optimize a single prompt using MIPROv2-style approach"""
        prompt = self.storage.get_prompt(prompt_id)
        
        # Gather execution examples
        examples = self.storage.get_examples(prompt_id, limit=20)
        
        # Generate candidate variants
        candidates = self.generate_variants(prompt, examples)
        
        # Evaluate candidates on held-out examples
        held_out = examples[-5:]  # Last 5 as test set
        training = examples[:-5]
        
        scores = []
        for candidate in candidates:
            score = self.evaluate_candidate(candidate, training)
            scores.append((candidate, score))
        
        # Select best candidate
        best_candidate, best_score = max(scores, key=lambda x: x[1])
        
        # If best candidate is significantly better, promote it
        if best_score > performance["mean_score"] + 0.1:
            self.storage.promote_prompt_variant(prompt_id, best_candidate)
            return {
                "prompt_id": prompt_id,
                "old_score": performance["mean_score"],
                "new_score": best_score,
                "improvement": best_score - performance["mean_score"],
                "variant_id": best_candidate.id
            }
        
        return {
            "prompt_id": prompt_id,
            "old_score": performance["mean_score"],
            "new_score": best_score,
            "improvement": 0,
            "status": "no_improvement"
        }
    
    def generate_variants(self, prompt, examples):
        """Generate candidate prompt variants"""
        variants = []
        
        # Variant 1: Add examples (few-shot)
        variants.append(self.add_examples_variant(prompt, examples[:3]))
        
        # Variant 2: Restructure instructions
        variants.append(self.restructure_variant(prompt))
        
        # Variant 3: Add explicit constraints
        variants.append(self.add_constraints_variant(prompt))
        
        # Variant 4: LLM-generated optimization
        variants.append(self.llm_optimize_variant(prompt, examples))
        
        return variants
    
    def evaluate_candidate(self, candidate, examples):
        """Evaluate a candidate prompt on training examples"""
        scores = []
        for example in examples:
            output = self.llm.generate_with_prompt(candidate, example.input)
            score = self.score_output(output, example.expected_output)
            scores.append(score)
        return np.mean(scores)
```

### A/B Testing Framework

Deploy optimized prompts alongside originals:

```python
class ABTestFramework:
    def __init__(self, storage):
        self.storage = storage
        self.active_tests = {}
    
    def start_test(self, prompt_id, variant_id, traffic_split=0.5):
        """Start an A/B test for a prompt variant"""
        test_id = generate_id()
        self.active_tests[test_id] = ABTest(
            id=test_id,
            prompt_id=prompt_id,
            control_version=prompt_id,
            variant_version=variant_id,
            traffic_split=traffic_split,
            start_time=datetime.now(),
            control_results=[],
            variant_results=[]
        )
        return test_id
    
    def assign_variant(self, test_id):
        """Assign control or variant based on traffic split"""
        test = self.active_tests[test_id]
        if random.random() < test.traffic_split:
            return "variant"
        return "control"
    
    def record_result(self, test_id, variant, score):
        """Record a test result"""
        test = self.active_tests[test_id]
        if variant == "control":
            test.control_results.append(score)
        else:
            test.variant_results.append(score)
    
    def check_significance(self, test_id):
        """Check if the test has reached statistical significance"""
        test = self.active_tests[test_id]
        
        if len(test.control_results) < 30 or len(test.variant_results) < 30:
            return {"significant": False, "reason": "insufficient_samples"}
        
        # T-test
        t_stat, p_value = ttest_ind(test.control_results, test.variant_results)
        
        if p_value < 0.05:
            effect_size = (
                np.mean(test.variant_results) - np.mean(test.control_results)
            )
            return {
                "significant": True,
                "p_value": p_value,
                "effect_size": effect_size,
                "winner": "variant" if effect_size > 0 else "control"
            }
        
        return {"significant": False, "p_value": p_value}
    
    def conclude_test(self, test_id):
        """Conclude the test and promote winner if significant"""
        result = self.check_significance(test_id)
        
        if result["significant"] and result["winner"] == "variant":
            test = self.active_tests[test_id]
            self.storage.promote_prompt(test.prompt_id, test.variant_version)
            return {"action": "promoted_variant", "result": result}
        
        return {"action": "kept_control", "result": result}
```

---

## 4. Skill Loop Evolution

### Problem

Skills (reusable behavioral patterns) are defined once and never evolve. There's no mechanism for improving skills based on performance data or adapting them to new contexts.

### Solution: Evolutionary Skill Management

### Skill Versioning

```python
class SkillVersion:
    def __init__(self, skill_id, version, definition, metadata=None):
        self.skill_id = skill_id
        self.version = version
        self.definition = definition  # The actual skill definition
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.status = "active"  # active, deprecated, experimental
        self.performance_metrics = {}
    
    def record_performance(self, metric_name, value):
        """Record a performance metric for this version"""
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        self.performance_metrics[metric_name].append({
            "value": value,
            "timestamp": datetime.now()
        })
```

### Mutation Operators

Systematic ways to modify skills:

```python
class SkillMutator:
    def __init__(self, llm):
        self.llm = llm
    
    def mutate(self, skill, operator, context=None):
        """Apply a mutation operator to a skill"""
        operators = {
            "add_step": self.add_step,
            "remove_step": self.remove_step,
            "reorder_steps": self.reorder_steps,
            "add_condition": self.add_condition,
            "simplify": self.simplify,
            "elaborate": self.elaborate,
            "add_error_handling": self.add_error_handling,
            "optimize_prompt": self.optimize_prompt
        }
        
        return operators[operator](skill, context)
    
    def add_step(self, skill, context):
        """Add a new step to the skill"""
        prompt = f"""
Given this skill:
{skill.definition}

And this context about where it fails:
{context}

Suggest one additional step that would improve this skill.
Return only the new step in the same format as existing steps.
"""
        new_step = self.llm.generate(prompt, max_tokens=200)
        return skill.with_added_step(new_step)
    
    def add_error_handling(self, skill, context):
        """Add error handling to the skill"""
        prompt = f"""
Given this skill:
{skill.definition}

And these failure modes:
{context}

Add error handling steps for each failure mode.
Return the updated skill definition.
"""
        updated = self.llm.generate(prompt, max_tokens=500)
        return skill.with_definition(updated)
    
    def simplify(self, skill, context):
        """Simplify the skill by removing unnecessary steps"""
        prompt = f"""
Given this skill:
{skill.definition}

Remove any steps that are redundant, unnecessary, or could be combined.
Return the simplified skill definition.
"""
        simplified = self.llm.generate(prompt, max_tokens=500)
        return skill.with_definition(simplified)
```

### Fitness Evaluation

```python
class SkillFitnessEvaluator:
    def __init__(self, test_cases):
        self.test_cases = test_cases
    
    def evaluate(self, skill_version):
        """Evaluate a skill version against test cases"""
        results = []
        for test in self.test_cases:
            result = self.run_test(skill_version, test)
            results.append(result)
        
        return FitnessResult(
            overall_score=np.mean([r.score for r in results]),
            pass_rate=sum(1 for r in results if r.passed) / len(results),
            avg_duration=np.mean([r.duration_ms for r in results]),
            details=results
        )
    
    def run_test(self, skill_version, test_case):
        """Run a single test case against a skill version"""
        start = time.time()
        output = skill_version.execute(test_case.input)
        duration = (time.time() - start) * 1000
        
        score = test_case.score(output)
        passed = score >= test_case.pass_threshold
        
        return TestResult(
            test_id=test_case.id,
            passed=passed,
            score=score,
            duration_ms=duration,
            output=output
        )
```

### Selection Mechanism

```python
class SkillSelector:
    def __init__(self, evaluator, mutator, storage):
        self.evaluator = evaluator
        self.mutator = mutator
        self.storage = storage
    
    def evolve(self, skill_id, generations=5, population_size=5):
        """Evolve a skill through multiple generations"""
        current = self.storage.get_active_skill(skill_id)
        baseline_fitness = self.evaluator.evaluate(current)
        
        best = current
        best_fitness = baseline_fitness
        
        for gen in range(generations):
            # Generate population
            population = [current]
            operators = ["add_step", "add_error_handling", "simplify", "elaborate", "add_condition"]
            
            for op in operators[:population_size-1]:
                try:
                    mutant = self.mutator.mutate(current, op)
                    population.append(mutant)
                except Exception:
                    pass
            
            # Evaluate population
            fitnesses = []
            for variant in population:
                fitness = self.evaluator.evaluate(variant)
                fitnesses.append((variant, fitness))
            
            # Select best
            best_variant, best_gen_fitness = max(fitnesses, key=lambda x: x[1].overall_score)
            
            if best_gen_fitness.overall_score > best_fitness.overall_score:
                best = best_variant
                best_fitness = best_gen_fitness
        
        # If improved, create new version
        if best_fitness.overall_score > baseline_fitness.overall_score + 0.05:
            new_version = self.storage.create_skill_version(
                skill_id=skill_id,
                definition=best.definition,
                parent_version=current.version
            )
            
            return EvolutionResult(
                skill_id=skill_id,
                original_fitness=baseline_fitness,
                new_fitness=best_fitness,
                improvement=best_fitness.overall_score - baseline_fitness.overall_score,
                new_version=new_version.version,
                generations_run=generations
            )
        
        return EvolutionResult(
            skill_id=skill_id,
            original_fitness=baseline_fitness,
            new_fitness=best_fitness,
            improvement=0,
            status="no_improvement"
        )
```

---

## 5. Integration Plan

### Phase 1: Reflexion (Week 1-3)
- Implement post-action reflection in agent loop
- Build reflection storage and retrieval
- Add error pattern detection (daily batch job)
- **Owner:** Engineer
- **Research support:** This spec

### Phase 2: Execution Traces (Week 3-5)
- Implement trace logging in agent loop
- Build trace storage and query interface
- Create weekly analysis pipeline
- **Owner:** Engineer
- **Research support:** Pattern extraction algorithms

### Phase 3: Prompt Optimization (Week 5-7)
- Implement prompt performance tracking
- Build optimization pipeline (MIPROv2-style)
- Add A/B testing framework
- **Owner:** Engineer + Solutions Engineer
- **Research support:** Optimization strategy evaluation

### Phase 4: Skill Evolution (Week 7-10)
- Implement skill versioning
- Build mutation operators
- Create fitness evaluation framework
- Add selection mechanism
- **Owner:** Engineer + Solutions Engineer
- **Research support:** Mutation operator design

### Phase 5: Integration & Measurement (Week 10-12)
- Connect all components into closed loop
- Deploy measurement dashboard
- Tune parameters based on production data
- **Owner:** QA Engineer + Engineer
- **Research support:** System-wide optimization analysis

---

## 6. Success Criteria

| Criterion | Measurement | Target |
|---|---|---|
| Repeated error rate | Same error type occurrences / week | <10% of previous rate |
| Trace analysis yield | Actionable improvements per week | >=3 |
| Prompt optimization success | Optimized prompts that pass A/B test | >=20% improvement |
| Skill evolution success | Skills improved through evolution | >=1 skill in 60 days |
| Self-correction accuracy | Correct corrections / total corrections | >80% |
| False positive rate | Unnecessary corrections / total corrections | <10% |

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Reflexion generates noise (false insights) | Minimum evidence threshold (3+ occurrences) before acting |
| Execution traces consume too much storage | Sampling strategy, retention policy (90 days for detailed, summary forever) |
| Prompt optimization overfits to test cases | Held-out test set, A/B validation before deployment |
| Skill evolution produces regressions | Fitness evaluation against full test suite before promotion |
| Self-correction creates infinite loops | Max correction depth (3), circuit breaker on repeated corrections |
| Optimization consumes too much compute | Budget caps, off-peak scheduling, small model for evaluation |

---

## 8. Appendix: Paper Summaries

### Reflexion (Shinn et al.)
**Finding:** Language agents that reflect on their failures and store verbal reflections as memory show significant improvement on sequential decision-making tasks. The key insight is that self-reflection provides a learning signal without requiring external rewards.
**Implication for Luke:** Implement lightweight reflection after every significant action. Store reflections as memories for pattern detection.

### MIPROv2
**Finding:** Automatic prompt optimization can significantly improve LLM performance without human intervention. The key is using a bootstrap-then-optimize approach: first gather demonstrations, then optimize the prompt using them.
**Implication for Luke:** Track prompt performance continuously, optimize weekly using accumulated examples.

### GEPA (Generative Prompt Adaptation)
**Finding:** Generative approaches to prompt adaptation outperform discrete search methods. Using an LLM to generate prompt variants based on performance feedback is more efficient than grid search.
**Implication for Luke:** Use LLM-generated variants for prompt optimization, not manual search.

### DSPy
**Finding:** Declarative programming of LM calls (specifying what, not how) enables automatic compilation and optimization of LM pipelines. The abstraction separates the program logic from the prompt text.
**Implication for Luke:** Consider adopting DSPy-style declarative skill definitions for easier optimization.
