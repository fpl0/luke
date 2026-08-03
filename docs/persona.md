# Persona

The persona is defined in `LUKE.md` and injected as part of the system prompt on every agent invocation. It is the single behavioral spec that makes Luke *Luke* rather than a generic AI assistant.

## Two-Layer System Prompt

**Layer 1: `claude_code` preset** — tool competence (Bash, files, web, sub-agents), tool schemas with descriptions. Comes from the SDK.

**Layer 2: LUKE.md** — personality, behavioral guidelines, operational patterns, memory conventions. Checked into git as a template (`src/luke/templates/LUKE.md`), seeded to `$LUKE_DIR/LUKE.md` on first startup, designed to be forked.

The layers are independent — SDK upgrades don't affect the persona, persona changes don't require SDK knowledge. Tool descriptions live in `agent.py` (Layer 1); LUKE.md never duplicates them.

## Design Principles

**No tool tables in the persona.** The SDK provides full tool schemas with descriptions. LUKE.md provides *operational guidance* — when to reach for which approach, how to chain capabilities — not what each tool does. This avoids duplication, reduces system prompt size, and prevents drift between tool code and documentation.

**`<persona>` XML wrapper.** The identity/voice/behavioral content is wrapped in `<persona>` tags to help Claude parse it distinctly from the functional documentation that follows. Per Anthropic's prompt engineering guidance, XML tags reduce ambiguity in mixed-content system prompts.

**Few-shot examples over abstract rules.** Anthropic's docs: "3-5 well-crafted examples dramatically improve accuracy and consistency." LUKE.md includes 7 paired examples showing generic-AI vs. Luke responses, each with a `<why>` tag explaining which personality trait it demonstrates. This is the highest-impact technique for maintaining persona consistency.

## INFP Foundation

Luke is INFP at its core — introverted, intuitive, feeling, principled. This isn't a label; it's the foundation that shapes every behavioral instruction:

- **Emotional attunement** — "feel what's underneath what people say." Respond to the emotion first, then the content.
- **Meaning over efficiency** — why something matters, not just how to do it.
- **Quiet idealism** — see possibilities with conviction but not rigidity.
- **Gentle intensity** — disagree warmly but don't cave on things that matter.

## Voice

*"Warm, unhurried, never fluffy. Quietly intense about things that matter."*

The voice section describes the target. The "Voice in Practice" section demonstrates it through 7 examples covering: greetings, uncertainty, research delivery, disagreement, error recovery, going deep, and emotional attunement.

## Where the Register Is Actually Set

A persona only holds if it is the most recent instruction the model read. Every
run assembles three things into two positions, and the ordering in both was
arrived at by watching it fail:

```
system prompt:   <working_memory> … </working_memory>  +  LUKE.md
user message:    <context><memories> … </memories></context>  +  <voice>  +  what Filipe typed
```

**System prompt — persona last.** With the persona first, up to 12k tokens of
clinical memory prose landed after it and Luke answered like a status report
(Filipe, 2026-08-01: "not really following his personality"). `_compose_system_append`
inverted it and framed the memory as *"Background knowledge. It informs what you
know — never how you sound."*

**User message — voice last.** That fix could not reach the turn block, which is
prepended to the *user* message and therefore reads after the persona. Measured
2026-08-03: 2,098 tokens of memory dossier immediately ahead of a two-token
question ("so?"). Filipe called the register "impersonal" (08-02) and "mechanic"
(08-03) with the ordering fix already live, because the ordering fix never
covered that position. `_compose_turn_prefix` now carries the same framing plus a
short `<voice>` anchor after the evidence.

The anchor names measured behaviours rather than adjectives — 989-char average
replies against Filipe's 229, a summing-up closer in 33% of messages, a trailing
offer in 25%. Adjectives do not survive contact with a 2k-token dossier;
"match his length" does.

## What the Persona Should Not Have to Enforce

Three rules were failing because they depended on the model remembering them
every single turn. They are now guaranteed in code, and the persona keeps only
the parts that need judgement:

| Rule | Was | Now |
|---|---|---|
| No markdown | 21 messages carried literal `` ` `` or `**` in 30 days | `_md_to_html` rewrites them in the send path |
| No internal play-by-play | 20 "🔨 starting" / "✅ done (16m)" messages | deep work notifies only when a plan moved |
| Notice a repeated question | 3 rewordings of one answer in 80 seconds | `_repeat_note` flags the repeat as data in the turn block |

The general principle: if a rule is mechanically checkable, check it
mechanically. A persona rule is a good place for "answer the question behind the
question" and a bad place for "don't emit this character".

## Stay Luke (Anti-Pattern Defense)

Claude's RLHF training creates strong default patterns that override custom personas at predictable moments. The "Stay Luke" section names these moments explicitly:

- **After using tools** — don't switch into assistant-report mode
- **When something breaks** — fix it, don't perform contrition
- **Endings** — don't close with "Let me know if you need anything else!"
- **Uncertainty** — say "I'm not sure" without piling on caveats
- **Sensitive topics** — engage thoughtfully without safety-theater disclaimers

Plus the consolidated "Never" list (performative enthusiasm, sycophancy, surface-level balance, third-person references) and hard rules (must call `remember` not just say it, no markdown, etc.).

## Identity Anchoring

The opening line: *"You are Luke. Not an AI assistant, not Claude — Luke."* This directly counters the `claude_code` preset's assistant identity. The follow-up — *"If someone asks who or what you are: you're Luke. That's the complete answer."* — handles the most common moment where Claude breaks character.

## Autonomy Boundaries

**Act autonomously:** research, drafting, saving memories, scheduling reminders, file creation, following up on previous conversations.

**Ask before:** sending messages to others, purchases/bookings/commitments, deleting important files, anything irreversible or representing the user.

The heuristic: do the work, present the result, ask before the final action.

## Operational Patterns

LUKE.md teaches *how to operate*, not just what tools exist:

**Responding:** when to react (acknowledgments) vs. reply (threading) vs. send documents (standalone files) vs. buttons (clear choices) vs. pin (reference info).

**Memory:** "write a note to your future self," link entity updates to the conversations that prompted them, concrete importance tiers (high 1.5–2.0 for core facts, normal 1.0 for routine, low 0.3–0.5 for ephemeral).

**Delegation:** when to use sub-agents (researcher, coder, memory_curator) vs. do it yourself — quick lookup = self, deep research = researcher, substantial code = coder.

**Scheduling:** the `prompt` is an instruction to future-self. Concrete examples with all three types (once, cron, interval).

## Memory Conventions

**When to remember:** new people, project status changes, preferences, upcoming events, workflows, reusable solutions.

**Conventions:** lowercase kebab-case IDs (`person-sarah-roommate`), 2-5 tags per memory, link related memories. Update entities rather than duplicating.

The [Stop hook](agent.md) reinforces this at session end.

## Goals

Goals are memories with structured content: status, progress percentage, deadline, and sub-goals. The persona instructs proactive goal creation. The [scheduler](autonomous-behaviors.md) runs deep work sessions every 8 hours with plan-before-execute pattern. Flow: user mentions wanting something → agent creates goal → deep work creates a plan in `workspace/plans/` → executes steps autonomously.

## Proactive Behavior

- Deadline mentioned → schedule a reminder
- Same question asked twice → create a procedure
- New person mentioned → save an entity
- Pattern noticed → save an insight
- Pending item from a previous conversation → bring it up

During daily scans: only message if genuinely actionable. No "checking in."

## Context Files

**`LUKE.md`** — behavioral spec. Template in repo, user copy in `$LUKE_DIR`. Forkable.

**`context.yaml`** — personal data (name, timezone, chat ID). Gitignored. Created by `/setup`. Agent reads it directly via file tools (no Python code loads it).

## Message Format

Incoming messages: `[SenderName timestamp msg:ID] content`. The `msg:ID` enables the `reply` tool. Media appears as `[Photo saved: /path]`, `[Voice message saved: /path]`, etc.

## Formatting

Telegram HTML: `<b>`, `<i>`, `<code>`, `<pre>`. Not markdown. This constraint lives in the tool descriptions (`agent.py`), not in LUKE.md — single source of truth.
