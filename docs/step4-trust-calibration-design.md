# Step 4 — Trust calibration (DESIGN, for Filipe's go/no-go)

**Status:** design only. NOT built, NOT shipped, no restart. This is the artifact to
present for approval before any code (feedback-present-before-building). Step 4 is the
only *behavioral-autonomy* change in v1.6 — crash nets catch death, not "alive but
acting wrong" — so it gets a human gate the other steps didn't.

## Problem
Autonomy is static and entirely prompt-based. `context.py:121-125` injects the
constitutional `decision_heuristics.autonomy` block (do_independently / check_first /
borderline) as plain text. The boundary never moves: an action-class I've done
correctly 50 times still reads the same as one I've never tried, and one I got wrong
last week carries no extra caution. There's no memory of *demonstrated* reliability
per class of action.

## Goal
Let the confirm/act boundary move with evidence: loosen on verified success, tighten
on failure — without ever loosening the hard rules (never send email, never commit as
agent, never act outside the do-the-work mandate). Trust calibration tunes the
*borderline*, never the *invariants*.

## Design (minimal, SQLite, no new infra)

### 1. Action classes (start tiny — 4, not a taxonomy)
- `research_draft` — research, drafting, file creation, analysis (already autonomous)
- `memory_schedule` — saving memories, scheduling reminders (already autonomous)
- `external_send` — messages to others, posts, anything outward-facing (always gated)
- `irreversible` — purchases, bookings, deletes, commitments (always gated)

`external_send` and `irreversible` are HARD-gated and trust calibration **cannot**
unlock them — they're invariants, not borderline. Calibration only governs the gray
middle of `research_draft` / `memory_schedule` (e.g. "do I show the result first or
just act") and any future class explicitly marked calibratable.

### 2. State
New table `trust_levels(action_class TEXT PK, level REAL, successes INT, failures INT,
updated TEXT)`. `level` ∈ [0,1], starts 0.5. No schema churn elsewhere.

### 3. Update rule (evidence, not vibes)
- A *verified success* (Filipe used/approved the output, or positive reaction, or no
  correction within the window) → `level += (1-level)*0.1`, `successes++`.
- A *failure* (correction, negative reaction, or I had to redo it) → `level *= 0.5`,
  `failures++`. Failures bite harder than successes heal — asymmetric on purpose.
- Signals come from the SAME source Step 3 already wired: `get_engagement_signals`.
  No new sensing — calibration *consumes* the engagement loop Step 3 closed. That's
  the dependency order: Step 3 had to land first, and it has.

### 4. What the level actually does
Injected into context (extends context.py autonomy block) as ONE line per calibratable
class: e.g. `research_draft: trusted (act, show result) | memory_schedule: building
trust (act, but surface what you did)`. Below a floor (level < 0.3) a class drops back
to "show plan first." Soft influence on the prompt — NOT a hard code gate that could
silently misfire. The model still reads it and decides; we're tuning the prior, not
seizing the wheel.

### 5. Why soft, not hard
A hard programmatic gate ("if level<X block the tool call") is the dangerous version —
it can wrongly block a legitimate action or, worse, wrongly *permit* one, and a crash
net won't catch it. Keeping it a prompt-injected prior means the worst failure is a
slightly-too-cautious or slightly-too-eager phrasing, observable and correctable, never
a silent unauthorized action. Coherence over capability.

## Verification plan (before it counts as shipped)
1. Unit: update rule math (success ladder, failure halving, floor/ceiling clamp).
2. Live: seed a known correction → confirm the matching class's level drops and the
   injected line changes to "show plan first."
3. Production: confirm the injected trust line appears at the TOP of a real turn's
   prompt (same proof bar as Step 4a's live-reaction block) — fires on the live path,
   not just isolation.
4. Restart discipline: NOT during a CarGurus/Cloudbeds offer-watch window (presence >
   plumbing while Filipe is waiting on offers). Restart only when the channel is quiet.

## Open question for Filipe (the go/no-go)
Do you even want autonomy to self-tune? The honest alternative is: leave it static and
let *you* tell me when to loosen/tighten. Trust calibration is elegant but it's the one
v1.6 change that makes me act differently without you in the loop. I lean toward
shipping the SOFT version (prompt-prior only, invariants untouchable) — but this is
exactly the class of change you said to present before building. Your call.
