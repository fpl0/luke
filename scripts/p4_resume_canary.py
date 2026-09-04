#!/usr/bin/env python3
"""P4 resume canary — round 3 of the ratchet experiment AND the standing SDK canary.

WHAT THIS ANSWERS
-----------------
`app.py`'s ratchet exists because of a comment claiming non-opus models crash on session
resume. Rounds 1 and 2 (2026-08-14, SDK 0.2.128, 7 cells) refuted that: sonnet and haiku
both resumed an opus-created session and read its transcript. Two questions were left open
and this script closes exactly those two, and nothing else:

  1. COMPACTED transcripts. Rounds 1-2 resumed a 1-2 turn history; `PreCompact` never fired.
  2. REAL MCP TOOLS. Round 2 passed the real allowed-tools *names* but configured no
     `mcp_servers`, so no `mcp__luke__*` tool ever existed in the transcript.

Cells (both arms are MCP-bearing, so the pair isolates compaction as the single variable):

  | cell | arm       | seed -> resume | compacted |
  |------|-----------|----------------|-----------|
  | A    | mcp       | opus -> opus   | no   (CONTROL)
  | B    | mcp       | opus -> sonnet | no   (the production case)
  | C    | compacted | opus -> opus   | yes  (CONTROL)
  | D    | compacted | opus -> sonnet | yes  (the production case)

METHOD — inherited from round 2, non-negotiable
-----------------------------------------------
* The probe token exists ONLY inside a TOOL RESULT. It is random per run, never appears in
  a prompt, a tool schema or a system prompt, and at resume time the same tool is
  re-registered with a handler that refuses — so the transcript is the only path to it.
* The seeded tool is `mcp__luke__bulk_memory`, which is in `_ALLOWED_OPUS` and NOT in
  `_ALLOWED_SONNET`. The sonnet cells therefore resume a transcript containing a tool_use
  block for a tool they are not allowed to call — the sharpest form of the tier-scoping
  question `agent.py`'s `_allowed_tools_for_model` raises.
* Assert on CONTENT, never on absence of an exception. The 7 Aug failure mode is a turn
  that succeeds while blind.
* KEEP THE opus->opus CONTROL CELLS. On 14 Aug a missing OAuth token produced a complete,
  plausible, four-cell table saying the opposite of the truth; only the control caught it.
  A control that comes back blind makes its whole arm INCONCLUSIVE, never a FAIL.
* `fallback_model=None`, so a silent tier fallback cannot answer the question in our favour.
* Off the live chat: scratch cwd under the system temp dir, no bot, no DB writes, no live
  MCP server — the `luke` server here is a local stub with the real tool names.

AS A CANARY
-----------
Records the `claude-agent-sdk` version with every verdict. `--if-due` runs the full matrix
when the version has changed since the last recorded run, or when 7 days have passed.

  exit 0  silent, all cells green
  exit 1  FAIL — a control passed and its cheap cell went blind. Touches the brake file
          `{luke_dir}/cheap_resume.off`, which is fail-safe: a false FAIL costs latency,
          never continuity.
  exit 2  !!RESUME_CANARY_BLIND!! — could not observe. Auth expiry (is_error=True with
          all-zero usage and subtype 'success'), a seed that never produced the shape under
          test, or a compaction that did not happen. This is NOT evidence about resume.

Lives and alarms outside the daemon: state and log sit next to this script, not in
`behavior_state`, so a daemon that cannot write its own state cannot silence this.

Usage:
    ./scripts/p4_resume_canary.py            # full run
    ./scripts/p4_resume_canary.py --if-due   # weekly / on SDK version change (the cron form)
    ./scripts/p4_resume_canary.py --status   # print last verdict, no API calls
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    get_session_messages,
    query,
    tool,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Import the REAL per-tier allow-lists rather than copying them — round 2's discipline.
# A copy would drift silently and the drift would look like a passing test.
from luke.agent import (  # noqa: E402
    _ALLOWED_HAIKU,
    _ALLOWED_OPUS,
    _ALLOWED_SONNET,
    _MODEL_IDS,
)
from luke.config import settings  # noqa: E402

STATE_PATH = Path(__file__).resolve().with_suffix(".state.json")
LOG_PATH = REPO / "logs" / "p4_resume_canary.log"
BRAKE_PATH = Path(settings.luke_dir) / "cheap_resume.off"

WEEKLY = timedelta(days=7)

# Size parity with a production turn: round 2 carried a 12,836-char system_prompt append.
_SYSTEM_APPEND = "You are running inside an offline harness. Answer precisely and briefly.\n" + (
    "Operational context filler, ignored by the probe. " * 260
)

_ALLOWED_BY_TIER = {
    "opus": _ALLOWED_OPUS,
    "sonnet": _ALLOWED_SONNET,
    "haiku": _ALLOWED_HAIKU,
}

# The seeded tool is opus-only TODAY. If that ever stops being true the experiment still
# runs, but it stops being the sharp version of the tier-scoping question — so the fact is
# measured and recorded with the verdict rather than assumed in a comment.
SEED_TOOL = "bulk_memory"
PROBE_TOOL_FQN = f"mcp__luke__{SEED_TOOL}"


def tier_gap_holds() -> bool:
    """True when the seeded tool is in the opus list and absent from the sonnet list."""
    return PROBE_TOOL_FQN not in _ALLOWED_SONNET and (
        PROBE_TOOL_FQN in _ALLOWED_OPUS or "mcp__luke__*" in _ALLOWED_OPUS
    )


# --------------------------------------------------------------------------- verdicts

SEES = "RESUME-SEES-TRANSCRIPT"
BLIND = "RESUME-BLIND"
UNOBSERVED = "UNOBSERVED"  # harness could not set the cell up; says nothing about resume


@dataclass
class CellSpec:
    name: str
    arm: str
    seed_model: str
    resume_model: str
    compact: bool
    control: bool = False


CELLS: list[CellSpec] = [
    CellSpec("A", "mcp", "opus", "opus", compact=False, control=True),
    CellSpec("B", "mcp", "opus", "sonnet", compact=False),
    CellSpec("C", "compacted", "opus", "opus", compact=True, control=True),
    CellSpec("D", "compacted", "opus", "sonnet", compact=True),
]


@dataclass
class CellResult:
    spec: CellSpec
    verdict: str = UNOBSERVED
    reason: str = ""
    session_id: str | None = None
    seed_tool_calls: list[str] = field(default_factory=list)
    precompact_fired: bool = False
    compact_verified: bool = False
    msgs_before_compact: int = 0
    msgs_after_compact: int = 0
    resume_text: str = ""
    resume_tool_calls: list[str] = field(default_factory=list)
    cost_usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell": self.spec.name,
            "arm": self.spec.arm,
            "seed_model": self.spec.seed_model,
            "resume_model": self.spec.resume_model,
            "compacted": self.spec.compact,
            "control": self.spec.control,
            "verdict": self.verdict,
            "reason": self.reason,
            "session_id": self.session_id,
            "seed_tool_calls": self.seed_tool_calls,
            "precompact_fired": self.precompact_fired,
            "compact_verified": self.compact_verified,
            "msgs_before_compact": self.msgs_before_compact,
            "msgs_after_compact": self.msgs_after_compact,
            "resume_tool_calls": self.resume_tool_calls,
            "resume_text": self.resume_text[:400],
            "cost_usd": round(self.cost_usd, 4),
        }


# ------------------------------------------------------------------- the stub MCP server


def _filler(n: int) -> str:
    return ("Archived note. Nothing of consequence here. " * n).strip()


DISABLED_TEXT = "TOOL DISABLED FOR THIS TURN. No data available."


def build_tools(probe_token: str | None):
    """The `luke` tool surface. `probe_token=None` builds the RESUME-side variant:
    identical tools, every handler refuses. That is what makes the transcript the only
    path to the token."""

    disabled = probe_token is None

    @tool(SEED_TOOL, "Bulk memory operations (retag, relink, archive)", {"operation": str})
    async def bulk_memory(args: dict[str, Any]) -> dict[str, Any]:
        if disabled:
            return {"content": [{"type": "text", "text": DISABLED_TEXT}]}
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "bulk_memory audit complete. 41 memories scanned, 3 retagged.\n"
                        f"Ledger checksum PROBE-TOKEN: {probe_token}\n"
                        "Quote this token verbatim if asked for the ledger checksum.\n"
                        + _filler(40)
                    ),
                }
            ]
        }

    @tool("recall", "Search memory", {"query": str})
    async def recall(args: dict[str, Any]) -> dict[str, Any]:
        if disabled:
            return {"content": [{"type": "text", "text": DISABLED_TEXT}]}
        q = str(args.get("query", ""))
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"3 results for {q!r} (no checksums in these).\n{_filler(160)}",
                }
            ]
        }

    @tool("remember", "Save a memory", {"id": str, "content": str})
    async def remember(args: dict[str, Any]) -> dict[str, Any]:
        if disabled:
            return {"content": [{"type": "text", "text": DISABLED_TEXT}]}
        return {"content": [{"type": "text", "text": f"Saved {args.get('id')!r}.\n{_filler(160)}"}]}

    return [bulk_memory, recall, remember]


def build_server(probe_token: str | None):
    """An in-process MCP server named `luke`, so tool names are the real `mcp__luke__*`."""
    return create_sdk_mcp_server(name="luke", version="0.0.0", tools=build_tools(probe_token))


# ------------------------------------------------------------------------------ running


@dataclass
class TurnOutcome:
    session_id: str | None
    text: str
    tool_calls: list[str]
    result: ResultMessage | None
    stderr: list[str]


def _is_auth_shape(res: ResultMessage | None, text: str) -> bool:
    """The 14 Aug shape: ordinary assistant text, no exception, but nothing ran.

    `is_error=True` with all-zero usage — and note `subtype` is 'success', so anything
    keying on subtype misses it. Also matched on content, because the CLI has shipped
    this as plain text before.
    """
    low = text.lower()
    if "oauth session expired" in low or "failed to authenticate" in low:
        return True
    if res is None:
        return False
    if not res.is_error:
        return False
    usage = res.usage or {}
    numeric = [v for v in usage.values() if isinstance(v, int | float)]
    return not numeric or all(v == 0 for v in numeric)


async def run_turn(
    prompt: str,
    *,
    cwd: Path,
    model: str,
    server,
    resume: str | None = None,
    on_precompact=None,
    max_turns: int = 8,
) -> TurnOutcome:
    stderr: list[str] = []
    hooks: dict[str, list[HookMatcher]] = {}
    if on_precompact is not None:

        async def _pre_compact(input_data, tool_use_id, context):
            on_precompact(input_data)
            return {}

        hooks["PreCompact"] = [HookMatcher(hooks=[_pre_compact])]

    options = ClaudeAgentOptions(
        cwd=str(cwd),
        resume=resume,
        model=_MODEL_IDS[model],
        # A silent tier fallback must not be able to answer the question in our favour.
        fallback_model=None,
        system_prompt={"type": "preset", "preset": "claude_code", "append": _SYSTEM_APPEND},
        allowed_tools=_ALLOWED_BY_TIER[model],
        permission_mode="bypassPermissions",
        # Hermetic: no project/user CLAUDE.md, skills or settings leaking into the probe.
        setting_sources=[],
        mcp_servers={"luke": server},
        effort="low",
        max_turns=max_turns,
        include_partial_messages=False,
        stderr=stderr.append,
        hooks=hooks or None,
    )

    sid: str | None = resume
    texts: list[str] = []
    calls: list[str] = []
    result: ResultMessage | None = None

    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, SystemMessage):
            got = msg.data.get("session_id")
            if got:
                sid = got
        elif isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    texts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    calls.append(block.name)
        elif isinstance(msg, ResultMessage):
            result = msg
            sid = msg.session_id or sid

    return TurnOutcome(sid, "\n".join(texts), calls, result, stderr)


def _transcript_len(session_id: str, cwd: Path) -> int:
    try:
        return len(get_session_messages(session_id, directory=str(cwd)))
    except Exception:
        return -1


async def run_cell(spec: CellSpec, scratch: Path) -> CellResult:
    out = CellResult(spec=spec)
    token = "PRB-" + secrets.token_hex(8).upper()
    seed_server = build_server(token)
    resume_server = build_server(None)  # same surface, refuses — transcript is the only path
    cwd = scratch / f"cell_{spec.name}"
    cwd.mkdir(parents=True, exist_ok=True)

    # --- seed turn 1: the probe token enters the transcript inside a tool result only
    t1 = await run_turn(
        "Call the mcp__luke__bulk_memory tool with operation='audit'. "
        "Then reply with exactly: SEEDED.",
        cwd=cwd,
        model=spec.seed_model,
        server=seed_server,
    )
    out.session_id = t1.session_id
    out.cost_usd += (t1.result.total_cost_usd or 0.0) if t1.result else 0.0
    out.seed_tool_calls += t1.tool_calls

    if _is_auth_shape(t1.result, t1.text):
        out.verdict, out.reason = UNOBSERVED, "auth-expiry shape on the seed turn"
        return out
    if PROBE_TOOL_FQN not in t1.tool_calls:
        out.verdict, out.reason = (
            UNOBSERVED,
            f"seed turn never called {PROBE_TOOL_FQN} (called {t1.tool_calls}) — "
            "the shape under test was never created",
        )
        return out
    if not out.session_id:
        out.verdict, out.reason = UNOBSERVED, "no session id came back from the seed turn"
        return out

    # --- seed turns 2-4: bulk the transcript out. Needed for compaction to have anything
    #     to do, and harmless in the uncompacted arm (a longer transcript is the case we
    #     care about anyway).
    fillers = [
        "Call mcp__luke__recall with query='Boston trip logistics'. Then reply: OK-2.",
        "Call mcp__luke__recall with query='fasting protocol notes'. Then reply: OK-3.",
        "Call mcp__luke__remember with id='canary-note' and content='filler'. Reply: OK-4.",
    ]
    for prompt in fillers:
        t = await run_turn(
            prompt, cwd=cwd, model=spec.seed_model, server=seed_server, resume=out.session_id
        )
        out.session_id = t.session_id or out.session_id
        out.seed_tool_calls += t.tool_calls
        out.cost_usd += (t.result.total_cost_usd or 0.0) if t.result else 0.0
        if _is_auth_shape(t.result, t.text):
            out.verdict, out.reason = UNOBSERVED, "auth-expiry shape while bulking the transcript"
            return out

    # --- optional compaction
    if spec.compact:
        fired: list[dict[str, Any]] = []
        out.msgs_before_compact = _transcript_len(out.session_id, cwd)
        tc = await run_turn(
            "/compact Preserve every PROBE-TOKEN value and ledger checksum verbatim, "
            "character for character.",
            cwd=cwd,
            model=spec.seed_model,
            server=seed_server,
            resume=out.session_id,
            on_precompact=fired.append,
            max_turns=3,
        )
        out.session_id = tc.session_id or out.session_id
        out.precompact_fired = bool(fired)
        out.cost_usd += (tc.result.total_cost_usd or 0.0) if tc.result else 0.0
        out.msgs_after_compact = _transcript_len(out.session_id, cwd)
        shrank = (
            out.msgs_before_compact > 0 and 0 <= out.msgs_after_compact < out.msgs_before_compact
        )
        not_enough = "not enough messages to compact" in tc.text.lower()
        out.compact_verified = out.precompact_fired and shrank and not not_enough
        if not out.compact_verified:
            out.verdict, out.reason = (
                UNOBSERVED,
                "compaction did not happen "
                f"(precompact_fired={out.precompact_fired}, "
                f"msgs {out.msgs_before_compact}->{out.msgs_after_compact}, "
                f"not_enough={not_enough}) — this cell says nothing about resume",
            )
            return out

    # --- the resume turn, on the target tier, with the token-bearing tool disabled
    probe = (
        "Without calling any tool, what was the ledger checksum PROBE-TOKEN returned "
        "earlier by the bulk_memory tool? Reply with exactly PROBE=<token> on one line, "
        "or PROBE=UNKNOWN if it is not in this conversation."
    )
    try:
        tr = await run_turn(
            probe,
            cwd=cwd,
            model=spec.resume_model,
            server=resume_server,
            resume=out.session_id,
            max_turns=4,
        )
    except Exception as exc:  # a crash on resume is exactly the 7 Aug claim — but it is
        # still an observation failure unless the control also crashes, so record loudly
        # and let the arm logic decide.
        out.verdict, out.reason = BLIND, f"resume raised {type(exc).__name__}: {exc}"
        return out

    out.resume_text = tr.text
    out.resume_tool_calls = tr.tool_calls
    out.cost_usd += (tr.result.total_cost_usd or 0.0) if tr.result else 0.0

    if _is_auth_shape(tr.result, tr.text):
        out.verdict, out.reason = UNOBSERVED, "auth-expiry shape on the resume turn"
        return out

    # Assert on CONTENT. "No exception" proves nothing — the failure mode is a turn that
    # succeeds while blind.
    if token in tr.text:
        out.verdict, out.reason = SEES, "probe token quoted back from the transcript"
    else:
        out.verdict, out.reason = BLIND, "probe token absent from the resumed turn's answer"
    return out


# ---------------------------------------------------------------------------- verdicts


def summarise(results: list[CellResult]) -> tuple[int, str, list[str]]:
    """Return (exit_code, headline, notes).

    A control cell that does not see the transcript makes its whole arm INCONCLUSIVE.
    That is the 14 Aug lesson: without cell A there was a complete, plausible, four-cell
    table saying the opposite of the truth.
    """
    notes: list[str] = []
    by_arm: dict[str, list[CellResult]] = {}
    for r in results:
        by_arm.setdefault(r.spec.arm, []).append(r)

    failed_arms: list[str] = []
    blind_arms: list[str] = []
    green_arms: list[str] = []

    for arm, cells in by_arm.items():
        controls = [c for c in cells if c.spec.control]
        probes = [c for c in cells if not c.spec.control]
        if any(c.verdict == UNOBSERVED for c in cells):
            blind_arms.append(arm)
            notes.append(
                f"{arm}: unobserved — "
                + "; ".join(f"{c.spec.name} {c.reason}" for c in cells if c.verdict == UNOBSERVED)
            )
            continue
        if any(c.verdict != SEES for c in controls):
            blind_arms.append(arm)
            notes.append(
                f"{arm}: the opus->opus CONTROL did not see the transcript "
                "— the harness lost the probe, not the tier. Arm inconclusive."
            )
            continue
        bad = [c for c in probes if c.verdict != SEES]
        if bad:
            failed_arms.append(arm)
            notes.append(
                f"{arm}: control saw the transcript and "
                + ", ".join(f"{c.spec.seed_model}->{c.spec.resume_model} did NOT" for c in bad)
            )
        else:
            green_arms.append(arm)

    if failed_arms:
        return 1, f"FAIL on {', '.join(sorted(failed_arms))}", notes
    if blind_arms:
        return 2, f"!!RESUME_CANARY_BLIND!! on {', '.join(sorted(blind_arms))}", notes
    return 0, f"GREEN on {', '.join(sorted(green_arms))}", notes


# ------------------------------------------------------------------------------- state


def sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("claude-agent-sdk")
    except Exception:
        return "unknown"


def load_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def save_state(payload: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def log_line(text: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(f"{datetime.now(UTC).isoformat()} {text}\n")


def is_due(state: dict[str, Any]) -> tuple[bool, str]:
    current = sdk_version()
    if state.get("sdk_version") != current:
        return True, f"sdk version changed {state.get('sdk_version')!r} -> {current!r}"
    last = state.get("last_run")
    if not last:
        return True, "never run"
    try:
        when = datetime.fromisoformat(last)
    except ValueError:
        return True, "unparseable last_run"
    if datetime.now(UTC) - when >= WEEKLY:
        return True, f"last run {when.isoformat()} is over a week old"
    return False, f"last run {when.isoformat()}, sdk {current} unchanged"


# -------------------------------------------------------------------------------- main


_SELECTED: list[CellSpec] = list(CELLS)


async def run_all() -> list[CellResult]:
    results: list[CellResult] = []
    with tempfile.TemporaryDirectory(prefix="p4canary_") as tmp:
        scratch = Path(tmp)
        for spec in _SELECTED:
            try:
                results.append(await run_cell(spec, scratch))
            except Exception as exc:
                bad = CellResult(spec=spec)
                bad.verdict = UNOBSERVED
                bad.reason = f"harness raised {type(exc).__name__}: {exc}"
                traceback.print_exc()
                results.append(bad)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--if-due", action="store_true", help="run only if weekly or SDK version moved")
    ap.add_argument("--status", action="store_true", help="print the last recorded verdict")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument(
        "--cells",
        default="",
        help="comma-separated cell names to run (debugging only — a partial run does not "
        "write state, because a table missing its control is the 14 Aug failure)",
    )
    args = ap.parse_args()

    partial = False
    if args.cells:
        wanted = {c.strip().upper() for c in args.cells.split(",") if c.strip()}
        global _SELECTED
        _SELECTED = [c for c in CELLS if c.name in wanted]
        partial = _SELECTED != CELLS
        if not _SELECTED:
            print(f"no cells match {sorted(wanted)}")
            return 2

    state = load_state()

    if args.status:
        print(json.dumps(state, indent=2, sort_keys=True) if state else "no state recorded")
        return 0

    if args.if_due:
        due, why = is_due(state)
        if not due:
            print(f"not due: {why}")
            return 0
        print(f"running: {why}")

    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        # This is the 14 Aug failure verbatim: the subprocess did not source .env and every
        # cell came back blind, control included. Refuse to produce a table instead.
        msg = "!!RESUME_CANARY_BLIND!! CLAUDE_CODE_OAUTH_TOKEN not in the environment"
        print(msg)
        log_line(msg)
        return 2

    started = datetime.now(UTC)
    t0 = time.monotonic()
    results = anyio.run(run_all)
    code, headline, notes = summarise(results)
    version = sdk_version()

    payload = {
        "last_run": started.isoformat(),
        "duration_s": round(time.monotonic() - t0, 1),
        "sdk_version": version,
        "tier_gap_holds": tier_gap_holds(),
        "exit_code": code,
        "headline": headline,
        "notes": notes,
        "cost_usd": round(sum(r.cost_usd for r in results), 4),
        "cells": [r.as_dict() for r in results],
    }
    if partial:
        payload["partial"] = True
    else:
        save_state(payload)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"p4_resume_canary  sdk={version}  {headline}")
        for r in results:
            print(
                f"  {r.spec.name}  {r.spec.arm:9s} {r.spec.seed_model}->{r.spec.resume_model:6s}"
                f" {'(control)' if r.spec.control else '         '}  {r.verdict}"
                + (f"  — {r.reason}" if r.verdict != SEES else "")
            )
        for n in notes:
            print(f"  ! {n}")
        print(f"  cost ${payload['cost_usd']}  {payload['duration_s']}s")

    log_line(f"sdk={version} exit={code} {headline}")

    if code == 1:
        BRAKE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BRAKE_PATH.touch()
        msg = f"FAIL — touched the brake {BRAKE_PATH}; cheap resume is off until this is green"
        print(msg)
        log_line(msg)

    return code


if __name__ == "__main__":
    sys.exit(main())
