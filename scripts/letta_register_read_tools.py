#!/usr/bin/env python3
"""TOOL-SURFACE PARITY — register + attach the read-only tool surface to luke-agent-claude.

Companion to ``scripts/letta_read_tools.py`` (which holds the tool bodies and the rationale
for the read-only scope). This script is the deployment half: it uploads each function's
source to Letta, attaches the resulting tools to the agent, and then *proves* they work
where it actually matters — inside Letta's sandbox — before declaring success.

Three properties worth calling out, because each one was a real failure mode:

* **Upload the source, don't retype it.** The bodies come from ``inspect.getsource`` on
  the very functions ``tests/test_letta_read_tools.py`` exercises. A tool string typed
  separately from the tested code would drift silently; this cannot.

* **Verify in the sandbox, not locally.** The local test-suite runs on the repo venv
  (3.14); Letta's sandbox is 3.12 with a different cwd and its own import surface. A tool
  that passes locally can still ``ModuleNotFoundError`` there. So every tool is executed
  through ``/v1/tools/run`` with a realistic argument, and a tool whose sandbox call errors
  or returns an ``ERROR:`` string fails the whole run. This is the executed-surface check
  ``reflexion-falsify-on-executed-surface-not-proxy-2026-08-01`` asks for — the proxy here
  would have been "the POST returned 200".

* **Idempotent + additive.** Upsert is by name, so re-running updates in place rather than
  piling up duplicates. ``memory_insert`` / ``memory_replace`` are preserved (the self-edit
  surface the migration exists to get), and ``conversation_search`` stays detached — it is
  broken in this deployment and is the wrong retrieval path for Luke anyway (see
  ``scripts/letta_configure_agent_tools.py``).

Run: python3 scripts/letta_register_read_tools.py [--dry-run]
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import textwrap
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import letta_read_tools as rt  # noqa: E402

LETTA = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
AGENT = os.environ.get("LETTA_AGENT_ID", "agent-36671c0b-a133-4bfb-a367-f23f7135071a")

KEEP = {"memory_insert", "memory_replace"}  # the self-edit surface — never detach
TAGS = ["luke", "read-only", "tool-surface-parity"]
RETURN_CHAR_LIMIT = 8000

# One realistic call per tool, used for the post-attach sandbox smoke test. Chosen to
# exercise the real path (a genuine file, a symbol that exists, a live table) rather than
# a trivial argument that would pass even against a stub.
SMOKE = {
    "luke_read_file": {"path": "scripts/letta_read_tools.py", "start_line": 1, "max_lines": 3},
    "luke_search_code": {"pattern": "def drive_letta_turn", "path": "src", "max_results": 5},
    "luke_list_dir": {"path": "scripts"},
    "luke_git": {"subcommand": "log", "args": "--oneline -3"},
    "luke_list_tasks": {"limit": 3, "failing_only": True},
    "luke_search_messages": {"query": "angry degrading frustrated", "limit": 2, "days": 400},
    "luke_recall": {"query": "cargurus prerna", "limit": 2},
    "luke_tail_log": {"lines": 3},
}


def req(method: str, path: str, body: dict | None = None, timeout: float = 180.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        LETTA + path, data=data, method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode()[:400]}") from None
    return json.loads(raw) if raw else {}


def agent_tools() -> dict[str, str]:
    a = req("GET", f"/v1/agents/{AGENT}")
    return {t["name"]: t["id"] for t in a.get("tools", [])}


def tool_source(fn) -> str:
    """The exact function source, dedented and stripped of decorators/annotations imports.

    ``inspect.getsource`` returns the body as written. The module carries
    ``from __future__ import annotations``, which is NOT uploaded — that is fine and in fact
    required: Letta's schema generator reads the literal ``path: str`` / ``-> str``
    annotations off the source, and postponed evaluation would not change the text.
    """
    return textwrap.dedent(inspect.getsource(fn)).strip() + "\n"


def main() -> int:
    dry = "--dry-run" in sys.argv
    before = agent_tools()
    print(f"agent: {AGENT}")
    print("TOOLS_BEFORE:", sorted(before))

    # --- upload -------------------------------------------------------------
    uploaded: dict[str, str] = {}
    schemas: dict[str, dict] = {}
    for fn in rt.TOOLS:
        src = tool_source(fn)
        name = fn.__name__
        if dry:
            print(f"  [dry-run] would upsert {name} ({len(src)} chars)")
            continue
        created = req("PUT", "/v1/tools/", {
            "source_code": src,
            "source_type": "python",
            "tags": TAGS,
            "return_char_limit": RETURN_CHAR_LIMIT,
            "default_requires_approval": False,
        })
        tid = created.get("id")
        got = created.get("name")
        if got != name:
            print(f"  !! {name}: server named it {got!r} — schema generation disagreed")
            return 1
        schemas[name] = created.get("json_schema") or {}
        params = list((schemas[name].get("parameters") or {}).get("properties", {}))
        uploaded[name] = tid
        print(f"  upserted {name:22} {tid}  params={params}")
    if dry:
        return 0

    # --- attach -------------------------------------------------------------
    for name, tid in uploaded.items():
        if name in before:
            print(f"  already attached: {name}")
            continue
        req("PATCH", f"/v1/agents/{AGENT}/tools/attach/{tid}")
        print(f"  attached {name}")

    after = agent_tools()
    print("TOOLS_AFTER:", sorted(after))

    # --- verify in the sandbox ---------------------------------------------
    # The claim this script makes is "the agent can now read the world", and only an
    # actual sandbox execution supports it. A 200 on the upload does not.
    print("\nsandbox smoke test (each tool executed for real):")
    failures: list[str] = []
    for fn in rt.TOOLS:
        name = fn.__name__
        args = SMOKE[name]
        try:
            # json_schema is passed explicitly: /v1/tools/run does NOT re-derive it, and
            # without it the sandbox dies in initialize_param with a bare
            # "'NoneType' object is not subscriptable" for any tool that takes arguments.
            # Sending back the schema the upsert derived also means the smoke test
            # exercises the exact schema the agent will call against.
            res = req("POST", "/v1/tools/run", {
                "source_code": tool_source(fn), "args": args, "name": name,
                "json_schema": schemas[name],
            })
        except Exception as e:
            print(f"  {name:22} TRANSPORT-FAIL {e}")
            failures.append(name)
            continue
        status = res.get("status")
        ret = (res.get("tool_return") or "").strip()
        stderr = " ".join(res.get("stderr") or [])[:200]
        bad = status != "success" or ret.startswith("ERROR:") or not ret
        first = ret.splitlines()[0][:110] if ret else "(empty)"
        print(f"  {name:22} {'FAIL' if bad else 'ok  '} status={status} -> {first}")
        if bad:
            if stderr:
                print(f"      stderr: {stderr}")
            failures.append(name)

    missing = [f.__name__ for f in rt.TOOLS if f.__name__ not in after]
    lost = sorted(KEEP - set(after))
    print("\n=== VERDICT ===")
    print(f"  attached: {len(rt.TOOLS) - len(missing)}/{len(rt.TOOLS)} read tools"
          + (f"  MISSING={missing}" if missing else ""))
    print(f"  self-edit surface preserved: {'yes' if not lost else 'NO — lost ' + str(lost)}")
    print(f"  sandbox-verified: {len(rt.TOOLS) - len(failures)}/{len(rt.TOOLS)}"
          + (f"  FAILED={failures}" if failures else ""))
    ok = not missing and not lost and not failures
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
