#!/usr/bin/env python3
"""Phase 4.2 verification — the Letta backend replaces the memory core + state, NOT the shell.

The migration invariant (plan Phase 4.2): flipping ``agent_backend``/``memory_backend`` to
``letta`` swaps *only* the memory core (store/self-edit surface) and the per-turn working-context
source. The shell — planner, attention budget, scheduler, media, dbwriter — must be **agnostic**
to the backend: same code path, same behavior, whether Luke runs on sqlite or Letta.

This is a static + import audit, so it is fully **bridge-independent** and runs **off the OAuth
pool** (no Claude turn). It proves the invariant three ways from ground truth:

  (A) BLAST RADIUS — neither backend flag is *referenced* in any shell module. Uses the AST
      (not grep) so comments/strings can't produce false negatives: we walk every module and
      flag a real ``settings.<flag>`` attribute access.
  (B) SEAM COMPLETENESS — the set of modules that DO branch on a backend flag is exactly the
      known integration seams (memory core + context assembly). A new, unaudited branch point
      anywhere else fails the test — this is the regression guard.
  (C) IMPORT AGNOSTICISM — each shell module imports cleanly and its public surface is byte-for-
      byte identical with the backend forced to "sdk" vs "letta". Import-time branching on the
      flag would change the surface; it doesn't.

Run: python3 scripts/probe_42_shell_isolation.py   (from repo root, app venv)
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "luke"

# The shell: subsystems Letta must NOT touch. Letta replaces memory core + state only.
SHELL_MODULES = ["planner", "attention", "scheduler", "media", "dbwriter"]

# The known, sanctioned integration seams that ARE allowed to branch on a backend flag.
# Any OTHER module that branches on a flag is an unaudited seam → regression.
SANCTIONED_SEAMS = {
    "agent",          # agent.py:1939 — working_ctx source (4.1)
    "memory",         # memory.py:766 — semantic source (letta adapter)
    "letta_writer",   # write-through (2.2c)
    "letta_adapter",  # recall read side
    "letta_agent",    # context assembly (4.1)
    "config",         # where the flags are DEFINED (not a branch)
}

BACKEND_FLAGS = {"agent_backend", "memory_backend"}


def _settings_flag_accesses(tree: ast.AST) -> set[str]:
    """Return the set of backend flags accessed as ``settings.<flag>`` in an AST.

    AST-based so a flag name inside a comment or docstring never counts — only a real
    attribute load on a ``settings`` name.
    """
    hits: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in BACKEND_FLAGS
            and isinstance(node.value, ast.Name)
            and node.value.id == "settings"
        ):
            hits.add(node.attr)
    return hits


def _module_flag_map() -> dict[str, set[str]]:
    """Map every luke module name → the backend flags it references (via AST)."""
    out: dict[str, set[str]] = {}
    for py in sorted(SRC.glob("*.py")):
        name = py.stem
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError as e:
            print(f"  ! parse error in {name}: {e}")
            continue
        hits = _settings_flag_accesses(tree)
        if hits:
            out[name] = hits
    return out


def check_a_blast_radius(flag_map: dict[str, set[str]]) -> bool:
    print("(A) BLAST RADIUS — no backend flag referenced in any shell module")
    ok = True
    for mod in SHELL_MODULES:
        hits = flag_map.get(mod, set())
        status = "clean" if not hits else f"LEAK {sorted(hits)}"
        print(f"    {mod:12s} {status}")
        if hits:
            ok = False
    return ok


def check_b_seam_completeness(flag_map: dict[str, set[str]]) -> bool:
    print("(B) SEAM COMPLETENESS — only sanctioned seams branch on a backend flag")
    branching = set(flag_map) - {"config"}  # config defines the flags, doesn't branch
    unaudited = branching - SANCTIONED_SEAMS
    for mod in sorted(branching):
        tag = "sanctioned" if mod in SANCTIONED_SEAMS else "UNAUDITED SEAM"
        print(f"    {mod:14s} {sorted(flag_map[mod])}  {tag}")
    if unaudited:
        print(f"    ! unaudited backend branch points: {sorted(unaudited)}")
        return False
    return True


def _public_surface(mod_name: str) -> list[str]:
    """Import a shell module fresh and return its sorted public attribute names."""
    full = f"luke.{mod_name}"
    if full in sys.modules:
        del sys.modules[full]
    mod = importlib.import_module(full)
    return sorted(n for n in dir(mod) if not n.startswith("_"))


def check_c_import_agnosticism() -> bool:
    print("(C) IMPORT AGNOSTICISM — shell public surface identical sdk vs letta")
    ok = True
    for mod in SHELL_MODULES:
        try:
            os.environ["AGENT_BACKEND"] = "sdk"
            os.environ["MEMORY_BACKEND"] = "sqlite"
            surf_sdk = _public_surface(mod)
            os.environ["AGENT_BACKEND"] = "letta"
            os.environ["MEMORY_BACKEND"] = "letta"
            surf_letta = _public_surface(mod)
        except Exception as e:
            print(f"    {mod:12s} IMPORT ERROR: {e}")
            ok = False
            continue
        identical = surf_sdk == surf_letta
        print(f"    {mod:12s} {'identical' if identical else 'DIVERGED'} ({len(surf_sdk)} names)")
        if not identical:
            ok = False
    os.environ.pop("AGENT_BACKEND", None)
    os.environ.pop("MEMORY_BACKEND", None)
    return ok


def main() -> int:
    print("=== Phase 4.2 — shell isolation from the Letta backend ===\n")
    flag_map = _module_flag_map()
    a = check_a_blast_radius(flag_map)
    print()
    b = check_b_seam_completeness(flag_map)
    print()
    c = check_c_import_agnosticism()
    print()
    passed = a and b and c
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}  (A={a} B={b} C={c})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
