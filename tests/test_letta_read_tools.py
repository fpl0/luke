"""Tests for the Letta read-only tool surface (TOOL-SURFACE PARITY).

These functions run inside Letta's sandbox as ``filipelm`` with full machine access —
the guards in each tool are the only boundary between a model-chosen argument and the
filesystem. So the escape cases are tested first and hardest: a passing "it reads a file"
test proves nothing if ``../../../.ssh/id_rsa`` also reads.

They are ordinary functions here, which is the point of writing them as a module and
uploading via ``inspect.getsource`` rather than as opaque strings — the exact code that
gets registered is the code exercised below.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import letta_read_tools as rt  # noqa: E402

REPO = rt.REPO_DIR
HAVE_DB = os.path.isfile(rt.DB_PATH)
HAVE_REPO = os.path.isdir(REPO)

import pytest  # noqa: E402

needs_repo = pytest.mark.skipif(not HAVE_REPO, reason="Luke source repo not present")
needs_db = pytest.mark.skipif(not HAVE_DB, reason="luke.db not present")


# --- containment: the guards, before anything else --------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "/Users/filipelm/.ssh/id_rsa",
        "../../../../etc/hosts",
        "/Users/filipelm/Luke/../.claude/.credentials.json",
        "/tmp",
    ],
)
def test_read_file_refuses_paths_outside_the_roots(path):
    out = rt.luke_read_file(path)
    assert out.startswith("ERROR:"), f"escaped containment with {path!r}: {out[:120]}"


@pytest.mark.parametrize(
    "path",
    ["/Users/filipelm/Code/luke/.env", "/Users/filipelm/Luke/.env", "/Users/filipelm/Code/luke/.git/config"],
)
def test_read_file_refuses_secrets_inside_the_roots(path):
    """Being inside the allowlist is not sufficient — secrets are denied on top of it."""
    assert rt.luke_read_file(path).startswith("ERROR:")


@needs_repo
def test_search_code_refuses_paths_outside_the_roots():
    assert rt.luke_search_code("root", path="/etc").startswith("ERROR:")


def test_list_dir_refuses_paths_outside_the_roots():
    assert rt.luke_list_dir("/Users/filipelm/.ssh").startswith("ERROR:")


@pytest.mark.parametrize("sub", ["push", "commit", "reset", "clean", "checkout", "rm", "config"])
def test_git_refuses_write_subcommands(sub):
    assert rt.luke_git(sub).startswith("ERROR:")


def test_git_refuses_shell_metacharacters():
    out = rt.luke_git("log", "--oneline; rm -rf /tmp/x")
    assert out.startswith("ERROR:") and "metacharacter" in out


# --- capability: each tool actually answers something -----------------------


@needs_repo
def test_read_file_returns_numbered_window():
    out = rt.luke_read_file("scripts/letta_read_tools.py", start_line=1, max_lines=5)
    assert not out.startswith("ERROR:")
    body = out.splitlines()
    assert body[0].endswith("scripts/letta_read_tools.py")
    assert body[1].startswith("1\t")
    # header + 5 lines + the truncation marker, which must say where to resume so the
    # agent can page rather than assume it saw the whole file.
    assert len(body) == 7
    assert body[-1].startswith("… (truncated") and "line 6" in body[-1]


@needs_repo
def test_read_file_respects_start_line():
    out = rt.luke_read_file("scripts/letta_read_tools.py", start_line=40, max_lines=2)
    assert "\n40\t" in out


@needs_repo
def test_search_code_finds_a_known_symbol():
    out = rt.luke_search_code(r"def drive_letta_turn", path="src")
    assert not out.startswith("ERROR:")
    assert "letta_agent.py" in out


@needs_repo
def test_search_code_reports_a_clean_miss_not_an_error():
    """A miss must read as a miss — an ERROR would make the agent think the tool broke."""
    out = rt.luke_search_code("zzz_no_such_symbol_zzz", path="src")
    assert out.startswith("(no matches")


@needs_repo
def test_list_dir_lists_the_repo_and_hides_vcs_internals():
    out = rt.luke_list_dir("")
    assert "scripts" in out and "src" in out
    names = {ln.split("  ")[-1] for ln in out.splitlines()[1:]}
    assert names.isdisjoint({".git", "node_modules", "__pycache__", ".venv", ".letta-venv"})


@needs_repo
def test_git_log_returns_history():
    out = rt.luke_git("log", "--oneline -3")
    assert not out.startswith("ERROR:")
    assert len(out.splitlines()) <= 3 and out.strip()


@needs_db
def test_list_tasks_returns_rows():
    out = rt.luke_list_tasks(limit=3)
    assert not out.startswith("ERROR:")


@needs_db
def test_list_tasks_filter_is_applied():
    out = rt.luke_list_tasks(status="completed", limit=3)
    assert out.startswith("(no tasks") or "[completed]" in out


@needs_db
def test_list_tasks_failing_only_surfaces_failures_recency_would_hide():
    """The default listing is recency-ordered and capped, so a job that has been failing
    for a while falls outside the window — found during live verification, where the
    currently-failing task was not in the default 25 rows."""
    out = rt.luke_list_tasks(failing_only=True, limit=10)
    assert not out.startswith("ERROR:")
    if out.startswith("(no tasks"):
        return
    # Every returned row must actually be a failure, and ordering must be worst-first.
    counts = [int(p.split("fails=")[1].split()[0]) for p in out.splitlines() if "fails=" in p]
    assert counts and all(c > 0 for c in counts), out
    assert counts == sorted(counts, reverse=True), f"not failure-ordered: {counts}"
    # The guarantee that matters: every failing task is reachable here regardless of how
    # far down the recency-ordered default listing it has slipped.
    import sqlite3

    con = sqlite3.connect(f"file:{rt.DB_PATH}?mode=ro", uri=True)
    expected = {r[0] for r in con.execute("SELECT id FROM tasks WHERE consecutive_failures > 0")}
    con.close()
    assert expected <= {ln.split("  ")[0].split("] ")[1] for ln in out.splitlines() if ln.startswith("[")}


@needs_db
def test_search_messages_finds_a_known_term():
    out = rt.luke_search_messages("cargurus", limit=3, days=400)
    assert not out.startswith("ERROR:")
    if not out.startswith("(no messages"):
        assert "cargurus" in out.lower()


@needs_db
def test_search_messages_ors_terms_so_a_paraphrase_still_hits():
    """The AND version returned nothing whenever one term was absent, which made the agent
    report that a message it could not word-match did not exist. Verified against the real
    26 July message: "your answer quality are degrading" contains "degrading" but none of
    the words someone would guess for it."""
    out = rt.luke_search_messages("angry degrading frustrated", limit=10, days=400)
    assert not out.startswith("ERROR:")
    assert not out.startswith("(no messages"), "OR semantics did not recover the message"
    assert "degrading" in out.lower()


@needs_db
def test_search_messages_ranks_by_match_count():
    out = rt.luke_search_messages("cargurus prerna", limit=10, days=400)
    if out.startswith("(no messages"):
        return
    scores = [int(ln.split(" [")[1].split("/")[0]) for ln in out.splitlines() if "/2 terms]" in ln]
    # Printed oldest-first, so the best match is last in the rendering.
    assert scores == sorted(scores), f"not ranked by match count: {scores}"


@needs_db
def test_search_messages_scopes_to_a_single_day():
    out = rt.luke_search_messages("", on_date="2026-07-26", limit=5)
    assert not out.startswith("ERROR:")
    if not out.startswith("(no messages"):
        assert all(ln.startswith("[2026-07-26") for ln in out.splitlines())


@needs_db
def test_search_messages_rejects_an_empty_request():
    assert rt.luke_search_messages("  ").startswith("ERROR:")


@needs_db
def test_recall_returns_archive_passages():
    out = rt.luke_recall("cargurus prerna", limit=3)
    assert not out.startswith("ERROR:")
    if not out.startswith("(nothing"):
        assert "<mem id=" in out


@needs_db
def test_recall_survives_fts_hostile_input():
    """Bare FTS operators used to raise; the LIKE fallback must absorb them."""
    out = rt.luke_recall('OR AND "unbalanced', limit=2)
    assert not out.startswith("ERROR:")


def test_tail_log_rejects_an_unknown_log():
    assert rt.luke_tail_log(log="/etc/passwd").startswith("ERROR:")


@pytest.mark.skipif(not os.path.isfile("/Users/filipelm/Luke/luke.log"), reason="no runtime log")
def test_tail_log_returns_bounded_output():
    out = rt.luke_tail_log(lines=5)
    assert not out.startswith("ERROR:")
    assert len(out.splitlines()) <= 5


# --- registration contract --------------------------------------------------


def test_every_tool_is_self_contained():
    """Letta uploads the function source alone, so a module-level reference would
    NameError at execution time. Every import and constant must be inside the body."""
    import inspect

    for fn in rt.TOOLS:
        src = inspect.getsource(fn)
        lines = [ln for ln in src.splitlines() if ln.strip()]
        assert lines[0].startswith("def "), f"{fn.__name__} carries a decorator"
        for name in ("RUNTIME_DIR", "REPO_DIR", "DB_PATH"):
            assert name not in src, f"{fn.__name__} references module-level {name}"


def test_no_tool_contains_a_nested_function():
    """Letta's schema generator walks the uploaded source and cannot cope with more than
    one def: with a nested helper present it first rejected the upload ("Function _resolve
    missing docstring") and then, once documented, registered the tool under the HELPER's
    name. Both failures only surface at registration, so pin the constraint here."""
    import ast
    import inspect
    import textwrap

    for fn in rt.TOOLS:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        defs = [
            getattr(n, "name", "<lambda>")
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        ]
        assert defs == [fn.__name__], f"{fn.__name__} must be the only def in its source; found {defs}"


def test_every_tool_has_a_schema_generating_docstring():
    """Letta derives the JSON schema from the docstring; a missing Args block ships a
    tool the model cannot call correctly."""
    for fn in rt.TOOLS:
        doc = fn.__doc__ or ""
        assert "Args:" in doc and "Returns:" in doc, f"{fn.__name__} docstring is incomplete"
        sig = __import__("inspect").signature(fn)
        for p in sig.parameters:
            assert f"{p} (" in doc, f"{fn.__name__} does not document parameter {p}"
        # `from __future__ import annotations` stringifies these locally; the uploaded
        # source carries the literal `-> str` that Letta's schema generator reads.
        assert sig.return_annotation in (str, "str"), f"{fn.__name__} must return str"


def test_no_tool_can_write():
    """The read-only guarantee is what makes attaching these safe during a live replay —
    assert it structurally rather than trusting review."""
    import inspect

    banned = ("open(", "shutil", "os.remove", "os.unlink", "os.rename", "os.mkdir", "subprocess.run")
    for fn in rt.TOOLS:
        src = inspect.getsource(fn)
        if "open(" in src:
            # Reads are fine; a write mode is not.
            assert '"w"' not in src and "'w'" not in src and '"a"' not in src, fn.__name__
        if "subprocess" in src:
            assert "shell=True" not in src, f"{fn.__name__} spawns a shell"
        assert "mode=ro" in src or "sqlite3" not in src, f"{fn.__name__} opens sqlite writable"
        assert not any(b in src for b in banned[1:6]), f"{fn.__name__} touches a mutating call"
