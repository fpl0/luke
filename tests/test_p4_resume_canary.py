"""Tests for the P4 resume canary's judgement, not for the SDK.

The API-calling half of `scripts/p4_resume_canary.py` is the experiment; it is exercised
by running it. What is tested here is everything that decides what a run MEANS, because
that is where the 14 Aug failure lived: a complete, plausible, four-cell table saying the
opposite of the truth, produced by a subprocess with no OAuth token. Three properties
carry that lesson:

  * a control cell that does not see the transcript makes its arm INCONCLUSIVE (exit 2),
    never a FAIL — the harness lost the probe, not the tier;
  * the auth-expiry shape (`is_error=True`, all-zero usage, `subtype='success'`) reads
    BLIND, never FAIL;
  * every arm structurally has a control, so a future cell cannot be added without one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import anyio
import pytest
from claude_agent_sdk import ResultMessage

_SPEC = importlib.util.spec_from_file_location(
    "p4_resume_canary",
    Path(__file__).resolve().parent.parent / "scripts" / "p4_resume_canary.py",
)
assert _SPEC and _SPEC.loader
canary = importlib.util.module_from_spec(_SPEC)
# Register before exec: @dataclass resolves annotations through sys.modules, and a module
# that isn't there yet fails collection with a bare AttributeError.
sys.modules["p4_resume_canary"] = canary
_SPEC.loader.exec_module(canary)


def cell(name, arm, resume_model, verdict, *, control=False, reason=""):
    spec = canary.CellSpec(
        name, arm, "opus", resume_model, compact=(arm == "compacted"), control=control
    )
    return canary.CellResult(spec=spec, verdict=verdict, reason=reason)


def _result(**kw) -> ResultMessage:
    base = dict(
        subtype="success",
        duration_ms=1200,
        duration_api_ms=900,
        is_error=False,
        num_turns=1,
        session_id="s",
        usage={"input_tokens": 900, "output_tokens": 40},
    )
    base.update(kw)
    return ResultMessage(**base)


# ------------------------------------------------------------------ arm-level verdicts


def test_all_cells_green_is_a_silent_zero():
    code, headline, notes = canary.summarise(
        [
            cell("A", "mcp", "opus", canary.SEES, control=True),
            cell("B", "mcp", "sonnet", canary.SEES),
        ]
    )
    assert code == 0
    assert "GREEN" in headline
    assert notes == []


def test_control_blind_makes_the_arm_inconclusive_not_a_failure():
    """The 14 Aug shape. Without this rule the run reports FAIL on a broken harness."""
    code, headline, notes = canary.summarise(
        [
            cell("A", "mcp", "opus", canary.BLIND, control=True),
            cell("B", "mcp", "sonnet", canary.BLIND),
        ]
    )
    assert code == 2
    assert "!!RESUME_CANARY_BLIND!!" in headline
    assert any("CONTROL" in n for n in notes)


def test_control_green_and_cheap_cell_blind_is_the_real_failure():
    code, headline, _ = canary.summarise(
        [
            cell("A", "mcp", "opus", canary.SEES, control=True),
            cell("B", "mcp", "sonnet", canary.BLIND),
        ]
    )
    assert code == 1
    assert "FAIL" in headline


def test_unobserved_cell_blinds_its_arm_even_with_a_green_control():
    """A compaction that did not happen is not evidence that resume works."""
    code, headline, notes = canary.summarise(
        [
            cell("C", "compacted", "opus", canary.SEES, control=True),
            cell("D", "compacted", "sonnet", canary.UNOBSERVED, reason="compaction did not happen"),
        ]
    )
    assert code == 2
    assert "!!RESUME_CANARY_BLIND!!" in headline
    assert any("compaction did not happen" in n for n in notes)


def test_a_real_failure_outranks_a_blind_arm():
    code, headline, _ = canary.summarise(
        [
            cell("A", "mcp", "opus", canary.SEES, control=True),
            cell("B", "mcp", "sonnet", canary.BLIND),
            cell("C", "compacted", "opus", canary.UNOBSERVED, control=True),
            cell("D", "compacted", "sonnet", canary.UNOBSERVED),
        ]
    )
    assert code == 1
    assert "FAIL on mcp" in headline


def test_arms_are_judged_independently():
    code, headline, _ = canary.summarise(
        [
            cell("A", "mcp", "opus", canary.SEES, control=True),
            cell("B", "mcp", "sonnet", canary.SEES),
            cell("C", "compacted", "opus", canary.SEES, control=True),
            cell("D", "compacted", "sonnet", canary.BLIND),
        ]
    )
    assert code == 1
    assert "compacted" in headline and "mcp" not in headline


# ------------------------------------------------------------------ the auth-expiry shape


def test_auth_shape_is_error_with_all_zero_usage():
    res = _result(is_error=True, subtype="success", usage={"input_tokens": 0, "output_tokens": 0})
    assert canary._is_auth_shape(res, "some text") is True


def test_auth_shape_is_caught_in_plain_assistant_text_with_no_error_flag():
    """14 Aug: the CLI returned the OAuth message as ordinary text, no exception raised."""
    text = "Failed to authenticate: OAuth session expired and could not be refreshed"
    assert canary._is_auth_shape(_result(), text) is True


def test_a_healthy_turn_is_not_the_auth_shape():
    assert canary._is_auth_shape(_result(), "PROBE=PRB-1234") is False


def test_a_genuine_error_with_real_usage_is_not_the_auth_shape():
    res = _result(is_error=True, usage={"input_tokens": 5000, "output_tokens": 12})
    assert canary._is_auth_shape(res, "tool loop exceeded") is False


def test_missing_result_is_not_asserted_as_auth_failure():
    assert canary._is_auth_shape(None, "PROBE=PRB-1234") is False


# ------------------------------------------------------------------------- due-ness


def _state(days_ago: float, version: str = "0.2.128") -> dict:
    return {
        "last_run": (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(),
        "sdk_version": version,
    }


def test_never_run_is_due():
    due, why = canary.is_due({})
    assert (due and "sdk version changed" in why) or due


def test_sdk_version_change_is_due_immediately(monkeypatch):
    monkeypatch.setattr(canary, "sdk_version", lambda: "0.3.0")
    due, why = canary.is_due(_state(0.1))
    assert due
    assert "sdk version changed" in why


def test_a_week_old_run_is_due(monkeypatch):
    monkeypatch.setattr(canary, "sdk_version", lambda: "0.2.128")
    due, _ = canary.is_due(_state(7.5))
    assert due


def test_a_fresh_run_on_the_same_sdk_is_not_due(monkeypatch):
    monkeypatch.setattr(canary, "sdk_version", lambda: "0.2.128")
    due, why = canary.is_due(_state(1))
    assert not due
    assert "unchanged" in why


def test_unparseable_last_run_is_due(monkeypatch):
    monkeypatch.setattr(canary, "sdk_version", lambda: "0.2.128")
    due, why = canary.is_due({"last_run": "whenever", "sdk_version": "0.2.128"})
    assert due
    assert "unparseable" in why


# ------------------------------------------------------------- structure of the matrix


def test_every_arm_has_a_control_cell():
    arms = {c.arm for c in canary.CELLS}
    for arm in arms:
        assert any(c.control for c in canary.CELLS if c.arm == arm), arm


def test_every_arm_has_a_cheap_probe_cell():
    arms = {c.arm for c in canary.CELLS}
    for arm in arms:
        probes = [c for c in canary.CELLS if c.arm == arm and not c.control]
        assert probes, arm
        assert all(c.resume_model != c.seed_model for c in probes), arm


def test_controls_resume_the_model_that_seeded_them():
    for c in canary.CELLS:
        if c.control:
            assert c.seed_model == c.resume_model


def test_the_two_arms_differ_only_in_compaction():
    mcp = {(c.seed_model, c.resume_model) for c in canary.CELLS if c.arm == "mcp"}
    compacted = {(c.seed_model, c.resume_model) for c in canary.CELLS if c.arm == "compacted"}
    assert mcp == compacted
    assert all(not c.compact for c in canary.CELLS if c.arm == "mcp")
    assert all(c.compact for c in canary.CELLS if c.arm == "compacted")


def test_haiku_is_not_a_cell():
    """opus->sonnet only. Haiku on a rich transcript is a quality change, not a routing
    change — the 14 Aug handover's constraint."""
    assert all(c.resume_model in {"opus", "sonnet"} for c in canary.CELLS)


def test_the_seeded_tool_is_outside_the_sonnet_tier_today():
    assert canary.tier_gap_holds()


# ------------------------------------------------------------------- the tool surface


def _call(tools, name, args):
    handler = next(t.handler for t in tools if t.name == name)
    return anyio.run(handler, args)


def _text(payload) -> str:
    return "".join(block["text"] for block in payload["content"])


def test_the_probe_token_appears_only_in_the_seed_tool_result():
    tools = canary.build_tools("PRB-DEADBEEF")
    assert "PRB-DEADBEEF" in _text(_call(tools, canary.SEED_TOOL, {"operation": "audit"}))
    assert "PRB-DEADBEEF" not in _text(_call(tools, "recall", {"query": "x"}))
    assert "PRB-DEADBEEF" not in _text(_call(tools, "remember", {"id": "a", "content": "b"}))
    assert all(
        "PRB-DEADBEEF" not in (t.description + json.dumps(str(t.input_schema))) for t in tools
    )


def test_the_resume_side_server_refuses_every_tool():
    """The transcript must be the only path to the token — otherwise a resumed turn can
    re-earn the answer by calling the tool again, and the cell passes while blind."""
    tools = canary.build_tools(None)
    for name, args in [
        (canary.SEED_TOOL, {"operation": "audit"}),
        ("recall", {"query": "x"}),
        ("remember", {"id": "a", "content": "b"}),
    ]:
        assert _text(_call(tools, name, args)) == canary.DISABLED_TEXT


def test_both_server_variants_expose_the_same_tool_names():
    assert [t.name for t in canary.build_tools("PRB-X")] == [
        t.name for t in canary.build_tools(None)
    ]


def test_the_system_append_carries_no_token_and_is_production_sized():
    assert "PRB-" not in canary._SYSTEM_APPEND
    assert len(canary._SYSTEM_APPEND) > 10_000


def test_the_brake_is_the_file_the_routing_change_reads():
    assert canary.BRAKE_PATH.name == "cheap_resume.off"


@pytest.mark.parametrize("code", [0, 1, 2])
def test_state_round_trips(tmp_path, monkeypatch, code):
    monkeypatch.setattr(canary, "STATE_PATH", tmp_path / "state.json")
    canary.save_state({"exit_code": code, "sdk_version": "0.2.128"})
    assert canary.load_state()["exit_code"] == code


def test_load_state_survives_a_corrupt_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    monkeypatch.setattr(canary, "STATE_PATH", path)
    assert canary.load_state() == {}
