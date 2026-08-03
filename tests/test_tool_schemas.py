"""Tests for MCP tool input schemas — optional params must stay optional.

The SDK's dict-shorthand schema builder sets ``required = list(properties)``, so
a param the handler reads with ``args.get()`` was still mandatory at the tool
boundary. That cost real calls: ``connect`` rejected every link that did not
carry ``supersedes_rel``, and the send tools forced a ``chat_id`` that
``_target()`` defaults anyway. ``_schema()`` emits the full JSON Schema instead.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import mcp.types as mt
import pytest

from luke.agent import _build_tools, _schema

AGENT_SRC = Path(__file__).resolve().parents[1] / "src" / "luke" / "agent.py"


def _registered_tools() -> dict[str, mt.Tool]:
    """List tools through the real MCP handler, not the decorator's own dict."""
    inst = _build_tools("8201332044", MagicMock())["instance"]
    handler = inst.request_handlers[mt.ListToolsRequest]
    res = asyncio.run(handler(mt.ListToolsRequest(method="tools/list")))
    return {t.name: t for t in res.root.tools}


# Params the handler resolves itself when absent. Requiring them means the model
# has to invent a value — which is how int("me@fpl0.io") happened.
EXPECTED_OPTIONAL = {
    "send_message": {"chat_id", "silent"},
    "send_photo": {"chat_id", "caption"},
    "send_document": {"chat_id", "caption"},
    "send_video": {"chat_id", "caption"},
    "send_voice": {"chat_id"},
    "send_location": {"chat_id"},
    "send_poll": {"chat_id", "is_anonymous"},
    "send_buttons": {"chat_id"},
    "reply": {"chat_id"},
    "react": {"chat_id"},
    "edit_message": {"chat_id"},
    "delete_message": {"chat_id"},
    "pin": {"chat_id"},
    "get_reactions": {"msg_id", "sender_id", "sentiment", "limit"},
    "connect": {"supersedes_rel"},
    "bulk_memory": {"ids", "tags", "link_to", "relationship"},
    "review_corrections": {"action", "correction_id", "corrected_content"},
    "pin_attention": {"related_id"},
    "get_cost_report": {"period"},
    "browse": {"selector", "screenshot"},
    "delegate": {"trigger_msg_id"},
}


@pytest.mark.parametrize("name,optional", sorted(EXPECTED_OPTIONAL.items()))
def test_optional_params_are_not_required(name: str, optional: set[str]) -> None:
    tool = _registered_tools()[name]
    required = set(tool.inputSchema.get("required", []))
    declared = set(tool.inputSchema.get("properties", {}))
    assert optional <= declared, f"{name}: schema lost params {optional - declared}"
    assert not (optional & required), f"{name}: still requires {sorted(optional & required)}"


def test_genuinely_required_params_stay_required() -> None:
    """The fix must not make everything optional."""
    tools = _registered_tools()
    for name, params in [
        ("send_message", {"text"}),
        ("connect", {"from_id", "to_id", "relationship"}),
        ("browse", {"url"}),
        ("schedule_task", {"prompt", "schedule_type", "schedule_value"}),
        ("remember", {"id", "type", "title", "content"}),
    ]:
        required = set(tools[name].inputSchema.get("required", []))
        assert params <= required, f"{name}: no longer requires {sorted(params - required)}"


def test_schema_rejects_unknown_optional() -> None:
    with pytest.raises(ValueError, match="not in schema"):
        _schema({"a": str}, ["b"])


def test_no_tool_requires_a_param_its_handler_treats_as_optional() -> None:
    """Structural guard: a new tool added with the dict shorthand fails here.

    Reads the source rather than the schema so it catches the shorthand itself,
    which is the thing that silently marks everything required.
    """
    tree = ast.parse(AGENT_SRC.read_text())
    offenders: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        deco = next(
            (
                d
                for d in node.decorator_list
                if isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id == "tool"
            ),
            None,
        )
        if deco is None or len(deco.args) < 3:
            continue
        schema_arg = deco.args[2]
        # Already converted via _schema(...) — the required list is explicit.
        if isinstance(schema_arg, ast.Call):
            continue
        if not isinstance(schema_arg, ast.Dict):
            continue
        params = [k.value for k in schema_arg.keys if isinstance(k, ast.Constant)]
        if "properties" in params:  # hand-written full JSON Schema
            continue

        body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        name = deco.args[0].value
        for p in params:
            hard = f"Subscript(value=Name(id='args', ctx=Load()), slice=Constant(value='{p}')" in body
            soft = f"attr='get'" in body and f"Constant(value='{p}')" in body
            if soft and not hard:
                offenders.append(f"{name}.{p}")

    assert not offenders, (
        "these params are declared required but the handler reads them with "
        f"args.get(): {offenders} — wrap the schema in _schema(..., [optional])"
    )
