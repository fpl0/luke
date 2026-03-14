---
name: customize
description: Add new capabilities or modify Luke's behavior. Use when user wants to add channels, change triggers, add integrations, modify tools, or make any other customizations. This is an interactive skill that asks questions to understand what the user wants.
---

# Luke Customization

This skill helps users add capabilities or modify behavior. Use AskUserQuestion to understand what they want before making changes.

## Workflow

1. **Understand the request** — Ask clarifying questions
2. **Plan the changes** — Identify files to modify
3. **Implement** — Make changes directly to the code
4. **Verify** — Run `uv run ruff check src/` and test

## Key Files

| File | Purpose |
|------|---------|
| `src/luke/app.py` | Telegram handlers, message dispatch, main loop |
| `src/luke/agent.py` | Claude SDK client, 20 MCP tools, stop hook |
| `src/luke/db.py` | SQLite schema and queries |
| `src/luke/config.py` | Settings from `.env` |
| `src/luke/scheduler.py` | Task execution loop |
| `$LUKE_DIR/LUKE.md` | Agent persona and behavior |
| `pyproject.toml` | Dependencies |

## Common Customization Patterns

### Adding a New MCP Tool

Add a new `@tool` decorated function inside `_build_tools()` in `agent.py`:

```python
@tool(
    "tool_name",
    "Description of what it does",
    {"param1": str, "param2": int},
)
async def my_tool(args: dict[str, Any]) -> dict[str, Any]:
    # Implementation — has access to bot, chat_id, luke_dir
    result = do_something(args["param1"])
    return {"content": [{"type": "text", "text": f"Done: {result}"}]}
```

Then add it to the `tools=[...]` list in `create_sdk_mcp_server()` at the bottom of `_build_tools()`, and add `"mcp__luke__tool_name"` to `allowed_tools` in `run_agent()` (or keep the existing `"mcp__luke__*"` wildcard).

### Adding a New Telegram Handler

Add a new handler in `app.py` following the existing pattern:

```python
@dp.message(F.some_filter)
async def on_something(msg: types.Message) -> None:
    if not msg.from_user:
        return
    db.store_message(
        chat_id=str(msg.chat.id),
        sender_name=msg.from_user.full_name,
        content="[Description of what was received]",
        timestamp=msg.date.isoformat(),
    )
    _dispatch(str(msg.chat.id))
```

### Adding a New Channel (e.g., Slack, WhatsApp)

Questions to ask:
- Which channel?
- Same trigger word or different?
Implementation:
1. Add the channel's library to `pyproject.toml` and run `uv add <package>`
2. Add connection/handler code in `app.py` (or a new module if large)
3. Messages flow through the same `db.store_message()` → `_dispatch()` → `run_agent()` pipeline
4. Add channel-specific send tools in `agent.py`

### Changing Agent Behavior

- **Persona** → edit `$LUKE_DIR/LUKE.md`
- **Assistant name** → change `ASSISTANT_NAME` in `.env`
- **Tools available** → modify `allowed_tools` in `run_agent()` in `agent.py`

### Adding a New Database Table

1. Add the `CREATE TABLE` statement to `_SCHEMA` in `db.py`
2. Add query functions below the schema
3. `db.init()` runs on startup and creates any missing tables

### Adding External MCP Servers

In `run_agent()` in `agent.py`, add to the `mcp_servers` dict:

```python
mcp_servers={
    "luke": _build_tools(chat_id, bot),
    "external": {"type": "stdio", "command": "server-command", "args": ["--flag"]},
}
```

## After Changes

```bash
# Lint
uv run ruff check src/

# Restart service
launchctl kickstart -k gui/$(id -u)/com.luke
```
