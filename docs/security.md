# Security

Luke runs with `bypassPermissions`. All security comes from validation at the boundaries.

## Path Traversal

**File-sending tools** — all pass paths through `_safe_path()`, which resolves the path (collapsing `..`), checks it's under an allowed root (`luke_dir` or `store_dir`), and verifies the file exists.

**Telegram filenames** — `Path(...).name` strips all directory components from user-supplied filenames.

**Memory IDs** — regex strips everything except `[a-zA-Z0-9_-]`.

## FTS5 Injection

FTS5 MATCH queries can fail on malformed input. `recall()` catches `sqlite3.OperationalError` and returns empty results instead of crashing.

## Schedule Validation

`create_task()` validates: type is one of `cron|interval|once`, cron expressions pass `croniter.is_valid()`, intervals are positive integers, once-tasks are valid ISO timestamps.

## Memory Type Allowlist

A frozen set derived from `MEMORY_DIRS` (`entity`, `episode`, `procedure`, `insight`, `goal`). Invalid types are rejected before any file I/O.

## Token Security

`SecretStr` from pydantic for the bot token. Masks value in `__repr__` and `__str__`. `.get_secret_value()` called once at `Bot()` construction.

## SQL Parameterization

All queries use `?` placeholders. One f-string in `recall()` interpolates hardcoded condition strings only, never user input.

## Rate Limiting

PreToolUse hook counts outbound Telegram sends per agent run (10 send-related tools tracked). Blocks after `max_sends_per_run` (default: 20). Counter resets per invocation.

## Budget Limits

| | User Messages | Behaviors |
|---|---|---|
| Max turns | 200 | 75 |
| Max budget | $5.00 | $1.00 |
| Timeout | 30 min | 30 min |

`asyncio.wait_for()` hard-kills the agent on timeout.

## Sandbox

Docker sandbox enabled with `autoAllowBashIfSandboxed`. Tool execution runs in containers.
