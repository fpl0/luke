---
name: debug
description: Debug Luke issues. Use when things aren't working, Telegram not connecting, agent not responding, memory problems, or to understand how the system works. Covers logs, environment, common issues.
---

# Luke Debugging

## Architecture Overview

```
Telegram (aiogram)          Claude Agent SDK
──────────────────────────────────────────────
app.py                      agent.py
    │                           │
    │ on_text/on_photo/...      │ run_agent()
    │      ↓                    │
    │ db.store_message()        │ ClaudeSDKClient
    │      ↓                    │   ├── 20 MCP tools (in-process)
    │ _dispatch()               │   ├── Docker Sandboxes (Bash)
    │      ↓                    │   ├── WebSearch, WebFetch
    │ process() → run_agent()   │   └── Agent teams
    │      ↓                    │
    │ result.texts → bot.send   │
    │                           │
    └── scheduler.py            │
        (cron/interval/once)    │
```

## Log Locations

| Log | Location | Content |
|-----|----------|---------|
| **stdout** | `$LUKE_DIR/luke.log` | Application output (structlog) |
| **stderr** | `$LUKE_DIR/luke.err` | Errors, tracebacks |
| **SQLite** | `$LUKE_DIR/luke.db` | Messages, sessions, tasks, memory index |

## Enabling Debug Logging

Set `LOG_LEVEL=debug` in `.env`:

```bash
echo 'LOG_LEVEL=debug' >> .env
launchctl kickstart -k gui/$(id -u)/com.luke
```

## Common Issues

### 1. Bot Not Responding

**Check service is running:**
```bash
launchctl list | grep luke
# PID should be non-zero. Status 0 = ok.
```

**Check logs for errors:**
```bash
tail -50 $LUKE_DIR/luke.err
tail -50 $LUKE_DIR/luke.log
```

**Check Telegram token:**
```bash
source .env
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe" | python3 -m json.tool
```

**Check main chat is configured:**
```bash
grep CHAT_ID .env
```
If missing, run `/setup` to register the chat.

**Check cursor isn't ahead of messages:**
```bash
sqlite3 $LUKE_DIR/luke.db "SELECT * FROM cursors;"
sqlite3 $LUKE_DIR/luke.db "SELECT chat_id, max(ts) FROM messages GROUP BY chat_id;"
```
If cursor is ahead of latest message, messages are being skipped. Fix:
```bash
sqlite3 $LUKE_DIR/luke.db "DELETE FROM cursors;"
```

### 2. Authentication Errors

**"Invalid API key" or SDK auth failure:**
```bash
cat .env | grep -E '(ANTHROPIC_API_KEY|CLAUDE_CODE_OAUTH_TOKEN)'
```
One of these must be set. For subscription users: `claude setup-token`.

### 3. Agent Runs But No Response

**Check if result.texts is empty:**
The agent may be sending responses via MCP tools (`send_message`) instead of returning text. Check logs for "Sent (id: ...)".

**Check CHAT_ID is set:**
```bash
grep CHAT_ID .env
```

### 4. Memory Not Working

**Check FTS5 index:**
```bash
sqlite3 $LUKE_DIR/luke.db "SELECT count(*) FROM memory_meta WHERE status='active';"
sqlite3 $LUKE_DIR/luke.db "SELECT id, type, title FROM memory_fts LIMIT 10;"
```

**Check memory files exist:**
```bash
ls -la $LUKE_DIR/memory/*/
```

**Rebuild index from files:**
```python
import sys; sys.path.insert(0, 'src')
from luke.db import init, rebuild_index
from luke.config import settings
init()
rebuild_index()
```

### 5. Scheduler Not Firing

**Check tasks exist:**
```bash
sqlite3 $LUKE_DIR/luke.db "SELECT id, schedule_type, schedule_value, status, last_run FROM tasks;"
```

**Check scheduler is running:** Look for "Scheduler started" in logs.

**Force a task to run:** Set `last_run` to NULL:
```bash
sqlite3 $LUKE_DIR/luke.db "UPDATE tasks SET last_run = NULL WHERE id = '<task_id>';"
```

### 6. Service Won't Start

**Check plist:**
```bash
cat ~/Library/LaunchAgents/com.luke.plist
```
Verify `ProgramArguments` has the correct `uv` path and `WorkingDirectory` is correct.

**Check uv path:**
```bash
which uv
```

**Check Python version:**
```bash
uv run python --version  # Should be 3.13+
```

## Quick Diagnostic

```bash
echo "=== Luke Diagnostic ==="

echo -e "\n1. Service running?"
launchctl list 2>/dev/null | grep luke || echo "NOT RUNNING"

echo -e "\n2. Auth configured?"
[ -f .env ] && (grep -q "CLAUDE_CODE_OAUTH_TOKEN=\|ANTHROPIC_API_KEY=" .env && echo "OK" || echo "MISSING") || echo "NO .env FILE"

echo -e "\n3. Telegram token valid?"
source .env 2>/dev/null && curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe" | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK -', d['result']['username']) if d.get('ok') else print('INVALID')" 2>/dev/null || echo "CANNOT CHECK"

echo -e "\n4. Main chat configured?"
grep -q CHAT_ID .env 2>/dev/null && grep CHAT_ID .env || echo "NOT SET"

echo -e "\n5. Recent messages?"
sqlite3 $LUKE_DIR/luke.db "SELECT count(*) || ' messages' FROM messages WHERE ts > datetime('now', '-1 hour');" 2>/dev/null || echo "NO DATABASE"

echo -e "\n6. Memory index?"
sqlite3 $LUKE_DIR/luke.db "SELECT count(*) || ' memories' FROM memory_meta WHERE status='active';" 2>/dev/null || echo "NO INDEX"

echo -e "\n7. Recent errors?"
tail -5 $LUKE_DIR/luke.err 2>/dev/null || echo "No error log"
```

## Session Management

Sessions are stored in SQLite. To clear a chat's session:
```bash
sqlite3 $LUKE_DIR/luke.db "DELETE FROM sessions WHERE chat_id = '<chat_id>';"
```

## Rebuilding

```bash
# Reinstall deps
uv sync

# Lint
uv run ruff check src/

# Restart
launchctl kickstart -k gui/$(id -u)/com.luke
```
