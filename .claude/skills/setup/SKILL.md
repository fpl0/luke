---
name: setup
description: Run initial Luke setup. Use when user wants to install dependencies, create a Telegram bot, register their main chat, or start the background service. Triggers on "setup", "install", "configure luke", or first-time setup requests.
---

# Luke Setup

Run setup steps automatically. Only pause when user action is required (creating bot, sending first message). Fix problems yourself — don't tell the user to fix things unless it genuinely requires their action.

**UX Note:** Use `AskUserQuestion` for all user-facing questions.

## 0. Prerequisites

Check Python, uv, and ffmpeg:

```bash
python3 --version  # Need 3.13+
uv --version       # Need uv installed
ffmpeg -version    # Need ffmpeg for voice transcription
```

- If Python < 3.13: install via `brew install python@3.13` (macOS) or `pyenv install 3.13`
- If uv missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- If ffmpeg missing: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)

## 1. Install Dependencies

```bash
cd ~/Code/luke  # or wherever the project lives
uv sync
```

If this fails, check error output. Common fix: delete `.venv` and retry.

## 2. Claude Authentication

AskUserQuestion: "Do you use a Claude subscription (Pro/Max) or an Anthropic API key?"

**Subscription:** Tell user to run `claude setup-token` in another terminal, copy the token, then:
```bash
echo 'CLAUDE_CODE_OAUTH_TOKEN=<token>' >> .env
```
Do NOT collect the token in chat.

**API key:** Tell user to add their key:
```bash
echo 'ANTHROPIC_API_KEY=<key>' >> .env
```

## 3. Telegram Bot Setup

AskUserQuestion: "Do you already have a Telegram bot token, or should I walk you through creating one?"

**Creating a new bot:**
1. Tell user to open Telegram and message @BotFather
2. Send `/newbot` to BotFather
3. Choose a name (e.g., "Luke") and username (e.g., "luke_assistant_bot")
4. Copy the token BotFather gives them

**Store the token:**
```bash
echo 'TELEGRAM_BOT_TOKEN=<token>' >> .env
```

**Optional settings:**
AskUserQuestion: "What name should the assistant use? (default: Luke)"
If different from "Luke":
```bash
echo 'ASSISTANT_NAME=<name>' >> .env
```

## 4. Choose Data Directory

AskUserQuestion: "Where should Luke store its data (database, memory, workspace)? Examples:\n- `~/.luke` (default, local)\n- `~/Documents/Luke`\n- Any path you prefer\n\nNote: avoid cloud-synced folders (Google Drive, Dropbox, iCloud) — SQLite databases can corrupt with cloud sync."

Store the chosen path (use `~/.luke` if user accepts default):
```bash
echo 'LUKE_DIR=<chosen_path>' >> .env
```

## 5. Register Main Chat

Tell the user: "Send any message to your bot on Telegram so I can detect your chat ID."

Then run Luke temporarily to capture the chat ID:

```bash
# Start luke briefly to see incoming messages
timeout 30 uv run luke 2>&1 | tee /tmp/luke-setup.log &
sleep 5
```

Wait for the user to confirm they sent a message. Then find the chat ID from logs:

```bash
grep "chat.id" /tmp/luke-setup.log
```

Once you have the chat ID, register it:

```bash
echo "CHAT_ID=<chat_id>" >> .env
```

Replace `<chat_id>` with the detected chat ID.

## 6. Create Context File

AskUserQuestion: "What's your name and timezone? (e.g., 'Alice, Europe/London')"

Read `LUKE_DIR` from `.env` and write `$LUKE_DIR/context.yaml` with the user's details:

```yaml
name: Alice
timezone: Europe/London
chat_id: "123456789"
```

This file lives in the data directory — personal data stays outside the repo.

## 7. Create Weekly Consolidation Task

Read `LUKE_DIR` from `.env` and register a weekly memory consolidation task:

```python
import sqlite3, uuid, os
from datetime import datetime, UTC

luke_dir = os.environ.get("LUKE_DIR", os.path.expanduser("~/.luke"))
db = sqlite3.connect(f"{luke_dir}/luke.db")
db.executescript("""
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY, chat_id TEXT NOT NULL,
    prompt TEXT NOT NULL, schedule_type TEXT NOT NULL, schedule_value TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active', last_run TEXT, created_at TEXT NOT NULL
);
""")
db.execute(
    "INSERT OR REPLACE INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    (
        str(uuid.uuid4())[:8], CHAT_ID,
        "Review this week's conversations and memories. Then:\n"
        "1. Create or update entity memories for people I interacted with.\n"
        "2. Distill episode patterns into insights.\n"
        "3. Archive episodes fully captured by insights or entity updates.\n"
        "4. Check for stale information.\n"
        "5. Identify anything I mentioned wanting to do but haven't — schedule reminders.",
        "cron", "0 3 * * 0",  # Sunday 3am
        "active", None, datetime.now(UTC).isoformat(),
    ),
)
db.commit()
db.close()
```

## 8. Install as Service (launchd)

Read `com.luke.plist`, replace placeholders (including `__LUKE_DIR__`), and install:

```bash
UV_PATH=$(which uv)
PROJECT_ROOT=$(pwd)
HOME_DIR=$HOME
LUKE_DIR_VAL=$(grep LUKE_DIR .env | cut -d= -f2 | sed "s|~|$HOME|")

sed -e "s|__UV_PATH__|$UV_PATH|g" \
    -e "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
    -e "s|__HOME__|$HOME_DIR|g" \
    -e "s|__LUKE_DIR__|$LUKE_DIR_VAL|g" \
    com.luke.plist > ~/Library/LaunchAgents/com.luke.plist

launchctl load ~/Library/LaunchAgents/com.luke.plist
```

## 9. Verify

```bash
# Check service is running
launchctl list | grep luke

# Check logs (in LUKE_DIR)
tail -20 "$LUKE_DIR_VAL/luke.log"

# Check for errors
tail -20 "$LUKE_DIR_VAL/luke.err"
```

Tell user to send a message to their bot. Watch logs with `tail -f "$LUKE_DIR_VAL/luke.log"`.

## Troubleshooting

**Service not starting:** Check `$LUKE_DIR/luke.err`. Common: wrong uv path, missing `.env`, Python version mismatch.

**Bot not responding:** Check `TELEGRAM_BOT_TOKEN` in `.env`. Verify bot exists with `curl https://api.telegram.org/bot<TOKEN>/getMe`.

**"No module named luke":** Run `uv sync` to reinstall the package.
