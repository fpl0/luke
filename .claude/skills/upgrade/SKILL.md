---
name: upgrade
description: Pull latest Luke code and propagate changes to data and running service. Use when user says "upgrade", "update luke", "pull latest", or after making code changes that need to reach the running instance.
---

# Luke Upgrade

Pull latest code, sync deps, apply schema changes, update persona, regenerate plist, and restart the service. Run steps automatically — only pause when a decision is needed.

## 1. Pull Latest Code

```bash
git pull --ff-only
```

If `--ff-only` fails (local changes), inform the user and stop. Don't force-pull.

## 2. Sync Dependencies

```bash
uv sync
```

If this fails, check error output. Common fix: delete `.venv` and retry with `uv sync`.

## 3. Run Type Checks and Lint

Quick sanity check that the pulled code is clean:

```bash
uv run ruff check src/
uv run pyright src/
```

If either fails, warn the user — the upgrade may be broken. Don't proceed to restart.

## 4. Apply Schema Migrations

Luke uses `CREATE TABLE IF NOT EXISTS` so new tables are auto-created on restart. But new columns on existing tables need explicit migration.

Check if any ALTER TABLE statements are needed by reading `db.py` for migration markers:

```bash
grep -n "ALTER TABLE\|-- migration" src/luke/db.py
```

If migrations are found, run them against the database:

```bash
source .env 2>/dev/null
LUKE_DIR="${LUKE_DIR:-$HOME/.luke}"
sqlite3 "$LUKE_DIR/luke.db" < <migration_sql>
```

If no migrations are found, this step is a no-op — `init()` on restart handles new tables/indexes.

## 5. Update LUKE.md Persona

The package template at `src/luke/templates/LUKE.md` may have improvements. The user's copy at `$LUKE_DIR/LUKE.md` was seeded on first run and never updated.

```bash
source .env 2>/dev/null
LUKE_DIR="${LUKE_DIR:-$HOME/.luke}"
```

Compare the two files:

```bash
diff "$LUKE_DIR/LUKE.md" src/luke/templates/LUKE.md
```

If they differ, AskUserQuestion: "The persona template (LUKE.md) has been updated. Would you like to:\n1. Replace your current persona with the new template\n2. See the diff and merge manually\n3. Keep your current persona unchanged"

- **Option 1:** Copy the new template over: `cp src/luke/templates/LUKE.md "$LUKE_DIR/LUKE.md"`
- **Option 2:** Show the diff output and let the user decide what to keep
- **Option 3:** Skip — no action needed

## 6. Regenerate launchd Plist

Check if the plist template changed since last install:

```bash
# Compare current installed plist structure with template
INSTALLED=~/Library/LaunchAgents/com.luke.plist
if [ -f "$INSTALLED" ]; then
    # Check if template has new keys not in installed version
    TEMPLATE_KEYS=$(grep '<key>' com.luke.plist | sort)
    INSTALLED_KEYS=$(grep '<key>' "$INSTALLED" | sort)
    if [ "$TEMPLATE_KEYS" != "$INSTALLED_KEYS" ]; then
        echo "PLIST_CHANGED"
    else
        echo "PLIST_OK"
    fi
else
    echo "PLIST_MISSING"
fi
```

If `PLIST_CHANGED` or `PLIST_MISSING`, regenerate:

```bash
UV_PATH=$(which uv)
PROJECT_ROOT=$(pwd)
HOME_DIR=$HOME
source .env 2>/dev/null
LUKE_DIR_VAL="${LUKE_DIR:-$HOME/.luke}"

# Expand ~ in LUKE_DIR if present
LUKE_DIR_VAL=$(eval echo "$LUKE_DIR_VAL")

launchctl unload ~/Library/LaunchAgents/com.luke.plist 2>/dev/null

sed -e "s|__UV_PATH__|$UV_PATH|g" \
    -e "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
    -e "s|__HOME__|$HOME_DIR|g" \
    -e "s|__LUKE_DIR__|$LUKE_DIR_VAL|g" \
    com.luke.plist > ~/Library/LaunchAgents/com.luke.plist

launchctl load ~/Library/LaunchAgents/com.luke.plist
```

If `PLIST_OK`, just restart:

```bash
launchctl kickstart -k gui/$(id -u)/com.luke
```

## 7. Verify

```bash
source .env 2>/dev/null
LUKE_DIR="${LUKE_DIR:-$HOME/.luke}"

echo "=== Post-Upgrade Check ==="

echo -e "\n1. Service running?"
launchctl list 2>/dev/null | grep luke || echo "NOT RUNNING"

echo -e "\n2. Recent logs (last 10 lines):"
tail -10 "$LUKE_DIR/luke.log" 2>/dev/null || echo "No log file"

echo -e "\n3. Any errors?"
tail -5 "$LUKE_DIR/luke.err" 2>/dev/null || echo "No error log"

echo -e "\n4. Memory index:"
sqlite3 "$LUKE_DIR/luke.db" "SELECT count(*) || ' active memories' FROM memory_meta WHERE status='active';" 2>/dev/null || echo "No database"
```

Tell the user: "Upgrade complete. Send a message to your bot to verify it's responding."
