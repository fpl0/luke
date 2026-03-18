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

Luke uses `CREATE TABLE IF NOT EXISTS` so new tables are auto-created on restart. But new columns on existing tables, changed indexes, or new virtual table columns need explicit migration.

**Do NOT just grep for migration markers.** Actively compare the schema definition in `db.py` against the live database:

```bash
source .env 2>/dev/null
LUKE_DIR="${LUKE_DIR:-$HOME/.luke}"

# Extract expected schema from code
grep -A2 "CREATE TABLE\|CREATE INDEX\|CREATE VIRTUAL" src/luke/db.py

# Get live schema
sqlite3 "$LUKE_DIR/luke.db" ".schema"

# Compare columns for each table
for table in messages memory_meta memory_links tasks task_logs sessions cursors behavior_state memory_history cost_log reaction_feedback schema_version; do
    sqlite3 "$LUKE_DIR/luke.db" "PRAGMA table_info($table);" 2>/dev/null
done
```

Compare the code's `_SCHEMA` string against the live DB output. Look for:
- **New columns** in existing tables → generate `ALTER TABLE ... ADD COLUMN` statements
- **New indexes** → generate `CREATE INDEX IF NOT EXISTS` statements
- **Changed column defaults** → may need UPDATE statements for existing rows
- **New tables** → handled automatically by `init()` on restart (no action needed)

If migrations are needed, build and run them:

```bash
sqlite3 "$LUKE_DIR/luke.db" "<migration SQL>"
```

If the live schema matches the code, this step is a no-op.

## 4b. Apply Data Migrations

Data migrations are versioned alongside schema migrations in the `schema_version` table. Check the current version and run any pending data migrations:

```bash
source .env 2>/dev/null
LUKE_DIR="${LUKE_DIR:-$HOME/.luke}"
CURRENT=$(sqlite3 "$LUKE_DIR/luke.db" "SELECT COALESCE(MAX(version), 0) FROM schema_version" 2>/dev/null || echo 0)
echo "Current schema version: $CURRENT"
```

**Version 4: Backfill valid_from + normalize ad-hoc relationship labels**
Only run if `$CURRENT < 4`:
```sql
UPDATE memory_links SET valid_from = created WHERE valid_from = '';
UPDATE memory_links SET relationship = 'derived_from' WHERE relationship IN ('derived from', 'derived-from', 'originated-from');
UPDATE memory_links SET relationship = 'caused' WHERE relationship = 'root cause involves';
UPDATE memory_links SET relationship = 'supports' WHERE relationship IN ('evidence-from', 'pattern exemplified by', 'reinforced-by');
UPDATE memory_links SET relationship = 'related' WHERE relationship IN ('escalation path involves', 'describes emotional pattern of', 'complements');
INSERT INTO schema_version (version) VALUES (4);
```

**Version 5: Re-label generic 'related' links by type pair**
Only run if `$CURRENT < 5`:
```sql
UPDATE memory_links SET relationship = 'involves' WHERE relationship = 'related' AND from_id IN (SELECT id FROM memory_meta WHERE type = 'episode') AND to_id IN (SELECT id FROM memory_meta WHERE type = 'entity');
UPDATE memory_links SET relationship = 'contributes_to' WHERE relationship = 'related' AND from_id IN (SELECT id FROM memory_meta WHERE type = 'episode') AND to_id IN (SELECT id FROM memory_meta WHERE type = 'goal');
UPDATE memory_links SET relationship = 'derived_from' WHERE relationship = 'related' AND from_id IN (SELECT id FROM memory_meta WHERE type = 'episode') AND to_id IN (SELECT id FROM memory_meta WHERE type = 'insight');
UPDATE memory_links SET relationship = 'uses' WHERE relationship = 'related' AND from_id IN (SELECT id FROM memory_meta WHERE type = 'episode') AND to_id IN (SELECT id FROM memory_meta WHERE type = 'procedure');
UPDATE memory_links SET relationship = 'about' WHERE relationship = 'related' AND from_id IN (SELECT id FROM memory_meta WHERE type = 'insight') AND to_id IN (SELECT id FROM memory_meta WHERE type = 'entity');
UPDATE memory_links SET relationship = 'about' WHERE relationship = 'related' AND from_id IN (SELECT id FROM memory_meta WHERE type = 'insight') AND to_id IN (SELECT id FROM memory_meta WHERE type = 'goal');
UPDATE memory_links SET relationship = 'supports' WHERE relationship = 'related' AND from_id IN (SELECT id FROM memory_meta WHERE type = 'insight') AND to_id IN (SELECT id FROM memory_meta WHERE type = 'insight');
UPDATE memory_links SET relationship = 'involves' WHERE relationship = 'related' AND from_id IN (SELECT id FROM memory_meta WHERE type = 'goal') AND to_id IN (SELECT id FROM memory_meta WHERE type = 'entity');
UPDATE memory_links SET relationship = 'informed_by' WHERE relationship = 'related' AND from_id IN (SELECT id FROM memory_meta WHERE type = 'goal') AND to_id IN (SELECT id FROM memory_meta WHERE type = 'insight');
UPDATE memory_links SET relationship = 'about' WHERE relationship = 'related' AND from_id IN (SELECT id FROM memory_meta WHERE type = 'procedure') AND to_id IN (SELECT id FROM memory_meta WHERE type = 'entity');
INSERT INTO schema_version (version) VALUES (5);
```
Remaining `related` links (episode↔episode, entity↔entity, etc.) stay as `related`.

**Version 6: Cleanup + trigger feedback consolidation**
Only run if `$CURRENT < 6`:
```bash
# Archive test-mem debugging artifact
sqlite3 "$LUKE_DIR/luke.db" "UPDATE memory_meta SET status = 'archived' WHERE id = 'test-mem'"
# Trigger initial feedback consolidation on next scheduler tick
sqlite3 "$LUKE_DIR/luke.db" "INSERT OR REPLACE INTO behavior_state (name, last_run) VALUES ('feedback_consolidation', '2000-01-01T00:00:00')"
sqlite3 "$LUKE_DIR/luke.db" "INSERT INTO schema_version (version) VALUES (6)"
```

Also fix duplicate `#` title headers in 4 memory files (if present):
- `$LUKE_DIR/memory/entities/person-filipe.md`
- `$LUKE_DIR/memory/entities/project-clio-i18n.md`
- `$LUKE_DIR/memory/procedures/procedure-morning-briefing.md`
- `$LUKE_DIR/memory/procedures/capability-voice-messages.md`

For each, check if the `# Title` line appears twice consecutively and remove the duplicate.

After all data migrations, verify:
```bash
echo "Schema version: $(sqlite3 "$LUKE_DIR/luke.db" "SELECT MAX(version) FROM schema_version")"
echo "Link labels: $(sqlite3 "$LUKE_DIR/luke.db" "SELECT relationship, COUNT(*) FROM memory_links GROUP BY relationship ORDER BY COUNT(*) DESC")"
```

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
