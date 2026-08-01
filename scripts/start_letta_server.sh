#!/bin/bash
# Launch the Letta server with the postgres URI EXPLICITLY set.
#
# Why this wrapper exists (recall-500 Fault 2, 2026-08-01):
# Letta connects to postgres via its DEFAULT uri (letta:letta@localhost:5432/letta)
# even when pg_uri is unset — so data lands in postgres. BUT settings.database_engine
# only reports POSTGRES when pg_uri (or the pg_* parts) is EXPLICITLY set; otherwise it
# reports SQLITE and build_agent_passage_query() takes the sqlite_vec branch, which
# `import sqlite_vec` (not installed) -> 500 on every archival search.
# Setting LETTA_PG_URI to the same DB makes the connection identical AND flips
# database_engine -> POSTGRES so pgvector cosine_distance is used. See
# docs/letta-recall-500-rootcause.md.
export LETTA_PG_URI='postgresql+pg8000://letta:letta@localhost:5432/letta'
cd /Users/filipelm/Code/luke
exec .letta-venv/bin/letta server --port 8283
