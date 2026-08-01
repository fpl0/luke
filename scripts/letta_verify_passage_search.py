#!/usr/bin/env python3
"""Verify Letta archival passage search end-to-end through the REST API.

Recall's Letta semantic source was 500ing (see docs/letta-recall-500-rootcause.md).
Fault 1 (embedding column vector(4096) holding 768-dim zero-padded data) is fixed;
Fault 2 (Letta's text-embedding call step) may still 500. This script is the one-command
gate: it prints SEARCH_OK only when Letta REST search returns the expected top-1 for a
known query whose answer lives only in the archive.

Run: .venv/bin/python scripts/letta_verify_passage_search.py   (sandbox-disabled — localhost)
"""
import sys, json, urllib.request

LETTA = "http://localhost:8283"
ARCHIVE = "archive-7654f6f5-542a-47b6-bdb9-3542f1cb9eca"  # bge titled archive
QUERY = "my US B1 visa interview date and address"
EXPECT_SUBSTR = "b-1 us business visa"  # entity-b1-visa-application title, lowercased


def main():
    body = json.dumps({"query": QUERY, "archive_id": ARCHIVE, "limit": 3}).encode()
    req = urllib.request.Request(
        f"{LETTA}/v1/passages/search", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except Exception as e:
        print(f"SEARCH_FAIL: REST error (Fault 2 likely still open): {repr(e)[:200]}")
        print("SEARCH_OK: False")
        return 1

    items = data if isinstance(data, list) else (data.get("results") or data.get("passages") or [])
    if not items:
        print("SEARCH_FAIL: 0 results returned")
        print("SEARCH_OK: False")
        return 1

    top = items[0]
    obj = top.get("passage", top) if isinstance(top, dict) else {}
    text = (obj.get("text") or "").lower()
    ok = EXPECT_SUBSTR in text
    print(f"top-1: {(obj.get('text') or '')[:80]}")
    print(f"SEARCH_OK: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
