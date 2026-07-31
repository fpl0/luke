"""Live 2.2b verification: connect/forget/restore write-through into the Letta ledger.

Creates two sentinel memories through the real index_memory path, exercises
link_memories / archive_memory / restore_memory, and asserts the Letta passages
reflect each change. Cleans up all sentinel rows + passages at the end.
Run sandbox-off (needs localhost Letta :8283 + Ollama embed).
"""
import json
import urllib.request
import sys

sys.path.insert(0, "src")
from luke.config import settings  # noqa: E402
from luke import memory  # noqa: E402
from luke.letta_adapter import letta_semantic_search  # noqa: E402

BASE = settings.letta_base_url.rstrip("/")
ARCH = settings.letta_archive_id
print(f"backend={settings.memory_backend} archive={ARCH}")


def _search(surface):
    body = json.dumps({"query": surface[:400], "archive_id": ARCH, "limit": 30}).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/passages/search", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=6) as r:
        data = json.load(r)
    if isinstance(data, dict):
        data = data.get("results") or data.get("passages") or []
    return data


def passages_for(luke_id, surface):
    out = []
    for p in _search(surface):
        o = p.get("passage", p) if isinstance(p, dict) else {}
        m = o.get("metadata") or {}
        if m.get("luke_id") == luke_id:
            out.append({"id": o.get("id"), "status": m.get("status"), "links": m.get("links")})
    return out


def delete_passage(pid):
    req = urllib.request.Request(f"{BASE}/v1/archives/{ARCH}/passages/{pid}", method="DELETE")
    with urllib.request.urlopen(req, timeout=6):
        pass


A, B = "test-22b-alpha-zzz", "test-22b-beta-zzz"
surfA = "test 22b alpha The quick sentinel alpha memory for write-through verification"
surfB = "test 22b beta A second sentinel beta memory used as a link target"

ok = True

# 1. create both through the real remember path
memory.index_memory(mem_id=A, mem_type="episode", title="test 22b alpha",
    content="The quick sentinel alpha memory for write-through verification",
    tags=["test22b"], links=[])
memory.index_memory(mem_id=B, mem_type="episode", title="test 22b beta",
    content="A second sentinel beta memory used as a link target",
    tags=["test22b"], links=[])
pa = passages_for(A, surfA)
print(f"[1 create]  A passages={len(pa)} status={[p['status'] for p in pa]} links={pa[0]['links'] if pa else None}")
ok &= len(pa) == 1 and pa[0]["status"] == "active"

# 2. connect A -> B ; from_id passage links metadata should now include B
created = memory.link_memories(A, B, "related")
pa = passages_for(A, surfA)
has_b = any(B in (p["links"] or []) for p in pa)
print(f"[2 connect] link_created={created} A.links={pa[0]['links'] if pa else None} contains_B={has_b}")
ok &= created and has_b

# 3. forget/archive A ; passage flips to archived + drops out of adapter recall
memory.archive_memory(A)
pa = passages_for(A, surfA)
print(f"[3 forget]  A passage statuses={[p['status'] for p in pa]}")
ok &= any(p["status"] == "archived" for p in pa)
hit = letta_semantic_search("sentinel alpha memory write-through verification",
    limit=10, include_private=True)
in_recall = hit is not None and any(h["id"] == A for h in hit)
print(f"[3 recall]  archived A in adapter recall? {in_recall} (expect False)")
ok &= not in_recall

# 4. restore A ; passage flips back to active
restored = memory.restore_memory(A)
pa = passages_for(A, surfA)
print(f"[4 restore] restored={restored} A passage statuses={[p['status'] for p in pa]}")
ok &= restored and any(p["status"] == "active" for p in pa)

# cleanup: delete sentinel passages + sqlite rows
db = memory._db()
for pid_set, surf in ((A, surfA), (B, surfB)):
    for p in passages_for(pid_set, surf):
        try:
            delete_passage(p["id"])
        except Exception as e:
            print("passage cleanup", p["id"], e)
for mid in (A, B):
    db.execute("DELETE FROM memory_meta WHERE id = ?", (mid,))
    db.execute("DELETE FROM memory_fts WHERE id = ?", (mid,))
    db.execute("DELETE FROM memory_links WHERE from_id = ? OR to_id = ?", (mid, mid))
    db.execute("DELETE FROM memory_vec WHERE memory_id = ?", (mid,))
db.commit()
residual = len(passages_for(A, surfA)) + len(passages_for(B, surfB))
print(f"[cleanup]   residual sentinel passages={residual}")
ok &= residual == 0

print("RESULT:", "ALL PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
