#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/embeddings server for Luke's embedding model
(BAAI/bge-base-en-v1.5 via fastembed). This is the SINGLE canonical copy of the
model weights: Luke's memory.py embeds through it (sqlite-vec indexing + recall)
and the Letta server embeds archive passages through it. Loading the model
in-process per consumer duplicated ~450MB of identical weights each.

Honors OpenAI's ``encoding_format``: Letta's client requests ``"base64"`` and decodes
the response as packed little-endian float32 — returning plain float lists there is
the recall-500 "fault 2" (2026-08-01). Float lists remain the default when the field
is absent.

Run: .venv/bin/python scripts/bge_embed_server.py  # serves on :17595
Supervised by com.luke.bgeembed.plist; GET /health is the liveness probe.
"""
import base64
import json
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastembed import TextEmbedding

MODEL = "BAAI/bge-base-en-v1.5"
PORT = 17595
_embedder = TextEmbedding(model_name=MODEL)


def embed(texts):
    return [v.tolist() for v in _embedder.embed(texts)]


def encode_vector(vec, encoding_format):
    if encoding_format == "base64":
        return base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode("ascii")
    return vec


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _respond(self, status, payload):
        out = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/health"):
            self._respond(200, {"status": "ok", "model": MODEL})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/embeddings"):
            self.send_response(404)
            self.end_headers()
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._respond(400, {"error": "invalid JSON body"})
            return
        inp = body.get("input", "")
        texts = [inp] if isinstance(inp, str) else list(inp)
        fmt = body.get("encoding_format", "float")
        vecs = embed(texts)
        data = [
            {"object": "embedding", "index": i, "embedding": encode_vector(v, fmt)}
            for i, v in enumerate(vecs)
        ]
        self._respond(
            200,
            {"object": "list", "data": data, "model": body.get("model", MODEL),
             "usage": {"prompt_tokens": 0, "total_tokens": 0}},
        )


if __name__ == "__main__":
    print(f"bge embed server on :{PORT} model={MODEL}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
