#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/embeddings server wrapping Luke's exact embedding
model (BAAI/bge-base-en-v1.5 via fastembed). Lets Letta embed passages with the SAME
model Luke's sqlite-vec recall uses, so the Letta backend is parity-or-better rather
than a regression from the weaker nomic model.

Run: .venv/bin/python scripts/bge_embed_server.py  # serves on :17595
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastembed import TextEmbedding

MODEL = "BAAI/bge-base-en-v1.5"
PORT = 17595
_embedder = TextEmbedding(model_name=MODEL)


def embed(texts):
    return [v.tolist() for v in _embedder.embed(texts)]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/embeddings"):
            self.send_response(404)
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        inp = body.get("input", "")
        texts = [inp] if isinstance(inp, str) else list(inp)
        vecs = embed(texts)
        data = [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vecs)
        ]
        out = json.dumps(
            {"object": "list", "data": data, "model": body.get("model", MODEL),
             "usage": {"prompt_tokens": 0, "total_tokens": 0}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


if __name__ == "__main__":
    print(f"bge embed server on :{PORT} model={MODEL}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
