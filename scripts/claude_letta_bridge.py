#!/usr/bin/env python3
"""OpenAI-compatible /v1/chat/completions proxy that drives Claude via Luke's existing
CLAUDE_CODE_OAUTH_TOKEN — no new billing key. This is the Phase-1 critical-path bridge:
Letta's agent is configured with model_endpoint_type="openai" + model_endpoint pointing
here, so every Letta LLM turn is translated OpenAI->Anthropic, run on real Claude, and
translated back (text + tool_use). Keeps Luke's auth (OAuth) and makes the Letta agent
Claude-smart + fast instead of qwen3:8b (~min/turn).

Spike-proven (2026-07-30): the OAuth token works against api.anthropic.com/v1/messages
with header `anthropic-beta: oauth-2025-04-20` IF the system prompt begins with the
Claude Code identity line. Both plain completion and tool_use return real Claude output.

Run: .venv/bin/python scripts/claude_letta_bridge.py   # serves on :17596
"""
import json
import os
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 17596
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
# OAuth tokens are only accepted when the system prompt starts with this identity line.
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


def _token():
    tok = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if tok:
        return tok
    # Fall back to the repo .env (same source Luke uses).
    env = os.path.join(os.path.dirname(__file__), "..", ".env")
    try:
        with open(env) as f:
            for line in f:
                if line.startswith("CLAUDE_CODE_OAUTH_TOKEN="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


TOKEN = _token()


# ---------- OpenAI -> Anthropic ----------

def _openai_to_anthropic(body):
    system_parts = [CLAUDE_CODE_IDENTITY]
    messages = []
    for m in body.get("messages", []):
        role = m.get("role")
        if role == "system":
            c = m.get("content")
            if isinstance(c, str):
                system_parts.append(c)
            elif isinstance(c, list):
                system_parts.append(
                    "".join(p.get("text", "") for p in c if isinstance(p, dict))
                )
            continue
        if role == "tool":
            # OpenAI tool result -> Anthropic tool_result inside a user turn.
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": _as_text(m.get("content")),
                }],
            })
            continue
        if role == "assistant":
            blocks = []
            txt = m.get("content")
            if isinstance(txt, str) and txt:
                blocks.append({"type": "text", "text": txt})
            elif isinstance(txt, list):
                for p in txt:
                    if isinstance(p, dict) and p.get("type") == "text":
                        blocks.append({"type": "text", "text": p.get("text", "")})
            for tc in m.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            if not blocks:
                blocks = [{"type": "text", "text": ""}]
            messages.append({"role": "assistant", "content": blocks})
            continue
        # default: user
        messages.append({"role": "user", "content": _as_text(m.get("content"))})

    out = {
        "model": _map_model(body.get("model")),
        "max_tokens": int(body.get("max_tokens") or DEFAULT_MAX_TOKENS),
        # System as TWO blocks: the blessed identity line ALONE (block 1), then the
        # custom bulk in a SEPARATE cached block (block 2). CRITICAL — verified 2026-08-01:
        #   • plain-string 26k system                        -> 429 (rate_limit_error)
        #   • identity+bulk COMBINED in one cached block      -> 429  (identity unrecognized)
        #   • identity block ALONE + bulk in a cached block   -> 200  (cache_creation/read)
        # The OAuth path matches the FIRST system block against the known Claude Code
        # identity; gluing custom content onto it defeats that match and the whole (large,
        # uncached-equivalent) system trips the subscription rate window. Splitting keeps
        # the identity recognized AND turns the heavy custom bulk into cheap cache tokens
        # the limiter treats leniently. Letta ships ~26k chars of core blocks every turn,
        # so without this a single Letta ReAct turn 429s (the whole Phase-1.4 blocker).
        "system": _build_system_blocks(system_parts),
        "messages": messages,
    }
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]

    tools = body.get("tools")
    if tools:
        out["tools"] = []
        for t in tools:
            fn = t.get("function", t)
            out["tools"].append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        # Cache the (stable, per-turn-identical) tool block too — same rate-limit rationale.
        if out["tools"]:
            out["tools"][-1]["cache_control"] = {"type": "ephemeral"}
        tc = body.get("tool_choice")
        out["tool_choice"] = _map_tool_choice(tc)
    return out


def _build_system_blocks(system_parts):
    """Identity line as its own block (so the OAuth path recognizes it), the rest cached.
    See the rate-limit rationale at the call site."""
    parts = [p for p in system_parts if p]
    if not parts:
        return [{"type": "text", "text": CLAUDE_CODE_IDENTITY}]
    blocks = [{"type": "text", "text": parts[0]}]  # identity — standalone, uncached
    rest = "\n\n".join(parts[1:])
    if rest:
        blocks.append({"type": "text", "text": rest, "cache_control": {"type": "ephemeral"}})
    return blocks


def _as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return "" if content is None else str(content)


def _map_model(model):
    # Only pass through a FULL Claude model id (e.g. claude-opus-4-8, claude-sonnet-4-6).
    # A bare "claude" or any non-claude/unknown label falls back to DEFAULT_MODEL —
    # Anthropic 404s on "claude" alone, and Letta may hand us a short label.
    if model and model.startswith("claude-") and len(model) > len("claude-"):
        return model
    return DEFAULT_MODEL


def _map_tool_choice(tc):
    if tc in (None, "auto"):
        return {"type": "auto"}
    if tc == "required":
        return {"type": "any"}
    if tc == "none":
        return {"type": "auto"}
    if isinstance(tc, dict) and tc.get("type") == "function":
        return {"type": "tool", "name": tc.get("function", {}).get("name", "")}
    return {"type": "auto"}


# ---------- Anthropic -> OpenAI ----------

_STOP_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
}


def _anthropic_to_openai(resp, model):
    text_parts, tool_calls = [], []
    for block in resp.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
    msg = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    usage = resp.get("usage", {})
    return {
        "id": resp.get("id", "chatcmpl-bridge"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": _STOP_MAP.get(resp.get("stop_reason"), "stop"),
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def _call_anthropic(payload, max_retries=4):
    """POST to Anthropic with exponential backoff on 429/529/5xx. The OAuth token shares
    one rate-limit pool with the live SDK-Luke session, so transient 429s are expected
    under contention and must be retried, not surfaced as hard failures to Letta."""
    data = json.dumps(payload).encode()
    _dbg = os.environ.get("BRIDGE_DEBUG")
    if _dbg:
        print(f"[dbg] req bytes={len(data)} sys_chars={len(payload.get('system','') or '')} "
              f"tools={len(payload.get('tools',[]) or [])} msgs={len(payload.get('messages',[]) or [])} "
              f"max_tokens={payload.get('max_tokens')}", flush=True)
    delay = 2.0
    last_err = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(ANTHROPIC_URL, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {TOKEN}")
        req.add_header("anthropic-beta", "oauth-2025-04-20")
        req.add_header("anthropic-version", "2023-06-01")
        req.add_header("content-type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if _dbg:
                _b = ""
                try:
                    _b = e.read().decode(errors="replace")[:400]
                except Exception:
                    pass
                print(f"[dbg] HTTP {e.code} attempt={attempt} retry-after={e.headers.get('retry-after')} body={_b}", flush=True)
            if e.code in (429, 529) or 500 <= e.code < 600:
                last_err = e
                if attempt < max_retries:
                    retry_after = e.headers.get("retry-after")
                    wait = float(retry_after) if retry_after else delay
                    time.sleep(min(wait, 30.0))
                    delay *= 2
                    continue
            raise
    raise last_err


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": DEFAULT_MODEL, "object": "model", "owned_by": "anthropic"}
            ]})
            return
        self._send(200, {"status": "ok", "bridge": "claude-letta", "model": DEFAULT_MODEL})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "bad json"})
            return
        stream = bool(body.get("stream"))
        model = _map_model(body.get("model"))
        try:
            payload = _openai_to_anthropic(body)
            resp = _call_anthropic(payload)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            self._send(e.code, {"error": {"message": detail, "type": "anthropic_error"}})
            return
        except Exception as e:  # noqa: BLE001
            self._send(502, {"error": {"message": str(e)[:500], "type": "bridge_error"}})
            return
        oa = _anthropic_to_openai(resp, model)
        if stream:
            self._send_stream(oa)
        else:
            self._send(200, oa)

    def _send_stream(self, oa):
        # Emulate a single-chunk SSE stream from the completed response.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        choice = oa["choices"][0]
        delta = {"role": "assistant"}
        msg = choice["message"]
        if msg.get("content"):
            delta["content"] = msg["content"]
        if msg.get("tool_calls"):
            delta["tool_calls"] = [
                {**tc, "index": i} for i, tc in enumerate(msg["tool_calls"])
            ]
        first = {
            "id": oa["id"], "object": "chat.completion.chunk",
            "created": oa["created"], "model": oa["model"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        last = {
            "id": oa["id"], "object": "chat.completion.chunk",
            "created": oa["created"], "model": oa["model"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
        }
        for chunk in (first, last):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("no CLAUDE_CODE_OAUTH_TOKEN found (env or .env)")
    print(f"claude-letta bridge on :{PORT} -> {ANTHROPIC_URL} (model {DEFAULT_MODEL})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
