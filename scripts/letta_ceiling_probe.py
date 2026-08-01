#!/usr/bin/env python3
"""
letta_ceiling_probe.py — find the exact uncached-input ceiling of the OAuth bridge.

Root cause of the Phase-1.4 block (corrected 2026-08-01): the Letta->bridge->Anthropic
429 is triggered by per-call INPUT SIZE, not OAuth pool contention. The bridge already
sends the SYSTEM prompt as cache_control:ephemeral blocks, so a warm system is cheap.
What a real Letta ReAct turn keeps *growing* is UNCACHED input: conversation history,
archive-search results injected as messages, and tool results. This probe isolates the
uncached-input ceiling by holding the system constant (small) and escalating the size of
a single user message, binary-searching the 200<->429 boundary.

Decision this informs: if the ceiling is comfortably above a trimmed Luke turn
(core blocks ~5.5k tok live in the cached system; the uncached remainder = recent
history + a few archive hits + tool schemas), then lever (b) context-trimming closes
1.4 on Path B with no new key. If the ceiling sits below even a minimal uncached
remainder, Path A (dedicated ANTHROPIC_API_KEY) is the honest recommendation.

Bare urllib, no deps. Talks to the live bridge daemon on :17596. Off the OAuth pool
insofar as this IS the driver — but per the corrected root cause, concurrency is not
the variable; size is. We vary ONLY size.
"""
import json
import sys
import time
import urllib.request
import urllib.error

BRIDGE = "http://localhost:17596/v1/chat/completions"
# ~4 chars/token is the usual English rule of thumb; we report both chars and est-tokens.
CHARS_PER_TOK = 4
PAD_UNIT = "word context filler token stream " * 1  # ~33 chars; repeat to size


def est_tokens(chars):
    return chars // CHARS_PER_TOK


def make_body(pad_chars):
    filler = ("lorem ipsum dolor sit amet consectetur adipiscing elit "
              * ((pad_chars // 55) + 1))[:pad_chars]
    return {
        "model": "claude",
        "max_tokens": 16,
        "messages": [
            {"role": "system", "content": "You are a terse echo. Reply with the single word OK."},
            {"role": "user", "content": f"{filler}\n\nReply with exactly: OK"},
        ],
    }


def probe(pad_chars, timeout=90):
    body = json.dumps(make_body(pad_chars)).encode()
    req = urllib.request.Request(BRIDGE, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return ("200", round(time.monotonic() - t0, 2))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:120]
        except Exception:
            pass
        return (f"{e.code}", round(time.monotonic() - t0, 2), detail)
    except Exception as e:
        return (f"ERR:{type(e).__name__}", round(time.monotonic() - t0, 2), str(e)[:120])


def main():
    # Coarse ladder first (chars). est-tokens shown for interpretation.
    ladder = [500, 2000, 6000, 12000, 20000, 32000, 48000, 64000]
    print("=== COARSE LADDER (uncached user-message size) ===")
    results = {}
    last_ok = 0
    first_fail = None
    for chars in ladder:
        res = probe(chars)
        results[chars] = res
        code = res[0]
        print(f"chars={chars:>6}  ~tok={est_tokens(chars):>6}  -> {res}")
        if code == "200":
            last_ok = chars
        elif code == "429" and first_fail is None:
            first_fail = chars
            break
        time.sleep(1.5)  # small gap; size is the variable, not rate
    if first_fail is None:
        print(f"\nNo 429 up to {ladder[-1]} chars (~{est_tokens(ladder[-1])} tok). "
              f"Ceiling is ABOVE the tested range — trimming is trivially sufficient.")
        return
    # Binary search between last_ok and first_fail
    print(f"\n=== BINARY SEARCH between OK={last_ok} and 429={first_fail} chars ===")
    lo, hi = last_ok, first_fail
    while hi - lo > 1500:
        mid = (lo + hi) // 2
        res = probe(mid)
        code = res[0]
        print(f"chars={mid:>6}  ~tok={est_tokens(mid):>6}  -> {res}")
        if code == "200":
            lo = mid
        elif code == "429":
            hi = mid
        else:
            print(f"  non-decisive ({code}); retrying once after 3s")
            time.sleep(3)
            res2 = probe(mid)
            print(f"  retry -> {res2}")
            if res2[0] == "200":
                lo = mid
            elif res2[0] == "429":
                hi = mid
            else:
                print("  still non-decisive; stopping binary search")
                break
        time.sleep(1.5)
    print(f"\n>>> UNCACHED-INPUT CEILING between ~{est_tokens(lo)} and ~{est_tokens(hi)} tokens "
          f"({lo}-{hi} chars). Last-OK={lo}, first-429={hi}.")


if __name__ == "__main__":
    main()
