#!/usr/bin/env python3
"""Concurrency test with warmup to eliminate cold-start noise.

Sends a few serial warmup requests, then immediately fires N concurrent
requests to measure steady-state concurrency.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

BASE_URL = "https://ollama.com/api"
PROMPT = "Count from 1 to 10."
MAX_TOKENS = 64
TIMEOUT = 120.0


def fire_one(api_key: str, model: str) -> tuple[int, float, float, int]:
    """Send one request. Returns (status, wall_s, server_s, tokens)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": MAX_TOKENS},
    }).encode()
    req = urllib.request.Request(
        url=f"{BASE_URL}/chat", data=payload, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
            t1 = time.monotonic()
            return (200, t1 - t0,
                    data.get("total_duration", 0) / 1e9,
                    data.get("eval_count", 0))
    except urllib.error.HTTPError as exc:
        t1 = time.monotonic()
        return (exc.code, t1 - t0, 0.0, 0)
    except Exception:
        t1 = time.monotonic()
        return (0, t1 - t0, 0.0, 0)


def concurrent_burst(api_key: str, model: str, n: int):
    """Fire n requests simultaneously and return per-request timings."""
    barrier = threading.Barrier(n)
    results = [None] * n

    def worker(i: int):
        barrier.wait()
        start = time.monotonic()
        status, wall, server, tokens = fire_one(api_key, model)
        end = time.monotonic()
        results[i] = {
            "id": i, "status": status,
            "start": start, "end": end,
            "wall": wall, "server": server, "tokens": tokens,
        }

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def main():
    api_key = os.environ.get("OLLAMA_API_KEY") or os.environ.get("LLAMA_API_KEY")
    if not api_key:
        print("OLLAMA_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    model = "glm-4.7:cloud"

    # Phase 1: Warmup (3 serial requests to ensure model is loaded)
    print("=== Phase 1: Warmup (3 serial requests) ===")
    for i in range(3):
        status, wall, server, tokens = fire_one(api_key, model)
        print(f"  Warmup {i+1}: status={status} wall={wall:.2f}s "
              f"server={server:.2f}s tokens={tokens}")

    # Phase 2: Test different concurrency levels
    for n in [2, 3, 4, 5, 6, 8]:
        print(f"\n=== Phase 2: Burst of {n} concurrent requests ===")
        # Small pause to let server settle.
        time.sleep(2)

        results = concurrent_burst(api_key, model, n)
        ok = [r for r in results if r["status"] == 200]
        t0 = min(r["start"] for r in ok)

        print(f"  {'ID':>4} {'Start':>7} {'End':>7} {'Wall':>7} "
              f"{'Server':>8} {'Tokens':>7}")
        for r in sorted(ok, key=lambda x: x["start"]):
            print(f"  {r['id']:>4} {r['start']-t0:>7.2f} {r['end']-t0:>7.2f} "
                  f"{r['wall']:>7.2f} {r['server']:>8.2f} {r['tokens']:>7}")

        span = max(r["end"] for r in ok) - min(r["start"] for r in ok)
        sum_wall = sum(r["wall"] for r in ok)
        ratio = sum_wall / span if span > 0 else 1.0
        # Also compute based on server-side time
        sum_server = sum(r["server"] for r in ok)
        server_ratio = sum_server / span if span > 0 else 1.0

        print(f"  Span: {span:.2f}s  SumWall: {sum_wall:.2f}s  "
              f"Ratio(wall): {ratio:.2f}x  Ratio(server): {server_ratio:.2f}x")

        # Check if any request had anomalous server time (>3x median)
        servers = sorted(r["server"] for r in ok)
        median_server = servers[len(servers) // 2]
        anomalies = [r for r in ok if r["server"] > median_server * 3]
        if anomalies:
            print(f"  WARNING: {len(anomalies)} request(s) with anomalous "
                  f"server time (>3x median={median_server:.2f}s)")


if __name__ == "__main__":
    main()
