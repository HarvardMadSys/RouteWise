#!/usr/bin/env python3
"""Test actual request-level concurrency on Ollama Cloud.

Sends N concurrent requests to the same model and measures whether they
execute in parallel or get serialized. This determines whether Ollama's
"3 concurrent cloud models" translates to request-level concurrency.

Usage:
    source .venv/bin/activate
    export OLLAMA_API_KEY=...
    python scripts/perf/ollama_concurrency_test.py --model glm-4.7:cloud --concurrency 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

BASE_URL = "https://ollama.com/api"
# Use a prompt that generates a predictable, moderate-length response.
DEFAULT_PROMPT = "Count from 1 to 20, one number per line."
DEFAULT_MAX_TOKENS = 128
TIMEOUT_SECONDS = 120.0


@dataclass
class Result:
    """Timing result for one concurrent request."""

    worker_id: int
    status: int
    start_time: float
    end_time: float
    wall_seconds: float
    server_total_duration_ns: int
    eval_count: int
    error: str | None


def send_request(
    worker_id: int,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str,
    barrier: threading.Barrier,
) -> Result:
    """Send one request, synchronized via barrier for simultaneous launch."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }).encode("utf-8")

    request = urllib.request.Request(
        url=f"{BASE_URL}/chat",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    # Wait for all workers to be ready before firing.
    barrier.wait()

    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
            end = time.monotonic()
            data = json.loads(raw) if raw else {}
            return Result(
                worker_id=worker_id,
                status=resp.getcode(),
                start_time=start,
                end_time=end,
                wall_seconds=end - start,
                server_total_duration_ns=int(data.get("total_duration") or 0),
                eval_count=int(data.get("eval_count") or 0),
                error=None,
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        end = time.monotonic()
        error_msg = raw
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                error_msg = str(parsed.get("error") or raw)
        except json.JSONDecodeError:
            pass
        return Result(
            worker_id=worker_id,
            status=exc.code,
            start_time=start,
            end_time=end,
            wall_seconds=end - start,
            server_total_duration_ns=0,
            eval_count=0,
            error=error_msg,
        )
    except Exception as exc:
        end = time.monotonic()
        return Result(
            worker_id=worker_id,
            status=0,
            start_time=start,
            end_time=end,
            wall_seconds=end - start,
            server_total_duration_ns=0,
            eval_count=0,
            error=str(exc),
        )


def analyze_concurrency(results: list[Result]) -> None:
    """Analyze timing overlap to determine effective concurrency."""
    ok_results = [r for r in results if r.status == 200]
    failed = [r for r in results if r.status != 200]

    print(f"\n{'='*70}")
    print(f"  Ollama Cloud Concurrency Test Results")
    print(f"{'='*70}")
    print(f"  Total requests sent:  {len(results)}")
    print(f"  Successful:           {len(ok_results)}")
    print(f"  Failed:               {len(failed)}")

    if failed:
        print(f"\n  --- Failures ---")
        for r in failed:
            print(f"  Worker {r.worker_id}: status={r.status} error={r.error}")

    if not ok_results:
        print("  No successful requests to analyze.")
        return

    # Normalize times relative to earliest start.
    t0 = min(r.start_time for r in ok_results)
    print(f"\n  --- Timing (relative to first request) ---")
    print(f"  {'Worker':>8} {'Start':>8} {'End':>8} {'Wall(s)':>8} "
          f"{'Server(s)':>10} {'Tokens':>7}")
    for r in sorted(ok_results, key=lambda x: x.start_time):
        server_s = r.server_total_duration_ns / 1e9
        print(f"  {r.worker_id:>8} {r.start_time - t0:>8.2f} "
              f"{r.end_time - t0:>8.2f} {r.wall_seconds:>8.2f} "
              f"{server_s:>10.2f} {r.eval_count:>7}")

    # Compute overlap: at each point in time, how many requests are in-flight?
    events: list[tuple[float, int]] = []
    for r in ok_results:
        events.append((r.start_time - t0, +1))
        events.append((r.end_time - t0, -1))
    events.sort(key=lambda e: (e[0], e[1]))

    max_concurrent = 0
    current = 0
    for _, delta in events:
        current += delta
        max_concurrent = max(max_concurrent, current)

    # Serialization check: if fully parallel, total wall time ~= max single request.
    # If fully serial, total wall time ~= sum of all wall times.
    wall_times = [r.wall_seconds for r in ok_results]
    total_span = max(r.end_time for r in ok_results) - min(r.start_time for r in ok_results)
    sum_wall = sum(wall_times)
    max_wall = max(wall_times)
    parallelism_ratio = sum_wall / total_span if total_span > 0 else 1.0

    print(f"\n  --- Concurrency Analysis ---")
    print(f"  Peak in-flight requests:   {max_concurrent}")
    print(f"  Total span (wall):         {total_span:.2f}s")
    print(f"  Sum of individual walls:   {sum_wall:.2f}s")
    print(f"  Max single wall:           {max_wall:.2f}s")
    print(f"  Parallelism ratio:         {parallelism_ratio:.2f}x")
    print()

    if parallelism_ratio > len(ok_results) * 0.8:
        print(f"  CONCLUSION: Requests appear SERIALIZED (ratio ~= N).")
        print(f"  Effective request-level concurrency: ~1")
    elif parallelism_ratio < 1.5:
        print(f"  CONCLUSION: Requests appear FULLY PARALLEL.")
        print(f"  Effective request-level concurrency: >= {len(ok_results)}")
    else:
        inferred_c = round(parallelism_ratio)
        print(f"  CONCLUSION: Partial parallelism detected.")
        print(f"  Estimated effective request-level concurrency: ~{inferred_c}")

    print(f"{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Ollama Cloud request-level concurrency."
    )
    parser.add_argument("--model", default="glm-4.7:cloud",
                        help="Model to test (default: glm-4.7:cloud)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of concurrent requests to send (default: 10)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="Prompt to use for each request")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Max tokens per request (default: 128)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of rounds to run (default: 1)")
    args = parser.parse_args()

    api_key = os.environ.get("OLLAMA_API_KEY") or os.environ.get("LLAMA_API_KEY")
    if not api_key:
        print("OLLAMA_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Testing concurrency on {args.model} with {args.concurrency} "
          f"simultaneous requests...")

    for round_num in range(1, args.rounds + 1):
        if args.rounds > 1:
            print(f"\n--- Round {round_num}/{args.rounds} ---")

        barrier = threading.Barrier(args.concurrency)
        results: list[Result] = [None] * args.concurrency  # type: ignore[list-item]

        def worker(wid: int) -> None:
            results[wid] = send_request(
                worker_id=wid,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                api_key=api_key,
                barrier=barrier,
            )

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(args.concurrency)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        analyze_concurrency(results)  # type: ignore[arg-type]

        if round_num < args.rounds:
            print("Sleeping 5s between rounds...")
            time.sleep(5)


if __name__ == "__main__":
    main()
