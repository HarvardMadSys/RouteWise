#!/usr/bin/env python3
"""Stress test tool for the hybrid inference server.

Usage:
  python scripts/perf/llama_stress.py \
    --model llama-3.3-70b-instruct \
    --rps 5 --duration 30

Notes:
- Tests the server at http://localhost:8000 (configurable via --base-url)
- Uses standard OpenAI-compatible API format
- Collects metrics that will be visible in Grafana
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

import aiohttp


async def make_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    request_id: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Make a single request to the server."""
    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
    }

    start_time = time.time()
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            latency = (time.time() - start_time) * 1000  # ms

            if response.status == 200:
                data = await response.json()
                return {
                    "success": True,
                    "latency_ms": latency,
                    "status": response.status,
                    "model": data.get("model", model),
                    "usage": data.get("usage", {}),
                }
            else:
                error_text = await response.text()
                if verbose:
                    print(f"Request {request_id} failed: {response.status} - {error_text[:100]}")
                return {
                    "success": False,
                    "latency_ms": latency,
                    "status": response.status,
                    "error": error_text[:200],
                }
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        if verbose:
            print(f"Request {request_id} exception: {e}")
        return {
            "success": False,
            "latency_ms": latency,
            "status": 0,
            "error": str(e)[:200],
        }


async def run_stress_test(
    base_url: str,
    model: str,
    prompt: str,
    rps: float,
    duration: float,
    max_concurrent: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the stress test."""
    # Create session with connection pooling
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        start_time = time.time()
        request_interval = 1.0 / rps
        request_count = 0
        tasks = []
        results = []

        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_request(req_id: str):
            async with semaphore:
                result = await make_request(session, base_url, model, prompt, req_id, verbose)
                results.append(result)
                if verbose and len(results) % 10 == 0:
                    success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
                    print(f"  {len(results)} requests completed, success rate: {success_rate:.1f}%")
                return result

        # Generate requests at the specified rate
        while (time.time() - start_time) < duration:
            request_id = f"stress_{request_count:06d}"
            task = asyncio.create_task(bounded_request(request_id))
            tasks.append(task)
            request_count += 1

            # Wait for the next request slot
            await asyncio.sleep(request_interval)

        # Wait for all pending requests to complete
        if tasks:
            print(f"\nWaiting for {len(tasks)} requests to complete...")
            await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate statistics
        elapsed_time = time.time() - start_time
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        latencies = [r["latency_ms"] for r in successful]
        if latencies:
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
            min_lat = min(latencies)
            max_lat = max(latencies)
            avg_lat = sum(latencies) / len(latencies)
        else:
            p50 = p95 = p99 = min_lat = max_lat = avg_lat = 0

        # Error analysis
        error_types = {}
        for r in failed:
            status = r.get("status", 0)
            key = f"Status {status}"
            error_types[key] = error_types.get(key, 0) + 1

        return {
            "duration_seconds": elapsed_time,
            "total_requests": len(results),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / max(1, len(results)) * 100,
            "actual_rps": len(results) / max(1, elapsed_time),
            "latency": {
                "min": min_lat,
                "max": max_lat,
                "avg": avg_lat,
                "p50": p50,
                "p95": p95,
                "p99": p99,
            },
            "error_types": error_types,
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stress test the hybrid inference server")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-instruct",
        help="Model to test (default: llama-3.3-70b-instruct)",
    )
    parser.add_argument(
        "--prompt",
        default="Hello, please respond briefly.",
        help="Prompt to send",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=2.0,
        help="Requests per second (default: 2.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Test duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="Max concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )
    return parser.parse_args()


async def main():
    """Run the main stress test program."""
    args = parse_args()

    print("=== Hybrid Inference Server Stress Test ===")
    print(f"Server: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Target RPS: {args.rps}")
    print(f"Duration: {args.duration}s")
    print(f"Max Concurrent: {args.concurrent}")
    print("Starting test...\n")

    results = await run_stress_test(
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        rps=args.rps,
        duration=args.duration,
        max_concurrent=args.concurrent,
        verbose=args.verbose,
    )

    # Print results
    print("\n=== Llama Stress Test Results ===")
    print(f"Model: {args.model}")
    print(f"Target RPS: {args.rps}, Duration: {args.duration}s")
    print(f"Total Requests: {results['total_requests']}")
    print(f"Success: {results['successful_requests']} ({results['success_rate']:.1f}%)")
    print(f"Failures: {results['failed_requests']}")
    print(f"Actual RPS: {results['actual_rps']:.2f}")

    if results["successful_requests"] > 0:
        print("Latency (ms):")
        lat = results["latency"]
        print(f"  P50: {lat['p50']:.1f}")
        print(f"  P95: {lat['p95']:.1f}")
        print(f"  P99: {lat['p99']:.1f}")
        print(f"  Min: {lat['min']:.1f}")
        print(f"  Max: {lat['max']:.1f}")

    if results["error_types"]:
        print("\nError Summary:")
        for error_type, count in sorted(results["error_types"].items(), key=lambda x: -x[1])[:5]:
            print(f"  {error_type}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
