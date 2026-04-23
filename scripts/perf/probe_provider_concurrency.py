"""Generic concurrency probe for OpenAI-compatible endpoints.

Works for any provider that implements /v1/chat/completions with streaming:
Featherless, Chutes, Ollama Cloud, MiniMax, OpenRouter, etc. The script
measures TTFT + E2E at several concurrency levels so we can fit the
lambda(u) congestion curve used by the joint router's effective cost.

Usage example:
    python probe_provider_concurrency.py \
        --provider featherless \
        --base-url https://api.featherless.ai \
        --api-key-env FEATHERLESS_API_KEY \
        --model MiniMaxAI/MiniMax-M2.5 \
        --concurrencies 1 2 3 4 \
        --samples 20 \
        --output /tmp/featherless_minimax.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import httpx


_DEFAULT_PROMPT = (
    "Write one concise paragraph about transformer attention."
)


async def _single_request(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
    concurrency: int,
    extra_headers: dict | None,
    extra_payload: dict | None,
) -> dict:
    """Fire a streaming chat completion and record TTFT and E2E."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }
    if extra_payload:
        payload.update(extra_payload)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    t_start = time.perf_counter()
    ttft_ms: float | None = None
    first_byte_ms: float | None = None
    error: str | None = None
    status: int | None = None
    finished = False

    try:
        async with client.stream(
            "POST", endpoint, json=payload, headers=headers, timeout=180.0,
        ) as response:
            status = response.status_code
            if status != 200:
                body = await response.aread()
                error = f"HTTP {status}: {body[:300].decode(errors='replace')}"
            else:
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    if first_byte_ms is None:
                        first_byte_ms = (time.perf_counter() - t_start) * 1000.0
                    text = chunk.decode(errors="replace")
                    if ttft_ms is None and ('"content"' in text or '"delta"' in text):
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                    if "[DONE]" in text:
                        finished = True
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    e2e_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "request_id": request_id,
        "concurrency": concurrency,
        "ttft_ms": ttft_ms,
        "first_byte_ms": first_byte_ms,
        "e2e_ms": e2e_ms,
        "status": status,
        "error": error,
        "finished": finished,
    }


async def _run_at_concurrency(
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    samples: int,
    extra_headers: dict | None,
    extra_payload: dict | None,
) -> list[dict]:
    results: list[dict] = []
    request_idx = 0

    async def worker(worker_id: int, client: httpx.AsyncClient) -> None:
        nonlocal request_idx
        per_worker = samples // concurrency + (
            1 if worker_id < samples % concurrency else 0
        )
        for _ in range(per_worker):
            rid = request_idx
            request_idx += 1
            res = await _single_request(
                client,
                endpoint,
                api_key,
                model,
                prompt,
                max_tokens,
                rid,
                concurrency,
                extra_headers,
                extra_payload,
            )
            res["worker_id"] = worker_id
            results.append(res)

    async with httpx.AsyncClient(http2=False) as client:
        workers = [worker(i, client) for i in range(concurrency)]
        await asyncio.gather(*workers)

    return results


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return float(s[k])

    return {
        "count": n,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "mean": float(statistics.fmean(s)),
        "min": float(s[0]),
        "max": float(s[-1]),
    }


def _summarize(raw: list[dict]) -> dict:
    successes = [r for r in raw if r["error"] is None and r["ttft_ms"] is not None]
    errors = [r for r in raw if r["error"] is not None]

    ttft_stats = _percentiles([r["ttft_ms"] for r in successes])
    e2e_stats = _percentiles([r["e2e_ms"] for r in successes])

    return {
        "concurrency": raw[0]["concurrency"] if raw else None,
        "total_requests": len(raw),
        "successes": len(successes),
        "errors": len(errors),
        "error_samples": [r["error"] for r in errors[:3]],
        "ttft_ms": ttft_stats,
        "e2e_ms": e2e_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, help="Provider label for the output record")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True, help="Env var name holding the API key")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--endpoint-path",
        default="/v1/chat/completions",
        help="Path appended to base-url (default: /v1/chat/completions)",
    )
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--output", default="probe.json")
    parser.add_argument(
        "--extra-headers",
        default=None,
        help="JSON dict merged into the request headers",
    )
    parser.add_argument(
        "--extra-payload",
        default=None,
        help="JSON dict merged into the request body",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Seconds between concurrency levels",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"env var {args.api_key_env} is empty")

    endpoint = args.base_url.rstrip("/") + args.endpoint_path
    extra_headers = json.loads(args.extra_headers) if args.extra_headers else None
    extra_payload = json.loads(args.extra_payload) if args.extra_payload else None

    print(f"Provider={args.provider}")
    print(f"Endpoint={endpoint}")
    print(f"Model={args.model}")
    print(f"Concurrencies={args.concurrencies} samples/level={args.samples}")

    report: dict = {
        "provider": args.provider,
        "base_url": args.base_url,
        "endpoint": endpoint,
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "samples_per_concurrency": args.samples,
        "levels": [],
    }

    for c in args.concurrencies:
        print(f"\n--- concurrency={c} ---")
        t0 = time.perf_counter()
        raw = asyncio.run(
            _run_at_concurrency(
                endpoint=endpoint,
                api_key=api_key,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                concurrency=c,
                samples=args.samples,
                extra_headers=extra_headers,
                extra_payload=extra_payload,
            ),
        )
        elapsed = time.perf_counter() - t0
        summary = _summarize(raw)
        summary["elapsed_seconds"] = round(elapsed, 2)

        ttft = summary["ttft_ms"]
        print(
            f"  n={summary['successes']}/{summary['total_requests']}  "
            f"TTFT p50={ttft.get('p50', 'n/a')}  p95={ttft.get('p95', 'n/a')}  "
            f"p99={ttft.get('p99', 'n/a')}  errors={summary['errors']}  "
            f"({elapsed:.1f}s)"
        )
        for err in summary["error_samples"][:2]:
            print(f"  ! {err}")

        report["levels"].append({
            "concurrency": c,
            "summary": summary,
            "raw": raw,
        })

        if c != args.concurrencies[-1]:
            time.sleep(args.cooldown)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
