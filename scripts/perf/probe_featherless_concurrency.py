"""Probe Featherless AI to measure concurrency-bound latency scaling.

Purpose
-------
Featherless provides S_C (concurrency-limited) inference. We need the real
P50/P99 TTFT at each concurrency level to calibrate the synthetic simulator's
S_C provider, so the λ(u) curve reflects actual behavior.

We hold total request volume roughly constant across concurrency levels, so
differences in latency come from queueing/saturation rather than workload.

Usage (intended for gpu2 to minimize RTT):
    python probe_featherless_concurrency.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --concurrencies 1 2 3 4 \
        --samples 30 \
        --output featherless_probe.json

Outputs a JSON summary: P50/P95/P99 TTFT per concurrency level, plus the
raw per-request measurements (for empirical resampling).
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


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


_DEFAULT_PROMPT = (
    "Write a one-paragraph explanation of the Transformer architecture."
)


async def _single_request(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
) -> dict:
    """Fire a streaming chat completion and return TTFT + E2E."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    t_start = time.perf_counter()
    ttft_ms = None
    first_byte_ms = None
    error: str | None = None
    status: int | None = None

    try:
        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=120.0,
        ) as response:
            status = response.status_code
            if status != 200:
                body = await response.aread()
                error = f"HTTP {status}: {body[:200].decode(errors='replace')}"
            else:
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    if first_byte_ms is None:
                        first_byte_ms = (time.perf_counter() - t_start) * 1000.0
                    # The first content chunk (not just headers) defines TTFT.
                    text = chunk.decode(errors="replace")
                    if '"content"' in text and ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                    # Drain the rest.
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    e2e_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "request_id": request_id,
        "ttft_ms": ttft_ms,
        "first_byte_ms": first_byte_ms,
        "e2e_ms": e2e_ms,
        "status": status,
        "error": error,
    }


async def _run_at_concurrency(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    samples: int,
) -> list[dict]:
    """Issue `samples` requests using `concurrency` workers."""
    results: list[dict] = []
    request_idx = 0

    async def worker(worker_id: int, client: httpx.AsyncClient) -> None:
        nonlocal request_idx
        per_worker = samples // concurrency + (1 if worker_id < samples % concurrency else 0)
        for _ in range(per_worker):
            rid = request_idx
            request_idx += 1
            res = await _single_request(
                client, base_url, api_key, model, prompt, max_tokens, rid,
            )
            res["worker_id"] = worker_id
            res["concurrency"] = concurrency
            results.append(res)

    async with httpx.AsyncClient(http2=False) as client:
        workers = [worker(i, client) for i in range(concurrency)]
        await asyncio.gather(*workers)

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        # Nearest-rank percentile for small n; avoid numpy dependency.
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return float(s[k])

    return {
        "count": n,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "mean": float(statistics.fmean(s)),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model id as expected by Featherless (OpenAI-compatible endpoint)",
    )
    parser.add_argument(
        "--concurrencies",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Concurrency levels to sweep",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Requests per concurrency level",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="max_tokens per request (keep small to control cost)",
    )
    parser.add_argument(
        "--prompt",
        default=_DEFAULT_PROMPT,
        help="User prompt (kept short for consistent input size)",
    )
    parser.add_argument(
        "--output",
        default="featherless_probe.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    base_url = os.environ.get("FEATHERLESS_BASE_URL", "https://api.featherless.ai")
    api_key = os.environ.get("FEATHERLESS_API_KEY")
    if not api_key:
        raise SystemExit("FEATHERLESS_API_KEY not set in environment")

    print(f"Probing {base_url} with model={args.model}")
    print(f"Concurrencies: {args.concurrencies}  samples per level: {args.samples}")

    report: dict = {
        "base_url": base_url,
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
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                concurrency=c,
                samples=args.samples,
            ),
        )
        elapsed = time.perf_counter() - t0
        summary = _summarize(raw)
        summary["elapsed_seconds"] = round(elapsed, 2)
        print(
            f"  n={summary['successes']}/{summary['total_requests']}  "
            f"TTFT p50={summary['ttft_ms'].get('p50', 'n/a')}  "
            f"p95={summary['ttft_ms'].get('p95', 'n/a')}  "
            f"p99={summary['ttft_ms'].get('p99', 'n/a')}  "
            f"errors={summary['errors']}  "
            f"(elapsed {elapsed:.1f}s)"
        )
        if summary["error_samples"]:
            for err in summary["error_samples"][:2]:
                print(f"  ! {err}")

        report["levels"].append({
            "concurrency": c,
            "summary": summary,
            "raw": raw,  # keep full samples for empirical resampling
        })

        # Small cooldown between levels so rate limiting doesn't spill over.
        if c != args.concurrencies[-1]:
            time.sleep(3)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
