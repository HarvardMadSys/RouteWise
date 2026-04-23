"""Sustained quota probe — measure rate limits and quota depletion over time.

Unlike the concurrency probe, this script issues sequential (concurrency=1)
requests for a configurable duration (default 30 min), recording the success
rate and any rate-limit errors. The resulting data tells us:

  - Sustained request-per-minute rate the provider allows.
  - Whether/when 429-style rate-limit kicks in.
  - Extrapolated quota size for a longer window (e.g. 5h).

Usage example (on gpu2 for minimum RTT):
    python probe_quota_sustained.py \
        --provider ollama \
        --base-url https://ollama.com \
        --api-key-env OLLAMA_API_KEY \
        --model minimax-m2.5 \
        --duration-minutes 30 \
        --output ollama_quota_mm25.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import httpx


_DEFAULT_PROMPT = "Write one short paragraph about transformers."


def _single_request(
    client: httpx.Client,
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    extra_headers: dict | None,
    extra_payload: dict | None,
) -> dict:
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
    error: str | None = None
    status: int | None = None
    finished = False

    try:
        with client.stream(
            "POST", endpoint, json=payload, headers=headers, timeout=120.0,
        ) as response:
            status = response.status_code
            if status != 200:
                body = response.read()
                error = f"HTTP {status}: {body[:300].decode(errors='replace')}"
            else:
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    text = chunk.decode(errors="replace")
                    if ttft_ms is None and ('"content"' in text or '"delta"' in text):
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                    if "[DONE]" in text:
                        finished = True
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    e2e_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "t_wall": time.time(),
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "status": status,
        "error": error,
        "finished": finished,
    }


def _flush_summary(records: list[dict]) -> dict:
    successes = [r for r in records if r["error"] is None]
    errors = [r for r in records if r["error"] is not None]
    rate_limit_errors = [
        r for r in errors
        if r["status"] in (429, 529) or "429" in (r["error"] or "") or "rate" in (r["error"] or "").lower()
    ]
    ttfts = sorted([r["ttft_ms"] for r in successes if r["ttft_ms"] is not None])

    def pct(p: float) -> float:
        if not ttfts:
            return 0.0
        k = max(0, min(len(ttfts) - 1, int(round(p * (len(ttfts) - 1)))))
        return float(ttfts[k])

    if records:
        duration_sec = records[-1]["t_wall"] - records[0]["t_wall"]
    else:
        duration_sec = 0.0

    return {
        "total_requests": len(records),
        "successes": len(successes),
        "errors": len(errors),
        "rate_limit_errors": len(rate_limit_errors),
        "first_rate_limit_at_request": next(
            (i for i, r in enumerate(records) if r in rate_limit_errors), None,
        ),
        "duration_sec": duration_sec,
        "effective_rate_per_min": len(successes) / max(duration_sec / 60.0, 0.01),
        "extrapolated_quota_1h": int(len(successes) / max(duration_sec / 3600.0, 0.001)),
        "extrapolated_quota_5h": int(len(successes) * 5 / max(duration_sec / 3600.0, 0.001)),
        "ttft_p50_ms": pct(0.50),
        "ttft_p95_ms": pct(0.95),
        "ttft_p99_ms": pct(0.99),
        "ttft_mean_ms": float(statistics.fmean(ttfts)) if ttfts else 0.0,
        "error_samples": [
            r["error"] for r in errors[:5]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint-path", default="/v1/chat/completions")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument(
        "--max-rate-per-sec",
        type=float,
        default=2.0,
        help="Cap on request rate to be a good citizen; 2.0 req/s is gentle",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--output", default="quota_probe.json")
    parser.add_argument(
        "--stop-on-sustained-limit",
        type=int,
        default=30,
        help="Stop early if this many consecutive rate-limit errors occur",
    )
    parser.add_argument("--extra-headers", default=None)
    parser.add_argument("--extra-payload", default=None)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"env var {args.api_key_env} is empty")

    endpoint = args.base_url.rstrip("/") + args.endpoint_path
    extra_headers = json.loads(args.extra_headers) if args.extra_headers else None
    extra_payload = json.loads(args.extra_payload) if args.extra_payload else None

    print(f"Provider={args.provider}  endpoint={endpoint}  model={args.model}")
    print(
        f"Duration={args.duration_minutes}min  "
        f"max rate={args.max_rate_per_sec}/s  "
        f"early-stop after {args.stop_on_sustained_limit} consecutive rate-limits"
    )

    end_time = time.time() + args.duration_minutes * 60.0
    records: list[dict] = []
    consecutive_rate_limits = 0
    min_interval = 1.0 / max(args.max_rate_per_sec, 0.01)

    with httpx.Client(http2=False) as client:
        next_ok = time.time()
        while time.time() < end_time:
            now = time.time()
            if now < next_ok:
                time.sleep(next_ok - now)
            t0 = time.time()
            r = _single_request(
                client,
                endpoint,
                api_key,
                args.model,
                args.prompt,
                args.max_tokens,
                extra_headers,
                extra_payload,
            )
            records.append(r)

            # Detect consecutive rate-limit behavior for early stop.
            is_rl = (
                r["status"] in (429, 529)
                or (r["error"] and ("429" in r["error"] or "rate" in r["error"].lower()))
            )
            if is_rl:
                consecutive_rate_limits += 1
            else:
                consecutive_rate_limits = 0
            if consecutive_rate_limits >= args.stop_on_sustained_limit:
                print(
                    f"  ! {consecutive_rate_limits} consecutive rate-limits, "
                    f"stopping early at request #{len(records)}"
                )
                break

            # Status every 20 requests.
            if len(records) % 20 == 0:
                s = _flush_summary(records)
                print(
                    f"  #{len(records):<4d} rate={s['effective_rate_per_min']:.1f}/min "
                    f"succ={s['successes']}  errs={s['errors']}  "
                    f"p50={s['ttft_p50_ms']:.0f}ms"
                )

            next_ok = t0 + min_interval

    report = {
        "provider": args.provider,
        "base_url": args.base_url,
        "endpoint": endpoint,
        "model": args.model,
        "duration_minutes": args.duration_minutes,
        "max_rate_per_sec": args.max_rate_per_sec,
        "summary": _flush_summary(records),
        "records": records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out_path}")
    s = report["summary"]
    print(
        f"Final: {s['successes']}/{s['total_requests']} succ  "
        f"rate={s['effective_rate_per_min']:.1f}/min  "
        f"extrapolated 5h quota={s['extrapolated_quota_5h']}  "
        f"first_rate_limit_at=#{s['first_rate_limit_at_request']}"
    )


if __name__ == "__main__":
    main()
