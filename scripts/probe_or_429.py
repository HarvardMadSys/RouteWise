"""Probe whether 429s on OpenRouter sub-providers are caused by our single OR
account or by upstream sub-provider capacity.

Reads multiple OpenRouter keys from ``OPENROUTER_API_KEY1..9`` and bursts short
streaming requests at a pinned sub-provider. Compares:

* single key, sequentially: baseline (warm cache, no contention)
* single key, N concurrent: stresses one account
* K keys, M concurrent each: distributes load across accounts

If 429s appear in single-key bursts but go away when load is distributed across
keys, the bottleneck is per-OR-account QPS. If 429s persist even with K=9
distinct accounts, the bottleneck is upstream sub-provider capacity (and
adding keys does not help).

This script does NOT touch the runner or any RouteWise state; it issues raw
requests to ``/api/v1/chat/completions`` with ``provider.order`` pinned.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests

OR_BASE = "https://openrouter.ai/api/v1"
MODEL = "minimax/minimax-m2.5"
PROMPT = "Reply with one word: hi"
MAX_TOKENS = 8
REQUEST_TIMEOUT_SEC = 30.0


@dataclass
class ProbeResult:
    key_index: int
    provider_hint: str
    http_status: int | None
    ttft_ms: float | None
    e2e_ms: float
    rate_limited: bool
    error: str | None
    completion_tokens: int | None
    reported_cost_usd: float | None
    retry_after: str | None


def send_one(key_index: int, key: str, provider_hint: str) -> ProbeResult:
    start = time.perf_counter()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.org/routewise-artifact",
        "X-Title": "RouteWise OR 429 probe",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
        "provider": {
            "order": [provider_hint],
            "allow_fallbacks": False,
        },
    }
    try:
        resp = requests.post(
            f"{OR_BASE}/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        return ProbeResult(
            key_index=key_index,
            provider_hint=provider_hint,
            http_status=None,
            ttft_ms=None,
            e2e_ms=(time.perf_counter() - start) * 1000.0,
            rate_limited=False,
            error=type(exc).__name__,
            completion_tokens=None,
            reported_cost_usd=None,
            retry_after=None,
        )

    status = resp.status_code
    if status != 200:
        retry_after = resp.headers.get("Retry-After")
        text = ""
        try:
            text = resp.text[:200]
        except Exception:
            text = ""
        resp.close()
        return ProbeResult(
            key_index=key_index,
            provider_hint=provider_hint,
            http_status=status,
            ttft_ms=None,
            e2e_ms=(time.perf_counter() - start) * 1000.0,
            rate_limited=(status == 429),
            error=text or None,
            completion_tokens=None,
            reported_cost_usd=None,
            retry_after=retry_after,
        )

    first_token_at: float | None = None
    completion_tokens: int | None = None
    reported_cost_usd: float | None = None
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data_str = raw[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content and first_token_at is None:
                    first_token_at = time.perf_counter()
            usage = chunk.get("usage") or {}
            if isinstance(usage, dict):
                if usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                if usage.get("cost") is not None:
                    reported_cost_usd = float(usage["cost"])
    finally:
        resp.close()

    ttft_ms = None if first_token_at is None else (first_token_at - start) * 1000.0
    return ProbeResult(
        key_index=key_index,
        provider_hint=provider_hint,
        http_status=200,
        ttft_ms=ttft_ms,
        e2e_ms=(time.perf_counter() - start) * 1000.0,
        rate_limited=False,
        error=None,
        completion_tokens=completion_tokens,
        reported_cost_usd=reported_cost_usd,
        retry_after=None,
    )


def verify_keys(keys: list[str]) -> None:
    """Hit /api/v1/key per key, print account label + remaining credits."""
    print("\n=== Key verification (/api/v1/key) ===")
    for ki, key in enumerate(keys, start=1):
        try:
            resp = requests.get(
                f"{OR_BASE}/key",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if resp.status_code != 200:
                print(f"  key {ki}: HTTP {resp.status_code}")
                continue
            data = resp.json().get("data") or {}
            label = data.get("label", "<no label>")
            usage = data.get("usage")
            limit = data.get("limit")
            limit_remaining = data.get("limit_remaining")
            print(
                f"  key {ki}: label={label!r} usage={usage} "
                f"limit={limit} remaining={limit_remaining}"
            )
        except requests.RequestException as exc:
            print(f"  key {ki}: error {type(exc).__name__}: {exc}")


def burst(
    keys: list[str],
    provider_hint: str,
    per_key_concurrency: int,
    label: str,
) -> list[ProbeResult]:
    tasks = [(ki, key) for ki, key in enumerate(keys, start=1) for _ in range(per_key_concurrency)]
    n_total = len(tasks)
    print(
        f"\n--- {label}: provider={provider_hint!r} keys={len(keys)} "
        f"concurrency/key={per_key_concurrency} total={n_total} ---"
    )
    started = time.perf_counter()
    results: list[ProbeResult] = []
    with ThreadPoolExecutor(max_workers=n_total or 1) as ex:
        futures = [ex.submit(send_one, ki, key, provider_hint) for ki, key in tasks]
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.perf_counter() - started
    print(f"  elapsed: {elapsed:.2f}s")

    successes = [r for r in results if r.http_status == 200]
    rate_limited = [r for r in results if r.rate_limited]
    other_err = [
        r for r in results if r.http_status not in (200, 429) or r.error is not None and not r.rate_limited
    ]
    other_err = [r for r in other_err if not (r.http_status == 200 and r.error is None)]

    n = len(results)
    print(
        f"  success: {len(successes)}/{n} ({100 * len(successes) / max(n, 1):.0f}%)  "
        f"429: {len(rate_limited)}/{n} ({100 * len(rate_limited) / max(n, 1):.0f}%)  "
        f"other_err: {len(other_err)}"
    )

    # Per-key breakdown only when multi-key
    if len(keys) > 1:
        per_key: dict[int, dict[str, int]] = {}
        for r in results:
            d = per_key.setdefault(r.key_index, {"n": 0, "ok": 0, "429": 0})
            d["n"] += 1
            if r.http_status == 200:
                d["ok"] += 1
            if r.rate_limited:
                d["429"] += 1
        for ki in sorted(per_key):
            d = per_key[ki]
            print(f"    key {ki}: {d['ok']}/{d['n']} ok, {d['429']} 429")

    ttfts = [r.ttft_ms for r in successes if r.ttft_ms is not None]
    if ttfts:
        ttfts_sorted = sorted(ttfts)
        p50 = ttfts_sorted[len(ttfts_sorted) // 2]
        p90 = ttfts_sorted[min(len(ttfts_sorted) - 1, int(len(ttfts_sorted) * 0.9))]
        mx = ttfts_sorted[-1]
        print(
            f"  TTFT (success only): n={len(ttfts)} mean={statistics.mean(ttfts):.0f}ms "
            f"p50={p50:.0f}ms p90={p90:.0f}ms max={mx:.0f}ms"
        )

    # Sample first error message for inspection
    if rate_limited:
        sample = rate_limited[0]
        retry_after = sample.retry_after or "<none>"
        snippet = (sample.error or "")[:160]
        print(f"  sample 429: Retry-After={retry_after} body={snippet!r}")

    costs = [r.reported_cost_usd for r in successes if r.reported_cost_usd is not None]
    if costs:
        print(f"  total reported usage cost: ${sum(costs):.6f}  mean=${statistics.mean(costs):.6f}")

    return results


def main() -> int:
    keys: list[str] = []
    for i in range(1, 10):
        v = os.environ.get(f"OPENROUTER_API_KEY{i}", "")
        if v:
            keys.append(v)
    if not keys:
        print("No OPENROUTER_API_KEY{1..9} found", file=sys.stderr)
        return 2
    print(f"Loaded {len(keys)} OpenRouter keys")

    verify_keys(keys)

    plan = [
        ("Inceptron", 1, 1, "phase A: 1 key sequential baseline"),
        ("Inceptron", 1, 5, "phase B: 1 key x 5 concurrent (single-account stress)"),
        ("Inceptron", len(keys), 3, "phase C: 9 keys x 3 concurrent (user's design, 27 total)"),
        ("Inceptron", len(keys), 8, "phase D: 9 keys x 8 concurrent (72 total)"),
        ("Chutes", 1, 1, "phase E: 1 key sequential baseline"),
        ("Chutes", 1, 5, "phase F: 1 key x 5 concurrent (single-account stress)"),
        ("Chutes", len(keys), 3, "phase G: 9 keys x 3 concurrent (27 total)"),
        ("Chutes", len(keys), 8, "phase H: 9 keys x 8 concurrent (72 total)"),
    ]

    cooldown_between_phases_sec = 8.0
    aggregated: list[tuple[str, list[ProbeResult]]] = []
    for provider, n_keys, conc, label in plan:
        key_slice = keys[:n_keys]
        results = burst(key_slice, provider, conc, label)
        aggregated.append((label, results))
        time.sleep(cooldown_between_phases_sec)

    print("\n=== Headline ===")
    for label, results in aggregated:
        n = len(results)
        ok = sum(1 for r in results if r.http_status == 200)
        rl = sum(1 for r in results if r.rate_limited)
        print(f"  {label}: ok={ok}/{n} 429={rl}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
