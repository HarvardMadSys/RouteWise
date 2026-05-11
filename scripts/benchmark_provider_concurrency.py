"""Benchmark per-provider concurrency before 429 saturation.

For each OR provider, send N concurrent requests at increasing N and
measure: success / 429 / error rate, TTFT distribution. Used to decide
which providers belong in real-eval inventory.

Each provider gets its OWN OpenRouter key (from OPENROUTER_API_KEY_1..N
in .env), and all providers test in parallel. This mimics the 8h
experiment setup (one key per policy) and isolates per-key rate limits.

Usage:
    set -a && . ./.env.runner && set +a
    uv run python scripts/benchmark_provider_concurrency.py

Cost estimate: ~$0.05-0.10 total (small prompts, 20 output tokens).
Time: ~2-3 minutes (parallel; each provider runs 5 levels x 3 trials).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "minimax/minimax-m2.5"
# Short prompt to keep cost low. Real workload prompts are 500-700 tokens
# but rate-limit behavior is mostly request-count not token-count gated.
PROMPT = "Write one sentence describing thunderstorm weather."
MAX_TOKENS = 20
PROVIDERS_DEFAULT = [
    "AtlasCloud",
    "Chutes",
    "DeepInfra",
    "Friendli",
    "Inceptron",
    "SambaNova",
]
CONCURRENCY_LEVELS_DEFAULT = [1, 2, 4, 8, 16]
TRIALS_PER_LEVEL = 3
TRIAL_GAP_SEC = 5
HTTP_TIMEOUT = 30.0


def send_one(api_key: str, provider_hint: str) -> tuple[int | str, float, str]:
    """Send one OR request pinned to ``provider_hint``. Return (status, ttft_ms, err)."""
    t0 = time.time()
    try:
        r = requests.post(
            OR_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": MAX_TOKENS,
                "stream": False,
                "provider": {"order": [provider_hint], "allow_fallbacks": False},
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=HTTP_TIMEOUT,
        )
        elapsed_ms = (time.time() - t0) * 1000.0
        if r.status_code == 200:
            return (200, elapsed_ms, "")
        # OR 429 body has provider details
        body = r.text[:200] if r.text else ""
        return (r.status_code, elapsed_ms, body)
    except requests.RequestException as exc:
        return ("ERR", (time.time() - t0) * 1000.0, str(exc)[:200])


def burst_test(api_key: str, provider: str, n_concurrent: int) -> list[tuple]:
    """Fire ``n_concurrent`` requests in parallel, collect results."""
    with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
        futures = [executor.submit(send_one, api_key, provider) for _ in range(n_concurrent)]
        return [f.result() for f in as_completed(futures)]


def summarize(results: list[tuple]) -> dict:
    n = len(results)
    by_status: dict[str, int] = {}
    for s, _, _ in results:
        by_status[str(s)] = by_status.get(str(s), 0) + 1
    ttfts = sorted(t for s, t, _ in results if s == 200)
    return {
        "n": n,
        "ok": by_status.get("200", 0),
        "rl_429": by_status.get("429", 0),
        "err": sum(v for k, v in by_status.items() if k not in ("200", "429")),
        "by_status": by_status,
        "ttft_p50": ttfts[len(ttfts) // 2] if ttfts else None,
        "ttft_max": max(ttfts) if ttfts else None,
    }


def benchmark_one_provider(
    provider: str,
    api_key: str,
    levels: list[int],
    trials: int,
    gap_sec: float,
) -> dict:
    """Run all concurrency levels sequentially for one provider on a dedicated key."""
    by_level: dict[str, dict] = {}
    first_429_n: int | None = None
    ttft_at_n1: float | None = None
    print(f"  [{provider}] start", flush=True)

    for n in levels:
        agg_ok = 0
        agg_429 = 0
        agg_err = 0
        agg_ttft: list[float] = []
        for _trial in range(trials):
            results = burst_test(api_key, provider, n)
            agg_ok += sum(1 for s, _, _ in results if s == 200)
            agg_429 += sum(1 for s, _, _ in results if s == 429)
            agg_err += sum(1 for s, _, _ in results if s not in (200, 429))
            agg_ttft.extend(t for s, t, _ in results if s == 200)
            time.sleep(gap_sec)

        if first_429_n is None and agg_429 > 0:
            first_429_n = n
        if n == 1 and agg_ttft:
            ttft_at_n1 = sorted(agg_ttft)[len(agg_ttft) // 2]

        by_level[f"n_{n}"] = {
            "ok": agg_ok,
            "rl_429": agg_429,
            "err": agg_err,
            "total": n * trials,
            "ok_rate": agg_ok / (n * trials) if trials else 0.0,
            "ttft_ms_p50": (
                sorted(agg_ttft)[len(agg_ttft) // 2] if agg_ttft else None
            ),
            "ttft_ms_max": max(agg_ttft) if agg_ttft else None,
        }
        print(
            f"  [{provider}] n={n}: ok={agg_ok}/{n * trials}, "
            f"429={agg_429}, err={agg_err}",
            flush=True,
        )

    return {
        "provider": provider,
        "first_429_at": first_429_n,
        "ttft_ms_at_n1": ttft_at_n1,
        "by_level": by_level,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", nargs="+", default=PROVIDERS_DEFAULT)
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=CONCURRENCY_LEVELS_DEFAULT,
    )
    parser.add_argument("--trials", type=int, default=TRIALS_PER_LEVEL)
    parser.add_argument("--gap-sec", type=float, default=TRIAL_GAP_SEC)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmarks/provider_concurrency.json"),
    )
    args = parser.parse_args()

    # Map each provider to a dedicated OR API key (from .env numbered keys).
    # Falls back to OPENROUTER_API_KEY for any provider missing a numbered slot.
    primary_key = os.environ.get("OPENROUTER_API_KEY")
    provider_keys: dict[str, str] = {}
    for idx, provider in enumerate(args.providers, start=1):
        env_var = f"OPENROUTER_API_KEY_{idx}"
        key = os.environ.get(env_var) or primary_key
        if not key:
            print(
                f"ERROR: no key for provider {provider!r} "
                f"(tried {env_var} and OPENROUTER_API_KEY)",
                file=sys.stderr,
            )
            return 1
        provider_keys[provider] = key
        print(
            f"  {provider:15s} -> {env_var}",
            file=sys.stderr,
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("\n# Provider concurrency benchmark (parallel, 1 key per provider)")
    print(
        f"\nmodel={MODEL}, max_tokens={MAX_TOKENS}, "
        f"levels={args.levels}, trials={args.trials}, gap={args.gap_sec}s",
        flush=True,
    )
    print(
        f"\nLaunching {len(args.providers)} parallel benchmarks "
        f"(each provider on its own OR key)\n",
        flush=True,
    )

    all_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(args.providers)) as executor:
        futures = {
            executor.submit(
                benchmark_one_provider,
                provider,
                provider_keys[provider],
                args.levels,
                args.trials,
                args.gap_sec,
            ): provider
            for provider in args.providers
        }
        for fut in as_completed(futures):
            provider = futures[fut]
            try:
                result = fut.result()
                all_results[provider] = result
            except Exception as exc:
                print(
                    f"  [{provider}] FAILED: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                all_results[provider] = {"error": str(exc)}

    print("\n# Final results table\n", flush=True)
    print(
        "| Provider | "
        + " | ".join(f"N={n}" for n in args.levels)
        + " | First 429 at | TTFT P50@N=1 |"
    )
    print("|---|" + "---:|" * len(args.levels) + "---|---:|")
    for provider in args.providers:
        r = all_results.get(provider, {})
        if "error" in r:
            print(f"| {provider} | ERROR: {r['error']} |")
            continue
        cells = []
        for n in args.levels:
            level = r["by_level"][f"n_{n}"]
            ok = level["ok"]
            tot = level["total"]
            label = f"{ok}/{tot}"
            if level["rl_429"]:
                label += f" (429x{level['rl_429']})"
            if level["err"]:
                label += f" (errx{level['err']})"
            cells.append(label)
        first_429 = (
            f"N={r['first_429_at']}" if r.get("first_429_at") else "no 429"
        )
        ttft = (
            f"{r['ttft_ms_at_n1']:.0f}ms"
            if r.get("ttft_ms_at_n1") is not None
            else "-"
        )
        cells.append(first_429)
        cells.append(ttft)
        print(f"| {provider} | " + " | ".join(cells) + " |")

    args.output.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull JSON: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
