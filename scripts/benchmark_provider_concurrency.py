"""Benchmark per-provider concurrency before 429 saturation.

For each OR provider, send N concurrent requests at increasing N and
measure: success / 429 / error rate, TTFT distribution. Used to decide
which providers belong in real-eval inventory.

Usage:
    OPENROUTER_API_KEY=sk-or-... uv run python scripts/benchmark_provider_concurrency.py

Cost estimate: ~$0.05-0.10 total (small prompts, 20 output tokens, ~558 requests).
Time: ~12-15 minutes (6 providers x 5 concurrency levels x 3 trials with gaps).
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
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="env var name for OR API key (try OPENROUTER_API_KEY_1 if main missing)",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        # fallback to numbered keys
        for i in range(1, 10):
            v = os.environ.get(f"OPENROUTER_API_KEY_{i}")
            if v:
                api_key = v
                print(f"using OPENROUTER_API_KEY_{i}", file=sys.stderr)
                break
    if not api_key:
        print(
            "ERROR: no OPENROUTER_API_KEY found in env (set one of "
            "OPENROUTER_API_KEY, OPENROUTER_API_KEY_1..9)",
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    print("\n# Provider concurrency benchmark")
    print(f"\nmodel={MODEL}, prompt={MAX_TOKENS} max_tokens, levels={args.levels}, trials={args.trials}, gap={args.gap_sec}s\n")
    print("| Provider | " + " | ".join(f"N={n}" for n in args.levels) + " | First 429 at | TTFT P50@N=1 |")
    print("|---|" + "---|" * (len(args.levels) + 2))

    for provider in args.providers:
        row_cells: list[str] = []
        first_429_n: int | None = None
        ttft_at_n1: float | None = None
        provider_results: dict[str, dict] = {}

        for n in args.levels:
            agg_ok = 0
            agg_429 = 0
            agg_err = 0
            agg_ttft: list[float] = []
            for _trial in range(args.trials):
                results = burst_test(api_key, provider, n)
                summary = summarize(results)
                agg_ok += summary["ok"]
                agg_429 += summary["rl_429"]
                agg_err += summary["err"]
                agg_ttft.extend(t for s, t, _ in results if s == 200)
                time.sleep(args.gap_sec)

            total = n * args.trials
            label = f"{agg_ok}/{total}"
            if agg_429:
                label += f" (429x{agg_429})"
            if agg_err:
                label += f" (errx{agg_err})"
            row_cells.append(label)

            if first_429_n is None and agg_429 > 0:
                first_429_n = n
            if n == 1 and agg_ttft:
                ttft_at_n1 = sorted(agg_ttft)[len(agg_ttft) // 2]

            provider_results[f"n_{n}"] = {
                "ok": agg_ok,
                "rl_429": agg_429,
                "err": agg_err,
                "ttft_ms_p50": (
                    sorted(agg_ttft)[len(agg_ttft) // 2] if agg_ttft else None
                ),
                "ttft_ms_max": max(agg_ttft) if agg_ttft else None,
            }

        row_cells.append(f"N={first_429_n}" if first_429_n else "no 429")
        row_cells.append(f"{ttft_at_n1:.0f}ms" if ttft_at_n1 is not None else "-")
        print(f"| {provider} | " + " | ".join(row_cells) + " |")
        all_results[provider] = {
            "first_429_at": first_429_n,
            "ttft_ms_at_n1": ttft_at_n1,
            "by_level": provider_results,
        }

    args.output.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull JSON: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
