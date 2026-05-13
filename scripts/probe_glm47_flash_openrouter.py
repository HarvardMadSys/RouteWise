"""Probe OpenRouter provider concurrency for z-ai/glm-4.7-flash.

This is a narrow live probe for one OpenRouter account key. It pins each
request to one provider via provider.order and disables fallbacks, then bursts
small requests at increasing concurrency levels.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


OR_BASE = "https://openrouter.ai/api/v1"
MODEL = "z-ai/glm-4.7-flash"
PROMPT = "Reply with exactly one word: ok"
MAX_TOKENS = 4
DEFAULT_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
REQUEST_TIMEOUT_SEC = 60.0
REQUEST_HARD_TIMEOUT_SEC = 35


def _deadline_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError(f"request exceeded {REQUEST_HARD_TIMEOUT_SEC}s hard deadline")


@dataclass
class RequestResult:
    provider: str
    level: int
    http_status: int | None
    ttft_ms: float | None
    e2e_ms: float
    error: str | None
    retry_after: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None


def env_key() -> str:
    key = (
        os.environ.get("OPENROUTER_API_KEY1")
        or os.environ.get("OPENROUTER_API_KEY_1")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY1, OPENROUTER_API_KEY_1, or OPENROUTER_API_KEY is required"
        )
    return key.removeprefix("Bearer ").strip()


def get_endpoints() -> list[dict[str, Any]]:
    resp = requests.get(f"{OR_BASE}/models/{MODEL}/endpoints", timeout=20)
    resp.raise_for_status()
    endpoints = ((resp.json().get("data") or {}).get("endpoints")) or []
    return endpoints


def send_one(api_key: str, provider: str, level: int) -> RequestResult:
    start = time.perf_counter()
    first_token_at: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/HarvardSys/hybridInference",
        "X-Title": "RouteWise GLM 4.7 Flash provider probe",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": False,
        "provider": {"order": [provider], "allow_fallbacks": False},
    }
    try:
        signal.signal(signal.SIGALRM, _deadline_handler)
        signal.alarm(REQUEST_HARD_TIMEOUT_SEC)
        resp = requests.post(
            f"{OR_BASE}/chat/completions",
            headers=headers,
            json=payload,
            stream=False,
            timeout=REQUEST_TIMEOUT_SEC,
        )
    except (requests.RequestException, TimeoutError) as exc:
        return RequestResult(
            provider=provider,
            level=level,
            http_status=None,
            ttft_ms=None,
            e2e_ms=(time.perf_counter() - start) * 1000,
            error=f"{type(exc).__name__}: {exc}",
            retry_after=None,
            prompt_tokens=None,
            completion_tokens=None,
            cost_usd=None,
        )
    finally:
        signal.alarm(0)

    if resp.status_code != 200:
        retry_after = resp.headers.get("Retry-After")
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            body = ""
        resp.close()
        return RequestResult(
            provider=provider,
            level=level,
            http_status=resp.status_code,
            ttft_ms=None,
            e2e_ms=(time.perf_counter() - start) * 1000,
            error=body or None,
            retry_after=retry_after,
            prompt_tokens=None,
            completion_tokens=None,
            cost_usd=None,
        )

    try:
        data = resp.json()
        usage = data.get("usage") or {}
        if isinstance(usage, dict):
            if usage.get("prompt_tokens") is not None:
                prompt_tokens = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            if usage.get("cost") is not None:
                cost_usd = float(usage["cost"])
    except (requests.RequestException, ValueError) as exc:
        return RequestResult(
            provider=provider,
            level=level,
            http_status=resp.status_code,
            ttft_ms=None if first_token_at is None else (first_token_at - start) * 1000,
            e2e_ms=(time.perf_counter() - start) * 1000,
            error=f"{type(exc).__name__}: {exc}",
            retry_after=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
        )
    finally:
        resp.close()

    return RequestResult(
        provider=provider,
        level=level,
        http_status=200,
        ttft_ms=None if first_token_at is None else (first_token_at - start) * 1000,
        e2e_ms=(time.perf_counter() - start) * 1000,
        error=None,
        retry_after=None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int((len(ordered) - 1) * p))
    return ordered[idx]


def summarize_level(results: list[RequestResult]) -> dict[str, Any]:
    ok = [r for r in results if r.http_status == 200 and not r.error]
    ttft = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    e2e = [r.e2e_ms for r in ok]
    statuses: dict[str, int] = {}
    for r in results:
        statuses[str(r.http_status)] = statuses.get(str(r.http_status), 0) + 1
    return {
        "total": len(results),
        "ok": len(ok),
        "ok_rate": len(ok) / len(results) if results else 0.0,
        "status_counts": statuses,
        "ttft_ms_mean": statistics.mean(ttft) if ttft else None,
        "ttft_ms_p50": percentile(ttft, 0.50),
        "ttft_ms_p90": percentile(ttft, 0.90),
        "ttft_ms_max": max(ttft) if ttft else None,
        "e2e_ms_mean": statistics.mean(e2e) if e2e else None,
        "e2e_ms_p50": percentile(e2e, 0.50),
        "e2e_ms_p90": percentile(e2e, 0.90),
        "e2e_ms_max": max(e2e) if e2e else None,
        "cost_usd_sum": sum(r.cost_usd for r in ok if r.cost_usd is not None),
        "sample_error": next((r.error for r in results if r.error), None),
        "retry_after": next((r.retry_after for r in results if r.retry_after), None),
    }


def burst(api_key: str, provider: str, level: int) -> list[RequestResult]:
    with ProcessPoolExecutor(max_workers=level) as ex:
        futures = [ex.submit(send_one, api_key, provider, level) for _ in range(level)]
        return [f.result() for f in as_completed(futures)]


def price_per_million(pricing: dict[str, Any], key: str) -> float | None:
    value = pricing.get(key)
    if value is None:
        return None
    return float(value) * 1_000_000


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# OpenRouter {MODEL} Provider Probe")
    lines.append("")
    lines.append(f"- run_at: {report['run_at']}")
    lines.append(f"- levels: {report['levels']}")
    lines.append(f"- prompt: {PROMPT!r}, max_tokens={MAX_TOKENS}")
    lines.append("")
    lines.append(
        "| Provider | Input $/M | Output $/M | Cache read $/M | Last 100% OK | Last >=95% OK | First degraded | Total cost | E2E p50@1 | E2E p50@max usable |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|")
    completed_providers = [
        provider for provider in report["provider_order"] if provider in report["providers"]
    ]
    for provider in completed_providers:
        item = report["providers"][provider]
        pricing = item["pricing"]
        max95 = item.get("max_level_ok_rate_gte_95")
        ttft_max95 = None
        if max95 is not None:
            ttft_max95 = item["levels"][str(max95)]["e2e_ms_p50"]
        lines.append(
            "| {provider} | {inp} | {out} | {cache} | {full} | {max95} | {degraded} | ${cost:.6f} | {ttft1} | {ttftmax} |".format(
                provider=provider,
                inp=f"{pricing.get('input_per_million'):.3f}" if pricing.get("input_per_million") is not None else "-",
                out=f"{pricing.get('output_per_million'):.3f}" if pricing.get("output_per_million") is not None else "-",
                cache=f"{pricing.get('cache_read_per_million'):.3f}" if pricing.get("cache_read_per_million") is not None else "-",
                full=item.get("max_level_100_ok") or "-",
                max95=max95 or "-",
                degraded=item.get("first_degraded_level") or "-",
                cost=item.get("total_reported_cost_usd") or 0.0,
                ttft1=fmt_ms(item["levels"].get("1", {}).get("e2e_ms_p50")),
                ttftmax=fmt_ms(ttft_max95),
            )
        )
    lines.append("")
    for provider in completed_providers:
        lines.append(f"## {provider}")
        lines.append("")
        lines.append("| N | OK | statuses | TTFT mean/p50/p90/max | E2E mean/p50/p90/max | cost |")
        lines.append("|---:|---:|---|---:|---:|---:|")
        for level in report["levels"]:
            s = report["providers"][provider]["levels"][str(level)]
            lines.append(
                "| {n} | {ok}/{total} | {statuses} | {ttft} | {e2e} | ${cost:.6f} |".format(
                    n=level,
                    ok=s["ok"],
                    total=s["total"],
                    statuses=json.dumps(s["status_counts"], ensure_ascii=False),
                    ttft=fmt_tuple(s, "ttft_ms"),
                    e2e=fmt_tuple(s, "e2e_ms"),
                    cost=s["cost_usd_sum"] or 0.0,
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}ms"


def fmt_tuple(summary: dict[str, Any], prefix: str) -> str:
    return "/".join(
        fmt_ms(summary.get(f"{prefix}_{suffix}"))
        for suffix in ("mean", "p50", "p90", "max")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=DEFAULT_LEVELS)
    parser.add_argument("--providers", nargs="+")
    parser.add_argument("--gap-sec", type=float, default=5.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/benchmarks/glm47_flash_openrouter_probe.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/benchmarks/glm47_flash_openrouter_probe.md"),
    )
    args = parser.parse_args()

    api_key = env_key()
    endpoints = get_endpoints()
    endpoint_by_provider = {e["provider_name"]: e for e in endpoints}
    providers = args.providers or list(endpoint_by_provider)
    missing = [p for p in providers if p not in endpoint_by_provider]
    if missing:
        print(f"Unknown providers for {MODEL}: {missing}", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "model": MODEL,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "levels": args.levels,
        "provider_order": providers,
        "providers": {},
    }
    print(f"model={MODEL} providers={providers} levels={args.levels}", flush=True)

    for provider in providers:
        endpoint = endpoint_by_provider[provider]
        pricing = endpoint.get("pricing") or {}
        item: dict[str, Any] = {
            "endpoint_name": endpoint.get("name"),
            "context_length": endpoint.get("context_length"),
            "max_completion_tokens": endpoint.get("max_completion_tokens"),
            "pricing": {
                "input_per_million": price_per_million(pricing, "prompt"),
                "output_per_million": price_per_million(pricing, "completion"),
                "cache_read_per_million": price_per_million(pricing, "input_cache_read"),
                "raw": pricing,
            },
            "levels": {},
            "total_reported_cost_usd": 0.0,
        }
        print(f"\n[{provider}] start", flush=True)
        for level in args.levels:
            started = time.perf_counter()
            results = burst(api_key, provider, level)
            summary = summarize_level(results)
            item["levels"][str(level)] = summary
            item["total_reported_cost_usd"] += summary["cost_usd_sum"] or 0.0
            elapsed = time.perf_counter() - started
            print(
                f"[{provider}] n={level}: ok={summary['ok']}/{summary['total']} "
                f"statuses={summary['status_counts']} "
                f"ttft_p50={fmt_ms(summary['ttft_ms_p50'])} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if args.gap_sec:
                time.sleep(args.gap_sec)

        full_ok_levels = [
            n
            for n in args.levels
            if item["levels"][str(n)]["ok"] == item["levels"][str(n)]["total"]
        ]
        ok95_levels = [
            n for n in args.levels if item["levels"][str(n)]["ok_rate"] >= 0.95
        ]
        degraded = next(
            (
                n
                for n in args.levels
                if item["levels"][str(n)]["ok"] < item["levels"][str(n)]["total"]
            ),
            None,
        )
        item["max_level_100_ok"] = max(full_ok_levels) if full_ok_levels else None
        item["max_level_ok_rate_gte_95"] = max(ok95_levels) if ok95_levels else None
        item["first_degraded_level"] = degraded
        report["providers"][provider] = item

        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        write_markdown(args.output_md, report)

    print(f"\nJSON: {args.output_json}")
    print(f"Markdown: {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
