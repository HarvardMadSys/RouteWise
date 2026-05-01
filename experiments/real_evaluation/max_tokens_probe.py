"""Probe whether providers generate up to ``max_tokens`` or stop naturally.

This is a real-API diagnostic, not a paper experiment. It sends the same prompt
to one configured provider while sweeping ``max_tokens`` and records actual
completion tokens, TTFT, E2E latency, and cost.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from experiments.real_evaluation.inventory import load_inventory
from experiments.real_evaluation.transports import build_transport

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Reply with exactly one word: OK"
DEFAULT_SWEEP = "1,2,4,8,16,32,64,128,256"

CSV_FIELDS = [
    "provider",
    "trial",
    "max_tokens",
    "status",
    "http_status",
    "ttft_ms",
    "e2e_ms",
    "prompt_tokens",
    "completion_tokens",
    "billed_cost_usd",
    "retry_count",
    "retry_sleep_ms",
    "rate_limited",
    "error_message",
]


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value <= 0:
            raise ValueError("max_tokens values must be positive")
        values.append(value)
    if not values:
        raise ValueError("at least one max_tokens value is required")
    return values


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["max_tokens"])].append(row)

    by_max_tokens: list[dict[str, Any]] = []
    for max_tokens in sorted(grouped):
        group = grouped[max_tokens]
        successes = [row for row in group if row["status"] == "success"]
        completion = [int(row["completion_tokens"]) for row in successes]
        ttft = [float(row["ttft_ms"]) for row in successes]
        e2e = [float(row["e2e_ms"]) for row in successes]
        by_max_tokens.append(
            {
                "max_tokens": max_tokens,
                "n": len(group),
                "successes": len(successes),
                "completion_tokens_min": min(completion) if completion else None,
                "completion_tokens_mean": (
                    statistics.fmean(completion) if completion else None
                ),
                "completion_tokens_max": max(completion) if completion else None,
                "ttft_ms_mean": statistics.fmean(ttft) if ttft else None,
                "e2e_ms_mean": statistics.fmean(e2e) if e2e else None,
                "rate_limited": sum(1 for row in group if row["rate_limited"]),
                "failures": len(group) - len(successes),
            }
        )

    return {
        "by_max_tokens": by_max_tokens,
        "interpretation": (
            "If completion_tokens stays roughly flat as max_tokens grows, "
            "max_tokens is acting as a cap. If completion_tokens tracks "
            "max_tokens, the provider/prompt is generating until the cap."
        ),
    }


def run_probe(
    *,
    inventory_path: Path,
    provider_name: str,
    prompt: str,
    max_tokens_values: list[int],
    repeats: int,
    timeout_sec: int,
    sleep_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = load_inventory(inventory_path)
    spec = next((p for p in inventory.providers if p.name == provider_name), None)
    if spec is None:
        known = ", ".join(p.name for p in inventory.providers)
        raise ValueError(f"unknown provider {provider_name!r}; known providers: {known}")

    transport = build_transport(spec.transport_cfg, requests.Session())
    rows: list[dict[str, Any]] = []
    for max_tokens in max_tokens_values:
        for trial in range(repeats):
            logger.info(
                "provider=%s max_tokens=%d trial=%d/%d",
                provider_name,
                max_tokens,
                trial + 1,
                repeats,
            )
            result = transport.send(prompt, max_tokens=max_tokens, timeout=timeout_sec)
            row = {
                "provider": provider_name,
                "trial": trial,
                "max_tokens": max_tokens,
                "status": result.status,
                "http_status": result.http_status,
                "ttft_ms": result.ttft_ms,
                "e2e_ms": result.e2e_ms,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "billed_cost_usd": result.billed_cost_usd,
                "retry_count": result.retry_count,
                "retry_sleep_ms": result.retry_sleep_ms,
                "rate_limited": result.rate_limited,
                "error_message": result.error_message,
            }
            rows.append(row)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    return rows, _summarize(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep max_tokens for one real-eval provider."
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--provider", required=True, help="Provider name in inventory.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", default=DEFAULT_SWEEP)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    rows, summary = run_probe(
        inventory_path=args.inventory,
        provider_name=args.provider,
        prompt=args.prompt,
        max_tokens_values=_parse_int_list(args.max_tokens),
        repeats=max(args.repeats, 1),
        timeout_sec=args.timeout_sec,
        sleep_sec=args.sleep_sec,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s and %s", args.output, summary_path)
    logger.info("summary: %s", json.dumps(summary["by_max_tokens"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
