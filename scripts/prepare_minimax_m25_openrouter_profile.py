"""Prepare MiniMax M2.5 OpenRouter TTFT samples from the cached phase5 run.

The source directory is named ``phase5_minimax_m25_24h``, but the available
CSV currently covers less than a full day. The generated metadata records the
observed duration explicitly so downstream users do not mistake the artifact
for a complete 24-hour measurement.

Run from the RouteWise package root:

    python -m scripts.prepare_minimax_m25_openrouter_profile \
        --endpoints-json /tmp/openrouter_minimax_m25_endpoints.json

Output:
    experiments/simulation/latency_profiles/minimax_m25_openrouter_24h.npz
    experiments/simulation/latency_profiles/minimax_m25_openrouter_24h.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOG = Path(
    os.environ.get(
        "ROUTEWISE_MINIMAX_M25_SOURCE_LOG",
        REPO_ROOT / "artifacts" / "private" / "minimax_m25_evaluation_log.csv",
    )
)
PROFILE_DIR = REPO_ROOT / "experiments" / "simulation" / "latency_profiles"
OUT_NPZ = PROFILE_DIR / "minimax_m25_openrouter_24h.npz"
OUT_META = PROFILE_DIR / "minimax_m25_openrouter_24h.json"

MIN_SAMPLES_PER_PROVIDER = 1_000
MAX_SAMPLES_PER_PROVIDER = 50_000
SUBSAMPLE_SEED = 42


def collect_ttft_per_provider(csv_path: Path) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Parse the phase5 log into ``{provider: [ttft_ms, ...]}`` plus run stats."""
    ttft: dict[str, list[float]] = defaultdict(list)
    timestamps: list[float] = []
    total_rows = 0
    skipped_rows = 0
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if row.get("status") != "success":
                skipped_rows += 1
                continue
            provider = row.get("actual_provider", "")
            if not provider or provider == "unknown":
                skipped_rows += 1
                continue
            try:
                value = float(row.get("ttft_ms", 0))
                ts = float(row.get("timestamp", 0))
            except ValueError:
                skipped_rows += 1
                continue
            if value <= 0 or ts <= 0:
                skipped_rows += 1
                continue
            ttft[provider].append(value)
            timestamps.append(ts)

    stats: dict[str, Any] = {
        "total_rows": total_rows,
        "valid_ttft_rows": sum(len(v) for v in ttft.values()),
        "skipped_rows": skipped_rows,
    }
    if timestamps:
        start_ts = min(timestamps)
        end_ts = max(timestamps)
        stats.update(
            {
                "timestamp_min": _iso_utc(start_ts),
                "timestamp_max": _iso_utc(end_ts),
                "observed_duration_hours": (end_ts - start_ts) / 3600.0,
                "is_full_day_observation": (end_ts - start_ts) >= 23.5 * 3600.0,
            }
        )
    return dict(ttft), stats


def subsample_if_large(
    ttft: dict[str, list[float]],
    cap: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Cap each provider's samples at ``cap`` via uniform random subsample."""
    output: dict[str, np.ndarray] = {}
    for provider, samples in ttft.items():
        arr = np.asarray(samples, dtype=np.float64)
        if arr.size > cap:
            idx = rng.choice(arr.size, size=cap, replace=False)
            idx.sort()
            arr = arr[idx]
        output[provider] = arr
    return output


def filter_too_few(
    ttft: dict[str, np.ndarray],
    min_samples: int,
) -> dict[str, np.ndarray]:
    """Drop providers with too few samples for stable empirical bootstrap."""
    return {p: arr for p, arr in ttft.items() if arr.size >= min_samples}


def load_openrouter_prices(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load provider price snapshot from OpenRouter endpoints JSON, if supplied."""
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    endpoints = raw.get("data", {}).get("endpoints", [])
    prices: dict[str, dict[str, Any]] = {}
    for endpoint in endpoints:
        provider = endpoint.get("provider_name")
        pricing = endpoint.get("pricing", {})
        if not provider:
            continue
        prompt = _price_per_m(pricing.get("prompt"))
        completion = _price_per_m(pricing.get("completion"))
        candidate = {
            "provider_name": provider,
            "tag": endpoint.get("tag"),
            "quantization": endpoint.get("quantization"),
            "input_price_per_m": prompt,
            "output_price_per_m": completion,
            "input_cache_read_price_per_m": _price_per_m(
                pricing.get("input_cache_read")
            ),
            "context_length": endpoint.get("context_length"),
            "max_completion_tokens": endpoint.get("max_completion_tokens"),
            "uptime_last_1d": endpoint.get("uptime_last_1d"),
            "status": endpoint.get("status"),
        }
        existing = prices.get(provider)
        if existing is None or (
            prompt,
            completion,
            str(candidate["tag"]),
        ) < (
            existing["input_price_per_m"],
            existing["output_price_per_m"],
            str(existing["tag"]),
        ):
            prices[provider] = candidate
    return prices


def build_metadata(
    ttft: dict[str, np.ndarray],
    *,
    source: Path,
    run_stats: dict[str, Any],
    prices: dict[str, dict[str, Any]],
    endpoints_json: Path | None,
    subsample_seed: int,
    max_samples: int,
    min_samples: int,
) -> dict[str, Any]:
    """Build provenance and per-provider summary metadata."""
    providers_meta: dict[str, dict[str, Any]] = {}
    for provider, arr in sorted(ttft.items()):
        meta = {
            "n_samples": int(arr.size),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "max_ms": float(np.max(arr)),
        }
        if provider in prices:
            meta["openrouter_price"] = prices[provider]
        providers_meta[provider] = meta

    return {
        "schema_version": "1.0",
        "artifact": OUT_NPZ.name,
        "source_log": source.name,
        "source_run_label": "phase5_minimax_m25_24h",
        "source_note": (
            "The full source path is omitted to keep metadata environment-independent. "
            "The run_stats.observed_duration_hours field records the actual CSV coverage."
        ),
        "model": "minimax_m25",
        "model_family": "minimax-m2.5",
        "tier": "S_A",
        "subsample_seed": subsample_seed,
        "max_samples_per_provider": max_samples,
        "min_samples_per_provider": min_samples,
        "run_stats": run_stats,
        "price_snapshot": {
            "source": (
                "https://openrouter.ai/api/v1/models/minimax/minimax-m2.5/endpoints"
            ),
            "captured_from": endpoints_json.name if endpoints_json else None,
            "unit": "USD per 1M tokens",
            "note": "Provider prices can change; refresh before paper numbers.",
        },
        "n_providers": len(providers_meta),
        "providers": providers_meta,
    }


def _price_per_m(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) * 1_000_000.0


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-log", type=Path, default=SOURCE_LOG)
    parser.add_argument("--out-npz", type=Path, default=OUT_NPZ)
    parser.add_argument("--out-meta", type=Path, default=OUT_META)
    parser.add_argument(
        "--endpoints-json",
        type=Path,
        default=None,
        help="Optional JSON captured from the OpenRouter endpoints API.",
    )
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_PROVIDER)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_PER_PROVIDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Reading {args.source_log}")
    raw, run_stats = collect_ttft_per_provider(args.source_log)
    print(
        f"  collected {len(raw)} providers, "
        f"{sum(len(v) for v in raw.values()):,} valid TTFT samples"
    )
    print(
        f"  observed duration: "
        f"{run_stats.get('observed_duration_hours', 0.0):.2f} hours"
    )

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    sub = subsample_if_large(raw, cap=args.max_samples, rng=rng)
    sub = filter_too_few(sub, min_samples=args.min_samples)
    print(
        f"  after cap={args.max_samples:,} and min={args.min_samples:,}: "
        f"{len(sub)} providers, {sum(arr.size for arr in sub.values()):,} samples"
    )

    prices = load_openrouter_prices(args.endpoints_json)
    if args.endpoints_json:
        matched = len(set(sub) & set(prices))
        print(f"  loaded price snapshot for {len(prices)} providers ({matched} matched)")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **sub)
    print(f"  wrote {args.out_npz} ({args.out_npz.stat().st_size / 1024:.1f} KB)")

    metadata = build_metadata(
        sub,
        source=args.source_log,
        run_stats=run_stats,
        prices=prices,
        endpoints_json=args.endpoints_json,
        subsample_seed=SUBSAMPLE_SEED,
        max_samples=args.max_samples,
        min_samples=args.min_samples,
    )
    with args.out_meta.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  wrote {args.out_meta}")

    print("\nPer-provider summary:")
    print(
        f"  {'provider':>12} {'n':>7} {'P50':>7} {'P90':>7} "
        f"{'P99':>8} {'$/M in':>8} {'$/M out':>8}"
    )
    for provider, meta in metadata["providers"].items():
        price = meta.get("openrouter_price", {})
        in_price = price.get("input_price_per_m")
        out_price = price.get("output_price_per_m")
        print(
            f"  {provider:>12} {meta['n_samples']:>7,} "
            f"{meta['p50_ms']:>7.0f} {meta['p90_ms']:>7.0f} "
            f"{meta['p99_ms']:>8.0f} "
            f"{'' if in_price is None else f'{in_price:.3f}':>8} "
            f"{'' if out_price is None else f'{out_price:.3f}':>8}"
        )


if __name__ == "__main__":
    main()
