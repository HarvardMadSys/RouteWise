"""Prepare per-provider TTFT latency profiles from real measurement logs.

Reads an ``evaluation_log.csv`` produced by a measurement run, extracts
per-provider TTFT samples, subsamples them, and writes a compact ``.npz``
plus a ``.json`` metadata sidecar for the simulator's empirical latency model.
It can also read real-eval ``shared_profile_events.jsonl`` logs, excluding
OpenRouter aggregate pseudo-providers such as ``__or_auto__`` by default.

The script is model-agnostic. It tolerates logs that lack a ``status`` or
``timestamp`` column: the ``status == "success"`` filter applies only when the
column is present, and run-duration stats are emitted only when timestamps are
available.

Run from the RouteWise package root, for example:

    python -m scripts.prepare_latency_profile \\
        --model qwen3_235b \\
        --source-log /path/to/qwen3/evaluation_log.csv \\
        --out-npz experiments/simulation/latency_profiles/qwen3_24h.npz

    python -m scripts.prepare_latency_profile \\
        --model minimax_m25 --model-family minimax-m2.5 \\
        --source-log /path/to/minimax/evaluation_log.csv \\
        --out-npz experiments/simulation/latency_profiles/minimax_m25_openrouter_24h.npz \\
        --endpoints-json /tmp/openrouter_minimax_m25_endpoints.json \\
        --price-source https://openrouter.ai/api/v1/models/minimax/minimax-m2.5/endpoints \\
        --run-label phase5_minimax_m25_24h
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

MIN_SAMPLES_PER_PROVIDER = 1_000
MAX_SAMPLES_PER_PROVIDER = 50_000
SUBSAMPLE_SEED = 42
PSEUDO_PROVIDER_RE = re.compile(r"^__.*__$")


def normalize_provider(raw: str) -> str:
    """Normalize real-eval provider labels to simulator profile keys."""
    provider = raw.strip()
    if "@" in provider:
        provider = provider.split("@", 1)[1]
    if provider.startswith("OR_"):
        provider = provider[3:]
    return provider


def is_pseudo_provider(provider: str) -> bool:
    """Return whether a provider label is an aggregate baseline profile."""
    return bool(PSEUDO_PROVIDER_RE.match(provider))


def collect_ttft_per_provider(
    csv_path: Path,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Parse an evaluation log into ``{provider: [ttft_ms, ...]}`` plus run stats.

    Row inclusion requires a known provider and ``ttft_ms > 0``. The
    ``status == "success"`` filter is applied only when the log has a
    ``status`` column. Timestamps are recorded for run-duration stats when
    present, but a missing or invalid timestamp never discards an otherwise
    valid TTFT sample.
    """
    ttft: dict[str, list[float]] = defaultdict(list)
    timestamps: list[float] = []
    total_rows = 0
    skipped_rows = 0
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            status = row.get("status")
            if status is not None and status != "success":
                skipped_rows += 1
                continue
            provider = row.get("actual_provider", "")
            if not provider or provider == "unknown":
                skipped_rows += 1
                continue
            try:
                value = float(row.get("ttft_ms", 0))
            except (TypeError, ValueError):
                skipped_rows += 1
                continue
            if value <= 0:
                skipped_rows += 1
                continue
            ttft[provider].append(value)
            raw_ts = row.get("timestamp")
            if raw_ts not in (None, ""):
                try:
                    ts = float(raw_ts)
                except (TypeError, ValueError):
                    ts = 0.0
                if ts > 0:
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


def collect_ttft_from_shared_profile_events(
    events_path: Path,
    *,
    allowed_sources: set[str] | None,
    include_pseudo: bool,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Parse shared profile events into ``{provider: [ttft_ms, ...]}``.

    Shared profile events include successful samples, transport failures, probe
    samples, and aggregate baseline labels. Only successful ``ttft_ms > 0``
    samples are used for the empirical distributions. Aggregate labels such as
    ``__or_sort_latency__`` are excluded unless explicitly requested.
    """
    ttft: dict[str, list[float]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    valid_source_counts: Counter[str] = Counter()
    excluded_pseudo: Counter[str] = Counter()
    error_counts: Counter[tuple[str, str]] = Counter()
    n_total = 0
    n_bad_json = 0
    n_missing_provider = 0
    n_missing_ttft = 0
    n_source_filtered = 0
    timestamps: list[float] = []

    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n_total += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                n_bad_json += 1
                continue

            source = str(event.get("source") or "")
            source_counts[source] += 1
            if allowed_sources is not None and source not in allowed_sources:
                n_source_filtered += 1
                continue

            provider = normalize_provider(str(event.get("provider") or ""))
            if not provider:
                n_missing_provider += 1
                continue
            if is_pseudo_provider(provider) and not include_pseudo:
                excluded_pseudo[provider] += 1
                continue

            error_type = event.get("error_type")
            if error_type:
                error_counts[(provider, str(error_type))] += 1
                continue

            try:
                value = float(event.get("ttft_ms") or 0.0)
            except (TypeError, ValueError):
                n_missing_ttft += 1
                continue
            if value <= 0.0:
                n_missing_ttft += 1
                continue

            ttft[provider].append(value)
            valid_source_counts[source] += 1
            try:
                ts = float(event.get("ts") or 0.0)
            except (TypeError, ValueError):
                ts = 0.0
            if ts > 0.0:
                timestamps.append(ts)

    stats: dict[str, Any] = {
        "total_events": n_total,
        "valid_ttft_events": sum(len(v) for v in ttft.values()),
        "bad_json_events": n_bad_json,
        "source_filtered_events": n_source_filtered,
        "missing_provider_events": n_missing_provider,
        "missing_or_nonpositive_ttft_events": n_missing_ttft,
        "source_counts": dict(sorted(source_counts.items())),
        "valid_source_counts": dict(sorted(valid_source_counts.items())),
        "excluded_pseudo_provider_counts": dict(sorted(excluded_pseudo.items())),
        "error_counts": {
            f"{provider}:{error_type}": count
            for (provider, error_type), count in sorted(error_counts.items())
        },
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


def load_inventory_provider_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load provider metadata from a real-eval inventory, keyed by profile name."""
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    metadata: dict[str, dict[str, Any]] = {}
    for provider in payload.get("providers", []):
        if not isinstance(provider, dict):
            continue
        raw_name = str(provider.get("name") or "")
        if not raw_name:
            continue
        key = normalize_provider(raw_name)
        entry = {
            field: provider[field]
            for field in (
                "name",
                "tier",
                "transport",
                "model",
                "provider_hint",
                "billing_mode",
                "quota_requests",
                "quota_window_sec",
                "concurrency_limit",
                "subscription_plan",
                "stream_cancel_billing",
                "notes",
            )
            if field in provider
        }
        input_price = provider.get("input_price_per_m")
        output_price = provider.get("output_price_per_m")
        cached_input = provider.get("cached_input_price_per_m")
        if input_price is not None:
            entry["input_price_per_m"] = float(input_price)
        if output_price is not None:
            entry["output_price_per_m"] = float(output_price)
        if cached_input is not None:
            entry["cached_input_price_per_m"] = float(cached_input)
        if (
            provider.get("transport") == "openrouter"
            and input_price is not None
            and output_price is not None
        ):
            entry["openrouter_price"] = {
                "provider_name": provider.get("provider_hint") or key,
                "input_price_per_m": float(input_price),
                "output_price_per_m": float(output_price),
                "input_cache_read_price_per_m": (
                    None if cached_input is None else float(cached_input)
                ),
                "source": "real_eval_inventory",
                "inventory_provider_name": raw_name,
            }
        metadata[key] = entry
    return metadata


def build_metadata(
    ttft: dict[str, np.ndarray],
    *,
    artifact_name: str,
    source: Path,
    model: str,
    model_family: str,
    tier: str,
    run_label: str | None,
    note: str | None,
    source_format: str,
    run_stats: dict[str, Any],
    prices: dict[str, dict[str, Any]],
    provider_metadata: dict[str, dict[str, Any]],
    price_source: str | None,
    endpoints_json: Path | None,
    inventory_json: Path | None,
    subsample_seed: int,
    max_samples: int,
    min_samples: int,
    include_pseudo_providers: bool,
    included_sources: set[str] | None,
) -> dict[str, Any]:
    """Build provenance and per-provider summary metadata."""
    providers_meta: dict[str, dict[str, Any]] = {}
    for provider, arr in sorted(ttft.items()):
        meta: dict[str, Any] = {
            "n_samples": int(arr.size),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "max_ms": float(np.max(arr)),
        }
        if provider in provider_metadata:
            meta.update(provider_metadata[provider])
        if provider in prices:
            meta["openrouter_price"] = prices[provider]
        providers_meta[provider] = meta

    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact": artifact_name,
        "source_log": str(source),
        "source_format": source_format,
        "model": model,
        "model_family": model_family,
        "tier": tier,
        "include_pseudo_providers": include_pseudo_providers,
        "included_sources": None if included_sources is None else sorted(included_sources),
    }
    if run_label:
        metadata["source_run_label"] = run_label
    if note:
        metadata["source_note"] = note
    metadata["subsample_seed"] = subsample_seed
    metadata["max_samples_per_provider"] = max_samples
    metadata["min_samples_per_provider"] = min_samples
    metadata["run_stats"] = run_stats
    if prices or provider_metadata or endpoints_json or inventory_json or price_source:
        metadata["price_snapshot"] = {
            "source": price_source,
            "captured_from": str(endpoints_json or inventory_json)
            if endpoints_json or inventory_json
            else None,
            "unit": "USD per 1M tokens",
            "note": "Provider prices can change; refresh before paper numbers.",
        }
    metadata["n_providers"] = len(providers_meta)
    metadata["providers"] = providers_meta
    return metadata


def _price_per_m(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) * 1_000_000.0


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier recorded in metadata, e.g. qwen3_235b or minimax_m25.",
    )
    parser.add_argument(
        "--model-family",
        default=None,
        help="Model family label. Defaults to --model when omitted.",
    )
    parser.add_argument(
        "--source-log",
        type=Path,
        required=True,
        help=(
            "Path to the source measurement log. Defaults to evaluation_log.csv; "
            "use --source-format shared_profile_events for real-eval JSONL events."
        ),
    )
    parser.add_argument(
        "--source-format",
        choices=("evaluation_log", "shared_profile_events"),
        default="evaluation_log",
        help="Input log format. Defaults to evaluation_log.",
    )
    parser.add_argument(
        "--out-npz",
        type=Path,
        required=True,
        help="Output .npz path (e.g. under experiments/simulation/latency_profiles/).",
    )
    parser.add_argument(
        "--out-meta",
        type=Path,
        default=None,
        help="Output .json metadata path. Defaults to --out-npz with a .json suffix.",
    )
    parser.add_argument(
        "--tier",
        default="S_A",
        help="Latency tier label recorded in metadata (default: S_A).",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional provenance label for the source measurement run.",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Optional free-text caveat recorded in metadata (e.g. partial coverage).",
    )
    parser.add_argument(
        "--endpoints-json",
        type=Path,
        default=None,
        help="Optional JSON captured from the OpenRouter endpoints API.",
    )
    parser.add_argument(
        "--price-source",
        default=None,
        help="Optional URL recorded as the price-snapshot source.",
    )
    parser.add_argument(
        "--inventory-json",
        type=Path,
        default=None,
        help=(
            "Optional real-eval inventory JSON. Useful with shared profile events "
            "to attach provider tier/transport/pricing metadata."
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help=(
            "For shared_profile_events input, include only this event source "
            "(e.g. natural, probe, warmup). Repeatable."
        ),
    )
    parser.add_argument(
        "--include-pseudo-providers",
        action="store_true",
        help="Keep aggregate labels such as __or_auto__ in shared-profile output.",
    )
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_PROVIDER)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_PER_PROVIDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_npz: Path = args.out_npz
    out_meta: Path = args.out_meta or out_npz.with_suffix(".json")
    model_family: str = args.model_family or args.model

    print(f"Reading {args.source_log}")
    included_sources = set(args.sources) if args.sources else None
    if args.source_format == "shared_profile_events":
        raw, run_stats = collect_ttft_from_shared_profile_events(
            args.source_log,
            allowed_sources=included_sources,
            include_pseudo=args.include_pseudo_providers,
        )
    else:
        raw, run_stats = collect_ttft_per_provider(args.source_log)
    print(
        f"  collected {len(raw)} providers, "
        f"{sum(len(v) for v in raw.values()):,} valid TTFT samples"
    )
    if "observed_duration_hours" in run_stats:
        print(f"  observed duration: {run_stats['observed_duration_hours']:.2f} hours")
    if run_stats.get("excluded_pseudo_provider_counts"):
        print(f"  excluded pseudo providers: {run_stats['excluded_pseudo_provider_counts']}")

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    sub = subsample_if_large(raw, cap=args.max_samples, rng=rng)
    sub = filter_too_few(sub, min_samples=args.min_samples)
    print(
        f"  after cap={args.max_samples:,} and min={args.min_samples:,}: "
        f"{len(sub)} providers, {sum(arr.size for arr in sub.values()):,} samples"
    )

    prices = load_openrouter_prices(args.endpoints_json)
    provider_metadata = load_inventory_provider_metadata(args.inventory_json)
    if args.endpoints_json:
        matched = len(set(sub) & set(prices))
        print(f"  loaded price snapshot for {len(prices)} providers ({matched} matched)")
    if args.inventory_json:
        matched = len(set(sub) & set(provider_metadata))
        print(
            f"  loaded inventory metadata for {len(provider_metadata)} providers "
            f"({matched} matched)"
        )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **sub)
    print(f"  wrote {out_npz} ({out_npz.stat().st_size / 1024:.1f} KB)")

    metadata = build_metadata(
        sub,
        artifact_name=out_npz.name,
        source=args.source_log,
        model=args.model,
        model_family=model_family,
        tier=args.tier,
        run_label=args.run_label,
        note=args.note,
        source_format=args.source_format,
        run_stats=run_stats,
        prices=prices,
        provider_metadata=provider_metadata,
        price_source=args.price_source,
        endpoints_json=args.endpoints_json,
        inventory_json=args.inventory_json,
        subsample_seed=SUBSAMPLE_SEED,
        max_samples=args.max_samples,
        min_samples=args.min_samples,
        include_pseudo_providers=args.include_pseudo_providers,
        included_sources=included_sources,
    )
    with out_meta.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  wrote {out_meta}")

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
