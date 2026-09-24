"""Prepare a TTFT latency profile from the 1h or8 effmean retry real-eval run.

Source: outputs/real_eval/real_eval_1h_gpu1_or8_effmean_baselines_rw75_*_or8_effmean_1h_retry/
which has 7 policies x 500 requests against an inventory of 8 OR providers
(Friendli, DeepInfra, Novita, Minimax, Phala, SiliconFlow, Nebius, WandB)
plus Chutes_SQ (subscription quota) and Featherless_SC (subscription concurrency).

Samples are pooled across all 7 policies because per-policy provider routing is
sparse (e.g. SiliconFlow received only a few requests under any single policy).
Pooling gives 47-1076 samples per provider.

Output:
    experiments/simulation/latency_profiles/minimax_m25_or8_1h_effmean.npz
    experiments/simulation/latency_profiles/minimax_m25_or8_1h_effmean.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPO_ROOT
    / "outputs/real_eval/"
    "real_eval_1h_gpu1_or8_effmean_baselines_rw75_20260512_014542_or8_effmean_1h_retry"
)
PROFILE_DIR = REPO_ROOT / "experiments" / "simulation" / "latency_profiles"
OUT_NPZ = PROFILE_DIR / "minimax_m25_or8_1h_effmean.npz"
OUT_META = PROFILE_DIR / "minimax_m25_or8_1h_effmean.json"

MIN_SAMPLES_PER_PROVIDER = 20
MAX_SAMPLES_PER_PROVIDER = 50_000
SUBSAMPLE_SEED = 42

PROVIDER_META: dict[str, dict[str, str]] = {
    "Friendli":       {"tier": "api", "transport": "openrouter"},
    "DeepInfra":      {"tier": "api", "transport": "openrouter"},
    "Novita":         {"tier": "api", "transport": "openrouter"},
    "Minimax":        {"tier": "api", "transport": "openrouter"},
    "Phala":          {"tier": "api", "transport": "openrouter"},
    "SiliconFlow":    {"tier": "api", "transport": "openrouter"},
    "Nebius":         {"tier": "api", "transport": "openrouter"},
    "WandB":          {"tier": "api", "transport": "openrouter"},
    "Chutes_SQ":      {"tier": "quota", "transport": "chutes"},
    "Featherless_SC": {"tier": "concurrency", "transport": "featherless"},
}


def normalize_provider(raw: str) -> str:
    """Strip the ``__policy__@`` prefix that OR providers carry in the CSV."""
    if "@" in raw:
        return raw.split("@", 1)[1]
    return raw


def collect_ttft(run_dir: Path) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Pool TTFT samples per provider across every policy under ``run_dir``."""
    ttft: dict[str, list[float]] = defaultdict(list)
    per_policy_rows: dict[str, int] = {}
    n_success = 0
    n_total = 0
    n_skipped = 0
    ts_min = float("inf")
    ts_max = float("-inf")
    for policy_dir in sorted(run_dir.iterdir()):
        csv_path = policy_dir / "requests.csv"
        if not csv_path.exists():
            continue
        rows_for_policy = 0
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                n_total += 1
                rows_for_policy += 1
                if row.get("status") != "success":
                    continue
                n_success += 1
                ttft_raw = row.get("ttft_ms") or ""
                ts_raw = row.get("ts") or ""
                try:
                    value = float(ttft_raw)
                    ts = float(ts_raw) if ts_raw else 0.0
                except ValueError:
                    n_skipped += 1
                    continue
                if value <= 0:
                    n_skipped += 1
                    continue
                provider = normalize_provider(row.get("actual_provider") or "")
                if not provider:
                    n_skipped += 1
                    continue
                ttft[provider].append(value)
                if ts > 0:
                    ts_min = min(ts_min, ts)
                    ts_max = max(ts_max, ts)
        per_policy_rows[policy_dir.name] = rows_for_policy

    stats: dict[str, Any] = {
        "policies": per_policy_rows,
        "total_rows": n_total,
        "success_rows": n_success,
        "skipped_rows": n_skipped,
    }
    if ts_min != float("inf"):
        stats["timestamp_min_unix"] = ts_min
        stats["timestamp_max_unix"] = ts_max
        stats["observed_duration_hours"] = (ts_max - ts_min) / 3600.0
    return dict(ttft), stats


def subsample_if_large(
    ttft: dict[str, list[float]], cap: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for provider, samples in ttft.items():
        arr = np.asarray(samples, dtype=np.float64)
        if arr.size > cap:
            idx = rng.choice(arr.size, size=cap, replace=False)
            idx.sort()
            arr = arr[idx]
        out[provider] = arr
    return out


def filter_too_few(
    ttft: dict[str, np.ndarray], min_samples: int
) -> dict[str, np.ndarray]:
    return {p: arr for p, arr in ttft.items() if arr.size >= min_samples}


def build_metadata(
    ttft: dict[str, np.ndarray],
    *,
    run_dir: Path,
    run_stats: dict[str, Any],
    subsample_seed: int,
    max_samples: int,
    min_samples: int,
) -> dict[str, Any]:
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
        # Annotate tier/transport when we know the provider; unknown providers
        # (none expected from this inventory) just get latency-only metadata.
        side = PROVIDER_META.get(provider)
        if side is not None:
            meta.update(side)
        providers_meta[provider] = meta

    source_rel = run_dir.relative_to(REPO_ROOT) if run_dir.is_absolute() else run_dir
    return {
        "schema_version": "1.0",
        "artifact": OUT_NPZ.name,
        "source_run_dir": str(source_rel),
        "source_run_label": run_dir.name,
        "model": "minimax_m25",
        "model_family": "minimax-m2.5",
        "description": (
            "TTFT samples pooled across 7 policies in a 1h real-eval retry run "
            "against 8 OR providers + Chutes_SQ + Featherless_SC."
        ),
        "subsample_seed": subsample_seed,
        "max_samples_per_provider": max_samples,
        "min_samples_per_provider": min_samples,
        "run_stats": run_stats,
        "n_providers": len(providers_meta),
        "providers": providers_meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-npz", type=Path, default=OUT_NPZ)
    parser.add_argument("--out-meta", type=Path, default=OUT_META)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_PROVIDER)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_PER_PROVIDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Reading {args.run_dir}")
    raw, run_stats = collect_ttft(args.run_dir)
    total_samples = sum(len(v) for v in raw.values())
    print(
        f"  collected {len(raw)} providers, {total_samples:,} valid TTFT samples"
    )

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    sub = subsample_if_large(raw, cap=args.max_samples, rng=rng)
    dropped = {p: arr.size for p, arr in sub.items() if arr.size < args.min_samples}
    sub = filter_too_few(sub, min_samples=args.min_samples)
    if dropped:
        print(f"  dropped {len(dropped)} providers under min={args.min_samples}: {dropped}")
    print(
        f"  after cap={args.max_samples:,} and min={args.min_samples:,}: "
        f"{len(sub)} providers, {sum(arr.size for arr in sub.values()):,} samples"
    )

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **sub)
    print(f"  wrote {args.out_npz} ({args.out_npz.stat().st_size / 1024:.1f} KB)")

    metadata = build_metadata(
        sub,
        run_dir=args.run_dir,
        run_stats=run_stats,
        subsample_seed=SUBSAMPLE_SEED,
        max_samples=args.max_samples,
        min_samples=args.min_samples,
    )
    with args.out_meta.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  wrote {args.out_meta}")

    print("\nPer-provider summary:")
    print(f"  {'provider':<16} {'tier':<12} {'n':>5}  {'P50':>6} {'P90':>6} {'P99':>6}  {'mean':>6}")
    for provider, meta in metadata["providers"].items():
        print(
            f"  {provider:<16} {meta.get('tier','?'):<12} {meta['n_samples']:>5}  "
            f"{meta['p50_ms']:>6.0f} {meta['p90_ms']:>6.0f} {meta['p99_ms']:>6.0f}  "
            f"{meta['mean_ms']:>6.0f}"
        )


if __name__ == "__main__":
    main()
