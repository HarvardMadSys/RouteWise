"""Prepare Qwen3 235B real-world TTFT samples for S_A experiments.

Reads cached OpenRouter evaluation log, extracts per-provider TTFT samples,
and writes compact profile artifacts for the EmpiricalDistribution class.

Run:
    python -m scripts.prepare_latency_profiles

Output:
    experiments/simulation/latency_profiles/qwen3_24h.npz       (per-provider TTFT arrays)
    experiments/simulation/latency_profiles/qwen3_24h.json      (metadata sidecar)
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOG = Path(
    os.environ.get(
        "ROUTEWISE_QWEN3_SOURCE_LOG",
        REPO_ROOT / "artifacts" / "private" / "qwen3_evaluation_log.csv",
    )
)
PROFILE_DIR = REPO_ROOT / "experiments" / "simulation" / "latency_profiles"
OUT_NPZ = PROFILE_DIR / "qwen3_24h.npz"
OUT_META = PROFILE_DIR / "qwen3_24h.json"

# Drop providers with too few samples to bootstrap robustly.
MIN_SAMPLES_PER_PROVIDER = 1_000

# Cap to keep .npz size reasonable. 50K samples is enough for stable
# bootstrap to within ~1% on P99.
MAX_SAMPLES_PER_PROVIDER = 50_000

# Subsample seed (for reproducibility of which 50K samples we kept).
SUBSAMPLE_SEED = 42


def collect_ttft_per_provider(csv_path: Path) -> dict[str, list[float]]:
    """Parse the evaluation log into {provider: [ttft_ms, ...]}."""
    ttft: dict[str, list[float]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            provider = row.get("actual_provider", "")
            if not provider or provider == "unknown":
                continue
            try:
                value = float(row.get("ttft_ms", 0))
            except ValueError:
                continue
            if value <= 0:
                continue
            ttft[provider].append(value)
    return dict(ttft)


def subsample_if_large(
    ttft: dict[str, list[float]],
    cap: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Cap each provider's samples at `cap` via uniform random subsample."""
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
    """Drop providers with fewer than min_samples (cannot bootstrap safely)."""
    return {p: arr for p, arr in ttft.items() if arr.size >= min_samples}


def build_metadata(
    ttft: dict[str, np.ndarray],
    *,
    source: Path,
    subsample_seed: int,
    max_samples: int,
) -> dict:
    """Per-provider summary stats + provenance."""
    providers_meta = {}
    for provider, arr in sorted(ttft.items()):
        providers_meta[provider] = {
            "n_samples": int(arr.size),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "max_ms": float(np.max(arr)),
        }
    return {
        "schema_version": "1.0",
        "source_log": source.name,
        "source_note": "Full source path omitted to keep metadata environment-independent.",
        "model": "qwen3_235b",
        "tier": "S_A",
        "subsample_seed": subsample_seed,
        "max_samples_per_provider": max_samples,
        "n_providers": len(providers_meta),
        "providers": providers_meta,
    }


def main() -> None:
    print(f"Reading {SOURCE_LOG}")
    raw = collect_ttft_per_provider(SOURCE_LOG)
    print(f"  collected {len(raw)} providers, {sum(len(v) for v in raw.values())} total samples")

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    sub = subsample_if_large(raw, cap=MAX_SAMPLES_PER_PROVIDER, rng=rng)
    sub = filter_too_few(sub, min_samples=MIN_SAMPLES_PER_PROVIDER)
    print(
        f"  after subsample to {MAX_SAMPLES_PER_PROVIDER:,} cap and filter to "
        f"≥{MIN_SAMPLES_PER_PROVIDER:,}: {len(sub)} providers, "
        f"{sum(arr.size for arr in sub.values())} total samples"
    )

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **sub)
    print(f"  wrote {OUT_NPZ} ({OUT_NPZ.stat().st_size / 1024:.1f} KB)")

    metadata = build_metadata(
        sub,
        source=SOURCE_LOG,
        subsample_seed=SUBSAMPLE_SEED,
        max_samples=MAX_SAMPLES_PER_PROVIDER,
    )
    with OUT_META.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  wrote {OUT_META}")

    print("\nPer-provider summary:")
    print(f"  {'provider':>14} {'n':>7} {'P50':>7} {'P90':>7} {'P99':>7} {'max':>9}")
    for provider, meta in metadata["providers"].items():
        print(
            f"  {provider:>14} {meta['n_samples']:>7,} "
            f"{meta['p50_ms']:>7.0f} {meta['p90_ms']:>7.0f} "
            f"{meta['p99_ms']:>7.0f} {meta['max_ms']:>9.0f}"
        )


if __name__ == "__main__":
    main()
