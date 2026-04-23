#!/usr/bin/env python3
r"""Plot provider-level TTFT ECDFs from a Phase 5 evaluation log.

Usage:
    source .venv/bin/activate
    python -m experiment.scripts.plot.latency.provider_ttft_ecdf \
        --input-dir experiment/results/phase5_qwen3_7d_clean/run_20260410_171624/snapshots/snap_20260411_171625 \
        --output-dir experiment/results/phase5_qwen3_7d_clean/run_20260410_171624/analysis
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment.scripts.plot.common import (
    PROVIDER_COLORS,
    apply_style,
    compute_ecdf,
    save_figure,
)

apply_style("paper")

# Extended provider colors for current OpenRouter providers.
QWEN_PROVIDER_COLORS = {
    **PROVIDER_COLORS,
    "WandB": "#3498db",
    "Alibaba": "#e74c3c",
    "SiliconFlow": "#f39c12",
    "DeepInfra": "#2ecc71",
    "Chutes": "#9b59b6",
    "Google": "#1abc9c",
    "Novita": "#e67e22",
    "AtlasCloud": "#34495e",
    "Together": "#d35400",
    "Cerebras": "#27ae60",
    "Friendli": "#8e44ad",
    "Parasail": "#16a085",
    "unknown": "#7f8c8d",
}


def _load_provider_ttft(
    input_dir: Path,
    min_requests: int,
) -> dict[str, np.ndarray]:
    """Loads successful TTFT samples grouped by actual provider."""
    provider_ttft: dict[str, list[float]] = defaultdict(list)
    log_path = input_dir / "evaluation_log.csv"

    with log_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["status"] != "success":
                continue
            if row["policy"].startswith("_"):
                continue
            provider = row["actual_provider"].strip()
            if not provider:
                continue
            try:
                ttft_ms = float(row["ttft_ms"])
            except ValueError:
                continue
            provider_ttft[provider].append(ttft_ms)

    filtered = {
        provider: np.array(values, dtype=float)
        for provider, values in provider_ttft.items()
        if len(values) >= min_requests
    }
    return filtered


def _write_summary(
    provider_ttft: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Writes summary quantiles for each provider."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "provider_ttft_summary.csv"
    rows = []
    for provider, values in sorted(
        provider_ttft.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        rows.append(
            {
                "provider": provider,
                "count": len(values),
                "p50_ttft_ms": np.percentile(values, 50),
                "p90_ttft_ms": np.percentile(values, 90),
                "p99_ttft_ms": np.percentile(values, 99),
                "mean_ttft_ms": values.mean(),
            }
        )

    with summary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "provider",
                "count",
                "p50_ttft_ms",
                "p90_ttft_ms",
                "p99_ttft_ms",
                "mean_ttft_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {summary_path}")


def plot_provider_ttft_ecdf(
    provider_ttft: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Plots provider-level TTFT ECDFs."""
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    providers_sorted = sorted(
        provider_ttft.items(),
        key=lambda item: (np.percentile(item[1], 50), item[0]),
    )

    for provider, values in providers_sorted:
        x, y = compute_ecdf(values)
        color = QWEN_PROVIDER_COLORS.get(provider, "#7f8c8d")
        p50 = np.percentile(values, 50)
        p99 = np.percentile(values, 99)
        ax.plot(
            x,
            y,
            label=f"{provider} (n={len(values):,}, p50={p50:.0f}ms, p99={p99:.0f}ms)",
            color=color,
            alpha=0.95,
            linewidth=2.0,
        )

    ax.set_xscale("log")
    ax.set_xlabel("TTFT (ms, log scale)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("TTFT ECDF by Provider (successful non-probing requests)")
    ax.set_xlim(left=100)
    ax.set_ylim(0.0, 1.0)
    ax.legend(
        loc="lower right",
        fontsize=9,
        framealpha=0.95,
        ncol=1,
    )

    fig.tight_layout()
    save_figure(fig, output_dir, "provider_ttft_ecdf", formats=["png", "pdf"])
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing evaluation_log.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write plots and summary CSV",
    )
    parser.add_argument(
        "--min-requests",
        type=int,
        default=500,
        help="Minimum successful requests required to include a provider",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the provider TTFT plotting pipeline."""
    args = parse_args()
    provider_ttft = _load_provider_ttft(args.input_dir, args.min_requests)
    if not provider_ttft:
        raise ValueError("No providers satisfied the minimum request threshold.")

    _write_summary(provider_ttft, args.output_dir)
    plot_provider_ttft_ecdf(provider_ttft, args.output_dir)


if __name__ == "__main__":
    main()
