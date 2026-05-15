"""Plot production TTFT versus prompt length for the background section.

The source CSV is exported from the admin Provider -> Performance scatter plot.
It contains one row per production request with prompt_tokens, ttft_ms, and
cache_hit. The figure bins requests by prompt length and visualizes TTFT
distributions with box plots to show whether tail TTFT grows with context size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIMULATOR_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = SIMULATOR_DIR.parent
DEFAULT_SOURCE_CSV = Path.home() / "Downloads" / "ttft_glm-5.1_zai.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "paper" / "figures"

BIN_SPECS: tuple[tuple[int, int | None, str], ...] = (
    (0, 20_000, "<=20K"),
    (20_000, 40_000, "20-40K"),
    (40_000, 80_000, "40-80K"),
    (80_000, None, ">80K"),
)


def load_points(source_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(source_csv)
    required = {"prompt_tokens", "ttft_ms", "cache_hit"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{source_csv} is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["prompt_tokens", "ttft_ms"]).copy()
    df["prompt_tokens"] = df["prompt_tokens"].astype(int)
    df["ttft_ms"] = df["ttft_ms"].astype(float)
    df = df[(df["prompt_tokens"] > 0) & (df["ttft_ms"] > 0)]
    if df.empty:
        raise ValueError(f"{source_csv} has no positive prompt_tokens/ttft_ms rows")
    df["ttft_s"] = df["ttft_ms"] / 1000.0
    return df


def bucket_values(df: pd.DataFrame) -> tuple[list[np.ndarray], list[str], list[dict[str, float | int | str]]]:
    values: list[np.ndarray] = []
    labels: list[str] = []
    summary: list[dict[str, float | int | str]] = []

    for lower, upper, label in BIN_SPECS:
        mask = df["prompt_tokens"] > lower
        if upper is not None:
            mask &= df["prompt_tokens"] <= upper
        bucket = df.loc[mask, "ttft_s"].to_numpy()
        if bucket.size == 0:
            continue

        values.append(bucket)
        labels.append(f"{label}\nn={bucket.size}")
        summary.append(
            {
                "prompt_bucket": label,
                "n": int(bucket.size),
                "mean_s": float(np.mean(bucket)),
                "p50_s": float(np.percentile(bucket, 50)),
                "p90_s": float(np.percentile(bucket, 90)),
                "p95_s": float(np.percentile(bucket, 95)),
                "p99_s": float(np.percentile(bucket, 99)),
            }
        )

    return values, labels, summary


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
        }
    )


def plot_boxplot(values: list[np.ndarray], labels: list[str], output_dir: Path, basename: str) -> None:
    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    bp = ax.boxplot(
        values,
        whis=(5, 95),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0f172a", "linewidth": 1.5},
        boxprops={"edgecolor": "#0f766e", "linewidth": 1.0},
        whiskerprops={"color": "#0f766e", "linewidth": 1.0},
        capprops={"color": "#0f766e", "linewidth": 1.0},
    )
    for box in bp["boxes"]:
        box.set_facecolor("#ccfbf1")
        box.set_alpha(0.75)

    p95 = [float(np.percentile(v, 95)) for v in values]
    ax.plot(
        np.arange(1, len(values) + 1),
        p95,
        color="#dc2626",
        marker="D",
        markersize=3.6,
        linewidth=1.2,
        label="P95",
        zorder=3,
    )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("TTFT (s)")
    ax.set_ylim(0, max(25.0, max(p95) * 1.12))
    ax.grid(axis="y", linestyle="--")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    for suffix in ("pdf", "png"):
        path = output_dir / f"{basename}.{suffix}"
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default="figB01-ttft-context-length")
    args = parser.parse_args()

    df = load_points(args.source_csv)
    values, labels, summary = bucket_values(df)
    plot_boxplot(values, labels, args.output_dir, args.basename)

    summary_path = args.output_dir / f"{args.basename}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
