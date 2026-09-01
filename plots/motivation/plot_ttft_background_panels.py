"""Plot the two TTFT background panels as one single-column figure.

The paper places these panels side by side inside one ACM column. Generating a
combined figure keeps axis fonts at the final printed size instead of shrinking
two standalone column-width PDFs down to half width.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.motivation.plot_ttft_context_length import bucket_values, load_points
from plots.motivation.plot_ttft_duration_share import (
    DEFAULT_PROVIDER_SERIES,
    load_provider_series,
)

SIMULATOR_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = SIMULATOR_DIR.parent
DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "paper" / "figures"
DEFAULT_SOURCE_JSONL = (
    SIMULATOR_DIR
    / "data"
    / "motivation"
    / "ttft_duration"
    / "requests_minimax-m2.5_mixed-providers_20260301-20260515.jsonl"
)


def load_points_jsonl(source_jsonl: Path) -> pd.DataFrame:
    """Load the committed production export into the context-length schema."""
    records = []
    with source_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame.from_records(records)
    status = pd.to_numeric(df.get("status_code"), errors="coerce")
    df = df[status.isna() | (status < 400)]
    df = df.dropna(subset=["prompt_tokens", "ttft_ms"]).copy()
    df["prompt_tokens"] = df["prompt_tokens"].astype(int)
    df["ttft_ms"] = df["ttft_ms"].astype(float)
    df = df[(df["prompt_tokens"] > 0) & (df["ttft_ms"] > 0)]
    if df.empty:
        raise ValueError(f"{source_jsonl} has no positive prompt_tokens/ttft_ms rows")
    df["ttft_s"] = df["ttft_ms"] / 1000.0
    return df


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.8,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.alpha": 0.6,
            "grid.linewidth": 0.45,
        }
    )


def plot_ttft_share_panel(ax: plt.Axes, output_input_ratio: float) -> dict[str, Any]:
    df, summary, groups = load_provider_series(DEFAULT_PROVIDER_SERIES, output_input_ratio)
    colors = ["#6FA8DC", "#E69138", "#4CA259"]
    compact_labels = {
        "GPT-5.4": "GPT",
        "Claude Opus 4.7": "Claude",
        "MiniMax-M2.5": "MiniMax",
    }

    for idx, group in enumerate(groups):
        model_df = df.loc[df["series"] == group]
        shares = np.sort(model_df["ttft_share"].clip(lower=0, upper=1).to_numpy()) * 100.0
        cdf = np.arange(1, len(shares) + 1) / len(shares) * 100.0
        color = colors[idx % len(colors)]
        ax.step(
            shares,
            cdf,
            where="post",
            color=color,
            linewidth=1.25,
            label=compact_labels.get(group, group),
        )
        ax.scatter(
            [float(np.median(shares))],
            [50],
            s=8,
            color=color,
            edgecolors="white",
            linewidths=0.35,
            zorder=4,
        )

    ax.axhline(50, color="#888888", linewidth=0.8, linestyle="--", alpha=0.9)
    ax.text(4.0, 53.5, "50%", color="#475569", fontsize=7.0, va="bottom")
    ax.set_xlim(0, 101.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("TTFT share (%)", labelpad=0.8)
    ax.set_ylabel("CDF (%)", labelpad=0.8)
    ax.grid(True, which="major", linestyle=":", linewidth=0.45, alpha=0.6)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=3,
        frameon=False,
        handlelength=1.25,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    ax.text(
        0.5,
        1.06,
        "(a)",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.0,
        fontweight="bold",
        clip_on=False,
    )
    return summary


def plot_context_length_panel(ax: plt.Axes, source: Path) -> list[dict[str, float | int | str]]:
    df = load_points_jsonl(source) if source.suffix == ".jsonl" else load_points(source)
    values, _labels, summary = bucket_values(df)
    compact_labels = ["<=20", "20-40", "40-80", ">80"]

    bp = ax.boxplot(
        values,
        whis=(5, 95),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0f172a", "linewidth": 1.2},
        boxprops={"edgecolor": "#4CA259", "linewidth": 0.95},
        whiskerprops={"color": "#4CA259", "linewidth": 0.95},
        capprops={"color": "#4CA259", "linewidth": 0.95},
    )
    for box in bp["boxes"]:
        box.set_facecolor("#D9EAD3")
        box.set_alpha(0.85)

    p95 = [float(np.percentile(v, 95)) for v in values]
    ax.set_xticks(np.arange(1, len(compact_labels) + 1))
    ax.set_xticklabels(compact_labels)
    ax.set_xlabel("Prompt length (K)", labelpad=0.8)
    ax.set_ylabel("TTFT (s)", labelpad=0.8)
    ax.set_ylim(0, max(25.0, max(p95) * 1.12))
    ax.set_yticks([0, 10, 20])
    ax.grid(axis="y", linestyle=":", linewidth=0.45, alpha=0.6)
    ax.grid(axis="x", visible=False)
    ax.text(
        0.5,
        1.06,
        "(b)",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.0,
        fontweight="bold",
        clip_on=False,
    )
    return summary


def plot_combined(
    source: Path,
    output_dir: Path,
    basename: str,
    output_input_ratio: float,
) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_share, ax_context) = plt.subplots(
        1,
        2,
        figsize=(3.35, 1.58),
        gridspec_kw={"width_ratios": [0.98, 1.12], "wspace": 0.5},
    )
    share_summary = plot_ttft_share_panel(ax_share, output_input_ratio)
    context_summary = plot_context_length_panel(ax_context, source)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.26, top=0.83, wspace=0.52)

    for suffix in ("pdf", "png"):
        path = output_dir / f"{basename}.{suffix}"
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.close(fig)

    summary = {
        "ttft_share": share_summary,
        "ttft_context_length": context_summary,
    }
    summary_path = output_dir / f"{basename}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved: {summary_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_JSONL,
        help=(
            "Panel (b) source: the committed production JSONL export by default, "
            "or a CSV with prompt_tokens/ttft_ms/cache_hit columns."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default="figure03_ttft_background_panels")
    parser.add_argument("--output-input-ratio", type=float, default=0.01)
    args = parser.parse_args(argv)

    plot_combined(args.source, args.output_dir, args.basename, args.output_input_ratio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
