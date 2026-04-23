"""Rebuild drift_wall_clock figures from raw llmAPI_bench CSVs.

Reconstructs the two introduction figures `drift_wall_clock_llama` and
`drift_wall_clock_gpt4o` from the archived raw TTFT CSVs. The original
plotting script was never committed; this script reproduces the paper
figure style and is parameterised by rolling window size so that peaks
match the paper PNG.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class DriftSpec:
    csv_path: str
    title: str
    output_basename: str
    window_size: int
    ylim: tuple[float, float] | None = None


def load_raw(csv_path: str, input_len: int = 10) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"duration (ms)": "latency_ms"})
    df = df[df["input_len"] == input_len].reset_index(drop=True)
    df = df[["created_at", "latency_ms"]].dropna().reset_index(drop=True)
    df = df[df["latency_ms"] > 0].reset_index(drop=True)
    df = df.sort_values("created_at").reset_index(drop=True)
    t0 = df["created_at"].iloc[0]
    df["wall_clock_hours"] = (df["created_at"] - t0) / 3600.0
    return df


def rolling_percentile(values: np.ndarray, window: int, pct: float) -> np.ndarray:
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=window).quantile(pct / 100.0).to_numpy()


def plot_drift(spec: DriftSpec, output_dir: str) -> dict:
    df = load_raw(spec.csv_path)
    t_hours = df["wall_clock_hours"].to_numpy()
    lat = df["latency_ms"].to_numpy()

    roll_p50 = rolling_percentile(lat, spec.window_size, 50)
    roll_p99 = rolling_percentile(lat, spec.window_size, 99)
    global_p99 = float(np.percentile(lat, 99))

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(t_hours, roll_p50, color="#6FA8DC", linewidth=1.0, label="Rolling P50")
    ax.plot(t_hours, roll_p99, color="#4CA259", linewidth=1.2, label="Rolling P99")
    ax.axhline(
        global_p99,
        color="#888888",
        linestyle="--",
        linewidth=1.0,
        label=f"Global P99 ({global_p99:.0f}ms)",
    )
    ax.set_xlabel("Wall Clock Time (hours)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(spec.title)
    if spec.ylim is not None:
        ax.set_ylim(*spec.ylim)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    pdf_path = os.path.join(output_dir, f"{spec.output_basename}.pdf")
    png_path = os.path.join(output_dir, f"{spec.output_basename}.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "csv": spec.csv_path,
        "window": spec.window_size,
        "rows": len(df),
        "hours_span": float(t_hours[-1] - t_hours[0]),
        "global_p99_ms": global_p99,
        "max_rolling_p99_ms": float(np.nanmax(roll_p99)),
        "pdf": pdf_path,
        "png": png_path,
    }


def build_specs(source_dir: str, window_llama: int, window_gpt4o: int) -> Iterable[DriftSpec]:
    yield DriftSpec(
        csv_path=os.path.join(source_dir, "TTFT_prompt_len_Llama-3.3-70B-Instruct.csv"),
        title="(a) Meta: Llama-3.3-70B (input=10 tokens)",
        output_basename="drift_wall_clock_llama",
        window_size=window_llama,
    )
    yield DriftSpec(
        csv_path=os.path.join(source_dir, "TTFT_prompt_len_gpt-4o-mini.csv"),
        title="(b) OpenAI: gpt-4o-mini (input=10 tokens)",
        output_basename="drift_wall_clock_gpt4o",
        window_size=window_gpt4o,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default="/Users/realtmxi/Desktop/6991d665791ce21ba05287b8/vector_figures_review_20260421/source_audit",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-llama", type=int, default=100)
    parser.add_argument("--window-gpt4o", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for spec in build_specs(args.source_dir, args.window_llama, args.window_gpt4o):
        stats = plot_drift(spec, args.output_dir)
        print(stats)


if __name__ == "__main__":
    main()
