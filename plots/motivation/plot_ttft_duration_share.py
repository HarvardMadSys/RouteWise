"""Plot TTFT share of end-to-end duration for agentic production requests.

The source JSONL files are sanitized exports from the admin User Requests table.
We filter to successful streaming-style rows with positive TTFT and end-to-end
latency, then keep agentic-like requests whose output length is at most 1% of
input length. This matches the production workload shape discussed in the paper:
long context and short output. The default paper plot is a provider-filtered CDF
of TTFT's share of total duration; exploratory scatter and boxplot views are also
available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIMULATOR_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = SIMULATOR_DIR.parent
DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "paper" / "figures"
DATA_DIR = SIMULATOR_DIR / "data" / "motivation" / "ttft_duration"
DEFAULT_SOURCE_JSONL = DATA_DIR / "requests_minimax-m2.5_mixed-providers_20260301-20260515.jsonl"
DEFAULT_MODELS = ["minimax-m2.5"]


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    source_jsonl: Path
    model_id: str
    provider: str
    ttft_cap_ms: int | None = None


DEFAULT_PROVIDER_SERIES = (
    SeriesSpec(
        label="GPT-5.4",
        source_jsonl=DATA_DIR / "requests_gpt-5.4_openai_20260301-20260515.jsonl",
        model_id="gpt-5.4",
        provider="openai",
    ),
    SeriesSpec(
        label="Claude Opus 4.7",
        source_jsonl=DATA_DIR / "requests_claude-opus-4.7_anthropic_20260301-20260515.jsonl",
        model_id="claude-opus-4.7",
        provider="anthropic",
        ttft_cap_ms=20_000,
    ),
    SeriesSpec(
        label="MiniMax-M2.5",
        source_jsonl=DATA_DIR / "requests_minimax-m2.5_mixed-providers_20260301-20260515.jsonl",
        model_id="minimax-m2.5",
        provider="minimax",
    ),
)


def iter_records(source_jsonl: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with source_jsonl.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_agentic_requests(source_jsonl: Path, output_input_ratio: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = iter_records(source_jsonl)
    df = pd.DataFrame.from_records(rows)
    required = {"model_id", "status_code", "ttft_ms", "latency_ms", "prompt_tokens", "completion_tokens"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{source_jsonl} is missing columns: {sorted(missing)}")

    for col in required.difference({"model_id"}):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_mask = (
        df["status_code"].between(200, 399)
        & (df["ttft_ms"] > 0)
        & (df["latency_ms"] >= df["ttft_ms"])
        & (df["prompt_tokens"] > 0)
        & (df["completion_tokens"] > 0)
    )
    valid = df.loc[valid_mask].copy()
    agentic_mask = valid["completion_tokens"] <= output_input_ratio * valid["prompt_tokens"]
    agentic = valid.loc[agentic_mask].copy()
    if agentic.empty:
        raise ValueError("No rows remain after applying the agentic workload filter")

    agentic["ttft_s"] = agentic["ttft_ms"] / 1000.0
    agentic["duration_s"] = agentic["latency_ms"] / 1000.0
    agentic["ttft_share"] = agentic["ttft_ms"] / agentic["latency_ms"]

    summary = {
        "source": str(source_jsonl),
        "rows": int(len(df)),
        "valid_rows": int(len(valid)),
        "agentic_rows": int(len(agentic)),
        "output_input_ratio_filter": output_input_ratio,
        "ttft_share_mean": float(agentic["ttft_share"].mean()),
        "ttft_share_p50": float(agentic["ttft_share"].quantile(0.50)),
        "ttft_share_p90": float(agentic["ttft_share"].quantile(0.90)),
        "ttft_share_p95": float(agentic["ttft_share"].quantile(0.95)),
        "share_ttft_at_least_50pct": float((agentic["ttft_share"] >= 0.50).mean()),
        "share_ttft_at_least_80pct": float((agentic["ttft_share"] >= 0.80).mean()),
        "share_ttft_at_least_95pct": float((agentic["ttft_share"] >= 0.95).mean()),
        "prompt_tokens_p50": float(agentic["prompt_tokens"].quantile(0.50)),
        "completion_tokens_p50": float(agentic["completion_tokens"].quantile(0.50)),
    }
    return agentic, summary


def _prepare_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model_id",
        "provider",
        "status_code",
        "ttft_ms",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    for col in required.difference({"model_id", "provider"}):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.loc[
        df["status_code"].between(200, 399)
        & (df["ttft_ms"] > 0)
        & (df["latency_ms"] >= df["ttft_ms"])
        & (df["prompt_tokens"] > 0)
        & (df["completion_tokens"] > 0)
    ].copy()


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ttft_s"] = df["ttft_ms"] / 1000.0
    df["duration_s"] = df["latency_ms"] / 1000.0
    df["ttft_share"] = df["ttft_ms"] / df["latency_ms"]
    return df


def summarize_frame(df: pd.DataFrame) -> dict[str, Any]:
    share = df["ttft_share"]
    return {
        "rows": int(len(df)),
        "ttft_share_p50": float(share.quantile(0.50)),
        "ttft_share_p90": float(share.quantile(0.90)),
        "ttft_share_p95": float(share.quantile(0.95)),
        "share_ttft_at_least_50pct": float((share >= 0.50).mean()),
        "share_ttft_at_least_80pct": float((share >= 0.80).mean()),
        "share_ttft_at_least_95pct": float((share >= 0.95).mean()),
        "ttft_s_p50": float(df["ttft_s"].quantile(0.50)),
        "ttft_s_p90": float(df["ttft_s"].quantile(0.90)),
        "duration_s_p50": float(df["duration_s"].quantile(0.50)),
        "duration_s_p90": float(df["duration_s"].quantile(0.90)),
        "prompt_tokens_p50": float(df["prompt_tokens"].quantile(0.50)),
        "completion_tokens_p50": float(df["completion_tokens"].quantile(0.50)),
    }


def load_provider_series(
    specs: tuple[SeriesSpec, ...], output_input_ratio: float
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    frames: list[pd.DataFrame] = []
    series_summary: dict[str, Any] = {}

    for spec in specs:
        rows = iter_records(spec.source_jsonl)
        raw = pd.DataFrame.from_records(rows)
        valid = _prepare_valid_rows(raw)
        scoped = valid.loc[
            (valid["model_id"] == spec.model_id) & (valid["provider"] == spec.provider)
        ].copy()
        agentic = scoped.loc[
            scoped["completion_tokens"] <= output_input_ratio * scoped["prompt_tokens"]
        ].copy()
        before_cap = len(agentic)
        if spec.ttft_cap_ms is not None:
            agentic = agentic.loc[agentic["ttft_ms"] <= spec.ttft_cap_ms].copy()
        if agentic.empty:
            raise ValueError(f"No rows remain for {spec.label}")

        agentic = _add_derived_columns(agentic)
        agentic["series"] = spec.label
        frames.append(agentic)
        series_summary[spec.label] = {
            "source": str(spec.source_jsonl),
            "model_id": spec.model_id,
            "provider": spec.provider,
            "raw_rows": int(len(raw)),
            "valid_rows": int(len(valid)),
            "scoped_valid_rows": int(len(scoped)),
            "agentic_rows_before_ttft_cap": int(before_cap),
            "ttft_cap_ms": spec.ttft_cap_ms,
            **summarize_frame(agentic),
        }

    combined = pd.concat(frames, ignore_index=True)
    labels = [spec.label for spec in specs]
    summary = {
        "series_mode": "provider_filtered",
        "output_input_ratio_filter": output_input_ratio,
        "agentic_rows": int(len(combined)),
        "series": series_summary,
        **summarize_frame(combined),
    }
    return combined, summary, labels


def select_models(df: pd.DataFrame, requested_models: list[str] | None, max_models: int) -> list[str]:
    if requested_models:
        missing = sorted(set(requested_models).difference(df["model_id"].unique()))
        if missing:
            raise ValueError(f"Requested models not found after filtering: {missing}")
        return requested_models
    default_models = [model for model in DEFAULT_MODELS if model in set(df["model_id"])]
    if default_models:
        return default_models[:max_models]
    return df["model_id"].value_counts().head(max_models).index.tolist()


def summarize_models(df: pd.DataFrame, models: list[str]) -> dict[str, dict[str, float | int]]:
    model_summary: dict[str, dict[str, float | int]] = {}
    for model in models:
        model_df = df.loc[df["model_id"] == model]
        share = model_df["ttft_share"]
        model_summary[model] = {
            "rows": int(len(model_df)),
            "ttft_share_p50": float(share.quantile(0.50)),
            "ttft_share_p90": float(share.quantile(0.90)),
            "share_ttft_at_least_80pct": float((share >= 0.80).mean()),
            "prompt_tokens_p50": float(model_df["prompt_tokens"].quantile(0.50)),
            "completion_tokens_p50": float(model_df["completion_tokens"].quantile(0.50)),
        }
    return model_summary


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.1,
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


def save_figure(fig: plt.Figure, output_dir: Path, basename: str) -> None:
    for suffix in ("pdf", "png"):
        path = output_dir / f"{basename}.{suffix}"
        fig.savefig(path)
        print(f"Saved: {path}")
    plt.close(fig)


def plot_ttft_share_cdf(
    df: pd.DataFrame, summary: dict[str, Any], output_dir: Path, basename: str, groups: list[str]
) -> None:
    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    colors = ["#0072B2", "#D55E00", "#009E73"]
    group_col = "series" if "series" in df.columns else "model_id"
    for idx, group in enumerate(groups):
        model_df = df.loc[df[group_col] == group]
        shares = np.sort(model_df["ttft_share"].clip(lower=0, upper=1).to_numpy()) * 100.0
        cdf = np.arange(1, len(shares) + 1) / len(shares) * 100.0
        color = colors[idx % len(colors)]
        ax.step(
            shares,
            cdf,
            where="post",
            color=color,
            linewidth=1.35,
            label=group,
        )
        median_share = float(np.median(shares))
        ax.scatter(
            [median_share],
            [50],
            s=13,
            color=color,
            edgecolors="white",
            linewidths=0.45,
            zorder=4,
        )

    ax.axhline(50, color="#64748b", linewidth=0.8, linestyle="--", alpha=0.75)
    ax.text(2.0, 52.0, "50% request fraction", color="#475569", fontsize=6.2, va="bottom")

    ax.set_xlim(0, 101.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("TTFT share of duration (%)", labelpad=1.5)
    ax.set_ylabel("CDF of requests (%)", labelpad=1.5)
    ax.grid(True, which="major", linestyle="--")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(groups),
        frameon=False,
        fontsize=6.7,
        handlelength=1.4,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.35)
    save_figure(fig, output_dir, basename)


def plot_ttft_share_boxplot(
    df: pd.DataFrame, summary: dict[str, Any], output_dir: Path, basename: str, groups: list[str]
) -> None:
    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#0072B2", "#D55E00", "#009E73"]
    group_col = "series" if "series" in df.columns else "model_id"
    values = [
        df.loc[df[group_col] == group, "ttft_share"].clip(lower=0, upper=1).to_numpy() * 100.0
        for group in groups
    ]
    labels = groups

    fig, ax = plt.subplots(figsize=(3.35, 1.72))
    parts = ax.boxplot(
        values,
        vert=False,
        tick_labels=labels,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        widths=0.52,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "#64748b", "linewidth": 0.8},
        capprops={"color": "#64748b", "linewidth": 0.8},
    )
    for patch, color in zip(parts["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)

    ax.set_xlim(0, 101.5)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("TTFT share of duration (%)", labelpad=1.5)
    ax.set_ylabel("")
    ax.grid(True, axis="x", which="major", linestyle="--")
    ax.grid(False, axis="y")
    fig.tight_layout(pad=0.35)
    save_figure(fig, output_dir, basename)


def plot_ttft_duration_scatter(df: pd.DataFrame, summary: dict[str, Any], output_dir: Path, basename: str) -> None:
    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.scatter(
        df["duration_s"],
        df["ttft_s"],
        s=4,
        color="#14b8a6",
        alpha=0.12,
        linewidths=0,
        rasterized=True,
    )

    low = max(0.05, float(min(df["duration_s"].min(), df["ttft_s"].min())) * 0.8)
    high = float(max(df["duration_s"].max(), df["ttft_s"].max())) * 1.15
    ax.plot([low, high], [low, high], color="#dc2626", linewidth=1.3, label="TTFT = duration")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("End-to-end duration (s)")
    ax.set_ylabel("TTFT (s)")
    ax.grid(True, which="major", linestyle="--")
    ax.grid(True, which="minor", linestyle=":", alpha=0.12)

    note = (
        f"n={summary['agentic_rows']:,}\n"
        f"median share={summary['ttft_share_p50'] * 100:.1f}%\n"
        f">=80% share={summary['share_ttft_at_least_80pct'] * 100:.1f}%"
    )
    ax.text(
        0.04,
        0.96,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, basename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-input-ratio", type=float, default=0.01)
    parser.add_argument("--basename", default="figB02-ttft-duration-share")
    parser.add_argument("--plot", choices=("cdf", "boxplot", "scatter"), default="cdf")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max-models", type=int, default=4)
    parser.add_argument(
        "--single-source",
        action="store_true",
        help="Use --source-jsonl and select models from one file instead of the provider-filtered paper series.",
    )
    args = parser.parse_args()

    if args.single_source:
        df, summary = load_agentic_requests(args.source_jsonl, args.output_input_ratio)
        groups = select_models(df, args.models, args.max_models)
        summary["selected_models"] = summarize_models(df, groups)
    else:
        df, summary, groups = load_provider_series(DEFAULT_PROVIDER_SERIES, args.output_input_ratio)

    if args.plot == "cdf":
        plot_ttft_share_cdf(df, summary, args.output_dir, args.basename, groups)
    elif args.plot == "boxplot":
        plot_ttft_share_boxplot(df, summary, args.output_dir, args.basename, groups)
    else:
        plot_ttft_duration_scatter(df, summary, args.output_dir, args.basename)

    summary_path = args.output_dir / f"{args.basename}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
