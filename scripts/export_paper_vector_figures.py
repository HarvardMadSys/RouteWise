#!/usr/bin/env python3
"""Export review-ready vector figures for the ICML paper.

This script does not modify the paper source tree. It creates a separate
review folder with PDF figures and a manifest that records provenance for each
figure.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_PAPER_DIR = Path("/Users/realtmxi/Desktop/6991d665791ce21ba05287b8")
INTERNAL_PAPER_DIR = PROJECT_ROOT / "6991d665791ce21ba05287b8"
DEFAULT_PAPER_DIR = EXTERNAL_PAPER_DIR if EXTERNAL_PAPER_DIR.exists() else INTERNAL_PAPER_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_PAPER_DIR / "vector_figures_review_20260421"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "blue": "#2B6CB0",
    "blue_light": "#CFE2F3",
    "green": "#2F855A",
    "green_light": "#C6F6D5",
    "orange": "#DD6B20",
    "orange_light": "#FEEBC8",
    "red": "#C53030",
    "red_light": "#FED7D7",
    "purple": "#6B46C1",
    "gray": "#4A5568",
    "gray_light": "#EDF2F7",
    "yellow": "#D69E2E",
}

LLMAPI_BENCH_REMOTE_HOST = "freeinference-direct"
LLMAPI_BENCH_REMOTE_GIT_DIR = "/home/murphy/test/hybridInference/.git/modules/llmAPI_bench"


@dataclass
class ManifestEntry:
    """Manifest metadata for one paper figure."""

    name: str
    paper_asset: str
    status: str
    output: str | None
    method: str
    inputs: list[str]
    notes: str


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _save_pdf(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _compile_standalone_tex(src_tex: Path, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vector_fig_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        temp_tex = temp_dir / src_tex.name
        shutil.copy2(src_tex, temp_tex)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", temp_tex.name],
            cwd=temp_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        built_pdf = temp_dir / f"{src_tex.stem}.pdf"
        shutil.copy2(built_pdf, output_pdf)


def _fetch_remote_git_file(remote_host: str, git_dir: str, object_path: str) -> str:
    """Fetch one tracked file from a remote git object store."""
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            remote_host,
            f"git --git-dir={git_dir} show HEAD:{object_path}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _export_remote_source_copy(raw_text: str, output_path: Path) -> None:
    """Write an audit copy of fetched source data into the review folder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(raw_text)


def _prepare_llmapi_bench_drift_df(raw_csv_text: str) -> pd.DataFrame:
    """Load long-horizon TTFT measurements and keep the short-prompt trace."""
    df = pd.read_csv(io.StringIO(raw_csv_text), skipinitialspace=True)
    df = df[df["input_len"] == 10].copy()
    df["created_at"] = pd.to_numeric(df["created_at"])
    df["duration_ms"] = pd.to_numeric(df["duration (ms)"])
    df = df.sort_values("created_at").reset_index(drop=True)
    df["elapsed_h"] = (df["created_at"] - df["created_at"].min()) / 3600.0
    return df


def _plot_llmapi_bench_drift(
    raw_csv_text: str,
    output_pdf: Path,
    title: str,
    rolling_window_points: int,
    rolling_min_periods: int = 1,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """Rebuild the paper's wall-clock drift figure from llmAPI_bench raw CSV."""
    df = _prepare_llmapi_bench_drift_df(raw_csv_text)
    rolling = df["duration_ms"].rolling(
        window=rolling_window_points,
        min_periods=rolling_min_periods,
    )
    rolling_p50 = rolling.quantile(0.50)
    rolling_p99 = rolling.quantile(0.99)
    global_p99 = float(df["duration_ms"].quantile(0.99))

    fig, ax = plt.subplots(figsize=(10.24, 6.83))
    ax.plot(
        df["elapsed_h"],
        rolling_p50,
        color="#6AA0C8",
        linewidth=2.2,
        label="Rolling P50",
    )
    ax.plot(
        df["elapsed_h"],
        rolling_p99,
        color="#2CA02C",
        linewidth=2.6,
        label="Rolling P99",
    )
    ax.axhline(
        global_p99,
        color="#7F7F7F",
        linestyle="--",
        linewidth=2.0,
        label=f"Global P99 ({int(round(global_p99))}ms)",
    )
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_xlabel("Wall Clock Time (hours)", fontsize=32)
    ax.set_ylabel("Latency (ms)", fontsize=34)
    ax.set_title(title, fontsize=44, pad=18)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=18)
    ax.spines["left"].set_linewidth(2.8)
    ax.spines["bottom"].set_linewidth(2.8)
    ax.tick_params(axis="both", labelsize=20, width=1.4, length=8)
    _save_pdf(fig, output_pdf)


def _export_llmapi_bench_drift_figure(
    remote_object_path: str,
    audit_copy_path: Path,
    output_pdf: Path,
    title: str,
    rolling_window_points: int,
    rolling_min_periods: int = 1,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """Fetch remote raw data, keep an audit copy, and export the vector PDF."""
    raw_csv_text = _fetch_remote_git_file(
        LLMAPI_BENCH_REMOTE_HOST,
        LLMAPI_BENCH_REMOTE_GIT_DIR,
        remote_object_path,
    )
    _export_remote_source_copy(raw_csv_text, audit_copy_path)
    _plot_llmapi_bench_drift(
        raw_csv_text=raw_csv_text,
        output_pdf=output_pdf,
        title=title,
        rolling_window_points=rolling_window_points,
        rolling_min_periods=rolling_min_periods,
        y_limits=y_limits,
    )


def _plot_sensitivity(results_path: Path, dataset_label: str, output_pdf: Path) -> None:
    data = _load_json(results_path)["fig2_parameter_sensitivity"]
    q_values = data["q_values"]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.axhline(
        y=data["all_api"][0],
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.8,
        label="All-API",
        alpha=0.9,
    )
    ax.plot(
        q_values,
        data["greedy"],
        marker="s",
        color=COLORS["blue"],
        linewidth=2.2,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="Greedy",
    )
    ax.plot(
        q_values,
        data["optimal"],
        marker="o",
        color=COLORS["orange"],
        linewidth=2.2,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="Optimal",
    )
    ax.fill_between(
        q_values,
        data["greedy"],
        data["optimal"],
        color=COLORS["red_light"],
        alpha=0.75,
        label="Opportunity cost",
    )
    ax.set_xlabel("Daily quota (Q)")
    ax.set_ylabel("API cost ($)")
    ax.set_title(dataset_label)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    _save_pdf(fig, output_pdf)


def _plot_online_calibration(results_path: Path, output_pdf: Path) -> None:
    data = _load_json(results_path)
    stats = data["LA-PD-P10"]["strategy_stats"]
    hist_cal = stats["predictor_calibration"]
    ema_cal = stats["ema_calibration"]

    quantiles = ["q10", "q50", "q90"]
    ideal = [0.10, 0.50, 0.90]
    hist_cov = [hist_cal[f"{quantile}_coverage"] for quantile in quantiles]
    ema_cov = [ema_cal[f"{quantile}_coverage"] for quantile in quantiles]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    x = np.arange(len(quantiles))
    width = 0.24

    bars_ideal = ax.bar(
        x - width,
        ideal,
        width,
        label="Ideal",
        color="#E2E8F0",
        edgecolor=COLORS["gray"],
        linewidth=1.3,
        hatch="///",
    )
    bars_hist = ax.bar(
        x,
        hist_cov,
        width,
        label="Histogram",
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=1.0,
    )
    bars_ema = ax.bar(
        x + width,
        ema_cov,
        width,
        label="EMA",
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=1.0,
    )

    for bars, values in ((bars_ideal, ideal), (bars_hist, hist_cov), (bars_ema, ema_cov)):
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel(r"Coverage $P(\mathrm{actual} \leq \mathrm{predicted})$")
    ax.set_xticks(x)
    ax.set_xticklabels(["P10", "P50", "P90"])
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95)
    _save_pdf(fig, output_pdf)


def _plot_single_model_comparison(results_path: Path, output_pdf: Path) -> None:
    data = _load_json(results_path)
    models = ["llama-4-scout", "qwen3-coder-30b", "llama-3.3-70b-instruct"]
    labels = {
        "llama-4-scout": "Small\n(mult=1)",
        "qwen3-coder-30b": "Medium\n(mult=2)",
        "llama-3.3-70b-instruct": "Large\n(mult=4)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    x = np.arange(len(models))
    width = 0.25

    quota_costs = [data[model]["results"]["daily_quota_only"]["costs"]["api"] for model in models]
    conc_costs = [data[model]["results"]["concurrency_only"]["costs"]["api"] for model in models]
    ilp_costs = [data[model]["results"]["ilp_optimal"]["costs"]["api"] for model in models]
    axes[0].bar(x - width, quota_costs, width, label="Quota-Only", color=COLORS["blue"])
    axes[0].bar(x, conc_costs, width, label="Conc.-Only", color=COLORS["green"])
    axes[0].bar(x + width, ilp_costs, width, label="ILP-Optimal", color=COLORS["orange"])
    axes[0].set_ylabel("API cost ($)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[model] for model in models])
    axes[0].set_title("(a) API cost")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper left", framealpha=0.95)

    quota_reqs = [
        data[model]["results"]["daily_quota_only"]["requests"]["subscription"] for model in models
    ]
    conc_reqs = [
        data[model]["results"]["concurrency_only"]["requests"]["subscription"] for model in models
    ]
    ilp_reqs = [data[model]["results"]["ilp_optimal"]["requests"]["subscription"] for model in models]
    axes[1].bar(x - width, quota_reqs, width, label="Quota-Only", color=COLORS["blue"])
    axes[1].bar(x, conc_reqs, width, label="Conc.-Only", color=COLORS["green"])
    axes[1].bar(x + width, ilp_reqs, width, label="ILP-Optimal", color=COLORS["orange"])
    axes[1].set_ylabel("Subscription requests")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[model] for model in models])
    axes[1].set_title("(b) Subscription utilization")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    eff_conc = [data[model]["effective_concurrency"] for model in models]
    conc_ratio = [conc / quota if quota > 0 else 0 for conc, quota in zip(conc_reqs, quota_reqs, strict=True)]
    bars = axes[2].bar(
        x,
        conc_ratio,
        color=[COLORS["blue_light"], COLORS["green_light"], COLORS["orange_light"]],
        edgecolor="white",
    )
    axes[2].axhline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1.2, alpha=0.8)
    for bar, concurrency in zip(bars, eff_conc, strict=True):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"C={concurrency}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[2].set_ylabel("Conc. / quota requests")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([labels[model] for model in models])
    axes[2].set_title("(c) Concurrency impact")
    axes[2].grid(axis="y", linestyle="--", alpha=0.3)

    _save_pdf(fig, output_pdf)


def _plot_cost_breakdown(results_path: Path, ilp_path: Path, output_tiers: Path, output_cost: Path) -> None:
    results = _load_json(results_path)
    ilp_data = _load_json(ilp_path)
    assignments = ilp_data["assignments"]

    counts = {"daily": 0, "concurrency": 0, "api": 0}
    for assignment in assignments.values():
        counts[assignment] = counts.get(assignment, 0) + 1

    total_requests = results["All-API"]["requests"]["api"]
    algorithms = [
        {
            "name": "ILP-Optimal",
            "sq": counts.get("daily", 0),
            "sc": counts.get("concurrency", 0),
            "sa": counts.get("api", 0),
            "api_cost": results["Offline-Optimal-ILP"]["costs"]["api"],
        },
        {
            "name": "PD-EMA",
            "sq": None,
            "sc": None,
            "subscription": results["PrimalDual-Online-Stage2"]["requests"]["subscription"],
            "sa": results["PrimalDual-Online-Stage2"]["requests"]["api"],
            "api_cost": results["PrimalDual-Online-Stage2"]["costs"]["api"],
        },
        {
            "name": "PD-Hist",
            "sq": results["LA-PD-Unified-Stage2"]["strategy_stats"]["sq_routed"],
            "sc": results["LA-PD-Unified-Stage2"]["strategy_stats"]["sc_routed"],
            "sa": results["LA-PD-Unified-Stage2"]["strategy_stats"]["api_routed"],
            "api_cost": results["LA-PD-Unified-Stage2"]["costs"]["api"],
        },
        {
            "name": "Greedy",
            "sq": None,
            "sc": None,
            "subscription": results["Greedy-Online-Stage2"]["requests"]["subscription"],
            "sa": results["Greedy-Online-Stage2"]["requests"]["api"],
            "api_cost": results["Greedy-Online-Stage2"]["costs"]["api"],
        },
        {
            "name": "All-API",
            "sq": 0,
            "sc": 0,
            "sa": results["All-API"]["requests"]["api"],
            "api_cost": results["All-API"]["costs"]["api"],
        },
    ]

    sq_pct = []
    sc_pct = []
    sa_pct = []
    for algorithm in algorithms:
        if algorithm["sq"] is None:
            sq_pct.append(algorithm["subscription"] / total_requests * 100)
            sc_pct.append(0.0)
        else:
            sq_pct.append(algorithm["sq"] / total_requests * 100)
            sc_pct.append(algorithm["sc"] / total_requests * 100)
        sa_pct.append(algorithm["sa"] / total_requests * 100)

    x = np.arange(len(algorithms))
    labels = [algorithm["name"] for algorithm in algorithms]

    fig1, ax1 = plt.subplots(figsize=(4.8, 3.5))
    ax1.bar(x, sq_pct, label=r"$S_Q$ (quota)", color=COLORS["blue"])
    ax1.bar(x, sc_pct, bottom=sq_pct, label=r"$S_C$ (concurrency)", color=COLORS["green"])
    stacked_bottom = [sq + sc for sq, sc in zip(sq_pct, sc_pct, strict=True)]
    ax1.bar(x, sa_pct, bottom=stacked_bottom, label=r"$S_A$ (API)", color=COLORS["orange"])
    for index, algorithm in enumerate(algorithms):
        if algorithm["sq"] is None:
            ax1.text(x[index], 2.0, "*", ha="center", va="center", color=COLORS["gray"], fontsize=13)
    ax1.text(
        0.99,
        -0.18,
        "* combined subscription in raw JSON",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=COLORS["gray"],
    )
    ax1.set_ylabel("Requests (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right", framealpha=0.95)
    _save_pdf(fig1, output_tiers)

    fig2, ax2 = plt.subplots(figsize=(4.8, 3.5))
    api_costs = [algorithm["api_cost"] for algorithm in algorithms]
    bars = ax2.bar(
        x,
        api_costs,
        color=[COLORS["orange"], COLORS["blue"], COLORS["green"], "#F6AD55", "#FC8181"],
    )
    for bar, cost in zip(bars, api_costs, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(api_costs) * 0.015,
            f"${cost:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2.set_ylabel("API overflow cost ($)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    _save_pdf(fig2, output_cost)


def _plot_provider_distribution(log_path: Path, output_pdf: Path) -> None:
    display_names = {
        "openrouter_auto": "OpenRouter Auto",
        "sort_price": "sort=price",
        "sort_throughput": "sort=throughput",
        "sort_latency": "sort=latency",
        "cheapest_fixed": "Cheapest Fixed",
        "fastest_fixed": "Fastest Fixed",
        "lp_mix": "Ours: Latency-Aware",
        "smart_hedge": "Ours: Smart Hedge",
    }

    policy_providers: dict[str, dict[str, int]] = {policy: {} for policy in display_names}
    policy_totals = {policy: 0 for policy in display_names}

    with log_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            policy = row["policy"]
            provider = row["actual_provider"]
            if policy not in display_names or row["status"] != "success" or not provider:
                continue
            policy_providers[policy][provider] = policy_providers[policy].get(provider, 0) + 1
            policy_totals[policy] += 1

    provider_totals: dict[str, int] = {}
    for providers in policy_providers.values():
        for provider, count in providers.items():
            provider_totals[provider] = provider_totals.get(provider, 0) + count

    top_providers = [provider for provider, _count in sorted(
        provider_totals.items(), key=lambda item: item[1], reverse=True
    )[:7]]
    categories = top_providers + ["Other"]
    colors = ["#2C7FB8", "#41B6C4", "#7FCDBB", "#F6E8C3", "#F4A259", "#E76F51", "#9C89B8", "#CBD5E0"]

    matrix = []
    labels = []
    for policy in display_names:
        total = policy_totals[policy]
        if total == 0:
            continue
        row = []
        other_share = 0.0
        for category in top_providers:
            row.append(policy_providers[policy].get(category, 0) / total * 100)
        for provider, count in policy_providers[policy].items():
            if provider not in top_providers:
                other_share += count / total * 100
        row.append(other_share)
        matrix.append(row)
        labels.append(display_names[policy])

    data = np.array(matrix)
    y = np.arange(len(labels))
    left = np.zeros(len(labels))

    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    for index, category in enumerate(categories):
        widths = data[:, index]
        ax.barh(y, widths, left=left, height=0.62, label=category, color=colors[index])
        left += widths

    ax.set_xlabel("Requests (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", framealpha=0.95)
    ax.invert_yaxis()
    _save_pdf(fig, output_pdf)


def _plot_latency_pareto(summary_path: Path, output_pdf: Path) -> None:
    df = pd.read_csv(summary_path)
    pricing = {
        "Groq": 0.27,
        "SambaNova": 0.40,
        "Cerebras": 0.60,
        "Cloudflare": 0.35,
        "Together": 0.80,
        "Fireworks": 0.90,
        "Parasail": 0.10,
        "Nebius": 0.50,
        "Hyperbolic": 0.40,
        "Crusoe": 1.20,
        "Novita": 0.70,
        "Friendli": 0.80,
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    ax1, ax2 = axes

    top_providers = df.sort_values("p99").head(6)["provider"].tolist()
    for _, row in df.iterrows():
        provider = row["provider"]
        if provider in top_providers:
            if row["p99"] < 2000:
                color = COLORS["green"]
            elif row["p99"] < 5000:
                color = COLORS["orange"]
            else:
                color = COLORS["red"]
            ax1.scatter(row["p50"], row["p99"], s=90, color=color, edgecolors="white", linewidths=1.2)
            ax1.annotate(provider, (row["p50"], row["p99"]), xytext=(6, 0), textcoords="offset points", fontsize=8)
    ax1.axhline(2000, color=COLORS["gray"], linestyle="--", linewidth=1.4, label="SLO = 2s")
    ax1.set_xlabel("P50 latency (ms)")
    ax1.set_ylabel("P99 latency (ms)")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right", framealpha=0.95)

    for _, row in df.iterrows():
        provider = row["provider"]
        cost = pricing.get(provider, 0.5)
        if row["p99"] < 1500:
            color = COLORS["green"]
        elif row["p99"] < 5000:
            color = COLORS["orange"]
        else:
            color = COLORS["red"]
        ax2.scatter(row["p99"], cost, s=90, color=color, edgecolors="white", linewidths=1.2)
        ax2.annotate(provider, (row["p99"], cost), xytext=(6, 0), textcoords="offset points", fontsize=8)
    ax2.axvline(2000, color=COLORS["gray"], linestyle="--", linewidth=1.4, label="SLO = 2s")
    ax2.set_xlabel("P99 latency (ms)")
    ax2.set_ylabel("Cost ($ / 1M tokens)")
    ax2.set_xscale("log")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.95)
    _save_pdf(fig, output_pdf)


def _plot_algorithm_cost_router(output_pdf: Path) -> None:
    request_labels = [r"$r_1$", r"$r_2$", r"$r_3$", r"$r_4$", r"$r_5$"]
    values = np.array([0.62, 0.08, 0.85, 0.28, 0.91])
    thresholds = np.array([0.05, 0.30, 0.30, 0.75, 0.75])
    routed_to_quota = np.array([True, False, True, False, True])
    slots_after = [1, 1, 2, 2, 3]
    x = np.arange(len(request_labels))

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    colors = [COLORS["purple"] if route else COLORS["orange"] for route in routed_to_quota]
    bars = ax.bar(x, values, width=0.65, color=colors, edgecolor="white", linewidth=1.2)
    ax.step(x, thresholds, where="mid", color=COLORS["red"], linewidth=2.2, label="Shadow-price threshold")
    ax.scatter(x, thresholds, color=COLORS["red"], s=28, zorder=4)

    for bar, used_slots in zip(bars, slots_after, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            1.03,
            f"{used_slots}/3 slots",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["gray"],
        )

    ax.annotate(
        r"$r_2$: \$0.08 < \$0.30" "\nroute to API",
        xy=(x[1], values[1]),
        xytext=(0.4, 0.48),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": COLORS["orange"], "lw": 1.3},
        fontsize=9,
        color=COLORS["gray"],
    )
    ax.annotate(
        r"$r_3$: \$0.85 > \$0.30" "\nuse quota",
        xy=(x[2], values[2]),
        xytext=(2.6, 0.98),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": COLORS["purple"], "lw": 1.3},
        fontsize=9,
        color=COLORS["gray"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(request_labels)
    ax.set_ylabel("Estimated request value ($)")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=COLORS["red"], lw=2.2, label="Shadow-price threshold"),
            Patch(facecolor=COLORS["purple"], label="Route to quota"),
            Patch(facecolor=COLORS["orange"], label="Route to API"),
        ],
        loc="upper left",
        framealpha=0.95,
    )
    _save_pdf(fig, output_pdf)


def _plot_algorithm_hedging(output_pdf: Path) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )

    slo_ms = 2000
    hedge_ms = 1100
    primary_finish_ms = 2280
    backup_finish_ms = 1480

    ax_top.broken_barh([(0, primary_finish_ms)], (18, 8), facecolors=COLORS["blue"], alpha=0.85)
    ax_top.broken_barh([(hedge_ms, backup_finish_ms - hedge_ms)], (4, 8), facecolors=COLORS["green"], alpha=0.85)
    ax_top.text(80, 22, "Primary request", color="white", fontsize=9, va="center", fontweight="bold")
    ax_top.text(hedge_ms + 60, 8, "Backup request", color="white", fontsize=9, va="center", fontweight="bold")
    ax_top.axvline(hedge_ms, color=COLORS["orange"], linestyle="--", linewidth=1.5)
    ax_top.axvline(slo_ms, color=COLORS["red"], linestyle="--", linewidth=1.5)
    ax_top.axvline(backup_finish_ms, color=COLORS["green"], linestyle=":", linewidth=1.5)
    ax_top.annotate(
        "Backup dispatched",
        xy=(hedge_ms, 26),
        xytext=(hedge_ms + 120, 30),
        arrowprops={"arrowstyle": "->", "color": COLORS["orange"], "lw": 1.2},
        fontsize=9,
    )
    ax_top.annotate(
        "First response returned",
        xy=(backup_finish_ms, 12),
        xytext=(backup_finish_ms - 560, 30),
        arrowprops={"arrowstyle": "->", "color": COLORS["green"], "lw": 1.2},
        fontsize=9,
    )
    ax_top.text(slo_ms + 25, 29, "SLO deadline", color=COLORS["red"], fontsize=9, va="center")
    ax_top.set_ylim(0, 34)
    ax_top.set_yticks([])
    ax_top.set_ylabel("Timeline")

    t = np.linspace(0, 2200, 300)
    violation_risk = 0.10 + 1.10 / (1.0 + np.exp(-(t - 1200) / 220.0))
    backup_cost = np.full_like(t, 0.68)
    ax_bottom.plot(t, violation_risk, color=COLORS["red"], linewidth=2.2, label="Expected violation loss")
    ax_bottom.plot(t, backup_cost, color=COLORS["gray"], linestyle="--", linewidth=1.6, label="Backup cost")
    ax_bottom.fill_between(
        t,
        backup_cost,
        violation_risk,
        where=violation_risk >= backup_cost,
        color=COLORS["red_light"],
        alpha=0.8,
    )
    ax_bottom.axvline(hedge_ms, color=COLORS["orange"], linestyle="--", linewidth=1.5)
    ax_bottom.text(hedge_ms + 25, 1.18, r"Trigger: expected loss $>$ backup cost", fontsize=9, color=COLORS["orange"])
    ax_bottom.set_xlabel("Elapsed time since dispatch (ms)")
    ax_bottom.set_ylabel("Relative cost")
    ax_bottom.grid(axis="y", linestyle="--", alpha=0.3)
    ax_bottom.legend(loc="upper left", framealpha=0.95)
    _save_pdf(fig, output_pdf)


def _write_manifest(output_dir: Path, entries: list[ManifestEntry]) -> None:
    manifest_path = output_dir / "manifest.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "figures": [
            {
                "name": entry.name,
                "paper_asset": entry.paper_asset,
                "status": entry.status,
                "output": entry.output,
                "method": entry.method,
                "inputs": entry.inputs,
                "notes": entry.notes,
            }
            for entry in entries
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Vector Figure Export",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "| Figure | Status | Output | Method | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        output_name = Path(entry.output).name if entry.output else "-"
        lines.append(
            f"| `{entry.name}` | `{entry.status}` | `{output_name}` | "
            f"`{entry.method}` | {entry.notes} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def export_vector_figures(paper_dir: Path, output_dir: Path) -> list[ManifestEntry]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "source_audit"
    entries: list[ManifestEntry] = []

    tasks = [
        (
            "system_architecture",
            paper_dir / "images/system_architecture.png",
            output_dir / "system_architecture.pdf",
            "compiled_tikz",
            [str(PROJECT_ROOT / "overleaf_patched/fig_preview.tex")],
            "Used the standalone TikZ version from overleaf_patched because the paper tree only stores a raster PNG plus a drawio file.",
            lambda target: _compile_standalone_tex(PROJECT_ROOT / "overleaf_patched/fig_preview.tex", target),
        ),
        (
            "fig2_sensitivity_sharegpt",
            paper_dir / "images/fig2_sensitivity_sharegpt.png",
            output_dir / "fig2_sensitivity_sharegpt.pdf",
            "replotted_from_json",
            [str(PROJECT_ROOT / "experiment/results/stage1/sharegpt_results.json")],
            "Directly replotted from Stage 1 result JSON.",
            lambda target: _plot_sensitivity(
                PROJECT_ROOT / "experiment/results/stage1/sharegpt_results.json",
                "ShareGPT",
                target,
            ),
        ),
        (
            "fig2_sensitivity_freeinference",
            paper_dir / "images/fig2_sensitivity_freeinference.png",
            output_dir / "fig2_sensitivity_freeinference.pdf",
            "replotted_from_json",
            [str(PROJECT_ROOT / "experiment/results/stage1/freeinference_results.json")],
            "Directly replotted from Stage 1 result JSON.",
            lambda target: _plot_sensitivity(
                PROJECT_ROOT / "experiment/results/stage1/freeinference_results.json",
                "FreeInference",
                target,
            ),
        ),
        (
            "single_model_comparison_rednote",
            paper_dir / "images/single_model_comparison_rednote.png",
            output_dir / "single_model_comparison_rednote.pdf",
            "replotted_from_json",
            [str(PROJECT_ROOT / "experiment/results/stage2/single_model_comparison_rednote.json")],
            "Directly replotted from Stage 2 single-model comparison JSON.",
            lambda target: _plot_single_model_comparison(
                PROJECT_ROOT / "experiment/results/stage2/single_model_comparison_rednote.json",
                target,
            ),
        ),
        (
            "online_calibration",
            paper_dir / "images/online_calibration.png",
            output_dir / "online_calibration.pdf",
            "replotted_from_json",
            [str(PROJECT_ROOT / "experiment/results/online/stage1_online_results.json")],
            "Directly replotted from Stage 1 online evaluation JSON.",
            lambda target: _plot_online_calibration(
                PROJECT_ROOT / "experiment/results/online/stage1_online_results.json",
                target,
            ),
        ),
        (
            "cost_breakdown_tiers",
            paper_dir / "images/cost_breakdown_tiers.png",
            output_dir / "cost_breakdown_tiers.pdf",
            "replotted_from_json",
            [
                str(PROJECT_ROOT / "experiment/results/online/stage2_online_results.json"),
                str(
                    PROJECT_ROOT
                    / "experiment/cache/ilp/stage2_online_featherless_scale_n371214_d1.0_Q5000_C8_cbc_slo0_pe4f3b39d_s0ca43dae.json"
                ),
            ],
            "Tier distribution replotted from Stage 2 online results plus the ILP assignment cache.",
            lambda target: _plot_cost_breakdown(
                PROJECT_ROOT / "experiment/results/online/stage2_online_results.json",
                PROJECT_ROOT
                / "experiment/cache/ilp/stage2_online_featherless_scale_n371214_d1.0_Q5000_C8_cbc_slo0_pe4f3b39d_s0ca43dae.json",
                target,
                output_dir / "cost_breakdown_api_cost.pdf",
            ),
        ),
        (
            "provider_distribution",
            paper_dir / "images/provider_distribution.png",
            output_dir / "provider_distribution.pdf",
            "replotted_from_csv",
            [str(PROJECT_ROOT / "experiment/results/phase5_qwen3_235b/run_20260405_073250/evaluation_log.csv")],
            "Replotted from the Qwen3-235B production evaluation log.",
            lambda target: _plot_provider_distribution(
                PROJECT_ROOT / "experiment/results/phase5_qwen3_235b/run_20260405_073250/evaluation_log.csv",
                target,
            ),
        ),
        (
            "cost_breakdown_api_cost",
            paper_dir / "images/cost_breakdown_api_cost.png",
            output_dir / "cost_breakdown_api_cost.pdf",
            "replotted_from_json",
            [
                str(PROJECT_ROOT / "experiment/results/online/stage2_online_results.json"),
                str(
                    PROJECT_ROOT
                    / "experiment/cache/ilp/stage2_online_featherless_scale_n371214_d1.0_Q5000_C8_cbc_slo0_pe4f3b39d_s0ca43dae.json"
                ),
            ],
            "API overflow cost replotted from Stage 2 online results plus the ILP assignment cache.",
            lambda target: _plot_cost_breakdown(
                PROJECT_ROOT / "experiment/results/online/stage2_online_results.json",
                PROJECT_ROOT
                / "experiment/cache/ilp/stage2_online_featherless_scale_n371214_d1.0_Q5000_C8_cbc_slo0_pe4f3b39d_s0ca43dae.json",
                output_dir / "cost_breakdown_tiers.pdf",
                target,
            ),
        ),
        (
            "latency_pareto_openrouter",
            paper_dir / "images/latency_pareto_openrouter.png",
            output_dir / "latency_pareto_openrouter.pdf",
            "replotted_from_csv",
            [str(PROJECT_ROOT / "experiment/results/latency_phase1_24h/latency_summary_stats.csv")],
            "Replotted from exported provider percentile statistics.",
            lambda target: _plot_latency_pareto(
                PROJECT_ROOT / "experiment/results/latency_phase1_24h/latency_summary_stats.csv",
                target,
            ),
        ),
        (
            "algorithm_cost_router",
            paper_dir / "images/algorithm_cost_router.png",
            output_dir / "algorithm_cost_router.pdf",
            "redrawn_vector",
            [str(paper_dir / "sections/03-design.tex")],
            "The repository only stores a raster PNG for this conceptual figure. This PDF is a newly redrawn vector version based on the paper caption and surrounding text.",
            lambda target: _plot_algorithm_cost_router(target),
        ),
        (
            "algorithm_hedging",
            paper_dir / "images/algorithm_hedging.png",
            output_dir / "algorithm_hedging.pdf",
            "redrawn_vector",
            [str(paper_dir / "sections/03-design.tex")],
            "The repository only stores a raster PNG for this conceptual figure. This PDF is a newly redrawn vector version based on the paper caption and surrounding text.",
            lambda target: _plot_algorithm_hedging(target),
        ),
        (
            "drift_wall_clock_llama",
            paper_dir / "images/drift_wall_clock_llama.png",
            output_dir / "drift_wall_clock_llama.pdf",
            "replotted_from_remote_llmapi_bench_csv",
            [
                "remote_git:/home/murphy/test/hybridInference/.git/modules/llmAPI_bench:HEAD:data/meta/TTFT_prompt_len_Llama-3.3-70B-Instruct.csv",
                str(audit_dir / "TTFT_prompt_len_Llama-3.3-70B-Instruct.csv"),
            ],
            "Replotted from the long-horizon llmAPI_bench raw CSV fetched from freeinference-direct. The original plotting script is not in the repository, so this vector rebuild uses the archived raw series plus a reverse-engineered rolling window (120 points) chosen to match the paper PNG.",
            lambda target: _export_llmapi_bench_drift_figure(
                remote_object_path="data/meta/TTFT_prompt_len_Llama-3.3-70B-Instruct.csv",
                audit_copy_path=audit_dir / "TTFT_prompt_len_Llama-3.3-70B-Instruct.csv",
                output_pdf=target,
                title="(a) Meta: Llama-3.3-70B (input=10 tokens)",
                rolling_window_points=120,
                y_limits=(200, 5500),
            ),
        ),
        (
            "drift_wall_clock_gpt4o",
            paper_dir / "images/drift_wall_clock_gpt4o.png",
            output_dir / "drift_wall_clock_gpt4o.pdf",
            "replotted_from_remote_llmapi_bench_csv",
            [
                "remote_git:/home/murphy/test/hybridInference/.git/modules/llmAPI_bench:HEAD:data/harvard_openai/TTFT_prompt_len_gpt-4o-mini.csv",
                str(audit_dir / "TTFT_prompt_len_gpt-4o-mini.csv"),
            ],
            "Replotted from the long-horizon llmAPI_bench raw CSV fetched from freeinference-direct. The original plotting script is not in the repository, so this vector rebuild uses the archived raw series plus a reverse-engineered rolling window (192 points) chosen to match the paper PNG.",
            lambda target: _export_llmapi_bench_drift_figure(
                remote_object_path="data/harvard_openai/TTFT_prompt_len_gpt-4o-mini.csv",
                audit_copy_path=audit_dir / "TTFT_prompt_len_gpt-4o-mini.csv",
                output_pdf=target,
                title="(b) OpenAI: gpt-4o-mini (input=10 tokens)",
                rolling_window_points=192,
                rolling_min_periods=2,
                y_limits=(300, 5000),
            ),
        ),
    ]

    for name, paper_asset, output_pdf, method, inputs, notes, task in tasks:
        try:
            task(output_pdf)
            entries.append(
                ManifestEntry(
                    name=name,
                    paper_asset=str(paper_asset),
                    status="generated",
                    output=str(output_pdf),
                    method=method,
                    inputs=inputs,
                    notes=notes,
                )
            )
        except Exception as exc:  # pylint: disable=broad-except
            entries.append(
                ManifestEntry(
                    name=name,
                    paper_asset=str(paper_asset),
                    status="failed",
                    output=None,
                    method=method,
                    inputs=inputs,
                    notes=f"{notes} Failure: {exc}",
                )
            )

    _write_manifest(output_dir, entries)
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export vector PDFs for the paper review folder.")
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR, help="Paper directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Vector figure output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = export_vector_figures(args.paper_dir, args.output_dir)
    generated = sum(entry.status == "generated" for entry in entries)
    missing = sum(entry.status == "missing_source_data" for entry in entries)
    failed = sum(entry.status == "failed" for entry in entries)
    print(f"Output directory: {args.output_dir}")
    print(f"Generated: {generated}")
    print(f"Missing source data: {missing}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
