#!/usr/bin/env python3
"""Analyze Ollama Cloud quota probe results and estimate safe session quota.

Reads the JSONL request log produced by ``ollama_quota_probe.py``, computes
consumption metrics, extrapolates to a full 5-hour session window, and
produces a quota estimate with multiple safety margins.

Outputs:
    quota_estimate.json  -- All metrics and quota estimates
    consumption_rate.png -- Cumulative requests & GPU-time vs wall-clock time

Example:
    source .venv/bin/activate
    python experiment/scripts/analyze_ollama_quota.py \
        --input-dir experiment/results/ollama_quota_probe/glm47_60min \
        --session-hours 5
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SESSION_HOURS_DEFAULT = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Ollama quota probe results."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing requests.jsonl and summary.json from the probe.",
    )
    parser.add_argument(
        "--session-hours",
        type=float,
        default=SESSION_HOURS_DEFAULT,
        help=f"Target session window in hours (default: {SESSION_HOURS_DEFAULT}).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for analysis. Defaults to input-dir.",
    )
    return parser.parse_args()


@dataclass
class RequestMetrics:
    """Parsed metrics from one probe request."""

    index: int
    label: str
    ok: bool
    limit_hit: bool
    wall_time_seconds: float
    total_duration_ns: int
    load_duration_ns: int
    prompt_eval_count: int
    prompt_eval_duration_ns: int
    eval_count: int
    eval_duration_ns: int
    effective_usage_ns: int
    response_chars: int


def load_requests(input_dir: pathlib.Path) -> list[RequestMetrics]:
    """Load and parse the JSONL request log."""
    path = input_dir / "requests.jsonl"
    if not path.exists():
        print(f"requests.jsonl not found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    records: list[RequestMetrics] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(
                RequestMetrics(
                    index=data["index"],
                    label=data["label"],
                    ok=data["ok"],
                    limit_hit=data["limit_hit"],
                    wall_time_seconds=data["wall_time_seconds"],
                    total_duration_ns=data["total_duration_ns"],
                    load_duration_ns=data["load_duration_ns"],
                    prompt_eval_count=data["prompt_eval_count"],
                    prompt_eval_duration_ns=data["prompt_eval_duration_ns"],
                    eval_count=data["eval_count"],
                    eval_duration_ns=data["eval_duration_ns"],
                    effective_usage_ns=data["effective_usage_ns"],
                    response_chars=data["response_chars"],
                )
            )
    return records


def load_summary(input_dir: pathlib.Path) -> dict:
    """Load the probe summary JSON."""
    path = input_dir / "summary.json"
    if not path.exists():
        print(f"summary.json not found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    with path.open() as f:
        return json.load(f)


def percentile_val(values: list[float], pct: float) -> float:
    """Compute a percentile with linear interpolation."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    if len(sorted_v) == 1:
        return sorted_v[0]
    rank = (len(sorted_v) - 1) * (pct / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_v[lower]
    weight = rank - lower
    return sorted_v[lower] * (1.0 - weight) + sorted_v[upper] * weight


def analyze(
    records: list[RequestMetrics],
    summary: dict,
    session_hours: float,
) -> dict:
    """Compute consumption metrics and extrapolate quota."""
    successes = [r for r in records if r.ok]
    limit_hit = any(r.limit_hit for r in records)
    limit_index = next(
        (r.index for r in records if r.limit_hit), None
    )

    if not successes:
        return {"error": "No successful requests found."}

    n = len(successes)
    total_wall_seconds = sum(r.wall_time_seconds for r in successes)
    # Ollama Cloud may not return detailed GPU breakdowns (effective_usage_ns=0).
    # Fall back to total_duration_ns which is always reported.
    use_total_duration = all(r.effective_usage_ns == 0 for r in successes)
    total_gpu_ns = sum(
        (r.total_duration_ns if use_total_duration else r.effective_usage_ns)
        for r in successes
    )
    total_tokens = sum(r.prompt_eval_count + r.eval_count for r in successes)
    total_output_tokens = sum(r.eval_count for r in successes)
    total_input_tokens = sum(r.prompt_eval_count for r in successes)

    # Per-request distributions
    gpu_ns_per_req = [
        (r.total_duration_ns if use_total_duration else r.effective_usage_ns)
        for r in successes
    ]
    tokens_per_req = [r.prompt_eval_count + r.eval_count for r in successes]
    output_tokens_per_req = [r.eval_count for r in successes]
    wall_time_per_req = [r.wall_time_seconds for r in successes]

    elapsed_minutes = summary.get("elapsed_minutes", total_wall_seconds / 60.0)
    session_seconds = session_hours * 3600.0

    # Consumption rates
    requests_per_minute = n / elapsed_minutes if elapsed_minutes > 0 else 0
    gpu_ns_per_minute = total_gpu_ns / elapsed_minutes if elapsed_minutes > 0 else 0

    # Extrapolation to full session
    session_minutes = session_hours * 60.0
    if limit_hit:
        # Limit was hit during probe - use observed count directly
        extrapolated_requests = n
        extrapolation_method = "limit_hit_in_probe"
    else:
        # Extrapolate linearly from observed rate
        extrapolated_requests = int(requests_per_minute * session_minutes)
        extrapolation_method = "linear_from_rate"

    # Safety margins
    quota_aggressive = extrapolated_requests
    quota_safe_80 = int(extrapolated_requests * 0.80)
    quota_safe_70 = int(extrapolated_requests * 0.70)

    # Per-label breakdown (short/medium/long)
    label_stats: dict[str, dict] = {}
    for r in successes:
        prefix = r.label.rsplit("_", 1)[0] if "_" in r.label else r.label
        if prefix not in label_stats:
            label_stats[prefix] = {
                "count": 0,
                "total_gpu_ns": 0,
                "total_tokens": 0,
                "total_output_tokens": 0,
                "wall_times": [],
            }
        stats = label_stats[prefix]
        stats["count"] += 1
        stats["total_gpu_ns"] += (
            r.total_duration_ns if use_total_duration else r.effective_usage_ns
        )
        stats["total_tokens"] += r.prompt_eval_count + r.eval_count
        stats["total_output_tokens"] += r.eval_count
        stats["wall_times"].append(r.wall_time_seconds)

    label_summary = {}
    for prefix, stats in sorted(label_stats.items()):
        cnt = stats["count"]
        label_summary[prefix] = {
            "count": cnt,
            "avg_gpu_ms": (stats["total_gpu_ns"] / cnt / 1e6) if cnt > 0 else 0,
            "avg_tokens": stats["total_tokens"] / cnt if cnt > 0 else 0,
            "avg_output_tokens": stats["total_output_tokens"] / cnt if cnt > 0 else 0,
            "avg_wall_seconds": (
                sum(stats["wall_times"]) / cnt if cnt > 0 else 0
            ),
        }

    gpu_metric_source = "total_duration_ns" if use_total_duration else "effective_usage_ns"

    result = {
        "model": summary.get("model", "unknown"),
        "gpu_metric_source": gpu_metric_source,
        "probe_elapsed_minutes": round(elapsed_minutes, 2),
        "session_hours": session_hours,
        "successful_requests": n,
        "limit_hit": limit_hit,
        "limit_hit_at_request": limit_index,
        "extrapolation_method": extrapolation_method,
        # Consumption rates
        "requests_per_minute": round(requests_per_minute, 2),
        "gpu_ns_per_minute": int(gpu_ns_per_minute),
        "tokens_per_minute": round(total_tokens / elapsed_minutes, 1) if elapsed_minutes > 0 else 0,
        # Per-request stats
        "per_request": {
            "gpu_ms_mean": round(np.mean(gpu_ns_per_req) / 1e6, 1),
            "gpu_ms_p50": round(percentile_val(gpu_ns_per_req, 50) / 1e6, 1),
            "gpu_ms_p95": round(percentile_val(gpu_ns_per_req, 95) / 1e6, 1),
            "gpu_ms_p99": round(percentile_val(gpu_ns_per_req, 99) / 1e6, 1),
            "tokens_mean": round(np.mean(tokens_per_req), 1),
            "tokens_p50": round(percentile_val(tokens_per_req, 50), 1),
            "tokens_p95": round(percentile_val(tokens_per_req, 95), 1),
            "output_tokens_mean": round(np.mean(output_tokens_per_req), 1),
            "wall_seconds_mean": round(np.mean(wall_time_per_req), 2),
            "wall_seconds_p95": round(percentile_val(wall_time_per_req, 95), 2),
        },
        # Totals
        "total_gpu_seconds": round(total_gpu_ns / 1e9, 2),
        "total_tokens": total_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        # Quota estimates
        "quota_estimates": {
            "extrapolated_5h_requests": extrapolated_requests,
            "quota_aggressive": quota_aggressive,
            "quota_safe_80pct": quota_safe_80,
            "quota_safe_70pct": quota_safe_70,
        },
        # By prompt category
        "by_category": label_summary,
        # Config snippet for experiment.yaml
        "yaml_config_snippet": {
            "ollama_pro": {
                "window_hours": session_hours,
                "quota_per_window": quota_safe_80,
                "quota_unit": "message",
                "monthly_fee": 20.0,
                "supported_models": [summary.get("model", "unknown").replace(":cloud", "")],
                "notes": (
                    f"Estimated from {elapsed_minutes:.0f}min probe: "
                    f"{n} reqs observed, {requests_per_minute:.1f} req/min rate, "
                    f"extrapolated {extrapolated_requests} per {session_hours}h, "
                    f"80% safety margin applied."
                ),
            }
        },
    }
    return result


def plot_consumption(
    records: list[RequestMetrics],
    analysis: dict,
    output_path: pathlib.Path,
    session_hours: float,
) -> None:
    """Plot cumulative requests and GPU-time vs wall-clock time."""
    successes = [r for r in records if r.ok]
    if not successes:
        return

    # Compute cumulative wall-clock time and metrics
    cum_wall_minutes: list[float] = []
    cum_requests: list[int] = []
    cum_gpu_seconds: list[float] = []
    cum_tokens: list[int] = []

    running_wall = 0.0
    running_gpu = 0.0
    running_tokens = 0

    use_total_duration = all(r.effective_usage_ns == 0 for r in successes)
    for i, r in enumerate(successes):
        running_wall += r.wall_time_seconds
        gpu_ns = r.total_duration_ns if use_total_duration else r.effective_usage_ns
        running_gpu += gpu_ns / 1e9
        running_tokens += r.prompt_eval_count + r.eval_count
        cum_wall_minutes.append(running_wall / 60.0)
        cum_requests.append(i + 1)
        cum_gpu_seconds.append(running_gpu)
        cum_tokens.append(running_tokens)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Ollama Cloud Quota Probe: {analysis.get('model', '?')}",
        fontsize=14,
        fontweight="bold",
    )

    # Top-left: Cumulative requests vs wall-clock time
    ax = axes[0, 0]
    ax.plot(cum_wall_minutes, cum_requests, "b-", linewidth=1.5)
    ax.set_xlabel("Wall-Clock Time (minutes)")
    ax.set_ylabel("Cumulative Requests")
    ax.set_title("Request Count Over Time")
    ax.grid(True, alpha=0.3)
    # Add extrapolation line
    if cum_wall_minutes:
        rate = analysis["requests_per_minute"]
        session_min = session_hours * 60
        extrap_x = [0, session_min]
        extrap_y = [0, rate * session_min]
        ax.plot(extrap_x, extrap_y, "r--", alpha=0.5, label=f"Extrapolated ({rate:.1f} req/min)")
        ax.axvline(x=cum_wall_minutes[-1], color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=9)

    # Top-right: Cumulative GPU-time vs wall-clock time
    ax = axes[0, 1]
    ax.plot(cum_wall_minutes, cum_gpu_seconds, "g-", linewidth=1.5)
    ax.set_xlabel("Wall-Clock Time (minutes)")
    ax.set_ylabel("Cumulative GPU Time (seconds)")
    ax.set_title("GPU Utilization Over Time")
    ax.grid(True, alpha=0.3)

    # Bottom-left: Per-request GPU time distribution
    ax = axes[1, 0]
    gpu_ms = [
        (r.total_duration_ns if use_total_duration else r.effective_usage_ns) / 1e6
        for r in successes
    ]
    ax.hist(gpu_ms, bins=min(50, len(gpu_ms)), color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(
        x=analysis["per_request"]["gpu_ms_p50"],
        color="red",
        linestyle="--",
        label=f"P50={analysis['per_request']['gpu_ms_p50']:.0f}ms",
    )
    ax.axvline(
        x=analysis["per_request"]["gpu_ms_p95"],
        color="orange",
        linestyle="--",
        label=f"P95={analysis['per_request']['gpu_ms_p95']:.0f}ms",
    )
    ax.set_xlabel("GPU Time per Request (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Request GPU Time Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Per-request output tokens distribution
    ax = axes[1, 1]
    out_tokens = [r.eval_count for r in successes]
    ax.hist(out_tokens, bins=min(50, len(out_tokens)), color="coral", edgecolor="white", alpha=0.8)
    ax.axvline(
        x=analysis["per_request"]["output_tokens_mean"],
        color="red",
        linestyle="--",
        label=f"Mean={analysis['per_request']['output_tokens_mean']:.0f}",
    )
    ax.set_xlabel("Output Tokens per Request")
    ax.set_ylabel("Count")
    ax.set_title("Output Token Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def main() -> None:
    args = parse_args()
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else input_dir

    print(f"Loading probe results from {input_dir} ...")
    records = load_requests(input_dir)
    summary = load_summary(input_dir)

    print(f"Loaded {len(records)} records ({sum(1 for r in records if r.ok)} successful)")

    analysis = analyze(records, summary, args.session_hours)

    # Save analysis
    output_dir.mkdir(parents=True, exist_ok=True)
    estimate_path = output_dir / "quota_estimate.json"
    with estimate_path.open("w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Quota estimate saved to {estimate_path}")

    # Plot
    plot_path = output_dir / "consumption_rate.png"
    plot_consumption(records, analysis, plot_path, args.session_hours)

    # Print summary
    print()
    print("=" * 60)
    print("  Ollama Cloud Quota Estimation Summary")
    print("=" * 60)
    print(f"  Model:                  {analysis.get('model', '?')}")
    print(f"  Probe duration:         {analysis['probe_elapsed_minutes']:.1f} min")
    print(f"  Successful requests:    {analysis['successful_requests']}")
    print(f"  Limit hit:              {analysis['limit_hit']}")
    print(f"  Request rate:           {analysis['requests_per_minute']:.2f} req/min")
    print(f"  Avg GPU time/req:       {analysis['per_request']['gpu_ms_mean']:.0f} ms")
    print(f"  Avg tokens/req:         {analysis['per_request']['tokens_mean']:.0f}")
    print(f"  Avg output tokens/req:  {analysis['per_request']['output_tokens_mean']:.0f}")
    print()
    print(f"  --- Quota Estimates ({analysis['session_hours']}h session) ---")
    est = analysis["quota_estimates"]
    print(f"  Extrapolated:           {est['extrapolated_5h_requests']}")
    print(f"  Aggressive (no margin): {est['quota_aggressive']}")
    print(f"  Safe (80%):             {est['quota_safe_80pct']}")
    print(f"  Safe (70%):             {est['quota_safe_70pct']}")
    print()
    print(f"  Method: {analysis['extrapolation_method']}")
    print()

    # Print by-category breakdown
    if analysis.get("by_category"):
        print("  --- By Prompt Category ---")
        for cat, stats in analysis["by_category"].items():
            print(
                f"  {cat:12s}: n={stats['count']:3d}  "
                f"gpu={stats['avg_gpu_ms']:.0f}ms  "
                f"tokens={stats['avg_tokens']:.0f}  "
                f"out={stats['avg_output_tokens']:.0f}  "
                f"wall={stats['avg_wall_seconds']:.1f}s"
            )
        print()

    # Print YAML snippet
    print("  --- experiment.yaml snippet ---")
    snippet = analysis["yaml_config_snippet"]["ollama_pro"]
    print(f"  ollama_pro:")
    print(f"    window_hours: {snippet['window_hours']}")
    print(f"    quota_per_window: {snippet['quota_per_window']}")
    print(f"    quota_unit: {snippet['quota_unit']}")
    print(f"    monthly_fee: {snippet['monthly_fee']}")
    print(f"    supported_models:")
    for m in snippet["supported_models"]:
        print(f"      - {m}")
    print(f'    notes: "{snippet["notes"]}"')
    print("=" * 60)


if __name__ == "__main__":
    main()
