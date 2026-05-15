"""Plot end-to-end simulator frontier results for the paper.

The input is ``experiments.simulation.end_to_end``'s ``summary.csv``. The
script emits one metric per PDF so LaTeX can resize and arrange figures without
regenerating multi-panel plots.

Example:

    uv run python -m plots.end_to_end.plot_simulation_frontier \
      --summary-csv outputs/simulation/end_to_end_rw8_minimax_q1_c1_slo3_20260514/summary.csv \
      --histograms-json outputs/simulation/end_to_end_rw8_minimax_q1_c1_slo3_20260514/ttft_histograms.json \
      --frontier-out ../paper/figures/stage3_e2e_cost_latency_frontier.pdf \
      --slo-out ../paper/figures/stage3_e2e_slo_cost_frontier.pdf \
      --tier-out ../paper/figures/stage3_e2e_tier_mix.pdf \
      --provider-latency-out ../paper/figures/stage3_e2e_provider_latency.pdf \
      --provider-mix-out ../paper/figures/stage3_e2e_provider_mix.pdf \
      --p99-bar-out ../paper/figures/stage3_e2e_hedging_p99.pdf \
      --cdf-out ../paper/figures/stage3_e2e_ttft_cdf.pdf \
      --table-out ../paper/tables/simulation_e2e_q1_c1_slo3_rows.tex \
      --summary-out ../paper/tables/simulation_e2e_q1_c1_slo3_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.simulation.latency_profiles import load_empirical_distribution
from experiments.simulation.provider_profiles import load_provider_pool
from plots.end_to_end.frontier_plotting import (
    MIX_FIGSIZE,
    PROVIDER_COLOR_CYCLE,
    PROVIDER_MIX_COLORS,
    TIER_MIX_SEGMENTS,
    BoxSeries,
    CdfSeries,
    FrontierPoint,
    MixRow,
    MixSegment,
    plot_hedging_p99 as plot_common_hedging_p99,
    plot_mean_ttft_frontier,
    plot_slo_frontier,
    plot_stacked_mix,
    plot_ttft_boxplot as plot_common_ttft_boxplot,
    plot_ttft_cdf as plot_common_ttft_cdf,
    policy_plot_label,
)
from plots.style import apply_style

POLICY_LABELS = {
    "greedy_cost": "Greedy-cost",
    "or_sort_cost": "OR-price",
    "random": "Random",
    "greedy_latency": "Greedy-latency",
    "or_sort_latency": "OR-latency",
}
BASELINE_ORDER = (
    "greedy_cost",
    "greedy_latency",
    "random",
)
TABLE_POLICIES = (
    "greedy_cost",
    "greedy_latency",
    "random",
)
ROUTEWISE_TABLE_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ROUTEWISE_FIGURE_POLICIES = (
    "ablation_lp_only_p0",
    "ablation_lp_only_p25",
    "ablation_lp_only_p50",
    "ablation_lp_only_p75",
    "ablation_lp_only_p100",
)
CDF_POLICIES = (
    *ROUTEWISE_FIGURE_POLICIES,
    "greedy_cost",
    "greedy_latency",
    "random",
)
BOXPLOT_POLICIES = (
    *ROUTEWISE_FIGURE_POLICIES,
    "greedy_cost",
    "greedy_latency",
)
PROVIDER_MIX_POLICIES = (
    *ROUTEWISE_FIGURE_POLICIES,
    "greedy_cost",
    "greedy_latency",
    "random",
)


@dataclass(frozen=True)
class Row:
    scenario: str
    policy: str
    label: str
    alpha: float | None
    hedging: bool
    provider_mix: dict[str, float]
    total_cost_usd: float
    mean_ttft_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    slo_ms: float
    slo_violation_rate: float
    hedge_rate: float
    tier_mix: dict[str, float]
    real_world_pool: str | None
    api_provider_names: tuple[str, ...]
    api_latency_provider_names: tuple[str, ...]
    latency_profile: str | None
    quota_latency_profile_provider: str | None
    concurrency_latency_profile_provider: str | None
    subscription_plan: str | None
    concurrency_plan: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--histograms-json", type=Path, default=None)
    parser.add_argument("--frontier-out", type=Path, required=True)
    parser.add_argument("--slo-out", type=Path, required=True)
    parser.add_argument("--tier-out", type=Path, default=None)
    parser.add_argument(
        "--provider-latency-out",
        type=Path,
        default=None,
        help="Optional output path for a configured provider mean-TTFT bar PDF.",
    )
    parser.add_argument(
        "--provider-mix-out",
        type=Path,
        default=None,
        help="Optional output path for a provider-mix stacked-bar PDF.",
    )
    parser.add_argument(
        "--provider-mix-policies",
        nargs="+",
        default=None,
        help=(
            "Optional exact policy names to include in the provider-mix figure. "
            "Defaults to representative policies excluding random."
        ),
    )
    parser.add_argument("--p99-bar-out", type=Path, default=None)
    parser.add_argument("--cdf-out", type=Path, default=None)
    parser.add_argument(
        "--boxplot-out",
        type=Path,
        default=None,
        help="Optional output path for a per-policy TTFT percentile boxplot PDF.",
    )
    parser.add_argument(
        "--boxplot-policies",
        nargs="+",
        default=None,
        help=(
            "Optional exact policy names to include in the TTFT boxplot. "
            "Defaults to representative policies excluding random."
        ),
    )
    parser.add_argument(
        "--frontier-baselines",
        nargs="+",
        default=None,
        help=(
            "Optional exact baseline policy names to include in cost-frontier "
            "figures. Defaults to all simulator baselines."
        ),
    )
    parser.add_argument(
        "--routewise-frontier-policies",
        nargs="+",
        default=None,
        help=(
            "RouteWise policies to show in compact frontier figures. "
            "Defaults to the simulator alpha sweep."
        ),
    )
    parser.add_argument("--table-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def parse_alpha(policy: str) -> float | None:
    if "_p" not in policy:
        return None
    suffix = policy.rsplit("_p", 1)[1]
    try:
        return int(suffix) / 100.0
    except ValueError:
        return None


def row_label(policy: str, alpha: float | None, hedging: bool) -> str:
    if policy in POLICY_LABELS:
        return POLICY_LABELS[policy]
    if alpha is None:
        return policy.replace("_", r"\_")
    suffix = " + hedge" if hedging else ""
    if alpha == 0.25:
        return f"RouteWise-{alpha:g}{suffix}"
    return rf"\sysname{{}} ($\alpha={alpha:g}${suffix})"


def load_histograms(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["policy"]: item["histogram"] for item in payload}


def parse_json_object(raw: str | None) -> dict[str, float]:
    if raw in {None, ""}:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        return {}
    return {str(key): float(value) for key, value in payload.items()}


def parse_json_list(raw: str | None) -> tuple[str, ...]:
    if raw in {None, ""}:
        return ()
    payload = json.loads(raw)
    if not isinstance(payload, list):
        return ()
    return tuple(str(item) for item in payload)


def histogram_quantile(histogram: dict[str, object] | None, q: float) -> float:
    if histogram is None:
        return float("nan")
    n = int(histogram["n"])
    if n <= 0:
        return float("nan")
    min_ms = float(histogram["min_ms"])
    max_ms = float(histogram["max_ms"])
    if min_ms == max_ms:
        return min_ms
    edges = [float(value) for value in histogram["bin_edges_ms"]]
    counts = [int(value) for value in histogram["counts"]]
    target = q * n
    cumulative = 0
    for idx, count in enumerate(counts):
        next_cumulative = cumulative + count
        if target <= next_cumulative:
            if idx == 0:
                return min_ms
            if idx == len(counts) - 1:
                return max_ms
            lo = edges[idx - 1]
            hi = edges[idx]
            if count <= 1:
                return 0.5 * (lo + hi)
            frac = (target - cumulative) / count
            return lo + frac * (hi - lo)
        cumulative = next_cumulative
    return edges[-1]


def histogram_cdf(histogram: dict[str, object], value_ms: float) -> float:
    n = int(histogram["n"])
    if n <= 0:
        return 0.0
    edges = [float(value) for value in histogram["bin_edges_ms"]]
    counts = [int(value) for value in histogram["counts"]]
    if value_ms < edges[0]:
        return 0.0
    if value_ms >= edges[-1]:
        return 1.0
    bin_idx = 0
    while bin_idx < len(edges) and edges[bin_idx] <= value_ms:
        bin_idx += 1
    cumulative = counts[0] + sum(counts[1:bin_idx])
    lo = edges[bin_idx - 1]
    hi = edges[bin_idx]
    count = counts[bin_idx]
    if count:
        cumulative += count * ((value_ms - lo) / (hi - lo))
    return min(max(cumulative / n, 0.0), 1.0)


def load_rows(
    path: Path,
    histograms: dict[str, dict[str, object]] | None = None,
) -> list[Row]:
    rows: list[Row] = []
    histograms = histograms or {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            policy = raw["policy"]
            alpha = parse_alpha(policy)
            hedging = raw.get("hedging_enabled") == "True"
            tier_mix_raw = raw.get("tier_mix") or "{}"
            p95_raw = raw.get("p95_ms")
            p95_ms = (
                float(p95_raw)
                if p95_raw not in {None, ""}
                else histogram_quantile(histograms.get(policy), 0.95)
            )
            rows.append(
                Row(
                    scenario=raw["scenario"],
                    policy=policy,
                    label=row_label(policy, alpha, hedging),
                    alpha=alpha,
                    hedging=hedging,
                    provider_mix=parse_json_object(raw.get("provider_mix")),
                    total_cost_usd=float(raw["total_cost_usd"]),
                    mean_ttft_ms=float(raw["mean_ttft_ms"]),
                    p50_ms=float(raw["p50_ms"]),
                    p90_ms=float(raw["p90_ms"]),
                    p95_ms=p95_ms,
                    p99_ms=float(raw["p99_ms"]),
                    slo_ms=float(raw.get("slo_ms") or 3000.0),
                    slo_violation_rate=float(raw["slo_violation_rate"]),
                    hedge_rate=float(raw["hedge_rate"]),
                    tier_mix=json.loads(tier_mix_raw),
                    real_world_pool=raw.get("real_world_pool") or None,
                    api_provider_names=parse_json_list(raw.get("api_provider_names")),
                    api_latency_provider_names=parse_json_list(
                        raw.get("api_latency_provider_names")
                    ),
                    latency_profile=raw.get("latency_profile") or None,
                    quota_latency_profile_provider=(
                        raw.get("quota_latency_profile_provider") or None
                    ),
                    concurrency_latency_profile_provider=(
                        raw.get("concurrency_latency_profile_provider") or None
                    ),
                    subscription_plan=raw.get("subscription_plan") or None,
                    concurrency_plan=raw.get("concurrency_plan") or None,
                )
            )
    return rows


def routewise_rows(rows: list[Row], *, hedging: bool) -> list[Row]:
    selected = [row for row in rows if row.alpha is not None and row.hedging is hedging]
    return sorted(selected, key=lambda row: row.alpha or 0.0)


def frontier_points(rows: list[Row]) -> list[FrontierPoint]:
    return [
        FrontierPoint(
            policy=row.policy,
            label=policy_plot_label(
                row.policy,
                alpha=row.alpha,
                hedging=row.hedging,
            ),
            alpha=row.alpha,
            hedging=row.hedging,
            total_cost_usd=row.total_cost_usd,
            mean_ttft_ms=row.mean_ttft_ms,
            slo_violation_rate=row.slo_violation_rate,
            p99_ms=row.p99_ms,
            hedge_rate=row.hedge_rate,
        )
        for row in rows
    ]


def selected_frontier_points(
    rows: list[Row],
    baseline_policies: list[str] | None = None,
    routewise_policies: list[str] | None = None,
) -> list[FrontierPoint]:
    baselines = set(baseline_policies or BASELINE_ORDER)
    routewise = set(routewise_policies or ROUTEWISE_FIGURE_POLICIES)
    return [
        point
        for point in frontier_points(rows)
        if (point.alpha is None and point.policy in baselines)
        or point.policy in routewise
    ]


def plot_frontier(
    rows: list[Row],
    output_path: Path,
    baseline_policies: list[str] | None = None,
    routewise_policies: list[str] | None = None,
) -> None:
    mean_baselines = [
        policy for policy in (baseline_policies or list(BASELINE_ORDER)) if policy != "random"
    ]
    plot_mean_ttft_frontier(
        selected_frontier_points(rows, mean_baselines, routewise_policies),
        output_path,
    )


def plot_slo(
    rows: list[Row],
    output_path: Path,
    baseline_policies: list[str] | None = None,
    routewise_policies: list[str] | None = None,
) -> None:
    plot_slo_frontier(
        selected_frontier_points(rows, baseline_policies, routewise_policies),
        output_path,
    )


def plot_tier_mix(rows: list[Row], output_path: Path) -> None:
    selected = routewise_rows(rows, hedging=False)
    mix_rows = [
        MixRow(
            label="RouteWise-0.25"
            if row.alpha == 0.25
            else rf"RW $\alpha={row.alpha:g}$",
            shares=row.tier_mix,
        )
        for row in selected
    ]
    segments = [MixSegment(key, label, color) for key, label, color in TIER_MIX_SEGMENTS]
    plot_stacked_mix(mix_rows, segments, output_path, legend_ncols=3)


def provider_label(provider: str) -> str:
    if provider.endswith("_quota"):
        return r"$\mathcal{P}_Q$"
    if provider.endswith("_concurrency"):
        return r"$\mathcal{P}_C$"
    if provider.startswith("api_"):
        return provider.removeprefix("api_").replace("_", " ")
    return provider.replace("_", " ")


def provider_color(provider: str, idx: int) -> str:
    if provider.endswith("_quota"):
        return PROVIDER_MIX_COLORS.get("MiniMax_Plus_SQ", "#6b4c3b")
    if provider.endswith("_concurrency"):
        return PROVIDER_MIX_COLORS.get("Featherless_SC", "#59a14f")
    color_key = (
        f"OR_{provider.removeprefix('api_')}"
        if provider.startswith("api_")
        else provider
    )
    return PROVIDER_MIX_COLORS.get(
        color_key,
        PROVIDER_COLOR_CYCLE[idx % len(PROVIDER_COLOR_CYCLE)],
    )


def provider_tier(provider: str) -> str:
    if provider.endswith("_quota"):
        return "quota"
    if provider.endswith("_concurrency"):
        return "concurrency"
    return "api"


def provider_sort_key(provider: str, totals: Counter[str]) -> tuple[int, float, str]:
    tier_order = {"quota": 0, "concurrency": 1, "api": 2}
    return (tier_order.get(provider_tier(provider), 9), -totals[provider], provider)


def provider_mix_policy_rows(
    rows: list[Row],
    requested_policies: list[str] | None,
) -> list[Row]:
    by_policy = {row.policy: row for row in rows}
    policies = requested_policies or list(PROVIDER_MIX_POLICIES)
    missing = [policy for policy in policies if policy not in by_policy]
    if requested_policies is not None and missing:
        raise ValueError(f"missing provider-mix policies: {missing}")
    selected = [by_policy[policy] for policy in policies if policy in by_policy]
    if not selected:
        raise ValueError("no rows available for provider-mix plot")
    return selected


def plot_provider_mix(
    rows: list[Row],
    output_path: Path,
    requested_policies: list[str] | None = None,
) -> None:
    selected = provider_mix_policy_rows(rows, requested_policies)
    totals: Counter[str] = Counter()
    for row in selected:
        for provider, share in row.provider_mix.items():
            if share > 0.0:
                totals[provider] += share

    providers = sorted(totals, key=lambda provider: provider_sort_key(provider, totals))
    segments = [
        MixSegment(
            key=provider,
            label=provider_label(provider),
            color=provider_color(provider, idx),
        )
        for idx, provider in enumerate(providers)
    ]
    mix_rows = [
        MixRow(
            label=policy_plot_label(row.policy, alpha=row.alpha, hedging=row.hedging),
            shares={provider: row.provider_mix.get(provider, 0.0) for provider in providers},
        )
        for row in selected
    ]
    plot_stacked_mix(
        mix_rows,
        segments,
        output_path,
        legend_ncols=5,
        legend_fontsize=8.2,
        font_size=10.8,
        label_fontsize=11.5,
        tick_fontsize=9.8,
        margins=(0.43, 0.98, 0.18, 0.80),
        show_legend=True,
        x_max=102.0,
    )


def provider_latency_means(row: Row) -> dict[str, float]:
    latencies: dict[str, float] = {}
    if row.real_world_pool:
        pool = load_provider_pool(row.real_world_pool)
        pool_by_name = pool.by_name()
        for provider, latency_provider in zip(
            row.api_provider_names,
            row.api_latency_provider_names,
            strict=False,
        ):
            entry = pool_by_name.get(latency_provider)
            if entry is not None:
                latencies[provider] = entry.ttft_dist.mean()

    if row.latency_profile and row.quota_latency_profile_provider and row.subscription_plan:
        dist = load_empirical_distribution(
            row.latency_profile,
            row.quota_latency_profile_provider,
        )
        latencies[f"{row.subscription_plan}_quota"] = dist.mean()

    if (
        row.latency_profile
        and row.concurrency_latency_profile_provider
        and row.concurrency_plan
    ):
        dist = load_empirical_distribution(
            row.latency_profile,
            row.concurrency_latency_profile_provider,
        )
        latencies[f"{row.concurrency_plan}_concurrency"] = dist.mean()

    return latencies


def plot_provider_latency(
    rows: list[Row],
    output_path: Path,
    requested_policies: list[str] | None = None,
) -> None:
    selected = provider_mix_policy_rows(rows, requested_policies)
    totals: Counter[str] = Counter()
    for row in selected:
        for provider, share in row.provider_mix.items():
            if share > 0.0:
                totals[provider] += share

    means = provider_latency_means(selected[0])
    items = [
        (provider, mean_ms)
        for provider, mean_ms in means.items()
        if math.isfinite(mean_ms)
    ]
    if not items:
        raise ValueError("no provider latency means available for selected provider mix")
    items = sorted(items, key=lambda item: item[1])

    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "figure.figsize": MIX_FIGSIZE,
            "savefig.pad_inches": 0.01,
        }
    )
    labels = [provider_label(provider) for provider, _ in items]
    values = [mean_ms / 1000.0 for _, mean_ms in items]
    colors = [provider_color(provider, idx) for idx, (provider, _) in enumerate(items)]

    fig, ax = plt.subplots(figsize=MIX_FIGSIZE, constrained_layout=False)
    y = list(range(len(items)))
    ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.35, height=0.72)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean TTFT (s)")
    ax.grid(axis="x", color="#9a9a9a", alpha=0.28, linewidth=0.5)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.29, right=0.99, bottom=0.16, top=0.97)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(output_path)
    plt.close(fig)


def plot_hedging_p99(rows: list[Row], output_path: Path) -> None:
    plot_common_hedging_p99(
        frontier_points(rows),
        output_path,
        alphas=ROUTEWISE_TABLE_ALPHAS,
    )


def cdf_label(row: Row) -> str:
    return policy_plot_label(row.policy, alpha=row.alpha, hedging=row.hedging)


def plot_ttft_cdf(
    rows: list[Row],
    histograms: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    by_policy = {row.policy: row for row in rows}
    p99_values = [
        by_policy[policy].p99_ms
        for policy in CDF_POLICIES
        if policy in by_policy and policy != "random"
    ]
    x_max_sec = max(12.0, (max(p99_values) / 1000.0 * 1.05) if p99_values else 12.0)
    x_ms = [x_max_sec * 1000.0 * idx / 299.0 for idx in range(300)]
    series: list[CdfSeries] = []
    for policy in CDF_POLICIES:
        histogram = histograms.get(policy)
        if histogram is None or policy not in by_policy:
            continue
        y = [histogram_cdf(histogram, value) for value in x_ms]
        row = by_policy[policy]
        series.append(
            CdfSeries(
                policy=policy,
                label=cdf_label(row),
                x_ms=x_ms,
                y_values=y,
                p99_ms=row.p99_ms,
                alpha=row.alpha,
                total_cost_usd=row.total_cost_usd,
                include_in_axis=policy != "random",
            )
        )
    plot_common_ttft_cdf(series, output_path, slo_sec=3.0, x_max_sec=x_max_sec)


def histogram_samples(histogram: dict[str, object]) -> list[float]:
    edges = [float(value) for value in histogram["bin_edges_ms"]]
    counts = [int(value) for value in histogram["counts"]]
    min_ms = float(histogram["min_ms"])
    max_ms = float(histogram["max_ms"])
    samples: list[float] = []
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        if idx == 0:
            value = min_ms
        elif idx == len(counts) - 1:
            value = max_ms
        else:
            value = 0.5 * (edges[idx - 1] + edges[idx])
        samples.extend([value] * count)
    return samples


def plot_ttft_boxplot(
    rows: list[Row],
    histograms: dict[str, dict[str, object]],
    output_path: Path,
    requested_policies: list[str] | None = None,
) -> None:
    by_policy = {row.policy: row for row in rows}
    policies = requested_policies or list(BOXPLOT_POLICIES)
    missing = [policy for policy in policies if policy not in by_policy]
    if requested_policies is not None and missing:
        raise ValueError(f"missing boxplot policies: {missing}")
    series: list[BoxSeries] = []
    for policy in policies:
        row = by_policy.get(policy)
        histogram = histograms.get(policy)
        if row is None or histogram is None:
            continue
        samples = histogram_samples(histogram)
        if not samples:
            continue
        series.append(
            BoxSeries(
                policy=policy,
                label=cdf_label(row),
                samples_ms=samples,
                alpha=row.alpha,
                total_cost_usd=row.total_cost_usd,
            )
        )
    if not series:
        raise ValueError("no histogram-backed series available for boxplot")
    slo_sec = rows[0].slo_ms / 1000.0 if rows else 3.0
    plot_common_ttft_boxplot(series, output_path, slo_sec=slo_sec)


def write_table(rows: list[Row], output_path: Path) -> None:
    by_policy = {row.policy: row for row in rows}
    lines: list[str] = []
    for policy in TABLE_POLICIES:
        row = by_policy[policy]
        lines.append(_format_table_row(row))
    lines.append(r"\midrule")
    for alpha in ROUTEWISE_TABLE_ALPHAS:
        policy = f"ablation_lp_only_p{int(alpha * 100)}"
        row = by_policy[policy]
        lines.append(_format_table_row(row))
    lines.append(r"\midrule")
    for alpha in ROUTEWISE_TABLE_ALPHAS:
        policy = f"ablation_lp_hedging_p{int(alpha * 100)}"
        row = by_policy[policy]
        lines.append(_format_table_row(row))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_table_row(row: Row, *, label: str | None = None) -> str:
    hedge = "--" if row.hedge_rate == 0 else f"{row.hedge_rate * 100.0:.1f}\\%"
    return (
        f"{label or row.label} & "
        f"{row.total_cost_usd:.2f} & "
        f"{row.mean_ttft_ms / 1000.0:.2f} & "
        f"{row.p50_ms / 1000.0:.2f} & "
        f"{row.p90_ms / 1000.0:.2f} & "
        f"{row.p95_ms / 1000.0:.2f} & "
        f"{row.p99_ms / 1000.0:.2f} & "
        f"{row.slo_violation_rate * 100.0:.2f}\\% & "
        f"{hedge} \\\\"
    )


def write_summary(rows: list[Row], output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = (
        "policy",
        "label",
        "alpha",
        "hedging",
        "total_cost_usd",
        "mean_ttft_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "slo_violation_rate",
        "hedge_rate",
        "tier_mix",
    )
    payload = [
        {field: getattr(row, field) for field in summary_fields}
        for row in rows
    ]
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    histograms = load_histograms(args.histograms_json)
    rows = load_rows(args.summary_csv, histograms)
    plot_frontier(
        rows,
        args.frontier_out,
        args.frontier_baselines,
        args.routewise_frontier_policies,
    )
    plot_slo(
        rows,
        args.slo_out,
        args.frontier_baselines,
        args.routewise_frontier_policies,
    )
    if args.tier_out is not None:
        plot_tier_mix(rows, args.tier_out)
    if args.provider_latency_out is not None:
        plot_provider_latency(
            rows,
            args.provider_latency_out,
            args.provider_mix_policies,
        )
    if args.provider_mix_out is not None:
        plot_provider_mix(
            rows,
            args.provider_mix_out,
            args.provider_mix_policies,
        )
    if args.p99_bar_out is not None:
        plot_hedging_p99(rows, args.p99_bar_out)
    if args.cdf_out is not None:
        if not histograms:
            raise SystemExit("--cdf-out requires --histograms-json")
        plot_ttft_cdf(rows, histograms, args.cdf_out)
    if args.boxplot_out is not None:
        if not histograms:
            raise SystemExit("--boxplot-out requires --histograms-json")
        plot_ttft_boxplot(rows, histograms, args.boxplot_out, args.boxplot_policies)
    write_table(rows, args.table_out)
    write_summary(rows, args.summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
