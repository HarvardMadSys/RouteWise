"""Compare FreeInference prefix-cache pricing against no-cache pricing.

The two inputs should be ``experiments.simulation.end_to_end`` summaries from
the same workload and provider setup. The script emits a paper-ready LaTeX row
file and, optionally, a single cost-comparison PDF.

Example:

    uv run python -m plots.end_to_end.plot_cache_comparison \
      --cache-summary-csv outputs/simulation/end_to_end_rw8_freeinference_1week_q1_c1_slo3s_real_cache_price_20260514/summary.csv \
      --no-cache-summary-csv outputs/simulation/end_to_end_rw8_freeinference_1week_q1_c1_slo3s_no_cache_price_20260514/summary.csv \
      --table-out ../paper/tables/freeinference_cache_comparison_rows.tex \
      --summary-out ../paper/tables/freeinference_cache_comparison_summary.json \
      --cost-figure-out ../paper/figures/freeinference_cache_cost_comparison.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from plots.style import apply_style

POLICIES = (
    "greedy_cost",
    "greedy_latency",
    "random",
    "ablation_lp_only_p0",
    "ablation_lp_only_p25",
    "ablation_lp_only_p50",
    "ablation_lp_only_p75",
    "ablation_lp_only_p100",
    "ablation_lp_hedging_p0",
    "ablation_lp_hedging_p25",
    "ablation_lp_hedging_p50",
    "ablation_lp_hedging_p75",
    "ablation_lp_hedging_p100",
)

POLICY_LABELS = {
    "greedy_cost": "Greedy-cost",
    "greedy_latency": "Greedy-latency",
    "random": "Random",
    "ablation_lp_only_p0": r"\sysname{} ($\alpha=0$)",
    "ablation_lp_only_p25": "RouteWise-0.25",
    "ablation_lp_only_p50": r"\sysname{} ($\alpha=0.5$)",
    "ablation_lp_only_p75": r"\sysname{} ($\alpha=0.75$)",
    "ablation_lp_only_p100": r"\sysname{} ($\alpha=1$)",
    "ablation_lp_hedging_p0": r"\sysname{} ($\alpha=0$ + hedge)",
    "ablation_lp_hedging_p25": "RouteWise-0.25 + hedge",
    "ablation_lp_hedging_p50": r"\sysname{} ($\alpha=0.5$ + hedge)",
    "ablation_lp_hedging_p75": r"\sysname{} ($\alpha=0.75$ + hedge)",
    "ablation_lp_hedging_p100": r"\sysname{} ($\alpha=1$ + hedge)",
}

PLOT_POLICIES = (
    "greedy_cost",
    "greedy_latency",
    "ablation_lp_only_p25",
    "ablation_lp_only_p50",
    "ablation_lp_hedging_p50",
    "ablation_lp_hedging_p75",
)

CACHE_COLOR = "#2f6f73"
NO_CACHE_COLOR = "#9b5a32"
FIGSIZE = (3.25, 2.3)


@dataclass(frozen=True)
class PolicyMetrics:
    policy: str
    total_cost_usd: float
    mean_ttft_ms: float
    p99_ms: float
    slo_violation_rate: float
    api_cost_usd: float
    subscription_fixed_cost_usd: float


@dataclass(frozen=True)
class Comparison:
    policy: str
    label: str
    cache_cost_usd: float
    no_cache_cost_usd: float
    cost_reduction_pct: float
    cache_slo_pct: float
    no_cache_slo_pct: float
    cache_p99_s: float
    no_cache_p99_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-summary-csv", type=Path, required=True)
    parser.add_argument("--no-cache-summary-csv", type=Path, required=True)
    parser.add_argument("--table-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--cost-figure-out", type=Path, default=None)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, PolicyMetrics]:
    rows: dict[str, PolicyMetrics] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            policy = raw["policy"]
            rows[policy] = PolicyMetrics(
                policy=policy,
                total_cost_usd=float(raw["total_cost_usd"]),
                mean_ttft_ms=float(raw["mean_ttft_ms"]),
                p99_ms=float(raw["p99_ms"]),
                slo_violation_rate=float(raw["slo_violation_rate"]),
                api_cost_usd=float(raw["api_cost_usd"]),
                subscription_fixed_cost_usd=float(raw["subscription_fixed_cost_usd"]),
            )
    return rows


def compare(
    cache_rows: dict[str, PolicyMetrics],
    no_cache_rows: dict[str, PolicyMetrics],
) -> list[Comparison]:
    comparisons: list[Comparison] = []
    for policy in POLICIES:
        cache = cache_rows[policy]
        no_cache = no_cache_rows[policy]
        reduction = (no_cache.total_cost_usd - cache.total_cost_usd) / no_cache.total_cost_usd
        comparisons.append(
            Comparison(
                policy=policy,
                label=POLICY_LABELS[policy],
                cache_cost_usd=cache.total_cost_usd,
                no_cache_cost_usd=no_cache.total_cost_usd,
                cost_reduction_pct=reduction * 100.0,
                cache_slo_pct=cache.slo_violation_rate * 100.0,
                no_cache_slo_pct=no_cache.slo_violation_rate * 100.0,
                cache_p99_s=cache.p99_ms / 1000.0,
                no_cache_p99_s=no_cache.p99_ms / 1000.0,
            )
        )
    return comparisons


def write_table(comparisons: list[Comparison], output_path: Path) -> None:
    lines = [_format_row(item) for item in comparisons]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_row(item: Comparison) -> str:
    return (
        f"{item.label} & "
        f"{item.cache_cost_usd:.2f} & "
        f"{item.no_cache_cost_usd:.2f} & "
        f"{item.cost_reduction_pct:+.1f}\\% & "
        f"{item.cache_slo_pct:.2f}\\% & "
        f"{item.no_cache_slo_pct:.2f}\\% & "
        f"{item.cache_p99_s:.2f} & "
        f"{item.no_cache_p99_s:.2f} \\\\"
    )


def write_summary(comparisons: list[Comparison], output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(item) for item in comparisons], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def apply_column_figure_style() -> None:
    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "legend.fontsize": 6.5,
            "savefig.pad_inches": 0.01,
        }
    )


def plot_costs(comparisons: list[Comparison], output_path: Path | None) -> None:
    if output_path is None:
        return
    apply_column_figure_style()
    selected = [item for item in comparisons if item.policy in PLOT_POLICIES]
    labels = [
        item.label.replace(r"\sysname{}", "RW")
        .replace(r"$\alpha=", "")
        .replace("$", "")
        .replace(" + hedge", "+H")
        for item in selected
    ]
    x = list(range(len(selected)))
    width = 0.38

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(
        [value - width / 2 for value in x],
        [item.cache_cost_usd for item in selected],
        width=width,
        color=CACHE_COLOR,
        label="cache",
    )
    ax.bar(
        [value + width / 2 for value in x],
        [item.no_cache_cost_usd for item in selected],
        width=width,
        color=NO_CACHE_COLOR,
        label="no cache",
    )
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Total cost ($)")
    ax.legend(frameon=False, loc="upper left")
    ax.margins(x=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cache_rows = load_summary(args.cache_summary_csv)
    no_cache_rows = load_summary(args.no_cache_summary_csv)
    comparisons = compare(cache_rows, no_cache_rows)
    write_table(comparisons, args.table_out)
    write_summary(comparisons, args.summary_out)
    plot_costs(comparisons, args.cost_figure_out)


if __name__ == "__main__":
    main()
