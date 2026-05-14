"""Plot real-world end-to-end frontier results for the paper.

Input is a real-eval output directory with one subdirectory per policy and a
``requests.csv`` file inside each policy directory. The script aggregates the
raw per-request logs into paper metrics and emits:

* one single-panel cost vs. mean TTFT figure;
* one single-panel cost vs. SLO-violation figure; and
* a LaTeX table-row fragment consumed by ``paper/sections/05-evaluation.tex``.

Example:

    .venv/bin/python -m plots.end_to_end.plot_real_world_frontier \
      --input-dir outputs/real_eval/real_eval_minimax_m25_burstgpt_cap10s_rwp_sweep_20260514_001724_minimax_m25_burstgpt_cap10s_rwp_sweep_slo3s \
      --mean-ttft-out ../paper/figures/real_world_partial_mean_ttft_placeholder.pdf \
      --slo-out ../paper/figures/real_world_partial_slo_placeholder.pdf \
      --table-out ../paper/tables/real_world_frontier_placeholder_rows.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from plots.palettes import ONLINE_POLICY_COLORS, ROUTER_STRATEGY_COLORS
from plots.style import apply_style


ROUTEWISE_PREFIX = "budget_range_p"
ROUTEWISE_HEDGE_SUFFIX = "_hedge"
ROUTEWISE_COLOR = "#2f6f73"
BASELINE_ORDER = (
    "greedy_cost",
    "greedy_latency",
    "or_auto",
    "or_sort_cost",
    "or_sort_latency",
)
POLICY_LABELS = {
    "greedy_cost": "Greedy-cost",
    "greedy_latency": "Greedy-latency",
    "random": "Random",
    "or_auto": "OpenRouter auto",
    "or_sort_cost": "OpenRouter sort=price",
    "or_sort_latency": "OpenRouter sort=latency",
}
PLOT_LABELS = {
    "greedy_cost": "Greedy-cost",
    "greedy_latency": "Greedy-latency",
    "random": "Random",
    "or_auto": "OR auto",
    "or_sort_cost": "OR price",
    "or_sort_latency": "OR latency",
}
POLICY_COLORS = {
    "greedy_cost": ROUTER_STRATEGY_COLORS.get("greedy_cost", "#4c78a8"),
    "greedy_latency": ROUTER_STRATEGY_COLORS.get("greedy_latency", "#f58518"),
    "random": ROUTER_STRATEGY_COLORS.get("random", "#7f7f7f"),
    "or_auto": ONLINE_POLICY_COLORS.get("openrouter_auto", "#b279a2"),
    "or_sort_cost": ONLINE_POLICY_COLORS.get("sort_price", "#e45756"),
    "or_sort_latency": ONLINE_POLICY_COLORS.get("sort_latency", "#72b7b2"),
}
ROUTEWISE_LABEL_OFFSETS = {
    0.0: (4, -8),
    0.25: (4, 3),
    0.5: (-8, 8),
    0.75: (4, -7),
    1.0: (4, 4),
}
ROUTEWISE_LABEL_OFFSETS_BY_METRIC = {
    "slo_violation_rate": {
        0.0: (7, 10),
        0.25: (7, 12),
        0.5: (-30, -16),
        0.75: (7, 18),
        1.0: (7, 12),
    },
    "ttft_mean_ms": {
        0.0: (7, 10),
        0.25: (7, 9),
        0.5: (-30, -16),
        0.75: (7, 15),
        1.0: (7, 8),
    },
}
BASELINE_LABEL_OFFSET = (4, 3)


@dataclass(frozen=True)
class PolicySummary:
    policy: str
    label: str
    alpha: float | None
    n: int
    span_hours: float
    total_cost_usd: float
    billed_cost_usd: float
    fixed_cost_usd: float
    physical_cost_usd: float
    mean_cost_usd: float
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    e2e_mean_ms: float
    e2e_p99_ms: float
    slo_violation_rate: float
    success_rate: float
    rate_limited: int
    hedge_rate: float
    hedge_backup_win_rate: float
    tier_mix: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Real-eval run directory containing policy subdirectories.",
    )
    parser.add_argument(
        "--mean-ttft-out",
        type=Path,
        default=Path("../paper/figures/real_world_partial_mean_ttft_placeholder.pdf"),
        help="Output path for the cost-vs-mean-TTFT PDF figure.",
    )
    parser.add_argument(
        "--slo-out",
        type=Path,
        default=Path("../paper/figures/real_world_partial_slo_placeholder.pdf"),
        help="Output path for the cost-vs-SLO-violation PDF figure.",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=Path("../paper/tables/real_world_frontier_placeholder_rows.tex"),
        help="Output path for the LaTeX table-row fragment.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional output path for machine-readable aggregated metrics.",
    )
    parser.add_argument("--slo-ms", type=float, default=3000.0)
    parser.add_argument(
        "--billing-duration-sec",
        type=float,
        default=28800.0,
        help="Billing horizon used to prorate subscription fixed costs.",
    )
    parser.add_argument(
        "--minimax-fixed-cost-usd",
        type=float,
        default=0.2381,
        help="MiniMax Plus prorated fixed cost for one policy run.",
    )
    parser.add_argument(
        "--featherless-monthly-cost-usd",
        type=float,
        default=25.0,
        help="Featherless monthly fixed cost used for prorating.",
    )
    parser.add_argument(
        "--fixed-cost-non-or",
        type=float,
        default=None,
        help=(
            "Override fixed cost added to non-OpenRouter policies. If omitted, "
            "the script uses MiniMax fixed cost plus prorated Featherless cost."
        ),
    )
    parser.add_argument(
        "--include-random",
        action="store_true",
        help="Include the random baseline in the paper table and figure.",
    )
    parser.add_argument(
        "--x-label",
        default="Amortized total cost ($)",
        help="X-axis label for generated frontier figures.",
    )
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    pos = (len(values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[int(pos)]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def truthy(value: str | None) -> bool:
    return value in {"1", "true", "True", "yes", "YES"}


def parse_alpha(policy: str) -> float | None:
    if not policy.startswith(ROUTEWISE_PREFIX):
        return None
    raw = policy[len(ROUTEWISE_PREFIX) :]
    if raw.endswith(ROUTEWISE_HEDGE_SUFFIX):
        raw = raw[: -len(ROUTEWISE_HEDGE_SUFFIX)]
    if raw == "100":
        return 1.0
    try:
        return float(raw) / 100.0
    except ValueError:
        return None


def table_label(policy: str, alpha: float | None) -> str:
    if alpha is not None:
        hedge_suffix = " + hedge" if policy.endswith(ROUTEWISE_HEDGE_SUFFIX) else ""
        return rf"\sysname{{}} ($\alpha={alpha:g}${hedge_suffix})"
    return POLICY_LABELS.get(policy, policy.replace("_", r"\_"))


def fixed_cost_for_policy(policy: str, args: argparse.Namespace) -> float:
    if policy.startswith("or_"):
        return 0.0
    if args.fixed_cost_non_or is not None:
        return args.fixed_cost_non_or
    featherless = (
        args.featherless_monthly_cost_usd
        * (args.billing_duration_sec / 86400.0)
        / 30.0
    )
    return args.minimax_fixed_cost_usd + featherless


def read_summary(policy_dir: Path, args: argparse.Namespace) -> PolicySummary:
    policy = policy_dir.name
    alpha = parse_alpha(policy)
    rows = list(csv.DictReader((policy_dir / "requests.csv").open(newline="")))
    if not rows:
        raise ValueError(f"{policy_dir}: requests.csv has no rows")

    timestamps = [float(row["ts"]) for row in rows if row.get("ts")]
    successes = [
        row
        for row in rows
        if row.get("status") == "success"
        and row.get("ttft_ms") not in {"", None}
        and float(row["ttft_ms"]) >= 0.0
    ]
    ttft = [float(row["ttft_ms"]) for row in successes]
    e2e = [
        float(row["e2e_ms"])
        for row in successes
        if row.get("e2e_ms") not in {"", None}
    ]

    billed_cost = sum(float(row.get("billed_cost_usd") or 0.0) for row in rows)
    physical_cost = sum(float(row.get("physical_cost_usd") or 0.0) for row in rows)
    fixed_cost = fixed_cost_for_policy(policy, args)
    total_cost = billed_cost + fixed_cost
    slo_violations = sum(
        1
        for row in rows
        if row.get("status") != "success"
        or (
            row.get("ttft_ms") not in {"", None}
            and float(row["ttft_ms"]) > args.slo_ms
        )
    )
    hedge_count = sum(1 for row in rows if truthy(row.get("hedge_triggered")))
    backup_wins = sum(1 for row in rows if row.get("hedge_winner") == "backup")
    tier_counts: dict[str, int] = {}
    for row in rows:
        tier = row.get("tier") or "openrouter"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    return PolicySummary(
        policy=policy,
        label=table_label(policy, alpha),
        alpha=alpha,
        n=len(rows),
        span_hours=(max(timestamps) - min(timestamps)) / 3600.0 if timestamps else 0.0,
        total_cost_usd=total_cost,
        billed_cost_usd=billed_cost,
        fixed_cost_usd=fixed_cost,
        physical_cost_usd=physical_cost,
        mean_cost_usd=total_cost / len(rows),
        ttft_mean_ms=mean(ttft) if ttft else float("nan"),
        ttft_p50_ms=percentile(ttft, 50.0),
        ttft_p90_ms=percentile(ttft, 90.0),
        ttft_p99_ms=percentile(ttft, 99.0),
        e2e_mean_ms=mean(e2e) if e2e else float("nan"),
        e2e_p99_ms=percentile(e2e, 99.0),
        slo_violation_rate=slo_violations / len(rows),
        success_rate=len(successes) / len(rows),
        rate_limited=sum(1 for row in rows if truthy(row.get("rate_limited"))),
        hedge_rate=hedge_count / len(rows),
        hedge_backup_win_rate=backup_wins / hedge_count if hedge_count else 0.0,
        tier_mix={tier: count / len(rows) for tier, count in sorted(tier_counts.items())},
    )


def collect_summaries(args: argparse.Namespace) -> list[PolicySummary]:
    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)
    summaries = []
    for policy_dir in sorted(path for path in args.input_dir.iterdir() if path.is_dir()):
        if not (policy_dir / "requests.csv").exists():
            continue
        if policy_dir.name == "random" and not args.include_random:
            continue
        summaries.append(read_summary(policy_dir, args))
    if not summaries:
        raise ValueError(f"{args.input_dir}: found no requests.csv files")
    return sorted(summaries, key=sort_key)


def sort_key(summary: PolicySummary) -> tuple[int, float]:
    if summary.policy == "greedy_cost":
        return (0, 0.0)
    if summary.policy == "greedy_latency":
        return (0, 1.0)
    if summary.alpha is not None:
        return (1, summary.alpha)
    baseline_rank = (
        float(BASELINE_ORDER.index(summary.policy))
        if summary.policy in BASELINE_ORDER
        else 99.0
    )
    return (2, baseline_rank)


def write_table_rows(
    summaries: list[PolicySummary],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in summaries:
        lines.append(
            f"{s.label} & "
            f"{s.total_cost_usd:.3f} & "
            f"{s.ttft_mean_ms / 1000.0:.2f} & "
            f"{s.ttft_p99_ms / 1000.0:.2f} & "
            f"{100.0 * s.slo_violation_rate:.2f}\\% & "
            f"{100.0 * s.success_rate:.2f}\\% \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_value(summary: PolicySummary, attr: str) -> float:
    value = getattr(summary, attr)
    if attr.endswith("_ms"):
        return value / 1000.0
    if attr.endswith("_rate"):
        return value * 100.0
    return value


def annotate(ax: plt.Axes, summary: PolicySummary, attr: str, *, color: str) -> None:
    label = (
        rf"$\alpha={summary.alpha:g}$"
        if summary.alpha is not None
        else PLOT_LABELS.get(summary.policy, summary.policy)
    )
    offset = BASELINE_LABEL_OFFSET
    if summary.alpha is not None:
        metric_offsets = ROUTEWISE_LABEL_OFFSETS_BY_METRIC.get(attr, {})
        offset = metric_offsets.get(summary.alpha, ROUTEWISE_LABEL_OFFSETS.get(summary.alpha, (4, 3)))
    ax.annotate(
        label,
        xy=(summary.total_cost_usd, metric_value(summary, attr)),
        xytext=offset,
        textcoords="offset points",
        fontsize=6.2,
        color=color,
        ha="right" if offset[0] < 0 else "left",
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85},
        clip_on=False,
    )


def pad_axes(ax: plt.Axes) -> None:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_pad = max((x_max - x_min) * 0.08, 0.006)
    y_pad = max((y_max - y_min) * 0.12, 0.2)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)


def plot_metric_frontier(
    summaries: list[PolicySummary],
    path: Path,
    *,
    attr: str,
    ylabel: str,
    xlabel: str,
    title: str | None = None,
) -> None:
    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.figsize": (3.35, 2.25),
        }
    )

    routewise = [s for s in summaries if s.alpha is not None]
    baselines = [s for s in summaries if s.alpha is None]
    fig, ax = plt.subplots(figsize=(3.35, 2.25), constrained_layout=True)

    if routewise:
        has_hedging = any(s.policy.endswith(ROUTEWISE_HEDGE_SUFFIX) for s in routewise)
        ax.plot(
            [s.total_cost_usd for s in routewise],
            [metric_value(s, attr) for s in routewise],
            color=ROUTEWISE_COLOR,
            marker="o",
            linewidth=1.4,
            markersize=4.2,
            label="RouteWise + hedge" if has_hedging else "RouteWise",
        )
        for summary in routewise:
            annotate(ax, summary, attr, color=ROUTEWISE_COLOR)
    for summary in baselines:
        color = POLICY_COLORS.get(summary.policy, "#555555")
        ax.scatter(
            summary.total_cost_usd,
            metric_value(summary, attr),
            marker="s",
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        annotate(ax, summary, attr, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    pad_axes(ax)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_summary(summaries: list[PolicySummary], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    summaries = collect_summaries(args)
    plot_metric_frontier(
        summaries,
        args.mean_ttft_out,
        attr="ttft_mean_ms",
        ylabel="Mean TTFT (s)",
        xlabel=args.x_label,
    )
    plot_metric_frontier(
        summaries,
        args.slo_out,
        attr="slo_violation_rate",
        ylabel="SLO violations (%)",
        xlabel=args.x_label,
    )
    write_table_rows(summaries, args.table_out)
    write_summary(summaries, args.summary_out)
    print(f"wrote {args.mean_ttft_out}")
    print(f"wrote {args.slo_out}")
    print(f"wrote {args.table_out}")
    if args.summary_out is not None:
        print(f"wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
