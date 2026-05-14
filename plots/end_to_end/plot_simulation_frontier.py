"""Plot end-to-end simulator frontier results for the paper.

The input is ``experiments.simulation.end_to_end``'s ``summary.csv``. The
script emits one metric per PDF so LaTeX can resize and arrange figures without
regenerating multi-panel plots.

Example:

    uv run python -m plots.end_to_end.plot_simulation_frontier \
      --summary-csv outputs/simulation/end_to_end_rw8_minimax_q1_c1_slo3_20260514/summary.csv \
      --frontier-out ../paper/figures/stage3_e2e_cost_latency_frontier.pdf \
      --slo-out ../paper/figures/stage3_e2e_slo_cost_frontier.pdf \
      --tier-out ../paper/figures/stage3_e2e_tier_mix.pdf \
      --table-out ../paper/tables/simulation_e2e_q1_c1_slo3_rows.tex \
      --summary-out ../paper/tables/simulation_e2e_q1_c1_slo3_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from plots.palettes import ROUTER_STRATEGY_COLORS, TIER_COLORS
from plots.style import apply_style


POLICY_LABELS = {
    "greedy_cost": "Greedy-cost",
    "or_sort_cost": "OR price",
    "random": "Random",
    "greedy_latency": "Greedy-latency",
    "or_sort_latency": "OR latency",
}
BASELINE_ORDER = (
    "greedy_cost",
    "or_sort_cost",
    "random",
    "greedy_latency",
    "or_sort_latency",
)
TABLE_POLICIES = (
    "greedy_cost",
    "or_sort_cost",
    "random",
    "greedy_latency",
    "or_sort_latency",
    "ablation_lp_only_p0",
    "ablation_lp_only_p25",
    "ablation_lp_only_p50",
    "ablation_lp_only_p75",
    "ablation_lp_hedging_p50",
    "ablation_lp_hedging_p75",
)
ROUTEWISE_LINE_COLOR = "#2f6f73"
ROUTEWISE_HEDGE_COLOR = ROUTER_STRATEGY_COLORS.get("ablation_lp_hedging", "#ff7f0e")


@dataclass(frozen=True)
class Row:
    policy: str
    label: str
    alpha: float | None
    hedging: bool
    total_cost_usd: float
    mean_ttft_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    slo_violation_rate: float
    hedge_rate: float
    tier_mix: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--frontier-out", type=Path, required=True)
    parser.add_argument("--slo-out", type=Path, required=True)
    parser.add_argument("--tier-out", type=Path, required=True)
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
    return rf"\sysname{{}} ($\alpha={alpha:g}${suffix})"


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            policy = raw["policy"]
            alpha = parse_alpha(policy)
            hedging = raw.get("hedging_enabled") == "True"
            tier_mix_raw = raw.get("tier_mix") or "{}"
            rows.append(
                Row(
                    policy=policy,
                    label=row_label(policy, alpha, hedging),
                    alpha=alpha,
                    hedging=hedging,
                    total_cost_usd=float(raw["total_cost_usd"]),
                    mean_ttft_ms=float(raw["mean_ttft_ms"]),
                    p50_ms=float(raw["p50_ms"]),
                    p90_ms=float(raw["p90_ms"]),
                    p99_ms=float(raw["p99_ms"]),
                    slo_violation_rate=float(raw["slo_violation_rate"]),
                    hedge_rate=float(raw["hedge_rate"]),
                    tier_mix=json.loads(tier_mix_raw),
                )
            )
    return rows


def routewise_rows(rows: list[Row], *, hedging: bool) -> list[Row]:
    selected = [
        row
        for row in rows
        if row.alpha is not None and row.hedging is hedging
    ]
    return sorted(selected, key=lambda row: row.alpha or 0.0)


def baseline_rows(rows: list[Row]) -> list[Row]:
    by_policy = {row.policy: row for row in rows}
    return [by_policy[name] for name in BASELINE_ORDER if name in by_policy]


def annotate_points(ax, rows: list[Row], *, y_value) -> None:
    for row in rows:
        ax.annotate(
            row.label.replace(r"\sysname{}", "RW"),
            (row.total_cost_usd, y_value(row)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )


def plot_frontier(rows: list[Row], output_path: Path) -> None:
    apply_style("paper")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    no_hedge = routewise_rows(rows, hedging=False)
    hedged = routewise_rows(rows, hedging=True)
    baselines = baseline_rows(rows)

    ax.plot(
        [row.total_cost_usd for row in no_hedge],
        [row.mean_ttft_ms / 1000.0 for row in no_hedge],
        marker="o",
        color=ROUTEWISE_LINE_COLOR,
        label="RW",
    )
    ax.plot(
        [row.total_cost_usd for row in hedged],
        [row.mean_ttft_ms / 1000.0 for row in hedged],
        marker="s",
        color=ROUTEWISE_HEDGE_COLOR,
        label="RW + hedge",
    )
    for row in baselines:
        ax.scatter(
            row.total_cost_usd,
            row.mean_ttft_ms / 1000.0,
            marker="x",
            s=55,
            color=ROUTER_STRATEGY_COLORS.get(row.policy, "#555555"),
        )
    annotate_points(ax, baselines, y_value=lambda row: row.mean_ttft_ms / 1000.0)
    ax.set_xlabel("Total cost ($)")
    ax.set_ylabel("Mean TTFT (s)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_slo(rows: list[Row], output_path: Path) -> None:
    apply_style("paper")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    no_hedge = routewise_rows(rows, hedging=False)
    hedged = routewise_rows(rows, hedging=True)
    baselines = baseline_rows(rows)

    ax.plot(
        [row.total_cost_usd for row in no_hedge],
        [row.slo_violation_rate * 100.0 for row in no_hedge],
        marker="o",
        color=ROUTEWISE_LINE_COLOR,
        label="RW",
    )
    ax.plot(
        [row.total_cost_usd for row in hedged],
        [row.slo_violation_rate * 100.0 for row in hedged],
        marker="s",
        color=ROUTEWISE_HEDGE_COLOR,
        label="RW + hedge",
    )
    for row in baselines:
        ax.scatter(
            row.total_cost_usd,
            row.slo_violation_rate * 100.0,
            marker="x",
            s=55,
            color=ROUTER_STRATEGY_COLORS.get(row.policy, "#555555"),
        )
    annotate_points(ax, baselines, y_value=lambda row: row.slo_violation_rate * 100.0)
    ax.set_xlabel("Total cost ($)")
    ax.set_ylabel("SLO violation (%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_tier_mix(rows: list[Row], output_path: Path) -> None:
    apply_style("paper")
    selected = routewise_rows(rows, hedging=False)
    labels = [rf"$\alpha={row.alpha:g}$" for row in selected]
    tiers = ("quota", "concurrency", "api")

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    bottom = [0.0] * len(selected)
    for tier in tiers:
        values = [row.tier_mix.get(tier, 0.0) * 100.0 for row in selected]
        ax.bar(
            labels,
            values,
            bottom=bottom,
            color=TIER_COLORS.get(tier, "#777777"),
            label=tier,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom = [old + value for old, value in zip(bottom, values)]
    ax.set_ylim(0, 100)
    ax.set_ylabel("Requests (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_table(rows: list[Row], output_path: Path) -> None:
    by_policy = {row.policy: row for row in rows}
    lines: list[str] = []
    for policy in TABLE_POLICIES:
        row = by_policy[policy]
        hedge = "--" if row.hedge_rate == 0 else f"{row.hedge_rate * 100.0:.1f}\\%"
        lines.append(
            f"{row.label} & "
            f"{row.total_cost_usd:.2f} & "
            f"{row.mean_ttft_ms / 1000.0:.2f} & "
            f"{row.p99_ms / 1000.0:.2f} & "
            f"{row.slo_violation_rate * 100.0:.2f}\\% & "
            f"{hedge} \\\\"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(rows: list[Row], output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.summary_csv)
    plot_frontier(rows, args.frontier_out)
    plot_slo(rows, args.slo_out)
    plot_tier_mix(rows, args.tier_out)
    write_table(rows, args.table_out)
    write_summary(rows, args.summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
