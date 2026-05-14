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
      --p99-bar-out ../paper/figures/stage3_e2e_hedging_p99.pdf \
      --cdf-out ../paper/figures/stage3_e2e_ttft_cdf.pdf \
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
    "or_sort_cost": "API price-sort",
    "random": "Random",
    "greedy_latency": "Greedy-latency",
    "or_sort_latency": "API latency-sort",
}
BASELINE_ORDER = (
    "greedy_cost",
    "or_sort_cost",
    "random",
    "greedy_latency",
    "or_sort_latency",
)
PLOT_BASELINE_ORDER = (
    "greedy_cost",
    "or_sort_cost",
    "greedy_latency",
    "or_sort_latency",
)
TABLE_POLICIES = (
    "greedy_cost",
    "or_sort_cost",
    "random",
    "greedy_latency",
    "or_sort_latency",
)
ROUTEWISE_TABLE_ALPHAS = (0.0, 0.25, 0.5, 0.75)
CDF_POLICIES = (
    "greedy_cost",
    "ablation_lp_only_p50",
    "ablation_lp_hedging_p50",
    "ablation_lp_hedging_p75",
    "greedy_latency",
)
COLUMN_FIGSIZE_TALL = (3.25, 2.25)
ROUTEWISE_LINE_COLOR = "#2f6f73"
ROUTEWISE_HEDGE_COLOR = ROUTER_STRATEGY_COLORS.get("ablation_lp_hedging", "#ff7f0e")
BASELINE_COLOR = "#6f7f80"
COLUMN_FIGSIZE = (3.25, 2.05)


def apply_column_figure_style() -> None:
    """Style for single-metric PDFs placed as one-column paper figures."""
    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.5,
            "savefig.pad_inches": 0.01,
        }
    )


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
    p95_ms: float
    p99_ms: float
    slo_violation_rate: float
    hedge_rate: float
    tier_mix: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--histograms-json", type=Path, default=None)
    parser.add_argument("--frontier-out", type=Path, required=True)
    parser.add_argument("--slo-out", type=Path, required=True)
    parser.add_argument("--tier-out", type=Path, default=None)
    parser.add_argument("--p99-bar-out", type=Path, default=None)
    parser.add_argument("--cdf-out", type=Path, default=None)
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


def load_histograms(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["policy"]: item["histogram"] for item in payload}


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
                    policy=policy,
                    label=row_label(policy, alpha, hedging),
                    alpha=alpha,
                    hedging=hedging,
                    total_cost_usd=float(raw["total_cost_usd"]),
                    mean_ttft_ms=float(raw["mean_ttft_ms"]),
                    p50_ms=float(raw["p50_ms"]),
                    p90_ms=float(raw["p90_ms"]),
                    p95_ms=p95_ms,
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


def plot_baseline_rows(rows: list[Row]) -> list[Row]:
    by_policy = {row.policy: row for row in rows}
    return [by_policy[name] for name in PLOT_BASELINE_ORDER if name in by_policy]


def annotate_points(ax, rows: list[Row], *, y_value) -> None:
    for row in rows:
        ax.annotate(
            row.label.replace(r"\sysname{}", "RW"),
            (row.total_cost_usd, y_value(row)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )


def _deduplicate_rows(rows: list[Row], *, y_value) -> list[tuple[float, float, str]]:
    """Group rows at the same (x, y) position and merge their labels."""
    seen: dict[tuple[float, float], list[str]] = {}
    order: list[tuple[float, float]] = []
    for row in rows:
        key = (round(row.total_cost_usd, 2), round(y_value(row), 4))
        if key not in seen:
            seen[key] = []
            order.append(key)
        lbl = (row.label if row.alpha is None else
               (f"{row.alpha:g}" if row.alpha < 0.75 else r"$\geq$0.75"))
        if lbl not in seen[key]:
            seen[key].append(lbl)
    return [(x, y, " / ".join(labels)) for (x, y), labels in
            ((k, seen[k]) for k in order)]


def _annotate_alpha(ax, rows: list[Row], *, y_value, fontsize: float = 5.5,
                    ha: str = "left", offset: tuple[int, int] = (3, 4)) -> None:
    for x, y, label in _deduplicate_rows(rows, y_value=y_value):
        ax.annotate(
            label, (x, y),
            xytext=offset, textcoords="offset points",
            fontsize=fontsize, color="#333333", ha=ha,
        )


BASELINE_SHORT = {
    "Greedy-latency / API latency-sort": "Greedy-lat. / API lat.-sort",
}


def _annotate_baselines(ax, rows: list[Row], *, y_value, fontsize: float = 5.5) -> None:
    for x, y, label in _deduplicate_rows(rows, y_value=y_value):
        label = BASELINE_SHORT.get(label, label)
        ax.annotate(
            label, (x, y),
            xytext=(-3, -8), textcoords="offset points",
            fontsize=fontsize, color=BASELINE_COLOR, ha="right",
        )


def plot_frontier(rows: list[Row], output_path: Path) -> None:
    apply_column_figure_style()
    fig, ax = plt.subplots(figsize=COLUMN_FIGSIZE)
    no_hedge = routewise_rows(rows, hedging=False)
    hedged = routewise_rows(rows, hedging=True)
    baselines = plot_baseline_rows(rows)

    ax.plot(
        [row.total_cost_usd for row in no_hedge],
        [row.mean_ttft_ms / 1000.0 for row in no_hedge],
        marker="o",
        color=ROUTEWISE_LINE_COLOR,
        label="RouteWise",
    )
    ax.plot(
        [row.total_cost_usd for row in hedged],
        [row.mean_ttft_ms / 1000.0 for row in hedged],
        marker="s",
        color=ROUTEWISE_HEDGE_COLOR,
        label="RouteWise + hedge",
    )
    ax.scatter(
        [row.total_cost_usd for row in baselines],
        [row.mean_ttft_ms / 1000.0 for row in baselines],
        marker="x",
        s=22,
        color=BASELINE_COLOR,
        linewidths=1.1,
        label="Baselines",
    )
    yv = lambda r: r.mean_ttft_ms / 1000.0
    _annotate_alpha(ax, no_hedge, y_value=yv, offset=(3, 4))
    _annotate_alpha(ax, hedged, y_value=yv, offset=(3, -10))
    _annotate_baselines(ax, baselines, y_value=yv)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.set_xlabel("Total cost ($)")
    ax.set_ylabel("Mean TTFT (s)")
    ax.margins(x=0.10, y=0.15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_slo(rows: list[Row], output_path: Path) -> None:
    apply_column_figure_style()
    fig, ax = plt.subplots(figsize=COLUMN_FIGSIZE)
    no_hedge = routewise_rows(rows, hedging=False)
    hedged = routewise_rows(rows, hedging=True)
    baselines = plot_baseline_rows(rows)

    ax.plot(
        [row.total_cost_usd for row in no_hedge],
        [row.slo_violation_rate * 100.0 for row in no_hedge],
        marker="o",
        color=ROUTEWISE_LINE_COLOR,
        label="RouteWise",
    )
    ax.plot(
        [row.total_cost_usd for row in hedged],
        [row.slo_violation_rate * 100.0 for row in hedged],
        marker="s",
        color=ROUTEWISE_HEDGE_COLOR,
        label="RouteWise + hedge",
    )
    ax.scatter(
        [row.total_cost_usd for row in baselines],
        [row.slo_violation_rate * 100.0 for row in baselines],
        marker="x",
        s=22,
        color=BASELINE_COLOR,
        linewidths=1.1,
        label="Baselines",
    )
    yv = lambda r: r.slo_violation_rate * 100.0
    _annotate_alpha(ax, no_hedge, y_value=yv, offset=(3, 4))
    _annotate_alpha(ax, hedged, y_value=yv, offset=(3, -10))
    _annotate_baselines(ax, baselines, y_value=yv)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.set_xlabel("Total cost ($)")
    ax.set_ylabel("SLO violation (%)")
    ax.margins(x=0.10, y=0.15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_tier_mix(rows: list[Row], output_path: Path) -> None:
    apply_column_figure_style()
    selected = routewise_rows(rows, hedging=False)
    labels = [f"{row.alpha:g}" for row in selected]
    tiers = ("quota", "concurrency", "api")
    tier_labels = {
        "quota": r"$\mathcal{P}_Q$ quota",
        "concurrency": r"$\mathcal{P}_C$ conc.",
        "api": r"$\mathcal{P}_O$ API",
    }

    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    bottom = [0.0] * len(selected)
    for tier in tiers:
        values = [row.tier_mix.get(tier, 0.0) * 100.0 for row in selected]
        ax.bar(
            labels,
            values,
            bottom=bottom,
            color=TIER_COLORS.get(tier, "#777777"),
            label=tier_labels.get(tier, tier),
            edgecolor="white",
            linewidth=0.5,
        )
        bottom = [old + value for old, value in zip(bottom, values)]
    ax.set_ylim(0, 100)
    ax.set_ylabel("Requests (%)")
    ax.set_xlabel(r"$\alpha$")
    ax.tick_params(axis="x", pad=1)
    ax.legend(
        frameon=False,
        fontsize=6.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncols=3,
        handlelength=0.9,
        columnspacing=0.9,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_hedging_p99(rows: list[Row], output_path: Path) -> None:
    apply_column_figure_style()
    fig, ax = plt.subplots(figsize=COLUMN_FIGSIZE_TALL)
    no_hedge = {row.alpha: row for row in routewise_rows(rows, hedging=False)}
    hedged = {row.alpha: row for row in routewise_rows(rows, hedging=True)}
    alphas = [alpha for alpha in ROUTEWISE_TABLE_ALPHAS if alpha in no_hedge and alpha in hedged]
    labels = [f"{alpha:g}" for alpha in alphas]
    x = list(range(len(alphas)))
    width = 0.36

    ax.bar(
        [value - width / 2 for value in x],
        [no_hedge[alpha].p99_ms / 1000.0 for alpha in alphas],
        width=width,
        color=ROUTEWISE_LINE_COLOR,
        label="RW",
    )
    ax.bar(
        [value + width / 2 for value in x],
        [hedged[alpha].p99_ms / 1000.0 for alpha in alphas],
        width=width,
        color=ROUTEWISE_HEDGE_COLOR,
        label="RW + hedge",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("P99 TTFT (s)")
    ax.set_ylim(0, max(no_hedge[alpha].p99_ms for alpha in alphas) / 1000.0 * 1.18)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_ttft_cdf(
    rows: list[Row],
    histograms: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    apply_column_figure_style()
    by_policy = {row.policy: row for row in rows}
    fig, ax = plt.subplots(figsize=COLUMN_FIGSIZE_TALL)
    x_ms = [200.0 * ((30_000.0 / 200.0) ** (idx / 179.0)) for idx in range(180)]
    colors = {
        "greedy_cost": BASELINE_COLOR,
        "greedy_latency": "#3a8f4f",
        "ablation_lp_only_p50": ROUTEWISE_LINE_COLOR,
        "ablation_lp_hedging_p50": ROUTEWISE_HEDGE_COLOR,
        "ablation_lp_hedging_p75": "#c95a17",
    }
    labels = {
        "greedy_cost": "Greedy-cost",
        "greedy_latency": "Greedy-latency",
        "ablation_lp_only_p50": r"RW $\alpha=0.5$",
        "ablation_lp_hedging_p50": r"RW $\alpha=0.5$ + hedge",
        "ablation_lp_hedging_p75": r"RW $\alpha=0.75$ + hedge",
    }
    styles = {
        "greedy_cost": (0, (3, 2)),
        "greedy_latency": (0, (1.5, 1.5)),
        "ablation_lp_only_p50": "solid",
        "ablation_lp_hedging_p50": "solid",
        "ablation_lp_hedging_p75": "solid",
    }
    for policy in CDF_POLICIES:
        histogram = histograms.get(policy)
        if histogram is None or policy not in by_policy:
            continue
        y = [histogram_cdf(histogram, value) for value in x_ms]
        ax.plot(
            [value / 1000.0 for value in x_ms],
            y,
            color=colors.get(policy, "#555555"),
            linestyle=styles.get(policy, "solid"),
            label=labels.get(policy, by_policy[policy].label),
        )
    ax.axvline(3.0, color="#444444", linewidth=0.8, linestyle=":", label="3s SLO")
    ax.set_xscale("log")
    ax.set_xlim(0.2, 30.0)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("TTFT (s)")
    ax.set_ylabel("CDF")
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


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
        if alpha == 0.75 and "ablation_lp_only_p100" in by_policy:
            lines.append(_format_table_row(row, label=rf"\sysname{{}} ($\alpha\ge{alpha:g}$)"))
        else:
            lines.append(_format_table_row(row))
    lines.append(r"\midrule")
    for alpha in ROUTEWISE_TABLE_ALPHAS:
        policy = f"ablation_lp_hedging_p{int(alpha * 100)}"
        row = by_policy[policy]
        if alpha == 0.75 and "ablation_lp_hedging_p100" in by_policy:
            lines.append(_format_table_row(row, label=rf"\sysname{{}} ($\alpha\ge{alpha:g}$ + hedge)"))
        else:
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
    payload = [asdict(row) for row in rows]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    histograms = load_histograms(args.histograms_json)
    rows = load_rows(args.summary_csv, histograms)
    plot_frontier(rows, args.frontier_out)
    plot_slo(rows, args.slo_out)
    if args.tier_out is not None:
        plot_tier_mix(rows, args.tier_out)
    if args.p99_bar_out is not None:
        plot_hedging_p99(rows, args.p99_bar_out)
    if args.cdf_out is not None:
        if not histograms:
            raise SystemExit("--cdf-out requires --histograms-json")
        plot_ttft_cdf(rows, histograms, args.cdf_out)
    write_table(rows, args.table_out)
    write_summary(rows, args.summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
