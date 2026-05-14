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

from plots.end_to_end.frontier_plotting import (
    CdfSeries,
    FrontierPoint,
    MixRow,
    MixSegment,
    TIER_MIX_SEGMENTS,
    plot_hedging_p99 as plot_common_hedging_p99,
    plot_mean_ttft_frontier,
    plot_slo_frontier,
    plot_stacked_mix,
    plot_ttft_cdf as plot_common_ttft_cdf,
    policy_plot_label,
)


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
)
ROUTEWISE_TABLE_ALPHAS = (0.0, 0.25, 0.5, 0.75)
ROUTEWISE_FRONTIER_POLICY = "ablation_lp_hedging_p25"
CDF_POLICIES = (
    "ablation_lp_hedging_p25",
    "greedy_cost",
    "greedy_latency",
    "or_sort_cost",
    "or_sort_latency",
    "random",
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


def selected_frontier_points(rows: list[Row]) -> list[FrontierPoint]:
    return [
        point
        for point in frontier_points(rows)
        if point.alpha is None or point.policy == ROUTEWISE_FRONTIER_POLICY
    ]


def plot_frontier(rows: list[Row], output_path: Path) -> None:
    plot_mean_ttft_frontier(selected_frontier_points(rows), output_path)


def plot_slo(rows: list[Row], output_path: Path) -> None:
    plot_slo_frontier(selected_frontier_points(rows), output_path)


def plot_tier_mix(rows: list[Row], output_path: Path) -> None:
    selected = routewise_rows(rows, hedging=False)
    mix_rows = [
        MixRow(label=rf"RW $\alpha={row.alpha:g}$", shares=row.tier_mix)
        for row in selected
    ]
    segments = [MixSegment(key, label, color) for key, label, color in TIER_MIX_SEGMENTS]
    plot_stacked_mix(mix_rows, segments, output_path, legend_ncols=3)


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
                include_in_axis=policy != "random",
            )
        )
    plot_common_ttft_cdf(series, output_path, slo_sec=3.0, x_max_sec=x_max_sec)


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
