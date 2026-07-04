"""Plot effective-cost quota-limit ablation line figures."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt

from experiments.ablations.effective_cost.presets import parse_ablation_policy_name
from experiments.simulation import common
from plots.helpers import save_figure
from plots.style import apply_style

DEFAULT_INPUT_DIRS = (
    Path("outputs/ablations/effective_cost_phaseA_qsweep_exp_linear"),
    Path("outputs/ablations/effective_cost_phaseA_qsweep_constants"),
)
DEFAULT_OUTPUT_DIR = Path("outputs/ablations/effective_cost_phaseA_qsweep_merged")
DEFAULT_Q_VALUES = (10_000, 20_000, 40_000, 60_000, 80_000)
CURVE_ORDER = ("constant_0", "exp_lu", "linear_lu", "constant_l", "constant_u")
CURVE_LABELS = {
    "constant_0": "constant 0",
    "constant_l": "constant L",
    "exp_lu": "exp L-U",
    "linear_lu": "linear L-U",
    "constant_u": "constant U",
}
CURVE_COLORS = {
    "constant_0": "#9467bd",
    "constant_l": "#2ca02c",
    "exp_lu": "#1f77b4",
    "linear_lu": "#ff7f0e",
    "constant_u": "#d62728",
}
# exp L-U and constant 0 are the headline curves: they close the legend and are
# drawn last (highlight on top) so overlapping curves cannot hide them.
LEGEND_ORDER = ("linear_lu", "constant_l", "constant_u", "exp_lu", "constant_0")
HIGHLIGHT_CURVE = "exp_lu"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        help="Input output directory containing summary.csv. Repeat for split runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Merged output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--workload",
        default="burstgpt",
        choices=("burstgpt", "sharegpt_burstgpt"),
        help="Workload used to compute binding-day fraction.",
    )
    parser.add_argument(
        "--quota-limit",
        type=int,
        action="append",
        dest="q_values",
        help=f"Expected quota limit. Repeat to override defaults {DEFAULT_Q_VALUES}.",
    )
    parser.add_argument(
        "--q",
        type=int,
        action="append",
        dest="q_values",
        help="Deprecated alias for --quota-limit.",
    )
    parser.add_argument(
        "--curve",
        action="append",
        choices=CURVE_ORDER,
        dest="curves",
        help=f"Expected curve. Repeat to override defaults {CURVE_ORDER}.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.0,
        help="Expected p value. Defaults to 0.",
    )
    parser.add_argument(
        "--skip-utilization-plot",
        action="store_true",
        help="Skip the quota subscription-utilization line plot.",
    )
    args = parser.parse_args(argv)

    input_dirs = tuple(args.input_dir or DEFAULT_INPUT_DIRS)
    q_values = tuple(args.q_values or DEFAULT_Q_VALUES)
    curves = tuple(args.curves or CURVE_ORDER)

    raw_rows = _load_summaries(input_dirs)
    rows = _normalize_rows(raw_rows, expected_p=float(args.p))
    _validate_grid(rows, q_values=q_values, curves=curves)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "summary.csv", raw_rows)

    percent_rows = _percent_delta_rows(rows, q_values=q_values, curves=curves)
    binding_rows = _binding_day_rows(q_values, workload=args.workload)
    _write_csv(args.output_dir / "effective_cost_qsweep_percent_delta.csv", percent_rows)
    _write_csv(args.output_dir / "effective_cost_qsweep_binding_days.csv", binding_rows)

    figure_dir = args.output_dir / "figures"
    _plot_total_cost_curves(rows, q_values=q_values, curves=curves, output_dir=figure_dir)
    if not args.skip_utilization_plot:
        _plot_subscription_utilization_curves(
            rows,
            q_values=q_values,
            curves=curves,
            output_dir=figure_dir,
        )
    _plot_subscription_request_share_curves(
        rows,
        q_values=q_values,
        curves=curves,
        output_dir=figure_dir,
    )

    print(
        json.dumps(
            {
                "input_dirs": [str(path) for path in input_dirs],
                "output_dir": str(args.output_dir),
                "rows": len(rows),
                "q_values": list(q_values),
                "curves": list(curves),
            },
            sort_keys=True,
        )
    )
    return 0


def _load_summaries(input_dirs: tuple[Path, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for input_dir in input_dirs:
        path = input_dir / "summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("no summary rows loaded")
    return rows


def _normalize_rows(
    raw_rows: list[dict[str, str]],
    *,
    expected_p: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        quota_curve, _, p_value = parse_ablation_policy_name(raw["policy"])
        if abs(p_value - expected_p) > 1e-9:
            raise ValueError(f"expected p={expected_p}, got p={p_value} for policy {raw['policy']}")
        quota_limit = _quota_limit(raw)
        quota_fraction = _quota_fraction(raw)
        rows.append(
            {
                "scenario": raw["scenario"],
                "policy": raw["policy"],
                "curve": quota_curve,
                "q": quota_limit,
                "total_cost_usd_per_run": float(raw["total_cost_usd_per_run"]),
                "api_cost_usd_per_run": float(raw["api_cost_usd_per_run"]),
                "subscription_fixed_cost_usd_per_run": float(
                    raw["subscription_fixed_cost_usd_per_run"]
                ),
                "quota_request_fraction": quota_fraction,
                "subscription_utilization": _quota_subscription_utilization(
                    raw,
                    quota_limit=quota_limit,
                    quota_fraction=quota_fraction,
                ),
                "mean_ttft_ms": float(raw["mean_ttft_ms"]),
                "p99_ms": float(raw["p99_ms"]),
                "slo_violation_rate": float(raw["slo_violation_rate"]),
            }
        )
    return rows


def _validate_grid(
    rows: list[dict[str, Any]],
    *,
    q_values: tuple[int, ...],
    curves: tuple[str, ...],
) -> None:
    expected = {(curve, q) for curve in curves for q in q_values}
    observed = {(str(row["curve"]), int(row["q"])) for row in rows}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"unexpected q-sweep grid; missing={missing}, extra={extra}")
    if len(rows) != len(expected):
        raise ValueError(f"expected {len(expected)} rows, got {len(rows)}")


def _percent_delta_rows(
    rows: list[dict[str, Any]],
    *,
    q_values: tuple[int, ...],
    curves: tuple[str, ...],
) -> list[dict[str, Any]]:
    row_by_curve_q = {(row["curve"], row["q"]): row for row in rows}
    baseline_by_q = {q: row_by_curve_q[("exp_lu", q)]["total_cost_usd_per_run"] for q in q_values}
    output = []
    for curve in curves:
        for q in q_values:
            row = row_by_curve_q[(curve, q)]
            baseline = baseline_by_q[q]
            total = row["total_cost_usd_per_run"]
            output.append(
                {
                    "curve": curve,
                    "q": q,
                    "total_cost_usd_per_run": total,
                    "baseline_total_cost_usd_per_run": baseline,
                    "delta_pct_vs_exp_lu": (total - baseline) / baseline * 100.0,
                }
            )
    return output


def _binding_day_rows(q_values: tuple[int, ...], *, workload: str) -> list[dict[str, Any]]:
    requests = common.load_workload(dataset=workload)
    if not requests:
        raise ValueError("cannot compute binding days for an empty workload")
    trace_start = float(requests[0].timestamp)
    counts = [0] * 30
    for request in requests:
        day = int((float(request.timestamp) - trace_start) // 86400.0)
        if 0 <= day < len(counts):
            counts[day] += 1
    return [
        {
            "q": q,
            "binding_day_fraction": sum(count > q for count in counts) / 30.0,
        }
        for q in q_values
    ]


def _plot_total_cost_curves(
    rows: list[dict[str, Any]],
    *,
    q_values: tuple[int, ...],
    curves: tuple[str, ...],
    output_dir: Path,
) -> None:
    _plot_curve_lines(
        rows,
        x_values=q_values,
        curves=curves,
        y_key="total_cost_usd_per_run",
        ylabel="Total cost ($)",
        xlabel="Quota limit (requests/day)",
        output_dir=output_dir,
        filename="effective_cost_quota_total_cost_curves",
    )


def _plot_subscription_utilization_curves(
    rows: list[dict[str, Any]],
    *,
    q_values: tuple[int, ...],
    curves: tuple[str, ...],
    output_dir: Path,
) -> None:
    _plot_curve_lines(
        rows,
        x_values=q_values,
        curves=curves,
        y_key="subscription_utilization",
        ylabel="Subscription utilization",
        xlabel="Quota limit (requests/day)",
        output_dir=output_dir,
        filename="effective_cost_quota_subscription_utilization_curves",
        y_min=0.0,
        y_max=1.05,
    )


def _plot_subscription_request_share_curves(
    rows: list[dict[str, Any]],
    *,
    q_values: tuple[int, ...],
    curves: tuple[str, ...],
    output_dir: Path,
) -> None:
    _plot_curve_lines(
        rows,
        x_values=q_values,
        curves=curves,
        y_key="quota_request_fraction",
        ylabel="Subscription request share",
        xlabel="Quota limit (requests/day)",
        output_dir=output_dir,
        filename="effective_cost_quota_subscription_request_share_curves",
        y_min=0.0,
        y_max=1.05,
    )


def _plot_curve_lines(
    rows: list[dict[str, Any]],
    *,
    x_values: tuple[int, ...],
    curves: tuple[str, ...],
    y_key: str,
    ylabel: str,
    xlabel: str,
    output_dir: Path,
    filename: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    _apply_effective_cost_style()
    fig, ax = plt.subplots(figsize=(2.3, 1.95))
    legend_curves = [curve for curve in LEGEND_ORDER if curve in curves]
    draw_order = sorted(legend_curves, key=lambda curve: curve == HIGHLIGHT_CURVE)
    handles: dict[str, Any] = {}
    for curve in draw_order:
        curve_rows = _rows_for_curve(rows, curve)
        xs = [row["q"] for row in curve_rows]
        ys = [row[y_key] for row in curve_rows]
        color = CURVE_COLORS[curve]
        (handles[curve],) = ax.plot(
            xs,
            ys,
            marker="o",
            color=color,
            linewidth=1.1,
            markersize=3.0,
            label=CURVE_LABELS[curve],
        )
    ax.set_xticks(list(x_values))
    ax.set_xticklabels(
        [f"{value // 1000}k" if value % 1000 == 0 else str(value) for value in x_values]
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.24)
    ax.legend(
        [handles[curve] for curve in legend_curves],
        [CURVE_LABELS[curve] for curve in legend_curves],
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        bbox_transform=fig.transFigure,
        columnspacing=0.6,
        handlelength=1.4,
        handletextpad=0.3,
    )
    fig.subplots_adjust(left=0.27, right=0.97, top=0.79, bottom=0.22)
    save_figure(fig, output_dir, filename, formats=["pdf", "png"], full_canvas=True)
    plt.close(fig)


def _rows_for_curve(rows: list[dict[str, Any]], curve: str) -> list[dict[str, Any]]:
    return sorted([row for row in rows if row["curve"] == curve], key=lambda row: row["q"])


def _quota_fraction(row: dict[str, str]) -> float:
    tier_mix = _parse_mapping(row["tier_mix"])
    if "quota" in tier_mix:
        return tier_mix["quota"]
    provider_mix = _parse_mapping(row["provider_mix"])
    return sum(
        fraction
        for provider, fraction in provider_mix.items()
        if "quota" in provider or "chutes" in provider
    )


def _quota_subscription_utilization(
    row: dict[str, str],
    *,
    quota_limit: int,
    quota_fraction: float,
) -> float:
    n_requests = int(float(row["n_requests"]))
    trace_days = float(row.get("trace_days") or 0.0)
    observed_days = max(1, math.ceil(trace_days))
    capacity = float(quota_limit) * observed_days
    if capacity <= 0.0:
        return 0.0
    return min(max(quota_fraction * n_requests / capacity, 0.0), 1.0)


def _quota_limit(row: dict[str, str]) -> int:
    raw_quota_limit = row.get("quota_limit")
    if raw_quota_limit not in (None, ""):
        return int(float(raw_quota_limit))
    return int(float(row["subscription_count"]))


def _parse_mapping(value: str) -> dict[str, float]:
    payload = ast.literal_eval(value)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping payload, got {value!r}")
    return {str(key): float(val) for key, val in payload.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _apply_effective_cost_style() -> None:
    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
