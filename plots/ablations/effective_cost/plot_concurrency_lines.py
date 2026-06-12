"""Plot effective-cost concurrency ablation line figures."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt

from experiments.ablations.effective_cost.presets import parse_ablation_policy_name
from plots.helpers import save_figure
from plots.style import apply_style

DEFAULT_INPUT_DIRS = (
    Path("outputs/ablations/effective_cost_phaseB_concurrency_core"),
    Path("outputs/ablations/effective_cost_phaseB_concurrency_sanity"),
)
DEFAULT_BASELINE_SUMMARY = Path("outputs/simulation/cost_layer_1_3_featherless_main/summary.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/ablations/effective_cost_phaseB_concurrency_merged")
DEFAULT_COUNTS = (6, 8, 10, 11, 12, 13, 14, 16)
CURVE_ORDER = (
    "constant_0",
    "exp_lu",
    "linear_lu",
    "constant_l",
    "constant_u",
)
CURVE_LABELS = {
    "constant_0": "constant 0",
    "constant_l": "constant L",
    "exp_lu": "exp L-U",
    "linear_lu": "linear L-U",
    "constant_u": "constant U",
}
CURVE_COLORS = {
    "constant_0": "#9467bd",
    "constant_l": "#1f77b4",
    "exp_lu": "#2ca02c",
    "linear_lu": "#ff7f0e",
    "constant_u": "#d62728",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        help="Input ablation output directory containing summary.csv. Repeat for split runs.",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_SUMMARY,
        help="Deprecated; kept for compatibility with older pipeline commands.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Merged output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--n",
        type=int,
        action="append",
        dest="counts",
        help=f"Expected concurrency count. Repeat to override defaults {DEFAULT_COUNTS}.",
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
        help="Skip the concurrency subscription-utilization line plot.",
    )
    args = parser.parse_args(argv)

    input_dirs = tuple(args.input_dir or DEFAULT_INPUT_DIRS)
    counts = tuple(args.counts or DEFAULT_COUNTS)
    curves = tuple(args.curves or CURVE_ORDER)

    raw_rows = _load_summaries(input_dirs)
    rows = _normalize_ablation_rows(raw_rows, expected_p=float(args.p))
    _validate_grid(rows, counts=counts, curves=curves)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "summary.csv", raw_rows)
    percent_rows = _percent_delta_rows(rows, counts=counts, curves=curves)
    _write_csv(args.output_dir / "effective_cost_concurrency_percent_delta.csv", percent_rows)

    figure_dir = args.output_dir / "figures"
    _plot_total_cost_curves(rows, counts=counts, curves=curves, output_dir=figure_dir)
    if not args.skip_utilization_plot:
        _plot_subscription_utilization_curves(
            rows,
            counts=counts,
            curves=curves,
            output_dir=figure_dir,
        )
    _plot_subscription_request_share_curves(
        rows,
        counts=counts,
        curves=curves,
        output_dir=figure_dir,
    )

    print(
        json.dumps(
            {
                "curves": list(curves),
                "input_dirs": [str(path) for path in input_dirs],
                "output_dir": str(args.output_dir),
                "counts": list(counts),
                "rows": len(rows),
            },
            sort_keys=True,
        )
    )
    return 0


def _load_summaries(input_dirs: tuple[Path, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for input_dir in input_dirs:
        rows.extend(_load_summary_file(input_dir / "summary.csv"))
    if not rows:
        raise ValueError("no ablation summary rows loaded")
    return rows


def _load_summary_file(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_ablation_rows(
    raw_rows: list[dict[str, str]],
    *,
    expected_p: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        _, concurrency_curve, p_value = parse_ablation_policy_name(raw["policy"])
        if abs(p_value - expected_p) > 1e-9:
            raise ValueError(f"expected p={expected_p}, got p={p_value} for policy {raw['policy']}")
        rows.append(_row_payload(raw, label=concurrency_curve))
    return rows


def _row_payload(raw: dict[str, str], *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "scenario": raw["scenario"],
        "policy": raw["policy"],
        "n": int(float(raw["concurrency_count"])),
        "total_cost_usd_per_run": float(raw["total_cost_usd_per_run"]),
        "api_cost_usd_per_run": float(raw["api_cost_usd_per_run"]),
        "subscription_fixed_cost_usd_per_run": float(raw["subscription_fixed_cost_usd_per_run"]),
        "concurrency_request_fraction": _concurrency_fraction(raw),
        "subscription_utilization": float(raw["mean_concurrency_utilization"]),
        "mean_concurrency_utilization": float(raw["mean_concurrency_utilization"]),
        "peak_used_concurrency_cost": float(raw["peak_used_concurrency_cost"]),
        "mean_ttft_ms": float(raw["mean_ttft_ms"]),
        "p99_ms": float(raw["p99_ms"]),
        "slo_violation_rate": float(raw["slo_violation_rate"]),
    }


def _validate_grid(
    rows: list[dict[str, Any]],
    *,
    counts: tuple[int, ...],
    curves: tuple[str, ...],
) -> None:
    expected = {(curve, count) for curve in curves for count in counts}
    observed = {(str(row["label"]), int(row["n"])) for row in rows}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"unexpected concurrency grid; missing={missing}, extra={extra}")
    if len(rows) != len(expected):
        raise ValueError(f"expected {len(expected)} rows, got {len(rows)}")


def _percent_delta_rows(
    rows: list[dict[str, Any]],
    *,
    counts: tuple[int, ...],
    curves: tuple[str, ...],
) -> list[dict[str, Any]]:
    row_by_curve_n = {(row["label"], row["n"]): row for row in rows}
    baseline_by_n = {
        n: row_by_curve_n[("exp_lu", n)]["total_cost_usd_per_run"] for n in counts
    }
    output = []
    for curve in curves:
        for n in counts:
            row = row_by_curve_n[(curve, n)]
            baseline = baseline_by_n[n]
            total = row["total_cost_usd_per_run"]
            output.append(
                {
                    "curve": curve,
                    "n": n,
                    "total_cost_usd_per_run": total,
                    "baseline_total_cost_usd_per_run": baseline,
                    "delta_pct_vs_exp_lu": (total - baseline) / baseline * 100.0,
                }
            )
    return output


def _plot_total_cost_curves(
    rows: list[dict[str, Any]],
    *,
    counts: tuple[int, ...],
    curves: tuple[str, ...],
    output_dir: Path,
) -> None:
    _plot_curve_lines(
        rows,
        x_values=counts,
        curves=curves,
        y_key="total_cost_usd_per_run",
        ylabel="Total cost ($)",
        xlabel="Featherless Premium accounts (n)",
        title="Concurrency ablation total cost",
        output_dir=output_dir,
        filename="effective_cost_concurrency_total_cost_curves",
        mark_min=True,
    )


def _plot_subscription_utilization_curves(
    rows: list[dict[str, Any]],
    *,
    counts: tuple[int, ...],
    curves: tuple[str, ...],
    output_dir: Path,
) -> None:
    _plot_curve_lines(
        rows,
        x_values=counts,
        curves=curves,
        y_key="subscription_utilization",
        ylabel="Subscription utilization",
        xlabel="Featherless Premium accounts (n)",
        title="Concurrency subscription utilization",
        output_dir=output_dir,
        filename="effective_cost_concurrency_subscription_utilization_curves",
        y_min=0.0,
        y_max=1.05,
        mark_min=False,
    )


def _plot_subscription_request_share_curves(
    rows: list[dict[str, Any]],
    *,
    counts: tuple[int, ...],
    curves: tuple[str, ...],
    output_dir: Path,
) -> None:
    _plot_curve_lines(
        rows,
        x_values=counts,
        curves=curves,
        y_key="concurrency_request_fraction",
        ylabel="Subscription request share",
        xlabel="Featherless Premium accounts (n)",
        title="Concurrency subscription request share",
        output_dir=output_dir,
        filename="effective_cost_concurrency_subscription_request_share_curves",
        y_min=0.0,
        y_max=1.05,
        mark_min=False,
    )


def _plot_curve_lines(
    rows: list[dict[str, Any]],
    *,
    x_values: tuple[int, ...],
    curves: tuple[str, ...],
    y_key: str,
    ylabel: str,
    xlabel: str,
    title: str,
    output_dir: Path,
    filename: str,
    y_min: float | None = None,
    y_max: float | None = None,
    mark_min: bool = False,
) -> None:
    _apply_effective_cost_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for curve in curves:
        curve_rows = _rows_for_label(rows, curve)
        xs = [row["n"] for row in curve_rows]
        ys = [row[y_key] for row in curve_rows]
        color = CURVE_COLORS[curve]
        ax.plot(
            xs,
            ys,
            marker="o",
            color=color,
            linewidth=1.55,
            markersize=3.8,
            label=CURVE_LABELS[curve],
        )
        if mark_min and curve_rows:
            best = min(curve_rows, key=lambda row: row[y_key])
            ax.scatter(
                [best["n"]],
                [best[y_key]],
                marker="*",
                s=72,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
    ax.set_xticks(list(x_values))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=5)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.24)
    ax.legend(frameon=False, ncols=2, loc="best")
    save_figure(fig, output_dir, filename, formats=["pdf", "png"])
    plt.close(fig)


def _rows_for_label(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if row["label"] == label],
        key=lambda row: row["n"],
    )


def _concurrency_fraction(row: dict[str, str]) -> float:
    tier_mix = _parse_mapping(row["tier_mix"])
    if "concurrency" in tier_mix:
        return tier_mix["concurrency"]
    provider_mix = _parse_mapping(row["provider_mix"])
    return sum(
        fraction
        for provider, fraction in provider_mix.items()
        if "concurrency" in provider or "featherless" in provider
    )


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
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.4,
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
