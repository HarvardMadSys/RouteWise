"""Plot effective-cost q-sweep heatmaps."""

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
import numpy as np

from experiments.ablations.effective_cost.presets import parse_ablation_policy_name
from experiments.simulation import common
from plots.helpers import save_figure
from plots.style import apply_style

DEFAULT_INPUT_DIRS = (
    Path("outputs/ablations/effective_cost_phaseA_qsweep_exp_linear"),
    Path("outputs/ablations/effective_cost_phaseA_qsweep_constants"),
)
DEFAULT_OUTPUT_DIR = Path("outputs/ablations/effective_cost_phaseA_qsweep_merged")
DEFAULT_Q_VALUES = (2, 4, 8, 12, 16)
CURVE_ORDER = ("constant_l", "exp_lu", "linear_lu", "constant_u")
CURVE_LABELS = {
    "constant_l": "constant L",
    "exp_lu": "exp L-U",
    "linear_lu": "linear L-U",
    "constant_u": "constant U",
}


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
        "--q",
        type=int,
        action="append",
        dest="q_values",
        help=f"Expected q value. Repeat to override defaults {DEFAULT_Q_VALUES}.",
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
        "--color-limit-pct",
        type=float,
        default=20.0,
        help="Symmetric heatmap color limit in percent. Defaults to +/-20.",
    )
    parser.add_argument(
        "--skip-quota-fraction",
        action="store_true",
        help="Skip the optional quota-request-fraction appendix heatmap.",
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
    _plot_percent_delta_heatmap(
        percent_rows,
        binding_rows,
        q_values=q_values,
        curves=curves,
        output_dir=figure_dir,
        color_limit_pct=float(args.color_limit_pct),
    )
    if not args.skip_quota_fraction:
        _plot_quota_fraction_heatmap(
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
        rows.append(
            {
                "scenario": raw["scenario"],
                "policy": raw["policy"],
                "curve": quota_curve,
                "q": int(float(raw["subscription_count"])),
                "total_cost_usd_per_run": float(raw["total_cost_usd_per_run"]),
                "api_cost_usd_per_run": float(raw["api_cost_usd_per_run"]),
                "subscription_fixed_cost_usd_per_run": float(
                    raw["subscription_fixed_cost_usd_per_run"]
                ),
                "quota_request_fraction": _quota_fraction(raw),
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
            "binding_day_fraction": sum(count > q * 5000 for count in counts) / 30.0,
        }
        for q in q_values
    ]


def _plot_percent_delta_heatmap(
    percent_rows: list[dict[str, Any]],
    binding_rows: list[dict[str, Any]],
    *,
    q_values: tuple[int, ...],
    curves: tuple[str, ...],
    output_dir: Path,
    color_limit_pct: float,
) -> None:
    _apply_effective_cost_style()
    matrix = _matrix(
        percent_rows,
        curves=curves,
        q_values=q_values,
        value_key="delta_pct_vs_exp_lu",
    )
    clipped = np.clip(matrix, -color_limit_pct, color_limit_pct)
    binding = [float(row["binding_day_fraction"]) for row in binding_rows]

    fig = plt.figure(figsize=(4.8, 3.0))
    grid = fig.add_gridspec(2, 1, height_ratios=(4.0, 1.15), hspace=0.08)
    ax = fig.add_subplot(grid[0])
    bar_ax = fig.add_subplot(grid[1], sharex=ax)

    image = ax.imshow(
        clipped,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-color_limit_pct,
        vmax=color_limit_pct,
    )
    ax.set_yticks(range(len(curves)))
    ax.set_yticklabels([CURVE_LABELS[curve] for curve in curves])
    ax.set_xticks(range(len(q_values)))
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_title("Total cost delta vs exp L-U (%)", pad=5)
    for row_idx, _curve in enumerate(curves):
        for col_idx, _q in enumerate(q_values):
            value = matrix[row_idx, col_idx]
            text_color = "white" if abs(value) >= 0.65 * color_limit_pct else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:+.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("delta (%)", rotation=270, labelpad=10)

    bar_ax.bar(range(len(q_values)), binding, color="#666666", width=0.62)
    bar_ax.set_xticks(range(len(q_values)))
    bar_ax.set_xticklabels([str(q) for q in q_values])
    bar_ax.set_xlabel("Subscriptions (q)")
    bar_ax.set_ylabel("binding\nfraction")
    bar_ax.set_ylim(0.0, 1.0)
    bar_ax.grid(True, axis="y", alpha=0.24)

    save_figure(
        fig,
        output_dir,
        "effective_cost_qsweep_percent_delta_heatmap",
        formats=["pdf", "png"],
    )
    plt.close(fig)


def _plot_quota_fraction_heatmap(
    rows: list[dict[str, Any]],
    *,
    q_values: tuple[int, ...],
    curves: tuple[str, ...],
    output_dir: Path,
) -> None:
    _apply_effective_cost_style()
    matrix = _matrix(
        rows,
        curves=curves,
        q_values=q_values,
        value_key="quota_request_fraction",
    )
    fig, ax = plt.subplots(figsize=(4.8, 2.35))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(curves)))
    ax.set_yticklabels([CURVE_LABELS[curve] for curve in curves])
    ax.set_xticks(range(len(q_values)))
    ax.set_xticklabels([str(q) for q in q_values])
    ax.set_xlabel("Subscriptions (q)")
    ax.set_title("Quota request fraction", pad=5)
    for row_idx, _curve in enumerate(curves):
        for col_idx, _q in enumerate(q_values):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.5 else "black",
                fontsize=7,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("fraction", rotation=270, labelpad=10)
    save_figure(
        fig,
        output_dir,
        "effective_cost_qsweep_quota_fraction_heatmap",
        formats=["pdf", "png"],
    )
    plt.close(fig)


def _matrix(
    rows: list[dict[str, Any]],
    *,
    curves: tuple[str, ...],
    q_values: tuple[int, ...],
    value_key: str,
) -> np.ndarray:
    value_by_curve_q = {(str(row["curve"]), int(row["q"])): float(row[value_key]) for row in rows}
    return np.asarray(
        [[value_by_curve_q[(curve, q)] for q in q_values] for curve in curves],
        dtype=float,
    )


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
