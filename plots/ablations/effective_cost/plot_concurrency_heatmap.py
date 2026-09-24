"""Plot effective-cost concurrency-only ablation figures."""

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
    "constant_l",
    "exp_lu",
    "linear_lu",
    "legacy_linear_u",
    "constant_u",
)
CURVE_LABELS = {
    "constant_l": "constant L",
    "exp_lu": "exp L-U",
    "linear_lu": "linear L-U",
    "legacy_linear_u": "legacy U*u",
    "constant_u": "constant U",
}
CURVE_COLORS = {
    "constant_l": "#1f77b4",
    "exp_lu": "#2ca02c",
    "linear_lu": "#ff7f0e",
    "legacy_linear_u": "#9467bd",
    "constant_u": "#d62728",
    "offline": "#555555",
    "greedy_cost": "#111111",
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
        help=f"§1.3 baseline summary.csv. Defaults to {DEFAULT_BASELINE_SUMMARY}.",
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
        "--color-limit-pct",
        type=float,
        default=20.0,
        help="Symmetric heatmap color limit in percent. Defaults to +/-20.",
    )
    parser.add_argument(
        "--skip-usage-heatmap",
        action="store_true",
        help="Skip the optional concurrency usage appendix heatmap.",
    )
    args = parser.parse_args(argv)

    input_dirs = tuple(args.input_dir or DEFAULT_INPUT_DIRS)
    counts = tuple(args.counts or DEFAULT_COUNTS)
    curves = tuple(args.curves or CURVE_ORDER)

    raw_rows = _load_summaries(input_dirs)
    rows = _normalize_ablation_rows(raw_rows, expected_p=float(args.p))
    _validate_grid(rows, counts=counts, curves=curves)

    baseline_rows = _normalize_baseline_rows(
        _load_summary_file(args.baseline_summary),
        counts=counts,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "summary.csv", raw_rows)
    percent_rows = _percent_delta_rows(rows, counts=counts, curves=curves)
    _write_csv(args.output_dir / "effective_cost_concurrency_percent_delta.csv", percent_rows)

    figure_dir = args.output_dir / "figures"
    _plot_percent_delta_heatmap(
        percent_rows,
        counts=counts,
        curves=curves,
        output_dir=figure_dir,
        color_limit_pct=float(args.color_limit_pct),
    )
    _plot_total_cost_curves(
        rows,
        baseline_rows,
        counts=counts,
        curves=curves,
        output_dir=figure_dir,
    )
    if not args.skip_usage_heatmap:
        _plot_concurrency_usage_heatmap(
            rows,
            counts=counts,
            curves=curves,
            output_dir=figure_dir,
        )

    print(
        json.dumps(
            {
                "baseline_summary": str(args.baseline_summary),
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
        rows.append(_row_payload(raw, label=concurrency_curve, kind="ablation"))
    return rows


def _normalize_baseline_rows(
    raw_rows: list[dict[str, str]],
    *,
    counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count_set = set(counts)
    for raw in raw_rows:
        if raw.get("public_scenario") != "concurrency":
            continue
        if raw.get("model") != "sharegpt":
            continue
        if raw.get("policy") not in {"offline", "greedy_cost"}:
            continue
        if int(float(raw["concurrency_count"])) not in count_set:
            continue
        rows.append(_row_payload(raw, label=raw["policy"], kind="baseline"))
    return rows


def _row_payload(raw: dict[str, str], *, label: str, kind: str) -> dict[str, Any]:
    return {
        "label": label,
        "kind": kind,
        "scenario": raw["scenario"],
        "policy": raw["policy"],
        "n": int(float(raw["concurrency_count"])),
        "total_cost_usd_per_run": float(raw["total_cost_usd_per_run"]),
        "api_cost_usd_per_run": float(raw["api_cost_usd_per_run"]),
        "subscription_fixed_cost_usd_per_run": float(raw["subscription_fixed_cost_usd_per_run"]),
        "concurrency_request_fraction": _concurrency_fraction(raw),
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
        n: row_by_curve_n[("legacy_linear_u", n)]["total_cost_usd_per_run"] for n in counts
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
                    "delta_pct_vs_legacy_linear_u": (total - baseline) / baseline * 100.0,
                }
            )
    return output


def _plot_percent_delta_heatmap(
    percent_rows: list[dict[str, Any]],
    *,
    counts: tuple[int, ...],
    curves: tuple[str, ...],
    output_dir: Path,
    color_limit_pct: float,
) -> None:
    _apply_effective_cost_style()
    matrix = _matrix(
        percent_rows,
        labels=curves,
        counts=counts,
        value_key="delta_pct_vs_legacy_linear_u",
    )
    clipped = np.clip(matrix, -color_limit_pct, color_limit_pct)

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    image = ax.imshow(
        clipped,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-color_limit_pct,
        vmax=color_limit_pct,
    )
    ax.set_yticks(range(len(curves)))
    ax.set_yticklabels([CURVE_LABELS[curve] for curve in curves])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(count) for count in counts])
    ax.set_xlabel("Featherless Premium accounts (n)")
    ax.set_title("Total cost delta vs legacy U*u (%)", pad=5)
    for row_idx, _curve in enumerate(curves):
        for col_idx, _count in enumerate(counts):
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
    save_figure(
        fig,
        output_dir,
        "effective_cost_concurrency_percent_delta_heatmap",
        formats=["pdf", "png"],
    )
    plt.close(fig)


def _plot_total_cost_curves(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    counts: tuple[int, ...],
    curves: tuple[str, ...],
    output_dir: Path,
) -> None:
    _apply_effective_cost_style()
    fig = plt.figure(figsize=(6.75, 3.65))
    grid = fig.add_gridspec(2, 1, height_ratios=(3.2, 1.35), hspace=0.08)
    ax = fig.add_subplot(grid[0])
    util_ax = fig.add_subplot(grid[1], sharex=ax)

    all_rows = rows + baseline_rows
    for baseline in ("offline", "greedy_cost"):
        best = _best_for_label(all_rows, baseline)
        if best is not None:
            ax.axhline(
                best["total_cost_usd_per_run"],
                color=CURVE_COLORS[baseline],
                linestyle=":" if baseline == "offline" else "--",
                linewidth=1.0,
                alpha=0.65,
            )

    for curve in curves:
        curve_rows = _rows_for_label(rows, curve)
        _plot_line_with_argmin(
            ax,
            curve_rows,
            label=CURVE_LABELS[curve],
            color=CURVE_COLORS[curve],
            linestyle="-",
        )

    for baseline in ("offline", "greedy_cost"):
        baseline_curve_rows = _rows_for_label(baseline_rows, baseline)
        if not baseline_curve_rows:
            continue
        _plot_line_with_argmin(
            ax,
            baseline_curve_rows,
            label=baseline.replace("_", " "),
            color=CURVE_COLORS[baseline],
            linestyle=":" if baseline == "offline" else "--",
            annotate=False,
        )

    ax.set_ylabel("Total cost ($)")
    ax.grid(True, alpha=0.24)
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend(
        frameon=False,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        columnspacing=0.9,
        handlelength=1.6,
    )

    util_by_count = {
        count: float(
            np.mean([row["mean_concurrency_utilization"] for row in rows if row["n"] == count])
        )
        for count in counts
    }
    util_ax.plot(
        list(counts),
        [util_by_count[count] for count in counts],
        marker="o",
        color="#666666",
        linewidth=1.25,
        markersize=3.2,
    )
    util_ax.set_xlabel("Featherless Premium accounts (n)")
    util_ax.set_ylabel("mean\nutil.")
    util_ax.set_xticks(list(counts))
    util_ax.set_ylim(bottom=0.0)
    util_ax.grid(True, axis="y", alpha=0.24)

    save_figure(
        fig,
        output_dir,
        "effective_cost_concurrency_total_cost_curves",
        formats=["pdf", "png"],
    )
    plt.close(fig)


def _plot_line_with_argmin(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    label: str,
    color: str,
    linestyle: str,
    annotate: bool = True,
) -> None:
    rows = sorted(rows, key=lambda row: row["n"])
    counts = [row["n"] for row in rows]
    totals = [row["total_cost_usd_per_run"] for row in rows]
    ax.plot(
        counts,
        totals,
        marker="o",
        color=color,
        linestyle=linestyle,
        linewidth=1.45,
        markersize=3.8,
        label=label,
    )
    best = min(rows, key=lambda row: row["total_cost_usd_per_run"])
    ax.scatter(
        [best["n"]],
        [best["total_cost_usd_per_run"]],
        marker="*",
        s=72,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    if annotate:
        offset_by_label = {
            "constant L": (5, -12),
            "exp L-U": (5, 4),
            "linear L-U": (5, 7),
            "legacy U*u": (5, 8),
            "constant U": (5, 7),
        }
        ax.annotate(
            f"({best['n']}, ${best['total_cost_usd_per_run']:.0f})",
            xy=(best["n"], best["total_cost_usd_per_run"]),
            xytext=offset_by_label.get(label, (5, 5)),
            textcoords="offset points",
            fontsize=5.8,
            color=color,
        )


def _plot_concurrency_usage_heatmap(
    rows: list[dict[str, Any]],
    *,
    counts: tuple[int, ...],
    curves: tuple[str, ...],
    output_dir: Path,
) -> None:
    _apply_effective_cost_style()
    matrix = _matrix(
        rows,
        labels=curves,
        counts=counts,
        value_key="concurrency_request_fraction",
    )
    fig, ax = plt.subplots(figsize=(5.2, 2.55))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(curves)))
    ax.set_yticklabels([CURVE_LABELS[curve] for curve in curves])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(count) for count in counts])
    ax.set_xlabel("Featherless Premium accounts (n)")
    ax.set_title("Concurrency request fraction", pad=5)
    for row_idx, _curve in enumerate(curves):
        for col_idx, _count in enumerate(counts):
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
        "effective_cost_concurrency_usage_heatmap",
        formats=["pdf", "png"],
    )
    plt.close(fig)


def _matrix(
    rows: list[dict[str, Any]],
    *,
    labels: tuple[str, ...],
    counts: tuple[int, ...],
    value_key: str,
) -> np.ndarray:
    value_by_label_n = {
        (str(row.get("label", row.get("curve"))), int(row["n"])): float(row[value_key])
        for row in rows
    }
    return np.asarray(
        [[value_by_label_n[(label, count)] for count in counts] for label in labels],
        dtype=float,
    )


def _rows_for_label(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if row["label"] == label],
        key=lambda row: row["n"],
    )


def _best_for_label(rows: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    selected = _rows_for_label(rows, label)
    if not selected:
        return None
    return min(selected, key=lambda row: row["total_cost_usd_per_run"])


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
            "legend.fontsize": 6.2,
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
