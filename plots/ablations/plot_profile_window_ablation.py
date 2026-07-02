"""Plot latency-profile window-length ablation figures."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from plots.helpers import save_figure
from plots.style import apply_style

DEFAULT_INPUT_DIR = Path("outputs/ablations/profile_window")

FAMILY_ORDER = ("lp", "hedge")
FAMILY_LABELS = {"lp": "LP", "hedge": "LP+Hedge"}
# One color per environment change period; static gets gray.
PERIOD_COLOR_CYCLE = ("#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b")
STATIC_COLOR = "#7f7f7f"

# (metric key, y-axis label, filename stem, draw oracle reference lines)
METRIC_PANELS = (
    ("p99_ms", "P99 TTFT (ms)", "profile_window_p99_lines", True),
    ("slo_violation_rate", "SLO violation rate", "profile_window_slo_lines", True),
    ("total_cost_usd", "Total cost (USD)", "profile_window_cost_lines", True),
    ("profile_fallback_rate", "Profile fallback rate", "profile_window_fallback_lines", False),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Ablation directory containing summary.csv. Defaults to {DEFAULT_INPUT_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to --input-dir.",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    rows = _load_rows(input_dir / "summary.csv")
    delta_rows = _delta_rows(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "profile_window_delta_summary.csv", delta_rows)

    figure_dir = output_dir / "figures"
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row["policy_family"] == family]
        if not family_rows:
            continue
        for metric, ylabel, stem, with_oracle in METRIC_PANELS:
            if not any(row.get(metric) is not None for row in family_rows):
                continue
            _plot_metric_lines(
                family_rows,
                metric=metric,
                ylabel=ylabel,
                name=f"{stem}_{family}",
                with_oracle=with_oracle,
                output_dir=figure_dir,
            )
        _plot_oracle_regret(
            [row for row in delta_rows if row["policy_family"] == family],
            name=f"profile_window_p99_regret_lines_{family}",
            output_dir=figure_dir,
        )

    print(
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "rows": len(rows),
            "figures": str(figure_dir),
        }
    )
    return 0


def _load_rows(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    rows: list[dict[str, Any]] = []
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            family = raw.get("policy_family")
            if family not in FAMILY_ORDER:
                continue
            rows.append(
                {
                    "scenario": raw.get("scenario"),
                    "policy": raw.get("policy"),
                    "policy_family": family,
                    "alpha": _float(raw.get("alpha")),
                    "window_min": _float(raw.get("window_min")),
                    "shift_period_min": _float(raw.get("shift_period_min")),
                    "shift_magnitude": _float(raw.get("shift_magnitude")),
                    "mean_ttft_ms": _float(raw.get("mean_ttft_ms")),
                    "p50_ms": _float(raw.get("p50_ms")),
                    "p99_ms": _float(raw.get("p99_ms")),
                    "slo_violation_rate": _float(raw.get("slo_violation_rate")),
                    "hedge_rate": _float(raw.get("hedge_rate")),
                    "mean_total_cost_usd": _float(raw.get("mean_total_cost_usd")),
                    "total_cost_usd": _float(raw.get("total_cost_usd")),
                    "profile_fallback_rate": _float(raw.get("profile_fallback_rate")),
                }
            )
    if not rows:
        raise ValueError(f"no profile-window rows found in {summary_path}")
    return rows


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute deltas of each observed cell vs its same-environment oracle."""
    oracle = {
        (row["scenario"], row["policy_family"], row["alpha"]): row
        for row in rows
        if row["window_min"] is None
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["window_min"] is None:
            continue
        base = oracle.get((row["scenario"], row["policy_family"], row["alpha"]))
        entry = dict(row)
        if base is not None:
            entry.update(
                {
                    "p99_delta_pct": _pct_delta(row["p99_ms"], base["p99_ms"]),
                    "mean_ttft_delta_pct": _pct_delta(row["mean_ttft_ms"], base["mean_ttft_ms"]),
                    "slo_delta_pp": (
                        (row["slo_violation_rate"] - base["slo_violation_rate"]) * 100.0
                        if None not in (row["slo_violation_rate"], base["slo_violation_rate"])
                        else None
                    ),
                    "total_cost_delta_pct": _pct_delta(
                        row["mean_total_cost_usd"], base["mean_total_cost_usd"]
                    ),
                }
            )
        output.append(entry)
    return sorted(
        output,
        key=lambda row: (
            row["shift_period_min"] or 0.0,
            FAMILY_ORDER.index(row["policy_family"]),
            row["window_min"],
        ),
    )


def _pct_delta(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0.0:
        return None
    return (value / base - 1.0) * 100.0


def _period_label(period_min: float | None) -> str:
    if not period_min:
        return "static"
    if period_min >= 1.0 and float(period_min).is_integer():
        return f"P={int(period_min)}m"
    return f"P={period_min:g}m"


def _window_axis(
    ax: plt.Axes,
    window_minutes: list[float],
    *,
    oracle_slot: bool = False,
) -> None:
    """Evenly spaced categorical x-axis, one slot per swept window value.

    With ``oracle_slot`` an extra rightmost position labels the
    configured-mode oracle cells, separated by a light vertical rule.
    """
    positions = list(range(len(window_minutes)))
    labels = [f"{value:g}" for value in window_minutes]
    if oracle_slot:
        separator = len(window_minutes) - 0.5
        ax.axvline(separator, color="#bbbbbb", linewidth=0.6)
        positions.append(len(window_minutes))
        labels.append("oracle")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Profile window (min)")


def _period_colors(period_minutes: list[float]) -> dict[float, str]:
    colors: dict[float, str] = {}
    cycle = iter(PERIOD_COLOR_CYCLE)
    for period in sorted(period_minutes):
        colors[period] = STATIC_COLOR if not period else next(cycle)
    return colors


def _period_series(
    rows: list[dict[str, Any]],
    *,
    period: float,
    metric: str,
) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if (row["shift_period_min"] or 0.0) == period
            and row["window_min"] is not None
            and row.get(metric) is not None
        ),
        key=lambda row: row["window_min"],
    )


def _finish_figure(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    handles: list[Any],
    labels: list[str],
    name: str,
    output_dir: Path,
) -> None:
    ax.grid(True, alpha=0.24)
    ax.legend(
        handles,
        labels,
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
    save_figure(fig, output_dir, name, formats=["pdf", "png"], full_canvas=True)
    plt.close(fig)


def _plot_metric_lines(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    name: str,
    with_oracle: bool,
    output_dir: Path,
) -> None:
    _apply_profile_window_style()
    periods = sorted({row["shift_period_min"] or 0.0 for row in rows})
    colors = _period_colors(periods)
    window_minutes = sorted({row["window_min"] for row in rows if row["window_min"] is not None})
    position = {window: index for index, window in enumerate(window_minutes)}
    oracle_position = len(window_minutes)
    fig, ax = plt.subplots(figsize=(2.3, 1.95))
    handles: list[Any] = []
    labels: list[str] = []
    drew_oracle = False
    for period in periods:
        selected = _period_series(rows, period=period, metric=metric)
        if not selected:
            continue
        (line,) = ax.plot(
            [position[row["window_min"]] for row in selected],
            [row[metric] for row in selected],
            marker="o",
            color=colors[period],
            linewidth=1.1,
            markersize=3.0,
            label=_period_label(period),
        )
        handles.append(line)
        labels.append(_period_label(period))
        if with_oracle:
            oracle = next(
                (
                    row
                    for row in rows
                    if (row["shift_period_min"] or 0.0) == period
                    and row["window_min"] is None
                    and row.get(metric) is not None
                ),
                None,
            )
            if oracle is not None:
                ax.plot(
                    [oracle_position],
                    [oracle[metric]],
                    marker="o",
                    color=colors[period],
                    markersize=3.4,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    linestyle="none",
                )
                drew_oracle = True
    _window_axis(ax, window_minutes, oracle_slot=drew_oracle)
    ax.set_ylabel(ylabel)
    _finish_figure(fig, ax, handles=handles, labels=labels, name=name, output_dir=output_dir)


def _plot_oracle_regret(
    delta_rows: list[dict[str, Any]],
    *,
    name: str,
    output_dir: Path,
) -> None:
    rows = [row for row in delta_rows if row.get("p99_delta_pct") is not None]
    if not rows:
        return
    _apply_profile_window_style()
    periods = sorted({row["shift_period_min"] or 0.0 for row in rows})
    colors = _period_colors(periods)
    window_minutes = sorted({row["window_min"] for row in rows})
    position = {window: index for index, window in enumerate(window_minutes)}
    fig, ax = plt.subplots(figsize=(2.3, 1.95))
    handles: list[Any] = []
    labels: list[str] = []
    for period in periods:
        selected = _period_series(rows, period=period, metric="p99_delta_pct")
        if not selected:
            continue
        (line,) = ax.plot(
            [position[row["window_min"]] for row in selected],
            [row["p99_delta_pct"] for row in selected],
            marker="o",
            color=colors[period],
            linewidth=1.1,
            markersize=3.0,
            label=_period_label(period),
        )
        handles.append(line)
        labels.append(_period_label(period))
    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.65)
    _window_axis(ax, window_minutes)
    ax.set_ylabel("P99 TTFT regret (%)")
    _finish_figure(fig, ax, handles=handles, labels=labels, name=name, output_dir=output_dir)


def _apply_profile_window_style() -> None:
    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        }
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
