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
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

from plots.helpers import save_figure
from plots.style import apply_style

DEFAULT_INPUT_DIR = Path("outputs/ablations/profile_window")

FAMILY_ORDER = ("lp", "hedge")
FAMILY_LABELS = {"lp": "LP", "hedge": "LP+Hedge"}
# One color per environment change period; static gets gray.
PERIOD_COLOR_CYCLE = ("#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b")
STATIC_COLOR = "#7f7f7f"

METRIC_PANELS = (
    ("p99_ms", "P99 TTFT (ms)", "profile_window_p99_lines"),
    ("slo_violation_rate", "SLO violation rate", "profile_window_slo_lines"),
    ("mean_total_cost_usd", "Mean total cost (USD)", "profile_window_cost_lines"),
    ("profile_fallback_rate", "Profile fallback rate", "profile_window_fallback_lines"),
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
    for metric, ylabel, figure_name in METRIC_PANELS:
        if not any(row.get(metric) is not None for row in rows):
            continue
        _plot_metric_lines(rows, metric=metric, ylabel=ylabel, name=figure_name, output_dir=figure_dir)
    _plot_oracle_regret(delta_rows, figure_dir)

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


def _window_log_axis(ax: plt.Axes, window_minutes: list[float]) -> None:
    """Log x-axis with one plain-number tick per swept window value."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(window_minutes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([f"{value:g}" for value in window_minutes])
    ax.set_xlabel("Profile window (min)")


def _period_colors(period_minutes: list[float]) -> dict[float, str]:
    colors: dict[float, str] = {}
    cycle = iter(PERIOD_COLOR_CYCLE)
    for period in sorted(period_minutes):
        colors[period] = STATIC_COLOR if not period else next(cycle)
    return colors


def _plot_metric_lines(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    name: str,
    output_dir: Path,
) -> None:
    apply_style("paper")
    periods = sorted({row["shift_period_min"] or 0.0 for row in rows})
    colors = _period_colors(periods)
    families = [
        family
        for family in FAMILY_ORDER
        if any(row["policy_family"] == family for row in rows)
    ]
    fig, axes = plt.subplots(
        1,
        len(families),
        figsize=(3.3 * len(families), 2.6),
        sharey=True,
        squeeze=False,
    )
    for ax, family in zip(axes[0], families, strict=False):
        for period in periods:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["policy_family"] == family
                    and (row["shift_period_min"] or 0.0) == period
                    and row["window_min"] is not None
                    and row.get(metric) is not None
                ),
                key=lambda row: row["window_min"],
            )
            if not selected:
                continue
            ax.plot(
                [row["window_min"] for row in selected],
                [row[metric] for row in selected],
                marker="o",
                color=colors[period],
                label=_period_label(period),
            )
            oracle = next(
                (
                    row
                    for row in rows
                    if row["policy_family"] == family
                    and (row["shift_period_min"] or 0.0) == period
                    and row["window_min"] is None
                    and row.get(metric) is not None
                ),
                None,
            )
            if oracle is not None:
                ax.axhline(
                    oracle[metric],
                    color=colors[period],
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.6,
                )
        window_minutes = sorted(
            {row["window_min"] for row in rows if row["window_min"] is not None}
        )
        _window_log_axis(ax, window_minutes)
        ax.set_title(FAMILY_LABELS[family], pad=4)
        ax.grid(True, alpha=0.24)
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    save_figure(fig, output_dir, name, ["pdf", "png"])
    plt.close(fig)


def _plot_oracle_regret(delta_rows: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [row for row in delta_rows if row.get("p99_delta_pct") is not None]
    if not rows:
        return
    apply_style("paper")
    periods = sorted({row["shift_period_min"] or 0.0 for row in rows})
    colors = _period_colors(periods)
    families = [
        family
        for family in FAMILY_ORDER
        if any(row["policy_family"] == family for row in rows)
    ]
    fig, axes = plt.subplots(
        1,
        len(families),
        figsize=(3.3 * len(families), 2.6),
        sharey=True,
        squeeze=False,
    )
    for ax, family in zip(axes[0], families, strict=False):
        for period in periods:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["policy_family"] == family
                    and (row["shift_period_min"] or 0.0) == period
                ),
                key=lambda row: row["window_min"],
            )
            if not selected:
                continue
            ax.plot(
                [row["window_min"] for row in selected],
                [row["p99_delta_pct"] for row in selected],
                marker="o",
                color=colors[period],
                label=_period_label(period),
            )
        ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.65)
        window_minutes = sorted({row["window_min"] for row in rows})
        _window_log_axis(ax, window_minutes)
        ax.set_title(FAMILY_LABELS[family], pad=4)
        ax.grid(True, alpha=0.24)
    axes[0][0].set_ylabel("P99 TTFT regret vs oracle (%)")
    axes[0][-1].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    save_figure(fig, output_dir, "profile_window_p99_regret_lines", ["pdf", "png"])
    plt.close(fig)


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
