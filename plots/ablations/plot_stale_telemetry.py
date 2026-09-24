"""Plot the stale-telemetry robustness experiment (hedging under latency spikes).

Consumes ``summary.csv`` and ``spike_timeseries.csv`` written by
``experiments.ablations.stale_telemetry`` and produces:

- ``stale_telemetry_spike_summary.csv``: compact spike-phase table.
- ``figures/stale_telemetry_ts_<metric>__<scenario>.{pdf,png}``: onset-aligned
  time series (one line per policy, spike window shaded).
- ``figures/stale_telemetry_spike_<metric>_vs_magnitude__dur=<D>m.{pdf,png}``:
  spike-phase metric against spike magnitude, one line per policy.
"""

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

DEFAULT_INPUT_DIR = Path("outputs/ablations/stale_telemetry")

BELIEF_ORDER = ("stale", "observed", "fresh")
BELIEF_COLORS = {"stale": "#d62728", "observed": "#ff7f0e", "fresh": "#1f77b4"}
FAMILY_ORDER = ("lp", "hedge")
FAMILY_LABELS = {"lp": "LP", "hedge": "LP+Hedge"}
# Dense dash pattern so hedging stays distinguishable inside short legend handles.
HEDGE_DASHES = (0, (2.2, 1.2))
SPIKE_SHADE = "#bbbbbb"

# (metric key in spike_timeseries.csv, y-axis label, filename stem)
TIMESERIES_PANELS = (
    ("slo_violation_rate", "SLO violation rate", "stale_telemetry_ts_slo"),
    ("mean_ttft_ms", "Mean TTFT (ms)", "stale_telemetry_ts_mean_ttft"),
    ("hedge_rate", "Hedge rate", "stale_telemetry_ts_hedge_rate"),
    (
        "spiked_provider_final_share",
        "Served by spiked provider",
        "stale_telemetry_ts_spiked_share",
    ),
)
# (metric key in summary.csv, y-axis label, filename stem)
MAGNITUDE_PANELS = (
    ("spike_slo_violation_rate", "Spike SLO violation rate", "stale_telemetry_spike_slo"),
    ("spike_p99_ms", "Spike P99 TTFT (ms)", "stale_telemetry_spike_p99"),
    ("spike_mean_ttft_ms", "Spike mean TTFT (ms)", "stale_telemetry_spike_mean_ttft"),
    (
        "spike_cost_multiplier_vs_stale_lp",
        "Spike cost vs. stale LP (x)",
        "stale_telemetry_spike_cost_multiplier",
    ),
)
SUMMARY_COLUMNS = (
    "scenario",
    "policy",
    "belief_mode",
    "policy_family",
    "profile_window_min",
    "spike_magnitude",
    "spike_duration_min",
    "spiked_provider",
    "spike_episode_count",
    "baseline_slo_violation_rate",
    "spike_slo_violation_rate",
    "post_spike_slo_violation_rate",
    "spike_mean_ttft_ms",
    "spike_p99_ms",
    "spike_hedge_rate",
    "spike_spiked_provider_final_share",
    "spike_mean_cost_usd",
    "spike_slo_violation_delta_vs_stale_lp_pp",
    "spike_p99_reduction_vs_stale_lp_pct",
    "spike_cost_multiplier_vs_stale_lp",
    "profile_mean_fallback_rate",
    "profile_cdf_fallback_rate",
)


def main(argv: list[str] | None = None) -> int:
    """Plot the stale-telemetry experiment from its summary and time-series CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Experiment directory containing summary.csv and spike_timeseries.csv. "
            f"Defaults to {DEFAULT_INPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to --input-dir.",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    summary_rows = _load_csv(input_dir / "summary.csv")
    timeseries_rows = _load_csv(input_dir / "spike_timeseries.csv")
    if not summary_rows:
        raise ValueError(f"no rows found in {input_dir / 'summary.csv'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "stale_telemetry_spike_summary.csv",
        [{key: row.get(key) for key in SUMMARY_COLUMNS} for row in summary_rows],
        fieldnames=list(SUMMARY_COLUMNS),
    )

    figure_dir = output_dir / "figures"
    figures = 0
    for scenario in sorted({row["scenario"] for row in timeseries_rows}):
        scenario_rows = [row for row in timeseries_rows if row["scenario"] == scenario]
        duration_min = scenario_rows[0]["spike_duration_min"]
        for metric, ylabel, stem in TIMESERIES_PANELS:
            if not any(row.get(metric) is not None for row in scenario_rows):
                continue
            _plot_timeseries(
                scenario_rows,
                metric=metric,
                ylabel=ylabel,
                duration_min=duration_min,
                name=f"{stem}__{scenario}",
                output_dir=figure_dir,
            )
            figures += 1
    for duration_min in sorted({row["spike_duration_min"] for row in summary_rows}):
        duration_rows = [row for row in summary_rows if row["spike_duration_min"] == duration_min]
        if len({row["spike_magnitude"] for row in duration_rows}) < 1:
            continue
        for metric, ylabel, stem in MAGNITUDE_PANELS:
            if not any(row.get(metric) is not None for row in duration_rows):
                continue
            _plot_vs_magnitude(
                duration_rows,
                metric=metric,
                ylabel=ylabel,
                name=f"{stem}_vs_magnitude__dur={duration_min:g}m",
                output_dir=figure_dir,
            )
            figures += 1

    print(
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "summary_rows": len(summary_rows),
            "timeseries_rows": len(timeseries_rows),
            "figures": figures,
            "figures_dir": str(figure_dir),
        }
    )
    return 0


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append({key: _typed(value) for key, value in raw.items()})
    return rows


def _typed(value: str | None) -> Any:
    if value is None or value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def _policy_key(row: dict[str, Any]) -> tuple[int, int]:
    return (
        BELIEF_ORDER.index(row["belief_mode"]),
        FAMILY_ORDER.index(row["policy_family"]),
    )


def _policy_label(row: dict[str, Any]) -> str:
    belief = row["belief_mode"]
    if belief == "observed":
        window = row.get("profile_window_min")
        belief_text = "Rolling" if window is None else f"Rolling {window:g}m"
    else:
        belief_text = belief.capitalize()
    return f"{belief_text} {FAMILY_LABELS[row['policy_family']]}"


def _line_style(row: dict[str, Any]) -> dict[str, Any]:
    style: dict[str, Any] = {"color": BELIEF_COLORS[row["belief_mode"]]}
    if row["policy_family"] == "hedge":
        style["linestyle"] = HEDGE_DASHES
    return style


def _policies(rows: list[dict[str, Any]]) -> list[str]:
    by_policy = {row["policy"]: row for row in rows}
    return sorted(by_policy, key=lambda name: _policy_key(by_policy[name]))


def _plot_timeseries(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    duration_min: float,
    name: str,
    output_dir: Path,
) -> None:
    _apply_stale_telemetry_style()
    fig, ax = plt.subplots(figsize=(3.1, 2.05))
    ax.axvspan(0.0, float(duration_min), color=SPIKE_SHADE, alpha=0.45, linewidth=0)
    handles: list[Any] = []
    labels: list[str] = []
    for policy in _policies(rows):
        selected = sorted(
            (row for row in rows if row["policy"] == policy and row.get(metric) is not None),
            key=lambda row: row["minutes_since_onset"],
        )
        if not selected:
            continue
        (line,) = ax.plot(
            # Bins are one minute wide; draw each at its centre.
            [row["minutes_since_onset"] + 0.5 for row in selected],
            [row[metric] for row in selected],
            linewidth=1.1,
            **_line_style(selected[0]),
        )
        handles.append(line)
        labels.append(_policy_label(selected[0]))
    ax.set_xlabel("Minutes since spike onset")
    ax.set_ylabel(ylabel)
    ax.axvline(0.0, color="#444444", linewidth=0.6, alpha=0.6)
    _finish_figure(fig, ax, handles=handles, labels=labels, name=name, output_dir=output_dir)


def _plot_vs_magnitude(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    name: str,
    output_dir: Path,
) -> None:
    _apply_stale_telemetry_style()
    magnitudes = sorted({row["spike_magnitude"] for row in rows})
    position = {magnitude: index for index, magnitude in enumerate(magnitudes)}
    fig, ax = plt.subplots(figsize=(3.1, 2.05))
    handles: list[Any] = []
    labels: list[str] = []
    for policy in _policies(rows):
        selected = sorted(
            (row for row in rows if row["policy"] == policy and row.get(metric) is not None),
            key=lambda row: row["spike_magnitude"],
        )
        if not selected:
            continue
        (line,) = ax.plot(
            [position[row["spike_magnitude"]] for row in selected],
            [row[metric] for row in selected],
            marker="o",
            linewidth=1.1,
            markersize=3.0,
            **_line_style(selected[0]),
        )
        handles.append(line)
        labels.append(_policy_label(selected[0]))
    ax.set_xticks(list(position.values()))
    ax.set_xticklabels([f"{magnitude:g}x" for magnitude in magnitudes])
    ax.set_xlabel("Spike magnitude")
    ax.set_ylabel(ylabel)
    _finish_figure(fig, ax, handles=handles, labels=labels, name=name, output_dir=output_dir)


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
        columnspacing=0.8,
        handlelength=1.6,
        handletextpad=0.3,
    )
    fig.subplots_adjust(left=0.2, right=0.97, top=0.8, bottom=0.21)
    save_figure(fig, output_dir, name, formats=["pdf", "png"], full_canvas=True)
    plt.close(fig)


def _apply_stale_telemetry_style() -> None:
    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.0,
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


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
