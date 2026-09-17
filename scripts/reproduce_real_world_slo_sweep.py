"""Recompute the real-world SLO sweep and plot its panels.

One RouteWise operating point (alpha = 0.5 with hedging) replayed the same
24-hour trace four times, once per SLO target, so the router's own SLO is the
only variable. Violations are counted against each run's own target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from plots.end_to_end.frontier_plotting import (
    ANNOTATION_FONT_SIZE,
    PAPER_PANEL_FIGSIZE,
    ROUTEWISE_COLOR,
    aligned_panel_geometry,
    apply_column_figure_style,
)
from plots.end_to_end.plot_real_world_frontier import percentile, truthy

ROOT = Path(__file__).resolve().parents[1]
# Prorated MiniMax Plus plus Featherless Premium over the 24-hour window, the
# value the run recorded; see data/real_eval_slo_sweep_m3/README.md.
FIXED_COST_USD = 1.5
METRICS = ("total_cost_usd", "ttft_mean_ms", "ttft_p99_ms", "slo_violation_rate", "hedge_rate")


@dataclass(frozen=True)
class SloRun:
    policy: str
    slo_ms: float
    n: int
    total_cost_usd: float
    billed_cost_usd: float
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    slo_violation_rate: float
    success_rate: float
    hedge_rate: float
    hedge_backup_win_rate: float


def load_runs(source: Path) -> list[SloRun]:
    runs: list[SloRun] = []
    for policy_dir in sorted(source.glob("*/requests.csv")):
        policy_dir = policy_dir.parent
        args = json.loads((policy_dir / "args.json").read_text(encoding="utf-8"))
        slo_ms = float(args["slo_ms"])
        rows = list(csv.DictReader((policy_dir / "requests.csv").open(newline="")))
        if not rows:
            raise ValueError(f"{policy_dir}: requests.csv has no rows")
        successes = [
            row
            for row in rows
            if row.get("status") == "success"
            and row.get("ttft_ms") not in {"", None}
            and float(row["ttft_ms"]) >= 0.0
        ]
        ttft = [float(row["ttft_ms"]) for row in successes]
        billed = sum(float(row.get("billed_cost_usd") or 0.0) for row in rows)
        violations = sum(
            1
            for row in rows
            if row.get("status") != "success"
            or (row.get("ttft_ms") not in {"", None} and float(row["ttft_ms"]) > slo_ms)
        )
        hedged = sum(1 for row in rows if truthy(row.get("hedge_triggered")))
        backup_wins = sum(1 for row in rows if row.get("hedge_winner") == "backup")
        runs.append(
            SloRun(
                policy=policy_dir.name,
                slo_ms=slo_ms,
                n=len(rows),
                total_cost_usd=billed + FIXED_COST_USD,
                billed_cost_usd=billed,
                ttft_mean_ms=sum(ttft) / len(ttft),
                ttft_p50_ms=percentile(ttft, 50.0),
                ttft_p90_ms=percentile(ttft, 90.0),
                ttft_p99_ms=percentile(ttft, 99.0),
                slo_violation_rate=violations / len(rows),
                success_rate=len(successes) / len(rows),
                hedge_rate=hedged / len(rows),
                hedge_backup_win_rate=backup_wins / hedged if hedged else 0.0,
            )
        )
    if not runs:
        raise SystemExit(f"no policy directories with requests.csv under {source}")
    return sorted(runs, key=lambda run: run.slo_ms)


def check_summary(actual_path: Path, reference_path: Path) -> None:
    """Compare independently generated metrics with the archived aggregates."""
    actual = {row["policy"]: row for row in json.loads(actual_path.read_text(encoding="utf-8"))}
    expected = {
        row["policy"]: row for row in json.loads(reference_path.read_text(encoding="utf-8"))
    }
    if actual.keys() != expected.keys():
        raise ValueError("generated and reference summaries contain different runs")
    for policy, row in actual.items():
        if row["n"] != expected[policy]["n"]:
            raise ValueError(f"{policy}: request count differs from the reference")
        for metric in METRICS:
            value, reference = float(row[metric]), float(expected[policy][metric])
            if not math.isclose(value, reference, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{policy}: {metric}={value}, reference={reference}")
    print(f"PASS: {len(actual)} SLO targets match the archived request counts and metrics.")


def _slo_axes(runs: list[SloRun]):
    figsize, margins = aligned_panel_geometry(len(runs))
    fig, ax = plt.subplots(figsize=figsize or PAPER_PANEL_FIGSIZE, constrained_layout=False)
    _, right, bottom, top = margins
    fig.subplots_adjust(left=0.20, right=right, bottom=bottom, top=top)
    ax.set_xticks(range(len(runs)), [f"{run.slo_ms / 1000:g}" for run in runs])
    ax.set_xlabel("SLO target (s)")
    ax.grid(axis="y", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _save(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_violations(runs: list[SloRun], output_path: Path) -> None:
    apply_column_figure_style()
    fig, ax = _slo_axes(runs)
    values = [100.0 * run.slo_violation_rate for run in runs]
    ax.bar(range(len(runs)), values, color=ROUTEWISE_COLOR, width=0.62)
    for idx, value in enumerate(values):
        ax.annotate(
            f"{value:.2f}%",
            (idx, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=ANNOTATION_FONT_SIZE,
        )
    ax.set_ylabel("SLO violations (%)")
    ax.set_ylim(0, max(values) * 1.22)
    _save(fig, output_path)


def plot_hedge_and_cost(runs: list[SloRun], output_path: Path) -> None:
    apply_column_figure_style()
    fig, ax = _slo_axes(runs)
    values = [100.0 * run.hedge_rate for run in runs]
    ax.bar(range(len(runs)), values, color=ROUTEWISE_COLOR, width=0.62)
    for idx, run in enumerate(runs):
        ax.annotate(
            f"cost ${run.total_cost_usd:.2f}",
            (idx, values[idx]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=ANNOTATION_FONT_SIZE - 1.0,
        )
    ax.set_ylabel("Hedged requests (%)")
    ax.set_ylim(0, max(values) * 1.22)
    _save(fig, output_path)


def write_table(runs: list[SloRun], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{run.slo_ms / 1000:g}s & "
        f"{run.total_cost_usd:.3f} & "
        f"{run.ttft_mean_ms / 1000:.2f} & "
        f"{run.ttft_p99_ms / 1000:.2f} & "
        f"{100.0 * run.slo_violation_rate:.2f}\\% & "
        f"{100.0 * run.hedge_rate:.2f}\\% & "
        f"{100.0 * run.hedge_backup_win_rate:.2f}\\% \\\\"
        for run in runs
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "real_eval_slo_sweep_m3")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "figures" / "real_world_slo_sweep"
    )
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help="Overwrite the archived reference summary instead of checking against it.",
    )
    args = parser.parse_args(argv)
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    runs = load_runs(source)
    summary = output / "slo_sweep_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps([asdict(run) for run in runs], indent=2) + "\n", encoding="utf-8")
    plot_violations(runs, output / "slo_sweep_violations.pdf")
    plot_hedge_and_cost(runs, output / "slo_sweep_hedge_rate.pdf")
    write_table(runs, output / "slo_sweep_rows.tex")
    reference = source / "reference_summary.json"
    if args.write_reference:
        reference.write_text(
            json.dumps([asdict(run) for run in runs], indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {reference}")
    else:
        check_summary(summary, reference)
    print(f"Panels and the recomputed summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
