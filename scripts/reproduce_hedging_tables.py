"""Recompute the two hedging tables of the revision from the committed records.

Table 1 asks what hedging buys at each RouteWise operating point: how often it
fires, how often the backup wins, and what happens to violations, mean TTFT,
P99 and cost. The "without hedge" column is the counterfactual the same run
supports: the primary leg answered on its own, so its observed time to first
token is what the request would have seen had no backup been dispatched.

Table 2 asks how that changes with the SLO target, using the sweep in which
one operating point replayed the same trace at four targets.

    uv run python scripts/reproduce_hedging_tables.py
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
from plots.end_to_end.plot_real_world_frontier import parse_alpha, percentile, truthy

ROOT = Path(__file__).resolve().parents[1]
HEDGING_ALPHAS = (0, 25, 50, 75, 100)
# Prorated MiniMax Plus plus Featherless Premium over the 24-hour window, the
# value the runs recorded.
FIXED_COST_USD = 1.5
HEDGING_METRICS = (
    "hedge_trigger_rate",
    "hedge_backup_win_rate",
    "hedged_meeting_slo_rate",
    "slo_violation_rate",
    "slo_violation_rate_no_hedge",
    "ttft_mean_ms",
    "ttft_mean_ms_no_hedge",
    "ttft_p99_ms",
    "ttft_p99_ms_no_hedge",
    "hedge_extra_cost_usd",
    "hedge_extra_cost_share_of_billed",
)
SLO_METRICS = ("total_cost_usd", "ttft_mean_ms", "ttft_p99_ms", "slo_violation_rate", "hedge_rate")


def _float(value: str | None) -> float | None:
    return None if value in {"", None} else float(value)


@dataclass(frozen=True)
class HedgingRow:
    policy: str
    alpha: float
    slo_ms: float
    n: int
    hedge_trigger_rate: float
    hedge_backup_win_rate: float
    hedged_meeting_slo_rate: float
    slo_violation_rate: float
    slo_violation_rate_no_hedge: float
    ttft_mean_ms: float
    ttft_mean_ms_no_hedge: float
    ttft_p99_ms: float
    ttft_p99_ms_no_hedge: float
    billed_cost_usd: float
    total_cost_usd: float
    hedge_extra_cost_usd: float
    # Share of the run's metered spend. The subscription tiers cost the same
    # whether or not a request is hedged, so the metered spend is the
    # denominator that hedging can actually move.
    hedge_extra_cost_share_of_billed: float
    censored_hedges: int


@dataclass(frozen=True)
class SloRow:
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


def _read_policy(policy_dir: Path) -> tuple[list[dict[str, str]], float]:
    with (policy_dir / "requests.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{policy_dir}: requests.csv has no rows")
    args = json.loads((policy_dir / "args.json").read_text(encoding="utf-8"))
    return rows, float(args["slo_ms"])


def _violates(row: dict[str, str], slo_ms: float) -> bool:
    ttft = _float(row.get("ttft_ms"))
    return row.get("status") != "success" or (ttft is not None and ttft > slo_ms)


def load_hedging_rows(source: Path) -> list[HedgingRow]:
    rows_out: list[HedgingRow] = []
    for alpha in HEDGING_ALPHAS:
        policy = f"budget_range_alpha{alpha}_hedge"
        policy_dir = source / policy
        rows, slo_ms = _read_policy(policy_dir)
        with (policy_dir / "hedge_legs.csv").open(newline="", encoding="utf-8") as handle:
            legs = {leg["req_id"]: leg for leg in csv.DictReader(handle)}
        if set(legs) != {row["req_id"] for row in rows}:
            raise ValueError(f"{policy}: hedge_legs.csv does not cover requests.csv")

        observed: list[float] = []
        counterfactual: list[float] = []
        violations = violations_no_hedge = 0
        scored_no_hedge = censored = 0
        hedged = backup_wins = hedged_meeting_slo = 0
        billed = extra_cost = 0.0
        for row in rows:
            leg = legs[row["req_id"]]
            billed += _float(row.get("billed_cost_usd")) or 0.0
            ttft = _float(row.get("ttft_ms"))
            success = row.get("status") == "success"
            if success and ttft is not None and ttft >= 0.0:
                observed.append(ttft)
            if _violates(row, slo_ms):
                violations += 1
            if not truthy(leg.get("hedge_triggered")):
                # No backup was dispatched, so the request is its own control.
                if success and ttft is not None and ttft >= 0.0:
                    counterfactual.append(ttft)
                if _violates(row, slo_ms):
                    violations_no_hedge += 1
                scored_no_hedge += 1
                continue
            hedged += 1
            if leg.get("hedge_winner") == "backup":
                backup_wins += 1
            if success and ttft is not None and 0.0 <= ttft <= slo_ms:
                hedged_meeting_slo += 1
            primary_ttft = _float(leg.get("primary_ttft_ms"))
            if primary_ttft is None:
                # The primary produced no token before it was canceled, so the
                # counterfactual is censored; see the table notes.
                censored += 1
                continue
            counterfactual.append(primary_ttft)
            scored_no_hedge += 1
            if primary_ttft > slo_ms:
                violations_no_hedge += 1
            extra_cost += _float(leg.get("loser_billed_cost_usd")) or 0.0

        rows_out.append(
            HedgingRow(
                policy=policy,
                alpha=parse_alpha(policy) or 0.0,
                slo_ms=slo_ms,
                n=len(rows),
                hedge_trigger_rate=hedged / len(rows),
                hedge_backup_win_rate=backup_wins / hedged if hedged else 0.0,
                hedged_meeting_slo_rate=hedged_meeting_slo / hedged if hedged else 0.0,
                slo_violation_rate=violations / len(rows),
                slo_violation_rate_no_hedge=violations_no_hedge / scored_no_hedge,
                ttft_mean_ms=sum(observed) / len(observed),
                ttft_mean_ms_no_hedge=sum(counterfactual) / len(counterfactual),
                ttft_p99_ms=percentile(observed, 99.0),
                ttft_p99_ms_no_hedge=percentile(counterfactual, 99.0),
                billed_cost_usd=billed,
                total_cost_usd=billed + FIXED_COST_USD,
                hedge_extra_cost_usd=extra_cost,
                hedge_extra_cost_share_of_billed=(extra_cost / billed if billed else 0.0),
                censored_hedges=censored,
            )
        )
    return rows_out


def load_slo_rows(source: Path) -> list[SloRow]:
    rows_out: list[SloRow] = []
    for policy_csv in sorted(source.glob("*/requests.csv")):
        policy_dir = policy_csv.parent
        rows, slo_ms = _read_policy(policy_dir)
        successes = [
            row
            for row in rows
            if row.get("status") == "success"
            and row.get("ttft_ms") not in {"", None}
            and float(row["ttft_ms"]) >= 0.0
        ]
        ttft = [float(row["ttft_ms"]) for row in successes]
        billed = sum(_float(row.get("billed_cost_usd")) or 0.0 for row in rows)
        hedged = sum(1 for row in rows if truthy(row.get("hedge_triggered")))
        backup_wins = sum(1 for row in rows if row.get("hedge_winner") == "backup")
        rows_out.append(
            SloRow(
                policy=policy_dir.name,
                slo_ms=slo_ms,
                n=len(rows),
                total_cost_usd=billed + FIXED_COST_USD,
                billed_cost_usd=billed,
                ttft_mean_ms=sum(ttft) / len(ttft),
                ttft_p50_ms=percentile(ttft, 50.0),
                ttft_p90_ms=percentile(ttft, 90.0),
                ttft_p99_ms=percentile(ttft, 99.0),
                slo_violation_rate=sum(1 for row in rows if _violates(row, slo_ms)) / len(rows),
                success_rate=len(successes) / len(rows),
                hedge_rate=hedged / len(rows),
                hedge_backup_win_rate=backup_wins / hedged if hedged else 0.0,
            )
        )
    return sorted(rows_out, key=lambda row: row.slo_ms)


def check_summary(actual: list, reference_path: Path, metrics: tuple[str, ...]) -> None:
    """Compare independently generated metrics with the archived aggregates."""
    generated = {row["policy"]: row for row in actual}
    expected = {
        row["policy"]: row for row in json.loads(reference_path.read_text(encoding="utf-8"))
    }
    if generated.keys() != expected.keys():
        raise ValueError(f"{reference_path.name}: generated and reference rows differ")
    for policy, row in generated.items():
        if row["n"] != expected[policy]["n"]:
            raise ValueError(f"{policy}: request count differs from the reference")
        for metric in metrics:
            value, archived = float(row[metric]), float(expected[policy][metric])
            if not math.isclose(value, archived, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{policy}: {metric}={value}, reference={archived}")
    print(f"PASS: {len(generated)} rows match {reference_path.name}.")


def write_hedging_table(rows: list[HedgingRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{row.alpha:g} & "
        f"{100.0 * row.hedge_trigger_rate:.2f}\\% & "
        f"{100.0 * row.hedge_backup_win_rate:.1f}\\% & "
        f"{100.0 * row.hedged_meeting_slo_rate:.1f}\\% & "
        f"{100.0 * row.slo_violation_rate:.2f}\\% / {100.0 * row.slo_violation_rate_no_hedge:.2f}\\% & "
        f"{row.ttft_mean_ms / 1000.0:.2f} / {row.ttft_mean_ms_no_hedge / 1000.0:.2f} & "
        f"{row.ttft_p99_ms / 1000.0:.2f} / {row.ttft_p99_ms_no_hedge / 1000.0:.2f} & "
        f"{row.hedge_extra_cost_usd:.4f} ({100.0 * row.hedge_extra_cost_share_of_billed:.2f}\\%) \\\\"
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def write_slo_table(rows: list[SloRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{row.slo_ms / 1000:g}s & "
        f"{row.total_cost_usd:.3f} & "
        f"{row.ttft_mean_ms / 1000.0:.2f} & "
        f"{row.ttft_p99_ms / 1000.0:.2f} & "
        f"{100.0 * row.slo_violation_rate:.2f}\\% & "
        f"{100.0 * row.hedge_rate:.2f}\\% & "
        f"{100.0 * row.hedge_backup_win_rate:.2f}\\% \\\\"
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _slo_axes(rows: list[SloRow]):
    figsize, margins = aligned_panel_geometry(len(rows))
    fig, ax = plt.subplots(figsize=figsize or PAPER_PANEL_FIGSIZE, constrained_layout=False)
    _, right, bottom, top = margins
    fig.subplots_adjust(left=0.20, right=right, bottom=bottom, top=top)
    ax.set_xticks(range(len(rows)), [f"{row.slo_ms / 1000:g}" for row in rows])
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


def plot_slo_violations(rows: list[SloRow], output_path: Path) -> None:
    apply_column_figure_style()
    fig, ax = _slo_axes(rows)
    values = [100.0 * row.slo_violation_rate for row in rows]
    ax.bar(range(len(rows)), values, color=ROUTEWISE_COLOR, width=0.62)
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


def plot_slo_hedge_rate(rows: list[SloRow], output_path: Path) -> None:
    apply_column_figure_style()
    fig, ax = _slo_axes(rows)
    values = [100.0 * row.hedge_rate for row in rows]
    ax.bar(range(len(rows)), values, color=ROUTEWISE_COLOR, width=0.62)
    for idx, row in enumerate(rows):
        ax.annotate(
            f"cost ${row.total_cost_usd:.2f}",
            (idx, values[idx]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=ANNOTATION_FONT_SIZE - 1.0,
        )
    ax.set_ylabel("Hedged requests (%)")
    ax.set_ylim(0, max(values) * 1.22)
    _save(fig, output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=ROOT / "data" / "real_eval_records_m3")
    parser.add_argument(
        "--slo-sweep-dir", type=Path, default=ROOT / "data" / "real_eval_slo_sweep_m3"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "figures" / "hedging_tables"
    )
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help="Overwrite the archived reference summaries instead of checking against them.",
    )
    args = parser.parse_args(argv)
    records, sweep = args.records_dir.resolve(), args.slo_sweep_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    hedging = [asdict(row) for row in load_hedging_rows(records)]
    slo = [asdict(row) for row in load_slo_rows(sweep)]
    (output / "table1_hedging.json").write_text(
        json.dumps(hedging, indent=2) + "\n", encoding="utf-8"
    )
    (output / "table2_slo_sweep.json").write_text(
        json.dumps(slo, indent=2) + "\n", encoding="utf-8"
    )
    write_hedging_table(load_hedging_rows(records), output / "table1_hedging.tex")
    write_slo_table(load_slo_rows(sweep), output / "table2_slo_sweep.tex")
    plot_slo_violations(load_slo_rows(sweep), output / "slo_sweep_violations.pdf")
    plot_slo_hedge_rate(load_slo_rows(sweep), output / "slo_sweep_hedge_rate.pdf")

    hedging_reference = records / "hedging_reference_summary.json"
    slo_reference = sweep / "reference_summary.json"
    if args.write_reference:
        hedging_reference.write_text(json.dumps(hedging, indent=2) + "\n", encoding="utf-8")
        slo_reference.write_text(json.dumps(slo, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {hedging_reference}\nwrote {slo_reference}")
    else:
        check_summary(hedging, hedging_reference, HEDGING_METRICS)
        check_summary(slo, slo_reference, SLO_METRICS)
    print(f"Tables and panels: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
