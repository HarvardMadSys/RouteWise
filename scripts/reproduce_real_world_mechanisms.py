"""Measure how RouteWise's components behaved in the MiniMax-M3 24-hour run.

Three measurements, from the committed records and the per-decision router
state of the five RouteWise runs:

1. Concurrency-limited provider. The table asks whether the LP considered the
   concurrency slot whenever it was free and took it whenever it was the
   lowest-latency candidate.
2. Quota-limited provider. The figure plots quota consumed in the current
   five-hour window over the day for RouteWise, Greedy-cost (the live policy
   closest to "use quota whenever it is available") and an offline replay that
   sends every arrival to quota until the window is exhausted.
3. What the budget reserves quota for. RouteWise never exhausts a window, and
   its quota use rises with alpha although a higher alpha is the less
   cost-sensitive setting. The figure resolves that: a tight budget does not
   use less quota indiscriminately, it spends quota on the long responses,
   where the metered alternative is dear. It is measured over the decisions
   where the concurrency slot was busy, the only ones in which the budget
   weighs quota against a metered price.

    uv run python scripts/reproduce_real_world_mechanisms.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.real_evaluation.inventory import load_inventory
from plots.end_to_end.frontier_plotting import (
    ANNOTATION_FONT_SIZE,
    PAPER_PANEL_FIGSIZE,
    POLICY_COLORS,
    POLICY_PLOT_LABELS,
    ROUTEWISE_COLOR,
    apply_column_figure_style,
)
from plots.end_to_end.plot_real_world_frontier import parse_alpha

ROOT = Path(__file__).resolve().parents[1]
ROUTEWISE_POLICIES = tuple(f"budget_range_alpha{alpha}_hedge" for alpha in (0, 25, 50, 75, 100))
BASELINE_POLICIES = ("greedy_latency", "greedy_cost")
QUOTA_FIRST_OFFLINE = "quota_first_offline"
# The two line figures carry their legend below the axes, so they are taller
# than the shared panel size.
LEGEND_BELOW_FIGSIZE = (PAPER_PANEL_FIGSIZE[0], 3.9)
CONCURRENCY_METRICS = (
    "free_slot_decisions",
    "offered_when_free_rate",
    "lowest_latency_when_free_rate",
    "picked_when_lowest_latency_rate",
    "picked_when_not_lowest_latency_rate",
    "share_of_requests",
)
# Coarser than the tier-mix bins: these count only the decisions where the
# concurrency slot was busy, and the long bins are thin.
QUOTA_LENGTH_BIN_EDGES = (0, 10, 50, 200, float("inf"))
QUOTA_LENGTH_BIN_LABELS = ("1-10", "11-50", "51-200", ">200")
# Sequential, because the bins are ordered; the ramp also survives greyscale.
QUOTA_LENGTH_COLORS = ("#dbe6ec", "#9dbdcf", "#4f86a0", "#1d4657")
QUOTA_METRICS = (
    "quota_requests",
    "peak_window_quota_requests",
    "peak_fraction_used",
    "requests_after_exhaustion",
)
QUOTA_LENGTH_METRICS = tuple(f"share_{name}" for name in QUOTA_LENGTH_BIN_LABELS)


def _alpha_color(alpha: float) -> str:
    """RouteWise teal, lighter for low alpha and darker for high alpha."""
    base = np.array(to_rgb(ROUTEWISE_COLOR))
    if alpha < 0.5:
        mix = 0.5 * (0.5 - alpha) / 0.5
        return to_hex(base + (1.0 - base) * mix)
    mix = 0.6 * (alpha - 0.5) / 0.5
    return to_hex(base * (1.0 - mix))


def _label(policy: str) -> str:
    alpha = parse_alpha(policy)
    if alpha is not None:
        return rf"RouteWise $\alpha={alpha:g}$"
    if policy == QUOTA_FIRST_OFFLINE:
        return "Quota whenever available"
    return POLICY_PLOT_LABELS.get(policy, policy)


def _color(policy: str) -> str:
    alpha = parse_alpha(policy)
    if alpha is not None:
        return _alpha_color(alpha)
    if policy == QUOTA_FIRST_OFFLINE:
        return "#7f7f7f"
    return POLICY_COLORS.get(policy, "#555555")


class Records:
    """One policy's requests joined with its router state, when it has one."""

    def __init__(self, policy_dir: Path) -> None:
        self.policy = policy_dir.name
        self.alpha = parse_alpha(self.policy)
        frame = pd.read_csv(policy_dir / "requests.csv", dtype={"req_id": str})
        # ts is the row-write time at completion; dispatch is what capacity
        # accounting and the trace timeline run on.
        frame["dispatch"] = frame["ts"] - frame["e2e_ms"] / 1000.0
        frame = frame.sort_values("dispatch").reset_index(drop=True)
        frame["hours"] = (frame["dispatch"] - frame["dispatch"].iloc[0]) / 3600.0
        state_path = policy_dir / "router_state.csv.gz"
        if state_path.exists():
            with gzip.open(state_path, "rt", newline="", encoding="utf-8") as handle:
                state = pd.read_csv(handle, dtype={"req_id": str, "unavailable": str})
            if set(state["req_id"]) != set(frame["req_id"]):
                raise ValueError(f"{self.policy}: router_state.csv.gz does not cover requests.csv")
            frame = frame.merge(state, on="req_id", how="left", validate="one_to_one")
            frame["unavailable"] = frame["unavailable"].fillna("")
        self.frame = frame
        self.args = json.loads((policy_dir / "args.json").read_text(encoding="utf-8"))

    @property
    def has_state(self) -> bool:
        return "c_min_usd" in self.frame.columns


def load_records(source: Path, policies: tuple[str, ...]) -> dict[str, Records]:
    return {policy: Records(source / policy) for policy in policies}


def inventory_for(records: Records):
    return load_inventory(ROOT / records.args["inventory"])


def provider_by_tier(inventory, tier: str) -> str:
    names = [spec.name for spec in inventory.providers if spec.tier == tier]
    if len(names) != 1:
        raise ValueError(f"expected one {tier!r} provider, found {names}")
    return names[0]


# ---------------------------------------------------------------------------
# 1. Concurrency-limited provider: offered whenever free, taken when fastest.
# ---------------------------------------------------------------------------


def concurrency_row(records: Records, inventory) -> dict[str, float]:
    concurrency = provider_by_tier(inventory, "concurrency")
    limit = next(
        spec.concurrency_limit for spec in inventory.providers if spec.tier == "concurrency"
    )
    frame = records.frame
    latency_cols = [
        column for column in frame.columns if column.startswith("latency_objective_ms:")
    ]
    free = frame["concurrency_in_flight"] < limit
    offered = ~frame["unavailable"].str.contains(concurrency)
    lowest = frame[f"latency_objective_ms:{concurrency}"] <= frame[latency_cols].min(axis=1)
    picked = frame["primary_provider"] == concurrency
    when_free = free & offered
    return {
        "policy": records.policy,
        "alpha": records.alpha,
        "n": len(frame),
        "free_slot_decisions": int(free.sum()),
        "offered_when_free_rate": float(offered[free].mean()),
        "lowest_latency_when_free_rate": float(lowest[when_free].mean()),
        "picked_when_lowest_latency_rate": float(picked[when_free & lowest].mean()),
        "picked_when_not_lowest_latency_rate": float(picked[when_free & ~lowest].mean()),
        "picked_when_busy": int(picked[~free].sum()),
        "share_of_requests": float(picked.mean()),
    }


def write_concurrency_table(rows: list[dict], path: Path) -> None:
    lines = [
        f"{row['alpha']:g} & "
        f"{row['free_slot_decisions']:,} & "
        f"{100.0 * row['offered_when_free_rate']:.0f}\\% & "
        f"{100.0 * row['lowest_latency_when_free_rate']:.0f}\\% & "
        f"{100.0 * row['picked_when_lowest_latency_rate']:.0f}\\% & "
        f"{100.0 * row['picked_when_not_lowest_latency_rate']:.0f}\\% & "
        f"{100.0 * row['share_of_requests']:.0f}\\% \\\\"
        for row in rows
    ]
    _write_text(path, lines)


# ---------------------------------------------------------------------------
# 2. Quota-limited provider: consumption over the day.
# ---------------------------------------------------------------------------


def quota_windows(inventory) -> tuple[float, int]:
    spec = next(spec for spec in inventory.providers if spec.tier == "quota")
    shortest = min(spec.quota_windows, key=lambda window: window.window_sec)
    return float(shortest.window_sec), int(shortest.requests)


def quota_legs(frame: pd.DataFrame, quota: str) -> pd.Series:
    """Requests that charged quota: a primary or a hedge backup on the quota tier."""
    return (frame["primary_provider"] == quota) | (frame["backup_provider"] == quota)


def quota_timeline(frame: pd.DataFrame, used: pd.Series, window_sec: float) -> pd.DataFrame:
    """Quota used so far in the current window, one point per dispatch."""
    window = (frame["dispatch"] - frame["dispatch"].iloc[0]) // window_sec
    return pd.DataFrame(
        {"hours": frame["hours"], "window": window, "used": used.groupby(window).cumsum()}
    )


def quota_first_offline_usage(frame: pd.DataFrame, window_sec: float, size: int) -> pd.Series:
    """Every arrival takes quota until the window is exhausted (the trace's own order)."""
    window = (frame["dispatch"] - frame["dispatch"].iloc[0]) // window_sec
    rank = window.groupby(window).cumcount()
    return rank < size


def quota_row(
    policy: str,
    frame: pd.DataFrame,
    used: pd.Series,
    window_sec: float,
    size: int,
) -> dict[str, float]:
    """The numbers the figure draws, kept so the reference check can see them."""
    timeline = quota_timeline(frame, used, window_sec)
    per_window = timeline.groupby("window")["used"].max()
    peak_window = int(per_window.idxmax())
    exhausted = timeline[(timeline["window"] == peak_window) & (timeline["used"] >= size)]
    exhausted_at = float(exhausted["hours"].iloc[0]) if len(exhausted) else None
    after = 0
    if exhausted_at is not None:
        in_window = timeline["window"] == peak_window
        after = int((in_window & (timeline["hours"] >= exhausted_at)).sum()) - 1
    return {
        "policy": policy,
        "alpha": parse_alpha(policy),
        "n": len(frame),
        "quota_requests": int(used.sum()),
        "quota_per_window": [int(value) for value in per_window],
        "peak_window": peak_window,
        "peak_window_quota_requests": int(per_window.max()),
        "peak_fraction_used": float(per_window.max() / size),
        "exhausted_at_hours": exhausted_at,
        "requests_after_exhaustion": after,
    }


def plot_quota_over_time(
    timelines: dict[str, pd.DataFrame],
    span_hours: float,
    window_sec: float,
    size: int,
    output: Path,
) -> None:
    apply_column_figure_style(legend_fontsize=ANNOTATION_FONT_SIZE - 1)
    fig, ax = plt.subplots(figsize=LEGEND_BELOW_FIGSIZE)
    span = float(span_hours)
    for boundary in np.arange(window_sec / 3600.0, span, window_sec / 3600.0):
        ax.axvline(boundary, color="#bbbbbb", linewidth=0.6, linestyle=":")
    ax.axhline(size, color="#444444", linewidth=0.8, linestyle="--")
    ax.annotate(
        f"quota {size:,}",
        (math.ceil(span) - 0.3, size),
        xytext=(0, -9),
        textcoords="offset points",
        fontsize=ANNOTATION_FONT_SIZE - 1,
        ha="right",
        color="#444444",
    )
    for policy, timeline in timelines.items():
        style = {"linestyle": "--"} if policy == QUOTA_FIRST_OFFLINE else {}
        # Break the line at each window reset instead of drawing the drop.
        plotted = timeline.copy()
        first = plotted["window"] != plotted["window"].shift()
        plotted.loc[first & (plotted.index > 0), "used"] = np.nan
        ax.plot(
            plotted["hours"],
            plotted["used"],
            linewidth=1.4,
            color=_color(policy),
            label=_label(policy),
            **style,
        )
    ax.set_xlim(0, math.ceil(span))
    ax.set_ylim(0, size * 1.12)
    ax.set_xlabel("hour of the run")
    ax.set_ylabel(f"quota used in {window_sec / 3600:g} h window")
    ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        fontsize=ANNOTATION_FONT_SIZE - 1,
        columnspacing=1.0,
    )
    fig.tight_layout()
    _save(fig, output)


# ---------------------------------------------------------------------------
# 3. What the budget reserves quota for.
# ---------------------------------------------------------------------------


def contested_decisions(records: Records, quota: str, concurrency: str) -> pd.Series:
    """Decisions that were a real quota-versus-metered choice.

    The concurrency slot is free of charge, so whenever it is available the
    budget never has to weigh quota against a metered price. Restricting to the
    decisions where the slot was busy isolates the comparison the budget makes,
    and keeps the five operating points comparable even though they leave the
    slot busy for different shares of the day.
    """
    frame = records.frame
    return frame["unavailable"].str.contains(concurrency) & frame[f"c_eff:{quota}"].notna()


def length_mix(lengths: pd.Series) -> pd.Series:
    """Share of the requests falling in each response-length bin."""
    bins = pd.cut(lengths, QUOTA_LENGTH_BIN_EDGES, labels=list(QUOTA_LENGTH_BIN_LABELS))
    counts = bins.value_counts().reindex(list(QUOTA_LENGTH_BIN_LABELS)).fillna(0)
    return counts / counts.sum()


def quota_length_mixes(
    routewise: dict[str, Records], quota: str, concurrency: str
) -> list[dict[str, float]]:
    """Length mix of the quota traffic, with the two pools it is drawn from.

    The reference rows matter because the three distributions differ. The
    concurrency slot takes most long requests whenever it is free, so the pool
    of decisions in which the budget actually has to weigh quota against a
    metered price holds far fewer long requests than the trace does. Comparing
    quota's mix with the trace alone would understate how much the budget
    favours long requests.
    """
    rows: list[dict[str, float]] = []
    for policy, records in routewise.items():
        frame = records.frame
        contested = contested_decisions(records, quota, concurrency)
        served = frame[contested & (frame["primary_provider"] == quota)]
        rows.append(
            {
                "policy": policy,
                "label": _label(policy),
                "alpha": records.alpha,
                "n": len(served),
                **{
                    f"share_{name}": value
                    for name, value in length_mix(served["max_tokens"]).items()
                },
            }
        )
    reference = next(iter(routewise.values()))
    frame = reference.frame
    contested = contested_decisions(reference, quota, concurrency)
    for name, label, lengths in (
        (
            "contested",
            f"Contested ($\\alpha={reference.alpha:g}$)",
            frame.loc[contested, "max_tokens"],
        ),
        ("trace", "All requests", frame["max_tokens"]),
    ):
        rows.append(
            {
                "policy": name,
                "label": label,
                "alpha": None,
                "n": len(lengths),
                **{f"share_{name}": value for name, value in length_mix(lengths).items()},
            }
        )
    return rows


def plot_quota_length_mix(rows: list[dict[str, float]], output: Path) -> None:
    apply_column_figure_style(legend_fontsize=ANNOTATION_FONT_SIZE - 1)
    fig, ax = plt.subplots(figsize=(PAPER_PANEL_FIGSIZE[0], 3.05))
    # A gap between the operating points and the two reference rows.
    positions = [index + (0.6 if row["alpha"] is None else 0.0) for index, row in enumerate(rows)]
    left = np.zeros(len(rows))
    for name, color in zip(QUOTA_LENGTH_BIN_LABELS, QUOTA_LENGTH_COLORS, strict=True):
        widths = np.array([100.0 * row[f"share_{name}"] for row in rows])
        ax.barh(positions, widths, left=left, height=0.68, color=color, label=name)
        for position, width, start in zip(positions, widths, left, strict=True):
            if width >= 7.0:
                ax.text(
                    start + width / 2.0,
                    position,
                    f"{width:.0f}",
                    ha="center",
                    va="center",
                    fontsize=ANNOTATION_FONT_SIZE - 1.5,
                    color="white" if color in QUOTA_LENGTH_COLORS[2:] else "#222222",
                )
        left += widths
    # The claim lives in the two dark segments, so spell their sum out rather
    # than making the reader add them.
    long_labels = QUOTA_LENGTH_BIN_LABELS[2:]
    ax.text(
        119.0,
        min(positions) - 0.85,
        r"$\geq$50 tok",
        ha="center",
        va="center",
        fontsize=ANNOTATION_FONT_SIZE - 1.5,
        color="#444444",
    )
    for position, row in zip(positions, rows, strict=True):
        share = 100.0 * sum(row[f"share_{name}"] for name in long_labels)
        ax.text(
            119.0,
            position,
            f"{share:.1f}%",
            ha="center",
            va="center",
            fontsize=ANNOTATION_FONT_SIZE - 1,
            color="#555555" if row["alpha"] is None else "#222222",
        )
    ax.set_yticks(positions, [row["label"] for row in rows], fontsize=ANNOTATION_FONT_SIZE)
    for tick, row in zip(ax.get_yticklabels(), rows, strict=True):
        if row["alpha"] is None:
            tick.set_color("#555555")
    ax.invert_yaxis()
    ax.set_xlim(0, 136)
    ax.set_xticks([0, 50, 100])
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xlabel("share of requests (%)")
    ax.grid(False)
    ax.legend(
        title="response length (tokens)",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        handlelength=1.1,
        columnspacing=0.9,
        handletextpad=0.5,
        title_fontsize=ANNOTATION_FONT_SIZE - 1,
    )
    fig.tight_layout()
    _save(fig, output)


# ---------------------------------------------------------------------------
# Output helpers and the reference check.
# ---------------------------------------------------------------------------


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"wrote {output}")


def check_summary(summary: dict, reference_path: Path) -> None:
    """Compare the recomputed aggregates with the archived ones."""
    expected = json.loads(reference_path.read_text(encoding="utf-8"))
    checks = (
        ("concurrency", CONCURRENCY_METRICS),
        ("quota", QUOTA_METRICS),
        ("quota_length_mix", QUOTA_LENGTH_METRICS),
    )
    compared = 0
    for table, metrics in checks:
        generated = {row["policy"]: row for row in summary[table]}
        archived = {row["policy"]: row for row in expected[table]}
        if generated.keys() != archived.keys():
            raise ValueError(f"{reference_path.name}: {table} rows differ from the reference")
        for policy, row in generated.items():
            for metric in metrics:
                value, reference = row[metric], archived[policy][metric]
                if value is None or reference is None:
                    if value != reference:
                        raise ValueError(
                            f"{table}/{policy}: {metric}={value}, reference={reference}"
                        )
                    continue
                if not math.isclose(float(value), float(reference), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"{table}/{policy}: {metric}={value}, reference={reference}")
                compared += 1
    print(f"PASS: {compared} aggregates match {reference_path.name}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=ROOT / "data" / "real_eval_records_m3")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "figures" / "real_world_m3"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Archived aggregates to check against; defaults to "
        "<records-dir>/mechanisms_reference_summary.json when that exists.",
    )
    args = parser.parse_args(argv)
    source, output = args.records_dir.resolve(), args.output_dir.resolve()
    reference = args.reference or source / "mechanisms_reference_summary.json"

    routewise = load_records(source, ROUTEWISE_POLICIES)
    baselines = load_records(source, BASELINE_POLICIES)
    inventory = inventory_for(next(iter(routewise.values())))
    quota = provider_by_tier(inventory, "quota")
    window_sec, size = quota_windows(inventory)

    # 1. Concurrency-limited provider.
    concurrency_rows = [concurrency_row(rec, inventory) for rec in routewise.values()]
    write_concurrency_table(concurrency_rows, output / "mechanism_concurrency_rows.tex")

    # 2. Quota-limited provider.
    timelines: dict[str, pd.DataFrame] = {}
    quota_rows: list[dict] = []
    for policy, rec in {**routewise, **baselines}.items():
        used = quota_legs(rec.frame, quota)
        timelines[policy] = quota_timeline(rec.frame, used, window_sec)
        quota_rows.append(quota_row(policy, rec.frame, used, window_sec, size))
    trace = baselines["greedy_cost"].frame
    offline_used = quota_first_offline_usage(trace, window_sec, size)
    timelines[QUOTA_FIRST_OFFLINE] = quota_timeline(trace, offline_used, window_sec)
    quota_rows.append(quota_row(QUOTA_FIRST_OFFLINE, trace, offline_used, window_sec, size))
    plot_quota_over_time(
        {
            policy: timelines[policy]
            for policy in (*ROUTEWISE_POLICIES, *BASELINE_POLICIES, QUOTA_FIRST_OFFLINE)
        },
        float(trace["hours"].max()),
        window_sec,
        size,
        output / "mechanism_quota_over_time.pdf",
    )

    # 3. What the budget reserves quota for.
    concurrency = provider_by_tier(inventory, "concurrency")
    quota_mix_rows = quota_length_mixes(routewise, quota, concurrency)
    plot_quota_length_mix(quota_mix_rows, output / "mechanism_quota_by_length.pdf")

    summary = {
        "quota_provider": quota,
        "quota_window_sec": window_sec,
        "quota_window_requests": size,
        "concurrency": concurrency_rows,
        "quota": quota_rows,
        "quota_length_mix": quota_mix_rows,
    }
    summary_path = output / "mechanisms_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    if reference.exists():
        check_summary(summary, reference)
    else:
        print(f"no reference at {reference}; skipped the aggregate check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
