"""Measure how RouteWise's components behaved in the MiniMax-M3 24-hour run.

Four questions from the revision plan, answered from the committed records and
the per-decision router state of the five RouteWise runs:

1. Effective cost. The quota tier is priced by the shadow price
   psi(z) = L (U/L)^z of its fraction used z, the metered tiers by their real
   request price. The figure asks whether that comparison is what decides the
   quota-versus-metered choice: the quota share of dispatches as a function of
   psi(z) over the cheapest metered price for the same request.
2. Concurrency-limited provider. The table asks whether the LP considered the
   concurrency slot whenever it was free and took it whenever it was the
   lowest-latency candidate.
3. Quota-limited provider. The figure plots quota consumed in the current
   five-hour window over the day for RouteWise, Greedy-cost (the live policy
   closest to "use quota whenever it is available") and an offline replay
   that sends every arrival to quota until the window is exhausted; the table
   adds what each unit of quota bought. A second figure and table explain why
   RouteWise's quota use rises with alpha although a higher alpha is less
   cost-sensitive: a tight budget does not use less quota indiscriminately, it
   reserves quota for the long requests, where the metered alternative is dear.
   Both are measured over the decisions where the concurrency slot was busy,
   the only ones in which the budget weighs quota against a metered price.
4. Output-length awareness. The figure and table ask whether short predicted
   responses went to the metered on-demand tier while the subscription tiers
   took the long ones.

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
from plots.palettes import TIER_COLORS

ROOT = Path(__file__).resolve().parents[1]
ROUTEWISE_POLICIES = tuple(f"budget_range_alpha{alpha}_hedge" for alpha in (0, 25, 50, 75, 100))
BASELINE_POLICIES = ("greedy_latency", "greedy_cost")
QUOTA_FIRST_OFFLINE = "quota_first_offline"
LENGTH_FIGURE_POLICIES = ("budget_range_alpha25_hedge", "greedy_cost")
# Trace output length (tokens) bins shared by every policy; the trace is the
# same for all of them, so the bins hold the same requests in every panel.
LENGTH_BIN_EDGES = (0, 10, 20, 50, 100, 200, 500, float("inf"))
LENGTH_BIN_LABELS = ("1-10", "11-20", "21-50", "51-100", "101-200", "201-500", ">500")
# psi(z) / cheapest metered price, log-spaced bins (a factor of ~1.2 apart).
RATIO_BIN_EDGES = np.logspace(np.log10(0.1), np.log10(4.0), 21)
RATIO_BIN_MIN_COUNT = 30
# The two line figures carry their legend below the axes, so they are taller
# than the shared panel size.
LEGEND_BELOW_FIGSIZE = (PAPER_PANEL_FIGSIZE[0], 3.9)
TIER_LABELS = {"api": "Metered API", "quota": "Quota", "concurrency": "Concurrency"}
TIER_ORDER = ("api", "quota", "concurrency")
CONCURRENCY_METRICS = (
    "free_slot_decisions",
    "offered_when_free_rate",
    "lowest_latency_when_free_rate",
    "picked_when_lowest_latency_rate",
    "picked_when_not_lowest_latency_rate",
    "share_of_requests",
)
QUOTA_METRICS = (
    "quota_requests",
    "peak_window_quota_requests",
    "peak_fraction_used",
    "requests_after_exhaustion",
    "quota_mean_output_tokens",
    "quota_api_equivalent_usd",
    "quota_api_equivalent_usd_per_1000",
)
# Coarser than the tier-mix bins: these count only the decisions where the
# concurrency slot was busy, and the long bins are thin.
QUOTA_LENGTH_BIN_EDGES = (0, 10, 50, 200, float("inf"))
QUOTA_LENGTH_BIN_LABELS = ("1-10", "11-50", "51-200", ">200")
# Sequential, because the bins are ordered; the ramp also survives greyscale.
QUOTA_LENGTH_COLORS = ("#dbe6ec", "#9dbdcf", "#4f86a0", "#1d4657")
QUOTA_LONG_TOKENS = 50
QUOTA_LENGTH_METRICS = (
    "contested_decisions",
    "quota_requests",
    "quota_mean_output_tokens",
    "quota_long_share",
    "quota_share_shortest_bin",
    "quota_share_longest_bin",
    "cheaper_share_max_gap",
)
LENGTH_METRICS = (
    "trace_output_tokens_api",
    "trace_output_tokens_quota",
    "trace_output_tokens_concurrency",
)


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


def cheapest_metered_cost(
    inventory, prompt_tokens: pd.Series, output_tokens: pd.Series
) -> pd.Series:
    """Cheapest metered price of each request at the inventory's list prices."""
    costs = [
        (prompt_tokens * spec.input_price_per_m + output_tokens * spec.output_price_per_m) / 1e6
        for spec in inventory.providers
        if spec.tier == "api"
    ]
    return pd.concat(costs, axis=1).min(axis=1)


# ---------------------------------------------------------------------------
# 1. Effective cost: quota share versus psi(z) / cheapest metered price.
# ---------------------------------------------------------------------------


def quota_share_by_price_ratio(records: Records, inventory) -> pd.DataFrame:
    quota = provider_by_tier(inventory, "quota")
    concurrency = provider_by_tier(inventory, "concurrency")
    frame = records.frame
    metered = [f"c_eff:{spec.name}" for spec in inventory.providers if spec.tier == "api"]
    # Only decisions that were a straight quota-versus-metered choice: quota
    # offered and the concurrency slot busy, so the zero-cost slot did not
    # absorb the request.
    mask = frame[f"c_eff:{quota}"].notna() & frame["unavailable"].str.contains(concurrency)
    sub = frame[mask]
    ratio = sub[f"c_eff:{quota}"] / sub[metered].min(axis=1)
    picked = sub["primary_provider"] == quota
    bins = pd.cut(ratio, RATIO_BIN_EDGES)
    grouped = picked.groupby(bins, observed=True)
    out = pd.DataFrame(
        {
            "ratio_geomean": ratio.groupby(bins, observed=True).apply(
                lambda values: float(np.exp(np.log(values).mean()))
            ),
            "n": grouped.size(),
            "quota_share": grouped.mean(),
        }
    )
    return out[out["n"] >= RATIO_BIN_MIN_COUNT]


def plot_quota_share_vs_ratio(curves: dict[str, pd.DataFrame], output: Path) -> None:
    apply_column_figure_style(legend_fontsize=ANNOTATION_FONT_SIZE - 1)
    fig, ax = plt.subplots(figsize=LEGEND_BELOW_FIGSIZE)
    for policy, curve in curves.items():
        ax.plot(
            curve["ratio_geomean"],
            100.0 * curve["quota_share"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=_color(policy),
            label=_label(policy),
        )
    ax.axvline(1.0, color="#444444", linewidth=0.8, linestyle="--")
    ax.annotate(
        "break-even",
        (0.93, 45),
        fontsize=ANNOTATION_FONT_SIZE - 1,
        ha="right",
        va="center",
        rotation=90,
        color="#444444",
    )
    ax.set_xscale("log")
    ax.set_xlim(0.12, 3.3)
    ax.set_xticks([0.2, 0.5, 1.0, 2.0], ["0.2", "0.5", "1", "2"])
    ax.set_xlabel(r"quota shadow price $\psi(z)$ / metered price")
    ax.set_ylabel("sent to quota (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.tight_layout()
    _save(fig, output)


# ---------------------------------------------------------------------------
# 2. Concurrency-limited provider: offered whenever free, taken when fastest.
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
# 3. Quota-limited provider: consumption over time and what it bought.
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
    inventory,
    window_sec: float,
    size: int,
) -> dict[str, float]:
    timeline = quota_timeline(frame, used, window_sec)
    per_window = timeline.groupby("window")["used"].max()
    peak_window = int(per_window.idxmax())
    exhausted = timeline[(timeline["window"] == peak_window) & (timeline["used"] >= size)]
    exhausted_at = float(exhausted["hours"].iloc[0]) if len(exhausted) else None
    after = 0
    if exhausted_at is not None:
        in_window = timeline["window"] == peak_window
        after = int((in_window & (timeline["hours"] >= exhausted_at)).sum()) - 1
    equivalent = cheapest_metered_cost(inventory, frame["prompt_tokens"], frame["max_tokens"])
    quota_value = float(equivalent[used].sum())
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
        "quota_mean_output_tokens": float(frame.loc[used, "max_tokens"].mean()),
        "quota_api_equivalent_usd": quota_value,
        "quota_api_equivalent_usd_per_1000": 1000.0 * quota_value / used.sum(),
    }


def write_quota_table(rows: list[dict], size: int, path: Path) -> None:
    lines = []
    for row in rows:
        exhausted = f"{row['exhausted_at_hours']:.1f}\\,h" if row["exhausted_at_hours"] else "--"
        lines.append(
            f"{_label(row['policy'])} & "
            f"{row['quota_requests']:,} & "
            f"{row['peak_window_quota_requests']:,}/{size:,} & "
            f"{exhausted} & "
            f"{row['requests_after_exhaustion']:,} & "
            f"{row['quota_mean_output_tokens']:.0f} & "
            f"{row['quota_api_equivalent_usd']:.3f} & "
            f"{row['quota_api_equivalent_usd_per_1000']:.3f} \\\\"
        )
    _write_text(path, lines)


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
# 3b. What the budget reserves quota for: length of the quota-served requests.
# ---------------------------------------------------------------------------


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; closed form, so it reproduces."""
    if total == 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denominator = 1.0 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


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


def quota_cheaper(records: Records, quota: str) -> pd.Series:
    """Whether the quota tier's shadow price undercut every metered candidate."""
    frame = records.frame
    metered = [
        column
        for column in frame.columns
        if column.startswith("c_eff:") and column != f"c_eff:{quota}"
    ]
    return frame[f"c_eff:{quota}"] < frame[metered].min(axis=1)


def quota_by_length(records: Records, quota: str, concurrency: str) -> pd.DataFrame:
    """Share of requests sent to quota per response-length bin.

    ``cheaper_share`` is the share of the same decisions in which quota was the
    cheaper option at all. At alpha = 0 the budget equals the cheapest
    effective cost, so the two columns should agree; that agreement is what
    ties the routing pattern to the budget rather than to latency.
    """
    frame = records.frame
    contested = contested_decisions(records, quota, concurrency)
    bins = pd.cut(frame["max_tokens"], QUOTA_LENGTH_BIN_EDGES, labels=list(QUOTA_LENGTH_BIN_LABELS))
    taken = pd.DataFrame(
        {
            "bin": bins,
            "quota": frame["primary_provider"] == quota,
            "cheaper": quota_cheaper(records, quota),
        }
    )[contested]
    grouped = taken.groupby("bin", observed=True)
    out = grouped.agg(
        n=("quota", "size"),
        quota_requests=("quota", "sum"),
        cheaper_share=("cheaper", "mean"),
    )
    out["quota_share"] = out["quota_requests"] / out["n"]
    intervals = [_wilson(row.quota_requests, row.n) for row in out.itertuples()]
    out["ci_low"] = [low for low, _ in intervals]
    out["ci_high"] = [high for _, high in intervals]
    return out.reset_index()


def quota_length_row(records: Records, quota: str, concurrency: str) -> dict[str, float]:
    frame = records.frame
    contested = contested_decisions(records, quota, concurrency)
    served = frame[contested & (frame["primary_provider"] == quota)]
    long_requests = int((served["max_tokens"] > QUOTA_LONG_TOKENS).sum())
    low, high = _wilson(long_requests, len(served))
    curve = quota_by_length(records, quota, concurrency).set_index("bin")
    return {
        "policy": records.policy,
        "alpha": records.alpha,
        "contested_decisions": int(contested.sum()),
        "quota_requests": len(served),
        "quota_mean_output_tokens": float(served["max_tokens"].mean()),
        "quota_median_output_tokens": float(served["max_tokens"].median()),
        "quota_long_share": long_requests / len(served),
        "quota_long_share_ci_low": low,
        "quota_long_share_ci_high": high,
        "quota_share_shortest_bin": float(curve.loc[QUOTA_LENGTH_BIN_LABELS[0], "quota_share"]),
        "quota_share_longest_bin": float(curve.loc[QUOTA_LENGTH_BIN_LABELS[-1], "quota_share"]),
        # Largest gap, over the bins, between what was sent to quota and what
        # quota was the cheaper option for. Near zero means the budget alone
        # explains the pattern.
        "cheaper_share_max_gap": float((curve["quota_share"] - curve["cheaper_share"]).abs().max()),
    }


def write_quota_length_table(rows: list[dict], path: Path) -> None:
    lines = [
        f"{row['alpha']:g} & "
        f"{row['contested_decisions']:,} & "
        f"{row['quota_requests']:,} & "
        f"{row['quota_mean_output_tokens']:.0f} & "
        f"{100.0 * row['quota_long_share']:.1f}\\% & "
        f"{100.0 * row['quota_share_shortest_bin']:.0f}\\% & "
        f"{100.0 * row['quota_share_longest_bin']:.0f}\\% \\\\"
        for row in rows
    ]
    _write_text(path, lines)


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
                "row": policy,
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
    for row, label, lengths in (
        (
            "contested",
            f"Contested ($\\alpha={reference.alpha:g}$)",
            frame.loc[contested, "max_tokens"],
        ),
        ("trace", "All requests", frame["max_tokens"]),
    ):
        rows.append(
            {
                "row": row,
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
# 4. Output-length awareness.
# ---------------------------------------------------------------------------


def length_row(records: Records) -> dict[str, float]:
    frame = records.frame
    row: dict[str, float] = {"policy": records.policy, "alpha": records.alpha, "n": len(frame)}
    for tier in TIER_ORDER:
        sub = frame[frame["tier"] == tier]
        row[f"requests_{tier}"] = len(sub)
        row[f"trace_output_tokens_{tier}"] = float(sub["max_tokens"].mean())
        row[f"predicted_output_tokens_{tier}"] = (
            float(sub["predicted_output_tokens"].mean()) if records.has_state else None
        )
    return row


def write_length_table(rows: list[dict], path: Path) -> None:
    lines = []
    for row in rows:
        predicted = " & ".join(
            f"{row[f'predicted_output_tokens_{tier}']:.0f}"
            if row[f"predicted_output_tokens_{tier}"] is not None
            else "--"
            for tier in TIER_ORDER
        )
        realized = " & ".join(f"{row[f'trace_output_tokens_{tier}']:.0f}" for tier in TIER_ORDER)
        lines.append(f"{_label(row['policy'])} & {predicted} & {realized} \\\\")
    _write_text(path, lines)


def tier_mix_by_length(frame: pd.DataFrame) -> pd.DataFrame:
    bins = pd.cut(frame["max_tokens"], LENGTH_BIN_EDGES, labels=LENGTH_BIN_LABELS)
    counts = pd.crosstab(bins, frame["tier"]).reindex(columns=list(TIER_ORDER), fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def plot_tier_mix_by_length(records: Records, output: Path) -> None:
    apply_column_figure_style(legend_fontsize=ANNOTATION_FONT_SIZE - 1)
    fig, ax = plt.subplots(figsize=PAPER_PANEL_FIGSIZE)
    mix = tier_mix_by_length(records.frame)
    bottom = np.zeros(len(mix))
    x = np.arange(len(mix))
    for tier in TIER_ORDER:
        values = 100.0 * mix[tier].to_numpy()
        ax.bar(
            x, values, bottom=bottom, color=TIER_COLORS[tier], width=0.72, label=TIER_LABELS[tier]
        )
        bottom += values
    ax.set_xticks(x, list(mix.index), fontsize=ANNOTATION_FONT_SIZE, rotation=35, ha="right")
    ax.set_xlabel("response length in the trace (tokens)")
    ax.set_ylabel("share of requests (%)")
    ax.set_ylim(0, 100)
    ax.set_title(_label(records.policy), fontsize=ANNOTATION_FONT_SIZE + 1, pad=20)
    ax.grid(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
        borderaxespad=0.1,
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
        ("quota_length", QUOTA_LENGTH_METRICS),
        ("length", LENGTH_METRICS),
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

    # 1. Effective cost.
    curves = {
        policy: quota_share_by_price_ratio(rec, inventory) for policy, rec in routewise.items()
    }
    plot_quota_share_vs_ratio(curves, output / "mechanism_quota_vs_effective_cost.pdf")

    # 2. Concurrency-limited provider.
    concurrency_rows = [concurrency_row(rec, inventory) for rec in routewise.values()]
    write_concurrency_table(concurrency_rows, output / "mechanism_concurrency_rows.tex")

    # 3. Quota-limited provider.
    timelines: dict[str, pd.DataFrame] = {}
    quota_rows: list[dict] = []
    for policy, rec in {**routewise, **baselines}.items():
        used = quota_legs(rec.frame, quota)
        timelines[policy] = quota_timeline(rec.frame, used, window_sec)
        quota_rows.append(quota_row(policy, rec.frame, used, inventory, window_sec, size))
    trace = baselines["greedy_cost"].frame
    offline_used = quota_first_offline_usage(trace, window_sec, size)
    timelines[QUOTA_FIRST_OFFLINE] = quota_timeline(trace, offline_used, window_sec)
    quota_rows.append(
        quota_row(QUOTA_FIRST_OFFLINE, trace, offline_used, inventory, window_sec, size)
    )
    write_quota_table(quota_rows, size, output / "mechanism_quota_rows.tex")
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

    # 3b. What the budget reserves quota for.
    concurrency = provider_by_tier(inventory, "concurrency")
    quota_length_curves = {
        policy: quota_by_length(rec, quota, concurrency) for policy, rec in routewise.items()
    }
    quota_length_rows = [quota_length_row(rec, quota, concurrency) for rec in routewise.values()]
    write_quota_length_table(quota_length_rows, output / "mechanism_quota_length_rows.tex")
    quota_mix_rows = quota_length_mixes(routewise, quota, concurrency)
    plot_quota_length_mix(quota_mix_rows, output / "mechanism_quota_by_length.pdf")

    # 4. Output-length awareness.
    length_rows = [length_row(rec) for rec in {**routewise, **baselines}.values()]
    write_length_table(length_rows, output / "mechanism_length_rows.tex")
    for policy in LENGTH_FIGURE_POLICIES:
        rec = routewise.get(policy) or baselines[policy]
        plot_tier_mix_by_length(rec, output / f"mechanism_tier_by_length_{policy}.pdf")

    summary = {
        "quota_provider": quota,
        "quota_window_sec": window_sec,
        "quota_window_requests": size,
        "concurrency": concurrency_rows,
        "quota": quota_rows,
        "quota_length": quota_length_rows,
        "length": length_rows,
        "quota_share_by_price_ratio": {
            policy: curve.reset_index(drop=True).to_dict(orient="records")
            for policy, curve in curves.items()
        },
        "quota_share_by_length": {
            policy: curve.to_dict(orient="records") for policy, curve in quota_length_curves.items()
        },
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
