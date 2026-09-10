"""Front-page BurstGPT teaser with RouteWise alpha sweep.

The plot intentionally uses only the policies needed for the teaser:
Greedy-cost, Greedy-latency, and LP-only RouteWise alpha points.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = (
    REPO_ROOT
    / "outputs"
    / "simulation"
    / "end_to_end_rw8_minimax_q1_c1_slo3_p005_20260515"
    / "summary.csv"
)
FALLBACK_CSV = (
    REPO_ROOT
    / "outputs"
    / "simulation"
    / "end_to_end_rw8_minimax_q1_c1_slo3_20260514"
    / "summary.csv"
)

ROUTEWISE_TEAL = "#2b777a"
GREEDY_COST = "#1f77b4"
GREEDY_LATENCY = "#d96ab7"
GRID = "#e7e7e7"
TEXT = "#202020"


@dataclass(frozen=True)
class MetricSpec:
    column: str
    ylabel: str
    scale: float
    panel_title: str
    stem_suffix: str


def _style() -> None:
    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.edgecolor": "#222222",
            "axes.labelcolor": TEXT,
            "axes.linewidth": 1.2,
            "axes.titlesize": 0.0,
            "axes.labelsize": 11.8,
            "xtick.color": TEXT,
            "xtick.labelsize": 10.6,
            "ytick.color": TEXT,
            "ytick.labelsize": 10.6,
            "legend.fontsize": 10.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    if "hedging_enabled" in df.columns:
        df = df[df["hedging_enabled"] == False].copy()  # noqa: E712
    return df


def _load_real_summary(path: Path) -> pd.DataFrame:
    """Build the teaser frame from plot_real_world_frontier --summary-out JSON.

    The real-world RouteWise sweep runs with hedging enabled (that is the
    deployed configuration), so unlike the simulator path no hedging filter
    is applied.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(
        [
            {
                "policy": item["policy"],
                "routewise_alpha": item["alpha"],
                "total_cost_usd": item["total_cost_usd"],
                "mean_ttft_ms": item["ttft_mean_ms"],
                "slo_violation_rate": item["slo_violation_rate"],
            }
            for item in payload
        ]
    )


def _load_histograms(csv: Path) -> dict[str, dict[str, object]]:
    path = csv.with_name("ttft_histograms.json")
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["policy"]): dict(row["histogram"]) for row in rows}


def _select(df: pd.DataFrame, policy: str) -> pd.Series:
    rows = df[df["policy"] == policy]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {policy!r}, got {len(rows)}")
    return rows.iloc[0]


ROUTEWISE_SWEEP_PREFIXES = ("ablation_lp_only_alpha", "budget_range_alpha")


def _routewise_rows(df: pd.DataFrame, base_cost: float) -> pd.DataFrame:
    rows = df[df["policy"].str.startswith(ROUTEWISE_SWEEP_PREFIXES)].copy()
    if rows.empty:
        raise ValueError("summary has no RouteWise sweep rows")
    rows = rows.sort_values("routewise_alpha").reset_index(drop=True)
    rows["norm_cost"] = rows["total_cost_usd"] / base_cost
    rows["alpha_label"] = rows["routewise_alpha"].map(_format_alpha)
    return rows


def _format_alpha(value: float) -> str:
    if math.isclose(value, 0.0):
        return "0"
    if math.isclose(value, 1.0):
        return "1"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _key_alpha_mask(values: pd.Series) -> np.ndarray:
    key = np.array([0.0, 1.0])
    raw = values.to_numpy(dtype=float)
    return np.array([np.any(np.isclose(value, key, atol=1e-9)) for value in raw])


def _label_offsets() -> dict[float, tuple[float, float]]:
    """Offsets in points from the alpha=0 / alpha=1 markers to their labels."""
    return {
        0.0: (5.0, 11.0),
        1.0: (9.0, 8.0),
    }


def _alpha_text(alpha: float) -> str:
    if math.isclose(alpha, 0.0):
        return r"$\alpha=0$"
    if math.isclose(alpha, 1.0):
        return r"$\alpha=1$"
    return rf"$\alpha={_format_alpha(alpha)}$"


def _draw_better_cue(ax: plt.Axes) -> None:
    # Bottom-right corner: the real-world sweep occupies the lower-left of
    # the panel, so the cue lives in the opposite free corner.
    ax.annotate(
        "",
        xy=(0.795, 0.090),
        xytext=(0.925, 0.230),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#737373",
            "linewidth": 1.1,
            "mutation_scale": 10,
            "alpha": 0.75,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=1,
    )
    ax.text(
        0.865,
        0.275,
        "better",
        transform=ax.transAxes,
        fontsize=9.4,
        color="#737373",
        fontweight="bold",
        ha="center",
        va="center",
        zorder=1,
    )


def _plot_panel(
    ax: plt.Axes,
    *,
    df: pd.DataFrame,
    sweep: pd.DataFrame,
    base_cost: float,
    metric: MetricSpec,
) -> None:
    stroke = [pe.withStroke(linewidth=2.6, foreground="white")]
    xs = sweep["norm_cost"].to_numpy(dtype=float)
    ys = (sweep[metric.column].to_numpy(dtype=float) * metric.scale)
    # Connect the whole alpha sweep in cost order. On simulator data this
    # coincides with the Pareto frontier; on real-world data the sweep is not
    # monotone in TTFT, where a frontier line would collapse to one point.
    line = sweep.sort_values("norm_cost")
    frontier_x = line["norm_cost"].to_numpy(dtype=float)
    frontier_y = line[metric.column].to_numpy(dtype=float) * metric.scale

    ax.scatter(
        xs,
        ys,
        s=28,
        color=ROUTEWISE_TEAL,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.90,
        zorder=4,
    )
    ax.plot(
        frontier_x,
        frontier_y,
        color=ROUTEWISE_TEAL,
        linewidth=2.2,
        zorder=3,
        solid_capstyle="round",
    )

    key_mask = _key_alpha_mask(sweep["routewise_alpha"])
    ax.scatter(
        xs[key_mask],
        ys[key_mask],
        s=64,
        color=ROUTEWISE_TEAL,
        edgecolor="white",
        linewidth=1.1,
        zorder=5,
    )
    label_offsets = _label_offsets()
    for _, row in sweep[key_mask].iterrows():
        alpha = float(row["routewise_alpha"])
        x = float(row["norm_cost"])
        y = float(row[metric.column]) * metric.scale
        dx, dy = label_offsets.get(alpha, (9.0, 0.0))
        ax.annotate(
            _alpha_text(alpha),
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if dx < 0 else "left",
            va="center",
            color="#4b4b4b",
            fontsize=10.4,
            fontweight="normal",
            path_effects=stroke,
            zorder=6,
        )

    baseline_specs = (
        ("Greedy-cost", "greedy_cost", GREEDY_COST, "s"),
        ("Greedy-latency", "greedy_latency", GREEDY_LATENCY, "s"),
    )
    for label, policy, color, marker in baseline_specs:
        row = _select(df, policy)
        x = float(row["total_cost_usd"]) / base_cost
        y = float(row[metric.column]) * metric.scale
        ax.scatter(
            [x],
            [y],
            s=76,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=7,
        )

    _draw_better_cue(ax)

    ax.set_xlabel("Normalized cost\n(Baseline Greedy-cost)")
    ax.set_ylabel(metric.ylabel)
    ax.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_min = min(xs.min(), 1.0) - 0.025
    x_max = max(xs.max(), 1.0, float(_select(df, "greedy_latency")["total_cost_usd"]) / base_cost) + 0.07
    ax.set_xlim(x_min, x_max)
    tick_start = math.ceil(x_min * 10) / 10
    tick_step = 0.1 if (x_max - x_min) <= 0.45 else 0.2
    ax.set_xticks(np.arange(tick_start, math.ceil(x_max * 10) / 10 + 0.001, tick_step))
    ax.set_xticklabels([f"{tick:.1f}x" for tick in ax.get_xticks()])

    values = [*ys]
    for policy in ("greedy_cost", "greedy_latency"):
        values.append(float(_select(df, policy)[metric.column]) * metric.scale)
    y_min = min(values)
    y_max = max(values)
    pad = max((y_max - y_min) * 0.18, 0.08 if metric.column == "mean_ttft_ms" else 0.5)
    ax.set_ylim(max(0.0, y_min - pad), y_max + pad)


def _plot_slo_bar_panel(
    ax: plt.Axes,
    *,
    df: pd.DataFrame,
    sweep: pd.DataFrame,
    base_cost: float,
    metric: MetricSpec,
) -> None:
    key_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels: list[str] = []
    values: list[float] = []
    costs: list[float] = []
    colors: list[str] = []
    for alpha in key_alphas:
        match = sweep[np.isclose(sweep["routewise_alpha"].to_numpy(dtype=float), alpha)]
        if match.empty:
            continue
        row = match.iloc[0]
        labels.append(f"RouteWise-{_format_alpha(float(row['routewise_alpha']))}")
        values.append(float(row[metric.column]) * metric.scale)
        costs.append(float(row["total_cost_usd"]) / base_cost)
        colors.append(ROUTEWISE_TEAL)

    for label, policy, color in (
        ("Greedy-cost", "greedy_cost", GREEDY_COST),
        ("Greedy-latency", "greedy_latency", GREEDY_LATENCY),
    ):
        row = _select(df, policy)
        labels.append(label)
        values.append(float(row[metric.column]) * metric.scale)
        costs.append(float(row["total_cost_usd"]) / base_cost)
        colors.append(color)

    if not values:
        raise ValueError("summary has no points for SLO bar panel")

    y = np.arange(len(values))
    max_value = max(values)
    text_pad = max(max_value * 0.03, 0.16)
    ax.barh(
        y,
        values,
        height=0.66,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for row_idx, (value, cost) in enumerate(zip(values, costs, strict=True)):
        ax.text(
            value + text_pad,
            row_idx,
            f"{value:.1f}%  {cost:.2f}x",
            va="center",
            ha="left",
            fontsize=8.4,
            color="#4b4b4b",
        )

    ax.set_xlabel("SLO violations (%)")
    ax.set_ylabel("")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.4)
    ax.invert_yaxis()
    ax.grid(True, axis="x", color=GRID, linewidth=1.0, zorder=0)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.0, max_value + max(max_value * 0.44, 2.0))


def _histogram_box_stats(label: str, histogram: dict[str, object]) -> dict[str, float | str]:
    return {
        "label": label,
        "whislo": _histogram_quantile(histogram, 0.05) / 1000.0,
        "q1": _histogram_quantile(histogram, 0.25) / 1000.0,
        "med": _histogram_quantile(histogram, 0.50) / 1000.0,
        "q3": _histogram_quantile(histogram, 0.75) / 1000.0,
        "whishi": _histogram_quantile(histogram, 0.95) / 1000.0,
    }


def _histogram_quantile(histogram: dict[str, object], q: float) -> float:
    edges = np.asarray(histogram["bin_edges_ms"], dtype=float)
    counts = np.asarray(histogram["counts"], dtype=float)
    n = float(histogram["n"])
    target = q * n
    cumulative = 0.0
    for idx, count in enumerate(counts):
        next_cumulative = cumulative + count
        if target <= next_cumulative:
            if idx == 0:
                return float(histogram["min_ms"])
            if idx == len(counts) - 1:
                return float(histogram["max_ms"])
            lo = float(edges[idx - 1])
            hi = float(edges[idx])
            frac = 0.5 if count <= 0 else (target - cumulative) / count
            return lo + frac * (hi - lo)
        cumulative = next_cumulative
    return float(histogram["max_ms"])


def _metrics() -> tuple[MetricSpec, MetricSpec]:
    return (
        MetricSpec(
            column="mean_ttft_ms",
            ylabel="Mean TTFT (s)",
            scale=1.0 / 1000.0,
            panel_title="",
            stem_suffix="ttft",
        ),
        MetricSpec(
            column="slo_violation_rate",
            ylabel="",
            scale=100.0,
            panel_title="",
            stem_suffix="slo",
        ),
    )


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=ROUTEWISE_TEAL,
            marker="o",
            markersize=8.0,
            linewidth=2.6,
            label="RouteWise",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="s",
            markerfacecolor=GREEDY_COST,
            markeredgecolor="white",
            markersize=9.0,
            label="Greedy-cost",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="s",
            markerfacecolor=GREEDY_LATENCY,
            markeredgecolor="white",
            markersize=9.0,
            label="Greedy-latency",
        ),
    ]


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def _render_combined(
    *,
    df: pd.DataFrame,
    sweep: pd.DataFrame,
    base_cost: float,
    histograms: dict[str, dict[str, object]],
    out_dir: Path,
    out_stem: str,
) -> None:
    metrics = _metrics()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.05))
    fig.subplots_adjust(left=0.082, right=0.988, bottom=0.205, top=0.760, wspace=0.310)
    fig.text(0.082, 0.955, "BurstGPT + ShareGPT", ha="left", va="top", fontsize=12.0, fontweight="bold")
    for ax, metric in zip(axes, metrics, strict=True):
        if metric.stem_suffix == "slo":
            _plot_slo_bar_panel(ax, df=df, sweep=sweep, base_cost=base_cost, metric=metric)
        else:
            _plot_panel(ax, df=df, sweep=sweep, base_cost=base_cost, metric=metric)

    fig.legend(
        handles=_legend_handles(),
        loc="upper right",
        bbox_to_anchor=(0.988, 0.965),
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    _save(fig, out_dir, out_stem)
    if out_stem != "routewise_front_page_burstgpt":
        _save(fig, out_dir, "routewise_front_page_burstgpt")
    plt.close(fig)


def _render_separate(
    *,
    df: pd.DataFrame,
    sweep: pd.DataFrame,
    base_cost: float,
    histograms: dict[str, dict[str, object]],
    out_dir: Path,
    out_stem: str,
    include_legend: bool = True,
) -> list[str]:
    stems: list[str] = []
    for metric in _metrics():
        stem = f"{out_stem}_{metric.stem_suffix}"
        fig, ax = plt.subplots(figsize=(2.35, 2.08))
        show_legend = include_legend and metric.stem_suffix != "slo"
        top = 0.755 if show_legend else 0.955
        if metric.stem_suffix == "slo":
            fig.subplots_adjust(left=0.455, right=0.985, bottom=0.255, top=0.955)
            _plot_slo_bar_panel(ax, df=df, sweep=sweep, base_cost=base_cost, metric=metric)
        else:
            fig.subplots_adjust(left=0.215, right=0.985, bottom=0.255, top=top)
            _plot_panel(ax, df=df, sweep=sweep, base_cost=base_cost, metric=metric)
        if show_legend:
            fig.legend(
                handles=_legend_handles(),
                loc="upper center",
                bbox_to_anchor=(0.56, 0.985),
                ncol=3,
                frameon=False,
                handlelength=1.0,
                columnspacing=0.38,
                borderaxespad=0.0,
                fontsize=9.8,
            )
        _save(fig, out_dir, stem)
        plt.close(fig)
        stems.append(stem)
    return stems


def render(
    csv: Path,
    out_dir: Path,
    *,
    out_stem: str,
    combined: bool = True,
    separate_legend: bool = True,
    real_summary_json: Path | None = None,
) -> list[str]:
    _style()
    if real_summary_json is not None:
        df = _load_real_summary(real_summary_json)
        histograms = {}
    else:
        df = _load(csv)
        histograms = _load_histograms(csv)
    base_cost = float(_select(df, "greedy_cost")["total_cost_usd"])
    sweep = _routewise_rows(df, base_cost)

    out_dir.mkdir(parents=True, exist_ok=True)
    stems = _render_separate(
        df=df,
        sweep=sweep,
        base_cost=base_cost,
        histograms=histograms,
        out_dir=out_dir,
        out_stem=out_stem,
        include_legend=separate_legend,
    )
    if combined:
        _render_combined(
            df=df,
            sweep=sweep,
            base_cost=base_cost,
            histograms=histograms,
            out_dir=out_dir,
            out_stem=out_stem,
        )
        stems.append(out_stem)
    return stems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV if DEFAULT_CSV.exists() else FALLBACK_CSV)
    parser.add_argument(
        "--real-summary-json",
        type=Path,
        default=None,
        help=(
            "Aggregated real-eval metrics from plot_real_world_frontier "
            "--summary-out. When set, --csv is ignored and the teaser is "
            "drawn from the real-world experiment."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "Desktop" / "output")
    parser.add_argument("--out-stem", default="routewise_front_page_burstgpt_scatter_slo")
    parser.add_argument("--no-combined", action="store_true", help="Only write separate TTFT/SLO panel files.")
    parser.add_argument("--no-separate-legend", action="store_true", help="Omit legends from separate panel files.")
    args = parser.parse_args()
    stems = render(
        args.csv,
        args.out_dir,
        out_stem=args.out_stem,
        combined=not args.no_combined,
        separate_legend=not args.no_separate_legend,
        real_summary_json=args.real_summary_json,
    )
    for stem in stems:
        print(f"wrote: {args.out_dir}/{stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
