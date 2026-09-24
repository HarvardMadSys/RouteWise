"""Shared plotting helpers for end-to-end frontier figure groups.

The real-world and simulator pipelines read different artifacts, but the paper
figures should share one plotting implementation. Each pipeline converts its
native output into the normalized data classes below, then calls these helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from plots.palettes import ONLINE_POLICY_COLORS, ROUTER_STRATEGY_COLORS, TIER_COLORS
from plots.style import apply_style

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


COLUMN_FIGSIZE = (3.35, 2.25)
SLO_BAR_FIGSIZE = COLUMN_FIGSIZE
MIX_FIGSIZE = (3.35, 2.55)
BASE_FONT_SIZE = 8.9
LABEL_FONT_SIZE = 9.4
TICK_FONT_SIZE = 8.3
ANNOTATION_FONT_SIZE = 7.3
ROUTEWISE_ANNOTATION_FONT_SIZE = 8.6
LEGEND_FONT_SIZE = 7.0
ROUTEWISE_COLOR = "#2f6f73"
ROUTEWISE_HEDGE_COLOR = "#f28e2b"

DEFAULT_BASELINE_ORDER = (
    "greedy_cost",
    "greedy_latency",
    "or_auto",
    "or_sort_latency",
    "or_sort_cost",
    "random",
)
POLICY_PLOT_LABELS = {
    "greedy_cost": "Greedy-cost",
    "greedy_latency": "Greedy-latency",
    "random": "Random",
    "or_auto": "OR-auto",
    "or_sort_cost": "OR-price",
    "or_sort_latency": "OR-latency",
}
POLICY_COLORS = {
    "greedy_cost": ROUTER_STRATEGY_COLORS.get("greedy_cost", "#1f77b4"),
    "greedy_latency": ROUTER_STRATEGY_COLORS.get("greedy_latency", "#2ca02c"),
    "random": ROUTER_STRATEGY_COLORS.get("random", "#7f8c8d"),
    "or_auto": ONLINE_POLICY_COLORS.get("openrouter_auto", "#1f77b4"),
    "or_sort_cost": ONLINE_POLICY_COLORS.get("sort_price", "#17becf"),
    "or_sort_latency": ONLINE_POLICY_COLORS.get("sort_latency", "#e377c2"),
}
POLICY_MARKERS = {
    "greedy_cost": "s",
    "greedy_latency": "s",
    "or_auto": "D",
    "or_sort_cost": "D",
    "or_sort_latency": "D",
    "random": "x",
}
BASELINE_LABEL_OFFSET = (4, 3)
BASELINE_LABEL_OFFSETS_BY_METRIC = {
    "slo_violation_rate": {
        "greedy_cost": (6, -10),
        "greedy_latency": (6, -9),
        "or_auto": (6, 4),
        "or_sort_cost": (8, -11),
        "or_sort_latency": (6, 5),
        "random": (6, 4),
    },
    "mean_ttft_ms": {
        "greedy_cost": (6, -9),
        "greedy_latency": (0, -12),
        "or_auto": (6, -12),
        "or_sort_cost": (6, -10),
        "or_sort_latency": (-6, 7),
        "random": (6, 4),
    },
}
ROUTEWISE_LABEL_OFFSETS_BY_METRIC = {
    "slo_violation_rate": {
        0.0: (5, 8),
        0.25: (5, 5),
        0.5: (5, 7),
        0.75: (5, 8),
        1.0: (5, 21),
    },
    "mean_ttft_ms": {
        0.0: (8, 10),
        0.25: (4, -24),
        0.5: (10, 14),
        0.75: (-8, -16),
        1.0: (10, 0),
    },
}
CDF_LINESTYLES = {
    "routewise": "solid",
    "greedy_cost": (0, (3, 2)),
    "greedy_latency": (0, (5, 1.6)),
    "or_auto": (0, (1.5, 1.5)),
    "or_sort_cost": (0, (5, 2)),
    "or_sort_latency": (0, (3, 1.2, 1, 1.2)),
    "random": (0, (1, 1.2)),
}
PROVIDER_COLOR_CYCLE = (
    "#2ca02c",
    "#9467bd",
    "#1f77b4",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#d62728",
)
PROVIDER_MIX_COLORS = {
    "OR_Inceptron": "#76b7b2",
    "OR_Friendli": "#9467bd",
    "OR_DeepInfra": "#4e79a7",
    "OR_SambaNova": "#e377c2",
    "OR_Venice": "#17becf",
    "MiniMax_Plus_SQ": "#6b4c3b",
    "GLM_OR_SQ": "#6b4c3b",
    "Featherless_SC": "#59a14f",
    "OR_AtlasCloud": "#f28e2b",
    "OR_AkashML": "#17becf",
    "OR_WandB": "#8e63b0",
    "OR_Chutes": "#d62728",
    "OR_Novita": "#e377c2",
    "OR_Phala": "#bcbd22",
    "OR_SiliconFlow": "#7f7f7f",
    "OR_Parasail": "#17becf",
}
TIER_MIX_SEGMENTS = (
    ("quota", r"$\mathcal{P}_Q$", TIER_COLORS.get("quota", "#2ca02c")),
    (
        "concurrency",
        r"$\mathcal{P}_C$",
        TIER_COLORS.get("concurrency", "#9467bd"),
    ),
    ("api", r"$\mathcal{P}_O$", TIER_COLORS.get("api", "#1f77b4")),
)


@dataclass(frozen=True)
class FrontierPoint:
    policy: str
    label: str
    alpha: float | None
    hedging: bool
    total_cost_usd: float
    mean_ttft_ms: float
    slo_violation_rate: float
    p99_ms: float | None = None
    hedge_rate: float = 0.0


@dataclass(frozen=True)
class CdfSeries:
    policy: str
    label: str
    x_ms: Sequence[float]
    y_values: Sequence[float]
    p99_ms: float
    alpha: float | None = None
    total_cost_usd: float | None = None
    include_in_axis: bool = True
    drawstyle: str = "default"


@dataclass(frozen=True)
class BoxSeries:
    policy: str
    label: str
    samples_ms: Sequence[float]
    alpha: float | None = None
    total_cost_usd: float | None = None


@dataclass(frozen=True)
class MixSegment:
    key: str
    label: str
    color: str


@dataclass(frozen=True)
class MixRow:
    label: str
    shares: Mapping[str, float]


def apply_column_figure_style(*, legend_fontsize: float = LEGEND_FONT_SIZE) -> None:
    """Style for single-panel PDFs that LaTeX arranges into figure groups."""
    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "axes.titlesize": LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": legend_fontsize,
            "figure.figsize": COLUMN_FIGSIZE,
            "savefig.pad_inches": 0.01,
        }
    )


def policy_plot_label(
    policy: str,
    *,
    alpha: float | None = None,
    hedging: bool = False,
) -> str:
    if alpha is not None:
        return f"RouteWise-{alpha:g}"
    return POLICY_PLOT_LABELS.get(policy, policy.replace("_", " "))


def policy_color(policy: str, *, alpha: float | None = None) -> str:
    if alpha is not None:
        return ROUTEWISE_COLOR
    return POLICY_COLORS.get(policy, "#555555")


def policy_marker(policy: str) -> str:
    return POLICY_MARKERS.get(policy, "s")


def cdf_color(policy: str, *, alpha: float | None = None) -> str:
    return policy_color(policy, alpha=alpha)


def cdf_linestyle(policy: str, *, alpha: float | None = None):
    if alpha is not None:
        return CDF_LINESTYLES["routewise"]
    return CDF_LINESTYLES.get(policy, "solid")


def metric_value(point: FrontierPoint, attr: str) -> float:
    value = getattr(point, attr)
    if attr.endswith("_ms"):
        return value / 1000.0
    if attr.endswith("_rate"):
        return value * 100.0
    return value


def routewise_points(
    points: Sequence[FrontierPoint],
    *,
    hedging: bool | None = None,
    alphas: Sequence[float] | None = None,
) -> list[FrontierPoint]:
    selected = [point for point in points if point.alpha is not None]
    if hedging is not None:
        selected = [point for point in selected if point.hedging is hedging]
    if alphas is not None:
        selected = [
            point
            for point in selected
            if any(math.isclose(point.alpha or 0.0, alpha) for alpha in alphas)
        ]
    return sorted(selected, key=lambda point: point.alpha or 0.0)


def baseline_points(
    points: Sequence[FrontierPoint],
    *,
    order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> list[FrontierPoint]:
    by_policy = {point.policy: point for point in points if point.alpha is None}
    return [by_policy[policy] for policy in order if policy in by_policy]


def _normalized_points(points: Sequence[FrontierPoint]) -> list[FrontierPoint]:
    if not points:
        return []
    baseline = min(point.total_cost_usd for point in points)
    if baseline <= 0:
        return list(points)
    return [replace(point, total_cost_usd=point.total_cost_usd / baseline) for point in points]


def _annotate_baseline(
    ax: plt.Axes,
    point: FrontierPoint,
    attr: str,
    *,
    label_offsets: Mapping[str, tuple[int, int]] | None = None,
) -> None:
    offset = (label_offsets or {}).get(
        point.policy,
        BASELINE_LABEL_OFFSETS_BY_METRIC.get(attr, {}).get(
            point.policy,
            BASELINE_LABEL_OFFSET,
        ),
    )
    ax.annotate(
        POLICY_PLOT_LABELS.get(point.policy, point.label),
        (point.total_cost_usd, metric_value(point, attr)),
        xytext=offset,
        textcoords="offset points",
        fontsize=ANNOTATION_FONT_SIZE,
        color=policy_color(point.policy),
        ha="center" if offset[0] == 0 else ("right" if offset[0] < 0 else "left"),
        bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.78},
        clip_on=False,
    )


def _annotate_routewise(
    ax: plt.Axes,
    point: FrontierPoint,
    attr: str,
    *,
    routewise_count: int,
    label_offsets: Mapping[float, tuple[int, int]] | None = None,
    label_texts: Mapping[float, str | None] | None = None,
    label_alignments: Mapping[float, str] | None = None,
) -> None:
    label_text = (label_texts or {}).get(
        point.alpha,
        rf"$\alpha={point.alpha:g}$"
        if routewise_count > 1
        else f"RouteWise-{point.alpha:g}",
    )
    if label_text is None:
        return
    offset = (label_offsets or {}).get(
        point.alpha,
        ROUTEWISE_LABEL_OFFSETS_BY_METRIC.get(attr, {}).get(
            point.alpha,
            (5, -14),
        ),
    )
    if attr == "mean_ttft_ms":
        ha = "right" if offset[0] < 0 else "left"
    else:
        x_min, x_max = ax.get_xlim()
        if point.total_cost_usd > (x_min + x_max) / 2.0:
            offset = (-abs(offset[0]), offset[1])
            ha = "right"
        else:
            offset = (abs(offset[0]), offset[1])
            ha = "left"
    ha = (label_alignments or {}).get(point.alpha, ha)
    ax.annotate(
        label_text,
        xy=(point.total_cost_usd, metric_value(point, attr)),
        xytext=offset,
        textcoords="offset points",
        fontsize=ROUTEWISE_ANNOTATION_FONT_SIZE,
        color=ROUTEWISE_COLOR,
        ha=ha,
        bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.78},
        clip_on=False,
    )


def _pad_axes(ax: plt.Axes) -> None:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_pad = max((x_max - x_min) * 0.08, 0.006)
    y_pad = max((y_max - y_min) * 0.12, 0.2)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)


def _plot_routewise_group(
    ax: plt.Axes,
    points: Sequence[FrontierPoint],
    *,
    attr: str,
    color: str,
    marker: str,
    label: str,
    annotate_count: int,
) -> None:
    if not points:
        return
    ordered = sorted(points, key=lambda point: point.alpha or 0.0)
    if len(ordered) > 1:
        ax.plot(
            [point.total_cost_usd for point in ordered],
            [metric_value(point, attr) for point in ordered],
            color=ROUTEWISE_COLOR,
            linewidth=1.0,
            alpha=0.85,
            zorder=2,
        )
    for point in ordered:
        ax.scatter(
            point.total_cost_usd,
            metric_value(point, attr),
            marker=marker,
            s=30,
            color=ROUTEWISE_COLOR,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )


def _ordered_slo_bar_points(
    points: Sequence[FrontierPoint],
    *,
    routewise_alphas: Sequence[float] | None = None,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> list[FrontierPoint]:
    return [
        *routewise_points(points, hedging=False, alphas=routewise_alphas),
        *routewise_points(points, hedging=True, alphas=routewise_alphas),
        *baseline_points(points, order=baseline_order),
    ]


def _short_slo_bar_label(point: FrontierPoint) -> str:
    if point.alpha is None:
        return POLICY_PLOT_LABELS.get(point.policy, point.label)
    return f"RouteWise-{point.alpha:g}"


def _normalized_cost_label(value: float, baseline: float) -> str:
    if baseline <= 0:
        return _cost_label(value)
    return f"{value / baseline:.2f}x"


def _cost_label(value: float) -> str:
    if value >= 100.0:
        return rf"\${value:.0f}"
    if value >= 10.0:
        return rf"\${value:.1f}"
    return rf"\${value:.2f}"


def _slo_bar_color(point: FrontierPoint) -> str:
    if point.alpha is not None:
        return ROUTEWISE_COLOR
    return policy_color(point.policy)


def plot_metric_frontier(
    points: Sequence[FrontierPoint],
    output_path: Path,
    *,
    attr: str,
    ylabel: str,
    xlabel: str = "Normalized cost",
    routewise_alphas: Sequence[float] | None = None,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
    routewise_label_offsets: Mapping[float, tuple[int, int]] | None = None,
    routewise_label_texts: Mapping[float, str | None] | None = None,
    routewise_label_alignments: Mapping[float, str] | None = None,
    baseline_label_offsets: Mapping[str, tuple[int, int]] | None = None,
    baseline_marker_sizes: Mapping[str, float] | None = None,
) -> None:
    apply_column_figure_style()
    routewise_no_hedge = routewise_points(
        points,
        hedging=False,
        alphas=routewise_alphas,
    )
    routewise_hedged = routewise_points(
        points,
        hedging=True,
        alphas=routewise_alphas,
    )
    baselines = baseline_points(points, order=baseline_order)
    plotted_points = [*routewise_no_hedge, *routewise_hedged, *baselines]
    normalized = _normalized_points(plotted_points)
    split_no_hedge = len(routewise_no_hedge)
    split_hedged = split_no_hedge + len(routewise_hedged)
    routewise_no_hedge = normalized[:split_no_hedge]
    routewise_hedged = normalized[split_no_hedge:split_hedged]
    baselines = normalized[split_hedged:]
    routewise_count = len(routewise_no_hedge) + len(routewise_hedged)

    fig, ax = plt.subplots(figsize=COLUMN_FIGSIZE, constrained_layout=True)
    _plot_routewise_group(
        ax,
        routewise_no_hedge,
        attr=attr,
        color=ROUTEWISE_COLOR,
        marker="o",
        label="RouteWise",
        annotate_count=routewise_count,
    )
    _plot_routewise_group(
        ax,
        routewise_hedged,
        attr=attr,
        color=ROUTEWISE_HEDGE_COLOR,
        marker="s",
        label="RouteWise + hedge",
        annotate_count=routewise_count,
    )
    for point in baselines:
        ax.scatter(
            point.total_cost_usd,
            metric_value(point, attr),
            marker=policy_marker(point.policy),
            s=(baseline_marker_sizes or {}).get(point.policy, 24),
            color=policy_color(point.policy),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        _annotate_baseline(
            ax,
            point,
            attr,
            label_offsets=baseline_label_offsets,
        )
    for point in [*routewise_no_hedge, *routewise_hedged]:
        _annotate_routewise(
            ax,
            point,
            attr,
            routewise_count=routewise_count,
            label_offsets=routewise_label_offsets,
            label_texts=routewise_label_texts,
            label_alignments=routewise_label_alignments,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    _pad_axes(ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(output_path)
    plt.close(fig)


def plot_slo_bar(
    points: Sequence[FrontierPoint],
    output_path: Path,
    *,
    routewise_alphas: Sequence[float] | None = None,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    apply_column_figure_style()
    rows = _ordered_slo_bar_points(
        points,
        routewise_alphas=routewise_alphas,
        baseline_order=baseline_order,
    )
    if not rows:
        raise ValueError("no points to plot")

    labels = [_short_slo_bar_label(point) for point in rows]
    slo_values = [metric_value(point, "slo_violation_rate") for point in rows]
    costs = [point.total_cost_usd for point in rows]
    cost_baseline = min(costs)
    y = list(range(len(rows)))
    max_slo = max(slo_values)
    text_pad = max(max_slo * 0.025, 0.12)

    fig, ax = plt.subplots(figsize=SLO_BAR_FIGSIZE, constrained_layout=False)
    ax.barh(
        y,
        slo_values,
        height=0.66,
        color=[_slo_bar_color(point) for point in rows],
        edgecolor="white",
        linewidth=0.45,
    )
    for row_idx, (slo_value, cost) in enumerate(zip(slo_values, costs, strict=True)):
        ax.text(
            slo_value + text_pad,
            row_idx,
            f"{slo_value:.1f}%  {_normalized_cost_label(cost, cost_baseline)}",
            va="center",
            ha="left",
            fontsize=ANNOTATION_FONT_SIZE,
            color="#333333",
        )

    ax.set_xlabel("SLO violations (%)")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.35, alpha=0.35)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    label_room = max(max_slo * 0.38, 2.0)
    ax.set_xlim(0.0, max_slo + label_room)
    fig.subplots_adjust(left=0.33, right=0.99, bottom=0.16, top=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(output_path)
    plt.close(fig)


def plot_mean_ttft_frontier(
    points: Sequence[FrontierPoint],
    output_path: Path,
    **kwargs,
) -> None:
    plot_metric_frontier(
        points,
        output_path,
        attr="mean_ttft_ms",
        ylabel="Mean TTFT (s)",
        **kwargs,
    )


def plot_slo_frontier(
    points: Sequence[FrontierPoint],
    output_path: Path,
    **kwargs,
) -> None:
    plot_slo_bar(points, output_path, **kwargs)


def plot_stacked_mix(
    rows: Sequence[MixRow],
    segments: Sequence[MixSegment],
    output_path: Path,
    *,
    legend_ncols: int,
    legend_fontsize: float = 5.8,
    font_size: float = BASE_FONT_SIZE,
    label_fontsize: float = LABEL_FONT_SIZE,
    tick_fontsize: float = TICK_FONT_SIZE,
    margins: tuple[float, float, float, float] = (0.34, 0.99, 0.16, 0.79),
    show_legend: bool = True,
    x_max: float = 100.0,
) -> None:
    apply_column_figure_style(legend_fontsize=legend_fontsize)
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": label_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
        }
    )
    fig, ax = plt.subplots(figsize=MIX_FIGSIZE, constrained_layout=False)
    y = list(range(len(rows)))
    left = [0.0] * len(rows)
    handles = []
    labels = []
    for segment in segments:
        values = [row.shares.get(segment.key, 0.0) * 100.0 for row in rows]
        if not any(values):
            continue
        bar = ax.barh(
            y,
            values,
            left=left,
            height=0.72,
            color=segment.color,
            edgecolor="white",
            linewidth=0.35,
        )
        handles.append(bar[0])
        labels.append(segment.label)
        left = [old + value for old, value in zip(left, values, strict=True)]
    ax.set_xlim(0, x_max)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Requests (%)")
    ax.set_yticks(y, [row.label for row in rows])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#9a9a9a", alpha=0.24, linewidth=0.5)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(
        left=margins[0],
        right=margins[1],
        bottom=margins[2],
        top=margins[3],
    )
    if show_legend:
        fig.legend(
            handles,
            labels,
            frameon=False,
            ncols=legend_ncols,
            loc="upper center",
            bbox_to_anchor=(0.50, 0.985),
            handlelength=0.9,
            handletextpad=0.28,
            columnspacing=0.55,
            labelspacing=0.35,
            borderaxespad=0.0,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(output_path)
    plt.close(fig)


def _cdf_legend_label(item: CdfSeries, cost_baseline: float | None) -> str:
    if item.total_cost_usd is None or cost_baseline is None:
        return item.label
    return f"{item.label} ({_normalized_cost_label(item.total_cost_usd, cost_baseline)})"


def plot_ttft_cdf(
    series: Sequence[CdfSeries],
    output_path: Path,
    *,
    slo_sec: float = 3.0,
    x_max_sec: float | None = None,
) -> None:
    if not series:
        raise ValueError("no CDF series to plot")
    apply_column_figure_style(legend_fontsize=LEGEND_FONT_SIZE)
    axis_p99 = [item.p99_ms for item in series if item.include_in_axis]
    if x_max_sec is None:
        axis_values = axis_p99 or [item.p99_ms for item in series]
        x_max_sec = max(slo_sec * 4.0, max(axis_values) / 1000.0 * 1.05)
    costs = [
        item.total_cost_usd
        for item in series
        if item.total_cost_usd is not None and item.total_cost_usd > 0
    ]
    cost_baseline = min(costs) if costs else None
    fig, ax = plt.subplots(figsize=COLUMN_FIGSIZE, constrained_layout=False)
    for item in series:
        ax.plot(
            [value / 1000.0 for value in item.x_ms],
            item.y_values,
            color=cdf_color(item.policy, alpha=item.alpha),
            linestyle=cdf_linestyle(item.policy, alpha=item.alpha),
            linewidth=1.25,
            label=_cdf_legend_label(item, cost_baseline),
            drawstyle=item.drawstyle,
        )
    ax.axvline(slo_sec, color="#444444", linewidth=0.8, linestyle=":")
    ax.text(
        slo_sec + 0.12,
        0.86,
        f"{slo_sec:g}s SLO",
        color="#444444",
        fontsize=ANNOTATION_FONT_SIZE,
        ha="left",
        va="top",
    )
    ax.set_xlim(0.0, x_max_sec)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("TTFT (s)")
    ax.set_ylabel("CDF")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(
        frameon=False,
        loc="lower right",
        ncols=1,
        handlelength=1.6,
        columnspacing=0.7,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.17, top=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _series_percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def plot_ttft_boxplot(
    series: Sequence[BoxSeries],
    output_path: Path,
    *,
    slo_sec: float = 3.0,
    x_max_sec: float | None = None,
) -> None:
    if not series:
        raise ValueError("no boxplot series to plot")
    apply_column_figure_style(legend_fontsize=LEGEND_FONT_SIZE)
    plt.rcParams.update({"figure.figsize": COLUMN_FIGSIZE})

    sorted_samples_sec: list[list[float]] = []
    p95_sec: list[float] = []
    for item in series:
        sorted_ms = sorted(item.samples_ms)
        sorted_samples_sec.append([value / 1000.0 for value in sorted_ms])
        p95_sec.append(_series_percentile(sorted_ms, 95.0) / 1000.0)

    if x_max_sec is None:
        valid_p95 = [value for value in p95_sec if not math.isnan(value)]
        tail = max(valid_p95) if valid_p95 else slo_sec * 1.6
        x_max_sec = max(slo_sec * 1.6, tail * 1.08)

    fig, ax = plt.subplots(figsize=COLUMN_FIGSIZE, constrained_layout=False)
    box = ax.boxplot(
        sorted_samples_sec,
        vert=False,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        widths=0.6,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"edgecolor": "#555555", "linewidth": 0.8},
    )
    for patch, item in zip(box["boxes"], series, strict=True):
        patch.set_facecolor(policy_color(item.policy, alpha=item.alpha))
        patch.set_alpha(0.78)

    ax.axvline(slo_sec, color="#444444", linestyle=":", linewidth=0.8)
    ax.text(
        slo_sec + 0.08,
        len(series) + 0.45,
        f"{slo_sec:g}s SLO",
        color="#444444",
        fontsize=ANNOTATION_FONT_SIZE,
        ha="left",
        va="bottom",
    )

    labels = [item.label for item in series]
    ax.set_yticks(range(1, len(labels) + 1), labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, x_max_sec)
    ax.set_xlabel("TTFT (s)")
    ax.grid(axis="x", color="#9a9a9a", alpha=0.28, linewidth=0.5)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.30, right=0.99, bottom=0.18, top=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(output_path)
    plt.close(fig)


def plot_hedging_p99(
    points: Sequence[FrontierPoint],
    output_path: Path,
    *,
    alphas: Sequence[float],
) -> None:
    apply_column_figure_style()
    no_hedge = {
        point.alpha: point
        for point in routewise_points(points, hedging=False)
        if point.p99_ms is not None
    }
    hedged = {
        point.alpha: point
        for point in routewise_points(points, hedging=True)
        if point.p99_ms is not None
    }
    selected_alphas = [alpha for alpha in alphas if alpha in no_hedge and alpha in hedged]
    if not selected_alphas:
        raise ValueError("no matching hedged/non-hedged RouteWise P99 points")
    labels = [f"{alpha:g}" for alpha in selected_alphas]
    x = list(range(len(selected_alphas)))
    width = 0.36
    fig, ax = plt.subplots(figsize=COLUMN_FIGSIZE, constrained_layout=True)
    ax.bar(
        [value - width / 2 for value in x],
        [no_hedge[alpha].p99_ms / 1000.0 for alpha in selected_alphas],
        width=width,
        color=ROUTEWISE_COLOR,
        label="RW",
    )
    ax.bar(
        [value + width / 2 for value in x],
        [hedged[alpha].p99_ms / 1000.0 for alpha in selected_alphas],
        width=width,
        color=ROUTEWISE_HEDGE_COLOR,
        label="RW + hedge",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("P99 TTFT (s)")
    ax.set_ylim(
        0,
        max(no_hedge[alpha].p99_ms for alpha in selected_alphas) / 1000.0 * 1.18,
    )
    ax.grid(axis="y", linewidth=0.35, alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
