"""Quota-subscription simulator figures for the cost-layer section.

Consumes one or two ``routewise simulator cost-layer`` output directories:

* a q-sweep directory containing ``summary.csv`` across subscription counts;
* optionally, a q-fixed distribution-robustness directory containing
  ``summary.csv`` across latency families.

The main §1.2 question is the optimal number of subscriptions under a fixed
latency family (heavy-tail / LogNormal by default). Distribution robustness is
reported separately and should not be used to pick the headline setting.
"""

from __future__ import annotations

import argparse
import ast
import csv
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots.helpers import save_figure
from plots.style import apply_style

POLICY_ORDER = (
    "offline",
    "ablation_lp_only_p0",
    "greedy_cost",
    "ablation_lp_only_p25",
    "ablation_lp_only_p50",
    "ablation_lp_only_p75",
    "ablation_lp_only_p100",
    "random",
)

MAIN_POLICIES = (
    "offline",
    "ablation_lp_only_p0",
    "greedy_cost",
    "random",
)

MARGINAL_POLICIES = (
    "offline",
    "greedy_cost",
    "ablation_lp_only_p0",
)

ROUTEWISE_P_POLICIES = (
    "ablation_lp_only_p0",
    "ablation_lp_only_p25",
    "ablation_lp_only_p50",
    "ablation_lp_only_p75",
    "ablation_lp_only_p100",
)

POLICY_LABELS = {
    "offline": "Offline",
    "ablation_lp_only_p0": "RW p=0",
    "ablation_lp_only_p25": "RW p=.25",
    "ablation_lp_only_p50": "RW p=.50",
    "ablation_lp_only_p75": "RW p=.75",
    "ablation_lp_only_p100": "RW p=1",
    "greedy_cost": "Greedy",
    "random": "Random",
}

POLICY_COLORS = {
    "offline": "#555555",
    "ablation_lp_only_p0": "#9467bd",
    "ablation_lp_only_p25": "#8c6bb1",
    "ablation_lp_only_p50": "#6a51a3",
    "ablation_lp_only_p75": "#54278f",
    "ablation_lp_only_p100": "#3f007d",
    "greedy_cost": "#1f77b4",
    "random": "#7f8c8d",
}

POLICY_LINESTYLES = {
    "offline": "-",
    "ablation_lp_only_p0": "-.",
    "greedy_cost": "-",
    "random": "--",
}

LATENCY_FAMILY_ORDER = (
    "uniform",
    "normal",
    "heavy_tail",
    "real_world",
)

LATENCY_FAMILY_LABELS = {
    "uniform": "Uniform",
    "normal": "Normal",
    "heavy_tail": "LogNormal",
    "real_world": "Real-world",
}


def apply_quota_style() -> None:
    """Use compact paper-panel typography consistent with §1.1 figures."""
    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.55,
            "lines.markersize": 4.2,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
        }
    )


def load_summary(input_dir: Path) -> list[dict[str, str]]:
    """Load simulator section summary rows."""
    path = input_dir / "summary.csv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_mapping(value: str) -> dict[str, float]:
    payload = ast.literal_eval(value)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping payload, got: {value!r}")
    return {str(key): float(val) for key, val in payload.items()}


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _policy_rows(
    rows: list[dict[str, str]],
    policy: str,
    *,
    plan: str | None = None,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["policy"] == policy and (plan is None or row["subscription_plan"] == plan)
    ]
    return sorted(selected, key=lambda row: _int(row, "subscription_count"))


def _available_policies(rows: list[dict[str, str]]) -> list[str]:
    present = {row["policy"] for row in rows}
    return [policy for policy in POLICY_ORDER if policy in present]


def _best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda row: _float(row, "total_cost_usd_per_run"))


def _quota_fraction(row: dict[str, str]) -> float:
    tier_mix = _parse_mapping(row["tier_mix"])
    if "quota" in tier_mix:
        return tier_mix["quota"]
    provider_mix = _parse_mapping(row["provider_mix"])
    return sum(
        fraction
        for provider, fraction in provider_mix.items()
        if "quota" in provider or "chutes" in provider
    )


def _quota_fits_in_trace(row: dict[str, str]) -> str:
    """Return the quota-fit exclusion flag from the current summary schema."""
    return row["quota_fits_in_trace"]


def _fixed_fee_per_subscription(rows: list[dict[str, str]]) -> float:
    fees = []
    for row in rows:
        count = _int(row, "subscription_count")
        if count > 0:
            fees.append(_float(row, "subscription_fixed_cost_usd_per_run") / count)
    if not fees:
        return 0.0
    return float(np.median(fees))


def _format_money(value: float) -> str:
    return f"${value / 1000:.1f}k" if value >= 1000 else f"${value:.0f}"


def _set_q_ticks(ax: plt.Axes, counts: list[int]) -> None:
    if len(counts) > 10:
        preferred = [1, 4, 8, 12, 16, 20, 24, 32, 40]
        ticks = [count for count in preferred if count in counts]
    else:
        ticks = counts
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(count) for count in ticks])


def plot_total_cost_main(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    plan: str,
) -> None:
    """Plot the headline q-sweep U-shape for the main policies."""
    fig, ax = plt.subplots(figsize=(3.55, 2.25))
    all_counts = sorted({_int(row, "subscription_count") for row in rows})
    for policy in MAIN_POLICIES:
        policy_rows = _policy_rows(rows, policy, plan=plan)
        if not policy_rows:
            continue
        counts = [_int(row, "subscription_count") for row in policy_rows]
        totals = [_float(row, "total_cost_usd_per_run") for row in policy_rows]
        ax.plot(
            counts,
            totals,
            marker="o",
            color=POLICY_COLORS[policy],
            linestyle=POLICY_LINESTYLES.get(policy, "-"),
            label=POLICY_LABELS[policy],
        )
        best = _best_row(policy_rows)
        ax.scatter(
            [_int(best, "subscription_count")],
            [_float(best, "total_cost_usd_per_run")],
            s=32,
            facecolor="white",
            edgecolor=POLICY_COLORS[policy],
            linewidth=1.2,
            zorder=4,
        )

    ax.set_xlabel("Subscriptions (q)")
    ax.set_ylabel("Total cost per run ($)")
    _set_q_ticks(ax, all_counts)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.34))
    save_figure(
        fig,
        output_dir,
        f"cost_layer_quota_{plan}_q_sweep_total_cost",
        formats=["pdf"],
    )
    plt.close(fig)


def plot_total_cost_all_policies(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    plan: str,
) -> None:
    """Plot every policy so the p-dependent q* shift is visible."""
    fig, ax = plt.subplots(figsize=(4.25, 2.4))
    all_counts = sorted({_int(row, "subscription_count") for row in rows})
    for policy in _available_policies(rows):
        policy_rows = _policy_rows(rows, policy, plan=plan)
        if not policy_rows:
            continue
        counts = [_int(row, "subscription_count") for row in policy_rows]
        totals = [_float(row, "total_cost_usd_per_run") for row in policy_rows]
        ax.plot(
            counts,
            totals,
            marker="o",
            color=POLICY_COLORS[policy],
            linestyle=POLICY_LINESTYLES.get(policy, "-"),
            label=POLICY_LABELS[policy],
            alpha=0.95,
        )
    ax.set_xlabel("Subscriptions (q)")
    ax.set_ylabel("Total cost per run ($)")
    _set_q_ticks(ax, all_counts)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.27))
    save_figure(
        fig,
        output_dir,
        f"cost_layer_quota_{plan}_q_sweep_all_policies",
        formats=["pdf"],
    )
    plt.close(fig)


def plot_quota_fraction(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    plan: str,
) -> None:
    """Plot how much traffic each policy sends to quota as q increases."""
    fig, ax = plt.subplots(figsize=(3.55, 2.2))
    all_counts = sorted({_int(row, "subscription_count") for row in rows})
    for policy in MAIN_POLICIES:
        policy_rows = _policy_rows(rows, policy, plan=plan)
        if not policy_rows:
            continue
        counts = [_int(row, "subscription_count") for row in policy_rows]
        fractions = [_quota_fraction(row) for row in policy_rows]
        ax.plot(
            counts,
            fractions,
            marker="o",
            color=POLICY_COLORS[policy],
            linestyle=POLICY_LINESTYLES.get(policy, "-"),
            label=POLICY_LABELS[policy],
        )
    ax.set_xlabel("Subscriptions (q)")
    ax.set_ylabel("Quota request fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    _set_q_ticks(ax, all_counts)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.34))
    save_figure(
        fig,
        output_dir,
        f"cost_layer_quota_{plan}_q_sweep_quota_fraction",
        formats=["pdf"],
    )
    plt.close(fig)


def plot_marginal_saving(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    plan: str,
) -> None:
    """Plot marginal API-cost saving per extra subscription against the fee."""
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    fee = _fixed_fee_per_subscription(rows)
    for policy in MARGINAL_POLICIES:
        policy_rows = _policy_rows(rows, policy, plan=plan)
        if len(policy_rows) < 2:
            continue
        x_values: list[float] = []
        y_values: list[float] = []
        for previous, current in pairwise(policy_rows):
            q0 = _int(previous, "subscription_count")
            q1 = _int(current, "subscription_count")
            delta_q = q1 - q0
            if delta_q <= 0:
                continue
            saving = _float(previous, "api_cost_usd_per_run") - _float(
                current,
                "api_cost_usd_per_run",
            )
            x_values.append((q0 + q1) / 2.0)
            y_values.append(saving / delta_q)
        ax.plot(
            x_values,
            y_values,
            marker="o",
            color=POLICY_COLORS[policy],
            linestyle=POLICY_LINESTYLES.get(policy, "-"),
            label=POLICY_LABELS[policy],
        )
    ax.axhline(
        fee,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"fee ({_format_money(fee)}/q)",
    )
    all_counts = sorted({_int(row, "subscription_count") for row in rows})
    ax.set_xlabel("Subscription interval midpoint")
    ax.set_ylabel("API saving per +1q ($)")
    _set_q_ticks(ax, all_counts)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="upper right")
    save_figure(
        fig,
        output_dir,
        f"cost_layer_quota_{plan}_q_sweep_marginal_saving",
        formats=["pdf"],
    )
    plt.close(fig)


def write_optimal_count_table(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    plan: str,
) -> None:
    """Emit the q* table used by the paper text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cost_layer_quota_{plan}_optimal_count_table.csv"
    fieldnames = (
        "policy",
        "q_star",
        "total_cost_usd_per_run",
        "api_cost_usd_per_run",
        "subscription_fixed_cost_usd_per_run",
        "quota_request_fraction",
        "mean_ttft_ms",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy in _available_policies(rows):
            policy_rows = _policy_rows(rows, policy, plan=plan)
            if not policy_rows:
                continue
            best = _best_row(policy_rows)
            writer.writerow(
                {
                    "policy": POLICY_LABELS[policy],
                    "q_star": _int(best, "subscription_count"),
                    "total_cost_usd_per_run": f"{_float(best, 'total_cost_usd_per_run'):.6f}",
                    "api_cost_usd_per_run": f"{_float(best, 'api_cost_usd_per_run'):.6f}",
                    "subscription_fixed_cost_usd_per_run": (
                        f"{_float(best, 'subscription_fixed_cost_usd_per_run'):.6f}"
                    ),
                    "quota_request_fraction": f"{_quota_fraction(best):.6f}",
                    "mean_ttft_ms": f"{_float(best, 'mean_ttft_ms'):.3f}",
                }
            )
    print(f"Saved: {path}")


def write_q_sweep_table(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    plan: str,
) -> None:
    """Emit a compact q-sweep table with paper-facing fields only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cost_layer_quota_{plan}_q_sweep_table.csv"
    fieldnames = (
        "policy",
        "subscription_count",
        "total_cost_usd_per_run",
        "api_cost_usd_per_run",
        "subscription_fixed_cost_usd_per_run",
        "quota_request_fraction",
        "trace_paper_grade",
        "quota_fits_in_trace",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy in _available_policies(rows):
            for row in _policy_rows(rows, policy, plan=plan):
                writer.writerow(
                    {
                        "policy": POLICY_LABELS[policy],
                        "subscription_count": _int(row, "subscription_count"),
                        "total_cost_usd_per_run": (
                            f"{_float(row, 'total_cost_usd_per_run'):.6f}"
                        ),
                        "api_cost_usd_per_run": f"{_float(row, 'api_cost_usd_per_run'):.6f}",
                        "subscription_fixed_cost_usd_per_run": (
                            f"{_float(row, 'subscription_fixed_cost_usd_per_run'):.6f}"
                        ),
                        "quota_request_fraction": f"{_quota_fraction(row):.6f}",
                        "trace_paper_grade": row["trace_paper_grade"],
                        "quota_fits_in_trace": _quota_fits_in_trace(row),
                    }
                )
    print(f"Saved: {path}")


def _distribution_rows_by_family_policy(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        family = row.get("latency_family", "")
        if family:
            mapping[(family, row["policy"])] = row
    return mapping


def plot_distribution_main(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    plan: str,
) -> None:
    """Plot q-fixed distribution robustness for the headline policies."""
    mapping = _distribution_rows_by_family_policy(rows)
    families = [family for family in LATENCY_FAMILY_ORDER if (family, "offline") in mapping]
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    x = np.arange(len(families))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MAIN_POLICIES))
    for offset, policy in zip(offsets, MAIN_POLICIES, strict=True):
        values = [
            _float(mapping[(family, policy)], "total_cost_usd_per_run")
            for family in families
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
        )
    q = _int(next(iter(mapping.values())), "subscription_count")
    ax.set_xlabel("Latency family")
    ax.set_ylabel("Total cost per run ($)")
    ax.set_xticks(x)
    ax.set_xticklabels([LATENCY_FAMILY_LABELS[family] for family in families])
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    save_figure(
        fig,
        output_dir,
        f"cost_layer_quota_{plan}_q{q}_distribution_total_cost_main",
        formats=["pdf"],
    )
    plt.close(fig)


def plot_distribution_routewise_p(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    plan: str,
) -> None:
    """Plot distribution robustness for the RouteWise p-sweep policies."""
    mapping = _distribution_rows_by_family_policy(rows)
    families = [
        family
        for family in LATENCY_FAMILY_ORDER
        if (family, "ablation_lp_only_p0") in mapping
    ]
    fig, ax = plt.subplots(figsize=(3.9, 2.2))
    x = np.arange(len(families))
    width = 0.14
    offsets = np.linspace(-2.0 * width, 2.0 * width, len(ROUTEWISE_P_POLICIES))
    for offset, policy in zip(offsets, ROUTEWISE_P_POLICIES, strict=True):
        values = [
            _float(mapping[(family, policy)], "total_cost_usd_per_run")
            for family in families
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
        )
    q = _int(next(iter(mapping.values())), "subscription_count")
    ax.set_xlabel("Latency family")
    ax.set_ylabel("Total cost per run ($)")
    ax.set_xticks(x)
    ax.set_xticklabels([LATENCY_FAMILY_LABELS[family] for family in families])
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.23))
    save_figure(
        fig,
        output_dir,
        f"cost_layer_quota_{plan}_q{q}_distribution_total_cost_routewise_p",
        formats=["pdf"],
    )
    plt.close(fig)


def write_distribution_table(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    plan: str,
) -> None:
    """Emit distribution robustness table for q-fixed runs."""
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    q = _int(rows[0], "subscription_count")
    path = output_dir / f"cost_layer_quota_{plan}_q{q}_distribution_table.csv"
    fieldnames = (
        "latency_family",
        "policy",
        "total_cost_usd_per_run",
        "quota_request_fraction",
        "mean_ttft_ms",
        "p50_ms",
        "p99_ms",
    )
    mapping = _distribution_rows_by_family_policy(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for family in LATENCY_FAMILY_ORDER:
            for policy in POLICY_ORDER:
                row = mapping.get((family, policy))
                if row is None:
                    continue
                writer.writerow(
                    {
                        "latency_family": LATENCY_FAMILY_LABELS[family],
                        "policy": POLICY_LABELS[policy],
                        "total_cost_usd_per_run": (
                            f"{_float(row, 'total_cost_usd_per_run'):.6f}"
                        ),
                        "quota_request_fraction": f"{_quota_fraction(row):.6f}",
                        "mean_ttft_ms": f"{_float(row, 'mean_ttft_ms'):.3f}",
                        "p50_ms": f"{_float(row, 'p50_ms'):.3f}",
                        "p99_ms": f"{_float(row, 'p99_ms'):.3f}",
                    }
                )
    print(f"Saved: {path}")


def make_quota_plots(
    q_sweep_input_dir: Path,
    output_dir: Path,
    *,
    plan: str,
    distribution_input_dir: Path | None = None,
) -> None:
    """Generate all quota-subscription simulator figures and tables."""
    apply_quota_style()
    q_rows = load_summary(q_sweep_input_dir)
    q_rows = [row for row in q_rows if row["subscription_plan"] == plan]
    if not q_rows:
        raise ValueError(f"no rows found for subscription plan {plan!r}")
    plot_total_cost_main(q_rows, output_dir, plan=plan)
    plot_total_cost_all_policies(q_rows, output_dir, plan=plan)
    plot_quota_fraction(q_rows, output_dir, plan=plan)
    plot_marginal_saving(q_rows, output_dir, plan=plan)
    write_optimal_count_table(q_rows, output_dir, plan=plan)
    write_q_sweep_table(q_rows, output_dir, plan=plan)

    if distribution_input_dir is not None:
        distribution_rows = load_summary(distribution_input_dir)
        distribution_rows = [
            row for row in distribution_rows if row["subscription_plan"] == plan
        ]
        if distribution_rows:
            plot_distribution_main(distribution_rows, output_dir, plan=plan)
            plot_distribution_routewise_p(distribution_rows, output_dir, plan=plan)
            write_distribution_table(distribution_rows, output_dir, plan=plan)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q-sweep-input-dir",
        type=Path,
        required=True,
        help="Quota q-sweep output directory containing summary.csv.",
    )
    parser.add_argument(
        "--distribution-input-dir",
        type=Path,
        default=None,
        help="Optional q-fixed distribution robustness directory containing summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure/table output directory. Defaults to <q-sweep-input-dir>/figures.",
    )
    parser.add_argument(
        "--plan",
        default="chutes",
        help="Subscription plan key to plot.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir or args.q_sweep_input_dir / "figures"
    make_quota_plots(
        args.q_sweep_input_dir,
        output_dir,
        plan=args.plan,
        distribution_input_dir=args.distribution_input_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
