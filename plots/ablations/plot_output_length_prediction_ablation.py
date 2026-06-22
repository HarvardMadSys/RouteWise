"""Plot output-length prediction error ablation figures."""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from plots.helpers import save_figure
from plots.style import apply_style

DEFAULT_INPUT_DIR = Path(
    "outputs/e2e_all/output_length_prediction_ablation/"
    "burstgpt_llama2048_minimax_q4_c12_scaled_latency_q035_c007"
)

POLICY_ORDER = (
    "ablation_lp_only_alpha0",
    "ablation_lp_only_alpha25",
    "ablation_lp_only_alpha50",
    "ablation_lp_hedging_alpha0",
    "ablation_lp_hedging_alpha25",
    "ablation_lp_hedging_alpha50",
)
POLICY_LABELS = {
    "ablation_lp_only_alpha0": "LP α=0",
    "ablation_lp_only_alpha25": "LP α=0.25",
    "ablation_lp_only_alpha50": "LP α=0.5",
    "ablation_lp_hedging_alpha0": "Hedge α=0",
    "ablation_lp_hedging_alpha25": "Hedge α=0.25",
    "ablation_lp_hedging_alpha50": "Hedge α=0.5",
}
# Legend column order: pair LP (solid) and Hedge (dashed) of the same alpha in
# one column. With matplotlib's column-major fill and ncols=3, consecutive
# entries fill each column top-to-bottom.
LEGEND_ORDER = (
    "ablation_lp_only_alpha0",
    "ablation_lp_hedging_alpha0",
    "ablation_lp_only_alpha25",
    "ablation_lp_hedging_alpha25",
    "ablation_lp_only_alpha50",
    "ablation_lp_hedging_alpha50",
)
POLICY_COLORS = {
    "ablation_lp_only_alpha0": "#1f77b4",
    "ablation_lp_only_alpha25": "#2ca02c",
    "ablation_lp_only_alpha50": "#ff7f0e",
    "ablation_lp_hedging_alpha0": "#1f77b4",
    "ablation_lp_hedging_alpha25": "#2ca02c",
    "ablation_lp_hedging_alpha50": "#ff7f0e",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Ablation root containing err_*/summary.csv. Defaults to {DEFAULT_INPUT_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--scenario",
        help="Optional scenario name to select when err_*/summary.csv contains multiple scenarios.",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    rows = _load_rows(input_dir, scenario=args.scenario)
    delta_rows = _delta_rows(rows)
    worst_rows = _worst_case_rows(delta_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "output_length_prediction_summary.csv", rows)
    _write_csv(output_dir / "output_length_prediction_delta_summary.csv", delta_rows)
    _write_csv(output_dir / "output_length_prediction_worst_case_summary.csv", worst_rows)

    figure_dir = output_dir / "figures"
    _plot_cost_delta_lines(delta_rows, figure_dir)
    _plot_worst_case_deviation(worst_rows, figure_dir)
    _plot_mix_shift(delta_rows, figure_dir)

    print(
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "rows": len(delta_rows),
            "figures": str(figure_dir),
        }
    )
    return 0


def _load_rows(input_dir: Path, *, scenario: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(input_dir.glob("err_*")):
        if not run_dir.is_dir():
            continue
        error_pct = _error_pct(run_dir.name)
        summary_path = run_dir / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        with summary_path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                if raw["policy"] not in POLICY_ORDER:
                    continue
                if scenario is not None and raw.get("scenario") != scenario:
                    continue
                tier_mix = _parse_mapping(raw["tier_mix"])
                rows.append(
                    {
                        "prediction_error_pct": error_pct,
                        "scenario": raw.get("scenario"),
                        "policy": raw["policy"],
                        "output_predictor": raw.get("output_predictor"),
                        "total_cost_usd": float(raw["total_cost_usd"]),
                        "api_cost_usd": float(raw["api_cost_usd"]),
                        "subscription_fixed_cost_usd": float(raw["subscription_fixed_cost_usd"]),
                        "mean_ttft_ms": float(raw["mean_ttft_ms"]),
                        "p50_ms": float(raw["p50_ms"]),
                        "p90_ms": float(raw["p90_ms"]),
                        "p99_ms": float(raw["p99_ms"]),
                        "slo_violation_rate": float(raw["slo_violation_rate"]),
                        "hedge_rate": float(raw["hedge_rate"]),
                        "api_mix": float(tier_mix.get("api", 0.0)),
                        "quota_mix": float(tier_mix.get("quota", 0.0)),
                        "concurrency_mix": float(tier_mix.get("concurrency", 0.0)),
                    }
                )
    if not rows:
        raise ValueError(f"no err_*/summary.csv rows found under {input_dir}")
    scenarios = sorted({str(row.get("scenario")) for row in rows})
    if scenario is None and len(scenarios) > 1:
        raise ValueError(
            "multiple scenarios found; pass --scenario to select one: " + ", ".join(scenarios)
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("scenario") or ""),
            row["prediction_error_pct"],
            POLICY_ORDER.index(row["policy"]),
        ),
    )


def _delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {row["policy"]: row for row in rows if row["prediction_error_pct"] == 0}
    missing = [policy for policy in POLICY_ORDER if policy not in baseline]
    if missing:
        raise ValueError(f"missing err_000 baseline rows for policies: {missing}")

    output: list[dict[str, Any]] = []
    for row in rows:
        base = baseline[row["policy"]]
        mix_tv = 0.5 * sum(
            abs(row[key] - base[key]) for key in ("api_mix", "quota_mix", "concurrency_mix")
        )
        output.append(
            {
                **row,
                "policy_label": POLICY_LABELS[row["policy"]],
                "total_cost_delta_pct": _pct_delta(row["total_cost_usd"], base["total_cost_usd"]),
                "api_cost_delta_pct": _pct_delta(row["api_cost_usd"], base["api_cost_usd"]),
                "mean_ttft_delta_pct": _pct_delta(row["mean_ttft_ms"], base["mean_ttft_ms"]),
                "p99_delta_pct": _pct_delta(row["p99_ms"], base["p99_ms"]),
                "slo_delta_pp": (row["slo_violation_rate"] - base["slo_violation_rate"]) * 100.0,
                "hedge_rate_delta_pp": (row["hedge_rate"] - base["hedge_rate"]) * 100.0,
                "provider_mix_tv_pct": mix_tv * 100.0,
            }
        )
    return output


def _worst_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for error_pct in sorted(
        {row["prediction_error_pct"] for row in rows if row["prediction_error_pct"] != 0}
    ):
        selected = [row for row in rows if row["prediction_error_pct"] == error_pct]
        output.append(
            {
                "prediction_error_pct": error_pct,
                "max_abs_total_cost_delta_pct": max(
                    abs(row["total_cost_delta_pct"]) for row in selected
                ),
                "max_abs_api_cost_delta_pct": max(
                    abs(row["api_cost_delta_pct"]) for row in selected
                ),
                "max_abs_mean_ttft_delta_pct": max(
                    abs(row["mean_ttft_delta_pct"]) for row in selected
                ),
                "max_abs_p99_delta_pct": max(abs(row["p99_delta_pct"]) for row in selected),
                "max_provider_mix_tv_pct": max(row["provider_mix_tv_pct"] for row in selected),
                "max_abs_hedge_rate_delta_pp": max(
                    abs(row["hedge_rate_delta_pp"]) for row in selected
                ),
            }
        )
    return output


def _plot_cost_delta_lines(rows: list[dict[str, Any]], output_dir: Path) -> None:
    _apply_output_length_style()
    fig, ax = plt.subplots(figsize=(2.3, 1.95))
    handles: dict[str, Any] = {}
    for policy in POLICY_ORDER:
        policy_rows = _policy_rows(rows, policy)
        linestyle = "--" if "hedging" in policy else "-"
        (handles[policy],) = ax.plot(
            [row["prediction_error_pct"] for row in policy_rows],
            [row["total_cost_delta_pct"] for row in policy_rows],
            marker="o",
            color=POLICY_COLORS[policy],
            linestyle=linestyle,
            label=POLICY_LABELS[policy],
        )
    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.65)
    ax.set_xlabel("Length prediction bias (%)")
    ax.set_ylabel("Total cost delta (%)")
    ax.set_ylim(-6.0, 6.0)
    ax.legend(
        [handles[policy] for policy in LEGEND_ORDER],
        [POLICY_LABELS[policy] for policy in LEGEND_ORDER],
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        bbox_transform=fig.transFigure,
        columnspacing=0.4,
        handlelength=1.3,
        handletextpad=0.25,
    )
    ax.grid(True, alpha=0.24)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.79, bottom=0.22)
    save_figure(
        fig,
        output_dir,
        "output_length_prediction_cost_delta_lines",
        ["pdf", "png"],
        full_canvas=True,
    )
    plt.close(fig)


def _plot_cost_delta_heatmap(rows: list[dict[str, Any]], output_dir: Path) -> None:
    _apply_output_length_style()
    errors = sorted({row["prediction_error_pct"] for row in rows})
    matrix = _matrix(rows, errors=errors, value_key="total_cost_delta_pct")
    limit = max(5.0, float(np.ceil(np.max(np.abs(matrix)))))
    fig, ax = plt.subplots(figsize=(5.2, 2.65))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_yticks(range(len(POLICY_ORDER)))
    ax.set_yticklabels([POLICY_LABELS[policy] for policy in POLICY_ORDER])
    ax.set_xticks(range(len(errors)))
    ax.set_xticklabels([str(error) for error in errors])
    ax.set_xlabel("Output-token prediction bias (%)")
    ax.set_title("Total cost delta vs zero-bias predictor (%)", pad=5)
    for row_idx, _policy in enumerate(POLICY_ORDER):
        for col_idx, _error in enumerate(errors):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:+.1f}",
                ha="center",
                va="center",
                color="white" if abs(value) >= 0.62 * limit else "black",
                fontsize=6.8,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("delta (%)", rotation=270, labelpad=10)
    save_figure(fig, output_dir, "output_length_prediction_cost_delta_heatmap", ["pdf", "png"])
    plt.close(fig)


def _plot_worst_case_deviation(rows: list[dict[str, Any]], output_dir: Path) -> None:
    _apply_output_length_style()
    errors = [row["prediction_error_pct"] for row in rows]
    fig, ax = plt.subplots(figsize=(5.0, 2.55))
    series = (
        ("max_abs_total_cost_delta_pct", "cost", "#1f77b4"),
        ("max_abs_p99_delta_pct", "P99 TTFT", "#d62728"),
        ("max_provider_mix_tv_pct", "mix TV", "#2ca02c"),
    )
    for key, label, color in series:
        ax.plot(errors, [row[key] for row in rows], marker="o", label=label, color=color)
    ax.set_xlabel("Output-token prediction bias (%)")
    ax.set_ylabel("Worst-case deviation (%)")
    ax.set_title("Max zero-bias deviation across policies", pad=5)
    ax.legend(frameon=False, ncols=3, loc="upper left", columnspacing=1.0)
    ax.grid(True, alpha=0.24)
    save_figure(fig, output_dir, "output_length_prediction_worst_case_deviation", ["pdf", "png"])
    plt.close(fig)


def _plot_mix_shift(rows: list[dict[str, Any]], output_dir: Path) -> None:
    _apply_output_length_style()
    errors = tuple(sorted({int(row["prediction_error_pct"]) for row in rows}))
    labels = ("API", "Quota", "Concurrency")
    keys = ("api_mix", "quota_mix", "concurrency_mix")
    colors = ("#4c78a8", "#72b7b2", "#f58518")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.65), sharey=True)
    for ax, policy in zip(
        axes,
        ("ablation_lp_only_alpha0", "ablation_lp_hedging_alpha0", "ablation_lp_hedging_alpha50"),
        strict=True,
    ):
        selected = [
            row for row in _policy_rows(rows, policy) if row["prediction_error_pct"] in errors
        ]
        bottoms = np.zeros(len(selected))
        x = np.arange(len(selected))
        for key, label, color in zip(keys, labels, colors, strict=True):
            values = np.asarray([row[key] * 100.0 for row in selected])
            ax.bar(x, values, bottom=bottoms, color=color, width=0.68, label=label)
            bottoms += values
        ax.set_title(POLICY_LABELS[policy], pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([str(row["prediction_error_pct"]) for row in selected])
        ax.set_xlabel("bias (%)")
        ax.grid(True, axis="y", alpha=0.18)
    axes[0].set_ylabel("Request mix (%)")
    axes[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save_figure(fig, output_dir, "output_length_prediction_request_mix_shift", ["pdf", "png"])
    plt.close(fig)


def _matrix(rows: list[dict[str, Any]], *, errors: list[int], value_key: str) -> np.ndarray:
    by_policy_error = {
        (row["policy"], int(row["prediction_error_pct"])): float(row[value_key]) for row in rows
    }
    return np.asarray(
        [[by_policy_error[(policy, error)] for error in errors] for policy in POLICY_ORDER],
        dtype=float,
    )


def _policy_rows(rows: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if row["policy"] == policy],
        key=lambda row: row["prediction_error_pct"],
    )


def _parse_mapping(value: str) -> dict[str, float]:
    payload = ast.literal_eval(value)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping payload, got {value!r}")
    return {str(key): float(val) for key, val in payload.items()}


def _error_pct(name: str) -> int:
    try:
        return int(name.removeprefix("err_"))
    except ValueError as exc:
        raise ValueError(f"invalid error directory name: {name}") from exc


def _pct_delta(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if value == 0.0 else float("nan")
    return (value - baseline) / baseline * 100.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _apply_output_length_style() -> None:
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
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
