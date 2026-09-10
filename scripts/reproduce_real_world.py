"""Recompute the recorded real-world results and plot Figures 1, 6, and 7."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICIES = (
    "budget_range_alpha0_hedge",
    "budget_range_alpha25_hedge",
    "budget_range_alpha50_hedge",
    "budget_range_alpha75_hedge",
    "budget_range_alpha100_hedge",
    "greedy_cost",
    "greedy_latency",
    "or_auto",
    "or_sort_latency",
    "or_sort_cost",
)
METRICS = ("total_cost_usd", "ttft_mean_ms", "ttft_p99_ms", "slo_violation_rate")


def check_summary(actual_path: Path, reference_path: Path) -> None:
    """Compare independently generated metrics with the archived aggregates."""
    actual_rows = json.loads(actual_path.read_text(encoding="utf-8"))
    expected_rows = json.loads(reference_path.read_text(encoding="utf-8"))
    actual = {row["policy"]: row for row in actual_rows}
    expected = {row["policy"]: row for row in expected_rows}
    if len(actual) != len(actual_rows) or len(expected) != len(expected_rows):
        raise ValueError("duplicate policies in the generated or reference summary")
    if actual.keys() != expected.keys():
        raise ValueError("generated and reference summaries contain different policies")
    for policy, row in actual.items():
        if row["n"] != expected[policy]["n"]:
            raise ValueError(f"{policy}: request count differs from the reference")
        for metric in METRICS:
            value, reference = float(row[metric]), float(expected[policy][metric])
            if not math.isclose(value, reference, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{policy}: {metric}={value}, reference={reference}")
    print(f"PASS: {len(actual)} policies match the archived request counts and metrics.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "real_eval_records")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "figures" / "real_world"
    )
    args = parser.parse_args(argv)
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    summary = output / "real_world_summary.json"
    command = [
        sys.executable,
        "-m",
        "plots.end_to_end.plot_real_world_frontier",
        "--input-dir",
        str(source),
        "--billing-duration-sec",
        "86400",
        "--fixed-cost-non-or",
        "1.5476333333333334",
        "--slo-ms",
        "3000",
        "--routewise-plot-alphas",
        "0",
        "0.25",
        "0.5",
        "0.75",
        "1",
        "--policies",
        *POLICIES,
        "--mean-ttft-out",
        str(output / "real_world_mean_ttft.pdf"),
        "--slo-out",
        str(output / "real_world_slo_frontier.pdf"),
        "--table-out",
        str(output / "real_world_rows.tex"),
        "--summary-out",
        str(summary),
        "--boxplot-out",
        str(output / "figure06a_ttft_distribution.pdf"),
        "--boxplot-policies",
        *POLICIES,
        "--provider-mix-out",
        str(output / "figure06b_provider_mix.pdf"),
        "--provider-mix-policies",
        *POLICIES,
        "--provider-latency-out",
        str(output / "figure07a_provider_latency.pdf"),
        "--provider-latency-values-json",
        str(ROOT / "plots/end_to_end/paper_minimax_provider_latency.json"),
        "--provider-pricing-out",
        str(output / "figure07b_provider_pricing.pdf"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "plots.front_page.make_front_page_teaser",
            "--real-summary-json",
            str(summary),
            "--out-dir",
            str(output),
            "--out-stem",
            "figure01",
            "--no-combined",
        ],
        cwd=ROOT,
        check=True,
    )
    check_summary(summary, source / "reference_summary.json")
    print(f"Figures and the recomputed summary: {output}")
    print(
        "Figure 7a redraws the author-reconstructed provider-latency snapshot; see data/real_eval_records/README.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
