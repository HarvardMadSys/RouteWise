"""Recompute the MiniMax-M3 real-world rerun and plot its six panels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.reproduce_real_world import check_summary

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
    "single_OR_Together",
)
# Prorated MiniMax Plus ($20) plus Featherless Premium ($25) over 24 h of a
# 30-day month, the value recorded with the run; see data/real_eval_records_m3/README.md.
FIXED_COST_NON_OR = "1.5"
# Label offsets (points) for the mean-TTFT frontier, where five policies share
# one cost level around 1.6x and the paper's default placements collide.
LABEL_OFFSETS = json.dumps(
    {
        "routewise": {
            "0.0": [-3, -12],
            "0.25": [-3, -12],
            "0.5": [-3, -11],
            "0.75": [-3, -15],
            "1.0": [4, -12],
        },
        "baselines": {
            "single_OR_Together": [-4, 14],
            "or_sort_latency": [6, 6],
            "greedy_latency": [6, -8],
        },
    }
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "real_eval_records_m3")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "figures" / "real_world_m3"
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
        FIXED_COST_NON_OR,
        "--slo-ms",
        "3000",
        "--label-offsets",
        LABEL_OFFSETS,
        "--emphasize-routewise",
        "--frontier-x-max",
        "1.95",
        "--drop-failed-mix",
        "--provider-latency-p99",
        "--provider-latency-xmax",
        "7",
        "--routewise-plot-alphas",
        "0",
        "0.25",
        "0.5",
        "0.75",
        "1",
        "--policies",
        *POLICIES,
        "--mean-ttft-out",
        str(output / "figure01_ttft.pdf"),
        "--slo-out",
        str(output / "figure01_slo.pdf"),
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
        "--provider-pricing-out",
        str(output / "figure07b_provider_pricing.pdf"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    check_summary(summary, source / "reference_summary.json")
    print(f"Figures and the recomputed summary: {output}")
    print("Figure 7a aggregates provider TTFT from the request records of this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
