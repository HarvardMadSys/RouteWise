"""Plot the 30-day MiniMax-M3 end-to-end simulation panels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Baselines first, then the LP-only sweep, then the hedging sweep, matching the
# order experiments.simulation.end_to_end writes them in.
BASELINES = ("greedy_cost", "greedy_latency", "random")
LP_ONLY = tuple(f"ablation_lp_only_alpha{alpha}" for alpha in (0, 25, 50, 75, 100))
HEDGING = tuple(f"ablation_lp_hedging_alpha{alpha}" for alpha in (0, 25, 50, 75, 100))
# Every panel shows the deployed configuration, which hedges. Both sweeps on
# one frontier would collide: the shared plot labels a point by its alpha only,
# so the two curves repeat each label. The no-hedging sweep is in the P99 panel,
# the table and the summary.
FRONTIER_ROUTEWISE = HEDGING
# alpha=0 sits just below and right of Greedy-cost, so its default up-left
# label runs into that one. Unlisted points keep the shared defaults.
LABEL_OFFSETS = json.dumps({"routewise": {"0.0": [-6, -13]}})
# The per-policy panels stay readable with the baselines and the hedging sweep;
# the LP-only sweep is in the frontier, the table and the summary.
PANEL_POLICIES = (*BASELINES, *HEDGING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "outputs" / "simulation" / "end_to_end_m3"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "figures" / "simulation_m3"
    )
    args = parser.parse_args(argv)
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    command = [
        sys.executable,
        "-m",
        "plots.end_to_end.plot_simulation_frontier",
        "--summary-csv",
        str(source / "summary.csv"),
        "--histograms-json",
        str(source / "ttft_histograms.json"),
        "--frontier-out",
        str(output / "e2e_m3_cost_latency_frontier.pdf"),
        "--slo-out",
        str(output / "e2e_m3_slo_violations.pdf"),
        "--routewise-frontier-policies",
        *FRONTIER_ROUTEWISE,
        "--label-offsets",
        LABEL_OFFSETS,
        "--tier-out",
        str(output / "e2e_m3_tier_mix.pdf"),
        "--tier-policies",
        *HEDGING,
        "--provider-latency-out",
        str(output / "e2e_m3_provider_latency.pdf"),
        "--provider-mix-out",
        str(output / "e2e_m3_provider_mix.pdf"),
        "--provider-mix-policies",
        *PANEL_POLICIES,
        "--boxplot-out",
        str(output / "e2e_m3_ttft_distribution.pdf"),
        "--boxplot-policies",
        *PANEL_POLICIES,
        "--p99-bar-out",
        str(output / "e2e_m3_hedging_p99.pdf"),
        "--cdf-out",
        str(output / "e2e_m3_ttft_cdf.pdf"),
        "--cdf-policies",
        *PANEL_POLICIES,
        "--table-out",
        str(output / "e2e_m3_rows.tex"),
        "--summary-out",
        str(output / "e2e_m3_summary.json"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"30-day MiniMax-M3 end-to-end panels: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
