"""Plot the MiniMax-M3 end-to-end simulation panels.

Each workload is simulated first, then plotted:

    # 30-day BurstGPT
    uv run python -m experiments.simulation.end_to_end \
        --scenario end_to_end_m3_rw6 --jobs 13 \
        --output-dir outputs/simulation/end_to_end_m3
    uv run python scripts/plot_simulation_m3.py --workload burstgpt30d

    # 7-day PROD export of 2026-08-25..08-31, same replay settings
    uv run python -m experiments.simulation.end_to_end \
        --scenario end_to_end_m3_rw6 --workload freeinference_20260825 \
        --prefix-cache-enabled --seed 42 --slo-ms 3000 --predictor bucket_mean \
        --jobs 13 --output-dir outputs/simulation/prod_20260825
    uv run python scripts/plot_simulation_m3.py --workload prod20260825

    # 7-day PROD released with the paper, with the Figure 8 replay settings
    uv run python -m experiments.simulation.end_to_end \
        --scenario end_to_end_m3_rw6 --workload freeinference \
        --prefix-cache-enabled --seed 42 --slo-ms 3000 --predictor bucket_mean \
        --jobs 13 --output-dir outputs/simulation/prod_m3
    uv run python scripts/plot_simulation_m3.py --workload prod
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("greedy_cost", "greedy_latency", "random")
LP_ONLY = tuple(f"ablation_lp_only_alpha{alpha}" for alpha in (0, 25, 50, 75, 100))
HEDGING = tuple(f"ablation_lp_hedging_alpha{alpha}" for alpha in (0, 25, 50, 75, 100))
# Every panel shows the deployed configuration, which hedges. Both sweeps on
# one frontier would collide: the shared plot labels a point by its alpha only,
# so the two curves repeat each label. The no-hedging sweep is in the P99
# panel, the table and the summary.
PANEL_POLICIES = (*BASELINES, *HEDGING)


@dataclass(frozen=True)
class Workload:
    input_dir: Path
    output_dir: Path
    prefix: str
    # Partial frontier label offsets in points; unlisted points keep the
    # shared defaults. The two workloads place the low-alpha points
    # differently, so each needs its own nudges.
    label_offsets: dict


WORKLOADS = {
    "burstgpt30d": Workload(
        input_dir=ROOT / "outputs" / "simulation" / "end_to_end_m3",
        output_dir=ROOT / "outputs" / "figures" / "simulation_m3",
        prefix="e2e_m3_",
        # alpha=0 sits just below and right of Greedy-cost.
        label_offsets={"routewise": {"0.0": [-6, -13]}},
    ),
    "prod20260825": Workload(
        input_dir=ROOT / "outputs" / "simulation" / "prod_20260825",
        output_dir=ROOT / "outputs" / "figures" / "simulation_m3_prod20260825",
        prefix="prod20260825_m3_",
        # alpha=0 is the leftmost point, so its default up-left label runs off
        # the axis and into Greedy-cost.
        label_offsets={"routewise": {"0.0": [-2, -13]}},
    ),
    "prod": Workload(
        input_dir=ROOT / "outputs" / "simulation" / "prod_m3",
        output_dir=ROOT / "outputs" / "figures" / "simulation_m3_prod",
        prefix="prod_m3_",
        # alpha=0 and alpha=0.25 sit close together at the cheap end, so their
        # labels are anchored on opposite sides instead of both centred.
        label_offsets={"routewise": {"0.0": [-8, -13], "0.25": [8, -11]}},
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="burstgpt30d")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    workload = WORKLOADS[args.workload]
    source = (args.input_dir or workload.input_dir).resolve()
    output = (args.output_dir or workload.output_dir).resolve()
    name = lambda stem: str(output / f"{workload.prefix}{stem}")  # noqa: E731
    command = [
        sys.executable,
        "-m",
        "plots.end_to_end.plot_simulation_frontier",
        "--summary-csv",
        str(source / "summary.csv"),
        "--histograms-json",
        str(source / "ttft_histograms.json"),
        "--frontier-out",
        name("cost_latency_frontier.pdf"),
        "--slo-out",
        name("slo_violations.pdf"),
        "--routewise-frontier-policies",
        *HEDGING,
        "--label-offsets",
        json.dumps(workload.label_offsets),
        "--tier-out",
        name("tier_mix.pdf"),
        "--tier-policies",
        *HEDGING,
        "--provider-latency-out",
        name("provider_latency.pdf"),
        "--provider-mix-out",
        name("provider_mix.pdf"),
        "--provider-mix-policies",
        *PANEL_POLICIES,
        "--boxplot-out",
        name("ttft_distribution.pdf"),
        "--boxplot-policies",
        *PANEL_POLICIES,
        "--p99-bar-out",
        name("hedging_p99.pdf"),
        "--cdf-out",
        name("ttft_cdf.pdf"),
        "--cdf-policies",
        *PANEL_POLICIES,
        "--table-out",
        name("rows.tex"),
        "--summary-out",
        name("summary.json"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"MiniMax-M3 {args.workload} end-to-end panels: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
