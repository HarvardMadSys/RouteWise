"""Replay the released PROD trace with the paper's simulator revision and plot Figure 8.

The bundled source is an unmodified git archive of the historical rwsim/ and
experiments/ trees. Later changes to concurrency cost accounting alter the
RouteWise points, so exact paper reproduction uses this explicit snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "99c5f3fd6504377fb57bd4edecef89e3865b7765"
SOURCE_SHA256 = "5e3299d7e56c1d8ae2421d0bb2dd61fff7d081fc6fdc1e7fea24af9372ce6dce"
POLICIES = (
    "ablation_lp_only_alpha0",
    "ablation_lp_only_alpha25",
    "ablation_lp_only_alpha50",
    "ablation_lp_only_alpha75",
    "ablation_lp_only_alpha100",
    "greedy_cost",
    "greedy_latency",
    "random",
)
METRICS = (
    "total_cost_usd",
    "mean_ttft_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "slo_violation_rate",
    "hedge_rate",
)


def normalize_policy_names(output: Path) -> None:
    """Adapt the old p/alpha spelling for today's plotter; leave numbers unchanged."""
    for name in ("summary.json", "ttft_histograms.json"):
        path = output / name
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            row["policy"] = row["policy"].replace("_only_p", "_only_alpha")
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    path = output / "summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["policy"] = row["policy"].replace("_only_p", "_only_alpha")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_summary(actual_path: Path, reference_path: Path) -> None:
    """Validate newly simulated values against the independently archived results."""
    with actual_path.open(newline="", encoding="utf-8") as handle:
        actual_rows = list(csv.DictReader(handle))
    with reference_path.open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    actual = {row["policy"]: row for row in actual_rows}
    reference = {row["policy"]: row for row in reference_rows}
    if len(actual_rows) != len(POLICIES) or len(reference_rows) != len(POLICIES):
        raise ValueError("expected eight simulator and reference rows")
    if actual.keys() != set(POLICIES) or actual.keys() != reference.keys():
        raise ValueError("missing or duplicate simulator/reference policies")
    for policy, row in actual.items():
        expected = reference[policy]
        if int(row["n_requests"]) != int(expected["n_requests"]):
            raise ValueError(f"{policy}: request count differs from the reference")
        for key in METRICS:
            if not math.isclose(float(row[key]), float(expected[key]), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{policy}: {key}={row[key]}, reference={expected[key]}")
        if json.loads(row["provider_mix"]) != json.loads(expected["provider_mix"]):
            raise ValueError(f"{policy}: provider request counts differ from the reference")
    print("PASS: all eight policies match the archived counts, provider mix, and metrics.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "figure8")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = args.output_dir.resolve()
    simulation, figures = output / "simulation", output / "figures"
    source = ROOT / "experiments/simulation/paper_figure8_source.tar.gz"
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("paper simulator source checksum mismatch")
    print(f"Replaying PROD using paper simulator commit {SOURCE_COMMIT}", flush=True)
    with tempfile.TemporaryDirectory(prefix="routewise-figure8-") as temporary:
        snapshot = Path(temporary)
        with tarfile.open(source, "r:gz") as archive:
            archive.extractall(snapshot, filter="data")
        (snapshot / "data").mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "data/freeinference.jsonl", snapshot / "data/freeinference.jsonl")
        command = [
            sys.executable,
            "-m",
            "experiments.simulation.end_to_end",
            "--scenario",
            "end_to_end_rw8",
            "--workload",
            "freeinference",
            "--prefix-cache-enabled",
            "--seed",
            "42",
            "--slo-ms",
            "3000",
            "--predictor",
            "bucket_mean",
            "--predictor-quantile",
            "q50",
            "--quota-plan",
            "chutes",
            "--quota-count",
            "1",
            "--concurrency-plan",
            "featherless_premium",
            "--concurrency-count",
            "1",
            "--model",
            "qwen3-235b",
            "--jobs",
            str(args.jobs),
            "--output-dir",
            str(simulation),
        ]
        for policy in POLICIES:
            command.extend(("--policy", policy.replace("_only_alpha", "_only_p")))
        subprocess.run(command, cwd=snapshot, check=True)
    normalize_policy_names(simulation)
    check_summary(simulation / "summary.csv", ROOT / "data/figure8_reference_summary.csv")
    (simulation / "source_revision.json").write_text(
        json.dumps(
            {
                "commit": SOURCE_COMMIT,
                "sha256": SOURCE_SHA256,
                "policy_name_mapping": "ablation_lp_only_pN -> ablation_lp_only_alphaN",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "plots.end_to_end.plot_simulation_frontier",
            "--summary-csv",
            str(simulation / "summary.csv"),
            "--histograms-json",
            str(simulation / "ttft_histograms.json"),
            "--frontier-out",
            str(figures / "figure08a_freeinference_mean_ttft.pdf"),
            "--slo-out",
            str(figures / "figure08b_slo_violations.pdf"),
            "--boxplot-out",
            str(figures / "figure08c_ttft_distribution.pdf"),
            "--boxplot-policies",
            *POLICIES,
            "--provider-mix-out",
            str(figures / "figure08d_provider_mix.pdf"),
            "--provider-mix-policies",
            *POLICIES,
            "--table-out",
            str(figures / "figure08_rows.tex"),
            "--summary-out",
            str(figures / "figure08_summary.json"),
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"Figure 8 panels and recomputed results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
