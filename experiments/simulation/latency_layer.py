"""§2.1 latency-layer simulator section.

Same cost across providers, different latency profiles. Three same-family
providers (fast / medium / slow) with configured mean TTFT anchors =
100 / 300 / 1000 ms and a target Q10-Q90 band coverage on the
(fast, medium) anchor pair (see
:mod:`experiments.simulation.latency_overlap`).

Scenario grid: 3 synthetic families x 2 overlap labels, plus one RW3
real-world scenario = 7 scenarios.

Policies: random, greedy_latency, plus one LP-only RouteWise setting at the
section default p=0.75. No hedging in §2.1; that lives in §2.2
(``hedging.py``).

Outputs:
- ``summary.json`` — full row dict including overlap metadata
- ``summary.csv`` — latency-specific columns including target / realised
  band coverage for the three pairwise comparisons
- ``metadata.json``, ``ttft_histograms.json`` — same as cost-layer
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.simulation.common import (
    DEFAULT_OUTPUT_PREDICTOR,
    DEFAULT_SEEDS,
    DEFAULT_WORKLOAD,
    OUTPUT_DIR,
    WORKLOAD_CHOICES,
    SectionCell,
    SectionCellResult,
    load_workload,
    make_routewise_presets,
    make_tps_distribution,
    routewise_lp_policy_name,
    run_policy,
    run_section,
    write_json,
)
from experiments.simulation.latency_factory import SYNTHETIC_LATENCY_VERSION
from experiments.simulation.latency_overlap import (
    LATENCY_FAMILIES,
    LATENCY_LAYER_MEAN_MS,
    OVERLAP_TARGETS,
    PROVIDER_NAMES,
    SYNTHETIC_FAMILIES,
    OverlapSpec,
    build_distributions,
    summarise_realised_overlap,
    verify_calibration,
)
from rwsim.const import DEFAULT_PRIMARY_SLO_MS
from rwsim.world.capacity import ProviderTier
from rwsim.world.providers import TieredProvider
from rwsim.world.scenarios import ScenarioConfig

SECTION_NAME = "latency-layer"

# Same cost for every provider — §2.1 isolates the latency dimension.
LATENCY_LAYER_INPUT_COST_PER_M_TOKENS: float = 1.0
LATENCY_LAYER_OUTPUT_COST_PER_M_TOKENS: float = 5.0

# Public scenario tag used by metadata / csv groupings.
PUBLIC_SCENARIO_TAG: str = "latency_layer"
REAL_WORLD_SCENARIO_NAME: str = "latency_layer_real_world"
DEFAULT_ROUTEWISE_P: float = 0.75


def _scenario_name(family: str, overlap_label: str | None = None) -> str:
    if family == "real_world":
        return REAL_WORLD_SCENARIO_NAME
    if overlap_label is None:
        raise ValueError(f"synthetic family {family!r} requires overlap_label")
    return f"latency_layer_{family}_{overlap_label}"


def list_scenarios() -> tuple[str, ...]:
    """Return all §2.1 scenario names (6 synthetic + 1 real-world)."""
    synthetic = tuple(
        _scenario_name(family, label)
        for family in SYNTHETIC_FAMILIES
        for label in OVERLAP_TARGETS
    )
    return (*synthetic, REAL_WORLD_SCENARIO_NAME)


def make_scenarios() -> dict[str, ScenarioConfig]:
    """Build all §2.1 scenarios keyed by scenario name."""
    return {name: make_scenario(name) for name in list_scenarios()}


def make_scenario(name: str) -> ScenarioConfig:
    """Build one §2.1 scenario by name."""
    parsed = _parse_scenario_name(name)
    if parsed is None:
        known = ", ".join(list_scenarios())
        raise ValueError(f"unknown latency-layer scenario {name!r}; known: {known}")
    family, overlap_label = parsed
    return _make_latency_layer_scenario(family, overlap_label)


def _parse_scenario_name(name: str) -> tuple[str, str | None] | None:
    if name == REAL_WORLD_SCENARIO_NAME:
        return ("real_world", None)
    prefix = "latency_layer_"
    if not name.startswith(prefix):
        return None
    body = name.removeprefix(prefix)
    # overlap_label suffix is one of OVERLAP_TARGETS keys; greedy match the longest.
    for label in sorted(OVERLAP_TARGETS, key=len, reverse=True):
        suffix = "_" + label
        if body.endswith(suffix):
            family = body.removesuffix(suffix)
            if family in SYNTHETIC_FAMILIES:
                return family, label
    return None


def _make_latency_layer_scenario(
    family: str,
    overlap_label: str | None,
) -> ScenarioConfig:
    """Construct three same-cost providers with calibrated overlap."""
    spec = OverlapSpec(
        family=family,
        overlap_label=overlap_label,
        mean_anchors_ms=LATENCY_LAYER_MEAN_MS,
    )
    if family in SYNTHETIC_FAMILIES:
        # Fail loudly if a refactor ever breaks the closed-form derivation.
        verify_calibration(spec)
    distributions = build_distributions(spec)
    realised = summarise_realised_overlap(spec, distributions)
    latency_metadata: dict[str, Any] = {}
    if family in SYNTHETIC_FAMILIES:
        latency_metadata = {
            "latency_generation_version": SYNTHETIC_LATENCY_VERSION,
            "latency_anchor_kind": "mean",
            "latency_anchor_ms": list(LATENCY_LAYER_MEAN_MS),
            "latency_distribution_mean_ms": [dist.mean() for dist in distributions],
            "latency_distribution_p50_ms": [dist.p50() for dist in distributions],
        }
    providers = [
        TieredProvider(
            name=provider_name,
            cost_per_token=LATENCY_LAYER_INPUT_COST_PER_M_TOKENS / 1_000_000.0,
            input_cost_per_token=LATENCY_LAYER_INPUT_COST_PER_M_TOKENS / 1_000_000.0,
            output_cost_per_token=LATENCY_LAYER_OUTPUT_COST_PER_M_TOKENS
            / 1_000_000.0,
            ttft_dist=ttft_dist,
            tps_dist=make_tps_distribution(),
            tier=ProviderTier.S_A,
        )
        for provider_name, ttft_dist in zip(PROVIDER_NAMES, distributions, strict=True)
    ]
    metadata: dict[str, Any] = {
        "public_scenario": PUBLIC_SCENARIO_TAG,
        "artifact_label": _scenario_name(family, overlap_label),
        "latency_family": family,
        "overlap_label": overlap_label,
        "overlap_construction_metric": "q10_q90_directional_band_coverage",
        "target_anchor_pair": "fast_medium" if overlap_label is not None else None,
        "overlap_metric_source": "configured_distribution",
        "mean_anchors_ms": list(LATENCY_LAYER_MEAN_MS),
        "provider_names": list(PROVIDER_NAMES),
        "input_cost_per_million_tokens": LATENCY_LAYER_INPUT_COST_PER_M_TOKENS,
        "output_cost_per_million_tokens": LATENCY_LAYER_OUTPUT_COST_PER_M_TOKENS,
        **latency_metadata,
        **realised.as_dict(),
    }
    if spec.target_coverage is None:
        target_text = "no synthetic target"
    else:
        target_text = f"target_band_coverage(fast, medium)={spec.target_coverage:.2f}"
    description = (
        f"§2.1 latency-layer: family={family} overlap={overlap_label}, "
        f"mean={LATENCY_LAYER_MEAN_MS} ms, {target_text}, "
        f"realised={realised.realised_band_coverage_fast_medium:.4f}. "
        "Cost is identical across providers; routing differentiates on latency only."
    )
    return ScenarioConfig(
        name=_scenario_name(family, overlap_label),
        description=description,
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=DEFAULT_PRIMARY_SLO_MS,
        metadata=metadata,
    )


def policies_for_section(p_value: float = DEFAULT_ROUTEWISE_P) -> tuple[str, ...]:
    """Return policies relevant to §2.1 (no hedging, no offline oracle)."""
    return (
        "random",
        "greedy_latency",
        routewise_lp_policy_name(p_value),
    )


def run_latency_layer_cell(
    cell: SectionCell,
    presets: dict[str, dict[str, Any]],
    workload_dataset: str,
    duration_sec: float | None,
    max_requests: int | None,
    retain_records: bool,
) -> SectionCellResult:
    """Run one §2.1 simulation cell in a worker process."""
    scenario = make_scenario(cell.scenario_name)
    requests = load_workload(
        dataset=workload_dataset,
        duration_sec=duration_sec,
        max_requests=max_requests,
    )
    run = run_policy(
        scenario,
        requests,
        cell.policy,
        presets=presets,
        seed=cell.seed,
        retain_records=retain_records,
    )
    return SectionCellResult(
        scenario_name=cell.scenario_name,
        policy=cell.policy,
        seed=cell.seed,
        run=run,
    )


# ---------------------------------------------------------------------------
# Latency-specific summary CSV (extends common.write_summary_csv)
# ---------------------------------------------------------------------------

_LATENCY_CSV_FIELDNAMES: tuple[str, ...] = (
    "scenario",
    "public_scenario",
    "artifact_label",
    "latency_family",
    "latency_generation_version",
    "latency_anchor_kind",
    "latency_anchor_ms",
    "latency_distribution_mean_ms",
    "latency_distribution_p50_ms",
    "overlap_label",
    "policy",
    "seeds",
    "n_requests",
    "mean_ttft_ms",
    "p10_ms",
    "p25_ms",
    "p50_ms",
    "p75_ms",
    "p90_ms",
    "p99_ms",
    "mean_api_cost_usd",
    "mean_total_cost_usd",
    "api_cost_usd",
    "api_cost_usd_per_run",
    "total_cost_usd",
    "total_cost_usd_per_run",
    "trace_paper_grade",
    "trace_days",
    "slo_violation_rate",
    "hedge_rate",
    "provider_mix",
    "overlap_construction_metric",
    "target_anchor_pair",
    "overlap_metric_source",
    "target_band_coverage_fast_medium",
    "realised_band_coverage_fast_medium",
    "realised_band_coverage_medium_slow",
    "realised_band_coverage_fast_slow",
    "normal_clip_fraction_fast",
    "normal_clip_fraction_medium",
    "normal_clip_fraction_slow",
    "percentile_source",
    "histogram_bins",
)


def _enrich_rows_with_overlap_metadata(
    rows: list[dict[str, Any]],
    scenarios: dict[str, ScenarioConfig],
) -> list[dict[str, Any]]:
    """Fold scenario.metadata overlap fields into each summary row."""
    enriched: list[dict[str, Any]] = []
    for row in rows:
        scenario = scenarios.get(row["scenario"])
        meta = dict(getattr(scenario, "metadata", {}) or {})
        merged = dict(row)
        for key in (
            "latency_family",
            "latency_generation_version",
            "latency_anchor_kind",
            "latency_anchor_ms",
            "latency_distribution_mean_ms",
            "latency_distribution_p50_ms",
            "overlap_label",
            "overlap_construction_metric",
            "target_anchor_pair",
            "overlap_metric_source",
            "target_band_coverage_fast_medium",
            "realised_band_coverage_fast_medium",
            "realised_band_coverage_medium_slow",
            "realised_band_coverage_fast_slow",
            "normal_clip_fraction_fast",
            "normal_clip_fraction_medium",
            "normal_clip_fraction_slow",
        ):
            if key in meta and merged.get(key) is None:
                merged[key] = meta[key]
        enriched.append(merged)
    return enriched


def _write_latency_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write §2.1-specific csv view of the section summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_LATENCY_CSV_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], sort_keys=True)
                        if isinstance(row.get(key), (dict, list))
                        else row.get(key)
                    )
                    for key in _LATENCY_CSV_FIELDNAMES
                }
            )


def main(argv: list[str] | None = None) -> int:
    """Run the §2.1 latency-layer simulator section."""
    parser = argparse.ArgumentParser(
        prog="routewise simulator latency-layer",
        description=__doc__,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=list(list_scenarios()),
        help="Scenario to run. Repeat to run multiple. Defaults to all 7.",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=list(LATENCY_FAMILIES),
        help=(
            "Latency family filter. Repeat to allow multiple. "
            "Synthetic families run selected overlap labels; real_world is one scenario."
        ),
    )
    parser.add_argument(
        "--overlap",
        action="append",
        choices=list(OVERLAP_TARGETS),
        help="Overlap label filter. Repeat to allow multiple.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        help="Policy to run. Repeat to run multiple. Defaults to this section's policy set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help=f"Seed to run. Repeat to run multiple. Defaults to {DEFAULT_SEEDS}.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=DEFAULT_ROUTEWISE_P,
        dest="p_value",
        help=f"Single RouteWise p value for LP-only policy. Defaults to {DEFAULT_ROUTEWISE_P}.",
    )
    parser.add_argument(
        "--workload",
        default=DEFAULT_WORKLOAD,
        choices=WORKLOAD_CHOICES,
        help="Trace workload to replay.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        help="Optional trace truncation for smoke runs.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        help="Optional request-count truncation for smoke runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "latency_layer",
        help="Directory for metadata.json, summary.json, and summary.csv.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel scenario-policy-seed cells to run. Defaults to 1.",
    )
    parser.add_argument(
        "--predictor",
        default=DEFAULT_OUTPUT_PREDICTOR,
        help=(
            "Optional output-length predictor for RouteWise S_A LP cost. Defaults "
            f"to {DEFAULT_OUTPUT_PREDICTOR}. Examples: none, oracle, bucket_mean, "
            "constant_mean, fixed:<value>."
        ),
    )

    args = parser.parse_args(argv)
    p_values = (float(args.p_value),)

    selected_scenarios = _select_scenarios(
        explicit_scenarios=tuple(args.scenario) if args.scenario else None,
        families=tuple(args.family) if args.family else None,
        overlaps=tuple(args.overlap) if args.overlap else None,
    )
    scenarios = {name: make_scenario(name) for name in selected_scenarios}

    presets = make_routewise_presets(
        p_values=p_values,
        include_hedging=False,
        output_predictor=args.predictor,
    )
    policies = tuple(args.policy) if args.policy else policies_for_section(args.p_value)
    unknown = [policy for policy in policies if policy not in presets]
    if unknown:
        known = ", ".join(sorted(presets))
        raise SystemExit(
            f"unknown latency-layer policy {unknown[0]!r}; known policies: {known}"
        )

    rows = run_section(
        section_name=SECTION_NAME,
        scenarios=scenarios,
        policies=policies,
        presets=presets,
        seeds=tuple(args.seed) if args.seed else DEFAULT_SEEDS,
        section_runners=None,
        workload_dataset=args.workload,
        duration_sec=args.duration_sec,
        max_requests=args.max_requests,
        output_dir=args.output_dir,
        jobs=args.jobs,
        parallel_cell_runner=run_latency_layer_cell,
    )
    enriched_rows = _enrich_rows_with_overlap_metadata(rows, scenarios)
    write_json(args.output_dir / "summary.json", enriched_rows)
    _write_latency_summary_csv(args.output_dir / "summary.csv", enriched_rows)
    print(
        json.dumps(
            {
                "section": SECTION_NAME,
                "rows": len(enriched_rows),
                "output_dir": str(args.output_dir),
            }
        )
    )
    return 0


def _select_scenarios(
    *,
    explicit_scenarios: tuple[str, ...] | None,
    families: tuple[str, ...] | None,
    overlaps: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Resolve the requested scenario set from (--scenario, --family, --overlap)."""
    if explicit_scenarios:
        return explicit_scenarios
    family_set = set(families) if families else set(LATENCY_FAMILIES)
    overlap_set = set(overlaps) if overlaps else set(OVERLAP_TARGETS)
    selected = [
        _scenario_name(family, label)
        for family in SYNTHETIC_FAMILIES
        if family in family_set
        for label in OVERLAP_TARGETS
        if label in overlap_set
    ]
    if "real_world" in family_set and overlaps is None:
        selected.append(REAL_WORLD_SCENARIO_NAME)
    return tuple(selected)


__all__ = [
    "PUBLIC_SCENARIO_TAG",
    "REAL_WORLD_SCENARIO_NAME",
    "SECTION_NAME",
    "list_scenarios",
    "main",
    "make_scenario",
    "make_scenarios",
    "policies_for_section",
    "run_latency_layer_cell",
]
