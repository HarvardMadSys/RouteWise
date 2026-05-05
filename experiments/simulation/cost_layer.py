"""Cost-layer simulator section."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.simulation.common import (
    COST_LAYER_P50_MS,
    COST_RATIO_PER_MILLION,
    DEFAULT_SEEDS,
    DEFAULT_WORKLOAD,
    OUTPUT_DIR,
    P_SWEEP,
    make_api_provider,
    make_concurrency_provider,
    make_quota_provider,
    make_routewise_presets,
    routewise_lp_policy_name,
    run_section,
)
from rwsim.world.scenarios import ScenarioConfig

SECTION_NAME = "cost-layer"

_SYNTHETIC_FAMILIES = ("uniform", "normal", "heavy_tail")
_QUOTA_SIZE_PER_PROVIDER = 2_000
_CONCURRENCY_LIMIT_PER_PROVIDER = 8


def list_scenarios() -> tuple[str, ...]:
    """Return all cost-layer scenario names."""
    return tuple(make_scenarios())


def make_scenarios() -> dict[str, ScenarioConfig]:
    """Build cost-layer scenarios keyed by scenario name."""
    scenarios: dict[str, ScenarioConfig] = {}
    for family in _SYNTHETIC_FAMILIES:
        scenario = _make_api_cost_scenario(family)
        scenarios[scenario.name] = scenario
    for count in range(1, 5):
        quota = _make_quota_scenario(count)
        scenarios[quota.name] = quota
    for count in range(1, 5):
        concurrency = _make_concurrency_scenario(count)
        scenarios[concurrency.name] = concurrency
    return scenarios


def policies_for_section(
    p_values: tuple[float, ...] = P_SWEEP,
) -> tuple[str, ...]:
    """Return policies relevant to cost-layer experiments."""
    return (
        "greedy_cost",
        "random",
        *(routewise_lp_policy_name(value) for value in p_values),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the cost-layer simulator section."""
    parser = argparse.ArgumentParser(
        prog="routewise simulator cost-layer",
        description=__doc__,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=list_scenarios(),
        help="Scenario to run. Repeat to run multiple. Defaults to all.",
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
        action="append",
        dest="p_values",
        help=f"RouteWise p value. Repeat to sweep. Defaults to {P_SWEEP}.",
    )
    parser.add_argument(
        "--workload",
        default=DEFAULT_WORKLOAD,
        choices=("sharegpt_burstgpt", "burstgpt"),
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
        default=OUTPUT_DIR / "cost_layer",
        help="Directory for metadata.json, summary.json, and summary.csv.",
    )

    args = parser.parse_args(argv)
    p_values = tuple(args.p_values) if args.p_values else P_SWEEP
    scenarios = make_scenarios()
    if args.scenario:
        scenarios = {name: scenarios[name] for name in args.scenario}

    presets = make_routewise_presets(p_values=p_values, include_hedging=False)
    policies = tuple(args.policy) if args.policy else policies_for_section(p_values)
    unknown = [policy for policy in policies if policy not in presets]
    if unknown:
        known = ", ".join(sorted(presets))
        raise SystemExit(f"unknown cost-layer policy {unknown[0]!r}; known policies: {known}")

    rows = run_section(
        section_name=SECTION_NAME,
        scenarios=scenarios,
        policies=policies,
        presets=presets,
        seeds=tuple(args.seed) if args.seed else DEFAULT_SEEDS,
        workload_dataset=args.workload,
        duration_sec=args.duration_sec,
        max_requests=args.max_requests,
        output_dir=args.output_dir,
    )
    print(json.dumps({"section": SECTION_NAME, "rows": len(rows), "output_dir": str(args.output_dir)}))
    return 0


def _make_api_cost_scenario(family: str) -> ScenarioConfig:
    providers = [
        make_api_provider(
            "api_cheap",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[0],
            latency_family=family,
        ),
        make_api_provider(
            "api_mid",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[1],
            latency_family=family,
        ),
        make_api_provider(
            "api_expensive",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[2],
            latency_family=family,
        ),
    ]
    return ScenarioConfig(
        name=f"cost_layer_{family}",
        description=(
            "Cost-layer on-demand API scenario: identical TTFT distribution "
            f"({family}, P50={COST_LAYER_P50_MS:.0f}ms), cost ratio $1/$2/$4 per million tokens."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=2000.0,
    )


def _make_quota_scenario(count: int) -> ScenarioConfig:
    providers = [
        make_quota_provider(
            f"quota_{idx + 1}",
            quota_size=_QUOTA_SIZE_PER_PROVIDER,
        )
        for idx in range(count)
    ]
    api_count = 2 if count == 1 else 1
    providers.extend(
        make_api_provider(
            f"api_fallback_{idx + 1}",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[-1],
            latency_family="heavy_tail",
        )
        for idx in range(api_count)
    )
    return ScenarioConfig(
        name=f"cost_layer_quota_q{count}",
        description=(
            f"Cost-layer quota scenario: {count} quota provider(s), "
            f"{api_count} on-demand fallback provider(s), LogNormal P50={COST_LAYER_P50_MS:.0f}ms."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=2000.0,
    )


def _make_concurrency_scenario(count: int) -> ScenarioConfig:
    providers = [
        make_concurrency_provider(
            f"concurrency_{idx + 1}",
            concurrency_limit=_CONCURRENCY_LIMIT_PER_PROVIDER,
        )
        for idx in range(count)
    ]
    api_count = 2 if count == 1 else 1
    providers.extend(
        make_api_provider(
            f"api_fallback_{idx + 1}",
            cost_per_million_tokens=COST_RATIO_PER_MILLION[-1],
            latency_family="heavy_tail",
        )
        for idx in range(api_count)
    )
    return ScenarioConfig(
        name=f"cost_layer_concurrency_c{count}",
        description=(
            f"Cost-layer concurrency scenario: {count} concurrency provider(s), "
            f"{api_count} on-demand fallback provider(s), LogNormal P50={COST_LAYER_P50_MS:.0f}ms."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=2000.0,
    )


__all__ = [
    "SECTION_NAME",
    "list_scenarios",
    "make_scenarios",
    "policies_for_section",
]


if __name__ == "__main__":
    raise SystemExit(main())
