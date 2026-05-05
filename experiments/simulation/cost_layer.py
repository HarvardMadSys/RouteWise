"""Cost-layer simulator section."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.simulation.common import (
    COST_LAYER_P50_MS,
    COST_RATIO_PER_MILLION,
    DEFAULT_SEEDS,
    DEFAULT_WORKLOAD,
    OUTPUT_COST_MULTIPLIER,
    OUTPUT_DIR,
    P_SWEEP,
    make_api_provider,
    make_concurrency_provider,
    make_quota_provider,
    make_routewise_presets,
    make_tps_distribution,
    routewise_lp_policy_name,
    run_section,
)
from experiments.simulation.real_profiles import load_pooled_distribution
from rwsim.metrics import PerRequestRecord, Run, Status
from rwsim.world.capacity import ProviderTier
from rwsim.world.providers import TieredProvider
from rwsim.world.scenarios import ScenarioConfig

if TYPE_CHECKING:
    from rwsim.schemas import Request

SECTION_NAME = "cost-layer"
OFFLINE_POLICY = "offline"
REAL_WORLD_SCENARIO = "cost_layer_real_world"

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
    real_world = _make_real_world_api_cost_scenario()
    scenarios[real_world.name] = real_world
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
        OFFLINE_POLICY,
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
    section_runners = {OFFLINE_POLICY: run_offline_policy}
    known_policies = set(presets) | set(section_runners)
    unknown = [policy for policy in policies if policy not in known_policies]
    if unknown:
        known = ", ".join(sorted(known_policies))
        raise SystemExit(f"unknown cost-layer policy {unknown[0]!r}; known policies: {known}")

    rows = run_section(
        section_name=SECTION_NAME,
        scenarios=scenarios,
        policies=policies,
        presets=presets,
        seeds=tuple(args.seed) if args.seed else DEFAULT_SEEDS,
        section_runners=section_runners,
        workload_dataset=args.workload,
        duration_sec=args.duration_sec,
        max_requests=args.max_requests,
        output_dir=args.output_dir,
    )
    print(json.dumps({"section": SECTION_NAME, "rows": len(rows), "output_dir": str(args.output_dir)}))
    return 0


def run_offline_policy(
    scenario: ScenarioConfig,
    requests: list[Request],
    seed: int,
) -> Run:
    """Run the cost-layer offline baseline with full trace knowledge."""
    del seed
    assignments = _offline_assignments(scenario, requests)
    records = [
        _offline_record(
            scenario=scenario,
            request=request,
            provider=assignments[request.id],
        )
        for request in requests
    ]
    return Run(
        records=records,
        policy=OFFLINE_POLICY,
        scenario_name=scenario.name,
        source="simulation",
    )


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
            f"({family}, P50={COST_LAYER_P50_MS:.0f}ms), input cost ratio $1/$2/$4 "
            "and output cost ratio $5/$10/$20 per million tokens."
        ),
        providers=providers,
        arrival_process="trace",
        primary_slo_ms=2000.0,
    )


def _make_real_world_api_cost_scenario() -> ScenarioConfig:
    ttft_dist = load_pooled_distribution("rw8_pooled")
    providers = [
        TieredProvider(
            name="api_cheap",
            cost_per_token=COST_RATIO_PER_MILLION[0] / 1_000_000.0,
            input_cost_per_token=COST_RATIO_PER_MILLION[0] / 1_000_000.0,
            output_cost_per_token=(
                COST_RATIO_PER_MILLION[0] * OUTPUT_COST_MULTIPLIER / 1_000_000.0
            ),
            ttft_dist=ttft_dist,
            tps_dist=make_tps_distribution(),
            tier=ProviderTier.S_A,
        ),
        TieredProvider(
            name="api_mid",
            cost_per_token=COST_RATIO_PER_MILLION[1] / 1_000_000.0,
            input_cost_per_token=COST_RATIO_PER_MILLION[1] / 1_000_000.0,
            output_cost_per_token=(
                COST_RATIO_PER_MILLION[1] * OUTPUT_COST_MULTIPLIER / 1_000_000.0
            ),
            ttft_dist=ttft_dist,
            tps_dist=make_tps_distribution(),
            tier=ProviderTier.S_A,
        ),
        TieredProvider(
            name="api_expensive",
            cost_per_token=COST_RATIO_PER_MILLION[2] / 1_000_000.0,
            input_cost_per_token=COST_RATIO_PER_MILLION[2] / 1_000_000.0,
            output_cost_per_token=(
                COST_RATIO_PER_MILLION[2] * OUTPUT_COST_MULTIPLIER / 1_000_000.0
            ),
            ttft_dist=ttft_dist,
            tps_dist=make_tps_distribution(),
            tier=ProviderTier.S_A,
        ),
    ]
    return ScenarioConfig(
        name=REAL_WORLD_SCENARIO,
        description=(
            "Cost-layer real-world API scenario: identical pooled Qwen3/OpenRouter "
            "TTFT distribution (rw8_pooled), input cost ratio $1/$2/$4 and "
            "output cost ratio $5/$10/$20 per million tokens."
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


def _offline_assignments(
    scenario: ScenarioConfig,
    requests: list[Request],
) -> dict[int, TieredProvider]:
    providers = list(scenario.providers)
    api_provider = _cheapest_api_provider(providers)
    if api_provider is None:
        raise ValueError(f"offline baseline requires at least one API provider: {scenario.name}")

    quota_providers = [provider for provider in providers if provider.tier == ProviderTier.S_Q]
    concurrency_providers = [
        provider for provider in providers if provider.tier == ProviderTier.S_C
    ]

    assignments = {request.id: api_provider for request in requests}
    if quota_providers:
        assignments.update(_quota_assignments(requests, quota_providers, api_provider))
    if concurrency_providers:
        assignments.update(
            _concurrency_assignments(requests, concurrency_providers, api_provider)
        )
    return assignments


def _quota_assignments(
    requests: list[Request],
    quota_providers: list[TieredProvider],
    api_provider: TieredProvider,
) -> dict[int, TieredProvider]:
    windows: dict[int, list[Request]] = defaultdict(list)
    window_sec = float(quota_providers[0].quota.window_sec if quota_providers[0].quota else 86400.0)
    for request in requests:
        windows[int(request.timestamp // window_sec)].append(request)

    assignments: dict[int, TieredProvider] = {}
    total_quota = sum(provider.quota.size for provider in quota_providers if provider.quota is not None)
    provider_cycle = [
        provider
        for provider in quota_providers
        for _ in range(provider.quota.size if provider.quota is not None else 0)
    ]
    for window_requests in windows.values():
        ranked = sorted(
            window_requests,
            key=lambda request: (
                _api_cost(api_provider, request),
                -(request.timestamp),
                -request.id,
            ),
            reverse=True,
        )
        for index, request in enumerate(ranked[:total_quota]):
            assignments[request.id] = provider_cycle[index % len(provider_cycle)]
    return assignments


def _concurrency_assignments(
    requests: list[Request],
    concurrency_providers: list[TieredProvider],
    api_provider: TieredProvider,
) -> dict[int, TieredProvider]:
    slots: list[tuple[TieredProvider, list[tuple[float, float]]]] = [
        (provider, [])
        for provider in concurrency_providers
        for _ in range(provider.concurrency.limit if provider.concurrency is not None else 0)
    ]
    assignments: dict[int, TieredProvider] = {}
    ranked = sorted(
        requests,
        key=lambda request: (
            _api_cost(api_provider, request),
            -(request.timestamp),
            -request.id,
        ),
        reverse=True,
    )
    for request in ranked:
        start = float(request.timestamp)
        end = start + _offline_service_time_sec(concurrency_providers[0], request)
        for provider, intervals in slots:
            if all(end <= lo or start >= hi for lo, hi in intervals):
                intervals.append((start, end))
                assignments[request.id] = provider
                break
    return assignments


def _offline_record(
    *,
    scenario: ScenarioConfig,
    request: Request,
    provider: TieredProvider,
) -> PerRequestRecord:
    ttft_ms = provider.true_p50_ms(float(request.timestamp))
    cost_usd = provider.marginal_cost_for_request(request, float(request.timestamp))
    return PerRequestRecord(
        request_id=str(request.id),
        elapsed_sec=float(request.timestamp),
        policy=OFFLINE_POLICY,
        prompt_tokens=int(request.request_tokens),
        completion_tokens_budget=(
            request.estimated_response_tokens
            if request.estimated_response_tokens is not None
            else request.response_tokens
        ),
        completion_tokens_actual=request.response_tokens,
        primary_provider=provider.name,
        primary_tier=provider.tier.value,
        final_provider=provider.name,
        final_tier=provider.tier.value,
        ttft_ms=ttft_ms,
        primary_local_ttft_ms=ttft_ms,
        slo_ms=scenario.primary_slo_ms,
        slo_violated=ttft_ms > scenario.primary_slo_ms,
        total_cost_usd=cost_usd,
        primary_cost_usd=cost_usd,
        status=Status.SUCCESS,
        metadata={"offline_cost_baseline": True},
    )


def _cheapest_api_provider(providers: list[TieredProvider]) -> TieredProvider | None:
    api_providers = [provider for provider in providers if provider.tier == ProviderTier.S_A]
    if not api_providers:
        return None
    return min(
        api_providers,
        key=lambda provider: (
            provider.effective_input_cost_per_token,
            provider.effective_output_cost_per_token,
            provider.name,
        ),
    )


def _api_cost(provider: TieredProvider, request: Request) -> float:
    return provider.marginal_cost_for_request(request, float(request.timestamp))


def _offline_service_time_sec(provider: TieredProvider, request: Request) -> float:
    ttft_ms = provider.true_p50_ms(float(request.timestamp))
    tps = max(provider.tps_dist.p50(), 1.0)
    response_tokens = request.response_tokens or 1
    return (ttft_ms + (response_tokens / tps) * 1000.0) / 1000.0


__all__ = [
    "OFFLINE_POLICY",
    "SECTION_NAME",
    "list_scenarios",
    "make_scenarios",
    "policies_for_section",
    "run_offline_policy",
]


if __name__ == "__main__":
    raise SystemExit(main())
