"""Unit tests for the hedging timing/backup-selection ablation."""

from __future__ import annotations

import copy
import csv
import json

from experiments.ablations.hedging import harness
from experiments.ablations.hedging.policy import HedgingAblationPolicy
from experiments.ablations.hedging.presets import (
    DEFAULT_P_VALUES,
    ablation_policy_name,
    make_ablation_presets,
    parse_ablation_policy_name,
    production_baseline_policy_name,
)
from routewise_cli.main import ABLATION_COMMANDS, main as routewise_main
from rwsim.engine.simulator import Simulator
from rwsim.engine.state import SimulationState
from rwsim.policies.hedging import BackupCandidate
from rwsim.policies.routewise import RouteWisePolicy
from rwsim.schemas import Request, RoutingDecision
from rwsim.world.capacity import ProviderTier
from rwsim.world.distributions import Uniform
from rwsim.world.providers import TieredProvider
from rwsim.world.scenarios import ScenarioConfig


def test_presets_define_core_hedging_ablation_grid() -> None:
    presets = make_ablation_presets()

    assert tuple(presets) == (
        "hedging__dispatch=latest_safe__backup=probability__p75",
        "hedging__dispatch=earliest_safe__backup=probability__p75",
        "hedging__dispatch=latest_safe__backup=random_feasible__p75",
    )
    assert production_baseline_policy_name() == (
        "hedging__dispatch=latest_safe__backup=probability__p75"
    )
    assert parse_ablation_policy_name(
        "hedging__dispatch=earliest_safe__backup=random_feasible__p25"
    ) == ("earliest_safe", "random_feasible", 0.25)
    assert presets[production_baseline_policy_name()]["params"]["backup_selection"] == (
        "probability"
    )
    assert presets[production_baseline_policy_name()]["params"]["latency_profile_mode"] == (
        "configured"
    )


def test_latest_probability_ablation_matches_production_routewise_tick() -> None:
    providers = _primary_with_safe_and_slow_backups()
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = _request()
    decision = RoutingDecision(primary_provider="primary", hedge_checkpoints=(0.5,))
    state.now = 0.5
    production = RouteWisePolicy(
        hedging="probability_target",
        explorer=False,
        p=0.75,
        seed=7,
        slo_ms=2000.0,
        cost_envelope=(1e-6, 1e-3),
    )
    ablation = HedgingAblationPolicy(
        dispatch_timing="latest_safe",
        backup_selection="probability",
        p=0.75,
        seed=7,
        slo_ms=2000.0,
        cost_envelope=(1e-6, 1e-3),
    )

    production_dispatch = production.tick(request, decision, 0.5, state)
    ablation_dispatch = ablation.tick(request, decision, 0.5, state)

    assert production_dispatch == ablation_dispatch
    assert ablation_dispatch is not None
    assert ablation_dispatch.backup_provider == "safe"


def test_latest_probability_ablation_matches_production_simulator_path() -> None:
    scenario = ScenarioConfig(
        name="hedging_equivalence",
        description="small deterministic hedging equivalence scenario",
        providers=[
            _provider("cheap_slow", 3000.0, 4000.0, cost=1e-6),
            _provider("fast_expensive", 100.0, 200.0, cost=4e-6),
        ],
        primary_slo_ms=2000.0,
    )
    requests = [
        Request(
            id=index,
            timestamp=float(index),
            request_tokens=100,
            response_tokens=50,
            total_tokens=150,
        )
        for index in range(1, 12)
    ]
    production = RouteWisePolicy(
        hedging="probability_target",
        explorer=False,
        p=0.75,
        seed=11,
        slo_ms=2000.0,
        cost_envelope=(1e-6, 1e-3),
    )
    ablation = HedgingAblationPolicy(
        dispatch_timing="latest_safe",
        backup_selection="probability",
        p=0.75,
        seed=11,
        slo_ms=2000.0,
        cost_envelope=(1e-6, 1e-3),
    )

    production_run = Simulator(scenario=scenario, seed=19).run(
        requests,
        production,
        policy_name="routewise",
    )
    ablation_run = Simulator(scenario=scenario, seed=19).run(
        requests,
        ablation,
        policy_name="routewise",
    )

    assert production_run.records == ablation_run.records


def test_earliest_safe_dispatches_when_latest_safe_waits() -> None:
    providers = _primary_with_safe_and_slow_backups()
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = _request()
    decision = RoutingDecision(primary_provider="primary", hedge_checkpoints=(0.5, 1.0))
    latest = HedgingAblationPolicy(
        dispatch_timing="latest_safe",
        backup_selection="probability",
        p=0.75,
        seed=7,
        slo_ms=2000.0,
        cost_envelope=(1e-6, 1e-3),
    )
    earliest = HedgingAblationPolicy(
        dispatch_timing="earliest_safe",
        backup_selection="probability",
        p=0.75,
        seed=7,
        slo_ms=2000.0,
        cost_envelope=(1e-6, 1e-3),
    )
    state.now = 0.5

    assert latest.tick(request, decision, 0.5, state) is None
    dispatch = earliest.tick(request, decision, 0.5, state)

    assert dispatch is not None
    assert dispatch.backup_provider == "safe"


def test_random_feasible_backup_selection_only_draws_from_feasible_non_primary() -> None:
    providers = [
        _provider("primary", 3000.0, 4000.0, cost=1e-6),
        _provider("too_slow", 3000.0, 4000.0, cost=1e-6),
        _provider("safe_one", 100.0, 200.0, cost=1e-6),
        _provider("safe_two", 100.0, 200.0, cost=2e-6),
    ]
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = _request()
    decision = RoutingDecision(primary_provider="primary", hedge_checkpoints=(0.5,))
    policy = HedgingAblationPolicy(
        dispatch_timing="earliest_safe",
        backup_selection="random_feasible",
        p=0.75,
        seed=3,
        slo_ms=2000.0,
        cost_envelope=(1e-6, 1e-3),
    )
    state.now = 0.5

    dispatch = policy.tick(request, decision, 0.5, state)

    assert dispatch is not None
    assert dispatch.backup_provider in {"safe_one", "safe_two"}
    assert dispatch.backup_provider != "primary"
    assert dispatch.backup_provider != "too_slow"


def test_random_feasible_backup_selection_does_not_advance_route_rng() -> None:
    providers = [
        _provider("cheap_slow", 500.0, 1500.0, cost=1e-6),
        _provider("fast_expensive", 50.0, 150.0, cost=4e-6),
    ]
    state = SimulationState.from_providers({provider.name: provider for provider in providers})
    request = _request()
    probability = HedgingAblationPolicy(
        dispatch_timing="latest_safe",
        backup_selection="probability",
        p=0.75,
        seed=3,
        slo_ms=2000.0,
        cost_envelope=(1e-6, 1e-3),
    )
    random_feasible = HedgingAblationPolicy(
        dispatch_timing="latest_safe",
        backup_selection="random_feasible",
        p=0.75,
        seed=3,
        slo_ms=2000.0,
        cost_envelope=(1e-6, 1e-3),
    )
    candidates = [
        BackupCandidate(
            provider=providers[0],
            success_probability=0.995,
            marginal_cost=1e-6,
            true_mean_ms=100.0,
        ),
        BackupCandidate(
            provider=providers[1],
            success_probability=0.996,
            marginal_cost=2e-6,
            true_mean_ms=100.0,
        ),
    ]
    before = copy.deepcopy(random_feasible.rng.bit_generator.state)

    assert random_feasible._select_backup_candidate(candidates) is not None

    assert random_feasible.rng.bit_generator.state == before
    assert probability.route(request, state) == random_feasible.route(request, state)


def test_harness_cli_writes_policy_metadata_and_production_deltas(tmp_path) -> None:
    output_dir = tmp_path / "hedging-ablation"

    assert routewise_main(
        [
            "ablation",
            "hedging",
            "--scenario",
            "hedging_heavy_tail",
            "--seed",
            "42",
            "--max-requests",
            "12",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    rows = json.loads((output_dir / "summary.json").read_text())
    assert [row["policy"] for row in rows] == list(
        make_ablation_presets(p_values=DEFAULT_P_VALUES)
    )
    assert rows[0]["dispatch_timing"] == "latest_safe"
    assert rows[0]["backup_selection"] == "probability"
    assert rows[0]["latency_profile_mode"] == "configured"
    assert rows[0]["production_baseline_policy"] == production_baseline_policy_name()
    assert rows[0]["cost_multiplier_basis"] == "mean_total_cost_usd"
    assert rows[0]["p99_delta_vs_production_ms"] == 0.0
    assert rows[2]["backup_selection"] == "random_feasible"
    assert rows[2]["backup_selection_semantics"] == (
        "random_among_feasible_non_primary"
    )

    with (output_dir / "summary.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 3
    assert csv_rows[0]["production_baseline_policy"] == production_baseline_policy_name()
    assert csv_rows[0]["cost_multiplier_basis"] == "mean_total_cost_usd"
    assert csv_rows[0]["latency_profile_mode"] == "configured"


def test_routewise_cli_registers_hedging_ablation() -> None:
    assert ABLATION_COMMANDS["hedging"] == "experiments.ablations.hedging.harness"


def test_harness_lists_section_hedging_scenarios() -> None:
    assert harness.list_scenarios() == (
        "hedging_heavy_tail",
        "hedging_real_world_rw3",
        "hedging_real_world_rw8",
    )
    assert harness.policies_for_ablation() == tuple(make_ablation_presets())
    assert ablation_policy_name("latest_safe", "probability", p=0.75) in (
        harness.policies_for_ablation()
    )


def _primary_with_safe_and_slow_backups() -> list[TieredProvider]:
    return [
        _provider("primary", 3000.0, 4000.0, cost=1e-6),
        _provider("cheap_but_too_slow", 1400.0, 3000.0, cost=1e-6),
        _provider("safe", 100.0, 200.0, cost=10e-6),
    ]


def _provider(name: str, low_ms: float, high_ms: float, *, cost: float) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=cost,
        ttft_dist=Uniform(low_ms, high_ms),
        tps_dist=Uniform(100.0, 200.0),
        tier=ProviderTier.S_A,
    )


def _request() -> Request:
    return Request(
        id=1,
        timestamp=0.0,
        request_tokens=100,
        response_tokens=50,
        total_tokens=150,
    )
