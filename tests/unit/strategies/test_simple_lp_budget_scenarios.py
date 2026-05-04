"""Tests for hand-checkable LP-budget toy scenarios."""

from __future__ import annotations

import numpy as np

from experiments.simulation.lp_budget_eval import (
    EvaluatedRun,
    RunDiagnostics,
    build_all_scenarios,
    summarize_main_metrics,
)
from experiments.simulation.simple_scenarios import (
    SIMPLE_SCENARIOS,
    make_simple_scenarios,
)
from rwsim.world.capacity import ProviderTier
from rwsim.metrics import StrategyRun


def test_simple_scenarios_are_registered_in_sidecar() -> None:
    scenarios = build_all_scenarios()

    for name in SIMPLE_SCENARIOS:
        assert name in scenarios


def test_simple_scenarios_use_api_only_providers() -> None:
    scenarios = make_simple_scenarios()

    for scenario in scenarios.values():
        assert all(provider.tier == ProviderTier.S_A for provider in scenario.providers)


def test_same_cost_scenario_only_varies_latency() -> None:
    scenario = make_simple_scenarios()["simple_same_cost_different_latency"]

    costs = {provider.cost_per_token for provider in scenario.providers}
    p50s = [provider.true_p50_ms() for provider in scenario.providers]

    assert len(costs) == 1
    assert min(p50s) == scenario.providers[-1].true_p50_ms()


def test_slow_cheap_scenario_has_opposed_cost_latency_order() -> None:
    scenario = make_simple_scenarios()["simple_slow_cheap_fast_expensive"]
    providers = scenario.providers

    costs = [provider.cost_per_token for provider in providers]
    p50s = [provider.true_p50_ms() for provider in providers]

    assert costs == sorted(costs)
    assert p50s == sorted(p50s, reverse=True)


def test_provider_mix_aggregation_counts_missing_seed_as_zero() -> None:
    scenario = make_simple_scenarios()["simple_slow_cheap_fast_expensive"]
    runs = [
        StrategyRun(
            strategy="test",
            ttft_ms=np.array([1.0, 1.0]),
            cost_usd=np.array([1.0, 1.0]),
            provider=["a", "a"],
            timestamp=np.array([0.0, 1.0]),
            hedge_triggered=np.array([False, False]),
            tier=["api", "api"],
        ),
        StrategyRun(
            strategy="test",
            ttft_ms=np.array([1.0, 1.0]),
            cost_usd=np.array([1.0, 1.0]),
            provider=["b", "b"],
            timestamp=np.array([0.0, 1.0]),
            hedge_triggered=np.array([False, False]),
            tier=["api", "api"],
        ),
    ]
    evaluated = [
        EvaluatedRun(run=run, diagnostics=RunDiagnostics("test", scenario.name, idx))
        for idx, run in enumerate(runs)
    ]

    summary = summarize_main_metrics(scenario, evaluated)

    assert summary["provider_mix"] == {"a": 0.5, "b": 0.5}
