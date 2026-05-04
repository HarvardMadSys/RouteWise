"""Scenario-level orchestration for simulation strategies."""

from __future__ import annotations

from rwsim.schemas import Request
from rwsim.strategies.tiered_impl import TIERED_STRATEGIES, StrategyRun, run_tiered_strategy
from rwsim.world.scenarios import ScenarioConfig


def run_simulation_scenario(
    scenario: ScenarioConfig,
    requests: list[Request],
    seeds: list[int] | None = None,
    strategies: list[str] | None = None,
) -> dict[str, list[StrategyRun]]:
    """Run the tier-aware strategy set on one simulation scenario."""
    if seeds is None:
        seeds = [42, 43, 44]
    if strategies is None:
        strategies = TIERED_STRATEGIES

    results: dict[str, list[StrategyRun]] = {s: [] for s in strategies}
    for strategy in strategies:
        for seed in seeds:
            run = run_tiered_strategy(scenario, requests, strategy, seed=seed)
            results[strategy].append(run)
    return results


__all__ = ["run_simulation_scenario"]
