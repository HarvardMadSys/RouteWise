"""Scenario-level orchestration for the tiered strategies.

Runs each strategy in TIERED_STRATEGIES across a scenario with multiple
seeds and aggregates the results.
"""

from __future__ import annotations

from legacy.experiment.data.schema import Request

from .scenarios import TieredScenarioConfig
from .strategies import TIERED_STRATEGIES, StrategyRun, run_tiered_strategy


def run_tiered_scenario(
    scenario: TieredScenarioConfig,
    requests: list[Request],
    seeds: list[int] | None = None,
    strategies: list[str] | None = None,
) -> dict[str, list[StrategyRun]]:
    """Run the given strategies on a scenario across multiple seeds.

    Parameters
    ----------
    scenario:
        A TieredScenarioConfig. Note that the scenario's provider state
        (quota/concurrency) is mutated during a run; the strategy runner
        resets it at the start of each invocation.
    requests:
        Pre-generated workload, shared across all strategies and seeds.
    seeds:
        RNG seeds for latency sampling. Defaults to [42, 43, 44].
    strategies:
        Subset of TIERED_STRATEGIES to run. Defaults to all.

    Returns
    -------
    Dict mapping strategy name to list of StrategyRun (one per seed).
    """
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
