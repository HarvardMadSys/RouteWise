"""Scenario-level orchestration for simulation policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rwsim.policies import available_policies
from rwsim.runner import run_policy

if TYPE_CHECKING:
    from rwsim.metrics import Run
    from rwsim.schemas import Request
    from rwsim.world.scenarios import ScenarioConfig


def run_simulation_scenario(
    scenario: ScenarioConfig,
    requests: list[Request],
    seeds: list[int] | None = None,
    policies: list[str] | None = None,
) -> dict[str, list[Run]]:
    """Run the policy set on one simulation scenario."""
    if seeds is None:
        seeds = [42, 43, 44]
    if policies is None:
        policies = list(available_policies())

    results: dict[str, list[Run]] = {name: [] for name in policies}
    for policy_name in policies:
        for seed in seeds:
            run = run_policy(scenario, requests, policy_name, seed=seed)
            results[policy_name].append(run)
    return results


__all__ = ["run_simulation_scenario"]
