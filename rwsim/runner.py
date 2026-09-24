"""Canonical policy runner for the RouteWise simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rwsim.engine.simulator import Simulator
from rwsim.policies import available_policies, build_policy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rwsim.metrics import Run
    from rwsim.schemas import Request
    from rwsim.world.scenarios import ScenarioConfig


def run_policy(
    scenario: ScenarioConfig,
    requests: Sequence[Request],
    policy_name: str,
    *,
    seed: int = 42,
) -> Run:
    """Run one named policy preset on one scenario."""
    policy = build_policy(policy_name, seed=seed)
    simulator = Simulator(scenario=scenario, seed=seed)
    return simulator.run(requests, policy, policy_name=policy_name)


POLICIES = available_policies()

__all__ = ["POLICIES", "run_policy"]
