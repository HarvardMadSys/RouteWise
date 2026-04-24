"""Dependency-light checks for the migration architecture scaffold."""

from __future__ import annotations

import unittest

from experiments import available_experiments, get_experiment
from experiments._configs import summarize_scenario
from rwsim.policies import available_aliases, get_pipeline_alias
from rwsim.scenarios import build_scenario
from rwsim.schemas import ProviderTier


class ArchitectureScaffoldTest(unittest.TestCase):
    def test_tiered_configs_are_discoverable_without_yaml_dependency(self) -> None:
        experiment = get_experiment("tiered_capacity")

        self.assertEqual(
            experiment.list_scenarios(),
            (
                "s6_slow_q_trap",
                "s7_quota_depletion",
                "s8_concurrency_saturation",
                "s9_quota_concurrency_priority",
                "unified_pool",
            ),
        )

    def test_generic_scenario_builder_stays_outside_paper_specific_factories(self) -> None:
        scenario = build_scenario(
            {
                "name": "unit_smoke",
                "primary_slo_ms": 1000.0,
                "workload": {
                    "source": "synthetic",
                    "n_requests": 1,
                    "duration_seconds": 1.0,
                },
                "providers": [
                    {
                        "name": "api",
                        "tier": "api",
                        "cost_per_token": 0.000001,
                        "ttft_distribution": {"name": "fixed", "value_ms": 10.0},
                    }
                ],
            }
        )

        self.assertEqual(scenario.name, "unit_smoke")
        self.assertEqual(scenario.providers[0].tier, ProviderTier.API)
        self.assertEqual(summarize_scenario(scenario)["providers"]["count"], 1)

    def test_strategy_aliases_map_to_pipeline_stages(self) -> None:
        self.assertIn("joint_hedge", available_aliases())
        alias = get_pipeline_alias("joint_hedge")

        self.assertEqual(alias.cost_router.name, "cheapest_effective")
        self.assertEqual(alias.latency_router.name, "p95_slo_filter")
        self.assertEqual(alias.hedger.name, "smart_economic")

    def test_experiment_registry_has_expected_packages(self) -> None:
        self.assertEqual(
            available_experiments(),
            ("estimator_ablation", "synthetic_latency", "tiered_capacity"),
        )


if __name__ == "__main__":
    unittest.main()
