"""Dependency-light checks for the migration architecture scaffold."""

from __future__ import annotations

import unittest
from pathlib import Path

from experiments import available_experiments, get_experiment
from experiments._configs import summarize_scenario
from rwsim.policies import available_aliases, get_pipeline_alias
from rwsim.scenarios import build_scenario
from rwsim.schemas import ProviderTier, Request


ROOT_DIR = Path(__file__).resolve().parents[1]


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

    def test_request_schema_keeps_legacy_convenience_properties(self) -> None:
        request = Request(
            id=1,
            timestamp=90000.0,
            request_tokens=100,
            response_tokens=50,
            total_tokens=150,
        )

        self.assertEqual(request.day, 1)
        self.assertEqual(request.time_of_day, 3600)
        self.assertEqual(request.latency_seconds, 0.075)

    def test_rwsim_does_not_depend_on_experiments_or_scripts(self) -> None:
        forbidden = ("from experiments", "import experiments", "from scripts", "import scripts")

        for path in (ROOT_DIR / "rwsim").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path} must not import {token!r}")

    def test_rwsim_scenarios_has_no_concrete_paper_factories(self) -> None:
        source = (ROOT_DIR / "rwsim" / "scenarios.py").read_text(encoding="utf-8")

        forbidden = (
            "make_s6",
            "make_s7",
            "make_s8",
            "make_s9",
            "make_unified_pool",
            "s6_slow_q_trap",
            "s7_quota_depletion",
            "unified_pool",
        )
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_policy_stage_directories_exist(self) -> None:
        for dirname in ("value_estimators", "cost_routers", "latency_routers", "hedgers"):
            self.assertTrue((ROOT_DIR / "rwsim" / "policies" / dirname).is_dir(), dirname)

    def test_world_modules_do_not_import_legacy_core(self) -> None:
        legacy_token = "experiment.scripts.simulate.synthetic._core"

        for path in (ROOT_DIR / "rwsim" / "world").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(legacy_token, source, f"{path} should own this implementation")

    def test_runner_uses_rwsim_strategy_registry(self) -> None:
        legacy_registry_token = "experiment.scripts.simulate.synthetic._core.strategies"

        for relpath in ("rwsim/runner.py", "rwsim/strategies/__init__.py"):
            source = (ROOT_DIR / relpath).read_text(encoding="utf-8")
            self.assertNotIn(legacy_registry_token, source)
            self.assertIn("rwsim.strategies.registry", source)

        for path in (ROOT_DIR / "rwsim" / "strategies").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(legacy_registry_token, source)

    def test_rwsim_does_not_load_legacy_strategy_modules(self) -> None:
        forbidden = (
            "experiment.strategies",
            "experiment/strategies",
            "spec_from_file_location",
        )

        for path in (ROOT_DIR / "rwsim").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path} should use rwsim policies")


if __name__ == "__main__":
    unittest.main()
