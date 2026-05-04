"""Dependency-light checks for the migration architecture scaffold."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments import available_experiments, get_experiment
from experiments._configs import summarize_scenario
from experiments.suites import available_suites, get_suite
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
            ("estimator_ablation", "tiered_capacity"),
        )

    def test_tiered_calibrated_scenarios_have_canonical_home(self) -> None:
        canonical = ROOT_DIR / "experiments" / "tiered_capacity" / "calibrated.py"

        self.assertTrue(canonical.exists())
        self.assertIn("make_calibrated_scenarios", canonical.read_text(encoding="utf-8"))

    def test_request_schema_keeps_convenience_properties(self) -> None:
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

    def test_rwsim_does_not_depend_on_application_layers(self) -> None:
        forbidden = (
            "from experiments",
            "import experiments",
            "from routewise_cli",
            "import routewise_cli",
        )

        for path in (ROOT_DIR / "rwsim").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path} must not import {token!r}")

    def test_experiments_do_not_depend_on_cli(self) -> None:
        forbidden = ("from routewise_cli", "import routewise_cli")

        for path in (ROOT_DIR / "experiments").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path} must not import {token!r}")

    def test_routewise_cli_is_application_layer(self) -> None:
        cli = ROOT_DIR / "routewise_cli" / "main.py"
        pyproject = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        source = cli.read_text(encoding="utf-8")

        self.assertTrue(cli.exists())
        self.assertIn('routewise = "routewise_cli.main:main"', pyproject)
        self.assertIn("from experiments", source)
        self.assertNotIn("from routewise_cli", (ROOT_DIR / "rwsim" / "__init__.py").read_text())

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

    def test_algorithm_document_has_stage_level_correctness_contracts(self) -> None:
        source = (ROOT_DIR / "docs" / "ALGORITHMS.md").read_text(encoding="utf-8")

        for stage in ("Value Estimator", "Cost Router", "Latency Router", "Hedger"):
            self.assertIn(f"### {stage}", source)
            section = source.split(f"### {stage}", 1)[1].split("\n### ", 1)[0]
            for token in (
                "Canonical Definition (authoritative)",
                "Pipeline Role",
                "Three-Column Reconciliation",
                "Stage Interface",
                "Unit-Test Contract",
                "Known Divergence to Resolve",
                "[eq. 1]",
            ):
                self.assertIn(token, section, f"{stage} is missing {token}")

        self.assertIn("P_viol(t) * F_b(R_b(t)) > C_b / V", source)
        self.assertIn("max(1.5 * P50, 0.5 * SLO)", source)

    def test_old_experiment_packages_are_removed(self) -> None:
        self.assertFalse((ROOT_DIR / "experiment").exists())
        self.assertFalse((ROOT_DIR / ("leg" + "acy")).exists())
        self.assertFalse((ROOT_DIR / "experiments" / "latency_phase3").exists())
        self.assertFalse((ROOT_DIR / "experiments" / "latency_phase4").exists())
        self.assertFalse((ROOT_DIR / "experiments" / "synthetic_latency").exists())

        for path in ROOT_DIR.rglob("*.py"):
            if ".venv" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            old_from = "from " + "experiment."
            old_import = "import " + "experiment."
            self.assertNotIn(old_from, source, f"{path} should use rwsim or experiments")
            self.assertNotIn(old_import, source, f"{path} should use rwsim or experiments")

    def test_experiment_runners_have_canonical_suite_home(self) -> None:
        for path in ROOT_DIR.glob("run_*.py"):
            self.fail(f"{path} should live under experiments/*/suites")
        self.assertFalse((ROOT_DIR / "replot_phase_diagram.py").exists())
        self.assertFalse((ROOT_DIR / "rwsim" / "drivers").exists())
        self.assertFalse((ROOT_DIR / "scripts").exists())

        tiered_suites = (
            "run_joint.py",
            "run_joint_mm25_baselines.py",
            "run_stress_tests.py",
            "run_joint_lp_budget_eval.py",
        )

        for relpath in tiered_suites:
            path = ROOT_DIR / "experiments" / "tiered_capacity" / "suites" / relpath
            self.assertTrue(path.exists(), relpath)

        deleted_import = "leg" + "acy" + ".experiment"
        for path in (ROOT_DIR / "experiments" / "tiered_capacity" / "suites").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(deleted_import, source, f"{path} should use canonical imports")
            self.assertNotIn("scripts.experiments", source, f"{path} should use experiment suites")

    def test_full_sweep_suites_are_registered(self) -> None:
        expected = (
            "joint",
            "joint_lp_budget_eval",
            "joint_mm25_baselines",
            "stress",
        )

        self.assertEqual(available_suites(), expected)
        for name in expected:
            self.assertTrue(get_suite(name).module.startswith("experiments."))

    def test_canonical_packages_do_not_import_deleted_packages(self) -> None:
        deleted_import = "leg" + "acy" + ".experiment"
        for dirname in ("rwsim", "experiments", "routewise_cli", "tests"):
            for path in (ROOT_DIR / dirname).rglob("*.py"):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(deleted_import, source, f"{path} should not import deleted packages")

    def test_value_estimators_live_under_policy_stage(self) -> None:
        from rwsim.policies.value_estimators import (
            EMAOutputPredictor,
            HistogramOutputPredictor,
            OracleOutputPredictor,
        )

        request = Request(
            id=1,
            timestamp=0,
            request_tokens=128,
            response_tokens=64,
            total_tokens=192,
            model="unit-model",
        )

        ema = EMAOutputPredictor(min_samples_warmup=1)
        ema.update(request)

        self.assertEqual(type(ema.predict(request)).__name__, "QuantilePrediction")
        self.assertEqual(OracleOutputPredictor().predict(request).q50, 64.0)
        self.assertFalse(HistogramOutputPredictor().predict(request).is_warmed_up)

    def test_offline_stage_core_has_canonical_home(self) -> None:
        from experiments.offline_stage import DEFAULT_CONFIG_PATH
        from rwsim.offline import CostCalculator, Request

        self.assertTrue((ROOT_DIR / "rwsim" / "offline" / "schemas.py").exists())
        self.assertTrue((ROOT_DIR / "rwsim" / "offline" / "simulator.py").exists())
        self.assertTrue(
            (ROOT_DIR / "experiments" / "offline_stage" / "configs" / "experiment.yaml").exists()
        )
        self.assertTrue((ROOT_DIR / "config" / "experiment.yaml").is_symlink())
        self.assertEqual(CostCalculator.__module__, "rwsim.offline.cost")
        self.assertEqual(Request.__module__, "rwsim.offline.schemas")
        self.assertEqual(
            DEFAULT_CONFIG_PATH,
            ROOT_DIR / "experiments" / "offline_stage" / "configs" / "experiment.yaml",
        )

    def test_offline_stage_strategies_have_canonical_home(self) -> None:
        canonical = ROOT_DIR / "experiments" / "offline_stage" / "strategies"

        for relpath in (
            "all_api.py",
            "greedy.py",
            "stage1_optimal.py",
            "stage1_optimal_v2.py",
            "stage1_latency.py",
            "stage2_baselines.py",
            "stage2_optimal.py",
            "online/base.py",
            "online/greedy.py",
            "online/learning_augmented.py",
            "online/primal_dual.py",
        ):
            self.assertTrue((canonical / relpath).exists(), relpath)

    def test_latency_profiling_probe_has_canonical_home(self) -> None:
        canonical = ROOT_DIR / "experiments" / "offline_stage" / "latency_profiling.py"

        self.assertTrue(canonical.exists())
        self.assertIn("PROJECT_ROOT", canonical.read_text(encoding="utf-8"))

    def test_basic_cost_routers_live_under_policy_stage(self) -> None:
        from rwsim.policies.cost_routers import cheapest_provider, provider_for_index

        providers = (
            SimpleNamespace(name="expensive", cost_per_token=3.0),
            SimpleNamespace(name="cheap", cost_per_token=1.0),
        )

        self.assertEqual(cheapest_provider(providers).name, "cheap")
        self.assertEqual(provider_for_index(providers, 3).name, "cheap")

    def test_world_modules_do_not_import_old_core(self) -> None:
        old_core_token = "experiment.scripts.simulate.synthetic._core"

        for path in (ROOT_DIR / "rwsim" / "world").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(old_core_token, source, f"{path} should own this implementation")

    def test_runner_uses_rwsim_strategy_registry(self) -> None:
        old_registry_token = "experiment.scripts.simulate.synthetic._core.strategies"

        for relpath in ("rwsim/runner.py", "rwsim/strategies/__init__.py"):
            source = (ROOT_DIR / relpath).read_text(encoding="utf-8")
            self.assertNotIn(old_registry_token, source)
            self.assertIn("rwsim.strategies.registry", source)

        for path in (ROOT_DIR / "rwsim" / "strategies").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(old_registry_token, source)

    def test_rwsim_does_not_load_old_strategy_modules(self) -> None:
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
