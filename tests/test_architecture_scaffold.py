"""Dependency-light checks for the target simulator architecture."""

from __future__ import annotations

import importlib
import inspect
import unittest
from pathlib import Path

from experiments import available_experiments
from experiments._configs import summarize_scenario
from llm_routewise.schemas import ProviderTier, Request
from llm_routewise.sim.policies import DEFAULT_PRESETS, available_policies, build_policy
from llm_routewise.sim.scenarios import build_scenario

ROOT_DIR = Path(__file__).resolve().parents[1]


class ArchitectureScaffoldTest(unittest.TestCase):
    def test_generic_scenario_builder_stays_outside_paper_specific_factories(self) -> None:
        scenario = build_scenario(
            {
                "name": "unit_smoke",
                "primary_slo_ms": 1000.0,
                "workload": {
                    "source": "trace",
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

    def test_policy_presets_are_flat_and_paper_named(self) -> None:
        self.assertEqual(
            available_policies(),
            (
                "greedy_cost",
                "greedy_latency",
                "random",
                "ablation_lp_only",
                "ablation_lp_hedging",
                "routewise",
            ),
        )
        routewise_names = {"ablation_lp_only", "ablation_lp_hedging", "routewise"}
        test_presets = {
            name: (
                {
                    **preset,
                    "params": {
                        **preset.get("params", {}),
                        "cost_envelope": (1e-6, 1e-3),
                    },
                }
                if name in routewise_names
                else preset
            )
            for name, preset in DEFAULT_PRESETS.items()
        }
        for name in available_policies():
            policy = build_policy(name, presets=test_presets, seed=123)
            self.assertTrue(callable(policy.route))
            self.assertTrue(callable(policy.tick))
            self.assertTrue(callable(policy.observe))

    def test_experiment_registry_has_expected_packages(self) -> None:
        self.assertEqual(
            available_experiments(),
            ("estimator_ablation",),
        )

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

    def test_routewise_package_does_not_depend_on_application_layers(self) -> None:
        forbidden = (
            "from experiments",
            "import experiments",
            "from routewise_cli",
            "import routewise_cli",
        )

        for path in (ROOT_DIR / "llm_routewise").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path} must not import {token!r}")

    def test_experiments_do_not_depend_on_cli(self) -> None:
        forbidden = ("from routewise_cli", "import routewise_cli")

        for path in (ROOT_DIR / "experiments").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path} must not import {token!r}")

    def test_artifact_package_is_reviewer_facing_only(self) -> None:
        forbidden = ("from artifact", "import artifact")

        for dirname in ("llm_routewise", "experiments", "routewise_cli", "plots"):
            for path in (ROOT_DIR / dirname).rglob("*.py"):
                source = path.read_text(encoding="utf-8")
                for token in forbidden:
                    self.assertNotIn(token, source, f"{path} must not import {token!r}")

    def test_routewise_cli_is_repository_only_and_uses_policy_flag(self) -> None:
        cli = ROOT_DIR / "routewise_cli" / "main.py"
        pyproject = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
        source = cli.read_text(encoding="utf-8")

        self.assertTrue(cli.exists())
        self.assertNotIn('[project.scripts]', pyproject)
        self.assertNotIn('routewise = "routewise_cli.main:main"', pyproject)
        self.assertIn("--policy", source)
        self.assertNotIn("--" + "strategy", source)

    def test_deleted_strategy_and_stage_modules_are_absent(self) -> None:
        deleted_paths = (
            # The pre-unification package root: everything moved under
            # llm_routewise/ (shared layers) and llm_routewise/sim/ (simulator).
            ROOT_DIR / "rwsim",
            ROOT_DIR / "llm_routewise" / "sim" / "strategies",
            ROOT_DIR / "llm_routewise" / "sim" / "registry.py",
            ROOT_DIR / "llm_routewise" / "sim" / "world" / "shadow_price.py",
            ROOT_DIR / "llm_routewise" / "sim" / "world" / "workload.py",
            ROOT_DIR / "llm_routewise" / "sim" / "policies" / "composer.py",
            ROOT_DIR / "llm_routewise" / "sim" / "policies" / "value_estimators",
            ROOT_DIR / "llm_routewise" / "sim" / "policies" / "cost_routers",
            ROOT_DIR / "llm_routewise" / "sim" / "policies" / "latency_routers",
            ROOT_DIR / "llm_routewise" / "sim" / "policies" / "hedgers",
            ROOT_DIR / "experiments" / "suites.py",
            ROOT_DIR / "experiments" / "simulation" / "configs",
            ROOT_DIR / "experiments" / "simulation" / "suites",
            ROOT_DIR / "experiments" / "simulation" / "eval_grid.py",
            ROOT_DIR / "experiments" / "simulation" / "experiment.py",
            ROOT_DIR / "experiments" / "simulation" / "lp_budget_eval.py",
            ROOT_DIR / "experiments" / "simulation" / "materialize.py",
            ROOT_DIR / "experiments" / "simulation" / "runner.py",
            ROOT_DIR / "experiments" / "simulation" / "scenarios.py",
            ROOT_DIR / "tests" / "golden" / "simulation",
        )
        for path in deleted_paths:
            self.assertFalse(path.exists(), str(path))

    def test_no_strategy_compatibility_import_surface_remains(self) -> None:
        forbidden = (
            "llm_routewise.sim." + "strategies",
            "run_registered_" + "strategy",
            "STRATEGY_" + "REGISTRY",
            "TIERED_" + "STRATEGIES",
            "from llm_routewise.sim.policies." + "cost_routers",
            "from llm_routewise.sim.policies." + "latency_routers",
            "from llm_routewise.sim.policies." + "hedgers",
            "from llm_routewise.sim.world import generate_" + "workload",
            "from experiments.simulation." + "eval_grid",
            "from experiments.simulation." + "lp_budget_eval",
            "from experiments.simulation." + "materialize",
            "experiments.simulation." + "suites",
        )
        for dirname in ("llm_routewise", "experiments", "routewise_cli", "tests"):
            for path in (ROOT_DIR / dirname).rglob("*.py"):
                source = path.read_text(encoding="utf-8")
                for token in forbidden:
                    self.assertNotIn(token, source, f"{path} still references {token}")

    def test_metrics_have_canonical_home(self) -> None:
        import llm_routewise.metrics
        import llm_routewise.sim.world
        from llm_routewise.metrics import Run

        self.assertTrue((ROOT_DIR / "llm_routewise" / "metrics" / "run.py").exists())
        self.assertFalse(
            (ROOT_DIR / "llm_routewise" / "sim" / "world" / "metrics.py").exists()
        )
        self.assertEqual(Run.__module__, "llm_routewise.metrics.run")
        self.assertNotIn("SimulationRun", llm_routewise.metrics.__all__)
        self.assertNotIn("SimulationRun", llm_routewise.sim.world.__all__)

    def test_policy_contract_supports_in_flight_ticks(self) -> None:
        from llm_routewise.sim.policies.base import Policy

        route_sig = inspect.signature(Policy.route)
        tick_sig = inspect.signature(Policy.tick)
        observe_sig = inspect.signature(Policy.observe)

        self.assertEqual(tuple(route_sig.parameters), ("self", "request", "state"))
        self.assertEqual(
            tuple(tick_sig.parameters),
            ("self", "request", "decision", "elapsed", "state"),
        )
        self.assertEqual(
            tuple(observe_sig.parameters),
            ("self", "request", "decision", "outcome"),
        )

    def test_routewise_algorithm_has_canonical_home(self) -> None:
        """The RouteWise algorithm lives once, in llm_routewise.core.router.

        Both environments must delegate to it: the simulator policy and the
        real-eval BudgetRange policies expose it via a ``router`` attribute,
        and neither adapter re-implements the LP body orchestration.
        """
        import llm_routewise.core as public_core
        from llm_routewise.core.beliefs import LatencyBeliefs
        from llm_routewise.core.router import RouteWiseRouter

        self.assertEqual(RouteWiseRouter.__module__, "llm_routewise.core.router")
        self.assertEqual(LatencyBeliefs.__module__, "llm_routewise.core.beliefs")
        self.assertIs(public_core.RouteWiseRouter, RouteWiseRouter)

        # Neither adapter may call the LP solver directly; the router owns
        # the body orchestration.
        sim_adapter = (
            ROOT_DIR / "llm_routewise" / "sim" / "policies" / "routewise.py"
        ).read_text()
        real_adapter = (
            ROOT_DIR / "experiments" / "real_evaluation" / "policies.py"
        ).read_text()
        for source, label in ((sim_adapter, "sim"), (real_adapter, "real-eval")):
            self.assertNotIn(
                "solve_budget_lp(",
                source,
                f"{label} adapter must delegate the budget LP to llm_routewise.core.router",
            )

    def test_offline_stage_core_has_canonical_home(self) -> None:
        from experiments.offline_stage import DEFAULT_CONFIG_PATH
        from llm_routewise.offline import CostCalculator, Request

        self.assertTrue((ROOT_DIR / "llm_routewise" / "offline" / "schemas.py").exists())
        self.assertTrue((ROOT_DIR / "llm_routewise" / "offline" / "simulator.py").exists())
        self.assertTrue(
            (ROOT_DIR / "experiments" / "offline_stage" / "configs" / "experiment.yaml").exists()
        )
        self.assertFalse((ROOT_DIR / "config").exists())
        self.assertEqual(CostCalculator.__module__, "llm_routewise.offline.cost")
        self.assertEqual(Request.__module__, "llm_routewise.offline.schemas")
        self.assertEqual(
            DEFAULT_CONFIG_PATH,
            ROOT_DIR / "experiments" / "offline_stage" / "configs" / "experiment.yaml",
        )

    def test_offline_value_estimators_are_not_sim_policy_stages(self) -> None:
        from experiments.offline_stage.value_estimators import (
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

    def test_section_simulator_phase0_surface_is_registered_incrementally(self) -> None:
        from routewise_cli.main import SIMULATOR_SECTIONS

        self.assertEqual(
            SIMULATOR_SECTIONS,
            {
                "cost-layer": "experiments.simulation.cost_layer",
                "latency-layer": "experiments.simulation.latency_layer",
                "hedging": "experiments.simulation.hedging",
                "end-to-end": "experiments.simulation.end_to_end",
            },
        )
        for module_name in (
            "experiments.simulation.cost_layer",
            "experiments.simulation.latency_layer",
            "experiments.simulation.hedging",
            "experiments.simulation.end_to_end",
        ):
            module = importlib.import_module(module_name)
            self.assertTrue(hasattr(module, "SECTION_NAME"))
            self.assertTrue(hasattr(module, "list_scenarios"))
            self.assertTrue(hasattr(module, "make_scenarios"))

        end_to_end = importlib.import_module("experiments.simulation.end_to_end")
        self.assertTrue(hasattr(end_to_end, "main"))


if __name__ == "__main__":
    unittest.main()
