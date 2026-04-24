"""Registry for full-sweep experiment suites.

Suites are experiment-layer entrypoints that run a paper sweep or diagnostic
grid. They are intentionally outside :mod:`rwsim`; the simulator core must not
know these application-level runners exist.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class SuiteSpec:
    """One callable full-sweep experiment suite."""

    name: str
    module: str
    description: str


_SUITES: dict[str, SuiteSpec] = {
    "synthetic": SuiteSpec(
        name="synthetic",
        module="experiments.synthetic_latency.suites.run_synthetic",
        description="Run all synthetic latency scenarios.",
    ),
    "sanity": SuiteSpec(
        name="sanity",
        module="experiments.synthetic_latency.suites.run_sanity_check",
        description="Run the synthetic latency sanity-check sweep.",
    ),
    "alpha_sweep": SuiteSpec(
        name="alpha_sweep",
        module="experiments.synthetic_latency.suites.run_alpha_sweep",
        description="Run the alpha-weighted LP sweep.",
    ),
    "alpha_on_sanity": SuiteSpec(
        name="alpha_on_sanity",
        module="experiments.synthetic_latency.suites.run_alpha_on_sanity",
        description="Run alpha-weighted LP on sanity-check scenarios.",
    ),
    "pareto": SuiteSpec(
        name="pareto",
        module="experiments.synthetic_latency.suites.run_pareto",
        description="Run Pareto frontier sweeps.",
    ),
    "phase_diagram": SuiteSpec(
        name="phase_diagram",
        module="experiments.synthetic_latency.suites.run_phase_diagram",
        description="Run the LP-vs-V2 phase diagram sweep.",
    ),
    "phase_diagram_v2": SuiteSpec(
        name="phase_diagram_v2",
        module="experiments.synthetic_latency.suites.run_phase_diagram_v2",
        description="Run the Llama-like phase diagram sweep.",
    ),
    "replot_phase_diagram": SuiteSpec(
        name="replot_phase_diagram",
        module="experiments.synthetic_latency.suites.replot_phase_diagram",
        description="Replot the cached phase diagram cells.",
    ),
    "joint": SuiteSpec(
        name="joint",
        module="experiments.tiered_capacity.suites.run_joint",
        description="Run tiered joint-vs-two-layer scenarios.",
    ),
    "joint_mm25_baselines": SuiteSpec(
        name="joint_mm25_baselines",
        module="experiments.tiered_capacity.suites.run_joint_mm25_baselines",
        description="Run MM25 calibrated baseline scenarios.",
    ),
    "stress": SuiteSpec(
        name="stress",
        module="experiments.tiered_capacity.suites.run_stress_tests",
        description="Run tiered stress scenarios.",
    ),
    "joint_lp_budget_eval": SuiteSpec(
        name="joint_lp_budget_eval",
        module="experiments.tiered_capacity.suites.run_joint_lp_budget_eval",
        description="Run the sidecar LP-budget evaluation.",
    ),
}


def available_suites() -> tuple[str, ...]:
    """Return registered full-sweep suite names."""
    return tuple(sorted(_SUITES))


def get_suite(name: str) -> SuiteSpec:
    """Return one full-sweep suite spec."""
    try:
        return _SUITES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown suite {name!r}. Available: {', '.join(available_suites())}") from exc


def load_suite_module(name: str) -> ModuleType:
    """Import and return the module for one suite."""
    return importlib.import_module(get_suite(name).module)


def run_suite(name: str, argv: list[str] | None = None) -> int:
    """Run a suite module's ``main()`` with optional suite-local arguments."""
    module = load_suite_module(name)
    main = getattr(module, "main", None)
    if main is None:
        raise AttributeError(f"Suite {name!r} module {module.__name__!r} has no main()")

    old_argv = sys.argv[:]
    sys.argv = [module.__name__, *(argv or [])]
    try:
        result = main()
    finally:
        sys.argv = old_argv
    return int(result or 0)


__all__ = [
    "SuiteSpec",
    "available_suites",
    "get_suite",
    "load_suite_module",
    "run_suite",
]
