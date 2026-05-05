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
    "simulator_grid": SuiteSpec(
        name="simulator_grid",
        module="experiments.simulation.suites.run_simulator_grid",
        description="Run the paper simulator grid evaluation.",
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
