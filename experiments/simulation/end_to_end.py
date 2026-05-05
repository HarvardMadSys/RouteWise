"""End-to-end simulator section planning module.

This file is intentionally not CLI-registered until empirical provider
profiles are wired into the simulator.
"""

from __future__ import annotations

SECTION_NAME = "end-to-end"
PLANNED_SCENARIOS = (
    "end_to_end_rw3_with_hedging",
    "end_to_end_rw3_no_hedge",
    "end_to_end_rw8_with_hedging",
    "end_to_end_rw8_no_hedge",
)


def list_scenarios() -> tuple[str, ...]:
    """Return planned scenario names for the unregistered end-to-end section."""
    return PLANNED_SCENARIOS


def make_scenarios() -> dict[str, object]:
    """Return no scenarios until empirical provider profiles are wired."""
    return {}


__all__ = ["PLANNED_SCENARIOS", "SECTION_NAME", "list_scenarios", "make_scenarios"]
