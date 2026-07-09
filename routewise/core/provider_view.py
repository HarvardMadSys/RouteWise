"""Read-only provider snapshot consumed by the core RouteWise router.

This protocol is the single seam between the environment-agnostic routing
algorithm (:mod:`routewise.core.router`) and a concrete world: the simulator binds
it over ``(Provider, Request, SimulationState)``, the real-eval harness over
``(ProviderState, RequestContext)``. Views are cheap request-bound objects
built by each adapter per routing call; the router never sees request schemas,
capacity ledgers, or transport details.

Only the signals the algorithm actually consumes appear here. How they are
produced (simulated ledgers vs live wall-clock and HTTP 429s) is adapter
business and intentionally NOT unified.
"""

from __future__ import annotations

from typing import Protocol


class ProviderView(Protocol):
    """One provider, bound to one request, at routing time."""

    @property
    def name(self) -> str:
        """Unique provider name (stable across the run)."""
        ...

    @property
    def tier(self) -> str:
        """Effective-cost tier: ``"api"``, ``"quota"``, or ``"concurrency"``."""
        ...

    def is_available(self, now: float) -> bool:
        """Whether the provider can admit a new request at ``now``."""
        ...

    def quota_fraction_used(self, now: float) -> float | None:
        """Quota window fraction consumed, or ``None`` when untracked."""
        ...

    def route_cost_usd(self, now: float) -> float:
        """Marginal USD cost of serving THIS request (route-time estimate).

        Each adapter applies its own costing here (output-token predictor,
        prefix-cache discount, subscription zero-cost tiers).
        """
        ...

    def hedge_cost_usd(self, now: float) -> float:
        """Marginal USD cost of dispatching THIS request as a hedge backup.

        The simulator prices hedges with trace-realized tokens while routing
        uses the predictor; real-eval uses the same estimate for both.
        """
        ...

    def prior_ttft_mean_ms(self, now: float) -> float | None:
        """Prior belief for mean TTFT, used when the rolling window is empty.

        Simulator adapter: the provider's configured (true) distribution —
        the oracle fallback. Real adapter: ``None`` (no prior exists).
        """
        ...

    def prior_ttft_cdf(self, value_ms: float, now: float) -> float | None:
        """Prior belief for ``P(TTFT <= value_ms)``; see ``prior_ttft_mean_ms``."""
        ...


__all__ = ["ProviderView"]
