"""Experiment-local RouteWise variant whose latency beliefs never refresh.

``RouteWisePolicy`` in ``configured`` mode reads each provider's *current*
true TTFT distribution, so its beliefs track a provider-side latency spike
instantly (a fresh-telemetry oracle). The stale-telemetry experiment also
needs the opposite limit: a latency profiler that has not been updated since
the spike began. ``StaleBeliefRouteWisePolicy`` keeps the LP body, the
hedging trigger, and the backup selection unchanged and only swaps the
provider view so every prior (mean and CDF) is read from the provider's
baseline ``ttft_dist``, ignoring any ``ttft_shift_schedule`` active at query
time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_routewise.sim.policies.routewise import RouteWisePolicy, _SimProviderView

if TYPE_CHECKING:
    from llm_routewise.schemas import Request
    from llm_routewise.sim.engine.state import SimulationState
    from llm_routewise.sim.policies.latency_profiles import LatencyProfileMode
    from llm_routewise.sim.world.providers import Provider


@dataclass(frozen=True)
class _BaselinePriorProviderView(_SimProviderView):
    """Provider view whose latency priors ignore time-varying TTFT shifts."""

    def prior_ttft_mean_ms(self, now: float) -> float | None:
        del now
        return self.provider.ttft_dist.mean()

    def prior_ttft_cdf(self, value_ms: float, now: float) -> float | None:
        del now
        return self.provider.ttft_dist.cdf(value_ms)


@dataclass
class StaleBeliefRouteWisePolicy(RouteWisePolicy):
    """RouteWise whose latency profile is frozen at the pre-spike baseline."""

    latency_profile_mode: LatencyProfileMode = "configured"

    def __post_init__(self, p: float | None) -> None:
        if self.latency_profile_mode != "configured":
            raise ValueError(
                "StaleBeliefRouteWisePolicy requires latency_profile_mode='configured'; "
                "the frozen baseline prior is the only belief source"
            )
        super().__post_init__(p)

    def _view(
        self,
        provider: Provider,
        request: Request,
        state: SimulationState,
    ) -> _SimProviderView:
        return _BaselinePriorProviderView(
            provider=provider,
            request=request,
            state=state,
            output_predictor=self.output_predictor,
        )


__all__ = ["StaleBeliefRouteWisePolicy"]
