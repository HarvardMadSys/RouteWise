"""Simulator fallback admission regressions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rwsim.engine.simulator import Simulator, _fallback_admission
from rwsim.schemas import HedgeDispatch, Request, RoutingDecision, RoutingOutcome
from rwsim.world.capacity import ConcurrencyState, ProviderTier
from rwsim.world.distributions import Uniform
from rwsim.world.providers import TieredProvider
from rwsim.world.scenarios import ScenarioConfig

if TYPE_CHECKING:
    from rwsim.engine.state import SimulationState


@dataclass(frozen=True)
class _FixedDistribution:
    value: float

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        del rng
        return np.full(size, self.value, dtype=float)

    def p50(self) -> float:
        return self.value

    def p95(self) -> float:
        return self.value

    def p99(self) -> float:
        return self.value

    def quantile(self, q: float) -> float:
        del q
        return self.value

    def mean(self) -> float:
        return self.value

    def std(self) -> float:
        return 0.0

    def cdf(self, value: float) -> float:
        return 1.0 if value >= self.value else 0.0


class _SequenceDistribution:
    def __init__(self, *values: float) -> None:
        self.values = values
        self.index = 0

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        del rng
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += size
        return np.full(size, value, dtype=float)

    def p50(self) -> float:
        return self.values[0]

    def p95(self) -> float:
        return self.values[0]

    def p99(self) -> float:
        return self.values[0]

    def quantile(self, q: float) -> float:
        del q
        return self.values[0]

    def mean(self) -> float:
        return self.values[0]

    def std(self) -> float:
        return 0.0

    def cdf(self, value: float) -> float:
        return 1.0 if value >= self.values[0] else 0.0


class _PrimaryOnlyPolicy:
    def __init__(
        self,
        primary_provider: str,
        *,
        future_primary_occupancy: tuple[int, float, float] | None = None,
    ) -> None:
        self.primary_provider = primary_provider
        self.future_primary_occupancy = future_primary_occupancy
        self._occupancy_seeded = False

    def route(self, request: Request, state: SimulationState) -> RoutingDecision:
        del request
        if self.future_primary_occupancy is not None and not self._occupancy_seeded:
            request_id, now, service_time_sec = self.future_primary_occupancy
            provider = state.providers[self.primary_provider]
            assert provider.concurrency is not None
            provider.concurrency.admit(request_id, now, service_time_sec)
            self._occupancy_seeded = True
        return RoutingDecision(primary_provider=self.primary_provider)

    def tick(
        self,
        request: Request,
        decision: RoutingDecision,
        elapsed: float,
        state: SimulationState,
    ) -> HedgeDispatch | None:
        del request, decision, elapsed, state
        return None

    def observe(
        self,
        request: Request,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
    ) -> None:
        del request, decision, outcome


def test_fallback_admission_skips_sc_candidate_with_interval_conflict() -> None:
    cheap_concurrency = ConcurrencyState(limit=1)
    cheap_concurrency.admit(request_id=99, now=1.0, service_time_sec=2.0)
    cheap_sc = _provider(
        "cheap_sc",
        cost=1e-7,
        tier=ProviderTier.S_C,
        ttft_dist=_FixedDistribution(10.0),
        service_time_dist=_FixedDistribution(2000.0),
        concurrency=cheap_concurrency,
    )
    api = _provider(
        "api",
        cost=1e-6,
        tier=ProviderTier.S_A,
        ttft_dist=_FixedDistribution(20.0),
    )

    admission = _fallback_admission(
        [cheap_sc, api],
        now=0.0,
        output_tokens=10,
        provider_rngs=_rngs([cheap_sc, api]),
    )

    assert admission is not None
    assert admission.provider.name == "api"
    assert admission.ttft_ms == 20.0
    assert admission.service_time == 0.0


def test_primary_admission_failure_does_not_reroll_primary_as_fallback() -> None:
    primary_concurrency = ConcurrencyState(limit=1)
    primary_service_times = _SequenceDistribution(2000.0, 500.0)
    primary = _provider(
        "primary",
        cost=1e-7,
        tier=ProviderTier.S_C,
        ttft_dist=_FixedDistribution(10.0),
        service_time_dist=primary_service_times,
        concurrency=primary_concurrency,
    )
    api = _provider(
        "api",
        cost=1e-6,
        tier=ProviderTier.S_A,
        ttft_dist=_FixedDistribution(20.0),
    )
    scenario = ScenarioConfig(
        name="fallback_excludes_failed_primary",
        description="primary cannot be retried as its own fallback",
        providers=[primary, api],
    )
    request = Request(
        id=1,
        timestamp=0.0,
        request_tokens=10,
        response_tokens=10,
        total_tokens=20,
    )

    run = Simulator(scenario=scenario, seed=17).run(
        [request],
        _PrimaryOnlyPolicy(
            primary_provider="primary",
            future_primary_occupancy=(99, 1.0, 2.0),
        ),
        policy_name="primary_only",
    )

    assert run.records[0].final_provider == "api"
    assert primary_service_times.index == 1


def test_fallback_admission_restores_rng_for_rejected_trial_candidate() -> None:
    cheap_concurrency = ConcurrencyState(limit=1)
    cheap_concurrency.admit(request_id=99, now=1.0, service_time_sec=2.0)
    cheap_sc = _provider(
        "cheap_sc",
        cost=1e-7,
        tier=ProviderTier.S_C,
        ttft_dist=Uniform(10.0, 11.0),
        service_time_dist=Uniform(2000.0, 2001.0),
        concurrency=cheap_concurrency,
    )
    api = _provider(
        "api",
        cost=1e-6,
        tier=ProviderTier.S_A,
        ttft_dist=_FixedDistribution(20.0),
    )
    rngs = _rngs([cheap_sc, api])
    cheap_rng_state = copy.deepcopy(rngs["cheap_sc"].bit_generator.state)

    admission = _fallback_admission(
        [cheap_sc, api],
        now=0.0,
        output_tokens=10,
        provider_rngs=rngs,
    )

    assert admission is not None
    assert admission.provider.name == "api"
    assert rngs["cheap_sc"].bit_generator.state == cheap_rng_state


def _provider(
    name: str,
    *,
    cost: float,
    tier: ProviderTier,
    ttft_dist,
    service_time_dist=None,
    concurrency: ConcurrencyState | None = None,
) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=cost,
        input_cost_per_token=cost,
        output_cost_per_token=cost,
        ttft_dist=ttft_dist,
        tps_dist=_FixedDistribution(100.0),
        tier=tier,
        concurrency=concurrency,
        service_time_dist=service_time_dist,
    )


def _rngs(providers: list[TieredProvider]) -> dict[str, np.random.Generator]:
    return {provider.name: np.random.default_rng(17) for provider in providers}
