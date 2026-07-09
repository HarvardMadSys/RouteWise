"""Simulator regressions for cancel-aware hedge capacity accounting."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pytest

from routewise.capacity import ConcurrencyState, ProviderTier, WeightedConcurrencyState
from routewise.metrics import Status
from routewise.schemas import HedgeDispatch, Request, RoutingDecision, RoutingOutcome
from routewise.sim.engine.simulator import Simulator
from routewise.sim.engine.state import SimulationState
from routewise.sim.world.providers import TieredProvider
from routewise.sim.world.scenarios import ScenarioConfig


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


class _StaticHedgePolicy:
    def __init__(
        self,
        *,
        primary_provider: str = "primary",
        backup_provider: str = "backup",
        checkpoint_sec: float = 0.2,
    ) -> None:
        self.primary_provider = primary_provider
        self.backup_provider = backup_provider
        self.checkpoint_sec = checkpoint_sec

    def route(self, request: Request, state: SimulationState) -> RoutingDecision:
        del request, state
        return RoutingDecision(
            primary_provider=self.primary_provider,
            hedge_checkpoints=(self.checkpoint_sec,),
        )

    def tick(
        self,
        request: Request,
        decision: RoutingDecision,
        elapsed: float,
        state: SimulationState,
    ) -> HedgeDispatch | None:
        del request, decision, elapsed, state
        return HedgeDispatch(backup_provider=self.backup_provider)

    def observe(
        self,
        request: Request,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
    ) -> None:
        del request, decision, outcome


def test_backup_winner_cancels_primary_concurrency_occupancy() -> None:
    primary = _provider("primary", ttft_ms=1000.0)
    backup = _provider("backup", ttft_ms=100.0)
    scenario = _scenario([primary, backup])
    requests = [
        _request(1, timestamp=0.0),
        # This arrives after the backup wins request 1 but before request 1's
        # uncanceled primary service interval would have ended.
        _request(2, timestamp=0.4),
    ]

    run = Simulator(scenario, seed=1, dispatch_overhead_ms=5.0).run(
        requests,
        _StaticHedgePolicy(checkpoint_sec=0.2),
        policy_name="static_hedge",
    )

    first, second = run.records
    assert first.hedge_winner == "backup"
    assert first.metadata["hedge_loser"] == "primary"
    assert first.metadata["hedge_loser_canceled"] is True
    assert first.metadata["hedge_cancel_time"] == pytest.approx(0.305)

    primary_interval = _interval_for(primary.concurrency, request_id=1)
    assert primary_interval[1] == pytest.approx(0.305)
    assert second.status == Status.SUCCESS
    assert second.primary_provider == "primary"
    assert second.final_provider == "primary"


def test_primary_winner_cancels_backup_concurrency_occupancy() -> None:
    primary = _provider("primary", ttft_ms=300.0)
    backup = _provider("backup", ttft_ms=1000.0)
    scenario = _scenario([primary, backup])

    run = Simulator(scenario, seed=1, dispatch_overhead_ms=0.0).run(
        [_request(1, timestamp=0.0)],
        _StaticHedgePolicy(checkpoint_sec=0.1),
        policy_name="static_hedge",
    )

    record = run.records[0]
    assert record.hedge_winner == "primary"
    assert record.metadata["hedge_loser"] == "backup"
    assert record.metadata["hedge_loser_canceled"] is True
    assert record.metadata["hedge_cancel_time"] == pytest.approx(0.3)

    backup_interval = _interval_for(backup.concurrency, request_id=10_000_001)
    assert backup_interval[0] == pytest.approx(0.1)
    assert backup_interval[1] == pytest.approx(0.3)


def test_backup_winner_cancels_weighted_primary_concurrency_occupancy() -> None:
    primary = _weighted_provider("primary", ttft_ms=1000.0)
    backup = _weighted_provider("backup", ttft_ms=100.0)
    scenario = _scenario([primary, backup])

    run = Simulator(scenario, seed=1, dispatch_overhead_ms=5.0).run(
        [_request(1, timestamp=0.0)],
        _StaticHedgePolicy(checkpoint_sec=0.2),
        policy_name="static_hedge",
    )

    record = run.records[0]
    assert record.hedge_winner == "backup"
    assert record.metadata["hedge_loser"] == "primary"
    assert record.metadata["hedge_loser_canceled"] is True

    assert isinstance(primary.concurrency, WeightedConcurrencyState)
    primary_interval = _weighted_interval_for(primary.concurrency, request_id=1)
    assert primary_interval[1] == pytest.approx(0.305)
    assert primary.concurrency.used_concurrency_cost(0.2) == 4
    assert primary.concurrency.used_concurrency_cost(primary_interval[1]) == 0
    assert primary.concurrency.total_capacity_unit_seconds_used == pytest.approx(4 * 0.305)


def test_backup_winner_bills_full_backup_plus_canceled_primary_prefill_cost() -> None:
    primary = _api_provider("primary", ttft_ms=1000.0, cost_per_token=1e-6)
    backup = _api_provider("backup", ttft_ms=100.0, cost_per_token=3e-6)
    scenario = _scenario([primary, backup])

    run = Simulator(scenario, seed=1, dispatch_overhead_ms=0.0).run(
        [_request(1, timestamp=0.0)],
        _StaticHedgePolicy(checkpoint_sec=0.2),
        policy_name="static_hedge",
    )

    record = run.records[0]
    assert record.hedge_winner == "backup"
    assert record.total_cost_usd == pytest.approx(430e-6)
    assert record.primary_cost_usd == pytest.approx(100e-6)
    assert record.backup_cost_usd == pytest.approx(330e-6)
    assert record.metadata["primary_cancel_cost_usd"] == pytest.approx(100e-6)
    assert record.metadata["backup_cancel_cost_usd"] == pytest.approx(300e-6)
    assert record.metadata["primary_uncanceled_cost_usd"] == pytest.approx(110e-6)
    assert record.metadata["backup_uncanceled_cost_usd"] == pytest.approx(330e-6)


def test_primary_winner_bills_full_primary_plus_canceled_backup_prefill_cost() -> None:
    primary = _api_provider("primary", ttft_ms=300.0, cost_per_token=1e-6)
    backup = _api_provider("backup", ttft_ms=1000.0, cost_per_token=3e-6)
    scenario = _scenario([primary, backup])

    run = Simulator(scenario, seed=1, dispatch_overhead_ms=0.0).run(
        [_request(1, timestamp=0.0)],
        _StaticHedgePolicy(checkpoint_sec=0.1),
        policy_name="static_hedge",
    )

    record = run.records[0]
    assert record.hedge_winner == "primary"
    assert record.total_cost_usd == pytest.approx(410e-6)
    assert record.primary_cost_usd == pytest.approx(110e-6)
    assert record.backup_cost_usd == pytest.approx(300e-6)
    assert record.metadata["primary_cancel_cost_usd"] == pytest.approx(100e-6)
    assert record.metadata["backup_cancel_cost_usd"] == pytest.approx(300e-6)
    assert record.metadata["primary_uncanceled_cost_usd"] == pytest.approx(110e-6)
    assert record.metadata["backup_uncanceled_cost_usd"] == pytest.approx(330e-6)


def _provider(name: str, *, ttft_ms: float) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=0.0,
        ttft_dist=_FixedDistribution(ttft_ms),
        tps_dist=_FixedDistribution(1.0),
        tier=ProviderTier.S_C,
        concurrency=ConcurrencyState(limit=1),
    )


def _api_provider(name: str, *, ttft_ms: float, cost_per_token: float) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=cost_per_token,
        ttft_dist=_FixedDistribution(ttft_ms),
        tps_dist=_FixedDistribution(1.0),
        tier=ProviderTier.S_A,
    )


def _weighted_provider(name: str, *, ttft_ms: float) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=0.0,
        ttft_dist=_FixedDistribution(ttft_ms),
        tps_dist=_FixedDistribution(1.0),
        tier=ProviderTier.S_C,
        concurrency=WeightedConcurrencyState(
            capacity_units=4,
            model_concurrency_costs_by_class=MappingProxyType({"m": 4}),
            fixed_model_class="m",
        ),
    )


def _scenario(providers: list[TieredProvider]) -> ScenarioConfig:
    return ScenarioConfig(
        name="cancel-aware-hedging",
        description="deterministic cancel-aware hedging regression",
        providers=providers,
        primary_slo_ms=5000.0,
    )


def _request(request_id: int, *, timestamp: float) -> Request:
    return Request(
        id=request_id,
        timestamp=timestamp,
        request_tokens=100,
        response_tokens=10,
        total_tokens=110,
    )


def _interval_for(
    concurrency: ConcurrencyState | None,
    *,
    request_id: int,
) -> tuple[float, float, int]:
    assert concurrency is not None
    return next(entry for entry in concurrency.active if entry[2] == request_id)


def _weighted_interval_for(
    concurrency: WeightedConcurrencyState,
    *,
    request_id: int,
) -> tuple[float, float, int, int, str, int]:
    return next(entry for entry in concurrency.active if entry[3] == request_id)
