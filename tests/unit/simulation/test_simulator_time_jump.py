"""Simulator-level regression for non-monotonic time queries.

The simulator's hedge tick loop temporarily advances ``state.now`` to a
future checkpoint and then reverts to outer trace time. Capacity state
reads must remain side-effect-free under this pattern, and the engine's
``gc_before(prev_trace_time)`` watermark must avoid dropping intervals
that subsequent earlier-time queries still need.

These tests exercise the full Simulator.run path with S_C providers
and probability-target hedging, covering both plain ConcurrencyState
and WeightedConcurrencyState.
"""

from __future__ import annotations

from types import MappingProxyType

from rwsim.engine.simulator import Simulator
from rwsim.policies.routewise import RouteWisePolicy
from rwsim.schemas import Request
from rwsim.world.capacity import (
    ConcurrencyState,
    ProviderTier,
    WeightedConcurrencyState,
)
from rwsim.world.distributions import Uniform
from rwsim.world.providers import TieredProvider
from rwsim.world.scenarios import ScenarioConfig


def _trace(count: int, interval_sec: float = 0.1) -> list[Request]:
    return [
        Request(
            id=index,
            timestamp=float(index) * interval_sec,
            request_tokens=100,
            response_tokens=50,
            total_tokens=150,
        )
        for index in range(count)
    ]


def _s_c_provider(
    name: str,
    *,
    ttft_ms: tuple[float, float],
    cost: float,
    limit: int = 8,
) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=cost,
        ttft_dist=Uniform(*ttft_ms),
        tps_dist=Uniform(80.0, 120.0),
        tier=ProviderTier.S_C,
        concurrency=ConcurrencyState(limit=limit),
    )


def _weighted_s_c_provider(
    name: str,
    *,
    ttft_ms: tuple[float, float],
    cost: float,
    capacity_units: int = 32,
    model_class: str = "m",
    class_cost: int = 4,
) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=cost,
        ttft_dist=Uniform(*ttft_ms),
        tps_dist=Uniform(80.0, 120.0),
        tier=ProviderTier.S_C,
        concurrency=WeightedConcurrencyState(
            capacity_units=capacity_units,
            model_concurrency_costs_by_class=MappingProxyType({model_class: class_cost}),
            fixed_model_class=model_class,
        ),
    )


def _routewise_with_hedging(slo_ms: float = 2000.0) -> RouteWisePolicy:
    return RouteWisePolicy(
        hedging="probability_target",
        explorer=False,
        p=0.75,
        seed=17,
        slo_ms=slo_ms,
        cost_envelope=(1e-6, 1e-3),
    )


def test_simulator_runs_cleanly_with_plain_s_c_concurrency_and_hedging() -> None:
    """Smoke test: hedge tick advances state.now to future checkpoints
    on a plain-ConcurrencyState provider; pure read APIs and gc_before
    must not corrupt the ledger across outer trace iterations."""
    scenario = ScenarioConfig(
        name="plain_sc_hedging_smoke",
        description="plain S_C providers with hedging-induced time jumps",
        providers=[
            _s_c_provider("primary_slow", ttft_ms=(1800.0, 2200.0), cost=1e-6),
            _s_c_provider("backup_fast", ttft_ms=(200.0, 400.0), cost=4e-6),
        ],
        primary_slo_ms=2000.0,
    )
    requests = _trace(count=20, interval_sec=0.1)

    run = Simulator(scenario=scenario, seed=23).run(
        requests,
        _routewise_with_hedging(),
        policy_name="routewise",
    )

    # Sanity: simulator produced one record per trace request.
    assert len(run.records) == len(requests)

    # Pure-read invariant survives: every active entry is a 3-tuple of the
    # post-refactor shape (start, end, request_id).
    for provider in scenario.providers:
        if provider.concurrency is None:
            continue
        for entry in provider.concurrency.active:
            assert len(entry) == 3
            start, end, _ = entry
            assert start <= end


def test_simulator_runs_cleanly_with_weighted_s_c_and_hedging() -> None:
    """Smoke test: same as above but on WeightedConcurrencyState. The
    previous _current_used_cost counter would over-count weighted
    capacity at earlier times when a hedge backup admit happened during
    the hedge tick loop's future-checkpoint window."""
    scenario = ScenarioConfig(
        name="weighted_sc_hedging_smoke",
        description="weighted S_C providers with hedging-induced time jumps",
        providers=[
            _weighted_s_c_provider(
                "primary_slow",
                ttft_ms=(1800.0, 2200.0),
                cost=1e-6,
                capacity_units=32,
                class_cost=4,
            ),
            _weighted_s_c_provider(
                "backup_fast",
                ttft_ms=(200.0, 400.0),
                cost=4e-6,
                capacity_units=32,
                class_cost=4,
            ),
        ],
        primary_slo_ms=2000.0,
    )
    requests = _trace(count=20, interval_sec=0.1)

    run = Simulator(scenario=scenario, seed=23).run(
        requests,
        _routewise_with_hedging(),
        policy_name="routewise",
    )

    assert len(run.records) == len(requests)

    # Every weighted active entry is a 6-tuple
    # (start, finish, seq, request_id, mc, cost) with start < finish,
    # confirming no legacy heap shape leaked in.
    for provider in scenario.providers:
        if provider.concurrency is None:
            continue
        assert isinstance(provider.concurrency, WeightedConcurrencyState)
        for entry in provider.concurrency.active:
            assert len(entry) == 6
            start, finish, _, request_id, _, cost = entry
            assert start < finish
            assert request_id >= 0
            assert cost > 0


def test_simulator_gc_keeps_ledger_bounded_across_long_trace() -> None:
    """The engine calls gc_before(prev_trace_time) once per outer
    iteration, so finished intervals are dropped and the ledger does
    not grow with the trace length. After a long monotonic trace,
    surviving entries must end strictly after the previous trace
    timestamp; otherwise the GC hook is silently broken."""
    scenario = ScenarioConfig(
        name="ledger_gc_smoke",
        description="long trace verifies gc_before integration",
        providers=[
            _s_c_provider("primary", ttft_ms=(800.0, 1200.0), cost=1e-6),
            _s_c_provider("backup", ttft_ms=(100.0, 200.0), cost=4e-6),
        ],
        primary_slo_ms=2000.0,
    )
    # 100 requests at 0.1s spacing; service time ~ 1-2s, so most intervals
    # from early requests should be GC'd by the time the last request runs.
    requests = _trace(count=100, interval_sec=0.1)

    Simulator(scenario=scenario, seed=29).run(
        requests,
        _routewise_with_hedging(),
        policy_name="routewise",
    )

    # gc_before(prev_trace_time) was called before the last iteration with
    # watermark = (count-2)*interval_sec. Surviving entries must have
    # end > watermark; the engine should never have admitted intervals
    # with end <= watermark and left them in the ledger.
    watermark = float(len(requests) - 2) * 0.1
    for provider in scenario.providers:
        if provider.concurrency is None:
            continue
        for entry in provider.concurrency.active:
            _, end, _ = entry
            assert end > watermark, (
                f"provider {provider.name!r} ledger entry with end={end} "
                f"should have been GC'd at watermark={watermark}"
            )
