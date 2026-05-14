"""Cross-layer consistency tests for effective-cost scarcity formulas."""

from __future__ import annotations

import pytest

from experiments.real_evaluation.inventory import ProviderSpec, ProviderState
from experiments.real_evaluation.shadow_price import (
    concurrency_shadow_price as real_concurrency_shadow_price,
    quota_shadow_price as real_quota_shadow_price,
)
from experiments.real_evaluation.transports import TransportConfig
from rwsim.policies.effective_cost_kernel import scarcity_price
from rwsim.policies.routewise import (
    concurrency_shadow_price as sim_concurrency_shadow_price,
    quota_shadow_price as sim_quota_shadow_price,
)
from rwsim.world.capacity import ConcurrencyState, ProviderTier, QuotaState
from rwsim.world.distributions import Uniform
from rwsim.world.providers import TieredProvider

L = 1e-4
U = 1e-2


@pytest.mark.parametrize(
    ("used", "z"),
    [(0, 0.0), (25, 0.25), (50, 0.5), (75, 0.75), (100, 1.0)],
)
def test_quota_shadow_prices_match_kernel_across_layers(used: int, z: float) -> None:
    now = 10.0
    sim_provider = _sim_quota_provider(used=used)
    real_state = _real_quota_state(used=used)

    kernel_value = scarcity_price("exp_lu", z, L=L, U=U)

    assert sim_quota_shadow_price(sim_provider, now, L=L, U=U) == kernel_value
    assert real_quota_shadow_price(real_state, now, L=L, U=U) == kernel_value


@pytest.mark.parametrize(
    "active",
    [0, 1, 2, 3, 4],
)
def test_concurrency_shadow_prices_match_kernel_across_layers(active: int) -> None:
    now = 10.0
    sim_provider = _sim_concurrency_provider(active=active, now=now)
    real_state = _real_concurrency_state(active=active, now=now)

    # Concurrency is availability-only in both simulator and real-eval; the
    # reusable slot has no marginal shadow price once admission succeeds.
    assert sim_concurrency_shadow_price(sim_provider, now, L=L, U=U) == 0.0
    assert real_concurrency_shadow_price(real_state, now, L=L, U=U) == 0.0


def _sim_quota_provider(*, used: int) -> TieredProvider:
    return TieredProvider(
        name="sim_quota",
        cost_per_token=0.0,
        ttft_dist=_dist(),
        tps_dist=_dist(),
        tier=ProviderTier.S_Q,
        quota=QuotaState(size=100, used=used),
    )


def _sim_concurrency_provider(*, active: int, now: float) -> TieredProvider:
    concurrency = ConcurrencyState(limit=4)
    for request_id in range(active):
        concurrency.admit(request_id, now, 60.0)
    return TieredProvider(
        name="sim_concurrency",
        cost_per_token=0.0,
        ttft_dist=_dist(),
        tps_dist=_dist(),
        tier=ProviderTier.S_C,
        concurrency=concurrency,
    )


def _real_quota_state(*, used: int) -> ProviderState:
    state = ProviderState.from_spec(
        ProviderSpec(
            name="real_quota",
            tier="quota",
            transport_cfg=_transport("real_quota"),
            quota_window_sec=3600.0,
            quota_requests=100,
        )
    )
    assert state.quota is not None
    state.quota.used = used
    return state


def _real_concurrency_state(*, active: int, now: float) -> ProviderState:
    state = ProviderState.from_spec(
        ProviderSpec(
            name="real_concurrency",
            tier="concurrency",
            transport_cfg=_transport("real_concurrency"),
            concurrency_limit=4,
        )
    )
    assert state.concurrency is not None
    for request_id in range(active):
        state.concurrency.admit(request_id, now, 60.0)
    return state


def _transport(name: str) -> TransportConfig:
    return TransportConfig(
        name=name,
        transport="openrouter",
        model="x",
        stream_cancel_billing="stops",
    )


def _dist() -> Uniform:
    return Uniform(100.0, 200.0)
