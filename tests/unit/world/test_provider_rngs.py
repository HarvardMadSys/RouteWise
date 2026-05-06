"""Tests for provider-scoped simulator randomness."""

from __future__ import annotations

import numpy as np

from rwsim.engine.simulator import _provider_rngs
from rwsim.world.capacity import ProviderTier
from rwsim.world.distributions import Uniform
from rwsim.world.providers import TieredProvider


def _provider(name: str) -> TieredProvider:
    return TieredProvider(
        name=name,
        cost_per_token=1e-6,
        ttft_dist=Uniform(1.0, 2.0),
        tps_dist=Uniform(1.0, 2.0),
        tier=ProviderTier.S_A,
    )


def test_provider_rng_streams_are_reproducible_and_provider_scoped():
    providers = [_provider("api_cheap"), _provider("api_mid")]

    first = _provider_rngs(42, providers)
    second = _provider_rngs(42, providers)
    reversed_order = _provider_rngs(42, list(reversed(providers)))

    cheap_draws = first["api_cheap"].random(4)
    mid_draws = first["api_mid"].random(4)

    assert np.array_equal(cheap_draws, second["api_cheap"].random(4))
    assert np.array_equal(cheap_draws, reversed_order["api_cheap"].random(4))
    assert not np.array_equal(cheap_draws, mid_draws)
