"""Tests for real-world latency profile loading."""

from __future__ import annotations

import pytest

from experiments.simulation import real_profiles


def test_load_pool_returns_provider_specific_empirical_distributions():
    pool = real_profiles.load_pool("rw3")

    assert tuple(pool) == ("WandB", "DeepInfra", "Novita")
    assert pool["WandB"].label == "qwen3_24h/WandB"
    assert pool["Novita"].p50() > pool["WandB"].p50()


def test_load_pooled_distribution_uses_rw8_source_samples():
    pooled = real_profiles.load_pooled_distribution("rw8_pooled")
    rw8 = real_profiles.load_pool("rw8")

    assert pooled.label == "qwen3_24h/rw8_pooled"
    assert pooled.samples.size == sum(dist.samples.size for dist in rw8.values())


def test_real_profile_loader_rejects_unknown_names():
    with pytest.raises(KeyError, match="unknown real-world profile pool"):
        real_profiles.load_pool("missing")
    with pytest.raises(KeyError, match="unknown pooled real-world profile"):
        real_profiles.load_pooled_distribution("missing")
