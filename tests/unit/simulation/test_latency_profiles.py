"""Tests for real-world latency profile loading."""

from __future__ import annotations

import pytest

from experiments.simulation import latency_profiles


def test_load_pool_returns_provider_specific_empirical_distributions():
    pool = latency_profiles.load_pool("rw3")

    assert tuple(pool) == ("WandB", "DeepInfra", "Novita")
    assert pool["WandB"].label == "qwen3_24h/WandB"
    assert pool["Novita"].p50() > pool["WandB"].p50()


def test_load_pooled_distribution_uses_rw8_source_samples():
    pooled = latency_profiles.load_pooled_distribution("rw8_pooled")
    rw8 = latency_profiles.load_pool("rw8")

    assert pooled.label == "qwen3_24h/rw8_pooled"
    assert pooled.samples.size == sum(dist.samples.size for dist in rw8.values())


def test_load_empirical_subscription_profile_distributions():
    profile = latency_profiles.load_empirical_profile("minimax_m25_subscriptions")

    assert tuple(profile) == ("chutes", "featherless", "minimax")
    assert profile["chutes"].label == "minimax_m25_subscriptions/chutes"
    assert profile["chutes"].samples.size == 210
    assert profile["featherless"].samples.size == 296
    assert profile["minimax"].samples.size == 506
    assert profile["chutes"].p50() == pytest.approx(1210.2729224134237)
    assert profile["featherless"].p50() == pytest.approx(8157.45)
    assert profile["minimax"].p50() == pytest.approx(2506.0)


def test_load_empirical_distribution_rejects_missing_provider():
    with pytest.raises(KeyError, match="not found"):
        latency_profiles.load_empirical_distribution(
            "minimax_m25_subscriptions",
            "missing",
        )


def test_latency_profile_loader_rejects_unknown_names():
    with pytest.raises(KeyError, match="unknown real-world profile pool"):
        latency_profiles.load_pool("missing")
    with pytest.raises(KeyError, match="unknown pooled real-world profile"):
        latency_profiles.load_pooled_distribution("missing")
