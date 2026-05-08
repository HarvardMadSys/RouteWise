"""Tests for real-world latency profile loading."""

from __future__ import annotations

import pytest

from experiments.simulation import latency_profiles
from experiments.simulation.provider_profiles import load_provider_pool

MINIMAX_M25_RW8 = (
    "Inceptron",
    "Friendli",
    "DeepInfra",
    "SambaNova",
    "Venice",
    "AtlasCloud",
    "Chutes",
    "SiliconFlow",
)


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


def test_load_minimax_m25_rw8_pool_uses_pool_artifact():
    pool = latency_profiles.load_pool("minimax_m25_rw8")

    assert tuple(pool) == MINIMAX_M25_RW8
    assert pool["Inceptron"].label == "minimax_m25_openrouter_24h/Inceptron"
    assert pool["SiliconFlow"].p50() > pool["Inceptron"].p50()


def test_load_minimax_m25_provider_pool_uses_metadata_prices():
    pool = load_provider_pool("minimax_m25_rw8")

    assert pool.profile_name == "minimax_m25_openrouter"
    assert pool.price_source == "metadata_openrouter_price"
    assert tuple(provider.name for provider in pool.providers) == MINIMAX_M25_RW8
    assert pool.by_name()["Inceptron"].input_per_m == pytest.approx(0.24)
    assert pool.by_name()["Inceptron"].output_per_m == pytest.approx(0.9)
    assert pool.by_name()["Chutes"].input_per_m == pytest.approx(0.15)
    assert pool.by_name()["Chutes"].output_per_m == pytest.approx(1.2)


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


def test_load_empirical_minimax_openrouter_profile_distributions():
    metadata = latency_profiles.load_empirical_profile_metadata(
        "minimax_m25_openrouter_24h"
    )
    profile = latency_profiles.load_empirical_profile("minimax_m25_openrouter_24h")

    assert metadata["run_stats"]["is_full_day_observation"] is False
    assert metadata["run_stats"]["observed_duration_hours"] == pytest.approx(
        2.816319104300605
    )
    assert tuple(profile) == tuple(metadata["providers"])
    assert profile["Inceptron"].samples.size == 50_000
    assert profile["SiliconFlow"].p50() > profile["Inceptron"].p50()
    assert metadata["providers"]["Inceptron"]["openrouter_price"][
        "output_price_per_m"
    ] == pytest.approx(0.9)
    assert "openrouter_price" not in metadata["providers"]["Together"]


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
