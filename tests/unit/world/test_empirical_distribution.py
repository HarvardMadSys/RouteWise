"""Tests for empirical latency distributions and profile artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from rwsim.world.empirical import EmpiricalDistribution

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = REPO_ROOT / "experiments" / "simulation" / "latency_profiles"
QWEN3_NPZ = PROFILE_DIR / "qwen3_24h.npz"
QWEN3_META = PROFILE_DIR / "qwen3_24h.json"
POOLS_YAML = PROFILE_DIR / "pools.yaml"

RW3 = ("WandB", "DeepInfra", "Novita")
RW8 = (
    "WandB",
    "DeepInfra",
    "Google",
    "Alibaba",
    "Novita",
    "Cerebras",
    "SiliconFlow",
    "AtlasCloud",
)
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


def _metadata() -> dict:
    return json.loads(QWEN3_META.read_text())


def test_from_npz_loads_provider_distribution_matching_metadata():
    metadata = _metadata()["providers"]["WandB"]

    dist = EmpiricalDistribution.from_npz(QWEN3_NPZ, "WandB")

    assert dist.label == "qwen3_24h/WandB"
    assert dist.samples.size == metadata["n_samples"]
    assert dist.p50() == pytest.approx(metadata["p50_ms"])
    assert dist.p99() == pytest.approx(metadata["p99_ms"])
    assert 0.0 < dist.cdf(dist.p50()) < 1.0


def test_pooled_from_npz_concatenates_provider_samples():
    metadata = _metadata()["providers"]
    expected_count = sum(metadata[name]["n_samples"] for name in RW8)

    dist = EmpiricalDistribution.pooled_from_npz(QWEN3_NPZ, RW8, label="rw8_pooled")

    assert dist.label == "qwen3_24h/rw8_pooled"
    assert dist.samples.size == expected_count
    assert dist.p50() == pytest.approx(float(np.percentile(dist.samples, 50)))
    assert dist.p99() == pytest.approx(float(np.percentile(dist.samples, 99)))


def test_profile_pools_yaml_locks_canonical_rw3_rw8_and_pooled_sets():
    config = yaml.safe_load(POOLS_YAML.read_text())

    assert config["schema_version"] == 2
    assert config["profiles"]["qwen3_24h"]["artifact"] == "qwen3_24h.npz"
    assert config["profiles"]["minimax_m25_openrouter"]["artifact"] == (
        "minimax_m25_openrouter_24h.npz"
    )
    assert config["pools"]["rw3"]["profile"] == "qwen3_24h"
    assert tuple(config["pools"]["rw3"]["providers"]) == RW3
    assert config["pools"]["rw8"]["profile"] == "qwen3_24h"
    assert tuple(config["pools"]["rw8"]["providers"]) == RW8
    assert config["pools"]["minimax_m25_rw8"]["profile"] == "minimax_m25_openrouter"
    assert tuple(config["pools"]["minimax_m25_rw8"]["providers"]) == MINIMAX_M25_RW8
    assert config["pooled"]["rw8_pooled"]["profile"] == "qwen3_24h"
    assert tuple(config["pooled"]["rw8_pooled"]["source_providers"]) == RW8


def test_sample_draws_from_empirical_support():
    rng = np.random.default_rng(123)
    dist = EmpiricalDistribution.from_npz(QWEN3_NPZ, "DeepInfra")

    samples = dist.sample(rng, size=128)

    assert samples.shape == (128,)
    assert set(samples).issubset(set(dist.samples))


def test_mean_std_and_quantile_match_numpy_reference():
    """Lock cached moments and indexed-interp quantile to NumPy's defaults.

    The Phase 1a perf fix replaces ``np.percentile`` with sorted-array linear
    interpolation and caches mean/std in ``__post_init__``. This test pins
    the equivalence so future micro-optimisations don't drift away from
    ``np.mean`` / ``np.std`` / ``np.percentile(..., method='linear')``.
    """
    rng = np.random.default_rng(0)
    samples = rng.exponential(scale=300.0, size=10_000)
    dist = EmpiricalDistribution(samples)

    assert dist.mean() == pytest.approx(float(np.mean(samples)), rel=1e-12)
    assert dist.std() == pytest.approx(float(np.std(samples)), rel=1e-12)

    for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        assert dist.quantile(q) == pytest.approx(
            float(np.percentile(samples, q * 100.0)),
            rel=1e-12,
        )


def test_empirical_distribution_rejects_bad_samples_and_missing_provider():
    with pytest.raises(ValueError, match="non-empty"):
        EmpiricalDistribution(np.array([]))
    with pytest.raises(ValueError, match="finite"):
        EmpiricalDistribution(np.array([1.0, float("nan")]))
    with pytest.raises(ValueError, match="non-negative"):
        EmpiricalDistribution(np.array([1.0, -1.0]))
    with pytest.raises(KeyError, match="not found"):
        EmpiricalDistribution.from_npz(QWEN3_NPZ, "DoesNotExist")
