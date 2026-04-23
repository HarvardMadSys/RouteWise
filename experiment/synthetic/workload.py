"""Synthetic workload generation.

Generates timestamped request streams with configurable output token
distribution and arrival process (Poisson or deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticRequest:
    """A single synthetic LLM request.

    Attributes:
        id: Monotonic request ID.
        timestamp: Arrival time in seconds from start.
        output_tokens: Number of output tokens (ground truth).
        input_tokens: Number of input tokens (for pricing input cost).
    """

    id: int
    timestamp: float
    output_tokens: int
    input_tokens: int = 200

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def generate_workload(
    n_requests: int,
    duration_sec: float,
    output_token_mu: float = 4.0,
    output_token_sigma: float = 1.0,
    output_token_min: int = 10,
    output_token_max: int = 4000,
    seed: int = 0,
) -> list[SyntheticRequest]:
    """Generate a synthetic Poisson-arrival workload.

    Output tokens follow a LogNormal distribution (typical for LLM responses).

    Args:
        n_requests: Total number of requests.
        duration_sec: Total workload duration in seconds.
        output_token_mu: LogNormal mu parameter for output tokens.
        output_token_sigma: LogNormal sigma parameter for output tokens.
        output_token_min: Minimum allowed output tokens (clipped).
        output_token_max: Maximum allowed output tokens (clipped).
        seed: Random seed for reproducibility.

    Returns:
        Sorted list of SyntheticRequest by timestamp.
    """
    rng = np.random.default_rng(seed)

    # Poisson arrivals via uniform sampling + sort.
    timestamps = np.sort(rng.uniform(0.0, duration_sec, size=n_requests))

    # Output tokens: LogNormal clipped.
    raw = rng.lognormal(output_token_mu, output_token_sigma, size=n_requests)
    tokens = np.clip(raw, output_token_min, output_token_max).astype(int)

    return [
        SyntheticRequest(id=i, timestamp=float(t), output_tokens=int(tok))
        for i, (t, tok) in enumerate(zip(timestamps, tokens))
    ]


def generate_bimodal_workload(
    n_requests: int,
    duration_sec: float,
    short_tokens: int = 50,
    long_tokens: int = 2000,
    long_fraction: float = 0.3,
    seed: int = 0,
) -> list[SyntheticRequest]:
    """Generate a bimodal workload with distinct short and long requests.

    Useful for testing value-density saving: long requests cost more under S_A,
    so a smart scheduler should reserve limited S_Q quota for them.
    """
    rng = np.random.default_rng(seed)
    timestamps = np.sort(rng.uniform(0.0, duration_sec, size=n_requests))

    is_long = rng.uniform(0.0, 1.0, size=n_requests) < long_fraction
    tokens = np.where(is_long, long_tokens, short_tokens).astype(int)

    return [
        SyntheticRequest(id=i, timestamp=float(t), output_tokens=int(tok))
        for i, (t, tok) in enumerate(zip(timestamps, tokens))
    ]
