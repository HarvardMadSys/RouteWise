"""Synthetic workload generator producing a stream of Request objects."""

from __future__ import annotations

import numpy as np

from rwsim.schemas import Request
from rwsim.world.distributions import LogNormal


def generate_workload(
    n_requests: int,
    duration_seconds: float,
    seed: int = 42,
    start_time: float = 0.0,
    arrival_process: str = "poisson",
    output_token_dist: LogNormal | None = None,
    input_tokens: int = 100,
) -> list[Request]:
    """Generate a synthetic stream of requests sorted by timestamp."""
    if output_token_dist is None:
        output_token_dist = LogNormal(mu=4.0, sigma=1.0)

    rng = np.random.default_rng(seed)
    rate = n_requests / duration_seconds

    if arrival_process == "poisson":
        inter_arrivals = rng.exponential(1.0 / rate, n_requests)
    elif arrival_process == "bursty":
        high_rate = rate * 3.0
        low_rate = rate * 0.25
        mask = rng.random(n_requests) < 0.6
        inter_arrivals = np.where(
            mask,
            rng.exponential(1.0 / high_rate, n_requests),
            rng.exponential(1.0 / low_rate, n_requests),
        )
    else:
        raise ValueError(f"Unknown arrival_process: {arrival_process!r}")

    timestamps = np.cumsum(inter_arrivals) + start_time
    valid = timestamps <= start_time + duration_seconds
    timestamps = timestamps[valid]
    n = len(timestamps)

    raw = rng.lognormal(output_token_dist.mu, output_token_dist.sigma, n)
    output_tokens = np.clip(raw, 1, 4096).astype(int)

    requests: list[Request] = []
    for i in range(n):
        resp = int(output_tokens[i])
        requests.append(
            Request(
                id=i,
                timestamp=int(timestamps[i]),
                request_tokens=input_tokens,
                response_tokens=resp,
                total_tokens=input_tokens + resp,
            )
        )
    return requests


__all__ = ["generate_workload"]
