"""Synthetic workload generator producing a stream of Request objects."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path so experiment.data can be imported.
_project_root = str(Path(__file__).resolve().parents[4])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from experiment.data.schema import Request

from .providers import LogNormal


def generate_workload(
    n_requests: int,
    duration_seconds: float,
    seed: int = 42,
    start_time: float = 0.0,
    arrival_process: str = "poisson",
    output_token_dist: LogNormal | None = None,
    input_tokens: int = 100,
) -> list[Request]:
    """Generate a synthetic stream of requests with timestamps.

    Args:
        n_requests: Number of requests to generate.
        duration_seconds: Simulation window in seconds.
        seed: RNG seed for reproducibility.
        start_time: Base Unix-style timestamp (seconds).
        arrival_process: "poisson" (exponential inter-arrivals) or "bursty"
            (periodic high-rate bursts).
        output_token_dist: LogNormal distribution for output tokens.
            Defaults to LogNormal(mu=4.0, sigma=1.0) — P50 ≈ 55 tokens.
        input_tokens: Fixed input token count per request.

    Returns:
        List of Request objects sorted by timestamp, clipped to duration.
    """
    if output_token_dist is None:
        output_token_dist = LogNormal(mu=4.0, sigma=1.0)

    rng = np.random.default_rng(seed)
    rate = n_requests / duration_seconds  # requests per second

    if arrival_process == "poisson":
        inter_arrivals = rng.exponential(1.0 / rate, n_requests)
    elif arrival_process == "bursty":
        # 60% of requests arrive at 3× rate (bursts), 40% in slow gaps.
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

    # Clip to simulation window.
    valid = timestamps <= start_time + duration_seconds
    timestamps = timestamps[valid]
    n = len(timestamps)

    # Sample output token counts, clipped to [1, 4096].
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
