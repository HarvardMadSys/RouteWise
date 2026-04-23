"""Synthetic provider with configurable latency distribution.

A SyntheticProvider belongs to one of three tiers (S_Q quota, S_C concurrency,
S_A API) and samples TTFT from a log-normal distribution. Cost model follows
the real-world convention: S_A charges per-token; S_Q/S_C have 0 marginal
cost but limited capacity.

Log-normal for TTFT matches measured provider behavior: right-skewed,
heavy-tailed. Parameters can be tuned per-scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from experiment.synthetic.workload import SyntheticRequest


TIER_S_Q = "S_Q"
TIER_S_C = "S_C"
TIER_S_A = "S_A"


@dataclass
class SyntheticProvider:
    """A synthetic provider with known latency and cost model.

    Attributes:
        name: Provider display name.
        tier: One of {TIER_S_Q, TIER_S_C, TIER_S_A}.
        price_per_m_output: USD per 1M output tokens (for S_A; 0 for S_Q/S_C).
        price_per_m_input: USD per 1M input tokens (for S_A; 0 for S_Q/S_C).
        daily_quota: Max requests per day (for S_Q; 0 otherwise).
        concurrency_limit: Max concurrent requests (for S_C; 0 otherwise).
        ttft_mu: Log-normal mu (in log-ms units).
        ttft_sigma: Log-normal sigma.
        tps: Tokens-per-second for e2e latency computation.
        ttft_mu_fn: Optional time-varying mu(t) function (overrides ttft_mu).
    """

    name: str
    tier: str
    price_per_m_output: float = 0.0
    price_per_m_input: float = 0.0
    daily_quota: int = 0
    concurrency_limit: int = 0
    ttft_mu: float = 5.0
    ttft_sigma: float = 0.5
    tps: float = 2000.0
    ttft_mu_fn: Callable[[float], float] | None = field(default=None, repr=False)

    def sample_ttft_ms(
        self, current_time: float, rng: np.random.Generator
    ) -> float:
        """Sample TTFT in milliseconds from the provider's distribution.

        Args:
            current_time: Current simulation time (for time-varying providers).
            rng: Random generator instance for reproducibility.
        """
        mu = self.ttft_mu_fn(current_time) if self.ttft_mu_fn is not None else self.ttft_mu
        return float(rng.lognormal(mu, self.ttft_sigma))

    def marginal_cost(self, request: SyntheticRequest) -> float:
        """Marginal USD cost of sending `request` to this provider.

        S_Q and S_C have zero marginal cost (subscription covers it).
        S_A charges per-token.
        """
        if self.tier == TIER_S_A:
            return (
                request.output_tokens * 1e-6 * self.price_per_m_output
                + request.input_tokens * 1e-6 * self.price_per_m_input
            )
        return 0.0

    @property
    def is_s_q(self) -> bool:
        return self.tier == TIER_S_Q

    @property
    def is_s_c(self) -> bool:
        return self.tier == TIER_S_C

    @property
    def is_s_a(self) -> bool:
        return self.tier == TIER_S_A
