"""Synthetic simulator.

Runs a routing strategy on a synthetic workload, sampling latencies from
the providers' distributions. Supports hedging (dual-dispatch with backup).
Records per-request outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from experiment.synthetic.provider import SyntheticProvider
from experiment.synthetic.state import SimState
from experiment.synthetic.strategies import RouteDecision, Strategy
from experiment.synthetic.workload import SyntheticRequest


SECONDS_PER_DAY = 86400.0


@dataclass
class RequestOutcome:
    """Outcome of a single routed request."""

    request_id: int
    timestamp: float
    primary: str
    backup: str | None
    actual_provider: str  # which provider's response was used
    hedged: bool
    ttft_ms: float  # final TTFT (min of primary/backup if hedged and both succeeded)
    primary_ttft_ms: float
    backup_ttft_ms: float | None
    cost_usd: float  # billed cost (both if hedged)
    slo_violated: bool


@dataclass
class SimResult:
    """Aggregate simulation result."""

    strategy_name: str
    scenario_name: str
    outcomes: list[RequestOutcome]

    def summary(self, slo_ms: float) -> dict:
        """Return key aggregate metrics."""
        if not self.outcomes:
            return {
                "strategy": self.strategy_name,
                "scenario": self.scenario_name,
                "n": 0,
            }
        ttfts = np.array([o.ttft_ms for o in self.outcomes])
        costs = np.array([o.cost_usd for o in self.outcomes])
        violations = np.array([1.0 if o.slo_violated else 0.0 for o in self.outcomes])
        hedged = np.array([1.0 if o.hedged else 0.0 for o in self.outcomes])

        # Provider distribution.
        from collections import Counter

        prov_counts = Counter(o.actual_provider for o in self.outcomes)
        prov_dist = {k: v / len(self.outcomes) for k, v in prov_counts.items()}

        return {
            "strategy": self.strategy_name,
            "scenario": self.scenario_name,
            "n": len(self.outcomes),
            "total_cost": float(costs.sum()),
            "mean_cost_usd": float(costs.mean()),
            "p50_ttft_ms": float(np.percentile(ttfts, 50)),
            "p90_ttft_ms": float(np.percentile(ttfts, 90)),
            "p99_ttft_ms": float(np.percentile(ttfts, 99)),
            "mean_ttft_ms": float(ttfts.mean()),
            "slo_violation_rate": float(violations.mean()),
            "hedge_rate": float(hedged.mean()),
            "provider_distribution": prov_dist,
        }


def simulate(
    strategy: Strategy,
    requests: list[SyntheticRequest],
    providers: list[SyntheticProvider],
    slo_ms: float,
    seed: int = 0,
    scenario_name: str = "",
) -> SimResult:
    """Run one strategy on one workload.

    Model assumptions:
      - Sampled TTFT is the intrinsic provider latency (no queueing delay
        modeled for S_A).
      - For S_C, a request can only be dispatched if there is a free slot;
        otherwise the strategy should avoid it. We do NOT implement queueing
        inside S_C here. If the strategy routes to a saturated S_C, we
        still serve the request but the concurrency fraction will reach 1.
      - Hedging: both primary and backup are dispatched if primary's sampled
        TTFT > hedge_after_ms. Whichever responds first is "used". Both
        are billed (charge the full cost of each dispatch).
      - Finish time for a dispatch = dispatch_time + ttft_ms / 1000 +
        output_tokens / tps. Used to free S_C slots.
    """
    rng = np.random.default_rng(seed)
    state = SimState.initialize(providers)
    provider_by_name = {p.name: p for p in providers}
    outcomes: list[RequestOutcome] = []

    for req in requests:
        t = req.timestamp
        state.advance_time(t, providers, SECONDS_PER_DAY)

        decision: RouteDecision = strategy.route(req, state)
        primary = provider_by_name[decision.primary]
        backup = provider_by_name.get(decision.backup) if decision.backup else None

        # Sample primary TTFT.
        primary_ttft_ms = primary.sample_ttft_ms(t, rng)
        primary_finish = t + primary_ttft_ms / 1000.0 + req.output_tokens / primary.tps
        primary_cost = primary.marginal_cost(req)
        state.register_dispatch(primary, primary_finish)
        state.profiles[primary.name].add(t, primary_ttft_ms)

        # Decide hedge based on decision.
        hedged = False
        backup_ttft_ms = None
        backup_cost = 0.0

        actual_ttft_ms = primary_ttft_ms
        actual_provider = primary.name

        if (
            backup is not None
            and decision.hedge_after_ms is not None
            and primary_ttft_ms > decision.hedge_after_ms
            and state.can_accept(backup)
        ):
            # Dispatch backup at hedge_time.
            hedge_time = t + decision.hedge_after_ms / 1000.0
            backup_ttft_ms = backup.sample_ttft_ms(hedge_time, rng)
            backup_finish = (
                hedge_time + backup_ttft_ms / 1000.0 + req.output_tokens / backup.tps
            )
            backup_cost = backup.marginal_cost(req)
            state.register_dispatch(backup, backup_finish)
            state.profiles[backup.name].add(hedge_time, backup_ttft_ms)
            hedged = True

            # Winner = whoever finishes TTFT first in wall-clock time.
            primary_first_token_time = t + primary_ttft_ms / 1000.0
            backup_first_token_time = hedge_time + backup_ttft_ms / 1000.0
            if backup_first_token_time < primary_first_token_time:
                actual_ttft_ms = (backup_first_token_time - t) * 1000.0
                actual_provider = backup.name

        total_cost = primary_cost + backup_cost
        slo_violated = actual_ttft_ms > slo_ms

        outcomes.append(
            RequestOutcome(
                request_id=req.id,
                timestamp=t,
                primary=primary.name,
                backup=backup.name if backup else None,
                actual_provider=actual_provider,
                hedged=hedged,
                ttft_ms=actual_ttft_ms,
                primary_ttft_ms=primary_ttft_ms,
                backup_ttft_ms=backup_ttft_ms,
                cost_usd=total_cost,
                slo_violated=slo_violated,
            )
        )

    return SimResult(
        strategy_name=strategy.name,
        scenario_name=scenario_name,
        outcomes=outcomes,
    )
