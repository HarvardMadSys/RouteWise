"""LP-only policy for the effective-cost formula ablation.

This module is intentionally experiment-scoped. It duplicates the small
RouteWise cost-router LP enumerator so formula sweeps do not add hooks or
fields to the production RouteWisePolicy surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rwsim.core.lp import (
    LP_EPS,
    cost_tiebroken_objective,
    normalize_weights,
    solve_simplex_lp,
)
from rwsim.policies.base import NoOpTickMixin
from rwsim.policies.effective_cost_kernel import ScarcityCurve, scarcity_price
from rwsim.policies.routewise import RollingLatencyProfile
from rwsim.schemas import Request, RoutingDecision, RoutingOutcome
from rwsim.world.capacity import ProviderTier

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rwsim.engine.state import SimulationState
    from rwsim.world.providers import Provider

_LP_EPS = LP_EPS


@dataclass
class LPOnlyAblationPolicy(NoOpTickMixin):
    """Cost-router-only ablation policy with experiment-scoped curves."""

    quota_curve: ScarcityCurve
    concurrency_curve: ScarcityCurve
    p: float
    cost_envelope: tuple[float, float]
    seed: int = 0
    profile_window_sec: float = 15 * 60.0
    rng: np.random.Generator = field(init=False, repr=False)
    profiles: dict[str, RollingLatencyProfile] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"LPOnlyAblationPolicy p must be in [0, 1], got {self.p}")
        L, U = (float(self.cost_envelope[0]), float(self.cost_envelope[1]))
        if not (math.isfinite(L) and math.isfinite(U) and 0.0 < L < U):
            raise ValueError(
                f"LPOnlyAblationPolicy cost_envelope must satisfy 0 < L < U; got L={L}, U={U}"
            )
        self.cost_envelope = (L, U)
        self.rng = np.random.default_rng(self.seed)

    def route(self, request: Request, state: SimulationState) -> RoutingDecision:
        """Solve the LP-only cost-router decision and return a primary provider."""
        self._ensure_profiles(state.providers)
        providers = [
            provider for provider in state.providers.values() if provider.is_available(state.now)
        ]
        if not providers:
            providers = list(state.providers.values())
        if not providers:
            raise ValueError("No providers configured for LPOnlyAblationPolicy.")

        L, U = self.cost_envelope
        c_eff = {
            provider.name: self.effective_cost_for_provider(
                provider,
                request,
                state.now,
                L=L,
                U=U,
            )
            for provider in providers
        }
        tbar = {
            provider.name: self._latency_objective_ms(provider, state.now) for provider in providers
        }

        names = [provider.name for provider in providers]
        c_min = min(c_eff.values())
        c_max = max(c_eff.values())
        budget = c_min + self.p * (c_max - c_min)

        latency_objective = [tbar[name] for name in names]
        effective_costs = [c_eff[name] for name in names]
        if self.p <= _LP_EPS:
            weights = _p_zero_weights(
                names,
                latency_objective=latency_objective,
                effective_costs=effective_costs,
                budget=budget,
            )
        else:
            success, vector = solve_simplex_lp(
                objective=cost_tiebroken_objective(
                    latency_objective,
                    effective_costs,
                ),
                upper_constraint=effective_costs,
                upper_bound=budget,
            )
            weights = normalize_weights(names, vector) if success and vector is not None else {}
        if not weights:
            best = min(
                providers,
                key=lambda provider: (c_eff[provider.name], tbar[provider.name]),
            )
            weights = {best.name: 1.0}

        return RoutingDecision(
            primary_provider=_sample_weighted(weights, self.rng),
            hedge_checkpoints=(),
            metadata={
                "weights": weights,
                "c_eff": c_eff,
                "budget": budget,
                "L": L,
                "U": U,
                "quota_curve": self.quota_curve,
                "concurrency_curve": self.concurrency_curve,
                "p": self.p,
            },
        )

    def effective_cost_for_provider(
        self,
        provider: Provider,
        request: Request,
        now: float,
        *,
        L: float,
        U: float,
    ) -> float:
        """Return candidate piecewise effective cost for one provider."""
        if provider.tier == ProviderTier.S_A:
            return provider.marginal_cost_for_request(request, now)
        if provider.tier == ProviderTier.S_Q:
            if provider.quota is None:
                return 0.0
            return scarcity_price(
                self.quota_curve,
                provider.quota.fraction_used(now),
                L=L,
                U=U,
            )
        if provider.tier == ProviderTier.S_C:
            if provider.concurrency is None:
                return 0.0
            return scarcity_price(
                self.concurrency_curve,
                provider.concurrency.utilization(now),
                L=L,
                U=U,
            )
        raise ValueError(
            f"unsupported provider tier for effective-cost ablation: {provider.tier!r}"
        )

    def observe(
        self,
        request: Request,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
    ) -> None:
        """Feed observed TTFT samples into the LP-only latency objective."""
        del request
        self._profile(decision.primary_provider).add_sample(
            float(outcome.metadata.get("primary_observed_at", 0.0)),
            float(outcome.metadata.get("primary_ttft_ms", outcome.ttft_ms)),
        )

    def _ensure_profiles(self, providers: Mapping[str, Provider]) -> None:
        for name in providers:
            self._profile(name)

    def _profile(self, name: str) -> RollingLatencyProfile:
        return self.profiles.setdefault(
            name,
            RollingLatencyProfile(window_sec=self.profile_window_sec),
        )

    def _latency_objective_ms(self, provider: Provider, now: float) -> float:
        mean = self._profile(provider.name).mean(now)
        if mean is not None:
            return mean
        return provider.true_mean_ms(now)


def _p_zero_weights(
    names: list[str],
    *,
    latency_objective: list[float],
    effective_costs: list[float],
    budget: float,
) -> dict[str, float]:
    """Return the p=0 LP optimum without running the generic enumerator."""
    objective = cost_tiebroken_objective(latency_objective, effective_costs)
    best_key: tuple[float, float, int, tuple[float, ...]] | None = None
    best_name: str | None = None
    n = len(names)
    for index, (name, cost) in enumerate(zip(names, effective_costs, strict=True)):
        if cost > budget + _LP_EPS:
            continue
        rounded = tuple(1.0 if position == index else 0.0 for position in range(n))
        key = (float(objective[index]), float(cost), 1, rounded)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
    return {best_name: 1.0} if best_name is not None else {}


def _sample_weighted(weights: dict[str, float], rng: np.random.Generator) -> str:
    names = list(weights)
    probs = np.asarray([weights[name] for name in names], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


__all__ = [
    "LPOnlyAblationPolicy",
    "RollingLatencyProfile",
]
