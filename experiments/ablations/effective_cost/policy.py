"""LP-only policy for the effective-cost formula ablation.

This module is intentionally experiment-scoped. It duplicates the small
RouteWise cost-router LP enumerator so formula sweeps do not add hooks or
fields to the production RouteWisePolicy surface.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from experiments.ablations.effective_cost.curves import ScarcityCurve, scarcity_price
from rwsim.policies.base import NoOpTickMixin
from rwsim.schemas import Request, RoutingDecision, RoutingOutcome
from rwsim.world.capacity import ProviderTier

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rwsim.engine.state import SimulationState
    from rwsim.world.providers import Provider

_LP_EPS = 1e-9
_COST_TIEBREAK_MS = 1e-3


@dataclass
class RollingLatencyProfile:
    """Causal moving-window empirical latency profile for one provider."""

    window_sec: float = 15 * 60.0
    samples: deque[tuple[float, float]] = field(default_factory=deque)

    def add_sample(self, timestamp: float, ttft_ms: float) -> None:
        self.samples.append((float(timestamp), float(ttft_ms)))

    def values(self, now: float) -> list[float]:
        cutoff = now - self.window_sec
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        return [ttft_ms for timestamp, ttft_ms in self.samples if timestamp <= now]

    def mean(self, now: float) -> float | None:
        values = self.values(now)
        if not values:
            return None
        return float(np.mean(values))


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

        success, vector = _solve_lp(
            objective=_cost_tiebroken_objective(
                [tbar[name] for name in names],
                [c_eff[name] for name in names],
            ),
            upper_constraint=[c_eff[name] for name in names],
            upper_bound=budget,
        )
        weights = _normalize_weights(names, vector) if success and vector is not None else {}
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


def _solve_lp(
    objective: list[float],
    *,
    upper_constraint: list[float],
    upper_bound: float,
) -> tuple[bool, np.ndarray | None]:
    """Solve the RouteWise simplex LP using the production enumerator."""
    n = len(objective)
    if n == 0 or n != len(upper_constraint):
        return False, None

    obj = np.asarray(objective, dtype=float)
    costs = np.asarray(upper_constraint, dtype=float)
    if not np.all(np.isfinite(obj)) or not np.all(np.isfinite(costs)):
        return False, None

    best_vector: np.ndarray | None = None
    best_key: tuple[float, float, int, tuple[float, ...]] | None = None

    def consider(vector: np.ndarray) -> None:
        nonlocal best_key, best_vector
        expected_cost = float(np.dot(costs, vector))
        if expected_cost > upper_bound + _LP_EPS:
            return
        value = float(np.dot(obj, vector))
        support = int(np.count_nonzero(vector > _LP_EPS))
        rounded = tuple(round(float(part), 12) for part in vector)
        key = (value, expected_cost, support, rounded)
        if best_key is None or key < best_key:
            best_key = key
            best_vector = vector

    for index, cost in enumerate(costs):
        if cost <= upper_bound + _LP_EPS:
            vector = np.zeros(n, dtype=float)
            vector[index] = 1.0
            consider(vector)

    for left in range(n):
        left_cost = float(costs[left])
        for right in range(left + 1, n):
            right_cost = float(costs[right])
            span = left_cost - right_cost
            if abs(span) <= _LP_EPS:
                continue
            left_weight = (upper_bound - right_cost) / span
            right_weight = 1.0 - left_weight
            if left_weight < -_LP_EPS or right_weight < -_LP_EPS:
                continue
            left_weight = min(max(left_weight, 0.0), 1.0)
            right_weight = min(max(right_weight, 0.0), 1.0)
            vector = np.zeros(n, dtype=float)
            vector[left] = left_weight
            vector[right] = right_weight
            consider(vector)

    if best_vector is None:
        return False, None
    return True, best_vector


def _cost_tiebroken_objective(
    latency_objective_ms: list[float],
    effective_costs: list[float],
) -> list[float]:
    """Prefer lower effective cost when LP latency objectives are equal."""
    if len(latency_objective_ms) != len(effective_costs):
        raise ValueError("latency objective and cost arrays must have the same length")
    if not latency_objective_ms:
        return []

    latencies = np.asarray(latency_objective_ms, dtype=float)
    costs = np.asarray(effective_costs, dtype=float)
    cost_span = float(costs.max() - costs.min())
    if cost_span <= _LP_EPS:
        return [float(value) for value in latencies]

    normalized_costs = (costs - costs.min()) / cost_span
    return [float(value) for value in latencies + _COST_TIEBREAK_MS * normalized_costs]


def _normalize_weights(names: list[str], vector: np.ndarray) -> dict[str, float]:
    weights = {
        name: float(vector[idx]) for idx, name in enumerate(names) if float(vector[idx]) > _LP_EPS
    }
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    return {name: value / total for name, value in weights.items()}


def _sample_weighted(weights: dict[str, float], rng: np.random.Generator) -> str:
    names = list(weights)
    probs = np.asarray([weights[name] for name in names], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


__all__ = [
    "LPOnlyAblationPolicy",
    "RollingLatencyProfile",
]
