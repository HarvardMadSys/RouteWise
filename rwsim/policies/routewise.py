"""RouteWise policy and its module-local helpers."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linprog

from rwsim.policies.base import NoOpObserveMixin, NoOpTickMixin
from rwsim.schemas import HedgeDispatch, Request, RoutingDecision, RoutingOutcome
from rwsim.world.capacity import ProviderTier

if TYPE_CHECKING:
    from rwsim.engine.state import SimulationState
    from rwsim.world.providers import Provider

_LP_EPS = 1e-9
_COST_TIEBREAK_MS = 1e-3
_DEFAULT_HEDGE_CHECKPOINTS = (0.25, 0.50, 0.75, 0.90)
_HEDGE_SUCCESS_TARGET = 0.99
_DISPATCH_OVERHEAD_MS = 50.0


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
        return [ttft_ms for ts, ttft_ms in self.samples if ts <= now]

    def mean(self, now: float) -> float | None:
        values = self.values(now)
        if not values:
            return None
        return float(np.mean(values))

    def cdf(self, value_ms: float, now: float) -> float | None:
        values = self.values(now)
        if not values:
            return None
        return float(np.mean(np.asarray(values) <= value_ms))


@dataclass
class RouteWisePolicy(NoOpTickMixin, NoOpObserveMixin):
    """Complete RouteWise policy plus LP-only and LP+hedging ablation modes."""

    hedging: str | bool = "probability_target"
    explorer: bool = True
    p: float = 0.75
    slo_ms: float = 2000.0
    seed: int = 0
    profile_window_sec: float = 15 * 60.0
    rng: np.random.Generator = field(init=False, repr=False)
    profiles: dict[str, RollingLatencyProfile] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"RouteWise p must be in [0, 1], got {self.p}")
        self.rng = np.random.default_rng(self.seed)

    def route(self, request: Request, state: SimulationState) -> RoutingDecision:
        """Solve LP-TTFT-budget and return a primary provider."""
        self._ensure_profiles(state.providers)
        providers = [provider for provider in state.providers.values() if provider.is_available(state.now)]
        if not providers:
            providers = list(state.providers.values())
        if not providers:
            raise ValueError("No providers configured for RouteWisePolicy.")

        L, U = calibrate_envelopes(list(state.providers.values()))
        c_eff = {
            provider.name: effective_cost(
                provider,
                request.total_tokens or 0,
                state.now,
                U=U,
                L=L,
            )
            for provider in providers
        }
        tbar = {
            provider.name: self._latency_objective_ms(provider, state.now)
            for provider in providers
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
            best = min(providers, key=lambda provider: (c_eff[provider.name], tbar[provider.name]))
            weights = {best.name: 1.0}

        primary_name = _sample_weighted(weights, self.rng)
        hedge_checkpoints = ()
        if self.hedging:
            hedge_checkpoints = tuple(
                (self.slo_ms / 1000.0) * frac
                for frac in _DEFAULT_HEDGE_CHECKPOINTS
            )

        return RoutingDecision(
            primary_provider=primary_name,
            hedge_checkpoints=hedge_checkpoints,
            metadata={
                "weights": weights,
                "c_eff": c_eff,
                "budget": budget,
                "L": L,
                "U": U,
            },
        )

    def tick(
        self,
        request: Request,
        decision: RoutingDecision,
        elapsed: float,
        state: SimulationState,
    ) -> HedgeDispatch | None:
        """Re-evaluate probability-target hedging at an in-flight checkpoint."""
        if not self.hedging:
            return None
        providers = state.providers
        primary = providers[decision.primary_provider]
        backup = self._pick_backup(primary, request, state)
        if backup is None:
            return None

        current = _combined_success_probability(
            self,
            primary,
            backup,
            elapsed_ms=elapsed * 1000.0,
            now=state.now,
            slo_ms=self.slo_ms,
        )
        if current < _HEDGE_SUCCESS_TARGET - _LP_EPS:
            return None

        future_safe = False
        for future_elapsed in decision.hedge_checkpoints:
            if future_elapsed <= elapsed + _LP_EPS:
                continue
            future = _combined_success_probability(
                self,
                primary,
                backup,
                elapsed_ms=future_elapsed * 1000.0,
                now=state.now,
                slo_ms=self.slo_ms,
            )
            if future >= _HEDGE_SUCCESS_TARGET - _LP_EPS:
                future_safe = True
                break
        if future_safe:
            return None

        return HedgeDispatch(
            backup_provider=backup.name,
            metadata={"combined_success": current},
        )

    def observe(
        self,
        request: Request,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
    ) -> None:
        """Feed observed TTFT samples into policy-owned profiles."""
        del request
        self._profile(decision.primary_provider).add_sample(
            outcome.metadata.get("primary_observed_at", 0.0),
            float(outcome.metadata.get("primary_ttft_ms", outcome.ttft_ms)),
        )
        if self.explorer and outcome.hedge_triggered:
            backup = outcome.metadata.get("hedge_provider")
            backup_ttft_ms = outcome.metadata.get("backup_ttft_ms")
            backup_observed_at = outcome.metadata.get("backup_observed_at")
            if backup and backup_ttft_ms is not None and backup_observed_at is not None:
                self._profile(str(backup)).add_sample(float(backup_observed_at), float(backup_ttft_ms))

    def _ensure_profiles(self, providers: dict[str, Provider]) -> None:
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

    def _cdf_ms(self, provider: Provider, value_ms: float, now: float) -> float:
        empirical = self._profile(provider.name).cdf(value_ms, now)
        if empirical is not None:
            return empirical
        return provider._active_ttft_dist(now).cdf(value_ms)

    def _pick_backup(
        self,
        primary: Provider,
        request: Request,
        state: SimulationState,
    ) -> Provider | None:
        candidates = [
            provider
            for provider in state.providers.values()
            if provider.name != primary.name and provider.is_available(state.now)
        ]
        if not candidates:
            return None
        if self.explorer:
            return candidates[int(self.rng.integers(0, len(candidates)))]
        return min(
            candidates,
            key=lambda provider: (
                provider.marginal_cost(request.total_tokens or 0, state.now),
                provider.true_mean_ms(state.now),
                provider.name,
            ),
        )


def quota_shadow_price(
    provider: Provider,
    now: float,
    *,
    U: float,
    L: float,
) -> float:
    """Exponential quota shadow price used by RouteWise effective cost."""
    if provider.tier != ProviderTier.S_Q:
        return 0.0
    if provider.quota is None:
        return 0.0
    if L <= 0 or U <= 0 or U <= L:
        raise ValueError(f"Require 0 < L < U; got L={L}, U={U}")
    z = min(max(provider.quota.fraction_used(now), 0.0), 0.9999)
    return L * math.pow(U / L, z)


def concurrency_shadow_price(
    provider: Provider,
    now: float,
    *,
    U: float,
    alpha: float = 1.0,
) -> float:
    """Concurrency shadow price used by RouteWise effective cost."""
    if provider.tier != ProviderTier.S_C:
        return 0.0
    if provider.concurrency is None:
        return 0.0
    return U * math.pow(provider.concurrency.utilization(now), alpha)


def effective_cost(
    provider: Provider,
    total_tokens: int,
    now: float,
    *,
    U: float,
    L: float,
    concurrency_alpha: float = 1.0,
) -> float:
    """RouteWise effective request cost."""
    return (
        provider.marginal_cost(total_tokens, now)
        + quota_shadow_price(provider, now, U=U, L=L)
        + concurrency_shadow_price(provider, now, U=U, alpha=concurrency_alpha)
    )


def calibrate_envelopes(
    providers: list[Provider],
    typical_tokens: int = 200,
    floor_ratio: float = 1e-3,
) -> tuple[float, float]:
    """Compute (L, U) from API provider prices."""
    api_costs = [
        provider.cost_per_token * typical_tokens
        for provider in providers
        if provider.tier == ProviderTier.S_A and provider.cost_per_token > 0
    ]
    if not api_costs:
        return (1e-6, 1e-3)
    U = max(api_costs)
    L = max(U * floor_ratio, 1e-9)
    return (L, U)


def _solve_lp(
    objective: list[float],
    *,
    upper_constraint: list[float],
    upper_bound: float,
) -> tuple[bool, np.ndarray | None]:
    n = len(objective)
    result = linprog(
        c=objective,
        A_ub=[upper_constraint],
        b_ub=[upper_bound],
        A_eq=[np.ones(n)],
        b_eq=[1.0],
        bounds=[(0.0, 1.0) for _ in range(n)],
        method="highs",
    )
    if not result.success:
        return False, None
    return True, result.x


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
        name: float(vector[idx])
        for idx, name in enumerate(names)
        if float(vector[idx]) > _LP_EPS
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


def _combined_success_probability(
    policy: RouteWisePolicy,
    primary: Provider,
    backup: Provider,
    *,
    elapsed_ms: float,
    now: float,
    slo_ms: float,
) -> float:
    primary_cdf_t = policy._cdf_ms(primary, elapsed_ms, now)
    primary_cdf_slo = policy._cdf_ms(primary, slo_ms, now)
    primary_survival_t = max(1.0 - primary_cdf_t, 0.0)
    if primary_survival_t <= _LP_EPS:
        return 0.0

    p_not_violate = max(primary_cdf_slo - primary_cdf_t, 0.0) / primary_survival_t
    p_violate = max(1.0 - primary_cdf_slo, 0.0) / primary_survival_t
    remaining_ms = slo_ms - elapsed_ms - _DISPATCH_OVERHEAD_MS
    backup_success = 0.0 if remaining_ms <= 0.0 else policy._cdf_ms(backup, remaining_ms, now)
    return float(min(max(p_not_violate + p_violate * backup_success, 0.0), 1.0))


__all__ = [
    "RollingLatencyProfile",
    "RouteWisePolicy",
    "calibrate_envelopes",
    "concurrency_shadow_price",
    "effective_cost",
    "quota_shadow_price",
]
