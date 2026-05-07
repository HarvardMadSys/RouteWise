"""RouteWise policy and its module-local helpers."""

from __future__ import annotations

import math
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

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
    """Causal moving-window empirical latency profile for one provider.

    The simulator queries profiles with nondecreasing route-observation time.
    ``mean`` keeps a defensive backwards-query fallback for tests and ad hoc
    probes, but hot-path callers should treat this as a monotonic-time API.
    """

    window_sec: float = 15 * 60.0
    samples: deque[tuple[float, float]] = field(default_factory=deque)
    _mean_pending: list[tuple[float, int, float]] = field(default_factory=list, init=False, repr=False)
    _mean_active: list[tuple[float, int, float]] = field(default_factory=list, init=False, repr=False)
    _mean_sum: float = field(default=0.0, init=False, repr=False)
    _mean_count: int = field(default=0, init=False, repr=False)
    _mean_clock_sec: float = field(default=float("-inf"), init=False, repr=False)
    _sample_seq: int = field(default=0, init=False, repr=False)

    def add_sample(self, timestamp: float, ttft_ms: float) -> None:
        ts = float(timestamp)
        value = float(ttft_ms)
        self.samples.append((ts, value))
        item = (ts, self._sample_seq, value)
        self._sample_seq += 1
        if ts <= self._mean_clock_sec:
            if ts >= self._mean_clock_sec - self.window_sec:
                heapq.heappush(self._mean_active, item)
                self._mean_sum += value
                self._mean_count += 1
        else:
            heapq.heappush(self._mean_pending, item)

    def values(self, now: float) -> list[float]:
        cutoff = now - self.window_sec
        kept: deque[tuple[float, float]] = deque()
        visible: list[float] = []
        for ts, ttft_ms in self.samples:
            if ts < cutoff:
                continue
            kept.append((ts, ttft_ms))
            if ts <= now:
                visible.append(ttft_ms)
        self.samples = kept
        return visible

    def mean(self, now: float) -> float | None:
        if now < self._mean_clock_sec:
            values = [ttft_ms for ts, ttft_ms in self.samples if now - self.window_sec <= ts <= now]
            if not values:
                return None
            return float(np.mean(values))

        self._advance_mean_window(float(now))
        if self._mean_count == 0:
            return None
        return self._mean_sum / self._mean_count

    def cdf(self, value_ms: float, now: float) -> float | None:
        values = self.values(now)
        if not values:
            return None
        return float(np.mean(np.asarray(values) <= value_ms))

    def _advance_mean_window(self, now: float) -> None:
        self._mean_clock_sec = now
        cutoff = now - self.window_sec
        while self._mean_pending and self._mean_pending[0][0] <= now:
            ts, seq, value = heapq.heappop(self._mean_pending)
            if ts >= cutoff:
                heapq.heappush(self._mean_active, (ts, seq, value))
                self._mean_sum += value
                self._mean_count += 1
        while self._mean_active and self._mean_active[0][0] < cutoff:
            _, _, value = heapq.heappop(self._mean_active)
            self._mean_sum -= value
            self._mean_count -= 1


@dataclass
class RouteWisePolicy(NoOpTickMixin, NoOpObserveMixin):
    """Complete RouteWise policy plus LP-only and LP+hedging ablation modes."""

    hedging: str | bool = "probability_target"
    explorer: bool = True
    p: float = 0.75
    cost_envelope: tuple[float, float] | None = None
    slo_ms: float = 2000.0
    seed: int = 0
    profile_window_sec: float = 15 * 60.0
    rng: np.random.Generator = field(init=False, repr=False)
    profiles: dict[str, RollingLatencyProfile] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"RouteWise p must be in [0, 1], got {self.p}")
        if self.cost_envelope is None:
            raise ValueError(
                "RouteWisePolicy requires an explicit cost_envelope. "
                "Simulation sections should pass the workload-level P10/P90 envelope."
            )
        L, U = (float(self.cost_envelope[0]), float(self.cost_envelope[1]))
        if not (math.isfinite(L) and math.isfinite(U) and 0.0 < L < U):
            raise ValueError(
                "RouteWise cost_envelope must satisfy 0 < L < U; "
                f"got L={L}, U={U}"
            )
        self.cost_envelope = (L, U)
        self.rng = np.random.default_rng(self.seed)

    def route(self, request: Request, state: SimulationState) -> RoutingDecision:
        """Solve LP-TTFT-budget and return a primary provider."""
        self._ensure_profiles(state.providers)
        providers = [provider for provider in state.providers.values() if provider.is_available(state.now)]
        if not providers:
            providers = list(state.providers.values())
        if not providers:
            raise ValueError("No providers configured for RouteWisePolicy.")

        L, U = self.cost_envelope
        c_eff = {
            provider.name: effective_cost(
                provider,
                request,
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
        objective = _cost_tiebroken_objective(
            [tbar[name] for name in names],
            [c_eff[name] for name in names],
        )

        if c_max - c_min <= _LP_EPS:
            weights = _same_cost_shortcut_weights(
                names,
                objective=objective,
                costs=[c_eff[name] for name in names],
            )
        else:
            success, vector = _solve_lp(
                objective=objective,
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
                provider.marginal_cost_for_request(request, state.now),
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
    request: Request,
    now: float,
    *,
    U: float,
    L: float,
    concurrency_alpha: float = 1.0,
) -> float:
    """RouteWise piecewise effective request cost.

    Matches the paper formulation exactly:
        c_eff(j) = c_A(j)   if j in S_A   (real API billing)
                 = psi(z_j) if j in S_Q   (quota opportunity cost)
                 = lambda(u_j) if j in S_C (concurrency opportunity cost)

    Numerically equivalent to the previous additive form
        marginal + psi + lambda
    because S_Q / S_C providers are constructed with marginal=0; this
    refactor only sharpens the semantics so the code mirrors the paper.
    """
    tier = provider.tier
    if tier == ProviderTier.S_A:
        return provider.marginal_cost_for_request(request, now)
    if tier == ProviderTier.S_Q:
        return quota_shadow_price(provider, now, U=U, L=L)
    if tier == ProviderTier.S_C:
        return concurrency_shadow_price(provider, now, U=U, alpha=concurrency_alpha)
    raise ValueError(f"Unsupported provider tier for RouteWise effective cost: {tier!r}")


def _solve_lp(
    objective: list[float],
    *,
    upper_constraint: list[float],
    upper_bound: float,
) -> tuple[bool, np.ndarray | None]:
    """Solve the RouteWise simplex LP without invoking a generic solver.

    The LP has one equality constraint (sum of weights is 1) and one budget
    inequality.  Its optimum is therefore either a single provider whose cost is
    within budget, or a mixture of two providers that lies exactly on the budget
    boundary.  Enumerating those candidates is equivalent to solving the LP and
    avoids paying scipy/HiGHS setup cost for every routed request.
    """
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


def _same_cost_shortcut_weights(
    names: list[str],
    *,
    objective: list[float],
    costs: list[float],
) -> dict[str, float]:
    """Return the one-hot LP optimum when every provider has the same cost."""
    best_name: str | None = None
    best_key: tuple[float, float, int, tuple[float, ...]] | None = None
    n = len(names)
    for index, name in enumerate(names):
        vector = tuple(1.0 if position == index else 0.0 for position in range(n))
        key = (float(objective[index]), float(costs[index]), 1, vector)
        if best_key is None or key < best_key:
            best_key = key
            best_name = name
    return {best_name: 1.0} if best_name is not None else {}


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
    "concurrency_shadow_price",
    "effective_cost",
    "quota_shadow_price",
]
