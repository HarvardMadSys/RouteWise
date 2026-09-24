"""RouteWise policy and its module-local helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from rwsim.policies.base import NoOpObserveMixin, NoOpTickMixin
from rwsim.policies.effective_cost_kernel import scarcity_price
from rwsim.policies.hedging import (
    DISPATCH_OVERHEAD_MS,
    HEDGE_SUCCESS_TARGET,
    BackupCandidate,
    collect_backup_candidates,
    combined_success_probability,
    has_feasible_backup,
    select_probability_backup,
)
from rwsim.policies.latency_profiles import (
    LatencyProfileMode,
    LatencyProfileStrategy,
    RollingLatencyProfile,
    make_latency_profile_strategy,
)
from rwsim.policies.prefix_cache import cache_aware_marginal_cost
from rwsim.schemas import HedgeDispatch, Request, RoutingDecision, RoutingOutcome
from rwsim.world.capacity import ProviderTier

if TYPE_CHECKING:
    from rwsim.engine.state import SimulationState
    from rwsim.world.providers import Provider


class OutputPredictor(Protocol):
    """Optional output-token predictor consulted at routing time."""

    def predict(self, request: Request) -> Any: ...

    def update(self, request: Request) -> None: ...


_LP_EPS = 1e-9
_COST_TIEBREAK_MS = 1e-3
_HEDGE_CHECKPOINT_START_FRACTION = 0.25
_HEDGE_CHECKPOINT_END_FRACTION = 0.90
_HEDGE_CHECKPOINT_INTERVAL_FRACTION = 0.025


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
    latency_profile_mode: LatencyProfileMode = "observed"
    output_predictor: OutputPredictor | None = None
    output_predictor_quantile: str = "q50"
    rng: np.random.Generator = field(init=False, repr=False)
    profiles: dict[str, RollingLatencyProfile] = field(default_factory=dict, init=False)
    _latency_profile: LatencyProfileStrategy = field(init=False, repr=False)

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
        if self.output_predictor_quantile not in {"q10", "q50", "q90"}:
            raise ValueError(
                "output_predictor_quantile must be one of q10, q50, q90; "
                f"got {self.output_predictor_quantile!r}"
            )
        self.rng = np.random.default_rng(self.seed)
        self._latency_profile = make_latency_profile_strategy(
            self.latency_profile_mode,
            window_sec=self.profile_window_sec,
            profiles=self.profiles,
        )

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
            provider.name: self._effective_cost_for_request(
                provider,
                request,
                state.now,
                U=U,
                L=L,
                state=state,
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
            hedge_checkpoints = _hedge_checkpoints_for_slo(self.slo_ms)

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
        elapsed_ms = elapsed * 1000.0
        candidates = self._collect_backup_candidates(
            primary,
            request,
            state,
            elapsed_ms=elapsed_ms,
        )
        selected = self._select_backup_candidate(candidates)
        if selected is None or not selected.feasible:
            return None

        future_feasible = self._has_future_feasible_backup(
            primary,
            selected,
            request,
            decision,
            elapsed,
            state,
        )
        if not self._should_dispatch_now(
            selected,
            future_feasible=future_feasible,
        ):
            return None

        return HedgeDispatch(
            backup_provider=selected.provider.name,
            metadata={"combined_success": selected.success_probability},
        )

    def observe(
        self,
        request: Request,
        decision: RoutingDecision,
        outcome: RoutingOutcome,
    ) -> None:
        """Feed observed TTFT samples into policy-owned profiles."""
        if self.output_predictor is not None:
            self.output_predictor.update(request)
        self._latency_profile.observe(
            decision.primary_provider,
            outcome.metadata.get("primary_observed_at", 0.0),
            float(outcome.metadata.get("primary_ttft_ms", outcome.ttft_ms)),
        )
        if self.explorer and outcome.hedge_triggered:
            backup = outcome.metadata.get("hedge_provider")
            backup_ttft_ms = outcome.metadata.get("backup_ttft_ms")
            backup_observed_at = outcome.metadata.get("backup_observed_at")
            if backup and backup_ttft_ms is not None and backup_observed_at is not None:
                self._latency_profile.observe(
                    str(backup),
                    float(backup_observed_at),
                    float(backup_ttft_ms),
                )

    def _effective_cost_for_request(
        self,
        provider: Provider,
        request: Request,
        now: float,
        *,
        U: float,
        L: float,
        state: SimulationState | None,
    ) -> float:
        """Effective cost with optional predictor-based S_A output tokens."""
        if self.output_predictor is None or provider.tier != ProviderTier.S_A:
            return effective_cost(provider, request, now, U=U, L=L, state=state)
        prediction = self.output_predictor.predict(request)
        predicted_response_tokens = max(
            float(getattr(prediction, self.output_predictor_quantile)),
            0.0,
        )
        request_tokens = int(getattr(request, "request_tokens", 0) or 0)
        cached_tokens = 0
        if state is not None:
            from rwsim.policies.prefix_cache import cached_input_tokens

            cached_tokens = cached_input_tokens(provider, request, state)
        return provider.token_cost(
            request_tokens=request_tokens,
            response_tokens=predicted_response_tokens,
            cached_input_tokens=cached_tokens,
        )

    def _ensure_profiles(self, providers: dict[str, Provider]) -> None:
        for name in providers:
            self._latency_profile.ensure_provider(name)

    def _profile(self, name: str) -> RollingLatencyProfile:
        return self.profiles.setdefault(
            name,
            RollingLatencyProfile(window_sec=self.profile_window_sec),
        )

    def _latency_objective_ms(self, provider: Provider, now: float) -> float:
        return self._latency_profile.mean_ms(provider, now)

    def _cdf_ms(self, provider: Provider, value_ms: float, now: float) -> float:
        return self._latency_profile.cdf_ms(provider, value_ms, now)

    def _collect_backup_candidates(
        self,
        primary: Provider,
        request: Request,
        state: SimulationState,
        *,
        elapsed_ms: float,
    ) -> list[BackupCandidate]:
        return collect_backup_candidates(
            state.providers.values(),
            primary,
            request,
            now=state.now,
            elapsed_ms=elapsed_ms,
            slo_ms=self.slo_ms,
            cdf_ms=lambda provider, value_ms: self._cdf_ms(
                provider,
                value_ms,
                state.now,
            ),
            success_target=HEDGE_SUCCESS_TARGET,
            dispatch_overhead_ms=DISPATCH_OVERHEAD_MS,
            marginal_cost=lambda provider: cache_aware_marginal_cost(
                provider,
                request,
                state,
                now=state.now,
            ),
        )

    def _select_backup_candidate(
        self,
        candidates: list[BackupCandidate],
    ) -> BackupCandidate | None:
        if not candidates:
            return None
        if self.explorer:
            # TODO(routewise-hedging-ablation): replace this coarse random
            # fallback with an explicit random/spread backup-selection mode for
            # hedge-as-probe experiments. The default hedging path below stays
            # probability-driven and optimizes the current request's SLO success.
            return candidates[int(self.rng.integers(0, len(candidates)))]
        return select_probability_backup(candidates)

    def _has_future_feasible_backup(
        self,
        primary: Provider,
        selected: BackupCandidate,
        request: Request,
        decision: RoutingDecision,
        elapsed: float,
        state: SimulationState,
    ) -> bool:
        for future_elapsed in decision.hedge_checkpoints:
            if future_elapsed <= elapsed + _LP_EPS:
                continue
            future_ms = future_elapsed * 1000.0
            if self.explorer:
                future = combined_success_probability(
                    lambda provider, value_ms: self._cdf_ms(
                        provider,
                        value_ms,
                        state.now,
                    ),
                    primary,
                    selected.provider,
                    elapsed_ms=future_ms,
                    slo_ms=self.slo_ms,
                    dispatch_overhead_ms=DISPATCH_OVERHEAD_MS,
                )
                if future >= HEDGE_SUCCESS_TARGET - _LP_EPS:
                    return True
                continue

            candidates = self._collect_backup_candidates(
                primary,
                request,
                state,
                elapsed_ms=future_ms,
            )
            if has_feasible_backup(candidates):
                return True
        return False

    def _should_dispatch_now(
        self,
        selected: BackupCandidate,
        *,
        future_feasible: bool,
    ) -> bool:
        return selected.feasible and not future_feasible


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
    return scarcity_price("exp_lu", provider.quota.fraction_used(now), L=L, U=U)


def concurrency_shadow_price(
    provider: Provider,
    now: float,
    *,
    U: float,
    L: float,
    alpha: float = 1.0,
) -> float:
    """Zero marginal shadow price for reusable concurrency capacity."""
    del now, U, L, alpha
    if provider.tier != ProviderTier.S_C:
        return 0.0
    if provider.concurrency is None:
        return 0.0
    return 0.0


def effective_cost(
    provider: Provider,
    request: Request,
    now: float,
    *,
    U: float,
    L: float,
    state: SimulationState | None = None,
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
        if state is not None:
            return cache_aware_marginal_cost(provider, request, state, now=now)
        return provider.marginal_cost_for_request(request, now)
    if tier == ProviderTier.S_Q:
        return quota_shadow_price(provider, now, U=U, L=L)
    if tier == ProviderTier.S_C:
        return concurrency_shadow_price(provider, now, U=U, L=L, alpha=concurrency_alpha)
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


def _hedge_checkpoints_for_slo(slo_ms: float) -> tuple[float, ...]:
    """Return dense latest-safe hedge checkpoints in seconds."""
    slo_ms_f = float(slo_ms)
    interval_ms = slo_ms_f * _HEDGE_CHECKPOINT_INTERVAL_FRACTION
    if interval_ms <= 0.0:
        return ()
    start_ms = _ceil_to_interval_ms(
        slo_ms_f * _HEDGE_CHECKPOINT_START_FRACTION,
        interval_ms,
    )
    end_ms = slo_ms_f * _HEDGE_CHECKPOINT_END_FRACTION
    if start_ms > end_ms + _LP_EPS:
        return ()

    checkpoints_ms: list[float] = []
    current_ms = start_ms
    while current_ms <= end_ms + _LP_EPS:
        checkpoints_ms.append(current_ms)
        current_ms += interval_ms
    return tuple(ms / 1000.0 for ms in checkpoints_ms)


def _ceil_to_interval_ms(value_ms: float, interval_ms: float) -> float:
    return math.ceil((value_ms - _LP_EPS) / interval_ms) * interval_ms


__all__ = [
    "RollingLatencyProfile",
    "RouteWisePolicy",
    "concurrency_shadow_price",
    "effective_cost",
    "quota_shadow_price",
]
