"""RouteWise policy and its module-local helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from rwsim.core.cost import (
    concurrency_effective_cost,
    effective_cost as core_effective_cost,
    quota_effective_cost,
)
from rwsim.core.hedging import (
    DISPATCH_OVERHEAD_MS,
    HEDGE_SUCCESS_TARGET,
    BackupCandidate,
    combined_success_probability,
    has_feasible_backup,
    hedge_checkpoints_for_slo,
    select_probability_backup,
)
from rwsim.core.lp import (
    LP_EPS,
    cost_tiebroken_objective,
    normalize_weights,
    solve_simplex_lp,
)
from rwsim.policies.base import NoOpObserveMixin, NoOpTickMixin
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


_LP_EPS = LP_EPS


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
        objective = cost_tiebroken_objective(
            [tbar[name] for name in names],
            [c_eff[name] for name in names],
        )

        if c_max - c_min <= _LP_EPS:
            weights = _same_cost_shortcut_weights(
                names,
                objective=objective,
                costs=[c_eff[name] for name in names],
            )
            lp_status = "single_candidate" if len(names) == 1 else "feasible"
        else:
            success, vector = solve_simplex_lp(
                objective=objective,
                upper_constraint=[c_eff[name] for name in names],
                upper_bound=budget,
            )
            weights = normalize_weights(names, vector) if success and vector is not None else {}
            # budget = c_min + p*(c_max - c_min) with p in [0, 1], so budget >= c_min
            # and the min-cost provider is always within budget. An empty solve is
            # therefore a solver fallback, not an over-budget infeasibility: the
            # fallback below picks the (feasible) min-cost provider.
            lp_status = "feasible"
        if not weights:
            best = min(providers, key=lambda provider: (c_eff[provider.name], tbar[provider.name]))
            weights = {best.name: 1.0}
            lp_status = "feasible"

        primary_name = _sample_weighted(weights, self.rng)
        hedge_checkpoints = ()
        if self.hedging:
            hedge_checkpoints = hedge_checkpoints_for_slo(self.slo_ms)

        primary_provider = next(p for p in providers if p.name == primary_name)
        routing_estimated_cost_usd = self._routing_dollar_cost(primary_provider, request, state)

        return RoutingDecision(
            primary_provider=primary_name,
            hedge_checkpoints=hedge_checkpoints,
            metadata={
                "weights": weights,
                "c_eff": c_eff,
                "budget": budget,
                "L": L,
                "U": U,
                "lp_status": lp_status,
                "routing_estimated_cost_usd": routing_estimated_cost_usd,
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
        predicted_response_tokens = _predicted_output_tokens_from_prediction(
            prediction,
            self.output_predictor_quantile,
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

    def _routing_dollar_cost(
        self,
        provider: Provider,
        request: Request,
        state: SimulationState | None,
    ) -> float | None:
        """Decision-time dollar token cost using predicted output tokens.

        This is the routing-time estimate, not the realized cost. It uses only
        information visible at routing time: the predictor output, or the trace's
        ``estimated_response_tokens``. It never uses the actual generated
        ``response_tokens`` — substituting that would make the estimate equal the
        realized cost. Returns ``None`` when no routing-time estimate exists.

        Mirrors real-eval's ``routing_cache_diagnostics`` /
        ``cache_aware_request_cost_usd``: raw token-priced dollars regardless of
        tier, so SIM and real-eval expose the same quantity.
        """
        if self.output_predictor is not None:
            prediction = self.output_predictor.predict(request)
            predicted_response_tokens = _predicted_output_tokens_from_prediction(
                prediction,
                self.output_predictor_quantile,
            )
        else:
            predicted = getattr(request, "estimated_response_tokens", None)
            if predicted is None:
                return None
            predicted_response_tokens = float(predicted)
        request_tokens = int(getattr(request, "request_tokens", 0) or 0)
        cached_tokens = 0
        if state is not None:
            from rwsim.policies.prefix_cache import cached_input_tokens

            cached_tokens = cached_input_tokens(provider, request, state)
        return float(
            provider.token_cost(
                request_tokens=request_tokens,
                response_tokens=predicted_response_tokens,
                cached_input_tokens=cached_tokens,
            )
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
        candidates: list[BackupCandidate] = []
        for provider in state.providers.values():
            if provider.name == primary.name or not provider.is_available(state.now):
                continue
            candidates.append(
                BackupCandidate(
                    provider=provider,
                    success_probability=combined_success_probability(
                        lambda value_ms: self._cdf_ms(primary, value_ms, state.now),
                        lambda value_ms, backup=provider: self._cdf_ms(
                            backup,
                            value_ms,
                            state.now,
                        ),
                        elapsed_ms=elapsed_ms,
                        slo_ms=self.slo_ms,
                        dispatch_overhead_ms=DISPATCH_OVERHEAD_MS,
                    ),
                    marginal_cost=cache_aware_marginal_cost(
                        provider,
                        request,
                        state,
                        now=state.now,
                    ),
                    true_mean_ms=provider.true_mean_ms(state.now),
                    success_target=HEDGE_SUCCESS_TARGET,
                )
            )
        return candidates

    def _select_backup_candidate(
        self,
        candidates: list[BackupCandidate],
    ) -> BackupCandidate | None:
        return select_probability_backup(candidates)

    def _has_future_feasible_backup(
        self,
        primary: Provider,
        request: Request,
        decision: RoutingDecision,
        elapsed: float,
        state: SimulationState,
    ) -> bool:
        for future_elapsed in decision.hedge_checkpoints:
            if future_elapsed <= elapsed + _LP_EPS:
                continue
            future_ms = future_elapsed * 1000.0
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
    return quota_effective_cost(provider.quota.fraction_used(now), L=L, U=U)


def concurrency_shadow_price(
    provider: Provider,
    now: float,
    *,
    U: float,
    L: float,
    alpha: float = 1.0,
) -> float:
    """Zero marginal shadow price for reusable concurrency capacity."""
    del now, alpha
    if provider.tier != ProviderTier.S_C:
        return 0.0
    if provider.concurrency is None:
        return 0.0
    return concurrency_effective_cost(None, L=L, U=U)


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
            request_cost = cache_aware_marginal_cost(provider, request, state, now=now)
        else:
            request_cost = provider.marginal_cost_for_request(request, now)
        return core_effective_cost(
            "api",
            request_cost_usd=request_cost,
            L=L,
            U=U,
        )
    if tier == ProviderTier.S_Q:
        return core_effective_cost(
            "quota",
            quota_fraction_used=(
                None if provider.quota is None else provider.quota.fraction_used(now)
            ),
            L=L,
            U=U,
        )
    if tier == ProviderTier.S_C:
        del concurrency_alpha
        return core_effective_cost(
            "concurrency",
            concurrency_utilization=None,
            L=L,
            U=U,
        )
    raise ValueError(f"Unsupported provider tier for RouteWise effective cost: {tier!r}")


def _predicted_output_tokens_from_prediction(prediction: Any, quantile: str) -> float:
    """Extract a route-time output-token estimate from a predictor result."""

    if hasattr(prediction, "tokens"):
        if quantile != "q50":
            raise ValueError(
                "output_predictor_quantile="
                f"{quantile!r} requires a quantile predictor; point predictors only support q50"
            )
        return max(float(prediction.tokens), 0.0)
    return max(float(getattr(prediction, quantile)), 0.0)


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


def _sample_weighted(weights: dict[str, float], rng: np.random.Generator) -> str:
    names = list(weights)
    probs = np.asarray([weights[name] for name in names], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


__all__ = [
    "RollingLatencyProfile",
    "RouteWisePolicy",
    "concurrency_shadow_price",
    "effective_cost",
    "quota_shadow_price",
]
