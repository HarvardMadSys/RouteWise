"""RouteWise simulator policy: a thin adapter over the core router.

The algorithm itself lives in :mod:`routewise.core.router`; this module binds it
to the simulator world. ``_SimProviderView`` closes over
``(Provider, Request, SimulationState)`` and supplies the environment-specific
signals: cache-aware/predictor-based request costs, quota fractions, and the
oracle prior (the provider's configured true distribution) that backs the
empty-window fallback and the ``configured`` latency-profile mode.
"""

from __future__ import annotations

import math
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from routewise.capacity import ProviderTier
from routewise.const import DEFAULT_PRIMARY_SLO_MS
from routewise.core.beliefs import LatencyBeliefs
from routewise.core.cost import (
    concurrency_effective_cost,
    effective_cost as core_effective_cost,
    quota_effective_cost,
)
from routewise.core.hedging import (
    DISPATCH_OVERHEAD_MS,
    HEDGE_SUCCESS_TARGET,
    BackupCandidate,
    select_probability_backup,
)
from routewise.core.router import (
    LPStatus,
    RouteWiseRouter,
    # Re-exported for tests that pin shortcut/LP tie-break equivalence.
    _same_cost_shortcut_weights as _same_cost_shortcut_weights,
    latest_safe_dispatch_gate,
)
from routewise.schemas import HedgeDispatch, Request, RoutingDecision, RoutingOutcome
from routewise.sim.policies.base import NoOpObserveMixin, NoOpTickMixin
from routewise.sim.policies.latency_profiles import (
    DEFAULT_PROFILE_WINDOW_SEC,
    LatencyProfileMode,
    RollingLatencyProfile,
)
from routewise.sim.policies.prefix_cache import cache_aware_marginal_cost, cached_input_tokens

if TYPE_CHECKING:
    from routewise.sim.engine.state import SimulationState
    from routewise.sim.world.providers import Provider


class OutputPredictor(Protocol):
    """Optional output-token predictor consulted at routing time."""

    def predict(self, request: Request) -> Any: ...

    def update(self, request: Request) -> None: ...


@dataclass(frozen=True)
class _SimProviderView:
    """ProviderView binding one simulator provider to one request."""

    provider: Provider
    request: Request
    state: SimulationState
    output_predictor: OutputPredictor | None

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def tier(self) -> str:
        return self.provider.tier.value

    def is_available(self, now: float) -> bool:
        return self.provider.is_available(now)

    def quota_fraction_used(self, now: float) -> float | None:
        if self.provider.quota is None:
            return None
        return self.provider.quota.fraction_used(now)

    def route_cost_usd(self, now: float) -> float:
        """Route-time API cost: predictor-based S_A tokens when available."""
        provider = self.provider
        if self.output_predictor is None or provider.tier != ProviderTier.S_A:
            return cache_aware_marginal_cost(provider, self.request, self.state, now=now)
        prediction = self.output_predictor.predict(self.request)
        predicted_response_tokens = _predicted_output_tokens_from_prediction(prediction)
        request_tokens = int(getattr(self.request, "request_tokens", 0) or 0)
        cached_tokens = cached_input_tokens(provider, self.request, self.state)
        return provider.token_cost(
            request_tokens=request_tokens,
            response_tokens=predicted_response_tokens,
            cached_input_tokens=cached_tokens,
        )

    def hedge_cost_usd(self, now: float) -> float:
        """Hedge-backup cost: trace-realized tokens (pre-router behavior)."""
        return cache_aware_marginal_cost(self.provider, self.request, self.state, now=now)

    def prior_ttft_mean_ms(self, now: float) -> float | None:
        return self.provider.true_mean_ms(now)

    def prior_ttft_cdf(self, value_ms: float, now: float) -> float | None:
        return self.provider._active_ttft_dist(now).cdf(value_ms)


@dataclass
class RouteWisePolicy(NoOpTickMixin, NoOpObserveMixin):
    """Complete RouteWise policy plus LP-only and LP+hedging ablation modes."""

    hedging: str | bool = "probability_target"
    explorer: bool = True
    alpha: float = 0.75
    p: InitVar[float | None] = None
    cost_envelope: tuple[float, float] | None = None
    slo_ms: float = DEFAULT_PRIMARY_SLO_MS
    seed: int = 0
    profile_window_sec: float = DEFAULT_PROFILE_WINDOW_SEC
    latency_profile_mode: LatencyProfileMode = "observed"
    output_predictor: OutputPredictor | None = None
    rng: np.random.Generator = field(init=False, repr=False)
    profiles: dict[str, RollingLatencyProfile] = field(default_factory=dict, init=False)
    router: RouteWiseRouter = field(init=False, repr=False)

    def __post_init__(self, p: float | None = None) -> None:
        if p is not None:
            self.alpha = float(p)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"RouteWise alpha must be in [0, 1], got {self.alpha}")
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
        if self.latency_profile_mode not in ("observed", "configured"):
            raise ValueError(
                "unknown latency_profile_mode "
                f"{self.latency_profile_mode!r}; expected 'observed' or 'configured'"
            )
        self.rng = np.random.default_rng(self.seed)
        self.router = RouteWiseRouter(
            alpha=self.alpha,
            slo_ms=self.slo_ms,
            beliefs=LatencyBeliefs(
                window_sec=self.profile_window_sec,
                mode=(
                    "prior_only"
                    if self.latency_profile_mode == "configured"
                    else "observed"
                ),
                profiles=self.profiles,
            ),
            cost_envelope=self.cost_envelope,
            hedging=bool(self.hedging),
            sampler=self._sample_primary,
            hedge_dispatch_overhead_ms=DISPATCH_OVERHEAD_MS,
            hedge_success_target=HEDGE_SUCCESS_TARGET,
            hedge_tiebreak_mean="prior",
            backup_selector=self._select_backup_candidate,
            dispatch_gate=self._dispatch_gate,
        )

    def route(self, request: Request, state: SimulationState) -> RoutingDecision:
        """Solve LP-TTFT-budget and return a primary provider."""
        self._ensure_profiles(state.providers)
        providers = [provider for provider in state.providers.values() if provider.is_available(state.now)]
        if not providers:
            providers = list(state.providers.values())
        if not providers:
            raise ValueError("No providers configured for RouteWisePolicy.")

        self._sync_router()
        views = [self._view(provider, request, state) for provider in providers]
        result = self.router.route(views, state.now)

        L, U = self.cost_envelope
        primary_provider = next(p for p in providers if p.name == result.primary)
        routing_estimated_cost_usd = self._routing_dollar_cost(primary_provider, request, state)

        return RoutingDecision(
            primary_provider=result.primary,
            hedge_checkpoints=result.hedge_checkpoints_sec,
            metadata={
                "weights": result.weights,
                "c_eff": result.c_eff,
                "budget": result.budget,
                "alpha": self.alpha,
                "L": L,
                "U": U,
                "lp_status": (
                    "single_candidate"
                    if result.lp_status is LPStatus.SINGLE_CANDIDATE
                    else "feasible"
                ),
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
        self._sync_router()
        views = [
            self._view(provider, request, state)
            for provider in state.providers.values()
        ]
        result = self.router.checkpoint_backup(
            decision.primary_provider,
            views,
            state.now,
            elapsed_sec=elapsed,
            future_checkpoints_sec=decision.hedge_checkpoints,
        )
        if result.backup is None:
            return None
        return HedgeDispatch(
            backup_provider=result.backup,
            metadata={"combined_success": result.success_probability},
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
        self.router.observe(
            decision.primary_provider,
            outcome.metadata.get("primary_observed_at", 0.0),
            float(outcome.metadata.get("primary_ttft_ms", outcome.ttft_ms)),
        )
        if self.explorer and outcome.hedge_triggered:
            backup = outcome.metadata.get("hedge_provider")
            backup_ttft_ms = outcome.metadata.get("backup_ttft_ms")
            backup_observed_at = outcome.metadata.get("backup_observed_at")
            if backup and backup_ttft_ms is not None and backup_observed_at is not None:
                self.router.observe(
                    str(backup),
                    float(backup_observed_at),
                    float(backup_ttft_ms),
                )

    def _view(
        self,
        provider: Provider,
        request: Request,
        state: SimulationState,
    ) -> _SimProviderView:
        return _SimProviderView(
            provider=provider,
            request=request,
            state=state,
            output_predictor=self.output_predictor,
        )

    def _sync_router(self) -> None:
        """Propagate post-init mutations of the public knobs to the router."""
        router = self.router
        router.alpha = self.alpha
        router.slo_ms = self.slo_ms
        router.cost_envelope = self.cost_envelope
        router.hedging = bool(self.hedging)

    def _sample_primary(self, weights: dict[str, float], now: float) -> str:
        del now
        return _sample_weighted(weights, self.rng)

    def _dispatch_gate(self, selected: BackupCandidate, future_feasible: bool) -> bool:
        return self._should_dispatch_now(selected, future_feasible=future_feasible)

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
            predicted_response_tokens = _predicted_output_tokens_from_prediction(prediction)
        else:
            predicted = getattr(request, "estimated_response_tokens", None)
            if predicted is None:
                return None
            predicted_response_tokens = float(predicted)
        request_tokens = int(getattr(request, "request_tokens", 0) or 0)
        cached_tokens = 0
        if state is not None:
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
            self.router.beliefs.ensure(name)

    def _select_backup_candidate(
        self,
        candidates: list[BackupCandidate],
    ) -> BackupCandidate | None:
        """Backup-selection hook; hedging-ablation subclasses override this."""
        return select_probability_backup(candidates)

    def _should_dispatch_now(
        self,
        selected: BackupCandidate,
        *,
        future_feasible: bool,
    ) -> bool:
        """Dispatch-timing hook; hedging-ablation subclasses override this."""
        return latest_safe_dispatch_gate(selected, future_feasible)


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

    The routing path applies the same formula through
    :meth:`routewise.core.router.RouteWiseRouter._effective_cost` over
    ``_SimProviderView``; this module-level form is kept for the effective-cost
    kernel-consistency contract with real-eval and for experiment scripts.
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


def _predicted_output_tokens_from_prediction(prediction: Any) -> float:
    """Extract the route-time point estimate from a predictor result."""

    if hasattr(prediction, "tokens"):
        return max(float(prediction.tokens), 0.0)
    if hasattr(prediction, "median"):
        return max(float(prediction.median), 0.0)
    if hasattr(prediction, "q50"):
        return max(float(prediction.q50), 0.0)
    raise TypeError("output predictor must return PointPrediction or a q50-compatible result")


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
