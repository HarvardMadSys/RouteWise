"""Adapter routing policies for real online evaluation.

ARCHITECTURE NOTE — read before extending.

This module is the **real-eval adapter** over the canonical RouteWise
algorithm in :mod:`rwsim.core.router`. The ``BudgetRange*`` policies bind
that router to live providers through ``_RealProviderView`` (empirical
rolling profiles, no oracle priors, per-call time-seeded sampling) and keep
the world-interaction concerns local: locking, capacity charging, transports,
OpenRouter sentinels, and the recorder-facing ``RoutingDecision``. The
simulator binds the same router in :mod:`rwsim.policies.routewise`; algorithm
changes belong in the core, environment changes belong here.

Policy taxonomy:

* **Baselines** (used in paper plots as comparison points):
  - ``OrAutoPolicy``          : OpenRouter's native auto-routing
  - ``OrSortLatencyPolicy``   : OpenRouter ``sort=latency``
  - ``OrSortCostPolicy``      : OpenRouter ``sort=price``
  - ``GreedyCostPolicy``      : cheapest feasible; zero-cost ties prefer
                                ``S_C`` -> ``S_Q`` -> ``S_A``
  - ``GreedyLatencyPolicy``   : always lowest empirical-latency available
  - ``RandomPolicy``          : uniform random over feasible providers
  - ``QuotaFirstPolicy`` / ``ConcurrencyFirstPolicy`` : tier-priority heuristics

* **Current paper line** (``LP-TTFT-budget`` + ``Hedge-ProbTarget``):
  - ``BudgetRangeHedgePolicy(alpha)`` : range-normalized cost budget
    ``B_alpha = (1 - alpha) c_min + alpha c_max``, probability-target hedge
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from experiments.offline_stage.value_estimators import (
    BucketMeanOutputPredictor,
    OutputTokenPredictor,
)
from experiments.real_evaluation.inventory import (
    PROFILE_WINDOW_SEC,
    ProviderSpec,
    ProviderState,
)
from experiments.real_evaluation.prefix_cache import (
    cache_aware_request_cost_usd,
    cached_input_tokens,
)
from experiments.real_evaluation.shadow_price import request_marginal_cost
from rwsim.core.beliefs import LatencyBeliefs
from rwsim.core.hedging import HEDGE_SUCCESS_TARGET
from rwsim.core.latency_profile import DEFAULT_PROFILE_WINDOW_SEC
from rwsim.core.router import LPStatus, RouteWiseRouter
from rwsim.schemas import Request as _PredictionRequest

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# HEDGE_SUCCESS_TARGET is re-exported above from `rwsim.core.hedging` so the
# simulator and real-eval share one paper-level RouteWise protocol constant.
RATE_LIMIT_ERROR_PENALTY_MS: float = 60_000.0
BODY_MEAN_MIN_SAMPLES: int = 5
# Penalty applied to providers with no usable profile data so the LP
# still yields a feasible solution but never picks them when *any*
# profiled candidate is available. ~11.5 days in milliseconds — clearly
# unphysical, but finite. Do NOT use ``U * 1000`` (units mismatch: U is
# USD, not seconds) — that yielded ~0.1 ms and made unprofiled
# providers look fastest.
UNPROFILED_LATENCY_PENALTY_MS: float = 1e9

# Sentinel provider names used for OpenRouter's native routing modes. The
# runner translates these into transport-level config when dispatching.
OR_AUTO_SENTINEL: str = "__or_auto__"
OR_SORT_LATENCY_SENTINEL: str = "__or_sort_latency__"
OR_SORT_COST_SENTINEL: str = "__or_sort_cost__"
OR_SORT_THROUGHPUT_SENTINEL: str = "__or_sort_throughput__"

# Map sentinel name -> OpenRouter ``sort`` mode. The runner uses this to
# build a transport config for sentinel dispatches without relying on
# string-substring tricks.
OR_SORT_SENTINEL_TO_MODE: dict[str, str] = {
    OR_SORT_LATENCY_SENTINEL: "latency",
    OR_SORT_COST_SENTINEL: "price",
    OR_SORT_THROUGHPUT_SENTINEL: "throughput",
}


class CapacityUnavailableError(RuntimeError):
    """Raised when a provider cannot reserve capacity at dispatch time."""

    def __init__(self, provider: str) -> None:
        super().__init__(f"Provider capacity unavailable: {provider}")
        self.provider = provider


@dataclass
class RequestContext:
    """Per-request inputs the policy needs to make a routing decision."""

    prompt_tokens: int
    completion_tokens_budget: int
    prefix_id: str | None = None
    trace_cached_input_tokens: int | None = None


@dataclass
class RoutingDecision:
    """Output of a policy's ``route`` call."""

    primary: str | None
    hedge: str | None = None
    hedge_delay_sec: float = float("inf")
    lp_weights: dict[str, float] | None = None
    lp_status: str = "n/a"
    budget_usd: float | None = None
    reference_cost_usd: float | None = None
    c_eff_map: dict[str, float] | None = None
    tier_mix: dict[str, float] | None = None
    notes: str = ""


@dataclass(frozen=True)
class CheckpointHedgeDecision:
    """Backup decision made at one in-flight hedge checkpoint."""

    backup: str | None
    elapsed_sec: float
    success_probability: float | None = None
    future_feasible: bool = False
    feasible_count: int = 0
    candidate_count: int = 0


def request_cost_for_spec(spec: ProviderSpec, ctx: RequestContext) -> float:
    """Pricing-model marginal cost for one request on this provider."""
    return request_marginal_cost(
        spec,
        ctx.prompt_tokens,
        ctx.completion_tokens_budget,
    )


def _sample_weighted(weights: dict[str, float], rng: random.Random) -> str:
    """Sample one provider name from a (possibly unnormalized) weight map."""
    providers = list(weights.keys())
    probs = np.array([weights[p] for p in providers], dtype=float)
    total = float(np.sum(probs))
    if total <= 0:
        return providers[0]
    probs /= total
    draw = rng.random()
    cdf = 0.0
    for provider, p in zip(providers, probs, strict=True):
        cdf += float(p)
        if draw <= cdf:
            return provider
    return providers[-1]


def _time_seeded_sampler(weights: dict[str, float], now: float) -> str:
    """LP-weight sampler with the historical per-call time-seeded RNG."""
    return _sample_weighted(weights, rng=random.Random(int(now * 1e6)))


def _tier_mix_from_weights(
    weights: dict[str, float],
    states: dict[str, ProviderState],
) -> dict[str, float]:
    mix: dict[str, float] = {"quota": 0.0, "concurrency": 0.0, "api": 0.0}
    for provider, w in weights.items():
        state = states.get(provider)
        if state is None:
            continue
        mix[state.spec.tier] = mix.get(state.spec.tier, 0.0) + w
    return mix


def _body_latency_proxy_ms(
    state: ProviderState,
    now: float,
    *,
    error_penalty_ms: float,
) -> tuple[float, bool]:
    """Empirical body-latency estimate ``T̄_j(t)`` for one provider.

    Returns ``(mean_ms, used_fallback)`` where ``used_fallback`` is True if
    the rolling profile had too few samples and we fell back to the
    mean from a small sample set or a large sentinel.

    Failed attempts, including HTTP 429s, do not have a meaningful TTFT.
    They enter the body objective as synthetic 60s samples so burst-sensitive
    providers are avoided after live rate-limit feedback.
    """
    n = state.profile.total_count(now)
    if n > 0:
        mean = state.profile.mean_with_errors_ms(now, error_penalty_ms=error_penalty_ms)
        if mean is not None:
            return float(mean), n < BODY_MEAN_MIN_SAMPLES
    return float("inf"), True


# ---------------------------------------------------------------------------
# Core-router binding: provider views + real-eval belief configuration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RealProviderView:
    """ProviderView binding one real-eval provider to one request.

    Priors are ``None``: the real world has no oracle distribution, so empty
    rolling windows fall back to the unprofiled penalty (means) or 0.0 (CDFs).
    """

    state: ProviderState
    cost_fn: Callable[[ProviderState], float]

    @property
    def name(self) -> str:
        return self.state.spec.name

    @property
    def tier(self) -> str:
        return self.state.spec.tier

    def is_available(self, now: float) -> bool:
        return self.state.is_available(now)

    def quota_fraction_used(self, now: float) -> float | None:
        if self.state.quota is None:
            return None
        return self.state.quota.fraction_used(now)

    def route_cost_usd(self, now: float) -> float:
        return self.cost_fn(self.state)

    def hedge_cost_usd(self, now: float) -> float:
        return self.cost_fn(self.state)

    def prior_ttft_mean_ms(self, now: float) -> float | None:
        return None

    def prior_ttft_cdf(self, value_ms: float, now: float) -> float | None:
        return None


def _real_eval_beliefs(states: dict[str, ProviderState]) -> LatencyBeliefs:
    """Beliefs over the exact profile objects held by ``ProviderState``.

    Sharing the objects (not copies) keeps the runner's snapshot/bootstrap
    paths and the baselines reading ``state.profile`` consistent with what
    the router learns on.
    """
    window_sec = (
        next(iter(states.values())).profile.window_sec if states else DEFAULT_PROFILE_WINDOW_SEC
    )
    return LatencyBeliefs(
        window_sec=window_sec,
        error_penalty_ms=RATE_LIMIT_ERROR_PENALTY_MS,
        unprofiled_penalty_ms=UNPROFILED_LATENCY_PENALTY_MS,
        profiles={name: state.profile for name, state in states.items()},
    )


# ---------------------------------------------------------------------------
# Hedging — probability-target backup selection + latest-safe dispatch time.
# ---------------------------------------------------------------------------


def select_checkpoint_backup(
    primary: str,
    states: dict[str, ProviderState],
    ctx: RequestContext,
    slo_sec: float,
    now: float,
    *,
    elapsed_sec: float,
    future_checkpoints_sec: Sequence[float] = (),
    success_target: float = HEDGE_SUCCESS_TARGET,
    cost_fn: Callable[[ProviderState], float] | None = None,
) -> CheckpointHedgeDecision:
    """Evaluate probability-target hedging at a single checkpoint.

    This is the simulator RouteWise tick, verbatim: both environments run
    :meth:`rwsim.core.router.RouteWiseRouter.checkpoint_backup`. Here the
    router is bound to the empirical rolling profiles with no oracle prior,
    zero dispatch overhead, and penalized-belief latency tie-breaks.
    """
    primary_state = states.get(primary)
    if primary_state is None:
        return CheckpointHedgeDecision(backup=None, elapsed_sec=float(elapsed_sec))

    router = RouteWiseRouter(
        alpha=0.0,  # unused by checkpoint evaluation
        slo_ms=float(slo_sec) * 1000.0,
        beliefs=_real_eval_beliefs(states),
        hedging=True,
        hedge_dispatch_overhead_ms=0.0,
        hedge_success_target=success_target,
        hedge_tiebreak_mean="belief",
    )
    if cost_fn is None:
        def cost_fn(state: ProviderState) -> float:
            return request_cost_for_spec(state.spec, ctx)
    views = [_RealProviderView(state, cost_fn) for state in states.values()]
    result = router.checkpoint_backup(
        primary,
        views,
        now,
        elapsed_sec=elapsed_sec,
        future_checkpoints_sec=tuple(future_checkpoints_sec),
    )
    return CheckpointHedgeDecision(
        backup=result.backup,
        elapsed_sec=float(elapsed_sec),
        success_probability=result.success_probability,
        future_feasible=result.future_feasible,
        feasible_count=result.feasible_count,
        candidate_count=result.candidate_count,
    )


# ---------------------------------------------------------------------------
# BasePolicy.
# ---------------------------------------------------------------------------


class BasePolicy:
    """Base class shared by all real-eval routing policies.

    Each policy owns an isolated ``states`` dict so its decisions do not
    interfere with other policies running in the same harness. Sample
    feedback (``add_sample``) and capacity charging (``charge_capacity``)
    are thread-safe.
    """

    name: str = "base"
    use_hedge: bool = False
    requires_latency_profile_bootstrap: ClassVar[bool] = False

    def __init__(
        self,
        specs: list[ProviderSpec],
        slo_ms: float,
        profile_window_sec: float = PROFILE_WINDOW_SEC,
        prefix_cache_routing: bool = False,
        output_predictor: OutputTokenPredictor | None = None,
    ) -> None:
        self.specs = specs
        self.slo_sec = slo_ms / 1000.0
        self.states: dict[str, ProviderState] = {
            spec.name: ProviderState.from_spec(spec, profile_window_sec) for spec in specs
        }
        self.prefix_cache_routing = bool(prefix_cache_routing)
        self.cost_envelope: tuple[float, float] | None = None
        self.output_predictor: OutputTokenPredictor = (
            output_predictor if output_predictor is not None else BucketMeanOutputPredictor()
        )
        self._lock = threading.RLock()
        self._next_capacity_request_id = 1

    def set_cost_envelope(self, envelope: tuple[float, float] | None) -> None:
        """Install a workload-level ``(L, U)`` envelope for shadow pricing.

        ``BudgetRangePolicy`` requires this envelope before routing. It must be
        computed once from the workload's P10/P90 cheapest-API request-cost
        distribution by the runner; there is intentionally no per-request
        fallback because that previously made subscription tiers appear nearly
        free across most of the quota.
        """
        if envelope is None:
            self.cost_envelope = None
            return
        L, U = (float(envelope[0]), float(envelope[1]))
        if not (0.0 < L < U):
            raise ValueError(
                f"cost_envelope must satisfy 0 < L < U; got L={L}, U={U}"
            )
        self.cost_envelope = (L, U)

    def _cost_envelope_or_raise(self) -> tuple[float, float]:
        if self.cost_envelope is None:
            raise RuntimeError(
                f"{self.name} requires a workload cost envelope; call "
                "set_cost_envelope((L, U)) before routing."
            )
        return self.cost_envelope

    def request_cost_for_state(self, state: ProviderState, ctx: RequestContext) -> float:
        """Return this policy's route-time marginal cost for one provider."""
        return self.request_cost_for_spec(state.spec, ctx)

    def _predicted_output_tokens(self, ctx: RequestContext) -> int:
        """Route-time output-token estimate from the policy's predictor.

        Real-eval policies historically used ``ctx.completion_tokens_budget``,
        which is the trace's realized output length — i.e. an oracle. This
        helper replaces the oracle with the policy's online predictor so the
        live LP path matches the simulator (default: bucket mean over
        log-spaced input-length buckets).

        Locking note: the predictor's internal state (e.g. bucket counts and
        running sums) is mutated by ``observe_response`` from result threads
        while ``route`` calls run in parallel. ``self._lock`` serializes both
        sides so a routing read cannot see a half-applied update.
        """
        stub = _PredictionRequest(
            id=0,
            timestamp=0.0,
            request_tokens=int(ctx.prompt_tokens),
        )
        with self._lock:
            prediction = self.output_predictor.predict(stub)
        predicted_tokens = (
            prediction.tokens if hasattr(prediction, "tokens") else prediction.median
        )
        return max(1, round(predicted_tokens))

    def request_cost_for_spec(self, spec: ProviderSpec, ctx: RequestContext) -> float:
        """Return route-time API cost, optionally using trace-reported cache hit."""
        return cache_aware_request_cost_usd(
            spec,
            prompt_tokens=ctx.prompt_tokens,
            completion_tokens=self._predicted_output_tokens(ctx),
            trace_cached_input_tokens=ctx.trace_cached_input_tokens,
            enabled=self.prefix_cache_routing,
        )

    def observe_response(
        self,
        ctx: RequestContext,
        completion_tokens: int | None,
    ) -> None:
        """Update the output-token predictor with a realized response."""
        if completion_tokens is None or completion_tokens <= 0:
            return
        stub = _PredictionRequest(
            id=0,
            timestamp=0.0,
            request_tokens=int(ctx.prompt_tokens),
            response_tokens=int(completion_tokens),
        )
        with self._lock:
            self.output_predictor.update(stub)

    def routing_cache_diagnostics(
        self,
        provider: str | None,
        ctx: RequestContext,
    ) -> tuple[int, float | None]:
        """Return route-time cached tokens and estimated cost for diagnostics."""
        if provider is None:
            return (0, None)
        state = self.states.get(provider)
        if state is None:
            return (0, None)
        cached_tokens = (
            cached_input_tokens(
                prompt_tokens=ctx.prompt_tokens,
                trace_cached_input_tokens=ctx.trace_cached_input_tokens,
            )
            if self.prefix_cache_routing
            else 0
        )
        estimated_cost = cache_aware_request_cost_usd(
            state.spec,
            prompt_tokens=ctx.prompt_tokens,
            completion_tokens=self._predicted_output_tokens(ctx),
            trace_cached_input_tokens=ctx.trace_cached_input_tokens,
            enabled=self.prefix_cache_routing,
        )
        return (cached_tokens, estimated_cost)

    def add_sample(
        self,
        provider: str,
        ts: float,
        ttft_ms: float,
        error_type: str | None = None,
    ) -> None:
        with self._lock:
            state = self.states.get(provider)
            if state is not None:
                state.profile.add_sample(ts, ttft_ms, error_type)

    def charge_capacity(
        self,
        provider: str,
        now: float,
        expected_service_sec: float,
    ) -> int | None:
        """Atomically reserve provider capacity for one dispatched request."""
        with self._lock:
            state = self.states.get(provider)
            if state is None:
                return None
            if not state.is_available(now):
                raise CapacityUnavailableError(provider)
            if state.quota is not None:
                state.quota.charge(now)
            if state.concurrency is not None:
                request_id = self._next_capacity_request_id
                self._next_capacity_request_id += 1
                del expected_service_sec
                return state.concurrency.admit(request_id=request_id, now=now)
            return None

    def route_and_charge_capacity(
        self,
        now: float,
        ctx: RequestContext,
        expected_service_sec: float,
    ) -> tuple[RoutingDecision, int | None]:
        """Route and reserve primary capacity while holding the policy lock."""
        capacity_error: CapacityUnavailableError | None = None
        with self._lock:
            route_attempts = max(1, len(self.states) + 1)
            for _ in range(route_attempts):
                decision = self.route(now, ctx)
                if decision.primary is None:
                    return decision, None
                try:
                    capacity_id = self.charge_capacity(
                        decision.primary,
                        now,
                        expected_service_sec,
                    )
                    return decision, capacity_id
                except CapacityUnavailableError as exc:
                    capacity_error = exc
            if capacity_error is not None:
                raise capacity_error
        return RoutingDecision(primary=None, notes="capacity_unavailable"), None

    def fallback_candidate_and_charge_capacity(
        self,
        now: float,
        ctx: RequestContext,
        *,
        excluded: set[str],
        expected_service_sec: float,
    ) -> tuple[str | None, int | None]:
        """Pick and reserve a 429 fallback candidate under one policy lock."""
        with self._lock:
            for _ in range(max(1, len(self.states))):
                fallback_candidates = self.rate_limit_fallback_candidates(
                    now,
                    ctx,
                    excluded=excluded,
                )
                provider = next(
                    (candidate for candidate in fallback_candidates if candidate not in excluded),
                    None,
                )
                if provider is None:
                    return None, None
                try:
                    capacity_id = self.charge_capacity(
                        provider,
                        now,
                        expected_service_sec,
                    )
                    return provider, capacity_id
                except CapacityUnavailableError:
                    excluded.add(provider)
            return None, None

    def checkpoint_backup_and_charge_capacity(
        self,
        *,
        primary: str,
        ctx: RequestContext,
        slo_sec: float,
        now: float,
        elapsed_sec: float,
        future_checkpoints_sec: tuple[float, ...],
        expected_service_sec: float,
        cost_fn: Callable[[ProviderState], float] | None = None,
    ) -> tuple[CheckpointHedgeDecision, int | None]:
        """Select and reserve checkpoint backup capacity under one policy lock."""
        with self._lock:
            checkpoint_decision = select_checkpoint_backup(
                primary=primary,
                states=self.states,
                ctx=ctx,
                slo_sec=slo_sec,
                now=now,
                elapsed_sec=elapsed_sec,
                future_checkpoints_sec=future_checkpoints_sec,
                cost_fn=cost_fn,
            )
            if checkpoint_decision.backup is None:
                return checkpoint_decision, None
            capacity_id = self.charge_capacity(
                checkpoint_decision.backup,
                now,
                expected_service_sec,
            )
            return checkpoint_decision, capacity_id

    def release_capacity(
        self,
        provider: str | None,
        request_id: int | None,
        now: float | None = None,
    ) -> None:
        """Release a concurrency lease once the real request has finished."""
        if provider is None or request_id is None:
            return
        with self._lock:
            state = self.states.get(provider)
            if state is not None and state.concurrency is not None:
                state.concurrency.release(request_id, now)

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        raise NotImplementedError

    def rate_limit_fallback_candidates(
        self,
        now: float,
        ctx: RequestContext,
        *,
        excluded: set[str],
    ) -> list[str]:
        """Policy-aware retry order after provider-local HTTP 429s."""
        candidates = [
            state.spec.name
            for state in self.states.values()
            if state.spec.name not in excluded and state.is_available(now)
        ]
        candidates.sort()
        return candidates


# ---------------------------------------------------------------------------
# Baselines.
# ---------------------------------------------------------------------------


class OrAutoPolicy(BasePolicy):
    """OpenRouter's default routing — one sentinel decision."""

    name = "or_auto"

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        has_or = any(s.transport_cfg.transport == "openrouter" for s in self.specs)
        if not has_or:
            return RoutingDecision(primary=None, notes="no_openrouter_spec")
        return RoutingDecision(primary=OR_AUTO_SENTINEL, notes="or_default")

    def rate_limit_fallback_candidates(
        self,
        now: float,
        ctx: RequestContext,
        *,
        excluded: set[str],
    ) -> list[str]:
        del now, ctx, excluded
        return []


class _OpenRouterSortPolicy(BasePolicy):
    """Base for OpenRouter ``sort=<mode>`` baselines.

    Subclasses set ``sort_mode`` and ``name``. Returns a sentinel that the
    runner translates into the right ``provider.sort`` payload field.
    """

    sort_mode: str = "latency"
    sentinel: str = OR_SORT_LATENCY_SENTINEL

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        has_or = any(s.transport_cfg.transport == "openrouter" for s in self.specs)
        if not has_or:
            return RoutingDecision(primary=None, notes="no_openrouter_spec")
        return RoutingDecision(primary=self.sentinel, notes=f"or_sort_{self.sort_mode}")

    def rate_limit_fallback_candidates(
        self,
        now: float,
        ctx: RequestContext,
        *,
        excluded: set[str],
    ) -> list[str]:
        del now, ctx, excluded
        return []


class OrSortLatencyPolicy(_OpenRouterSortPolicy):
    """OpenRouter ``sort=latency``."""

    name = "or_sort_latency"
    sort_mode = "latency"
    sentinel = OR_SORT_LATENCY_SENTINEL


class OrSortCostPolicy(_OpenRouterSortPolicy):
    """OpenRouter ``sort=price``. Named cost to match paper terminology."""

    name = "or_sort_cost"
    sort_mode = "price"
    sentinel = OR_SORT_COST_SENTINEL


class OrSortThroughputPolicy(_OpenRouterSortPolicy):
    """OpenRouter ``sort=throughput``."""

    name = "or_sort_throughput"
    sort_mode = "throughput"
    sentinel = OR_SORT_THROUGHPUT_SENTINEL


_GREEDY_COST_TIER_RANK: dict[str, int] = {
    "concurrency": 0,
    "quota": 1,
    "api": 2,
}
_UNKNOWN_TIER_RANK = 99


class _GreedyBase(BasePolicy):
    """Shared logic for dynamic greedy baselines.

    Subclasses set the candidate filter (joint pool vs. OpenRouter-only)
    and the sort key. Splitting joint vs. OR-only is paper-relevant: the
    joint pool baseline lets ``S_Q`` / ``S_C`` providers compete; the
    OR-only variant pins comparison to the metered-API tier alone, which
    is the apples-to-apples baseline against OpenRouter's native sort modes.
    """

    or_only: bool = False

    def _candidates(self, now: float) -> list[ProviderState]:
        states = list(self.states.values())
        if self.or_only:
            states = [s for s in states if s.spec.transport_cfg.transport == "openrouter"]
        return [s for s in states if s.is_available(now)]


class GreedyCostPolicy(_GreedyBase):
    """Joint-pool: cheapest currently-available provider across ALL tiers.

    Subscription tiers have zero marginal cost, so ties intentionally prefer
    perishable concurrency capacity before bankable quota before metered API:
    ``S_C`` -> ``S_Q`` -> ``S_A``. This mirrors the simulator's
    paper-facing ``greedy_cost`` baseline.

    Tie-break is purely cost-driven: ``(cost, tier_rank, spec.name)``.
    Latency is intentionally NOT consulted here — ``greedy_cost`` must not
    benefit from latency information that should be the territory of
    ``greedy_latency`` / ``budget_range_*`` policies. When two providers
    have identical cost and tier, the alphabetically-first ``spec.name``
    wins (deterministic and independent of profile state).
    """

    name = "greedy_cost"
    or_only = False

    def _ranked_candidates(self, now: float, ctx: RequestContext) -> list[ProviderState]:
        candidates = self._candidates(now)
        scored = []
        for state in candidates:
            cost = self.request_cost_for_state(state, ctx)
            scored.append(
                (
                    cost,
                    _GREEDY_COST_TIER_RANK.get(state.spec.tier, _UNKNOWN_TIER_RANK),
                    state.spec.name,
                    state,
                )
            )
        scored.sort(key=lambda item: item[:-1])
        return [item[-1] for item in scored]

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        candidates = self._ranked_candidates(now, ctx)
        if not candidates:
            return RoutingDecision(primary=None, notes="none_available")
        return RoutingDecision(primary=candidates[0].spec.name, notes="greedy_cost")

    def rate_limit_fallback_candidates(
        self,
        now: float,
        ctx: RequestContext,
        *,
        excluded: set[str],
    ) -> list[str]:
        return [
            state.spec.name
            for state in self._ranked_candidates(now, ctx)
            if state.spec.name not in excluded
        ]


class OrGreedyCostPolicy(GreedyCostPolicy):
    """OR-only greedy cost (filters to ``transport == 'openrouter'``)."""

    name = "or_greedy_cost"
    or_only = True


class GreedyLatencyPolicy(_GreedyBase):
    """Joint-pool: lowest empirical-latency provider across ALL tiers."""

    name = "greedy_latency"
    or_only = False
    requires_latency_profile_bootstrap = True

    def _ranked_candidates(self, now: float, ctx: RequestContext) -> list[ProviderState]:
        candidates = self._candidates(now)
        scored = []
        for state in candidates:
            latency, _ = _body_latency_proxy_ms(
                state,
                now,
                error_penalty_ms=RATE_LIMIT_ERROR_PENALTY_MS,
            )
            cost = self.request_cost_for_state(state, ctx)
            scored.append(
                (
                    latency if latency is not None else float("inf"),
                    cost,
                    state.spec.name,
                    state,
                )
            )
        scored.sort(key=lambda item: item[:-1])
        return [item[-1] for item in scored]

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        candidates = self._ranked_candidates(now, ctx)
        if not candidates:
            return RoutingDecision(primary=None, notes="none_available")
        return RoutingDecision(primary=candidates[0].spec.name, notes="greedy_latency")

    def rate_limit_fallback_candidates(
        self,
        now: float,
        ctx: RequestContext,
        *,
        excluded: set[str],
    ) -> list[str]:
        return [
            state.spec.name
            for state in self._ranked_candidates(now, ctx)
            if state.spec.name not in excluded
        ]


class OrGreedyLatencyPolicy(GreedyLatencyPolicy):
    """OR-only greedy latency (filters to ``transport == 'openrouter'``)."""

    name = "or_greedy_latency"
    or_only = True


class RandomPolicy(BasePolicy):
    """Joint-pool random baseline over currently available providers."""

    name = "random"

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        candidates = [state.spec.name for state in self.states.values() if state.is_available(now)]
        if not candidates:
            return RoutingDecision(primary=None, notes="none_available")
        candidates.sort()
        choice = random.Random(int(now * 1e6)).choice(candidates)
        return RoutingDecision(primary=choice, notes="random")

    def rate_limit_fallback_candidates(
        self,
        now: float,
        ctx: RequestContext,
        *,
        excluded: set[str],
    ) -> list[str]:
        candidates = [
            state.spec.name
            for state in self.states.values()
            if state.spec.name not in excluded and state.is_available(now)
        ]
        candidates.sort()
        rng = random.Random(int(now * 1e6) + len(excluded))
        rng.shuffle(candidates)
        return candidates


class TierFirstPolicy(BasePolicy):
    """Fill preferred tier first, spill on exhaustion."""

    tier_priority: tuple[str, ...] = ("quota", "concurrency", "api")
    requires_latency_profile_bootstrap = True

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        for tier in self.tier_priority:
            tier_candidates = [
                state
                for state in self.states.values()
                if state.spec.tier == tier and state.is_available(now)
            ]
            if not tier_candidates:
                continue
            tier_candidates.sort(
                key=lambda s: (s.profile.median_ms(now) or float("inf"), s.spec.name)
            )
            return RoutingDecision(
                primary=tier_candidates[0].spec.name,
                notes=f"{self.tier_priority[0]}_first",
            )
        return RoutingDecision(primary=None, notes="none_available")


class QuotaFirstPolicy(TierFirstPolicy):
    name = "quota_first"
    tier_priority = ("quota", "concurrency", "api")


class ConcurrencyFirstPolicy(TierFirstPolicy):
    name = "concurrency_first"
    tier_priority = ("concurrency", "quota", "api")


# ---------------------------------------------------------------------------
# Current paper line: LP-RangeBudget + Hedge-ProbTarget.
# ---------------------------------------------------------------------------


class BudgetRangePolicy(BasePolicy):
    """``LP-RangeBudget`` body router (current paper main method).

    Body selector: ``min sum pi_j T̄_j  s.t.  sum pi_j c_eff_j <= B_alpha``
    where ``B_alpha = (1 - alpha) c_min + alpha c_max``.

    The selector itself is :class:`rwsim.core.router.RouteWiseRouter` — the
    same code the simulator's ``RouteWisePolicy`` runs. This adapter binds it
    to the empirical rolling profiles (no oracle priors) and translates
    ``RouteResult`` into the recorder-facing ``RoutingDecision``.
    """

    use_hedge = False
    name_suffix = ""
    requires_latency_profile_bootstrap = True

    def __init__(
        self,
        specs: list[ProviderSpec],
        slo_ms: float,
        profile_window_sec: float = PROFILE_WINDOW_SEC,
        budget_alpha_percent: int = 100,
        budget_percentile: int | None = None,
        prefix_cache_routing: bool = False,
        output_predictor: OutputTokenPredictor | None = None,
    ) -> None:
        super().__init__(
            specs,
            slo_ms,
            profile_window_sec,
            prefix_cache_routing=prefix_cache_routing,
            output_predictor=output_predictor,
        )
        if budget_percentile is not None:
            budget_alpha_percent = budget_percentile
        if not 0 <= budget_alpha_percent <= 100:
            raise ValueError(
                f"budget_alpha_percent must be in [0, 100]; got {budget_alpha_percent}"
            )
        self.budget_alpha_percent = int(budget_alpha_percent)
        self.budget_percentile = self.budget_alpha_percent
        self.name = f"budget_range_alpha{self.budget_alpha_percent}{self.name_suffix}"
        self.router = RouteWiseRouter(
            alpha=self.budget_alpha_percent / 100.0,
            slo_ms=float(slo_ms),
            beliefs=_real_eval_beliefs(self.states),
            cost_envelope=None,  # installed by set_cost_envelope before routing
            hedging=self.use_hedge,
            sampler=_time_seeded_sampler,
            hedge_dispatch_overhead_ms=0.0,
            hedge_success_target=HEDGE_SUCCESS_TARGET,
            hedge_tiebreak_mean="belief",
        )

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        feasible = [s for s in self.states.values() if s.is_available(now)]
        if not feasible:
            return RoutingDecision(primary=None, notes="none_available")

        result = self.router.route(self._views(feasible, ctx), now)
        if result.lp_status is LPStatus.FALLBACK_MIN_COST:
            return RoutingDecision(
                primary=result.primary,
                lp_weights=result.weights,
                lp_status="fallback_in_budget_range",
                budget_usd=float(result.budget),
                reference_cost_usd=float(result.c_max),
                c_eff_map=result.c_eff,
                notes="fallback_affordable_range",
            )
        return RoutingDecision(
            primary=result.primary,
            lp_weights=result.weights,
            lp_status="optimal",
            budget_usd=float(result.budget),
            reference_cost_usd=float(result.c_max),
            c_eff_map=result.c_eff,
            tier_mix=_tier_mix_from_weights(result.weights, self.states),
            notes=self.name,
        )

    def rate_limit_fallback_candidates(
        self,
        now: float,
        ctx: RequestContext,
        *,
        excluded: set[str],
    ) -> list[str]:
        """Re-solve the RouteWise body objective after provider-local 429s.

        A 429 makes the provider temporarily infeasible for this request. The
        fallback order should therefore respect the same cost budget and
        latency objective as ``route()``, not fall back to alphabetical order.
        """
        feasible = [
            state
            for state in self.states.values()
            if state.spec.name not in excluded and state.is_available(now)
        ]
        if not feasible:
            return []
        return self.router.fallback_order(self._views(feasible, ctx), now)

    def _views(
        self,
        states: list[ProviderState],
        ctx: RequestContext,
    ) -> list[_RealProviderView]:
        """Request-bound views; envelope installation is checked here."""
        self.router.cost_envelope = self._cost_envelope_or_raise()
        request_costs = {
            state.spec.name: self.request_cost_for_state(state, ctx) for state in states
        }

        def cost_fn(state: ProviderState) -> float:
            return request_costs[state.spec.name]

        return [_RealProviderView(state, cost_fn) for state in states]


class BudgetRangeHedgePolicy(BudgetRangePolicy):
    """``LP-RangeBudget`` + ``Hedge-ProbTarget``."""

    use_hedge = True
    name_suffix = "_hedge"


# ---------------------------------------------------------------------------
# Policy registry — used by the runner CLI.
# ---------------------------------------------------------------------------


def build_policy(
    name: str,
    specs: list[ProviderSpec],
    slo_ms: float,
    profile_window_sec: float = PROFILE_WINDOW_SEC,
    prefix_cache_routing: bool = False,
    output_predictor: OutputTokenPredictor | None = None,
) -> BasePolicy:
    """Construct one policy by name.

    Recognized names:
      * Baselines: ``or_auto``, ``or_sort_latency``, ``or_sort_cost``,
        ``greedy_cost``, ``greedy_latency``, ``random``, ``quota_first``,
        ``concurrency_first``
      * Paper line: ``budget_range_alpha<PP>`` and
        ``budget_range_alpha<PP>_hedge`` (PP in ``[0, 100]``)
    """
    common = {
        "specs": specs,
        "slo_ms": slo_ms,
        "profile_window_sec": profile_window_sec,
        "prefix_cache_routing": prefix_cache_routing,
        "output_predictor": output_predictor,
    }
    if name == "or_auto":
        return OrAutoPolicy(**common)
    if name == "or_sort_latency":
        return OrSortLatencyPolicy(**common)
    if name == "or_sort_cost":
        return OrSortCostPolicy(**common)
    if name == "or_sort_throughput":
        return OrSortThroughputPolicy(**common)
    if name == "greedy_cost":
        return GreedyCostPolicy(**common)
    if name == "greedy_latency":
        return GreedyLatencyPolicy(**common)
    if name == "or_greedy_cost":
        return OrGreedyCostPolicy(**common)
    if name == "or_greedy_latency":
        return OrGreedyLatencyPolicy(**common)
    if name == "random":
        return RandomPolicy(**common)
    if name == "quota_first":
        return QuotaFirstPolicy(**common)
    if name == "concurrency_first":
        return ConcurrencyFirstPolicy(**common)

    if name.startswith("budget_range_alpha") and name.endswith("_hedge"):
        try:
            alpha = int(name[len("budget_range_alpha") : -len("_hedge")])
        except ValueError as exc:
            raise ValueError(f"Bad budget_range name: {name!r}") from exc
        return BudgetRangeHedgePolicy(**common, budget_alpha_percent=alpha)
    if name.startswith("budget_range_alpha"):
        try:
            alpha = int(name[len("budget_range_alpha") :])
        except ValueError as exc:
            raise ValueError(f"Bad budget_range name: {name!r}") from exc
        return BudgetRangePolicy(**common, budget_alpha_percent=alpha)
    if name.startswith("budget_range_p") and name.endswith("_hedge"):
        try:
            alpha = int(name[len("budget_range_p") : -len("_hedge")])
        except ValueError as exc:
            raise ValueError(f"Bad budget_range name: {name!r}") from exc
        return BudgetRangeHedgePolicy(**common, budget_alpha_percent=alpha)
    if name.startswith("budget_range_p"):
        try:
            alpha = int(name[len("budget_range_p") :])
        except ValueError as exc:
            raise ValueError(f"Bad budget_range name: {name!r}") from exc
        return BudgetRangePolicy(**common, budget_alpha_percent=alpha)

    raise ValueError(f"Unknown policy name: {name!r}")


__all__ = [
    "HEDGE_SUCCESS_TARGET",
    "OR_AUTO_SENTINEL",
    "OR_SORT_COST_SENTINEL",
    "OR_SORT_LATENCY_SENTINEL",
    "OR_SORT_SENTINEL_TO_MODE",
    "OR_SORT_THROUGHPUT_SENTINEL",
    "UNPROFILED_LATENCY_PENALTY_MS",
    "BasePolicy",
    "BudgetRangeHedgePolicy",
    "BudgetRangePolicy",
    "CapacityUnavailableError",
    "CheckpointHedgeDecision",
    "ConcurrencyFirstPolicy",
    "GreedyCostPolicy",
    "GreedyLatencyPolicy",
    "OrAutoPolicy",
    "OrGreedyCostPolicy",
    "OrGreedyLatencyPolicy",
    "OrSortCostPolicy",
    "OrSortLatencyPolicy",
    "OrSortThroughputPolicy",
    "QuotaFirstPolicy",
    "RandomPolicy",
    "RequestContext",
    "RoutingDecision",
    "TierFirstPolicy",
    "build_policy",
    "request_cost_for_spec",
    "select_checkpoint_backup",
]
