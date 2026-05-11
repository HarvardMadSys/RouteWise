"""Adapter routing policies for real online evaluation.

ARCHITECTURE NOTE — read before extending.

This module is a **real-eval adapter**, NOT a long-term routing system.
It exists to let real-online experiments call the same algorithm shapes
as the simulator without forcing a premature unification of three
incompatible policy frameworks (``rwsim.policies``, the
historical simulator-grid sidecar, and these phase6-derived classes). When the
canonical policy pipeline in ``rwsim.policies`` is mature enough to express
the live-evaluation harness, this module should be retired or re-grounded
on it.

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
  - ``BudgetRangeHedgePolicy(p)`` : range-normalized cost budget ``B_p =
    c_min + p (c_max - c_min)``, probability-target hedge

The ``BudgetRange*`` selector is a hand-port from the retired simulator-grid
range-budget selector. The simulator version was distribution-aware; this real
version uses the empirical rolling profile.
"""

from __future__ import annotations

import logging
import math
import random
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scipy.optimize import linprog

from experiments.real_evaluation.inventory import (
    PROFILE_WINDOW_SEC,
    ProviderSpec,
    ProviderState,
)
from experiments.real_evaluation.prefix_cache import (
    cache_aware_request_cost_usd,
    cached_input_tokens,
    record_prefix_cache_dispatch,
)
from experiments.real_evaluation.shadow_price import (
    effective_cost,
    request_marginal_cost,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

LP_EPS: float = 1e-9
_COST_TIEBREAK_MS: float = 1e-3
HEDGE_SUCCESS_TARGET: float = 0.99
HEDGE_DISPATCH_OVERHEAD_SEC: float = 0.05
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


@dataclass
class RequestContext:
    """Per-request inputs the policy needs to make a routing decision."""

    prompt_tokens: int
    completion_tokens_budget: int
    prefix_id: str | None = None


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


def _body_latency_proxy_ms(state: ProviderState, now: float) -> tuple[float, bool]:
    """Empirical body-latency estimate ``T̄_j(t)`` for one provider.

    Returns ``(mean_ms, used_fallback)`` where ``used_fallback`` is True if
    the rolling profile had too few samples and we fell back to the
    median or a large sentinel.
    """
    n = state.profile.sample_count(now)
    if n >= BODY_MEAN_MIN_SAMPLES:
        mean = state.profile.mean_ms(now)
        if mean is not None:
            return float(mean), False
    median = state.profile.median_ms(now)
    if median is not None:
        return float(median), True
    return float("inf"), True


def _solve_simplex_lp(
    objective: Sequence[float],
    *,
    upper_constraint: Sequence[float] | None = None,
    upper_bound: float | None = None,
) -> tuple[bool, np.ndarray | None]:
    """Solve ``min c·x  s.t.  a·x <= b,  sum(x) = 1,  x >= 0``."""
    n = len(objective)
    if n == 0:
        return False, None
    A_ub = None
    b_ub = None
    if upper_constraint is not None and upper_bound is not None:
        A_ub = np.array(upper_constraint, dtype=float).reshape(1, -1)
        b_ub = np.array([upper_bound], dtype=float)
    A_eq = np.ones((1, n), dtype=float)
    b_eq = np.array([1.0], dtype=float)
    bounds = [(0.0, 1.0) for _ in range(n)]
    try:
        result = linprog(
            c=np.array(objective, dtype=float),
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
    except Exception as exc:
        logging.warning("linprog failed: %s", exc)
        return False, None
    if not result.success:
        return False, None
    return True, np.asarray(result.x, dtype=float)


def _cost_tiebroken_objective(
    latency_objective_ms: Sequence[float],
    effective_costs: Sequence[float],
) -> list[float]:
    """Prefer lower effective cost when LP latency objectives are equal."""
    if len(latency_objective_ms) != len(effective_costs):
        raise ValueError("latency objective and cost arrays must have the same length")
    if len(latency_objective_ms) == 0:
        return []

    latencies = np.asarray(latency_objective_ms, dtype=float)
    costs = np.asarray(effective_costs, dtype=float)
    cost_span = float(costs.max() - costs.min())
    if cost_span <= LP_EPS:
        return [float(value) for value in latencies]

    normalized_costs = (costs - costs.min()) / cost_span
    return [float(value) for value in latencies + _COST_TIEBREAK_MS * normalized_costs]


def _normalize_weights(names: list[str], vector: np.ndarray) -> dict[str, float]:
    raw = {names[i]: float(vector[i]) for i in range(len(names)) if vector[i] > 1e-6}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Hedging — probability-target backup selection + latest-safe dispatch time.
# ---------------------------------------------------------------------------


def _combined_hedge_success_probability(
    primary_state: ProviderState,
    backup_state: ProviderState,
    *,
    elapsed_sec: float,
    slo_sec: float,
    now: float,
    dispatch_overhead_sec: float = HEDGE_DISPATCH_OVERHEAD_SEC,
) -> float:
    """Probability that hedging now keeps the request within SLO.

    Mirrors ``rwsim.policies.hedging.combined_success_probability`` for the
    real-eval ``ProviderState`` representation.
    """
    elapsed_ms = max(0.0, float(elapsed_sec)) * 1000.0
    slo_ms = slo_sec * 1000.0
    primary_cdf_t = primary_state.profile.cdf_at(elapsed_ms, now)
    primary_cdf_slo = primary_state.profile.cdf_at(slo_ms, now)
    primary_survival_t = max(1.0 - primary_cdf_t, 0.0)
    if primary_survival_t <= LP_EPS:
        return 0.0

    p_not_violate = max(primary_cdf_slo - primary_cdf_t, 0.0) / primary_survival_t
    p_violate = max(1.0 - primary_cdf_slo, 0.0) / primary_survival_t
    remaining_ms = (slo_sec - float(elapsed_sec) - dispatch_overhead_sec) * 1000.0
    backup_success = 0.0 if remaining_ms <= 0.0 else backup_state.profile.cdf_at(remaining_ms, now)
    return float(min(max(p_not_violate + p_violate * backup_success, 0.0), 1.0))


def compute_hedge_time_sec(
    primary_state: ProviderState,
    backup_state: ProviderState,
    slo_sec: float,
    now: float,
    *,
    success_target: float = HEDGE_SUCCESS_TARGET,
    dispatch_overhead_sec: float = HEDGE_DISPATCH_OVERHEAD_SEC,
    grid_step_sec: float = 0.05,
) -> float:
    """Return the latest backup-dispatch wait time meeting target success.

    Searches a uniform grid for the latest ``t`` such that

        P(not violate | wait t) + P(violate | wait t) * P(backup ok in remaining)
        >= success_target.

    Returns ``math.inf`` if no ``t`` in ``[0, slo_sec - dispatch_overhead]``
    meets the target — the runner interprets that as "do not hedge".

    Empirical version of ``rwsim``'s latest-safe probability-target hedging.
    """
    latest_safe: float | None = None
    max_elapsed_sec = max(0.0, slo_sec - dispatch_overhead_sec)
    grid = np.arange(0.0, max_elapsed_sec + 1e-9, grid_step_sec)
    if len(grid) == 0:
        grid = np.array([0.0])
    for elapsed_sec in grid:
        combined = _combined_hedge_success_probability(
            primary_state,
            backup_state,
            elapsed_sec=float(elapsed_sec),
            slo_sec=slo_sec,
            now=now,
            dispatch_overhead_sec=dispatch_overhead_sec,
        )
        if combined >= success_target:
            latest_safe = float(elapsed_sec)
    if latest_safe is not None:
        return latest_safe
    return float("inf")


def select_safe_cheapest_backup(
    primary: str,
    states: dict[str, ProviderState],
    ctx: RequestContext,
    slo_sec: float,
    now: float,
    *,
    success_target: float = HEDGE_SUCCESS_TARGET,
    dispatch_overhead_sec: float = HEDGE_DISPATCH_OVERHEAD_SEC,
    cost_fn: Callable[[ProviderState], float] | None = None,
) -> str | None:
    """Pick the cheapest feasible probability-target backup.

    This is the real-eval adapter of simulator ``select_probability_backup``.
    A backup is feasible only if some latest-safe dispatch time lets the
    primary+backup pair satisfy the combined SLO success target. If no backup is
    feasible, return ``None`` and let the runner keep the request primary-only.
    """
    primary_state = states.get(primary)
    if primary_state is None:
        return None

    feasible: list[tuple[float, float, float, str, ProviderState]] = []
    for name, state in states.items():
        if name == primary or not state.is_available(now):
            continue
        hedge_delay_sec = compute_hedge_time_sec(
            primary_state,
            state,
            slo_sec,
            now,
            success_target=success_target,
            dispatch_overhead_sec=dispatch_overhead_sec,
        )
        if not math.isfinite(hedge_delay_sec):
            continue
        success_probability = _combined_hedge_success_probability(
            primary_state,
            state,
            elapsed_sec=hedge_delay_sec,
            slo_sec=slo_sec,
            now=now,
            dispatch_overhead_sec=dispatch_overhead_sec,
        )
        cost = cost_fn(state) if cost_fn is not None else request_cost_for_spec(state.spec, ctx)
        mean_ms, _ = _body_latency_proxy_ms(state, now)
        if not math.isfinite(mean_ms):
            mean_ms = UNPROFILED_LATENCY_PENALTY_MS
        feasible.append((cost, -success_probability, mean_ms, state.spec.name, state))
    if not feasible:
        return None
    feasible.sort(key=lambda item: item[:-1])
    return feasible[0][-1].spec.name


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
    ) -> None:
        self.specs = specs
        self.slo_sec = slo_ms / 1000.0
        self.states: dict[str, ProviderState] = {
            spec.name: ProviderState.from_spec(spec, profile_window_sec) for spec in specs
        }
        self.prefix_cache_routing = bool(prefix_cache_routing)
        self.provider_prefix_cache: dict[str, dict[str, int]] = {}
        self.cost_envelope: tuple[float, float] | None = None
        self._lock = threading.Lock()
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

    def request_cost_for_spec(self, spec: ProviderSpec, ctx: RequestContext) -> float:
        """Return route-time API cost, optionally using policy-local cache state."""
        with self._lock:
            return cache_aware_request_cost_usd(
                spec,
                prompt_tokens=ctx.prompt_tokens,
                completion_tokens=ctx.completion_tokens_budget,
                prefix_id=ctx.prefix_id,
                provider_prefix_cache=self.provider_prefix_cache,
                enabled=self.prefix_cache_routing,
            )

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
        with self._lock:
            cached_tokens = (
                cached_input_tokens(
                    provider_name=provider,
                    prefix_id=ctx.prefix_id,
                    prompt_tokens=ctx.prompt_tokens,
                    provider_prefix_cache=self.provider_prefix_cache,
                )
                if self.prefix_cache_routing
                else 0
            )
            estimated_cost = cache_aware_request_cost_usd(
                state.spec,
                prompt_tokens=ctx.prompt_tokens,
                completion_tokens=ctx.completion_tokens_budget,
                prefix_id=ctx.prefix_id,
                provider_prefix_cache=self.provider_prefix_cache,
                enabled=self.prefix_cache_routing,
            )
        return (cached_tokens, estimated_cost)

    def record_prefix_cache_dispatch(
        self,
        provider: str | None,
        ctx: RequestContext,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Update this policy's provider-local cache state after dispatch."""
        if not self.prefix_cache_routing or provider is None or provider not in self.states:
            return
        with self._lock:
            record_prefix_cache_dispatch(
                provider_name=provider,
                prefix_id=ctx.prefix_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider_prefix_cache=self.provider_prefix_cache,
            )

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
        """Record one dispatched request against quota / concurrency state."""
        with self._lock:
            state = self.states.get(provider)
            if state is None:
                return None
            if state.quota is not None:
                state.quota.charge(now)
            if state.concurrency is not None:
                request_id = self._next_capacity_request_id
                self._next_capacity_request_id += 1
                del expected_service_sec
                return state.concurrency.admit(request_id=request_id, now=now)
            return None

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
        del ctx
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
    """

    name = "greedy_cost"
    or_only = False

    def _ranked_candidates(self, now: float, ctx: RequestContext) -> list[ProviderState]:
        candidates = self._candidates(now)
        scored = []
        for state in candidates:
            cost = self.request_cost_for_state(state, ctx)
            latency = state.profile.mean_ms(now)
            if latency is None:
                latency = state.profile.median_ms(now)
            scored.append(
                (
                    cost,
                    _GREEDY_COST_TIER_RANK.get(state.spec.tier, _UNKNOWN_TIER_RANK),
                    latency if latency is not None else float("inf"),
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
            latency = state.profile.mean_ms(now)
            if latency is None:
                latency = state.profile.median_ms(now)
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
        del ctx
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

    Body selector: ``min sum pi_j T̄_j  s.t.  sum pi_j c_eff_j <= B_p``
    where ``B_p = c_min + (p/100) * (c_max - c_min)``.

    Hand-ported from the retired simulator-grid range-budget selector. The
    simulator version read distributional means from provider distributions;
    here we use the empirical ``LatencyProfile.mean_ms``.
    """

    use_hedge = False
    name_suffix = ""
    requires_latency_profile_bootstrap = True

    def __init__(
        self,
        specs: list[ProviderSpec],
        slo_ms: float,
        profile_window_sec: float = PROFILE_WINDOW_SEC,
        budget_percentile: int = 100,
        prefix_cache_routing: bool = False,
    ) -> None:
        super().__init__(
            specs,
            slo_ms,
            profile_window_sec,
            prefix_cache_routing=prefix_cache_routing,
        )
        if not 0 <= budget_percentile <= 100:
            raise ValueError(f"budget_percentile must be in [0, 100]; got {budget_percentile}")
        self.budget_percentile = int(budget_percentile)
        self.name = f"budget_range_p{self.budget_percentile}{self.name_suffix}"

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        feasible = [s for s in self.states.values() if s.is_available(now)]
        if not feasible:
            return RoutingDecision(primary=None, notes="none_available")

        L, U = self._cost_envelope_or_raise()
        request_costs = {s.spec.name: self.request_cost_for_state(s, ctx) for s in feasible}
        c_eff = {
            s.spec.name: effective_cost(s, request_costs[s.spec.name], now, U=U, L=L)
            for s in feasible
        }

        tbar: dict[str, float] = {}
        for s in feasible:
            mean_ms, _ = _body_latency_proxy_ms(s, now)
            if not math.isfinite(mean_ms):
                mean_ms = UNPROFILED_LATENCY_PENALTY_MS
            tbar[s.spec.name] = mean_ms

        # Range budget: B_p = c_min + p (c_max - c_min).
        c_values = list(c_eff.values())
        c_min = min(c_values)
        c_max = max(c_values)
        p = self.budget_percentile / 100.0
        budget = float(c_min + p * (c_max - c_min))

        names = [s.spec.name for s in feasible]
        objective = _cost_tiebroken_objective(
            [tbar[name] for name in names],
            [c_eff[name] for name in names],
        )
        success, vector = _solve_simplex_lp(
            objective=objective,
            upper_constraint=[c_eff[name] for name in names],
            upper_bound=budget,
        )
        if success and vector is not None:
            weights = _normalize_weights(names, vector)
            if weights:
                primary = _sample_weighted(weights, rng=random.Random(int(now * 1e6)))
                return RoutingDecision(
                    primary=primary,
                    lp_weights=weights,
                    lp_status="optimal",
                    budget_usd=float(budget),
                    reference_cost_usd=float(c_max),
                    c_eff_map=c_eff,
                    tier_mix=_tier_mix_from_weights(weights, self.states),
                    notes=self.name,
                )

        return _fallback_in_budget(feasible, c_eff, budget, c_max, fallback_label="range")

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

        L, U = self._cost_envelope_or_raise()
        request_costs = {
            state.spec.name: self.request_cost_for_state(state, ctx) for state in feasible
        }
        c_eff = {
            state.spec.name: effective_cost(state, request_costs[state.spec.name], now, U=U, L=L)
            for state in feasible
        }
        tbar: dict[str, float] = {}
        for state in feasible:
            mean_ms, _ = _body_latency_proxy_ms(state, now)
            if not math.isfinite(mean_ms):
                mean_ms = UNPROFILED_LATENCY_PENALTY_MS
            tbar[state.spec.name] = mean_ms

        c_values = list(c_eff.values())
        c_min = min(c_values)
        c_max = max(c_values)
        budget = float(c_min + (self.budget_percentile / 100.0) * (c_max - c_min))
        names = [state.spec.name for state in feasible]
        objective = _cost_tiebroken_objective(
            [tbar[name] for name in names],
            [c_eff[name] for name in names],
        )
        success, vector = _solve_simplex_lp(
            objective=objective,
            upper_constraint=[c_eff[name] for name in names],
            upper_bound=budget,
        )

        lp_order: list[str] = []
        if success and vector is not None:
            weights = _normalize_weights(names, vector)
            lp_order = [
                name
                for name, weight in sorted(
                    weights.items(),
                    key=lambda item: (-item[1], tbar[item[0]], c_eff[item[0]], item[0]),
                )
                if weight > LP_EPS
            ]

        ordered = list(lp_order)
        seen = set(ordered)
        remainder = [name for name in names if name not in seen]
        remainder.sort(
            key=lambda name: (
                c_eff[name] > budget + LP_EPS,
                tbar[name],
                c_eff[name],
                name,
            )
        )
        ordered.extend(remainder)
        return ordered


class BudgetRangeHedgePolicy(BudgetRangePolicy):
    """``LP-RangeBudget`` + ``Hedge-ProbTarget``."""

    use_hedge = True
    name_suffix = "_hedge"


def _fallback_in_budget(
    feasible: list[ProviderState],
    c_eff: dict[str, float],
    budget: float,
    reference_cost: float,
    *,
    fallback_label: str,
) -> RoutingDecision:
    """Shared fallback when the LP fails or returns no positive weights."""
    affordable = [s for s in feasible if c_eff[s.spec.name] <= budget + LP_EPS]
    if affordable:
        choice = min(affordable, key=lambda s: c_eff[s.spec.name])
        return RoutingDecision(
            primary=choice.spec.name,
            lp_weights={choice.spec.name: 1.0},
            lp_status=f"fallback_in_budget_{fallback_label}",
            budget_usd=float(budget),
            reference_cost_usd=float(reference_cost),
            c_eff_map=c_eff,
            notes=f"fallback_affordable_{fallback_label}",
        )
    choice = min(feasible, key=lambda s: c_eff[s.spec.name])
    return RoutingDecision(
        primary=choice.spec.name,
        lp_weights={choice.spec.name: 1.0},
        lp_status=f"fallback_no_budget_{fallback_label}",
        budget_usd=float(budget),
        reference_cost_usd=float(reference_cost),
        c_eff_map=c_eff,
        notes=f"fallback_no_affordable_{fallback_label}",
    )


# ---------------------------------------------------------------------------
# Policy registry — used by the runner CLI.
# ---------------------------------------------------------------------------


def build_policy(
    name: str,
    specs: list[ProviderSpec],
    slo_ms: float,
    profile_window_sec: float = PROFILE_WINDOW_SEC,
    prefix_cache_routing: bool = False,
) -> BasePolicy:
    """Construct one policy by name.

    Recognized names:
      * Baselines: ``or_auto``, ``or_sort_latency``, ``or_sort_cost``,
        ``greedy_cost``, ``greedy_latency``, ``random``, ``quota_first``,
        ``concurrency_first``
      * Paper line: ``budget_range_p<PP>`` and
        ``budget_range_p<PP>_hedge`` (PP in ``[0, 100]``)
    """
    common = {
        "specs": specs,
        "slo_ms": slo_ms,
        "profile_window_sec": profile_window_sec,
        "prefix_cache_routing": prefix_cache_routing,
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

    if name.startswith("budget_range_p") and name.endswith("_hedge"):
        try:
            p = int(name[len("budget_range_p") : -len("_hedge")])
        except ValueError as exc:
            raise ValueError(f"Bad budget_range name: {name!r}") from exc
        return BudgetRangeHedgePolicy(**common, budget_percentile=p)
    if name.startswith("budget_range_p"):
        try:
            p = int(name[len("budget_range_p") :])
        except ValueError as exc:
            raise ValueError(f"Bad budget_range name: {name!r}") from exc
        return BudgetRangePolicy(**common, budget_percentile=p)

    raise ValueError(f"Unknown policy name: {name!r}")


__all__ = [
    "HEDGE_DISPATCH_OVERHEAD_SEC",
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
    "compute_hedge_time_sec",
    "request_cost_for_spec",
    "select_safe_cheapest_backup",
]
