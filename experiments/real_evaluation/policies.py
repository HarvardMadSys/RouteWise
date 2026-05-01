"""Adapter routing policies for real online evaluation.

ARCHITECTURE NOTE — read before extending.

This module is a **real-eval adapter**, NOT a long-term routing system.
It exists to let real-online experiments call the same algorithm shapes
as the simulator without forcing a premature unification of three
incompatible policy frameworks (``rwsim.policies``, the
``lp_budget_eval`` sidecar, and these phase6-derived classes). When the
canonical policy pipeline in ``rwsim.policies.composer`` is mature
enough to express both the legacy and current paper methods, this
module should be retired or re-grounded on it.

Policy taxonomy:

* **Baselines** (used in paper plots as comparison points):
  - ``OpenRouterAutoPolicy``  : OpenRouter's native auto-routing
  - ``SortLatencyPolicy``     : OpenRouter ``sort=latency``
  - ``CheapestFixedPolicy``   : always cheapest available
  - ``FastestFixedPolicy``    : always lowest empirical-P50 available
  - ``QuotaFirstPolicy`` / ``ConcurrencyFirstPolicy`` : tier-priority heuristics

* **Legacy / reference** (historical RouteWise iterations, kept for A/B):
  - ``OriginalLPHedgePolicy``        : ``LP-CDF`` (min cost s.t. SLO attainment)
  - ``BudgetedVhatHedgePolicy``      : ``LP-TTFT-budget`` with ``vhat`` anchor

* **Current paper line** (``LP-TTFT-budget`` + ``Hedge-ProbTarget``):
  - ``BudgetRangeHedgePolicy(p)`` : range-normalized cost budget ``B_p =
    c_min + p (c_max - c_min)``, probability-target hedge

The ``BudgetRange*`` selector is a hand-port of
``experiments.tiered_capacity.lp_budget_eval._select_budget_body``
(``_is_range_budget_variant`` branch). The simulator version is
distribution-aware (uses ``provider._active_ttft_dist().mean()``); this
real version uses the empirical rolling profile. A parity test should
pin the two to the same outputs given matched inputs.
"""

from __future__ import annotations

import logging
import math
import random
import threading
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from experiments.real_evaluation.inventory import (
    PROFILE_WINDOW_SEC,
    ProviderSpec,
    ProviderState,
)
from experiments.real_evaluation.shadow_price import (
    calibrate_envelopes,
    effective_cost,
    request_marginal_cost,
)

LP_EPS: float = 1e-9
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
OR_AUTO_SENTINEL: str = "__openrouter_auto__"
OR_SORT_LATENCY_SENTINEL: str = "__openrouter_sort_latency__"
OR_SORT_PRICE_SENTINEL: str = "__openrouter_sort_price__"
OR_SORT_THROUGHPUT_SENTINEL: str = "__openrouter_sort_throughput__"

# Map sentinel name -> OpenRouter ``sort`` mode. The runner uses this to
# build a transport config for sentinel dispatches without relying on
# string-substring tricks.
OR_SORT_SENTINEL_TO_MODE: dict[str, str] = {
    OR_SORT_LATENCY_SENTINEL: "latency",
    OR_SORT_PRICE_SENTINEL: "price",
    OR_SORT_THROUGHPUT_SENTINEL: "throughput",
}


@dataclass
class RequestContext:
    """Per-request inputs the policy needs to make a routing decision."""

    prompt_tokens: int
    completion_tokens_budget: int


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


def slo_safe_anchor_cost(
    states: dict[str, ProviderState],
    ctx: RequestContext,
    slo_sec: float,
    now: float,
    *,
    cdf_target: float = 0.9,
) -> float:
    """Empirical ``v_hat_i``: cheapest SLO-safe S_A request cost.

    A provider is "SLO-safe" when its rolling-profile CDF at the SLO
    threshold is >= ``cdf_target``. Falls back to cheapest S_A overall
    when no profile data is available, mirroring the simulator's
    ``_predicted_api_cost_anchor`` fallback chain.
    """
    api_states = [s for s in states.values() if s.spec.tier == "api"]
    if not api_states:
        return 0.0

    slo_threshold_ms = slo_sec * 1000.0
    safe: list[tuple[float, ProviderState]] = []
    for st in api_states:
        if st.profile.cdf_at(slo_threshold_ms, now) >= cdf_target:
            cost = request_cost_for_spec(st.spec, ctx)
            safe.append((cost, st))
    if safe:
        safe.sort(key=lambda pair: pair[0])
        return float(safe[0][0])

    costs = [(request_cost_for_spec(st.spec, ctx), st) for st in api_states]
    costs.sort(key=lambda pair: pair[0])
    return float(costs[0][0])


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


def _normalize_weights(names: list[str], vector: np.ndarray) -> dict[str, float]:
    raw = {names[i]: float(vector[i]) for i in range(len(names)) if vector[i] > 1e-6}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Hedging — probability-target backup selection + latest-safe dispatch time.
# ---------------------------------------------------------------------------


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

    Empirical version of ``lp_budget_eval._find_latest_safe_hedge_time_ms``.
    """
    slo_ms = slo_sec * 1000.0
    latest_safe: float | None = None
    grid = np.arange(0.0, max(slo_sec, grid_step_sec) + 1e-9, grid_step_sec)
    for elapsed_sec in grid:
        elapsed_ms = float(elapsed_sec) * 1000.0
        F_t = primary_state.profile.cdf_at(elapsed_ms, now)
        F_L = primary_state.profile.cdf_at(slo_ms, now)
        survived_t = max(1.0 - F_t, 1e-9)
        p_not_violate = max(0.0, min(1.0, (F_L - F_t) / survived_t))
        p_violate = max(0.0, min(1.0, (1.0 - F_L) / survived_t))
        remaining_sec = slo_sec - float(elapsed_sec) - dispatch_overhead_sec
        p_backup = backup_state.profile.cdf_at(remaining_sec * 1000.0, now)
        combined = p_not_violate + p_violate * p_backup
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
) -> str | None:
    """Pick the cheapest non-primary provider whose CDF passes the SLO threshold.

    Falls back to the lowest empirical-median provider when no candidate is
    SLO-safe. Returns ``None`` only when there's no other provider at all.
    """
    remaining_ms = (slo_sec - dispatch_overhead_sec) * 1000.0
    safe: list[tuple[float, ProviderState]] = []
    for name, state in states.items():
        if name == primary or not state.is_available(now):
            continue
        if state.profile.cdf_at(remaining_ms, now) >= success_target:
            safe.append((request_cost_for_spec(state.spec, ctx), state))
    if safe:
        safe.sort(key=lambda pair: (pair[0], pair[1].spec.name))
        return safe[0][1].spec.name

    others: list[tuple[float, ProviderState]] = []
    for name, state in states.items():
        if name == primary or not state.is_available(now):
            continue
        med = state.profile.median_ms(now)
        if med is None:
            med = float("inf")
        others.append((med, state))
    if not others:
        return None
    others.sort(key=lambda pair: (pair[0], pair[1].spec.name))
    return others[0][1].spec.name


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

    def __init__(
        self,
        specs: list[ProviderSpec],
        slo_ms: float,
        profile_window_sec: float = PROFILE_WINDOW_SEC,
    ) -> None:
        self.specs = specs
        self.slo_sec = slo_ms / 1000.0
        self.states: dict[str, ProviderState] = {
            spec.name: ProviderState.from_spec(spec, profile_window_sec)
            for spec in specs
        }
        self._lock = threading.Lock()

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
    ) -> None:
        """Record one dispatched request against quota / concurrency state."""
        with self._lock:
            state = self.states.get(provider)
            if state is None:
                return
            if state.quota is not None:
                state.quota.charge(now)
            if state.concurrency is not None:
                state.concurrency.admit(
                    request_id=int(now * 1000) & 0xFFFFFFFF,
                    now=now,
                    expected_hold_sec=expected_service_sec,
                )

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Baselines.
# ---------------------------------------------------------------------------


class OpenRouterAutoPolicy(BasePolicy):
    """OpenRouter's default routing — one sentinel decision."""

    name = "openrouter_auto"

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        has_or = any(s.transport_cfg.transport == "openrouter" for s in self.specs)
        if not has_or:
            return RoutingDecision(primary=None, notes="no_openrouter_spec")
        return RoutingDecision(primary=OR_AUTO_SENTINEL, notes="or_default")


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
        return RoutingDecision(
            primary=self.sentinel, notes=f"or_sort_{self.sort_mode}"
        )


class SortLatencyPolicy(_OpenRouterSortPolicy):
    """OpenRouter ``sort=latency``."""

    name = "sort_latency"
    sort_mode = "latency"
    sentinel = OR_SORT_LATENCY_SENTINEL


class SortPricePolicy(_OpenRouterSortPolicy):
    """OpenRouter ``sort=price``."""

    name = "sort_price"
    sort_mode = "price"
    sentinel = OR_SORT_PRICE_SENTINEL


class SortThroughputPolicy(_OpenRouterSortPolicy):
    """OpenRouter ``sort=throughput``."""

    name = "sort_throughput"
    sort_mode = "throughput"
    sentinel = OR_SORT_THROUGHPUT_SENTINEL


class _FixedBaseBase(BasePolicy):
    """Shared logic for cheapest/fastest fixed-pick baselines.

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
            states = [
                s for s in states
                if s.spec.transport_cfg.transport == "openrouter"
            ]
        return [s for s in states if s.is_available(now)]


class CheapestFixedPolicy(_FixedBaseBase):
    """Joint-pool: cheapest currently-available provider across ALL tiers."""

    name = "cheapest_fixed"
    or_only = False

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        candidates = self._candidates(now)
        if not candidates:
            return RoutingDecision(primary=None, notes="none_available")
        scored = [(s, request_cost_for_spec(s.spec, ctx)) for s in candidates]
        scored.sort(key=lambda pair: (pair[1], pair[0].spec.name))
        return RoutingDecision(primary=scored[0][0].spec.name, notes="cheapest")


class OpenRouterCheapestFixedPolicy(CheapestFixedPolicy):
    """OR-only cheapest fixed (filters to ``transport == 'openrouter'``)."""

    name = "or_cheapest_fixed"
    or_only = True


class FastestFixedPolicy(_FixedBaseBase):
    """Joint-pool: lowest-empirical-median provider across ALL tiers."""

    name = "fastest_fixed"
    or_only = False

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        candidates = self._candidates(now)
        if not candidates:
            return RoutingDecision(primary=None, notes="none_available")
        scored = [
            (s.profile.median_ms(now) or float("inf"), s) for s in candidates
        ]
        scored.sort(key=lambda pair: (pair[0], pair[1].spec.name))
        return RoutingDecision(primary=scored[0][1].spec.name, notes="fastest")


class OpenRouterFastestFixedPolicy(FastestFixedPolicy):
    """OR-only fastest fixed (filters to ``transport == 'openrouter'``)."""

    name = "or_fastest_fixed"
    or_only = True


class TierFirstPolicy(BasePolicy):
    """Fill preferred tier first, spill on exhaustion."""

    tier_priority: tuple[str, ...] = ("quota", "concurrency", "api")

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
# Legacy LP-CDF — kept for A/B vs current paper line.
# ---------------------------------------------------------------------------


class OriginalLPHedgePolicy(BasePolicy):
    """``LP-CDF``: ``min sum_j pi_j c_eff_j  s.t.  sum_j pi_j F_j(SLO) >= rho``."""

    name = "original_lp_hedge"
    use_hedge = True

    def __init__(
        self,
        specs: list[ProviderSpec],
        slo_ms: float,
        profile_window_sec: float = PROFILE_WINDOW_SEC,
        slo_target: float = HEDGE_SUCCESS_TARGET,
    ) -> None:
        super().__init__(specs, slo_ms, profile_window_sec)
        self._slo_target = slo_target

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        feasible = [s for s in self.states.values() if s.is_available(now)]
        if not feasible:
            return RoutingDecision(primary=None, notes="none_available")

        L, U = calibrate_envelopes(
            self.specs, ctx.prompt_tokens, ctx.completion_tokens_budget
        )
        request_costs = {
            s.spec.name: request_cost_for_spec(s.spec, ctx) for s in feasible
        }
        c_eff = {
            s.spec.name: effective_cost(
                s, request_costs[s.spec.name], now, U=U, L=L
            )
            for s in feasible
        }
        slo_threshold_ms = self.slo_sec * 1000.0
        f_at_slo = {
            s.spec.name: s.profile.cdf_at(slo_threshold_ms, now) for s in feasible
        }

        for relax in (self._slo_target, 0.95, 0.90, 0.0):
            weights = self._solve_relaxed_lp(feasible, c_eff, f_at_slo, relax)
            if weights:
                primary = _sample_weighted(
                    weights, rng=random.Random(int(now * 1e6))
                )
                return RoutingDecision(
                    primary=primary,
                    lp_weights=weights,
                    lp_status=f"optimal_f{relax:.2f}",
                    reference_cost_usd=U,
                    c_eff_map=c_eff,
                    tier_mix=_tier_mix_from_weights(weights, self.states),
                    notes=f"original_lp_f{relax:.2f}",
                )

        best = min(feasible, key=lambda s: c_eff[s.spec.name])
        return RoutingDecision(
            primary=best.spec.name,
            lp_weights={best.spec.name: 1.0},
            lp_status="best_effort",
            c_eff_map=c_eff,
            notes="original_lp_best_effort",
        )

    @staticmethod
    def _solve_relaxed_lp(
        feasible: list[ProviderState],
        c_eff: dict[str, float],
        f_at_slo: dict[str, float],
        slo_target: float,
    ) -> dict[str, float] | None:
        if not feasible:
            return None
        names = [s.spec.name for s in feasible]
        n = len(names)
        objective = np.array([c_eff[name] for name in names], dtype=float)
        # SLO attainment >= rho rewritten as -F·pi <= -rho.
        A_ub = np.array([[-f_at_slo[name] for name in names]], dtype=float)
        b_ub = np.array([-slo_target], dtype=float)
        A_eq = np.ones((1, n), dtype=float)
        b_eq = np.array([1.0], dtype=float)
        bounds = [(0.0, 1.0) for _ in range(n)]
        try:
            result = linprog(
                c=objective,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
        except Exception:
            return None
        if not result.success:
            return None
        return _normalize_weights(names, np.asarray(result.x, dtype=float))


# ---------------------------------------------------------------------------
# Legacy LP-TTFT-budget with vhat anchor — kept for A/B vs range-budget.
# ---------------------------------------------------------------------------


class BudgetedVhatHedgePolicy(BasePolicy):
    """``LP-TTFT-budget`` variant anchored on ``tau * v_hat``.

    The ``v_hat`` anchor is the cheapest SLO-safe S_A request cost. This was
    the formulation in the original Phase 6 SA-only experiment. Kept here
    for A/B comparison against the range-budget variant.
    """

    name = "budget_vhat_t100_hedge"
    use_hedge = True

    def __init__(
        self,
        specs: list[ProviderSpec],
        slo_ms: float,
        profile_window_sec: float = PROFILE_WINDOW_SEC,
        tau: float = 1.0,
    ) -> None:
        super().__init__(specs, slo_ms, profile_window_sec)
        self.tau = tau
        self.name = f"budget_vhat_t{int(tau * 100)}_hedge"

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        feasible = [s for s in self.states.values() if s.is_available(now)]
        if not feasible:
            return RoutingDecision(primary=None, notes="none_available")

        L, U = calibrate_envelopes(
            self.specs, ctx.prompt_tokens, ctx.completion_tokens_budget
        )
        request_costs = {
            s.spec.name: request_cost_for_spec(s.spec, ctx) for s in feasible
        }
        c_eff = {
            s.spec.name: effective_cost(
                s, request_costs[s.spec.name], now, U=U, L=L
            )
            for s in feasible
        }
        tbar: dict[str, float] = {}
        for s in feasible:
            mean_ms, _ = _body_latency_proxy_ms(s, now)
            if not math.isfinite(mean_ms):
                mean_ms = UNPROFILED_LATENCY_PENALTY_MS
            tbar[s.spec.name] = mean_ms

        v_hat = slo_safe_anchor_cost(self.states, ctx, self.slo_sec, now)
        budget = self.tau * v_hat

        names = [s.spec.name for s in feasible]
        success, vector = _solve_simplex_lp(
            objective=[tbar[name] for name in names],
            upper_constraint=[c_eff[name] for name in names],
            upper_bound=budget,
        )
        if success and vector is not None:
            weights = _normalize_weights(names, vector)
            if weights:
                primary = _sample_weighted(
                    weights, rng=random.Random(int(now * 1e6))
                )
                return RoutingDecision(
                    primary=primary,
                    lp_weights=weights,
                    lp_status="optimal",
                    budget_usd=float(budget),
                    reference_cost_usd=float(v_hat),
                    c_eff_map=c_eff,
                    tier_mix=_tier_mix_from_weights(weights, self.states),
                    notes=self.name,
                )

        return _fallback_in_budget(
            feasible, c_eff, budget, v_hat, fallback_label="vhat"
        )


# ---------------------------------------------------------------------------
# Current paper line: LP-RangeBudget + Hedge-ProbTarget.
# ---------------------------------------------------------------------------


class BudgetRangePolicy(BasePolicy):
    """``LP-RangeBudget`` body router (current paper main method).

    Body selector: ``min sum pi_j T̄_j  s.t.  sum pi_j c_eff_j <= B_p``
    where ``B_p = c_min + (p/100) * (c_max - c_min)``.

    Hand-ported from
    ``experiments.tiered_capacity.lp_budget_eval._select_budget_body``
    (the ``_is_range_budget_variant`` branch). The simulator version reads
    distributional means via ``provider._active_ttft_dist().mean()``;
    here we use the empirical ``LatencyProfile.mean_ms``. A parity test
    should pin both to identical outputs given matched inputs.
    """

    use_hedge = False
    name_suffix = ""

    def __init__(
        self,
        specs: list[ProviderSpec],
        slo_ms: float,
        profile_window_sec: float = PROFILE_WINDOW_SEC,
        budget_percentile: int = 100,
    ) -> None:
        super().__init__(specs, slo_ms, profile_window_sec)
        if not 0 <= budget_percentile <= 100:
            raise ValueError(
                f"budget_percentile must be in [0, 100]; got {budget_percentile}"
            )
        self.budget_percentile = int(budget_percentile)
        self.name = f"budget_range_p{self.budget_percentile}{self.name_suffix}"

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        feasible = [s for s in self.states.values() if s.is_available(now)]
        if not feasible:
            return RoutingDecision(primary=None, notes="none_available")

        L, U = calibrate_envelopes(
            self.specs, ctx.prompt_tokens, ctx.completion_tokens_budget
        )
        request_costs = {
            s.spec.name: request_cost_for_spec(s.spec, ctx) for s in feasible
        }
        c_eff = {
            s.spec.name: effective_cost(
                s, request_costs[s.spec.name], now, U=U, L=L
            )
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
        success, vector = _solve_simplex_lp(
            objective=[tbar[name] for name in names],
            upper_constraint=[c_eff[name] for name in names],
            upper_bound=budget,
        )
        if success and vector is not None:
            weights = _normalize_weights(names, vector)
            if weights:
                primary = _sample_weighted(
                    weights, rng=random.Random(int(now * 1e6))
                )
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

        return _fallback_in_budget(
            feasible, c_eff, budget, c_max, fallback_label="range"
        )


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
    **kwargs,
) -> BasePolicy:
    """Construct one policy by name.

    Recognized names:
      * Baselines: ``openrouter_auto``, ``sort_latency``, ``cheapest_fixed``,
        ``fastest_fixed``, ``quota_first``, ``concurrency_first``
      * Legacy LP-CDF: ``original_lp_hedge``
      * Legacy vhat budget: ``budget_vhat_t<NN>_hedge`` (NN ∈ {25, 50, 75, 100})
      * Current paper line: ``budget_range_p<PP>`` and
        ``budget_range_p<PP>_hedge`` (PP ∈ [0, 100])
    """
    common = dict(specs=specs, slo_ms=slo_ms, profile_window_sec=profile_window_sec)
    if name == "openrouter_auto":
        return OpenRouterAutoPolicy(**common)
    if name == "sort_latency":
        return SortLatencyPolicy(**common)
    if name == "sort_price":
        return SortPricePolicy(**common)
    if name == "sort_throughput":
        return SortThroughputPolicy(**common)
    if name == "cheapest_fixed":
        return CheapestFixedPolicy(**common)
    if name == "fastest_fixed":
        return FastestFixedPolicy(**common)
    if name == "or_cheapest_fixed":
        return OpenRouterCheapestFixedPolicy(**common)
    if name == "or_fastest_fixed":
        return OpenRouterFastestFixedPolicy(**common)
    if name == "quota_first":
        return QuotaFirstPolicy(**common)
    if name == "concurrency_first":
        return ConcurrencyFirstPolicy(**common)
    if name == "original_lp_hedge":
        return OriginalLPHedgePolicy(**common, **kwargs)

    if name.startswith("budget_vhat_t") and name.endswith("_hedge"):
        try:
            tau_pct = int(name[len("budget_vhat_t") : -len("_hedge")])
        except ValueError as exc:
            raise ValueError(f"Bad budget_vhat name: {name!r}") from exc
        return BudgetedVhatHedgePolicy(**common, tau=tau_pct / 100.0)

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
    "BasePolicy",
    "BudgetRangeHedgePolicy",
    "BudgetRangePolicy",
    "BudgetedVhatHedgePolicy",
    "CheapestFixedPolicy",
    "ConcurrencyFirstPolicy",
    "FastestFixedPolicy",
    "HEDGE_DISPATCH_OVERHEAD_SEC",
    "HEDGE_SUCCESS_TARGET",
    "OR_AUTO_SENTINEL",
    "OR_SORT_LATENCY_SENTINEL",
    "OR_SORT_PRICE_SENTINEL",
    "OR_SORT_SENTINEL_TO_MODE",
    "OR_SORT_THROUGHPUT_SENTINEL",
    "OpenRouterAutoPolicy",
    "OpenRouterCheapestFixedPolicy",
    "OpenRouterFastestFixedPolicy",
    "OriginalLPHedgePolicy",
    "QuotaFirstPolicy",
    "RequestContext",
    "RoutingDecision",
    "SortLatencyPolicy",
    "SortPricePolicy",
    "SortThroughputPolicy",
    "TierFirstPolicy",
    "UNPROFILED_LATENCY_PENALTY_MS",
    "build_policy",
    "compute_hedge_time_sec",
    "request_cost_for_spec",
    "select_safe_cheapest_backup",
    "slo_safe_anchor_cost",
]
