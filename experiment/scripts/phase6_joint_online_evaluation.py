#!/usr/bin/env python3
# ruff: noqa: I001
"""Phase 6 joint online evaluation for NSDI submission.

Single-file harness supporting all three tiers (S_Q + S_C + S_A) via multiple
transports, with the new budgeted LP + probability-based hedging.

Provider universe is declared in an inventory JSON; default is
experiment/data/joint_minimax_m25_online.json.

Policy set (all 8 run as separate interleaved streams on a shared trace):
  - openrouter_auto     : OpenRouter default                (baseline)
  - sort_latency        : OpenRouter sort=latency           (baseline)
  - cheapest_fixed      : fixed cheapest provider           (anchor)
  - fastest_fixed       : fixed fastest provider            (anchor)
  - quota_first         : S_Q first, spill S_C then S_A     (heuristic)
  - concurrency_first   : S_C first, spill S_Q then S_A     (heuristic)
  - original_lp_hedge   : old LP (SLO-attainment) + hedge   (prior method)
  - budget_vhat_t100_hedge : new budgeted LP (tau=1.0) + hedge (ours)

Each incoming trace request is dispatched to exactly one policy (round-robin
hash). Each policy maintains its own rolling profile and quota/concurrency
state, so their routing decisions do not interfere with each other. The
underlying subscription quota at the backend is still physically shared,
but at 1 req/s trace rate over 1h the pilot load stays well under the
7100 req/hr S_Q budget (see phase6_joint_online_evaluation_plan.md).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from scipy.optimize import linprog

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.strategies.joint_online_transport import (  # noqa: E402
    BaseTransport,
    SingleRequestResult,
    TransportConfig,
    build_transport,
    compute_request_cost_usd,
    resolve_transport_config,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROBE_PROMPT = "Write a one-sentence greeting."
DEFAULT_MAX_TOKENS = 128
DEFAULT_TIMEOUT_SEC = 60
PROFILE_WINDOW_SEC = 15 * 60.0
PROBE_RATE = 0.05
HEDGE_SUCCESS_TARGET = 0.99
HEDGE_DISPATCH_OVERHEAD_SEC = 0.05
SHADOW_PRICE_FLOOR_RATIO = 1e-3

ALL_POLICIES = [
    "openrouter_auto",
    "sort_latency",
    "cheapest_fixed",
    "fastest_fixed",
    "quota_first",
    "concurrency_first",
    "original_lp_hedge",
    "budget_vhat_t100_hedge",
]


# ---------------------------------------------------------------------------
# Inventory loader
# ---------------------------------------------------------------------------


@dataclass
class ProviderSpec:
    name: str
    tier: str  # "quota", "concurrency", "api"
    transport_cfg: TransportConfig
    quota_window_sec: float | None
    quota_requests: int | None
    concurrency_limit: int | None

    @property
    def input_price_per_m(self) -> float:
        return self.transport_cfg.input_price_per_m

    @property
    def output_price_per_m(self) -> float:
        return self.transport_cfg.output_price_per_m


@dataclass
class InventoryConfig:
    model_family: str
    openrouter_model_id: str
    primary_slo_ms: float
    slo_thresholds_ms: list[float]
    providers: list[ProviderSpec]


def load_inventory(path: Path) -> InventoryConfig:
    raw = json.loads(path.read_text())
    provider_specs: list[ProviderSpec] = []
    for entry in raw["providers"]:
        transport_cfg = resolve_transport_config(entry)
        provider_specs.append(
            ProviderSpec(
                name=entry["name"],
                tier=entry["tier"],
                transport_cfg=transport_cfg,
                quota_window_sec=(
                    float(entry["quota_window_sec"])
                    if entry.get("quota_window_sec") is not None
                    else None
                ),
                quota_requests=(
                    int(entry["quota_requests"])
                    if entry.get("quota_requests") is not None
                    else None
                ),
                concurrency_limit=(
                    int(entry["concurrency_limit"])
                    if entry.get("concurrency_limit") is not None
                    else None
                ),
            )
        )
    return InventoryConfig(
        model_family=raw["model_family"],
        openrouter_model_id=raw["openrouter_model_id"],
        primary_slo_ms=float(raw["primary_slo_ms"]),
        slo_thresholds_ms=[float(t) for t in raw["slo_thresholds_ms"]],
        providers=provider_specs,
    )


# ---------------------------------------------------------------------------
# Per-policy state: quota, concurrency, profile
# ---------------------------------------------------------------------------


@dataclass
class QuotaState:
    window_sec: float
    limit: int
    window_start: float = 0.0
    used: int = 0

    def tick(self, now: float) -> None:
        if self.window_start <= 0.0:
            self.window_start = now
            return
        if now - self.window_start >= self.window_sec:
            self.window_start = now
            self.used = 0

    def is_available(self, now: float) -> bool:
        self.tick(now)
        return self.used < self.limit

    def fraction_used(self, now: float) -> float:
        self.tick(now)
        if self.limit <= 0:
            return 0.0
        return float(min(self.used / self.limit, 0.9999))

    def charge(self, now: float) -> None:
        self.tick(now)
        self.used += 1


@dataclass
class ConcurrencyState:
    limit: int
    active: list[tuple[int, float]] = field(default_factory=list)

    def _prune(self, now: float) -> None:
        self.active = [(rid, deadline) for rid, deadline in self.active if deadline > now]

    def is_available(self, now: float) -> bool:
        self._prune(now)
        return len(self.active) < self.limit

    def utilization(self, now: float) -> float:
        self._prune(now)
        if self.limit <= 0:
            return 0.0
        return float(min(len(self.active) / self.limit, 0.9999))

    def admit(self, request_id: int, now: float, expected_hold_sec: float) -> None:
        self._prune(now)
        self.active.append((request_id, now + max(0.1, expected_hold_sec)))


@dataclass
class LatencyProfile:
    """Rolling window of (timestamp, ttft_ms) with optional error tracking."""

    window_sec: float
    samples: list[tuple[float, float]] = field(default_factory=list)
    error_samples: list[tuple[float, str]] = field(default_factory=list)

    def add_sample(self, ts: float, ttft_ms: float, error_type: str | None = None) -> None:
        if error_type is None and ttft_ms > 0:
            self.samples.append((ts, float(ttft_ms)))
        elif error_type is not None:
            self.error_samples.append((ts, error_type))

    def _active_samples(self, now: float) -> list[float]:
        cutoff = now - self.window_sec
        self.samples = [(ts, v) for ts, v in self.samples if ts >= cutoff]
        self.error_samples = [(ts, e) for ts, e in self.error_samples if ts >= cutoff]
        return [v for _, v in self.samples]

    def mean_ms(self, now: float) -> float | None:
        values = self._active_samples(now)
        if not values:
            return None
        return float(np.mean(values))

    def median_ms(self, now: float) -> float | None:
        values = self._active_samples(now)
        if not values:
            return None
        return float(np.percentile(values, 50))

    def cdf_at(self, threshold_ms: float, now: float) -> float:
        """Return P(TTFT <= threshold) treating errors as misses."""
        values = self._active_samples(now)
        n_errors = len(self.error_samples)
        n_total = len(values) + n_errors
        if n_total == 0:
            return 0.0
        n_under = int(sum(1 for v in values if v <= threshold_ms))
        return float(n_under / n_total)


@dataclass
class ProviderState:
    """Per-policy dynamic state for one provider."""

    spec: ProviderSpec
    profile: LatencyProfile
    quota: QuotaState | None
    concurrency: ConcurrencyState | None

    @classmethod
    def from_spec(cls, spec: ProviderSpec, window_sec: float) -> "ProviderState":
        quota = (
            QuotaState(
                window_sec=spec.quota_window_sec,
                limit=spec.quota_requests,
            )
            if spec.quota_requests is not None and spec.quota_window_sec is not None
            else None
        )
        concurrency = (
            ConcurrencyState(limit=spec.concurrency_limit)
            if spec.concurrency_limit is not None
            else None
        )
        return cls(
            spec=spec,
            profile=LatencyProfile(window_sec=window_sec),
            quota=quota,
            concurrency=concurrency,
        )

    def is_available(self, now: float) -> bool:
        if self.quota is not None and not self.quota.is_available(now):
            return False
        if self.concurrency is not None and not self.concurrency.is_available(now):
            return False
        return True


# ---------------------------------------------------------------------------
# Shadow price / effective cost
# ---------------------------------------------------------------------------


def quota_shadow_price(
    state: ProviderState,
    now: float,
    U: float,
    L: float,
) -> float:
    if state.quota is None:
        return 0.0
    z = state.quota.fraction_used(now)
    if L <= 0 or U <= L:
        return 0.0
    return L * math.pow(U / L, z)


def concurrency_shadow_price(
    state: ProviderState,
    now: float,
    U: float,
) -> float:
    if state.concurrency is None:
        return 0.0
    u = state.concurrency.utilization(now)
    return U * u


def effective_cost(
    state: ProviderState,
    request_cost: float,
    now: float,
    U: float,
    L: float,
) -> float:
    q_sp = quota_shadow_price(state, now, U, L)
    c_sp = concurrency_shadow_price(state, now, U)
    if state.spec.tier == "api":
        return request_cost + q_sp + c_sp
    return q_sp + c_sp


def calibrate_envelopes(
    specs: list[ProviderSpec],
    ctx_prompt_tokens: int,
    ctx_completion_tokens: int,
) -> tuple[float, float]:
    api_costs = []
    for spec in specs:
        if spec.tier == "api" and (spec.input_price_per_m > 0 or spec.output_price_per_m > 0):
            api_costs.append(
                compute_request_cost_usd(
                    ctx_prompt_tokens,
                    ctx_completion_tokens,
                    spec.input_price_per_m,
                    spec.output_price_per_m,
                )
            )
    if not api_costs:
        return (1e-6, 1e-3)
    U = float(max(api_costs))
    L = float(max(U * SHADOW_PRICE_FLOOR_RATIO, 1e-9))
    return (L, U)


# ---------------------------------------------------------------------------
# Request context + SLO-safe anchor
# ---------------------------------------------------------------------------


@dataclass
class RequestContext:
    prompt_tokens: int
    completion_tokens_budget: int


def request_cost_for_spec(spec: ProviderSpec, ctx: RequestContext) -> float:
    return compute_request_cost_usd(
        ctx.prompt_tokens,
        ctx.completion_tokens_budget,
        spec.input_price_per_m,
        spec.output_price_per_m,
    )


def slo_safe_anchor_cost(
    states: dict[str, ProviderState],
    ctx: RequestContext,
    slo_sec: float,
    now: float,
) -> float:
    """Return v_hat_i = SLO-safe cheapest S_A provider's request cost.

    "SLO-safe" means the provider's rolling profile CDF at slo_sec has a
    sufficient mass (>=0.9). Falls back to cheapest S_A overall if no
    profile data exists or nothing is marked SLO-safe.
    """
    api_states = [s for s in states.values() if s.spec.tier == "api"]
    if not api_states:
        return 0.0

    slo_threshold_ms = slo_sec * 1000.0
    safe: list[tuple[float, ProviderState]] = []
    for st in api_states:
        cdf = st.profile.cdf_at(slo_threshold_ms, now)
        if cdf >= 0.9:
            cost = request_cost_for_spec(st.spec, ctx)
            safe.append((cost, st))

    if safe:
        safe.sort(key=lambda pair: pair[0])
        return float(safe[0][0])

    costs = [(request_cost_for_spec(st.spec, ctx), st) for st in api_states]
    costs.sort(key=lambda pair: pair[0])
    return float(costs[0][0])


# ---------------------------------------------------------------------------
# Policy wrappers
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    primary: str | None
    hedge: str | None = None
    hedge_time_sec: float = float("inf")
    lp_weights: dict[str, float] | None = None
    lp_status: str = "n/a"
    budget_usd: float | None = None
    reference_cost_usd: float | None = None
    c_eff_map: dict[str, float] | None = None
    tier_mix: dict[str, float] | None = None
    notes: str = ""


class BasePolicy:
    """Base class for one policy's routing state + decision logic.

    Each policy owns an isolated copy of provider state so that its
    shadow-price and LP decisions are not influenced by other policies
    in the shared pilot. The underlying backend quotas are physically
    shared, but at pilot trace rates the aggregate load stays under
    the subscription budget (see plan doc).
    """

    name: str = "base"

    def __init__(
        self,
        specs: list[ProviderSpec],
        slo_ms: float,
        profile_window_sec: float,
    ) -> None:
        self.slo_sec = slo_ms / 1000.0
        self.specs = specs
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
            if provider in self.states:
                self.states[provider].profile.add_sample(ts, ttft_ms, error_type)

    def charge_capacity(
        self,
        provider: str,
        now: float,
        expected_service_sec: float,
    ) -> None:
        """Record a dispatched request against quota / concurrency state."""
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


class OpenRouterAutoPolicy(BasePolicy):
    name = "openrouter_auto"

    def __init__(self, specs: list[ProviderSpec], slo_ms: float, profile_window_sec: float) -> None:
        super().__init__(specs, slo_ms, profile_window_sec)
        self.has_openrouter = any(
            s.transport_cfg.transport == "openrouter" for s in specs
        )

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        if not self.has_openrouter:
            return RoutingDecision(primary=None, notes="no_openrouter_spec")
        # Use a sentinel. The executor will route this via OpenRouter with no
        # provider hint, which is the default auto-routing behaviour.
        return RoutingDecision(primary="__openrouter_auto__", notes="or_default")


class SortLatencyPolicy(BasePolicy):
    name = "sort_latency"

    def __init__(self, specs: list[ProviderSpec], slo_ms: float, profile_window_sec: float) -> None:
        super().__init__(specs, slo_ms, profile_window_sec)
        self.sentinel_spec = self._make_sort_latency_sentinel(specs)

    @staticmethod
    def _make_sort_latency_sentinel(specs: list[ProviderSpec]) -> ProviderSpec | None:
        base = next(
            (s for s in specs if s.transport_cfg.transport == "openrouter"),
            None,
        )
        if base is None:
            return None
        cfg = TransportConfig(
            name="__sort_latency__",
            transport="openrouter",
            model=base.transport_cfg.model,
            provider_hint=None,
            base_url=base.transport_cfg.base_url,
            api_key_env=base.transport_cfg.api_key_env,
            input_price_per_m=base.input_price_per_m,
            output_price_per_m=base.output_price_per_m,
            extra_headers=dict(base.transport_cfg.extra_headers),
        )
        # Payload builder uses "sort=latency" if provider_hint starts with
        # a sentinel prefix; actual wiring happens in the executor.
        return ProviderSpec(
            name="__sort_latency__",
            tier="api",
            transport_cfg=cfg,
            quota_window_sec=None,
            quota_requests=None,
            concurrency_limit=None,
        )

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        if self.sentinel_spec is None:
            return RoutingDecision(primary=None, notes="no_openrouter_spec")
        return RoutingDecision(primary="__sort_latency__", notes="or_sort_latency")


class CheapestFixedPolicy(BasePolicy):
    name = "cheapest_fixed"

    def __init__(self, specs: list[ProviderSpec], slo_ms: float, profile_window_sec: float) -> None:
        super().__init__(specs, slo_ms, profile_window_sec)

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        candidates = [
            (state, request_cost_for_spec(state.spec, ctx))
            for state in self.states.values()
            if state.is_available(now)
        ]
        if not candidates:
            return RoutingDecision(primary=None, notes="none_available")
        candidates.sort(key=lambda pair: (pair[1], pair[0].spec.name))
        return RoutingDecision(primary=candidates[0][0].spec.name, notes="cheapest")


class FastestFixedPolicy(BasePolicy):
    name = "fastest_fixed"

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        candidates = []
        for state in self.states.values():
            if not state.is_available(now):
                continue
            median = state.profile.median_ms(now)
            if median is None:
                median = float("inf")
            candidates.append((median, state))
        if not candidates:
            return RoutingDecision(primary=None, notes="none_available")
        candidates.sort(key=lambda pair: (pair[0], pair[1].spec.name))
        return RoutingDecision(primary=candidates[0][1].spec.name, notes="fastest")


class TierFirstPolicy(BasePolicy):
    """Heuristic: use the preferred tier first, spill to alternatives."""

    tier_priority: tuple[str, str, str] = ("quota", "concurrency", "api")

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


class OriginalLPHedgePolicy(BasePolicy):
    """min sum_j pi_j * c_eff_j s.t. sum_j pi_j * F_j(SLO) >= 0.99."""

    name = "original_lp_hedge"
    use_hedge = True

    def __init__(
        self,
        specs: list[ProviderSpec],
        slo_ms: float,
        profile_window_sec: float,
    ) -> None:
        super().__init__(specs, slo_ms, profile_window_sec)
        self._slo_target = 0.99

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        feasible: list[ProviderState] = [s for s in self.states.values() if s.is_available(now)]
        if not feasible:
            return RoutingDecision(primary=None, notes="none_available")

        L, U = calibrate_envelopes(
            self.specs,
            ctx.prompt_tokens,
            ctx.completion_tokens_budget,
        )
        request_costs = {
            state.spec.name: request_cost_for_spec(state.spec, ctx)
            for state in feasible
        }
        c_eff = {
            state.spec.name: effective_cost(
                state,
                request_costs[state.spec.name],
                now,
                U=U,
                L=L,
            )
            for state in feasible
        }
        slo_threshold_ms = self.slo_sec * 1000.0
        f_at_slo = {
            state.spec.name: state.profile.cdf_at(slo_threshold_ms, now)
            for state in feasible
        }

        for relax in (self._slo_target, 0.95, 0.90, 0.0):
            weights = _solve_original_lp(feasible, c_eff, f_at_slo, relax)
            if weights:
                primary = _sample_weighted(weights, rng=random.Random(int(now * 1e6)))
                tier_mix = _tier_mix_from_weights(weights, self.states)
                return RoutingDecision(
                    primary=primary,
                    lp_weights=weights,
                    lp_status=f"optimal_f{relax:.2f}",
                    reference_cost_usd=U,
                    c_eff_map=c_eff,
                    tier_mix=tier_mix,
                    notes=f"original_lp_f{relax:.2f}",
                )

        best_state = min(feasible, key=lambda s: c_eff[s.spec.name])
        return RoutingDecision(
            primary=best_state.spec.name,
            lp_weights={best_state.spec.name: 1.0},
            lp_status="best_effort",
            c_eff_map=c_eff,
            notes="original_lp_best_effort",
        )


class BudgetedVhatHedgePolicy(BasePolicy):
    """min sum_j pi_j * mean_TTFT_j s.t. sum_j pi_j * c_eff_j <= tau * v_hat_i."""

    name = "budget_vhat_t100_hedge"
    use_hedge = True

    def __init__(
        self,
        specs: list[ProviderSpec],
        slo_ms: float,
        profile_window_sec: float,
        tau: float = 1.0,
    ) -> None:
        super().__init__(specs, slo_ms, profile_window_sec)
        self.tau = tau

    def route(self, now: float, ctx: RequestContext) -> RoutingDecision:
        feasible: list[ProviderState] = [s for s in self.states.values() if s.is_available(now)]
        if not feasible:
            return RoutingDecision(primary=None, notes="none_available")

        L, U = calibrate_envelopes(
            self.specs,
            ctx.prompt_tokens,
            ctx.completion_tokens_budget,
        )
        request_costs = {
            state.spec.name: request_cost_for_spec(state.spec, ctx)
            for state in feasible
        }
        c_eff = {
            state.spec.name: effective_cost(
                state,
                request_costs[state.spec.name],
                now,
                U=U,
                L=L,
            )
            for state in feasible
        }

        tbar: dict[str, float] = {}
        for state in feasible:
            mean = state.profile.mean_ms(now)
            if mean is None:
                mean = state.profile.median_ms(now)
            if mean is None:
                mean = U * 1000.0  # optimistic large penalty
            tbar[state.spec.name] = float(mean)

        v_hat = slo_safe_anchor_cost(self.states, ctx, self.slo_sec, now)
        budget = self.tau * v_hat

        names = [s.spec.name for s in feasible]
        n = len(names)
        objective = np.array([tbar[name] for name in names], dtype=float)
        upper_constraint = np.array([c_eff[name] for name in names], dtype=float)

        A_ub = upper_constraint.reshape(1, -1)
        b_ub = np.array([budget], dtype=float)
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
        except Exception as exc:  # noqa: BLE001
            result = None
            logging.warning("linprog failed: %s", exc)

        if result is not None and result.success:
            weights_raw = {
                names[i]: float(result.x[i])
                for i in range(n)
                if result.x[i] > 1e-6
            }
            if weights_raw:
                total = sum(weights_raw.values())
                weights = {k: v / total for k, v in weights_raw.items()}
                primary = _sample_weighted(weights, rng=random.Random(int(now * 1e6)))
                tier_mix = _tier_mix_from_weights(weights, self.states)
                return RoutingDecision(
                    primary=primary,
                    lp_weights=weights,
                    lp_status="optimal",
                    budget_usd=float(budget),
                    reference_cost_usd=float(v_hat),
                    c_eff_map=c_eff,
                    tier_mix=tier_mix,
                    notes=f"budget_vhat_t{int(self.tau * 100)}",
                )

        # Fallback: cheapest feasible by c_eff within budget, else lowest c_eff
        affordable = [s for s in feasible if c_eff[s.spec.name] <= budget + 1e-12]
        if affordable:
            choice = min(affordable, key=lambda s: c_eff[s.spec.name])
            return RoutingDecision(
                primary=choice.spec.name,
                lp_weights={choice.spec.name: 1.0},
                lp_status="fallback_cheapest_in_budget",
                budget_usd=float(budget),
                reference_cost_usd=float(v_hat),
                c_eff_map=c_eff,
                notes="fallback_affordable",
            )
        choice = min(feasible, key=lambda s: c_eff[s.spec.name])
        return RoutingDecision(
            primary=choice.spec.name,
            lp_weights={choice.spec.name: 1.0},
            lp_status="fallback_cheapest_overall",
            budget_usd=float(budget),
            reference_cost_usd=float(v_hat),
            c_eff_map=c_eff,
            notes="fallback_no_affordable",
        )


def _solve_original_lp(
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
    # SLO attainment constraint rewritten as inequality: -F >= -slo_target
    A_ub_list = [[-f_at_slo[name] for name in names]]
    b_ub_list = [-slo_target]
    A_eq = np.ones((1, n), dtype=float)
    b_eq = np.array([1.0], dtype=float)
    bounds = [(0.0, 1.0) for _ in range(n)]
    try:
        result = linprog(
            c=objective,
            A_ub=np.array(A_ub_list, dtype=float),
            b_ub=np.array(b_ub_list, dtype=float),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
    except Exception:
        return None
    if not result.success:
        return None
    raw = {names[i]: float(result.x[i]) for i in range(n) if result.x[i] > 1e-6}
    if not raw:
        return None
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _sample_weighted(weights: dict[str, float], rng: random.Random) -> str:
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


# ---------------------------------------------------------------------------
# Hedging (probability-target)
# ---------------------------------------------------------------------------


def compute_hedge_time(
    primary_profile: LatencyProfile,
    backup_profile: LatencyProfile,
    slo_sec: float,
    dispatch_overhead_sec: float,
    now: float,
    success_target: float = HEDGE_SUCCESS_TARGET,
    grid_step_sec: float = 0.05,
) -> float:
    """Return the latest wait time t such that P(success after hedging at t) >= target."""
    slo_ms = slo_sec * 1000.0
    latest_safe: float | None = None
    grid = np.arange(0.0, max(slo_sec, grid_step_sec) + 1e-9, grid_step_sec)
    for elapsed_sec in grid:
        elapsed_ms = float(elapsed_sec) * 1000.0
        F_t = primary_profile.cdf_at(elapsed_ms, now)
        F_L = primary_profile.cdf_at(slo_ms, now)
        survived_t = max(1.0 - F_t, 1e-9)
        p_not_violate = max(0.0, min(1.0, (F_L - F_t) / survived_t))
        p_violate = max(0.0, min(1.0, (1.0 - F_L) / survived_t))
        remaining_sec = slo_sec - float(elapsed_sec) - dispatch_overhead_sec
        p_backup = backup_profile.cdf_at(remaining_sec * 1000.0, now)
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
    dispatch_overhead_sec: float,
    now: float,
    success_target: float = HEDGE_SUCCESS_TARGET,
) -> str | None:
    """Deterministic backup: cheapest among currently SLO-safe non-primary providers."""
    slo_ms = slo_sec * 1000.0
    remaining_ms = (slo_sec - dispatch_overhead_sec) * 1000.0
    safe: list[tuple[float, ProviderState]] = []
    for name, state in states.items():
        if name == primary:
            continue
        if not state.is_available(now):
            continue
        if state.profile.cdf_at(remaining_ms, now) >= success_target:
            safe.append((request_cost_for_spec(state.spec, ctx), state))
    if safe:
        safe.sort(key=lambda pair: (pair[0], pair[1].spec.name))
        return safe[0][1].spec.name

    # fallback: fastest non-primary available (by median)
    others = []
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
# Trace loader
# ---------------------------------------------------------------------------


@dataclass
class TraceRequest:
    arrival_time_sec: float
    prompt: str
    prompt_tokens: int
    max_tokens: int
    use_real_prompt: bool = True


def load_trace_jsonl(
    path: Path,
    max_requests: int | None = None,
    time_compression: float = 1.0,
    trace_start_sec: float = 0.0,
    trace_end_sec: float = float("inf"),
) -> list[TraceRequest]:
    """Load a trace JSONL. Supports both sharegpt-style and day3-style schemas.

    Recognized fields (first non-null wins):
        arrival_time_sec: arrived_at
        prompt:           prompt_text | prompt
        prompt_tokens:    num_prefill_tokens | prompt_tokens
        max_tokens:       num_decode_tokens | max_tokens

    Args:
        time_compression: divisor applied to arrival_time_sec so a long trace
            can be replayed faster (e.g. 8.0 compresses a 24h trace into 3h).
    """
    requests_list: list[TraceRequest] = []
    first_ts: float | None = None
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            ts_raw = record.get("arrived_at")
            ts = float(ts_raw) if ts_raw is not None else 0.0
            if ts < trace_start_sec or ts >= trace_end_sec:
                continue
            if first_ts is None:
                first_ts = ts
            prompt = record.get("prompt_text") or record.get("prompt") or PROBE_PROMPT
            prompt_tokens = (
                record.get("num_prefill_tokens")
                or record.get("prompt_tokens")
                or max(1, len(str(prompt)) // 4)
            )
            max_tokens = (
                record.get("num_decode_tokens")
                or record.get("max_tokens")
                or DEFAULT_MAX_TOKENS
            )
            compressed_arrival = (ts - first_ts) / max(time_compression, 1e-9)
            requests_list.append(
                TraceRequest(
                    arrival_time_sec=compressed_arrival,
                    prompt=str(prompt),
                    prompt_tokens=int(prompt_tokens),
                    max_tokens=int(max_tokens),
                )
            )
            if max_requests is not None and len(requests_list) >= max_requests:
                break
    return requests_list


# Compatibility alias
load_sharegpt_jsonl = load_trace_jsonl


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class JointExecutor:
    def __init__(
        self,
        inventory: InventoryConfig,
        policies: dict[str, BasePolicy],
        transports: dict[str, BaseTransport],
        out_dir: Path,
        run_tag: str,
        duration_sec: float,
        warmup_sec: float,
        slo_ms: float,
    ) -> None:
        self.inventory = inventory
        self.policies = policies
        self.transports = transports
        self.out_dir = out_dir
        self.run_tag = run_tag
        self.duration_sec = duration_sec
        self.warmup_sec = warmup_sec
        self.slo_sec = slo_ms / 1000.0
        self.start_ts = time.time()
        self.request_log_path = out_dir / f"requests_{run_tag}.csv"
        self._log_lock = threading.Lock()
        self._log_fieldnames = [
            "ts",
            "policy",
            "req_id",
            "prompt_tokens",
            "max_tokens",
            "primary_provider",
            "backup_provider",
            "hedge_triggered",
            "hedge_winner",
            "actual_provider",
            "tier",
            "transport",
            "ttft_ms",
            "e2e_ms",
            "status",
            "error_message",
            "billed_cost_usd",
            "primary_cost_usd",
            "backup_cost_usd",
            "lp_status",
            "budget_usd",
            "reference_cost_usd",
            "tier_mix",
            "notes",
        ]
        self._log_fh = self.request_log_path.open("w", newline="")
        self._log_writer = csv.DictWriter(self._log_fh, fieldnames=self._log_fieldnames)
        self._log_writer.writeheader()
        self._thread_local = threading.local()

    # ---- Session pool ----

    def _get_session(self) -> requests.Session:
        sess = getattr(self._thread_local, "session", None)
        if sess is not None:
            return sess
        sess = requests.Session()
        self._thread_local.session = sess
        return sess

    # ---- Low-level send ----

    def _send_to_provider(
        self,
        provider_name: str,
        prompt: str,
        max_tokens: int,
        ttft_event: threading.Event | None = None,
        ttft_info: dict[str, Any] | None = None,
    ) -> SingleRequestResult:
        # Sentinels for OpenRouter-native routing modes.
        if provider_name in ("__sort_latency__", "__openrouter_auto__"):
            base_tx = next(
                (tx for tx in self.transports.values() if tx.cfg.transport == "openrouter"),
                None,
            )
            if base_tx is None:
                return SingleRequestResult(
                    ttft_ms=-1.0,
                    e2e_ms=-1.0,
                    status="error",
                    provider=provider_name,
                    error_message="no_openrouter_transport",
                )
            cfg = TransportConfig(
                name=provider_name,
                transport="openrouter",
                model=base_tx.cfg.model,
                provider_hint=None,
                base_url=base_tx.cfg.base_url,
                api_key_env=base_tx.cfg.api_key_env,
                input_price_per_m=base_tx.cfg.input_price_per_m,
                output_price_per_m=base_tx.cfg.output_price_per_m,
                extra_headers=dict(base_tx.cfg.extra_headers),
            )
            if provider_name == "__sort_latency__":
                return self._send_openrouter_sort(cfg, prompt, max_tokens, sort_key="latency")
            # __openrouter_auto__: plain OR call with no provider hint.
            from experiment.strategies.joint_online_transport import (  # noqa: E402
                OpenAICompatStreamingTransport,
            )
            tx = OpenAICompatStreamingTransport(cfg, self._get_session())
            return tx.send(
                prompt=prompt,
                max_tokens=max_tokens,
                timeout=DEFAULT_TIMEOUT_SEC,
                ttft_event=ttft_event,
                ttft_info=ttft_info,
            )
        transport = self.transports.get(provider_name)
        if transport is None:
            return SingleRequestResult(
                ttft_ms=-1.0,
                e2e_ms=-1.0,
                status="error",
                provider=provider_name,
                error_message=f"unknown_provider:{provider_name}",
            )
        transport.session = self._get_session()
        return transport.send(
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=DEFAULT_TIMEOUT_SEC,
            ttft_event=ttft_event,
            ttft_info=ttft_info,
        )

    def _send_openrouter_sort(
        self,
        cfg: TransportConfig,
        prompt: str,
        max_tokens: int,
        sort_key: str,
    ) -> SingleRequestResult:
        """OpenRouter with sort=<key> payload (no explicit provider order)."""
        from experiment.strategies.joint_online_transport import OpenAICompatStreamingTransport

        tx = OpenAICompatStreamingTransport(cfg, self._get_session())

        # monkey-override payload to inject sort=latency
        original_build = tx._build_payload

        def _sorted_payload(p: str, mt: int) -> dict[str, Any]:
            payload = original_build(p, mt)
            payload["provider"] = {"sort": sort_key}
            return payload

        tx._build_payload = _sorted_payload  # type: ignore[assignment]
        return tx.send(prompt=prompt, max_tokens=max_tokens, timeout=DEFAULT_TIMEOUT_SEC)

    # ---- Hedged send ----

    def _send_hedged(
        self,
        policy: BasePolicy,
        primary: str,
        backup: str,
        hedge_delay_sec: float,
        prompt: str,
        max_tokens: int,
    ) -> tuple[SingleRequestResult, SingleRequestResult | None, str, bool]:
        primary_result_holder: dict[str, SingleRequestResult] = {}
        backup_result_holder: dict[str, SingleRequestResult | None] = {"r": None}
        primary_ttft_event = threading.Event()
        primary_ttft_info: dict[str, Any] = {}

        def primary_worker() -> None:
            primary_result_holder["r"] = self._send_to_provider(
                primary,
                prompt,
                max_tokens,
                ttft_event=primary_ttft_event,
                ttft_info=primary_ttft_info,
            )

        primary_thread = threading.Thread(target=primary_worker, daemon=True)
        primary_thread.start()

        hedge_triggered = False
        if math.isfinite(hedge_delay_sec) and hedge_delay_sec >= 0:
            fired_before_primary = primary_ttft_event.wait(timeout=hedge_delay_sec)
            if not fired_before_primary:
                hedge_triggered = True

                def backup_worker() -> None:
                    backup_result_holder["r"] = self._send_to_provider(
                        backup, prompt, max_tokens
                    )

                backup_thread = threading.Thread(target=backup_worker, daemon=True)
                backup_thread.start()
                primary_thread.join(timeout=DEFAULT_TIMEOUT_SEC)
                backup_thread.join(timeout=DEFAULT_TIMEOUT_SEC)
            else:
                primary_thread.join(timeout=DEFAULT_TIMEOUT_SEC)
        else:
            primary_thread.join(timeout=DEFAULT_TIMEOUT_SEC)

        primary_result = primary_result_holder.get("r") or SingleRequestResult(
            ttft_ms=-1.0,
            e2e_ms=-1.0,
            status="error",
            provider=primary,
            error_message="primary_missing",
        )
        backup_result = backup_result_holder.get("r")

        # Decide winner
        winner = "primary"
        if (
            hedge_triggered
            and backup_result is not None
            and backup_result.first_token_ts is not None
            and (
                primary_result.first_token_ts is None
                or backup_result.first_token_ts < primary_result.first_token_ts
            )
        ):
            winner = "backup"
        return primary_result, backup_result, winner, hedge_triggered

    # ---- Per-policy request handling ----

    def _run_policy_request(
        self,
        policy: BasePolicy,
        req: TraceRequest,
        now: float,
    ) -> None:
        ctx = RequestContext(
            prompt_tokens=max(1, req.prompt_tokens),
            completion_tokens_budget=max(1, req.max_tokens or DEFAULT_MAX_TOKENS),
        )
        decision = policy.route(now, ctx)

        if decision.primary is None:
            self._write_log(
                policy=policy.name,
                req=req,
                ctx=ctx,
                primary=None,
                backup=None,
                primary_result=None,
                backup_result=None,
                hedge_triggered=False,
                hedge_winner=None,
                decision=decision,
            )
            return

        # Estimated service time (for concurrency hold).
        expected_service_sec = max(
            0.5, req.max_tokens / 40.0 if req.max_tokens else 5.0
        )

        prompt = req.prompt if req.use_real_prompt else PROBE_PROMPT
        max_tokens = req.max_tokens or DEFAULT_MAX_TOKENS

        use_hedge = getattr(policy, "use_hedge", False)
        primary_result: SingleRequestResult
        backup_result: SingleRequestResult | None = None
        hedge_triggered = False
        hedge_winner: str | None = None
        backup_name: str | None = None

        if use_hedge:
            backup_name = select_safe_cheapest_backup(
                primary=decision.primary,
                states=policy.states,
                ctx=ctx,
                slo_sec=self.slo_sec,
                dispatch_overhead_sec=HEDGE_DISPATCH_OVERHEAD_SEC,
                now=now,
            )
            if (
                backup_name is not None
                and backup_name != decision.primary
                and decision.primary in policy.states
            ):
                primary_profile = policy.states[decision.primary].profile
                backup_profile = policy.states[backup_name].profile
                hedge_delay_sec = compute_hedge_time(
                    primary_profile=primary_profile,
                    backup_profile=backup_profile,
                    slo_sec=self.slo_sec,
                    dispatch_overhead_sec=HEDGE_DISPATCH_OVERHEAD_SEC,
                    now=now,
                )
                policy.charge_capacity(decision.primary, now, expected_service_sec)
                (
                    primary_result,
                    backup_result,
                    hedge_winner,
                    hedge_triggered,
                ) = self._send_hedged(
                    policy,
                    decision.primary,
                    backup_name,
                    hedge_delay_sec,
                    prompt,
                    max_tokens,
                )
                if hedge_triggered and backup_name is not None:
                    policy.charge_capacity(backup_name, now, expected_service_sec)
            else:
                policy.charge_capacity(decision.primary, now, expected_service_sec)
                primary_result = self._send_to_provider(decision.primary, prompt, max_tokens)
        else:
            policy.charge_capacity(decision.primary, now, expected_service_sec)
            primary_result = self._send_to_provider(decision.primary, prompt, max_tokens)

        # Feed profiles: primary always, backup if hedge fired.
        if primary_result.status == "success" and primary_result.ttft_ms > 0:
            policy.add_sample(
                provider=decision.primary,
                ts=primary_result.start_ts,
                ttft_ms=primary_result.ttft_ms,
                error_type=None,
            )
        elif primary_result.status != "success":
            policy.add_sample(
                provider=decision.primary,
                ts=primary_result.start_ts,
                ttft_ms=-1.0,
                error_type=primary_result.status,
            )
        if hedge_triggered and backup_result is not None and backup_name is not None:
            if backup_result.status == "success" and backup_result.ttft_ms > 0:
                policy.add_sample(
                    provider=backup_name,
                    ts=backup_result.start_ts,
                    ttft_ms=backup_result.ttft_ms,
                    error_type=None,
                )
            elif backup_result.status != "success":
                policy.add_sample(
                    provider=backup_name,
                    ts=backup_result.start_ts,
                    ttft_ms=-1.0,
                    error_type=backup_result.status,
                )

        self._write_log(
            policy=policy.name,
            req=req,
            ctx=ctx,
            primary=decision.primary,
            backup=backup_name,
            primary_result=primary_result,
            backup_result=backup_result,
            hedge_triggered=hedge_triggered,
            hedge_winner=hedge_winner,
            decision=decision,
        )

    def _write_log(
        self,
        policy: str,
        req: TraceRequest,
        ctx: RequestContext,
        primary: str | None,
        backup: str | None,
        primary_result: SingleRequestResult | None,
        backup_result: SingleRequestResult | None,
        hedge_triggered: bool,
        hedge_winner: str | None,
        decision: RoutingDecision,
    ) -> None:
        # Choose the effective served request for latency reporting
        served: SingleRequestResult | None = primary_result
        if hedge_triggered and backup_result is not None and hedge_winner == "backup":
            served = backup_result

        primary_cost = primary_result.billed_cost_usd if primary_result else 0.0
        backup_cost = backup_result.billed_cost_usd if backup_result else 0.0
        total_cost = primary_cost + backup_cost

        row = {
            "ts": time.time(),
            "policy": policy,
            "req_id": req.arrival_time_sec,
            "prompt_tokens": ctx.prompt_tokens,
            "max_tokens": ctx.completion_tokens_budget,
            "primary_provider": primary or "",
            "backup_provider": backup or "",
            "hedge_triggered": "1" if hedge_triggered else "0",
            "hedge_winner": hedge_winner or "",
            "actual_provider": served.provider if served else "",
            "tier": (
                "api"
                if primary in ("__sort_latency__", "__openrouter_auto__")
                else (
                    "unknown"
                    if primary is None
                    else next(
                        (p.tier for p in self.inventory.providers if p.name == primary),
                        "unknown",
                    )
                )
            ),
            "transport": (
                next(
                    (p.transport_cfg.transport for p in self.inventory.providers if p.name == primary),
                    "openrouter" if primary in ("__sort_latency__", "__openrouter_auto__") else "unknown",
                )
                if primary is not None
                else "unknown"
            ),
            "ttft_ms": served.ttft_ms if served else -1.0,
            "e2e_ms": served.e2e_ms if served else -1.0,
            "status": served.status if served else "no_primary",
            "error_message": (served.error_message if served else "") or "",
            "billed_cost_usd": total_cost,
            "primary_cost_usd": primary_cost,
            "backup_cost_usd": backup_cost,
            "lp_status": decision.lp_status,
            "budget_usd": decision.budget_usd if decision.budget_usd is not None else "",
            "reference_cost_usd": (
                decision.reference_cost_usd if decision.reference_cost_usd is not None else ""
            ),
            "tier_mix": json.dumps(decision.tier_mix or {}, sort_keys=True),
            "notes": decision.notes,
        }
        with self._log_lock:
            self._log_writer.writerow(row)
            self._log_fh.flush()

    # ---- Warmup ----

    def warmup(self, per_provider_probes: int = 5) -> None:
        logging.info("Warmup: %d probes per provider x %d providers",
                     per_provider_probes, len(self.inventory.providers))
        threads: list[threading.Thread] = []

        def probe_one(provider_name: str) -> None:
            for _ in range(per_provider_probes):
                try:
                    result = self._send_to_provider(
                        provider_name, PROBE_PROMPT, DEFAULT_MAX_TOKENS
                    )
                except Exception as exc:  # noqa: BLE001
                    logging.warning("warmup %s failed: %s", provider_name, exc)
                    continue
                # Feed all policies with this warmup sample
                if result.status == "success" and result.ttft_ms > 0:
                    for policy in self.policies.values():
                        policy.add_sample(
                            provider_name, result.start_ts, result.ttft_ms, None
                        )
                time.sleep(0.2)

        for spec in self.inventory.providers:
            t = threading.Thread(target=probe_one, args=(spec.name,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=120)

    # ---- Main replay ----

    def replay(self, trace: list[TraceRequest], policy_order: list[str]) -> None:
        run_start = time.time()
        logging.info("Replay: %d requests, %d policies, duration %ss",
                     len(trace), len(policy_order), self.duration_sec)

        policy_cycle_len = len(policy_order)

        # Dispatch each request to one policy via round-robin hash. We wait
        # until the request's intended arrival time (minus run_start) before
        # dispatching. A 5s polling step lets us honour --duration-sec even
        # during long inter-arrival gaps without dispatching early.
        workers: list[threading.Thread] = []
        duration_exhausted = False
        for idx, req in enumerate(trace):
            while True:
                elapsed = time.time() - run_start
                if elapsed >= self.duration_sec:
                    duration_exhausted = True
                    break
                wait_sec = req.arrival_time_sec - elapsed
                if wait_sec <= 0:
                    break
                time.sleep(min(wait_sec, 5.0))

            if duration_exhausted:
                break

            policy_name = policy_order[idx % policy_cycle_len]
            policy = self.policies[policy_name]

            def _run(p=policy, r=req, now_ts=time.time()) -> None:
                try:
                    self._run_policy_request(p, r, now_ts)
                except Exception as exc:  # noqa: BLE001
                    logging.warning("policy %s req %d failed: %s", p.name, r.arrival_time_sec, exc)

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            workers.append(t)

        # Wait for all dispatched threads
        for t in workers:
            t.join(timeout=DEFAULT_TIMEOUT_SEC + 30)
        self._log_fh.close()
        logging.info("Replay complete.")


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------


def build_policies(
    inventory: InventoryConfig,
    selected: list[str],
    profile_window_sec: float,
) -> dict[str, BasePolicy]:
    out: dict[str, BasePolicy] = {}
    for name in selected:
        if name == "openrouter_auto":
            out[name] = OpenRouterAutoPolicy(inventory.providers, inventory.primary_slo_ms, profile_window_sec)
        elif name == "sort_latency":
            out[name] = SortLatencyPolicy(inventory.providers, inventory.primary_slo_ms, profile_window_sec)
        elif name == "cheapest_fixed":
            out[name] = CheapestFixedPolicy(inventory.providers, inventory.primary_slo_ms, profile_window_sec)
        elif name == "fastest_fixed":
            out[name] = FastestFixedPolicy(inventory.providers, inventory.primary_slo_ms, profile_window_sec)
        elif name == "quota_first":
            out[name] = QuotaFirstPolicy(inventory.providers, inventory.primary_slo_ms, profile_window_sec)
        elif name == "concurrency_first":
            out[name] = ConcurrencyFirstPolicy(inventory.providers, inventory.primary_slo_ms, profile_window_sec)
        elif name == "original_lp_hedge":
            out[name] = OriginalLPHedgePolicy(inventory.providers, inventory.primary_slo_ms, profile_window_sec)
        elif name == "budget_vhat_t100_hedge":
            out[name] = BudgetedVhatHedgePolicy(
                inventory.providers,
                inventory.primary_slo_ms,
                profile_window_sec,
                tau=1.0,
            )
        else:
            raise ValueError(f"Unknown policy: {name}")
    return out


def build_transports(inventory: InventoryConfig, session: requests.Session) -> dict[str, BaseTransport]:
    tx_map: dict[str, BaseTransport] = {}
    for spec in inventory.providers:
        tx_map[spec.name] = build_transport(spec.transport_cfg, session)
    return tx_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 joint online evaluation.")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=REPO_ROOT / "experiment/data/joint_minimax_m25_online.json",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=REPO_ROOT / "experiment/data/sharegpt_prompts_7d.jsonl",
    )
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument(
        "--time-compression",
        type=float,
        default=1.0,
        help="Divide arrival_time_sec by this factor to replay longer traces faster.",
    )
    parser.add_argument(
        "--trace-start-sec",
        type=float,
        default=0.0,
        help="Skip requests with arrived_at below this threshold (in trace seconds).",
    )
    parser.add_argument(
        "--trace-end-sec",
        type=float,
        default=float("inf"),
        help="Skip requests with arrived_at at or above this threshold.",
    )
    parser.add_argument("--duration-sec", type=float, default=3600.0)
    parser.add_argument("--warmup-per-provider", type=int, default=5)
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        default=None,
        help="Policies to run (repeatable). Defaults to all 8.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiment/results/phase6_joint",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Path to .env file to load provider API keys.",
    )
    parser.add_argument("--connectivity-only", action="store_true",
                        help="Run only the warmup/connectivity probes and exit.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_dotenv(args.env)

    inventory = load_inventory(args.inventory)
    selected_policies = args.policies or list(ALL_POLICIES)

    policies = build_policies(
        inventory,
        selected_policies,
        profile_window_sec=PROFILE_WINDOW_SEC,
    )

    session = requests.Session()
    transports = build_transports(inventory, session)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"run_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inventory.json").write_text(
        json.dumps(
            {
                "source": str(args.inventory),
                "providers": [spec.name for spec in inventory.providers],
                "selected_policies": selected_policies,
                "slo_ms": inventory.primary_slo_ms,
                "duration_sec": args.duration_sec,
            },
            indent=2,
        )
    )

    executor = JointExecutor(
        inventory=inventory,
        policies=policies,
        transports=transports,
        out_dir=out_dir,
        run_tag=run_tag,
        duration_sec=args.duration_sec,
        warmup_sec=0,
        slo_ms=inventory.primary_slo_ms,
    )

    executor.warmup(per_provider_probes=args.warmup_per_provider)

    if args.connectivity_only:
        logging.info("Connectivity-only mode; exiting after warmup.")
        return

    trace = load_trace_jsonl(
        args.trace,
        max_requests=args.max_requests,
        time_compression=args.time_compression,
        trace_start_sec=args.trace_start_sec,
        trace_end_sec=args.trace_end_sec,
    )
    if not trace:
        raise RuntimeError(f"Empty trace loaded from {args.trace}")

    executor.replay(trace, policy_order=selected_policies)

    logging.info("Results in %s", out_dir)


if __name__ == "__main__":
    main()
