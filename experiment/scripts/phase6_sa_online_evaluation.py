#!/usr/bin/env python3
# ruff: noqa: I001
"""Phase 6 S_A-only online evaluation for the new LP + hedge design.

This script intentionally keeps the online setup simple:
- S_A-only (OpenRouter API providers only)
- one policy per run
- open-loop arrival-driven trace replay
- new body LP: min mean TTFT s.t. expected cost <= tau * v_hat_i
- new hedge trigger: conditional probability target
- deterministic safe-cheapest backup

It reuses the Phase 5 request execution, probing, warmup, and logging code,
but avoids the old all-policies-per-request replay mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase5_online_evaluation as phase5


SUPPORTED_POLICIES = (
    "openrouter_auto",
    "sort_price",
    "sort_throughput",
    "sort_latency",
    "fastest_fixed",
    "lp_mix",
    "smart_hedge",
    "budget_vhat_t75",
    "budget_vhat_t75_hedge",
    "budget_vhat_t75_hedge_explorer",
    "cheapest_fixed",
)

DEFAULT_PHASE6_POLICIES = (
    "openrouter_auto",
    "sort_price",
    "sort_throughput",
    "sort_latency",
    "cheapest_fixed",
    "fastest_fixed",
    "lp_mix",
    "smart_hedge",
    "budget_vhat_t75",
    "budget_vhat_t75_hedge",
    "budget_vhat_t75_hedge_explorer",
)


@dataclass
class RequestPricingContext:
    prompt_tokens: int
    completion_tokens_budget: int


def load_token_pricing(pricing_file: str) -> dict[str, tuple[float, float]]:
    """Load provider token prices from a JSON file.

    JSON format:
    {
      "Provider": [input_price_per_M, output_price_per_M],
      ...
    }
    """
    with open(pricing_file) as f:
        raw = json.load(f)

    token_pricing: dict[str, tuple[float, float]] = {}
    for provider, prices in raw.items():
        if not isinstance(prices, list | tuple) or len(prices) != 2:
            raise ValueError(f"Invalid pricing entry for provider {provider!r}: {prices!r}")
        token_pricing[provider] = (float(prices[0]), float(prices[1]))
    return token_pricing


def compute_request_cost_usd(
    token_price_per_m: tuple[float, float],
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Compute expected request cost in USD."""
    input_price, output_price = token_price_per_m
    return (input_price * prompt_tokens + output_price * completion_tokens) / 1_000_000.0


def infer_pricing_context(
    req: phase5.TraceRequest,
    default_prompt_tokens: int = 50,
    default_completion_tokens: int = phase5.DEFAULT_MAX_TOKENS,
) -> RequestPricingContext:
    """Infer request token counts for budget/cost calculations."""
    prompt_tokens = (
        req.original_prompt_tokens if req.original_prompt_tokens > 0 else default_prompt_tokens
    )
    completion_budget = req.max_tokens if req.max_tokens > 0 else default_completion_tokens
    return RequestPricingContext(
        prompt_tokens=int(prompt_tokens),
        completion_tokens_budget=int(completion_budget),
    )


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str | None:
    """Sample one provider from normalized weights."""
    if not weights:
        return None

    providers = list(weights.keys())
    probs = np.array([weights[p] for p in providers], dtype=float)
    total = float(np.sum(probs))
    if total <= 0:
        return None
    probs /= total

    draw = rng.random()
    cdf = 0.0
    for provider, prob in zip(providers, probs, strict=True):
        cdf += float(prob)
        if draw <= cdf:
            return provider
    return providers[-1]


class BudgetedSALatencyRouter:
    """Per-request budgeted LP router for S_A-only routing."""

    def __init__(
        self,
        token_pricing: dict[str, tuple[float, float]],
        slo_sec: float,
        tau: float = 0.75,
        window_sec: float = 15 * 60,
        reference_mode: str = "max_provider",
        rng_seed: int = 42,
    ):
        self.token_pricing = token_pricing
        self.slo_sec = slo_sec
        self.tau = tau
        self.reference_mode = reference_mode
        self.rng = random.Random(rng_seed)
        self.profiles: dict[str, phase5.ProviderProfile] = {
            provider: phase5.ProviderProfile(
                provider=provider,
                window_sec=window_sec,
                failure_mode=phase5.FailureMode.INFINITY,
            )
            for provider in token_pricing
        }
        self.last_lp_status: str = "not_run"
        self.last_lp_weights: dict[str, float] = {}
        self.last_budget_usd: float = 0.0
        self.last_reference_cost_usd: float = 0.0

    def add_sample(
        self,
        provider: str,
        timestamp: float,
        ttft_ms: float,
        error_type: str | None = None,
    ) -> None:
        if provider not in self.profiles:
            return
        self.profiles[provider].add_sample(timestamp, ttft_ms, error_type)

    def _body_latency_sec(self, provider: str, now: float) -> float:
        samples = self.profiles[provider].get_samples_before(now)
        if not samples:
            return float("inf")
        return float(np.mean(samples))

    def _reference_cost_usd(self, ctx: RequestPricingContext) -> float:
        costs = [
            compute_request_cost_usd(prices, ctx.prompt_tokens, ctx.completion_tokens_budget)
            for prices in self.token_pricing.values()
        ]
        if not costs:
            return 0.0
        if self.reference_mode == "max_provider":
            return max(costs)
        if self.reference_mode == "mean_provider":
            return float(np.mean(costs))
        raise ValueError(f"Unknown reference_mode: {self.reference_mode}")

    def _request_costs_usd(self, ctx: RequestPricingContext) -> dict[str, float]:
        return {
            provider: compute_request_cost_usd(
                prices, ctx.prompt_tokens, ctx.completion_tokens_budget
            )
            for provider, prices in self.token_pricing.items()
        }

    def route(
        self,
        now: float,
        ctx: RequestPricingContext,
    ) -> str | None:
        request_costs = self._request_costs_usd(ctx)
        reference_cost = self._reference_cost_usd(ctx)
        budget_usd = self.tau * reference_cost

        self.last_budget_usd = budget_usd
        self.last_reference_cost_usd = reference_cost

        providers = list(self.profiles.keys())
        if not providers:
            self.last_lp_status = "no_providers"
            self.last_lp_weights = {}
            return None

        body_latency = np.array([self._body_latency_sec(p, now) for p in providers], dtype=float)
        costs = np.array([request_costs[p] for p in providers], dtype=float)
        feasible_mask = np.isfinite(body_latency)

        if not np.any(feasible_mask):
            cheapest_provider = min(providers, key=lambda p: request_costs[p])
            self.last_lp_status = "no_profile_fallback"
            self.last_lp_weights = {cheapest_provider: 1.0}
            return cheapest_provider

        feasible_providers = [p for p, keep in zip(providers, feasible_mask, strict=True) if keep]
        feasible_lat = body_latency[feasible_mask]
        feasible_cost = costs[feasible_mask]

        affordable = feasible_cost <= budget_usd + 1e-12
        if not np.any(affordable):
            cheapest_provider = min(feasible_providers, key=lambda p: request_costs[p])
            self.last_lp_status = "no_in_budget_fallback"
            self.last_lp_weights = {cheapest_provider: 1.0}
            return cheapest_provider

        n = len(feasible_providers)
        A_ub = feasible_cost.reshape(1, -1)
        b_ub = np.array([budget_usd], dtype=float)
        A_eq = np.ones((1, n), dtype=float)
        b_eq = np.array([1.0], dtype=float)
        bounds = [(0.0, None) for _ in range(n)]

        result = linprog(
            c=feasible_lat,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if result.success:
            weights = {
                provider: float(weight)
                for provider, weight in zip(feasible_providers, result.x, strict=True)
                if weight > 1e-6
            }
            total = sum(weights.values())
            if total > 0:
                weights = {provider: float(weight / total) for provider, weight in weights.items()}
                self.last_lp_status = "optimal"
                self.last_lp_weights = weights
                return weighted_choice(self.rng, weights)

        cheapest_affordable = min(
            [p for p in feasible_providers if request_costs[p] <= budget_usd + 1e-12],
            key=lambda p: request_costs[p],
        )
        self.last_lp_status = "solver_failure_fallback"
        self.last_lp_weights = {cheapest_affordable: 1.0}
        return cheapest_affordable


def provider_cdf(profile: phase5.ProviderProfile, threshold_sec: float, now: float) -> float:
    """Return P(T <= threshold) with failures treated as misses."""
    if threshold_sec <= 0:
        return 0.0
    return float(np.clip(profile.get_cdf_at(threshold_sec, now), 0.0, 1.0))


def combined_success_probability_after_hedge(
    primary_profile: phase5.ProviderProfile,
    backup_profile: phase5.ProviderProfile,
    elapsed_sec: float,
    slo_sec: float,
    dispatch_overhead_sec: float,
    now: float,
) -> float:
    """Compute the conditional success probability after dispatching a backup at t."""
    F_t = provider_cdf(primary_profile, elapsed_sec, now)
    F_L = provider_cdf(primary_profile, slo_sec, now)
    survived_t = max(1.0 - F_t, 1e-9)

    p_not_violate = max(0.0, min(1.0, (F_L - F_t) / survived_t))
    p_violate = max(0.0, min(1.0, (1.0 - F_L) / survived_t))

    remaining = slo_sec - elapsed_sec - dispatch_overhead_sec
    p_backup_succeeds = provider_cdf(backup_profile, remaining, now)
    return p_not_violate + p_violate * p_backup_succeeds


def latest_safe_dispatch_time(
    primary_profile: phase5.ProviderProfile,
    backup_profile: phase5.ProviderProfile,
    slo_sec: float,
    dispatch_overhead_sec: float,
    now: float,
    success_target: float = 0.99,
    grid_step_sec: float = 0.05,
) -> float:
    """Return the latest backup dispatch time that still meets the target success probability."""
    latest_safe: float | None = None
    grid = np.arange(0.0, max(slo_sec, grid_step_sec) + 1e-9, grid_step_sec)

    for elapsed_sec in grid:
        combined = combined_success_probability_after_hedge(
            primary_profile=primary_profile,
            backup_profile=backup_profile,
            elapsed_sec=float(elapsed_sec),
            slo_sec=slo_sec,
            dispatch_overhead_sec=dispatch_overhead_sec,
            now=now,
        )
        if combined >= success_target:
            latest_safe = float(elapsed_sec)

    if latest_safe is not None:
        return latest_safe

    no_hedge_success = provider_cdf(primary_profile, slo_sec, now)
    immediate_hedge_success = combined_success_probability_after_hedge(
        primary_profile=primary_profile,
        backup_profile=backup_profile,
        elapsed_sec=0.0,
        slo_sec=slo_sec,
        dispatch_overhead_sec=dispatch_overhead_sec,
        now=now,
    )
    if immediate_hedge_success > no_hedge_success:
        return 0.0
    return float("inf")


def p50_latency_sec(profile: phase5.ProviderProfile, now: float) -> float:
    samples = profile.get_samples_before(now)
    if not samples:
        return float("inf")
    return float(np.percentile(samples, 50))


def select_safe_fastest_backup(
    primary: str,
    profiles: dict[str, phase5.ProviderProfile],
    slo_sec: float,
    dispatch_overhead_sec: float,
    now: float,
    success_target: float = 0.99,
) -> str | None:
    """Pick the fastest other provider that can satisfy the SLO if started immediately.

    Rationale: hedge is a rare SLO-rescue path, not a cost-optimization lever.
    The primary selector (budget-constrained LP) already optimizes cost; the
    backup must prioritize latency to actually save tail requests. This matches
    the simulator spec in JOINT_DESIGN.md (fastest cross-tier backup) and the
    smart_hedge policy inherited from phase 5.
    """
    remaining = slo_sec - dispatch_overhead_sec
    candidates = []
    for provider in profiles:
        if provider == primary:
            continue
        if provider_cdf(profiles[provider], remaining, now) >= success_target:
            candidates.append(provider)

    if candidates:
        return min(candidates, key=lambda p: p50_latency_sec(profiles[p], now))

    others = [provider for provider in profiles if provider != primary]
    if not others:
        return None
    return min(others, key=lambda p: p50_latency_sec(profiles[p], now))


class Phase6SAEvaluator(phase5.Phase5OnlineEvaluator):
    """S_A-only evaluator with the new budget LP and probability-based hedge."""

    def __init__(
        self,
        config: phase5.EvaluationConfig,
        avg_request_costs: dict[str, float],
        token_pricing: dict[str, tuple[float, float]],
        output_dir: Path,
        tau: float = 0.75,
        reference_mode: str = "max_provider",
        success_target: float = 0.99,
        rng_seed: int = 42,
    ):
        super().__init__(config=config, pricing=avg_request_costs, output_dir=output_dir)
        self.token_pricing = token_pricing
        self.success_target = success_target
        self.budget_router = BudgetedSALatencyRouter(
            token_pricing=token_pricing,
            slo_sec=config.slo_sec,
            tau=tau,
            window_sec=config.window_sec,
            reference_mode=reference_mode,
            rng_seed=rng_seed,
        )
        self.policies = {name: phase5.PolicyState(name=name) for name in SUPPORTED_POLICIES}
        self.eval_providers = sorted(token_pricing.keys())
        self.cheapest_provider = min(
            self.eval_providers,
            key=lambda p: avg_request_costs.get(p, float("inf")),
        )
        self.fastest_provider = "Groq" if "Groq" in avg_request_costs else self.cheapest_provider
        self._state_lock = threading.Lock()

    def _refresh_fastest_provider(self) -> None:
        """Update fastest_fixed from empirical P50s once profiles exist."""
        candidate_profiles = []
        now = time.time()
        for provider in self.eval_providers:
            samples = self.router.profiles[provider].get_samples_before(now)
            if samples:
                candidate_profiles.append((provider, float(np.percentile(samples, 50))))

        if candidate_profiles:
            self.fastest_provider = min(candidate_profiles, key=lambda item: item[1])[0]

    def _sync_budget_router_from_phase5_profiles(self) -> None:
        for provider in self.eval_providers:
            if provider not in self.router.profiles or provider not in self.budget_router.profiles:
                continue
            self.budget_router.profiles[provider].samples = list(
                self.router.profiles[provider].samples
            )
            self.budget_router.profiles[provider].error_samples = list(
                self.router.profiles[provider].error_samples
            )
        self._refresh_fastest_provider()

    def warmup(self, duration_sec: float | None = None) -> None:
        super().warmup(duration_sec)
        self._sync_budget_router_from_phase5_profiles()

    def _run_probing_round(self, log_to_csv: bool = True) -> dict[str, float]:
        results = super()._run_probing_round(log_to_csv=log_to_csv)
        self._sync_budget_router_from_phase5_profiles()
        return results

    def _update_router_from_result(self, result: phase5.RequestResult) -> None:
        super()._update_router_from_result(result)
        if result.selected_provider is None or result.actual_provider not in self.token_pricing:
            return
        error_type = None if result.status == "success" else "server_error"
        ttft_ms = result.ttft_ms if result.status == "success" else -1.0
        with self._router_lock:
            self.budget_router.add_sample(
                provider=result.actual_provider,
                timestamp=result.timestamp,
                ttft_ms=ttft_ms,
                error_type=error_type,
            )

    def _select_provider_for_policy(
        self,
        policy: str,
        now: float,
        ctx: RequestPricingContext,
    ) -> str | None:
        if policy == "openrouter_auto":
            return None
        if policy in phase5.SORT_POLICIES:
            return phase5.SORT_POLICIES[policy]
        if policy == "cheapest_fixed":
            return self.cheapest_provider
        if policy == "fastest_fixed":
            return self.fastest_provider
        if policy in ("lp_mix", "smart_hedge"):
            with self._router_lock:
                return self.router.route(now)
        if policy in (
            "budget_vhat_t75",
            "budget_vhat_t75_hedge",
            "budget_vhat_t75_hedge_explorer",
        ):
            with self._router_lock:
                return self.budget_router.route(now, ctx)
        raise ValueError(f"Unsupported policy: {policy}")

    def _record_direct_sample(
        self,
        provider: str,
        timestamp: float,
        ttft_ms: float,
        error_type: str | None = None,
    ) -> None:
        """Feed a direct sample into both the old and new routers."""
        with self._router_lock:
            self.router.add_sample(
                provider=provider,
                timestamp=timestamp,
                ttft_ms=ttft_ms,
                error_type=error_type,
            )
            self.v2_router.add_sample(
                provider=provider,
                timestamp=timestamp,
                ttft_ms=ttft_ms,
                error_type=error_type,
            )
            self.budget_router.add_sample(
                provider=provider,
                timestamp=timestamp,
                ttft_ms=ttft_ms,
                error_type=error_type,
            )

    def _execute_policy_request(
        self,
        policy: str,
        now: float,
        req: phase5.TraceRequest,
    ) -> phase5.RequestResult:
        request_id = str(req.arrival_time_sec).replace(".", "_")
        ctx = infer_pricing_context(req)
        selected_provider = self._select_provider_for_policy(policy, now, ctx)
        hedge_triggered = False
        hedge_provider = None
        hedge_winner = None
        cost_is_estimated = False

        prompt = req.prompt if req.use_real_prompt else None
        max_tokens = req.max_tokens if req.use_real_prompt else None

        if policy == "smart_hedge" and selected_provider is not None:
            with self._router_lock:
                backup = phase5.select_backup(
                    method=phase5.BackupSelectionMethod.FASTEST,
                    profiles=self.router.profiles,
                    costs=self.pricing,
                    primary=selected_provider,
                    slo_sec=self.config.slo_sec,
                    now=now,
                    lp_weights=self.router.last_lp_weights,
                )
                if backup is not None and backup != selected_provider:
                    hedge_time = self.hedger.compute_hedge_time(
                        selected_provider,
                        backup,
                        self.router.profiles,
                        now,
                    )
                else:
                    hedge_time = float("inf")

            hedge_provider = backup

            if backup is not None and backup != selected_provider and math.isfinite(hedge_time):
                primary_result, backup_result, winner, hedge_triggered = self._send_hedged_request(
                    primary_provider=selected_provider,
                    backup_provider=backup,
                    hedge_time_sec=hedge_time,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                hedge_winner = winner

                if winner == "primary" or backup_result is None:
                    result = primary_result
                    actual_provider = result.actual_provider if result else "unknown"
                else:
                    result = backup_result
                    actual_provider = result.actual_provider if result else "unknown"

                overall_start = primary_result.start_ts if primary_result else now
                if result and result.first_token_ts is not None and result.status == "success":
                    ttft_ms = (result.first_token_ts - overall_start) * 1000.0
                else:
                    ttft_ms = result.ttft_ms if result else -1.0

                if result and result.e2e_ms > 0:
                    end_ts = result.start_ts + (result.e2e_ms / 1000.0)
                    e2e_ms = (end_ts - overall_start) * 1000.0
                else:
                    e2e_ms = result.e2e_ms if result else -1.0

                status = result.status if result else "error"
                error_msg = result.error_message if result else "hedging_failed"
                prompt_tokens = 0
                completion_tokens = 0
                cost = 0.0
                if primary_result is not None:
                    prompt_tokens += primary_result.prompt_tokens
                    completion_tokens += primary_result.completion_tokens
                    if primary_result.cost > 0:
                        cost += primary_result.cost
                    else:
                        cost_is_estimated = True
                if hedge_triggered and backup_result is not None:
                    prompt_tokens += backup_result.prompt_tokens
                    completion_tokens += backup_result.completion_tokens
                    if backup_result.cost > 0:
                        cost += backup_result.cost
                    else:
                        cost_is_estimated = True
            else:
                result = self._send_request(
                    provider=selected_provider,
                    timeout=phase5.DEFAULT_TIMEOUT_SEC,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                ttft_ms = result.ttft_ms
                e2e_ms = result.e2e_ms
                status = result.status
                actual_provider = result.actual_provider
                error_msg = result.error_message
                prompt_tokens = result.prompt_tokens
                completion_tokens = result.completion_tokens
                cost = result.cost
                cost_is_estimated = cost <= 0

        elif (
            policy in ("budget_vhat_t75_hedge", "budget_vhat_t75_hedge_explorer")
            and selected_provider is not None
        ):
            backup = select_safe_fastest_backup(
                primary=selected_provider,
                profiles=self.budget_router.profiles,
                slo_sec=self.config.slo_sec,
                dispatch_overhead_sec=self.config.dispatch_overhead_sec,
                now=now,
                success_target=self.success_target,
            )
            hedge_provider = backup

            if backup is not None and backup != selected_provider:
                with self._router_lock:
                    hedge_time = latest_safe_dispatch_time(
                        primary_profile=self.budget_router.profiles[selected_provider],
                        backup_profile=self.budget_router.profiles[backup],
                        slo_sec=self.config.slo_sec,
                        dispatch_overhead_sec=self.config.dispatch_overhead_sec,
                        now=now,
                        success_target=self.success_target,
                    )

                if math.isfinite(hedge_time):
                    primary_result, backup_result, winner, hedge_triggered = (
                        self._send_hedged_request(
                            primary_provider=selected_provider,
                            backup_provider=backup,
                            hedge_time_sec=hedge_time,
                            prompt=prompt,
                            max_tokens=max_tokens,
                        )
                    )
                    hedge_winner = winner

                    if winner == "primary" or backup_result is None:
                        result = primary_result
                        actual_provider = result.actual_provider if result else "unknown"
                    else:
                        result = backup_result
                        actual_provider = result.actual_provider if result else "unknown"

                    overall_start = primary_result.start_ts if primary_result else now
                    if result and result.first_token_ts is not None and result.status == "success":
                        ttft_ms = (result.first_token_ts - overall_start) * 1000.0
                    else:
                        ttft_ms = result.ttft_ms if result else -1.0

                    if result and result.e2e_ms > 0:
                        end_ts = result.start_ts + (result.e2e_ms / 1000.0)
                        e2e_ms = (end_ts - overall_start) * 1000.0
                    else:
                        e2e_ms = result.e2e_ms if result else -1.0

                    status = result.status if result else "error"
                    error_msg = result.error_message if result else "hedging_failed"
                    prompt_tokens = 0
                    completion_tokens = 0
                    cost = 0.0
                    if primary_result is not None:
                        prompt_tokens += primary_result.prompt_tokens
                        completion_tokens += primary_result.completion_tokens
                        if primary_result.cost > 0:
                            cost += primary_result.cost
                        else:
                            cost_is_estimated = True
                    if hedge_triggered and backup_result is not None:
                        prompt_tokens += backup_result.prompt_tokens
                        completion_tokens += backup_result.completion_tokens
                        if backup_result.cost > 0:
                            cost += backup_result.cost
                        else:
                            cost_is_estimated = True

                        if (
                            policy == "budget_vhat_t75_hedge_explorer"
                            and backup_result.actual_provider != actual_provider
                        ):
                            backup_error = (
                                None if backup_result.status == "success" else "server_error"
                            )
                            backup_ttft_ms = (
                                backup_result.ttft_ms if backup_result.status == "success" else -1.0
                            )
                            self._record_direct_sample(
                                provider=backup_result.actual_provider,
                                timestamp=now,
                                ttft_ms=backup_ttft_ms,
                                error_type=backup_error,
                            )
                else:
                    result = self._send_request(
                        provider=selected_provider,
                        timeout=phase5.DEFAULT_TIMEOUT_SEC,
                        prompt=prompt,
                        max_tokens=max_tokens,
                    )
                    ttft_ms = result.ttft_ms
                    e2e_ms = result.e2e_ms
                    status = result.status
                    actual_provider = result.actual_provider
                    error_msg = result.error_message
                    prompt_tokens = result.prompt_tokens
                    completion_tokens = result.completion_tokens
                    cost = result.cost
                    cost_is_estimated = cost <= 0
            else:
                result = self._send_request(
                    provider=selected_provider,
                    timeout=phase5.DEFAULT_TIMEOUT_SEC,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
                ttft_ms = result.ttft_ms
                e2e_ms = result.e2e_ms
                status = result.status
                actual_provider = result.actual_provider
                error_msg = result.error_message
                prompt_tokens = result.prompt_tokens
                completion_tokens = result.completion_tokens
                cost = result.cost
                cost_is_estimated = cost <= 0
        else:
            result = self._send_request(
                provider=selected_provider,
                timeout=phase5.DEFAULT_TIMEOUT_SEC,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            ttft_ms = result.ttft_ms
            e2e_ms = result.e2e_ms
            status = result.status
            actual_provider = result.actual_provider
            error_msg = result.error_message
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
            cost = result.cost
            cost_is_estimated = cost <= 0

        slo_violated = (status != "success") or (ttft_ms / 1000.0 > self.config.slo_sec)

        return phase5.RequestResult(
            request_id=request_id,
            timestamp=now,
            policy=policy,
            selected_provider=selected_provider,
            actual_provider=actual_provider,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            status=status,
            slo_violated=slo_violated,
            cost_usd=cost,
            hedge_triggered=hedge_triggered,
            hedge_provider=hedge_provider,
            hedge_winner=hedge_winner,
            error_message=error_msg,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_is_estimated=cost_is_estimated,
        )

    def run_trace_replay_single_policy(
        self,
        trace: list[phase5.TraceRequest],
        policy: str,
        speedup: float = 1.0,
    ) -> dict[str, phase5.PolicyState]:
        """Replay a trace for one policy using open-loop arrival scheduling."""
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Unsupported policy {policy!r}. Expected one of: {SUPPORTED_POLICIES}"
            )
        if not trace:
            return self.policies

        start_time = time.time()
        trace_start = trace[0].arrival_time_sec
        last_probe_time = start_time
        threads: list[threading.Thread] = []

        def dispatch(req: phase5.TraceRequest) -> None:
            now = time.time()
            result = self._execute_policy_request(policy, now, req)
            with self._state_lock:
                self.policies[policy].results.append(result)
                self.policies[policy].total_cost += result.cost_usd
                if result.status != "success":
                    self.policies[policy].error_count += 1
                self.total_cost += result.cost_usd
            self._update_router_from_result(result)
            self._log_result(result)

        print(f"\n{'=' * 60}")
        print("Phase 6: S_A-only Online Evaluation")
        print(f"{'=' * 60}")
        print(f"Policy: {policy}")
        print(f"Trace requests: {len(trace)}")
        print(f"Speedup: {speedup}x")
        print(f"SLO: {self.config.slo_sec:.2f}s")
        print(f"Cost cap: ${self.config.cost_cap_usd:.2f}")
        print(f"Providers: {self.eval_providers}")

        for i, req in enumerate(trace):
            target_time = (req.arrival_time_sec - trace_start) / speedup
            elapsed = time.time() - start_time
            if target_time > elapsed:
                time.sleep(target_time - elapsed)

            if self.total_cost > self.config.cost_cap_usd:
                print(f"\nCost cap exceeded (${self.total_cost:.2f}). Stopping at request {i}.")
                break

            now = time.time()
            if (
                self.config.probing_interval_sec > 0
                and (now - last_probe_time) >= self.config.probing_interval_sec
            ):
                print("\n  [Probing] Refreshing provider profiles...")
                self._run_probing_round(log_to_csv=True)
                last_probe_time = now

            t = threading.Thread(target=dispatch, args=(req,), daemon=False)
            threads.append(t)
            t.start()

            # Periodically reap finished threads to keep the list bounded.
            if len(threads) % 25 == 0:
                threads = [thread for thread in threads if thread.is_alive()]

            if (i + 1) % 10 == 0:
                with self._state_lock:
                    completed = len(self.policies[policy].results)
                    total_cost = self.total_cost
                print(
                    f"[{i + 1}/{len(trace)} dispatched] completed={completed} cost=${total_cost:.4f}"
                )

        for thread in threads:
            thread.join()

        return self.policies


class SharedObservationLogger:
    """CSV logger for shared warmup/probing observations."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "shared_observation_log.csv"
        self.total_cost = 0.0
        self.probing_cost = 0.0
        self.warmup_cost = 0.0
        self._lock = threading.Lock()
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "phase",
                    "provider",
                    "actual_provider",
                    "ttft_ms",
                    "e2e_ms",
                    "status",
                    "cost_usd",
                    "cost_is_estimated",
                    "prompt_tokens",
                    "completion_tokens",
                    "error_message",
                ]
            )

    def log(
        self,
        phase: str,
        provider: str,
        result: phase5.SingleRequestResult,
        cost_usd: float,
        cost_is_estimated: bool,
    ) -> None:
        """Append one shared observation."""
        with self._lock, open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    time.time(),
                    phase,
                    provider,
                    result.actual_provider,
                    result.ttft_ms if result.status == "success" else -1.0,
                    result.e2e_ms,
                    result.status,
                    cost_usd,
                    cost_is_estimated,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.error_message,
                ]
            )
        self.total_cost += cost_usd
        if phase == "probing":
            self.probing_cost += cost_usd
        elif phase == "warmup":
            self.warmup_cost += cost_usd


def _make_phase6_evaluator(
    *,
    config: phase5.EvaluationConfig,
    avg_request_costs: dict[str, float],
    token_pricing: dict[str, tuple[float, float]],
    output_dir: Path,
    tau: float,
    reference_mode: str,
    rng_seed: int,
) -> Phase6SAEvaluator:
    """Create a Phase 6 evaluator with shared constructor defaults."""
    return Phase6SAEvaluator(
        config=config,
        avg_request_costs=avg_request_costs,
        token_pricing=token_pricing,
        output_dir=output_dir,
        tau=tau,
        reference_mode=reference_mode,
        rng_seed=rng_seed,
    )


def _force_refresh_router_state(evaluator: Phase6SAEvaluator) -> None:
    """Refresh derived router state after shared observations."""
    with evaluator._router_lock:
        evaluator.router.update_lp(force=True)
        evaluator.v2_router.update_lp(force=True)
    evaluator._refresh_fastest_provider()


def _broadcast_shared_observation(
    evaluators: dict[str, Phase6SAEvaluator],
    result: phase5.SingleRequestResult,
    timestamp: float,
) -> None:
    """Feed a shared warmup/probe result into every policy-local router."""
    error_type = None if result.status == "success" else "server_error"
    ttft_ms = result.ttft_ms if result.status == "success" else -1.0
    for evaluator in evaluators.values():
        evaluator._record_direct_sample(
            provider=result.actual_provider,
            timestamp=timestamp,
            ttft_ms=ttft_ms,
            error_type=error_type,
        )
        evaluator._refresh_fastest_provider()


def _shared_observation_cost(
    result: phase5.SingleRequestResult,
    provider: str,
    avg_request_costs: dict[str, float],
) -> tuple[float, bool]:
    """Return cost and whether it is estimated for a shared observation."""
    if result.cost > 0:
        return result.cost, False
    return avg_request_costs.get(provider, 0.0), True


def run_shared_warmup(
    *,
    prober: Phase6SAEvaluator,
    evaluators: dict[str, Phase6SAEvaluator],
    logger: SharedObservationLogger,
    avg_request_costs: dict[str, float],
    duration_sec: float,
) -> None:
    """Run one shared warmup stream and broadcast observations to all policies."""
    if duration_sec <= 0:
        return

    print(f"\n{'=' * 60}")
    print(f"Phase 6: Shared Warmup ({duration_sec / 60:.1f} minutes)")
    print(f"{'=' * 60}")

    start_time = time.time()
    probe_count = 0
    while time.time() - start_time < duration_sec:
        for provider in prober.eval_providers:
            if time.time() - start_time >= duration_sec:
                break

            probe_count += 1
            now = time.time()
            result = prober._send_request(
                provider=provider,
                timeout=phase5.DEFAULT_TIMEOUT_SEC,
            )
            _broadcast_shared_observation(evaluators, result, now)
            cost, cost_is_estimated = _shared_observation_cost(
                result=result,
                provider=provider,
                avg_request_costs=avg_request_costs,
            )
            logger.log(
                phase="warmup",
                provider=provider,
                result=result,
                cost_usd=cost,
                cost_is_estimated=cost_is_estimated,
            )

            status_str = (
                f"TTFT={result.ttft_ms:.0f}ms" if result.status == "success" else result.status
            )
            elapsed = time.time() - start_time
            print(
                f"  [Shared Warmup {probe_count}] {provider}: {status_str} "
                f"({elapsed / 60:.1f}/{duration_sec / 60:.1f} min)"
            )
            time.sleep(1.0)

    for evaluator in evaluators.values():
        _force_refresh_router_state(evaluator)
    print(f"\nShared warmup complete: {probe_count} probes")


def run_shared_probing_round(
    *,
    prober: Phase6SAEvaluator,
    evaluators: dict[str, Phase6SAEvaluator],
    logger: SharedObservationLogger,
    avg_request_costs: dict[str, float],
) -> dict[str, float]:
    """Probe every provider once, then broadcast the observations."""
    results: dict[str, float] = {}
    print("\n  [Shared Probing] Refreshing provider profiles...")
    for provider in prober.eval_providers:
        now = time.time()
        result = prober._send_request(
            provider=provider,
            timeout=phase5.DEFAULT_TIMEOUT_SEC,
        )
        _broadcast_shared_observation(evaluators, result, now)
        cost, cost_is_estimated = _shared_observation_cost(
            result=result,
            provider=provider,
            avg_request_costs=avg_request_costs,
        )
        logger.log(
            phase="probing",
            provider=provider,
            result=result,
            cost_usd=cost,
            cost_is_estimated=cost_is_estimated,
        )
        results[provider] = result.ttft_ms if result.status == "success" else -1.0
        time.sleep(0.5)

    for evaluator in evaluators.values():
        _force_refresh_router_state(evaluator)
    n_success = sum(1 for ttft_ms in results.values() if ttft_ms > 0)
    print(f"  [Shared Probing] {n_success}/{len(results)} providers responded")
    return results


def run_trace_replay_shared_probing(
    *,
    trace: list[phase5.TraceRequest],
    policies: list[str],
    evaluators: dict[str, Phase6SAEvaluator],
    prober: Phase6SAEvaluator,
    observation_logger: SharedObservationLogger,
    avg_request_costs: dict[str, float],
    speedup: float,
    cost_cap_usd: float,
    probing_interval_sec: float,
) -> dict[str, phase5.PolicyState]:
    """Replay a trace with parallel policies and one shared prober."""
    if not trace:
        return {}

    start_time = time.time()
    trace_start = trace[0].arrival_time_sec
    last_probe_time = start_time
    threads: list[threading.Thread] = []

    print(f"\n{'=' * 60}")
    print("Phase 6: Shared-Prober Paired Online Evaluation")
    print(f"{'=' * 60}")
    print(f"Policies: {policies}")
    print(f"Trace requests: {len(trace)}")
    print(f"Speedup: {speedup}x")
    print(f"Providers: {prober.eval_providers}")
    print("Shared observations feed all policy-local routers.")

    def dispatch(policy: str, req: phase5.TraceRequest) -> None:
        evaluator = evaluators[policy]
        now = time.time()
        result = evaluator._execute_policy_request(policy, now, req)
        with evaluator._state_lock:
            evaluator.policies[policy].results.append(result)
            evaluator.policies[policy].total_cost += result.cost_usd
            if result.status != "success":
                evaluator.policies[policy].error_count += 1
            evaluator.total_cost += result.cost_usd
        evaluator._update_router_from_result(result)
        evaluator._log_result(result)

    for i, req in enumerate(trace):
        target_time = (req.arrival_time_sec - trace_start) / speedup
        elapsed = time.time() - start_time
        if target_time > elapsed:
            time.sleep(target_time - elapsed)

        if any(evaluator.total_cost > cost_cap_usd for evaluator in evaluators.values()):
            print(f"\nCost cap exceeded by at least one policy. Stopping at request {i}.")
            break

        now = time.time()
        if probing_interval_sec > 0 and (now - last_probe_time) >= probing_interval_sec:
            run_shared_probing_round(
                prober=prober,
                evaluators=evaluators,
                logger=observation_logger,
                avg_request_costs=avg_request_costs,
            )
            last_probe_time = time.time()

        for policy in policies:
            thread = threading.Thread(target=dispatch, args=(policy, req), daemon=False)
            threads.append(thread)
            thread.start()

        if len(threads) >= max(25, len(policies) * 4):
            alive_threads = []
            for thread in threads:
                if thread.is_alive():
                    alive_threads.append(thread)
            threads = alive_threads

        if (i + 1) % 10 == 0:
            completed = {
                policy: len(evaluators[policy].policies[policy].results) for policy in policies
            }
            costs = {policy: evaluators[policy].total_cost for policy in policies}
            min_completed = min(completed.values()) if completed else 0
            max_cost = max(costs.values()) if costs else 0.0
            print(
                f"[{i + 1}/{len(trace)} dispatched] "
                f"min_completed={min_completed} max_policy_cost=${max_cost:.4f}"
            )

    for thread in threads:
        thread.join()

    merged: dict[str, phase5.PolicyState] = {}
    for policy, evaluator in evaluators.items():
        merged[policy] = evaluator.policies[policy]
    print(
        f"Shared observation cost: total=${observation_logger.total_cost:.4f} "
        f"warmup=${observation_logger.warmup_cost:.4f} "
        f"probing=${observation_logger.probing_cost:.4f}"
    )
    return merged


def save_combined_summary(
    *,
    evaluators: dict[str, Phase6SAEvaluator],
    output_dir: Path,
) -> None:
    """Save one combined summary JSON/CSV for shared-coordinator runs."""
    combined: dict[str, dict] = {}
    for evaluator in evaluators.values():
        stats = evaluator.analyze_results()
        combined.update(stats)
        evaluator.print_summary(stats)
        evaluator.save_summary(stats)

    json_path = output_dir / "combined_evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)

    csv_path = output_dir / "combined_evaluation_summary.csv"
    fieldnames = [
        "policy",
        "total_requests",
        "successful_requests",
        "error_rate",
        "slo_violation_rate",
        "total_cost",
        "avg_cost",
        "avg_cost_real",
        "real_cost_count",
        "estimated_cost_count",
        "p50_ms",
        "p90_ms",
        "p99_ms",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for policy, stats in combined.items():
            row = {"policy": policy}
            row.update({field: stats.get(field) for field in fieldnames if field != "policy"})
            writer.writerow(row)

    print(f"\nSaved combined JSON summary to {json_path}")
    print(f"Saved combined CSV summary to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 S_A-only online evaluation")
    parser.add_argument("--model", type=str, default="qwen/qwen3-235b-a22b-2507")
    parser.add_argument("--policy", type=str, choices=SUPPORTED_POLICIES)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=SUPPORTED_POLICIES,
        help="Run multiple policies in one shared-prober paired coordinator.",
    )
    parser.add_argument("--pricing", type=str, required=True, help="Path to provider pricing JSON")
    parser.add_argument("--trace", type=str, required=True, help="Path to trace CSV/JSONL")
    parser.add_argument("--output", type=str, default="experiment/results/phase6_sa_online")
    parser.add_argument("--slo", type=float, default=2.0)
    parser.add_argument("--warmup", type=float, default=300.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--probing-interval", type=float, default=300.0)
    parser.add_argument("--cost-cap", type=float, default=10.0)
    parser.add_argument("--speedup", type=float, default=1.0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--real-prompts", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--tau", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-mode",
        type=str,
        default="max_provider",
        choices=("max_provider", "mean_provider"),
        help="How to anchor v_hat_i from the provider pricing set.",
    )
    args = parser.parse_args()

    if args.policy is None and not args.policies:
        parser.error("one of --policy or --policies is required")
    if args.policy is not None and args.policies:
        parser.error("--policy and --policies are mutually exclusive")

    if not phase5.API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    if not os.path.exists(args.pricing):
        print(f"Error: pricing file not found: {args.pricing}")
        sys.exit(1)

    token_pricing = load_token_pricing(args.pricing)
    avg_request_costs = phase5.load_pricing(args.pricing)

    if args.trace.endswith(".jsonl"):
        trace = phase5.load_trace_from_jsonl(
            args.trace,
            max_requests=args.max_requests,
            time_limit_sec=args.time_limit,
            use_real_prompts=args.real_prompts,
        )
    else:
        trace = phase5.load_trace_from_csv(
            args.trace,
            max_requests=args.max_requests,
            time_limit_sec=args.time_limit,
        )

    if not trace:
        print("Error: no requests loaded from trace")
        sys.exit(1)

    config = phase5.EvaluationConfig(
        model=args.model,
        slo_sec=args.slo,
        warmup_sec=args.warmup,
        duration_sec=args.time_limit if args.time_limit is not None else 3600.0,
        cost_cap_usd=args.cost_cap,
        probing_interval_sec=args.probing_interval,
        max_tokens=args.max_output_tokens,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_slug = args.model.replace("/", "__").replace(":", "_").replace(".", "_").replace("-", "_")
    if args.policies:
        policies = list(dict.fromkeys(args.policies))
        output_dir = Path(args.output) / f"{model_slug}__shared_prober__{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        evaluators = {
            policy: _make_phase6_evaluator(
                config=config,
                avg_request_costs=avg_request_costs,
                token_pricing=token_pricing,
                output_dir=output_dir / policy,
                tau=args.tau,
                reference_mode=args.reference_mode,
                rng_seed=args.seed,
            )
            for policy in policies
        }
        prober = _make_phase6_evaluator(
            config=config,
            avg_request_costs=avg_request_costs,
            token_pricing=token_pricing,
            output_dir=output_dir / "_shared_prober_internal",
            tau=args.tau,
            reference_mode=args.reference_mode,
            rng_seed=args.seed,
        )
        observation_logger = SharedObservationLogger(output_dir=output_dir)

        if not args.skip_warmup and args.warmup > 0:
            run_shared_warmup(
                prober=prober,
                evaluators=evaluators,
                logger=observation_logger,
                avg_request_costs=avg_request_costs,
                duration_sec=args.warmup,
            )

        run_trace_replay_shared_probing(
            trace=trace,
            policies=policies,
            evaluators=evaluators,
            prober=prober,
            observation_logger=observation_logger,
            avg_request_costs=avg_request_costs,
            speedup=args.speedup,
            cost_cap_usd=args.cost_cap,
            probing_interval_sec=args.probing_interval,
        )
        save_combined_summary(evaluators=evaluators, output_dir=output_dir)
        return

    output_dir = Path(args.output) / f"{model_slug}__{args.policy}__{timestamp}"
    evaluator = _make_phase6_evaluator(
        config=config,
        avg_request_costs=avg_request_costs,
        token_pricing=token_pricing,
        output_dir=output_dir,
        tau=args.tau,
        reference_mode=args.reference_mode,
        rng_seed=args.seed,
    )

    if not args.skip_warmup and args.warmup > 0:
        evaluator.warmup(duration_sec=args.warmup)

    evaluator.run_trace_replay_single_policy(
        trace=trace,
        policy=args.policy,
        speedup=args.speedup,
    )

    stats = evaluator.analyze_results()
    evaluator.print_summary(stats)
    evaluator.save_summary(stats)


if __name__ == "__main__":
    main()
