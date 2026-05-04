"""Sidecar LP-budget evaluation harness for synthetic simulation scenarios.

This module intentionally does not register new strategies in the shared
registry. It reuses the merged world model and implements an isolated
evaluation path for the LP formulation pivot study.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
from pathlib import Path
import pickle
import re
from typing import NamedTuple

import numpy as np
from scipy.optimize import linprog

from experiments.simulation import list_scenarios, load_world_scenario
from experiments.simulation.minimax_m25 import make_mm25_scenarios
from experiments.simulation.simple_scenarios import make_simple_scenarios
from rwsim.data import DataLoader
from rwsim.policies.latency_routers.online_lp import SWRRSampler
from rwsim.schemas import Request
from rwsim.strategies.tiered_impl import (
    _pick_cross_tier_backup,
    _reset_all_state,
    _sample_service_time,
    _sample_ttft,
)
from rwsim.metrics import StrategyRun
from rwsim.world import (
    ProviderTier,
    ScenarioConfig,
    TieredProvider,
    calibrate_envelopes,
    effective_cost,
    generate_workload,
    quota_shadow_price,
)

MAIN_VARIANTS = [
    "original_lp",
    "original_lp_hedge",
    "original_lp_explorer",
    "budget_range_p0",
    "budget_range_p25",
    "budget_range_p50",
    "budget_range_p75",
    "budget_range_p100",
    "budget_range_p0_hedge",
    "budget_range_p25_hedge",
    "budget_range_p50_hedge",
    "budget_range_p75_hedge",
    "budget_range_p100_hedge",
    "budget_range_p0_explorer",
    "budget_range_p25_explorer",
    "budget_range_p50_explorer",
    "budget_range_p75_explorer",
    "budget_range_p100_explorer",
    "budget_vhat_t25",
    "budget_vhat_t50",
    "budget_vhat_t75",
    "budget_vhat_t100",
    "budget_vhat_t150",
    "budget_vhat_t200",
    "budget_vhat_t25_hedge",
    "budget_vhat_t50_hedge",
    "budget_vhat_t75_hedge",
    "budget_vhat_t100_hedge",
    "budget_vhat_t150_hedge",
    "budget_vhat_t200_hedge",
    "budget_vhat_t25_explorer",
    "budget_vhat_t50_explorer",
    "budget_vhat_t75_explorer",
    "budget_vhat_t100_explorer",
    "budget_vhat_t150_explorer",
    "budget_vhat_t200_explorer",
]
EXPLORER_VARIANTS = [
    "original_lp_explorer",
    "budget_range_p0_explorer",
    "budget_range_p25_explorer",
    "budget_range_p50_explorer",
    "budget_range_p75_explorer",
    "budget_range_p100_explorer",
    "budget_vhat_t25_explorer",
    "budget_vhat_t50_explorer",
    "budget_vhat_t75_explorer",
    "budget_vhat_t100_explorer",
    "budget_vhat_t150_explorer",
    "budget_vhat_t200_explorer",
]
BACKUP_EXPLORATION_VARIANTS = [
    "original_lp_hedge_randombackup",
    "original_lp_explorer_randombackup",
    "budget_range_p0_hedge_randombackup",
    "budget_range_p0_explorer_randombackup",
    "budget_range_p25_hedge_randombackup",
    "budget_range_p25_explorer_randombackup",
    "budget_range_p50_hedge_randombackup",
    "budget_range_p50_explorer_randombackup",
    "budget_range_p75_hedge_randombackup",
    "budget_range_p75_explorer_randombackup",
    "budget_range_p100_hedge_randombackup",
    "budget_range_p100_explorer_randombackup",
    "budget_vhat_t25_hedge_randombackup",
    "budget_vhat_t25_explorer_randombackup",
    "budget_vhat_t50_hedge_randombackup",
    "budget_vhat_t50_explorer_randombackup",
    "budget_vhat_t75_hedge_randombackup",
    "budget_vhat_t75_explorer_randombackup",
    "budget_vhat_t100_hedge_randombackup",
    "budget_vhat_t100_explorer_randombackup",
    "budget_vhat_t150_hedge_randombackup",
    "budget_vhat_t150_explorer_randombackup",
    "budget_vhat_t200_hedge_randombackup",
    "budget_vhat_t200_explorer_randombackup",
]
HEDGE_ABLATION_VARIANTS = [
    "original_lp_oldhedge",
    "budget_range_p75_oldhedge",
    "budget_vhat_t75_oldhedge",
]
PROVIDER_PERCENTILE_ABLATION_VARIANTS = [
    "budget_body_p25",
    "budget_body_p50",
    "budget_body_p75",
    "budget_body_p25_hedge",
    "budget_body_p50_hedge",
    "budget_body_p75_hedge",
]
# Shadow-price ablation: identical to ``original_lp`` but the LP cost
# vector uses :meth:`TieredProvider.marginal_cost` (zero for S_Q/S_C)
# instead of :func:`effective_cost` (which carries ψ(z) and λ(u) shadow
# prices). Stays out of MAIN_VARIANTS so default legacy runs are not
# polluted; opt in via ``--include-shadow-price-ablation`` on the runner.
SHADOW_PRICE_ABLATION_VARIANTS = [
    "original_lp_rawcost",
    "original_lp_rawcost_hedge",
]
CONTROL_VARIANTS = [
    "cheapest_available",
    "fastest_available",
    "quota_first",
    "concurrency_first",
]
FIRST_BATCH_SCENARIOS = [
    "unified_pool",
]

WINDOW_SEC = 15 * 60.0
WARMUP_WINDOW_SEC = 14 * 60.0
WARMUP_SAMPLES_PER_PROVIDER = 50
BODY_MEAN_MIN_SAMPLES = 5
OLD_TARGETS = (0.99, 0.98, 0.95, 0.90)
LP_EPS = 1e-9
WEIGHT_FLOOR = 1e-8
HEDGE_COST_MULTIPLIER = 10.0
HEDGE_SUCCESS_TARGET = 0.99
HEDGE_DISPATCH_OVERHEAD_MS = 50.0
HEDGE_TIME_GRID_MS = 10.0
BACKUP_RANDOM_PROB_BASE = 0.5
BACKUP_RANDOM_PROB_TARGET_SLO_VIOLATION = 0.05
BACKUP_RANDOM_PROB_ZERO_POINT = 0.10
BACKUP_VIOLATION_WINDOW = 256
TRIVIAL_SUPPORT_THRESHOLD = 1
# NOTE: A ``PROBE_RATE`` constant + dedicated active-probing helper used
# to live here. They were removed on 2026-04-29 along with the
# ``--probe-rate`` runner flag and the ``probe_rate`` parameter on
# ``run_variant()``. The paper-line simulator does not exercise active
# probing:
#   1. The LP body objective uses ground-truth analytical T̄ (see
#      ``_body_latency_proxy_ms``), so it does not depend on a sampled
#      rolling profile.
#   2. Explorer (hedge-as-probe) is excluded from the simulator paper
#      grid; profile freshness is not measured here.
#   3. Production probing cost / capacity / observation-time accounting
#      is a real-experiment concern, handled outside this module.
# If a future profile-freshness debug experiment needs active probing,
# write it as a self-contained script under ``experiments/debug/`` —
# do NOT thread probe_rate back through the paper-line dispatch.
BACKUP_SCOPES = ("any_provider", "cross_tier")
TRACE_WORKLOAD_DATASETS = ("freeinference", "rednote", "sharegpt")
ObservedSample = tuple[str, float, float]  # provider, observation timestamp, TTFT ms

VARIANT_ALIASES = {
    "old_body": "original_lp",
    "old_body_hedge": "original_lp_hedge",
    "old_body_oldhedge": "original_lp_oldhedge",
}

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_TRACE_DATA_ROOT = _WORKSPACE_ROOT / "data"
_DATASET_CACHE_ROOT = _WORKSPACE_ROOT / "outputs" / "cache" / "dataset"
_DATA_LOADER_CONFIG = {"dataset": {}}

_TRACE_DATASET_PATHS = {
    "freeinference": [
        _TRACE_DATA_ROOT / "freeinference_logs.csv",
    ],
    "rednote": [
        _TRACE_DATA_ROOT / "rednote_logs.csv",
    ],
    "sharegpt": [
        _TRACE_DATA_ROOT / "sharegpt_prompts_7d.jsonl",
        _TRACE_DATA_ROOT / "sharegpt_burstgpt" / "sharegpt_prompts_7d.jsonl",
        _TRACE_DATA_ROOT / "sharegpt_burstgpt" / "converted.csv",
    ],
}


def canonicalize_variant_name(variant: str) -> str:
    """Return the canonical public name for a sidecar variant."""
    return VARIANT_ALIASES.get(variant, variant)


def _is_budget_variant(variant: str) -> bool:
    return variant.startswith("budget_")


def _is_provider_percentile_variant(variant: str) -> bool:
    return _base_variant(variant).startswith("budget_body_p")


def _is_range_budget_variant(variant: str) -> bool:
    return _base_variant(variant).startswith("budget_range_p")


def _is_vhat_budget_variant(variant: str) -> bool:
    return _base_variant(variant).startswith("budget_vhat_t")


def _is_control_variant(variant: str) -> bool:
    return canonicalize_variant_name(variant) in CONTROL_VARIANTS


def _strip_random_backup_suffix(variant: str) -> str:
    if variant.endswith("_randombackup"):
        return variant[:-len("_randombackup")]
    return variant


def _hedge_mode(variant: str) -> str:
    variant = _strip_random_backup_suffix(canonicalize_variant_name(variant))
    if variant.endswith("_oldhedge"):
        return "old"
    if variant.endswith("_hedge") or variant.endswith("_explorer"):
        return "new"
    return "none"


def _use_hedge(variant: str) -> bool:
    return _hedge_mode(variant) != "none"


def _use_hedge_as_probe(variant: str) -> bool:
    variant = _strip_random_backup_suffix(canonicalize_variant_name(variant))
    return variant.endswith("_explorer")


def _use_random_backup_exploration(variant: str) -> bool:
    return canonicalize_variant_name(variant).endswith("_randombackup")


def _base_variant(variant: str) -> str:
    variant = _strip_random_backup_suffix(canonicalize_variant_name(variant))
    hedge_mode = _hedge_mode(variant)
    if hedge_mode == "old":
        return variant[:-len("_oldhedge")]
    if variant.endswith("_explorer"):
        return variant[:-len("_explorer")]
    if hedge_mode == "new":
        return variant[:-len("_hedge")]
    return variant


def _budget_level_from_variant(variant: str) -> int:
    match = re.search(r"_(?:p|t)(\d+)$", _base_variant(variant))
    if match is None:
        raise ValueError(f"Variant does not encode a budget level: {variant!r}")
    return int(match.group(1))


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _safe_quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def _normalize_weights(
    names: list[str],
    vector: np.ndarray,
    *,
    floor: float = WEIGHT_FLOOR,
) -> dict[str, float]:
    weights = {
        name: float(vector[idx])
        for idx, name in enumerate(names)
        if float(vector[idx]) > floor
    }
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    return {name: weight / total for name, weight in weights.items()}


def _single_provider_weight(provider: TieredProvider) -> dict[str, float]:
    return {provider.name: 1.0}


def _choose_lp_provider(
    weights: dict[str, float],
    sampler: SWRRSampler,
) -> str:
    if not weights:
        raise ValueError("Cannot choose a provider from empty weights.")
    sampler.update_weights(weights, smoothing=1.0)
    choice = sampler.next()
    if choice is not None:
        return choice
    return max(weights, key=weights.get)


def _mean_tier_fraction(runs: list[StrategyRun]) -> dict[str, float]:
    tier_names = sorted(
        {tier_name for run in runs for tier_name in run.tier_fractions()}
    )
    fractions: dict[str, list[float]] = {name: [] for name in tier_names}
    for run in runs:
        run_fractions = run.tier_fractions()
        for tier_name in tier_names:
            fractions[tier_name].append(run_fractions.get(tier_name, 0.0))
    return {name: float(np.mean(values)) for name, values in sorted(fractions.items())}


def _mean_provider_fraction(runs: list[StrategyRun]) -> dict[str, float]:
    provider_names = sorted(
        {provider_name for run in runs for provider_name in run.provider_fractions()}
    )
    fractions: dict[str, list[float]] = {name: [] for name in provider_names}
    for run in runs:
        run_fractions = run.provider_fractions()
        for provider_name in provider_names:
            fractions[provider_name].append(run_fractions.get(provider_name, 0.0))
    return {name: float(np.mean(values)) for name, values in sorted(fractions.items())}


class SelectorOutcome(NamedTuple):
    """Selector output for one routing decision."""

    weights: dict[str, float]
    status: str
    c_eff: dict[str, float]
    budget_tau: float | None
    expected_c_eff: float | None
    budget_utilization: float | None
    budget_slack: float | None
    true_p50_fallbacks: int
    feasible_count: int


@dataclass
class RollingLatencyProfile:
    """Moving-window TTFT profile used by the sidecar LP selectors."""

    window_sec: float = WINDOW_SEC
    _samples: deque[tuple[float, float]] = field(default_factory=deque)

    def add_sample(self, timestamp: float, ttft_ms: float) -> None:
        self._samples.append((timestamp, ttft_ms))

    def _prune(self, current_time: float) -> None:
        cutoff = current_time - self.window_sec
        self._samples = deque(
            (timestamp, ttft_ms)
            for timestamp, ttft_ms in self._samples
            if timestamp >= cutoff
        )

    def values(self, current_time: float) -> list[float]:
        self._prune(current_time)
        return [
            ttft_ms
            for timestamp, ttft_ms in self._samples
            if timestamp <= current_time
        ]

    def sample_count(self, current_time: float) -> int:
        return len(self.values(current_time))

    def mean_ttft(self, current_time: float) -> float | None:
        values = self.values(current_time)
        if len(values) < BODY_MEAN_MIN_SAMPLES:
            return None
        return float(np.mean(values))

    def median_ttft(self, current_time: float) -> float | None:
        values = self.values(current_time)
        if not values:
            return None
        return float(np.median(values))

    def slo_attainment(self, current_time: float, slo_ms: float) -> float:
        values = self.values(current_time)
        if not values:
            return 0.0
        return float(np.mean(np.array(values) <= slo_ms))


@dataclass
class RecentViolationTracker:
    """Tracks recent final SLO outcomes for adaptive backup selection."""

    maxlen: int = BACKUP_VIOLATION_WINDOW
    _violations: deque[bool] = field(default_factory=deque)

    def add_outcome(self, violated: bool) -> None:
        self._violations.append(bool(violated))
        while len(self._violations) > self.maxlen:
            self._violations.popleft()

    def violation_rate(self) -> float:
        if not self._violations:
            return 0.0
        return float(np.mean(np.array(self._violations, dtype=float)))


@dataclass
class RunDiagnostics:
    """Aggregated diagnostics for one run of one variant on one seed."""

    variant: str
    scenario_name: str
    seed: int
    solver_status_counts: Counter[str] = field(default_factory=Counter)
    fallback_counts: Counter[str] = field(default_factory=Counter)
    budget_values: list[float] = field(default_factory=list)
    expected_c_eff_values: list[float] = field(default_factory=list)
    utilization_values: list[float] = field(default_factory=list)
    slack_values: list[float] = field(default_factory=list)
    total_decisions: int = 0
    non_optimal_decisions: int = 0
    single_feasible_provider_decisions: int = 0
    trivial_single_provider_outcomes: int = 0
    true_p50_fallback_count: int = 0
    explorer_feedback_count: int = 0
    backup_selection_counts: Counter[str] = field(default_factory=Counter)
    backup_random_prob_values: list[float] = field(default_factory=list)

    def record(self, outcome: SelectorOutcome) -> None:
        self.total_decisions += 1
        self.solver_status_counts[outcome.status] += 1
        if outcome.feasible_count == 1:
            self.single_feasible_provider_decisions += 1
        if len(outcome.weights) <= TRIVIAL_SUPPORT_THRESHOLD:
            self.trivial_single_provider_outcomes += 1
        if outcome.status != "optimal":
            self.non_optimal_decisions += 1
        if "fallback" in outcome.status or outcome.status.startswith("best_effort"):
            self.fallback_counts[outcome.status] += 1
        self.true_p50_fallback_count += outcome.true_p50_fallbacks

        if outcome.budget_tau is not None:
            self.budget_values.append(outcome.budget_tau)
        if outcome.expected_c_eff is not None:
            self.expected_c_eff_values.append(outcome.expected_c_eff)
        if outcome.budget_utilization is not None and math.isfinite(outcome.budget_utilization):
            self.utilization_values.append(outcome.budget_utilization)
        if outcome.budget_slack is not None and math.isfinite(outcome.budget_slack):
            self.slack_values.append(outcome.budget_slack)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "solver_status_counts": dict(sorted(self.solver_status_counts.items())),
            "fallback_counts": dict(sorted(self.fallback_counts.items())),
            "non_optimal_decisions": self.non_optimal_decisions,
            "single_feasible_provider_decisions": self.single_feasible_provider_decisions,
            "trivial_single_provider_outcomes": self.trivial_single_provider_outcomes,
            "true_p50_fallback_count": self.true_p50_fallback_count,
            "explorer_feedback_count": self.explorer_feedback_count,
            "backup_selection_counts": dict(sorted(self.backup_selection_counts.items())),
            "mean_backup_random_prob": _safe_mean(self.backup_random_prob_values),
            "mean_B_tau": _safe_mean(self.budget_values),
            "mean_E_pi_c_eff": _safe_mean(self.expected_c_eff_values),
            "mean_budget_utilization": _safe_mean(self.utilization_values),
            "mean_budget_slack": _safe_mean(self.slack_values),
            "budget_utilization_p10": _safe_quantile(self.utilization_values, 10),
            "budget_utilization_p50": _safe_quantile(self.utilization_values, 50),
            "budget_utilization_p90": _safe_quantile(self.utilization_values, 90),
            "budget_slack_p10": _safe_quantile(self.slack_values, 10),
            "budget_slack_p50": _safe_quantile(self.slack_values, 50),
            "budget_slack_p90": _safe_quantile(self.slack_values, 90),
            "total_decisions": self.total_decisions,
        }


@dataclass
class EvaluatedRun:
    """One seed of one variant on one scenario."""

    run: StrategyRun
    diagnostics: RunDiagnostics


def _make_profiles(providers: list[TieredProvider]) -> dict[str, RollingLatencyProfile]:
    return {provider.name: RollingLatencyProfile() for provider in providers}


def _warm_up_profiles(
    profiles: dict[str, RollingLatencyProfile],
    providers: list[TieredProvider],
    t0: float,
    rng: np.random.Generator,
) -> None:
    for provider in providers:
        for idx in range(WARMUP_SAMPLES_PER_PROVIDER):
            t_sample = t0 - WARMUP_WINDOW_SEC + (
                idx / WARMUP_SAMPLES_PER_PROVIDER
            ) * WARMUP_WINDOW_SEC
            profiles[provider.name].add_sample(
                t_sample,
                provider.sample_ttft(rng, t_sample),
            )


def _adaptive_random_backup_probability(
    violation_tracker: RecentViolationTracker,
) -> float:
    """Return the current random-backup fraction for explorer hedging."""
    violation_rate = violation_tracker.violation_rate()
    if violation_rate <= BACKUP_RANDOM_PROB_TARGET_SLO_VIOLATION:
        return BACKUP_RANDOM_PROB_BASE
    if violation_rate >= BACKUP_RANDOM_PROB_ZERO_POINT:
        return 0.0
    excess = (
        (violation_rate - BACKUP_RANDOM_PROB_TARGET_SLO_VIOLATION)
        / (BACKUP_RANDOM_PROB_ZERO_POINT - BACKUP_RANDOM_PROB_TARGET_SLO_VIOLATION)
    )
    return float(BACKUP_RANDOM_PROB_BASE * (1.0 - excess))


def _fallback_api_provider(providers: list[TieredProvider]) -> TieredProvider:
    api_candidates = [provider for provider in providers if provider.tier == ProviderTier.S_A]
    if api_candidates:
        return min(api_candidates, key=lambda provider: provider.cost_per_token)
    return min(providers, key=lambda provider: provider.cost_per_token)


def _predicted_api_cost_anchor(
    providers: list[TieredProvider],
    req: Request,
    now: float,
    slo_ms: float,
) -> float:
    """Return the request-side API cost anchor used as v_hat_i.

    v_hat_i is the billed cost of routing request i to the cheapest SLO-safe
    S_A provider at time `now`. A provider is SLO-safe if its analytical
    P99 TTFT at the current time is within the request's SLO. Restricting
    to SLO-safe providers prevents the budget from being anchored on a
    pathologically slow but cheap S_A (the "trap" case), which would
    starve the LP of enough budget to ever pick a fast provider.

    Fallback order when no SLO-safe S_A exists:
        1. cheapest S_A overall (preserves previous behaviour),
        2. cheapest positive-cost provider,
        3. 0.
    """
    api_candidates = [
        provider for provider in providers if provider.tier == ProviderTier.S_A
    ]
    slo_safe_candidates = [
        provider
        for provider in api_candidates
        if provider.true_p99_ms(now) <= slo_ms
    ]
    if slo_safe_candidates:
        return float(
            min(
                provider.marginal_cost(req.total_tokens, now)
                for provider in slo_safe_candidates
            )
        )
    if api_candidates:
        return float(
            min(provider.marginal_cost(req.total_tokens, now) for provider in api_candidates)
        )

    positive_costs = [
        provider.cost_per_request(req.total_tokens)
        for provider in providers
        if provider.cost_per_request(req.total_tokens) > 0.0
    ]
    if positive_costs:
        return float(min(positive_costs))
    return 0.0


def _body_latency_proxy_ms(
    provider: TieredProvider,
    profile: RollingLatencyProfile,
    now: float,
) -> tuple[float, int]:
    """Return the LP's ``T̄_j(t)`` for one provider.

    The simulator knows each provider's true latency distribution
    (LogNormal / Normal / Uniform parameters set at scenario
    construction), so we use the **analytical** expected TTFT directly
    instead of the rolling-profile sample mean. This decouples the LP's
    decisions from finite-sample estimator noise, which would otherwise
    contaminate experiment intent — most starkly in Stage 0 where the
    grid is constructed with **identical true latency distributions**
    across all providers (so the LP should see ``T̄`` truly tied, not
    differing by ±10ms because of sampling variance from a few hundred
    observations).

    Production deployments cannot use this path (no access to
    ground-truth distribution parameters), but the production algorithm
    still solves ``min Σ π × T̄`` against a profile-based estimator.
    The simulator's job is to expose the **mathematical** behaviour of
    the LP under controlled conditions; it is the production layer's
    responsibility to handle finite-sample noise. See DESIGN_PRINCIPLES.md
    §1: "reasoning beats realism".

    Note: the distribution is read via ``provider._active_ttft_dist(now)``
    rather than ``provider.ttft_dist`` directly, so a configured
    ``shift_time`` / ``ttft_dist_after`` (provider drift) takes effect
    in the LP body. For stationary providers this is a no-op — the
    accessor returns ``ttft_dist`` — so the eval-grid paper line is
    numerically unchanged. ``dist.mean()`` is the analytical expected
    TTFT (e.g. ``exp(mu + sigma^2/2)`` for LogNormal), matching the
    algorithm spec ``T̄_j(t) = E[T_j]``.

    Fallbacks:
    - If the provider exposes the legacy ``_active_dist`` accessor
      (``ShiftingProvider``), it is honoured.
    - If neither accessor exists, fall back to ``provider.ttft_dist``
      (stationary path).
    - If the resulting distribution does not expose ``mean()`` (very
      legacy distributions), fall back to ``true_p50_ms``.

    The ``profile`` argument is no longer consulted on the main path
    but is kept in the signature so callers can stay polymorphic.
    """
    del profile  # ground-truth path does not need sampled history
    if hasattr(provider, "_active_ttft_dist"):
        dist = provider._active_ttft_dist(now)
    elif hasattr(provider, "_active_dist"):
        dist = provider._active_dist(now)
    else:
        dist = provider.ttft_dist
    if hasattr(dist, "mean"):
        return float(dist.mean()), 0
    return provider.true_p50_ms(now), 1


def _expected_weighted_value(
    weights: dict[str, float],
    values: dict[str, float],
) -> float:
    return float(sum(weights[name] * values[name] for name in weights))


def _solve_simplex_lp(
    objective: list[float],
    *,
    upper_constraint: list[float] | None = None,
    upper_bound: float | None = None,
) -> tuple[bool, np.ndarray | None]:
    n = len(objective)
    A_ub = None
    b_ub = None
    if upper_constraint is not None and upper_bound is not None:
        A_ub = [upper_constraint]
        b_ub = [upper_bound]
    A_eq = [np.ones(n)]
    b_eq = [1.0]
    bounds = [(0.0, 1.0) for _ in range(n)]
    result = linprog(
        c=objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return False, None
    return True, result.x


def _select_original_lp(
    providers: list[TieredProvider],
    req: Request,
    now: float,
    slo_ms: float,
    U: float,
    L: float,
    profiles: dict[str, RollingLatencyProfile],
) -> SelectorOutcome:
    """LP-CDF selector (the "old LP-Mix family", generalised to tiered).

    This selector is the SLO-CDF-anchored LP that originally appeared in
    the Phase-3 latency-only experiments as ``LP-Mix``. The original
    Phase-3 form used raw API pricing (single tier, no shadow prices) and
    target_cdf >= 0.99 with relaxation fallback. Here we **generalise it
    to the tiered setting** by replacing ``c_j = cost_per_token × tokens``
    with ``c_eff = effective_cost(provider, ...)`` so the same LP can run
    against ``S_A``, ``S_Q``, and ``S_C`` providers.

    This means ``original_lp`` is **not** a 1:1 replay of Phase-3 LP-Mix:
    it shares the SLO-CDF objective shape but already carries the ψ(z) /
    λ(u) shadow prices in its cost vector. To get the strict shadow-price
    ablation (Phase-3-equivalent on Stage 3), use
    :func:`_select_original_lp_rawcost`, which uses ``marginal_cost``
    (zero for S_Q / S_C) and isolates the contribution of shadow pricing.

    Naming convention used in metadata and figures:

    - ``original_lp``         — LP-CDF / old LP-Mix family, tiered
                                 effective-cost generalisation
    - ``original_lp_rawcost`` — same LP-CDF objective, but with raw
                                 marginal cost (no shadow prices)
    - ``budget_range_p*``     — new LP-TTFT-budget family (latency in
                                 objective, cost in constraint)
    """
    feasible = [provider for provider in providers if provider.is_available(now)]
    if not feasible:
        fallback = _fallback_api_provider(providers)
        c_eff = {
            fallback.name: effective_cost(fallback, req.total_tokens, now, U=U, L=L)
        }
        return SelectorOutcome(
            weights=_single_provider_weight(fallback),
            status="no_capacity_fallback",
            c_eff=c_eff,
            budget_tau=None,
            expected_c_eff=c_eff[fallback.name],
            budget_utilization=None,
            budget_slack=None,
            true_p50_fallbacks=0,
            feasible_count=0,
        )

    names = [provider.name for provider in feasible]
    c_eff = {
        provider.name: effective_cost(provider, req.total_tokens, now, U=U, L=L)
        for provider in feasible
    }
    slo_attainment = {
        provider.name: profiles[provider.name].slo_attainment(now, slo_ms)
        for provider in feasible
    }

    last_status = "solver_failure_fallback"
    for target in OLD_TARGETS:
        success, vector = _solve_simplex_lp(
            objective=[c_eff[name] for name in names],
            upper_constraint=[-slo_attainment[name] for name in names],
            upper_bound=-target,
        )
        if success and vector is not None:
            weights = _normalize_weights(names, vector)
            if weights:
                status = "optimal" if math.isclose(target, 0.99) else f"relaxed_{target:.2f}"
                return SelectorOutcome(
                    weights=weights,
                    status=status,
                    c_eff=c_eff,
                    budget_tau=None,
                    expected_c_eff=_expected_weighted_value(weights, c_eff),
                    budget_utilization=None,
                    budget_slack=None,
                    true_p50_fallbacks=0,
                    feasible_count=len(feasible),
                )
            last_status = "degenerate_feasible_fallback"

    best_provider = min(
        feasible,
        key=lambda provider: (-slo_attainment[provider.name], c_eff[provider.name]),
    )
    return SelectorOutcome(
        weights=_single_provider_weight(best_provider),
        status="best_effort" if last_status != "degenerate_feasible_fallback" else last_status,
        c_eff=c_eff,
        budget_tau=None,
        expected_c_eff=c_eff[best_provider.name],
        budget_utilization=None,
        budget_slack=None,
        true_p50_fallbacks=0,
        feasible_count=len(feasible),
    )


def _select_original_lp_rawcost(
    providers: list[TieredProvider],
    req: Request,
    now: float,
    slo_ms: float,
    U: float,
    L: float,
    profiles: dict[str, RollingLatencyProfile],
) -> SelectorOutcome:
    """Shadow-price ablation: same as :func:`_select_original_lp` but the
    LP cost vector uses :meth:`TieredProvider.marginal_cost` instead of
    :func:`effective_cost`.

    For ``S_A`` providers ``marginal_cost == effective_cost``, so on Stage 2
    (S_A only) this should be indistinguishable from ``original_lp``. On
    Stage 3 (joint provider) ``marginal_cost = 0`` for ``S_Q`` and ``S_C``,
    so the LP no longer sees any cost penalty for greedily exhausting
    subscription capacity. The expected behaviour is "S_Q / S_C are picked
    first until exhaustion, then sudden overflow to S_A" — exactly the
    failure mode that motivates ψ(z) and λ(u).

    Used to isolate the contribution of shadow pricing in the cost vector
    while keeping every other piece (profile, SLO targets, LP solver,
    capacity bookkeeping, hedger) identical to ``original_lp``.
    """
    feasible = [provider for provider in providers if provider.is_available(now)]
    if not feasible:
        fallback = _fallback_api_provider(providers)
        c_eff = {
            fallback.name: float(fallback.marginal_cost(req.total_tokens, now))
        }
        return SelectorOutcome(
            weights=_single_provider_weight(fallback),
            status="no_capacity_fallback",
            c_eff=c_eff,
            budget_tau=None,
            expected_c_eff=c_eff[fallback.name],
            budget_utilization=None,
            budget_slack=None,
            true_p50_fallbacks=0,
            feasible_count=0,
        )

    names = [provider.name for provider in feasible]
    # Raw marginal cost: API price for S_A, zero for S_Q / S_C (subscription
    # is sunk cost). This is the deliberate ablation point.
    c_eff = {
        provider.name: float(provider.marginal_cost(req.total_tokens, now))
        for provider in feasible
    }
    slo_attainment = {
        provider.name: profiles[provider.name].slo_attainment(now, slo_ms)
        for provider in feasible
    }

    last_status = "solver_failure_fallback"
    for target in OLD_TARGETS:
        success, vector = _solve_simplex_lp(
            objective=[c_eff[name] for name in names],
            upper_constraint=[-slo_attainment[name] for name in names],
            upper_bound=-target,
        )
        if success and vector is not None:
            weights = _normalize_weights(names, vector)
            if weights:
                status = "optimal" if math.isclose(target, 0.99) else f"relaxed_{target:.2f}"
                return SelectorOutcome(
                    weights=weights,
                    status=status,
                    c_eff=c_eff,
                    budget_tau=None,
                    expected_c_eff=_expected_weighted_value(weights, c_eff),
                    budget_utilization=None,
                    budget_slack=None,
                    true_p50_fallbacks=0,
                    feasible_count=len(feasible),
                )
            last_status = "degenerate_feasible_fallback"

    best_provider = min(
        feasible,
        key=lambda provider: (-slo_attainment[provider.name], c_eff[provider.name]),
    )
    return SelectorOutcome(
        weights=_single_provider_weight(best_provider),
        status="best_effort" if last_status != "degenerate_feasible_fallback" else last_status,
        c_eff=c_eff,
        budget_tau=None,
        expected_c_eff=c_eff[best_provider.name],
        budget_utilization=None,
        budget_slack=None,
        true_p50_fallbacks=0,
        feasible_count=len(feasible),
    )


def _select_budget_body(
    providers: list[TieredProvider],
    req: Request,
    now: float,
    slo_ms: float,
    U: float,
    L: float,
    profiles: dict[str, RollingLatencyProfile],
    variant: str,
) -> SelectorOutcome:
    feasible = [provider for provider in providers if provider.is_available(now)]
    if not feasible:
        fallback = _fallback_api_provider(providers)
        c_eff = {
            fallback.name: effective_cost(fallback, req.total_tokens, now, U=U, L=L)
        }
        return SelectorOutcome(
            weights=_single_provider_weight(fallback),
            status="no_capacity_fallback",
            c_eff=c_eff,
            budget_tau=None,
            expected_c_eff=c_eff[fallback.name],
            budget_utilization=None,
            budget_slack=None,
            true_p50_fallbacks=0,
            feasible_count=0,
        )

    names = [provider.name for provider in feasible]
    c_eff = {
        provider.name: effective_cost(provider, req.total_tokens, now, U=U, L=L)
        for provider in feasible
    }

    tbar: dict[str, float] = {}
    true_p50_fallbacks = 0
    for provider in feasible:
        proxy_ms, fallback_hits = _body_latency_proxy_ms(
            provider,
            profiles[provider.name],
            now,
        )
        tbar[provider.name] = proxy_ms
        true_p50_fallbacks += fallback_hits

    budget_level = _budget_level_from_variant(variant)
    if _is_provider_percentile_variant(variant):
        budget_tau = float(np.percentile(list(c_eff.values()), budget_level))
    elif _is_range_budget_variant(variant):
        p = budget_level / 100.0
        c_min = min(c_eff.values())
        c_max = max(c_eff.values())
        budget_tau = float(c_min + p * (c_max - c_min))
    elif _is_vhat_budget_variant(variant):
        tau = budget_level / 100.0
        budget_tau = tau * _predicted_api_cost_anchor(providers, req, now, slo_ms)
    else:
        raise ValueError(f"Unknown budgeted variant family: {variant!r}")
    success, vector = _solve_simplex_lp(
        objective=[tbar[name] for name in names],
        upper_constraint=[c_eff[name] for name in names],
        upper_bound=budget_tau,
    )
    if success and vector is not None:
        weights = _normalize_weights(names, vector)
        if weights:
            expected_c_eff = _expected_weighted_value(weights, c_eff)
            budget_slack = float(budget_tau - expected_c_eff)
            utilization = float(expected_c_eff / budget_tau) if budget_tau > LP_EPS else None
            return SelectorOutcome(
                weights=weights,
                status="optimal",
                c_eff=c_eff,
                budget_tau=budget_tau,
                expected_c_eff=expected_c_eff,
                budget_utilization=utilization,
                budget_slack=budget_slack,
                true_p50_fallbacks=true_p50_fallbacks,
                feasible_count=len(feasible),
            )

        in_budget = [
            provider for provider in feasible
            if c_eff[provider.name] <= budget_tau + LP_EPS
        ]
        if not in_budget:
            cheapest = min(feasible, key=lambda provider: c_eff[provider.name])
            return SelectorOutcome(
                weights=_single_provider_weight(cheapest),
                status="no_in_budget_fallback",
                c_eff=c_eff,
                budget_tau=budget_tau,
                expected_c_eff=c_eff[cheapest.name],
                budget_utilization=(
                    float(c_eff[cheapest.name] / budget_tau)
                    if budget_tau > LP_EPS else None
                ),
                budget_slack=float(budget_tau - c_eff[cheapest.name]),
                true_p50_fallbacks=true_p50_fallbacks,
                feasible_count=len(feasible),
            )

        best = min(
            in_budget,
            key=lambda provider: (tbar[provider.name], c_eff[provider.name]),
        )
        return SelectorOutcome(
            weights=_single_provider_weight(best),
            status="degenerate_feasible_fallback",
            c_eff=c_eff,
            budget_tau=budget_tau,
            expected_c_eff=c_eff[best.name],
            budget_utilization=(
                float(c_eff[best.name] / budget_tau)
                if budget_tau > LP_EPS else None
            ),
            budget_slack=float(budget_tau - c_eff[best.name]),
            true_p50_fallbacks=true_p50_fallbacks,
            feasible_count=len(feasible),
        )

    in_budget = [
        provider for provider in feasible
        if c_eff[provider.name] <= budget_tau + LP_EPS
    ]
    if in_budget:
        best = min(
            in_budget,
            key=lambda provider: (tbar[provider.name], c_eff[provider.name]),
        )
        return SelectorOutcome(
            weights=_single_provider_weight(best),
            status="solver_failure_fallback",
            c_eff=c_eff,
            budget_tau=budget_tau,
            expected_c_eff=c_eff[best.name],
            budget_utilization=(
                float(c_eff[best.name] / budget_tau)
                if budget_tau > LP_EPS else None
            ),
            budget_slack=float(budget_tau - c_eff[best.name]),
            true_p50_fallbacks=true_p50_fallbacks,
            feasible_count=len(feasible),
        )

    cheapest = min(feasible, key=lambda provider: c_eff[provider.name])
    return SelectorOutcome(
        weights=_single_provider_weight(cheapest),
        status="no_in_budget_fallback",
        c_eff=c_eff,
        budget_tau=budget_tau,
        expected_c_eff=c_eff[cheapest.name],
        budget_utilization=(
            float(c_eff[cheapest.name] / budget_tau) if budget_tau > LP_EPS else None
        ),
        budget_slack=float(budget_tau - c_eff[cheapest.name]),
        true_p50_fallbacks=true_p50_fallbacks,
        feasible_count=len(feasible),
    )


def _select_control(
    providers: list[TieredProvider],
    req: Request,
    now: float,
    U: float,
    L: float,
    variant: str,
) -> SelectorOutcome:
    feasible = [provider for provider in providers if provider.is_available(now)]
    if not feasible:
        fallback = _fallback_api_provider(providers)
        c_eff = {
            fallback.name: effective_cost(fallback, req.total_tokens, now, U=U, L=L)
        }
        return SelectorOutcome(
            weights=_single_provider_weight(fallback),
            status="no_capacity_fallback",
            c_eff=c_eff,
            budget_tau=None,
            expected_c_eff=c_eff[fallback.name],
            budget_utilization=None,
            budget_slack=None,
            true_p50_fallbacks=0,
            feasible_count=0,
        )

    c_eff = {
        provider.name: effective_cost(provider, req.total_tokens, now, U=U, L=L)
        for provider in feasible
    }
    if variant == "cheapest_available":
        chosen = min(
            feasible,
            key=lambda provider: (
                provider.marginal_cost(req.total_tokens, now),
                c_eff[provider.name],
                provider.true_p50_ms(now),
            ),
        )
        status = "control_cheapest_available"
    elif variant == "fastest_available":
        chosen = min(
            feasible,
            key=lambda provider: (
                provider.true_p50_ms(now),
                c_eff[provider.name],
            ),
        )
        status = "control_fastest_available"
    elif variant in ("quota_first", "concurrency_first"):
        if variant == "quota_first":
            tier_priority = (
                ProviderTier.S_Q,
                ProviderTier.S_C,
                ProviderTier.S_A,
            )
            status = "control_quota_first"
        else:
            tier_priority = (
                ProviderTier.S_C,
                ProviderTier.S_Q,
                ProviderTier.S_A,
            )
            status = "control_concurrency_first"

        tier_candidates: list[TieredProvider] = []
        for tier in tier_priority:
            tier_candidates = [p for p in feasible if p.tier == tier]
            if tier_candidates:
                break
        if not tier_candidates:
            tier_candidates = feasible

        chosen = min(
            tier_candidates,
            key=lambda provider: (
                provider.true_p50_ms(now),
                c_eff[provider.name],
                provider.marginal_cost(req.total_tokens, now),
            ),
        )
    else:
        raise ValueError(f"Unknown control variant: {variant!r}")

    return SelectorOutcome(
        weights=_single_provider_weight(chosen),
        status=status,
        c_eff=c_eff,
        budget_tau=None,
        expected_c_eff=c_eff[chosen.name],
        budget_utilization=None,
        budget_slack=None,
        true_p50_fallbacks=0,
        feasible_count=len(feasible),
    )


def _ttft_cdf_ms(
    provider: TieredProvider,
    now: float,
    threshold_ms: float,
) -> float:
    """Return the provider TTFT CDF at `threshold_ms`."""
    if threshold_ms <= 0.0:
        return 0.0
    return float(provider._active_ttft_dist(now).cdf(threshold_ms))


def _available_backup_candidates(
    providers: list[TieredProvider],
    primary: TieredProvider,
    now: float,
    *,
    backup_scope: str,
) -> list[TieredProvider]:
    """Return currently available backup candidates under the requested scope."""
    if backup_scope not in BACKUP_SCOPES:
        raise ValueError(
            f"Unknown backup_scope {backup_scope!r}. Expected one of {BACKUP_SCOPES!r}."
        )

    if backup_scope == "cross_tier":
        return [
            provider
            for provider in providers
            if provider.name != primary.name
            and provider.tier != primary.tier
            and provider.is_available(now)
        ]

    return [
        provider
        for provider in providers
        if provider.name != primary.name and provider.is_available(now)
    ]


def _pick_probability_target_backup(
    providers: list[TieredProvider],
    primary: TieredProvider,
    req: Request,
    *,
    now: float,
    slo_ms: float,
    rng: np.random.Generator,
    violation_tracker: RecentViolationTracker,
    allow_random_backup: bool,
    backup_scope: str = "any_provider",
    target_success: float = HEDGE_SUCCESS_TARGET,
    dispatch_overhead_ms: float = HEDGE_DISPATCH_OVERHEAD_MS,
) -> tuple[TieredProvider | None, str, float]:
    """Pick the backup candidate for probability-target hedging.

    `backup_scope` separates optional tier restrictions from the hedge formula
    itself. The canonical probability-target hedger considers any available
    non-primary provider. `cross_tier` is an explicit ablation knob for testing
    the older tier-separation constraint. Random backup exploration is also an
    explicit ablation knob and is not part of hedge-only or hedge-as-probe
    semantics.
    """
    others = _available_backup_candidates(
        providers,
        primary,
        now,
        backup_scope=backup_scope,
    )
    if not others:
        return None, "no_backup", 0.0

    random_prob = (
        _adaptive_random_backup_probability(violation_tracker)
        if allow_random_backup else 0.0
    )
    if allow_random_backup and rng.random() < random_prob:
        index = int(rng.integers(0, len(others)))
        return others[index], "random_backup", random_prob

    remaining_budget_ms = slo_ms - dispatch_overhead_ms
    safe = [
        provider for provider in others
        if _ttft_cdf_ms(provider, now, remaining_budget_ms) >= target_success - LP_EPS
    ]
    if safe:
        backup = min(
            safe,
            key=lambda provider: (
                provider.marginal_cost(req.total_tokens, now),
                provider.true_p50_ms(now),
                provider.cost_per_token,
            ),
        )
        return backup, "safe_cheapest", random_prob

    fallback = min(others, key=lambda provider: provider.true_p50_ms(now))
    return fallback, "fallback_fastest", random_prob


def _combined_success_probability_after_hedge(
    primary: TieredProvider,
    backup: TieredProvider,
    *,
    now: float,
    slo_ms: float,
    hedge_time_ms: float,
    dispatch_overhead_ms: float = HEDGE_DISPATCH_OVERHEAD_MS,
) -> float:
    """Return success probability if backup is dispatched at `hedge_time_ms`.

    The calculation matches the meeting formula:

      P(not violate | t) + P(violate | t) * P(backup succeeds)

    where conditioning is on the primary still being unfinished at time `t`.
    """
    if hedge_time_ms < 0.0:
        return 0.0

    primary_cdf_t = _ttft_cdf_ms(primary, now, hedge_time_ms)
    primary_cdf_slo = _ttft_cdf_ms(primary, now, slo_ms)
    primary_survival_t = max(1.0 - primary_cdf_t, 0.0)
    if primary_survival_t <= LP_EPS:
        return 0.0

    p_not_violate = max(primary_cdf_slo - primary_cdf_t, 0.0) / primary_survival_t
    p_violate = max(1.0 - primary_cdf_slo, 0.0) / primary_survival_t

    remaining_budget_ms = slo_ms - hedge_time_ms - dispatch_overhead_ms
    if remaining_budget_ms <= 0.0:
        backup_success = 0.0
    else:
        backup_now = now + hedge_time_ms / 1000.0
        backup_success = _ttft_cdf_ms(backup, backup_now, remaining_budget_ms)

    combined_success = p_not_violate + p_violate * backup_success
    return float(min(max(combined_success, 0.0), 1.0))


def _find_latest_safe_hedge_time_ms(
    primary: TieredProvider,
    backup: TieredProvider,
    *,
    now: float,
    slo_ms: float,
    target_success: float = HEDGE_SUCCESS_TARGET,
    dispatch_overhead_ms: float = HEDGE_DISPATCH_OVERHEAD_MS,
    grid_ms: float = HEDGE_TIME_GRID_MS,
) -> float | None:
    """Return the latest dispatch time whose combined success meets target.

    We interpret the new hedge rule as a *latest safe wait time* search rather
    than an earliest-trigger search. Otherwise the rule often collapses to
    immediate replication at `t = 0`.
    """
    max_dispatch_ms = slo_ms - dispatch_overhead_ms
    if max_dispatch_ms < 0.0:
        return None

    latest_safe_ms: float | None = None
    hedge_time_ms = 0.0
    while hedge_time_ms <= max_dispatch_ms + LP_EPS:
        combined_success = _combined_success_probability_after_hedge(
            primary,
            backup,
            now=now,
            slo_ms=slo_ms,
            hedge_time_ms=min(hedge_time_ms, max_dispatch_ms),
            dispatch_overhead_ms=dispatch_overhead_ms,
        )
        if combined_success >= target_success - LP_EPS:
            latest_safe_ms = min(hedge_time_ms, max_dispatch_ms)
        hedge_time_ms += grid_ms

    if latest_safe_ms is not None:
        return latest_safe_ms

    immediate_success = _combined_success_probability_after_hedge(
        primary,
        backup,
        now=now,
        slo_ms=slo_ms,
        hedge_time_ms=0.0,
        dispatch_overhead_ms=dispatch_overhead_ms,
    )
    primary_only_success = _ttft_cdf_ms(primary, now, slo_ms)
    if immediate_success > primary_only_success + LP_EPS:
        return 0.0
    return None


def _apply_existing_hedge(
    scenario: ScenarioConfig,
    providers: list[TieredProvider],
    primary: TieredProvider,
    req: Request,
    now: float,
    rng: np.random.Generator,
    c_eff: dict[str, float],
    U: float,
    L: float,
) -> tuple[float, float, bool, list[ObservedSample], float]:
    """Apply the current hedge rule unchanged from the tiered runner."""
    del c_eff  # oldhedge recomputes backup cost at its actual dispatch time
    primary_ttft_ms = _sample_ttft(primary, rng, now)
    # Observation time is when the response arrives, not when we
    # dispatched. The rolling profile must not see the sample until
    # ``now + ttft``; dispatching at ``now`` means the latency value
    # is only observable later. The backup observation time below is
    # already computed correctly as ``dispatch_time + ttft``.
    primary_observation_time = now + primary_ttft_ms / 1000.0
    observed_samples = [(primary.name, primary_observation_time, primary_ttft_ms)]
    final_ttft_ms = primary_ttft_ms
    billed_cost = primary.marginal_cost(req.total_tokens, now)
    hedged = False

    primary.account_request(
        req.id,
        now,
        _sample_service_time(primary, rng, now, req.response_tokens, primary_ttft_ms),
    )

    primary_p50 = primary.true_p50_ms(now)
    wait_threshold_ms = max(primary_p50 * 1.5, scenario.primary_slo_ms * 0.5)
    if primary_ttft_ms <= wait_threshold_ms:
        return final_ttft_ms, billed_cost, hedged, observed_samples, primary_ttft_ms

    dispatch_time_sec = now + wait_threshold_ms / 1000.0
    backup = _pick_cross_tier_backup(providers, primary, dispatch_time_sec)
    if backup is None:
        return final_ttft_ms, billed_cost, hedged, observed_samples, primary_ttft_ms

    p_viol = 0.5
    remaining_ms = scenario.primary_slo_ms - wait_threshold_ms
    f_backup = 1.0 if backup.true_p50_ms(dispatch_time_sec) < remaining_ms else 0.3
    backup_c_eff = effective_cost(backup, req.total_tokens, dispatch_time_sec, U=U, L=L)
    v_penalty = U * HEDGE_COST_MULTIPLIER
    if p_viol * f_backup <= backup_c_eff / v_penalty:
        return final_ttft_ms, billed_cost, hedged, observed_samples, primary_ttft_ms

    backup_ttft_ms = _sample_ttft(backup, rng, dispatch_time_sec)
    backup_service_time = _sample_service_time(
        backup,
        rng,
        dispatch_time_sec,
        req.response_tokens,
        backup_ttft_ms,
    )
    if (
        backup.tier == ProviderTier.S_C
        and backup.concurrency is not None
        and not backup.concurrency.can_admit_interval(
            dispatch_time_sec,
            dispatch_time_sec + backup_service_time,
        )
    ):
        return final_ttft_ms, billed_cost, hedged, observed_samples, primary_ttft_ms

    backup.account_request(
        req.id + 10_000_000,
        dispatch_time_sec,
        backup_service_time,
    )
    backup_observation_time = dispatch_time_sec + backup_ttft_ms / 1000.0
    observed_samples.append((backup.name, backup_observation_time, backup_ttft_ms))
    billed_cost += backup.marginal_cost(req.total_tokens, dispatch_time_sec)
    final_ttft_ms = min(primary_ttft_ms, wait_threshold_ms + backup_ttft_ms)
    hedged = True
    return final_ttft_ms, billed_cost, hedged, observed_samples, primary_ttft_ms


def _apply_probability_target_hedge(
    scenario: ScenarioConfig,
    providers: list[TieredProvider],
    primary: TieredProvider,
    req: Request,
    now: float,
    rng: np.random.Generator,
    violation_tracker: RecentViolationTracker,
    allow_random_backup: bool,
    backup_scope: str,
) -> tuple[float, float, bool, list[ObservedSample], float, str, float, str]:
    """Apply the new probability-only hedge trigger.

    A backup launched at time `t` is considered feasible if:

      P(not violate | t) + P(violate | t) * P(backup succeeds) >= 0.99

    The policy waits until the latest `t` that still satisfies this condition,
    then dispatches the backup if the primary has not completed by that time.
    """
    primary_ttft_ms = _sample_ttft(primary, rng, now)
    primary_service_time = _sample_service_time(primary, rng, now, req.response_tokens, primary_ttft_ms)
    if (
        primary.tier == ProviderTier.S_C
        and primary.concurrency is not None
        and not primary.concurrency.can_admit_interval(
            now,
            now + primary_service_time,
        )
    ):
        primary = _fallback_api_provider(providers)
        primary_ttft_ms = _sample_ttft(primary, rng, now)
        primary_service_time = _sample_service_time(primary, rng, now, req.response_tokens, primary_ttft_ms)
    # Observation time = dispatch time + observed TTFT. See the
    # equivalent comment in _apply_economic_hedge.
    primary_observation_time = now + primary_ttft_ms / 1000.0
    observed_samples = [(primary.name, primary_observation_time, primary_ttft_ms)]
    final_ttft_ms = primary_ttft_ms
    billed_cost = primary.marginal_cost(req.total_tokens, now)
    hedged = False

    primary.account_request(
        req.id,
        now,
        primary_service_time,
    )

    backup, backup_selection_mode, random_backup_prob = _pick_probability_target_backup(
        providers,
        primary,
        req,
        now=now,
        slo_ms=scenario.primary_slo_ms,
        rng=rng,
        violation_tracker=violation_tracker,
        allow_random_backup=allow_random_backup,
        backup_scope=backup_scope,
    )
    if backup is None:
        return (
            final_ttft_ms,
            billed_cost,
            hedged,
            observed_samples,
            primary_ttft_ms,
            backup_selection_mode,
            random_backup_prob,
            primary.name,
        )

    hedge_time_ms = _find_latest_safe_hedge_time_ms(
        primary,
        backup,
        now=now,
        slo_ms=scenario.primary_slo_ms,
    )
    if hedge_time_ms is None or primary_ttft_ms <= hedge_time_ms:
        return (
            final_ttft_ms,
            billed_cost,
            hedged,
            observed_samples,
            primary_ttft_ms,
            backup_selection_mode,
            random_backup_prob,
            primary.name,
        )

    dispatch_time_sec = now + hedge_time_ms / 1000.0
    if not backup.is_available(dispatch_time_sec):
        return (
            final_ttft_ms,
            billed_cost,
            hedged,
            observed_samples,
            primary_ttft_ms,
            "dispatch_unavailable",
            random_backup_prob,
            primary.name,
        )
    backup_ttft_ms = _sample_ttft(backup, rng, dispatch_time_sec)
    backup_service_time = _sample_service_time(
        backup,
        rng,
        dispatch_time_sec,
        req.response_tokens,
        backup_ttft_ms,
    )
    if (
        backup.tier == ProviderTier.S_C
        and backup.concurrency is not None
        and not backup.concurrency.can_admit_interval(
            dispatch_time_sec,
            dispatch_time_sec + backup_service_time,
        )
    ):
        return (
            final_ttft_ms,
            billed_cost,
            hedged,
            observed_samples,
            primary_ttft_ms,
            "dispatch_interval_unavailable",
            random_backup_prob,
            primary.name,
        )
    backup.account_request(
        req.id + 10_000_000,
        dispatch_time_sec,
        backup_service_time,
    )
    backup_observation_time = dispatch_time_sec + backup_ttft_ms / 1000.0
    observed_samples.append((backup.name, backup_observation_time, backup_ttft_ms))
    billed_cost += backup.marginal_cost(req.total_tokens, dispatch_time_sec)
    final_ttft_ms = min(
        primary_ttft_ms,
        hedge_time_ms + HEDGE_DISPATCH_OVERHEAD_MS + backup_ttft_ms,
    )
    hedged = True
    return (
        final_ttft_ms,
        billed_cost,
        hedged,
        observed_samples,
        primary_ttft_ms,
        backup_selection_mode,
        random_backup_prob,
        primary.name,
    )


def run_variant(
    scenario: ScenarioConfig,
    requests: list[Request],
    variant: str,
    *,
    seed: int,
    backup_scope: str = "any_provider",
) -> EvaluatedRun:
    """Run one sidecar variant on one scenario for one RNG seed."""
    variant = canonicalize_variant_name(variant)
    if backup_scope not in BACKUP_SCOPES:
        raise ValueError(
            f"Unknown backup_scope {backup_scope!r}. Expected one of {BACKUP_SCOPES!r}."
        )
    if (
        variant not in MAIN_VARIANTS
        and variant not in BACKUP_EXPLORATION_VARIANTS
        and variant not in HEDGE_ABLATION_VARIANTS
        and variant not in PROVIDER_PERCENTILE_ABLATION_VARIANTS
        and variant not in SHADOW_PRICE_ABLATION_VARIANTS
        and variant not in CONTROL_VARIANTS
    ):
        raise ValueError(f"Unknown LP-budget variant: {variant!r}")

    rng = np.random.default_rng(seed)
    _reset_all_state(scenario.providers)
    profiles = _make_profiles(scenario.providers)
    if requests:
        _warm_up_profiles(profiles, scenario.providers, float(requests[0].timestamp), rng)

    sampler = SWRRSampler()
    L, U = calibrate_envelopes(scenario.providers)
    hedge_as_probe = _use_hedge_as_probe(variant)
    random_backup_exploration = _use_random_backup_exploration(variant)
    diagnostics = RunDiagnostics(
        variant=variant,
        scenario_name=scenario.name,
        seed=seed,
    )
    violation_tracker = RecentViolationTracker()

    ttft_ms: list[float] = []
    cost_usd: list[float] = []
    provider_sel: list[str] = []
    tier_sel: list[str] = []
    timestamps: list[float] = []
    hedge_flags: list[bool] = []
    z_series: list[float] = []
    u_series: list[float] = []
    psi_q_series: list[float] = []

    providers_by_name = {provider.name: provider for provider in scenario.providers}

    for req in requests:
        now = float(req.timestamp)

        if _is_control_variant(variant):
            outcome = _select_control(
                scenario.providers,
                req,
                now,
                U,
                L,
                _base_variant(variant),
            )
        elif _base_variant(variant) == "original_lp":
            outcome = _select_original_lp(
                scenario.providers,
                req,
                now,
                scenario.primary_slo_ms,
                U,
                L,
                profiles,
            )
        elif _base_variant(variant) == "original_lp_rawcost":
            outcome = _select_original_lp_rawcost(
                scenario.providers,
                req,
                now,
                scenario.primary_slo_ms,
                U,
                L,
                profiles,
            )
        else:
            outcome = _select_budget_body(
                scenario.providers,
                req,
                now,
                scenario.primary_slo_ms,
                U,
                L,
                profiles,
                _base_variant(variant),
            )

        diagnostics.record(outcome)

        primary_name = _choose_lp_provider(outcome.weights, sampler)
        primary = providers_by_name[primary_name]

        hedge_mode = _hedge_mode(variant)
        if hedge_mode == "new":
            (
                final_ttft_ms,
                billed_cost,
                hedged,
                observed_samples,
                _,
                backup_selection_mode,
                random_backup_prob,
                actual_primary_name,
            ) = _apply_probability_target_hedge(
                scenario,
                scenario.providers,
                primary,
                req,
                now,
                rng,
                violation_tracker,
                random_backup_exploration,
                backup_scope,
            )
            primary = providers_by_name[actual_primary_name]
            diagnostics.backup_selection_counts[backup_selection_mode] += 1
            diagnostics.backup_random_prob_values.append(random_backup_prob)
        elif hedge_mode == "old":
            final_ttft_ms, billed_cost, hedged, observed_samples, _ = _apply_existing_hedge(
                scenario,
                scenario.providers,
                primary,
                req,
                now,
                rng,
                outcome.c_eff,
                U,
                L,
            )
        else:
            primary_ttft_ms = _sample_ttft(primary, rng, now)
            primary.account_request(
                req.id,
                now,
                _sample_service_time(primary, rng, now, req.response_tokens, primary_ttft_ms),
            )
            final_ttft_ms = primary_ttft_ms
            billed_cost = primary.marginal_cost(req.total_tokens, now)
            hedged = False
            # Observation time = dispatch time + observed TTFT (see
            # _apply_economic_hedge for the rationale).
            primary_observation_time = now + primary_ttft_ms / 1000.0
            observed_samples = [
                (primary.name, primary_observation_time, primary_ttft_ms)
            ]

        _, primary_sample_time, primary_ttft_ms = observed_samples[0]
        profiles[primary.name].add_sample(primary_sample_time, primary_ttft_ms)
        if hedge_as_probe and hedged and len(observed_samples) > 1:
            backup_name, backup_sample_time, backup_ttft_ms = observed_samples[1]
            if backup_name != primary.name:
                profiles[backup_name].add_sample(backup_sample_time, backup_ttft_ms)
                diagnostics.explorer_feedback_count += 1
        # Dedicated active probing was deleted on 2026-04-29 — see the
        # PROBE_RATE comment block at the top of this module.

        q_provider = next(
            (provider for provider in scenario.providers if provider.tier == ProviderTier.S_Q),
            None,
        )
        c_provider = next(
            (
                provider
                for provider in scenario.providers
                if provider.tier == ProviderTier.S_C
            ),
            None,
        )

        ttft_ms.append(final_ttft_ms)
        cost_usd.append(billed_cost)
        provider_sel.append(primary.name)
        tier_sel.append(primary.tier.value)
        timestamps.append(now)
        hedge_flags.append(hedged)
        z_series.append(q_provider.quota.fraction_used(now) if q_provider else 0.0)
        u_series.append(
            c_provider.concurrency.utilization(now) if c_provider else 0.0
        )
        psi_q_series.append(
            quota_shadow_price(q_provider, now, U=U, L=L) if q_provider else 0.0
        )
        violation_tracker.add_outcome(final_ttft_ms > scenario.primary_slo_ms)

    run = StrategyRun(
        strategy=variant,
        ttft_ms=np.array(ttft_ms),
        cost_usd=np.array(cost_usd),
        provider=provider_sel,
        tier=tier_sel,
        timestamp=np.array(timestamps),
        hedge_triggered=np.array(hedge_flags, dtype=bool),
        quota_fraction_used=np.array(z_series),
        concurrency_utilization=np.array(u_series),
        shadow_price_q=np.array(psi_q_series),
    )
    return EvaluatedRun(run=run, diagnostics=diagnostics)


def build_first_batch_scenarios() -> dict[str, ScenarioConfig]:
    """Return the mandatory first-batch scenario set for the LP study."""
    scenarios = build_all_scenarios()
    return {name: scenarios[name] for name in FIRST_BATCH_SCENARIOS}


def build_all_scenarios() -> dict[str, ScenarioConfig]:
    """Return every registered synthetic scenario available to the sidecar."""
    scenarios = {name: load_world_scenario(name) for name in list_scenarios()}
    scenarios.update(make_simple_scenarios())
    scenarios.update(make_mm25_scenarios())
    return scenarios


def _resolve_trace_dataset_path(dataset_name: str) -> Path | None:
    """Return the first available canonical path for one trace dataset."""
    for candidate in _TRACE_DATASET_PATHS.get(dataset_name, []):
        if candidate.exists():
            return candidate
    return None


def _dataset_cache_path(dataset_name: str) -> Path:
    return _DATASET_CACHE_ROOT / f"{dataset_name}.pkl"


def _load_sharegpt_jsonl_requests(filepath: Path) -> list[Request]:
    """Load ShareGPT JSONL trace into Request objects."""
    requests: list[Request] = []
    first_timestamp: float | None = None
    with filepath.open() as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            timestamp = float(record["arrived_at"])
            if first_timestamp is None:
                first_timestamp = timestamp
            request_tokens = int(record["num_prefill_tokens"])
            response_tokens = int(record["num_decode_tokens"])
            total_tokens = request_tokens + response_tokens
            if total_tokens <= 0:
                continue
            requests.append(
                Request(
                    id=idx,
                    timestamp=float(timestamp - first_timestamp),
                    request_tokens=request_tokens,
                    response_tokens=response_tokens,
                    total_tokens=total_tokens,
                    model="sharegpt",
                )
            )
    return requests


@lru_cache(maxsize=None)
def _load_trace_dataset_requests(dataset_name: str) -> tuple[Request, ...]:
    """Load one real workload dataset for trace-driven synthetic experiments.

    Delegates all cache logic — loading, manifest verification, and
    auto-building — to :mod:`experiments.simulation.dataset_cache`.
    This ensures the runtime loader never silently accepts a stale pickle
    when the source file has changed.

    Load order:
    1. Pickle cache with valid manifest → ~0.3 s.
    2. Stale/missing cache → auto-rebuild from raw source (writes pickle
       + manifest).
    3. Neither source nor cache → ``FileNotFoundError``.

    Build caches ahead of time with::

        python -m experiments.simulation.dataset_cache build
    """
    if dataset_name not in TRACE_WORKLOAD_DATASETS:
        raise ValueError(
            f"Unknown trace-driven dataset: {dataset_name!r}. "
            f"Expected one of {TRACE_WORKLOAD_DATASETS!r}."
        )

    from experiments.simulation.dataset_cache import (
        CacheStalenessError,
        build_cache,
        load_cached,
        verify_cache,
    )

    # --- fast path: verified pickle cache --------------------------------
    cache_path = _dataset_cache_path(dataset_name)
    if cache_path.exists():
        try:
            verify_cache(dataset_name, quick=True)
            return load_cached(dataset_name)
        except CacheStalenessError:
            # Cache is stale — fall through to rebuild.
            pass

    # --- slow path: build (or rebuild) from raw source -------------------
    source_path = _resolve_trace_dataset_path(dataset_name)
    if source_path is not None:
        build_cache(dataset_name, force=True)
        return load_cached(dataset_name)

    raise FileNotFoundError(
        "Could not locate trace data for dataset "
        f"{dataset_name!r}. Checked {list(_TRACE_DATASET_PATHS.get(dataset_name, []))} "
        f"and cache path {cache_path}."
    )


def _generate_trace_driven_workload(
    scenario: ScenarioConfig,
    *,
    dataset_name: str,
    seed: int,
) -> list[Request]:
    """Build a trace-driven workload by replaying the entire trace at its
    natural arrival rate.

    Uses the **entire** trace (no random slicing) and **does not rescale**
    timestamps — the simulator's "now" advances at the trace's natural
    pace. Wall-clock simulation time is unaffected (Python still processes
    events as fast as it can); what changes is the *simulated* time
    spanning whatever the trace actually spanned (e.g. 7 days for
    sharegpt, 89 days for freeinference, 83 days for rednote). This
    preserves quota-window rolls and concurrency saturation dynamics as
    they would occur in production.

    The ``seed`` argument is currently unused in this mode (no slicing /
    no resampling), but kept in the signature so the caller can stay
    polymorphic with the synthetic generator.

    Historical note: an earlier scaled-replay mode contiguously sliced
    ``scenario.n_requests`` (default 2000) requests from the trace and
    rescaled their timestamps into ``scenario.duration_seconds``
    (default 3600 s). For sharegpt that was a 1.2× compression
    (close to natural). For freeinference (89-day trace) it was 11.5×.
    For rednote (83-day trace) it was 73×. The compression artificially
    inflated arrival rate, fabricated capacity stress that does not
    exist in production, and distorted quota-window semantics.
    The scaled-replay path — along with its ``_select_trace_slice`` /
    ``_normalize_trace_slice`` helpers — was removed on 2026-04-29 so
    nobody can wire it back in by mistake.
    """
    del seed  # unused — full trace, no slicing, no resampling
    full_trace = _load_trace_dataset_requests(dataset_name)
    if not full_trace:
        return []

    # Rebase to start at t=0 but keep the inter-arrival pattern intact.
    base = float(full_trace[0].timestamp)
    rebased: list[Request] = []
    for idx, req in enumerate(full_trace):
        rebased.append(
            Request(
                id=idx,
                timestamp=float(float(req.timestamp) - base),
                request_tokens=req.request_tokens,
                response_tokens=req.response_tokens,
                total_tokens=req.total_tokens,
                model=req.model,
                provider=req.provider,
                actual_cost=req.actual_cost,
                latency_ms=req.latency_ms,
                ttft_ms=req.ttft_ms,
            )
        )
    return rebased


def generate_scenario_workload(
    scenario: ScenarioConfig,
    *,
    seed: int = 0,
    dataset_name: str = "synthetic",
) -> list[Request]:
    """Generate the shared workload for one scenario.

    Args:
        scenario: Scenario configuration.
        seed: RNG seed controlling synthetic generation. Unused for
            trace-driven datasets (the full trace is replayed in natural
            order; there is no slicing or resampling).
        dataset_name: Workload source. Use ``"synthetic"`` for the
            built-in Poisson generator, or one of ``TRACE_WORKLOAD_DATASETS``
            for trace-driven replay. Trace replay is always natural-rate;
            the earlier scaled-replay mode and its
            ``trace_replay_natural`` opt-in flag were removed on
            2026-04-29 (see :func:`_generate_trace_driven_workload`
            docstring for the rationale).
    """
    if dataset_name != "synthetic":
        return _generate_trace_driven_workload(
            scenario,
            dataset_name=dataset_name,
            seed=seed,
        )

    return generate_workload(
        n_requests=scenario.n_requests,
        duration_seconds=scenario.duration_seconds,
        seed=seed,
        start_time=0.0,
        arrival_process=scenario.arrival_process,
    )


def summarize_main_metrics(
    scenario: ScenarioConfig,
    evaluated_runs: list[EvaluatedRun],
) -> dict[str, object]:
    """Aggregate user-facing metrics across seeds for one variant."""
    runs = [evaluated.run for evaluated in evaluated_runs]
    return {
        "mean_ttft_ms": float(np.mean([float(np.mean(run.ttft_ms)) for run in runs])),
        "p50_ms": float(np.mean([run.p50_ms() for run in runs])),
        "p90_ms": float(np.mean([float(np.percentile(run.ttft_ms, 90)) for run in runs])),
        "p99_ms": float(np.mean([run.p99_ms() for run in runs])),
        "slo_violation_rate": float(
            np.mean([run.slo_violation_rate(scenario.primary_slo_ms) for run in runs])
        ),
        "avg_cost_usd": float(np.mean([run.mean_cost_usd() for run in runs])),
        "hedge_rate": float(np.mean([run.hedge_rate() for run in runs])),
        "provider_mix": _mean_provider_fraction(runs),
        "tier_mix": _mean_tier_fraction(runs),
    }


def summarize_diagnostics(
    evaluated_runs: list[EvaluatedRun],
) -> dict[str, object]:
    """Aggregate diagnostics across seeds for one variant."""
    merged = RunDiagnostics(
        variant=evaluated_runs[0].diagnostics.variant,
        scenario_name=evaluated_runs[0].diagnostics.scenario_name,
        seed=-1,
    )
    for evaluated in evaluated_runs:
        diagnostics = evaluated.diagnostics
        merged.solver_status_counts.update(diagnostics.solver_status_counts)
        merged.fallback_counts.update(diagnostics.fallback_counts)
        merged.budget_values.extend(diagnostics.budget_values)
        merged.expected_c_eff_values.extend(diagnostics.expected_c_eff_values)
        merged.utilization_values.extend(diagnostics.utilization_values)
        merged.slack_values.extend(diagnostics.slack_values)
        merged.total_decisions += diagnostics.total_decisions
        merged.non_optimal_decisions += diagnostics.non_optimal_decisions
        merged.single_feasible_provider_decisions += (
            diagnostics.single_feasible_provider_decisions
        )
        merged.trivial_single_provider_outcomes += (
            diagnostics.trivial_single_provider_outcomes
        )
        merged.true_p50_fallback_count += diagnostics.true_p50_fallback_count
        merged.explorer_feedback_count += diagnostics.explorer_feedback_count
        merged.backup_selection_counts.update(diagnostics.backup_selection_counts)
        merged.backup_random_prob_values.extend(diagnostics.backup_random_prob_values)
    return merged.to_summary_dict()


def build_hedge_delta(
    no_hedge: dict[str, object],
    hedged: dict[str, object],
) -> dict[str, float]:
    """Compute no-hedge -> hedge deltas for one selector family."""
    return {
        "delta_p50_ms": float(hedged["p50_ms"] - no_hedge["p50_ms"]),
        "delta_p90_ms": float(hedged["p90_ms"] - no_hedge["p90_ms"]),
        "delta_p99_ms": float(hedged["p99_ms"] - no_hedge["p99_ms"]),
        "delta_slo_violation_rate": float(
            hedged["slo_violation_rate"] - no_hedge["slo_violation_rate"]
        ),
        "delta_cost_usd": float(hedged["avg_cost_usd"] - no_hedge["avg_cost_usd"]),
    }


__all__ = [
    "BACKUP_SCOPES",
    "BACKUP_EXPLORATION_VARIANTS",
    "CONTROL_VARIANTS",
    "EvaluatedRun",
    "EXPLORER_VARIANTS",
    "FIRST_BATCH_SCENARIOS",
    "HEDGE_ABLATION_VARIANTS",
    "MAIN_VARIANTS",
    "PROVIDER_PERCENTILE_ABLATION_VARIANTS",
    "RunDiagnostics",
    "TRACE_WORKLOAD_DATASETS",
    "build_all_scenarios",
    "build_first_batch_scenarios",
    "build_hedge_delta",
    "canonicalize_variant_name",
    "generate_scenario_workload",
    "run_variant",
    "summarize_diagnostics",
    "summarize_main_metrics",
]
