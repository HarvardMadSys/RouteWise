"""The RouteWise routing algorithm, written once against ProviderView.

This module owns the orchestration that used to be duplicated between the
simulator policy (``routewise.sim.policies.routewise.RouteWisePolicy``) and the
real-eval harness (``experiments.real_evaluation.policies.BudgetRangePolicy``):

* body selector: effective-cost map -> range budget
  ``B_alpha = (1 - alpha) * c_min + alpha * c_max`` -> budget LP ->
  sample a primary from the LP weights,
* probability-target hedging: score backups at in-flight checkpoints with the
  conditional SLO success probability, dispatch the cheapest feasible backup
  only when no later checkpoint still has one,
* 429 fallback: re-solve the same body objective over the non-excluded
  providers to produce a fallback order,
* learning: feed observed TTFTs / errors into the rolling latency beliefs.

Environment differences are injected, never branched on: the
:class:`~routewise.core.provider_view.ProviderView` adapters carry costing and
prior beliefs, ``sampler`` carries each harness's RNG discipline verbatim, and
the ``hedge_*`` knobs carry the calibration deltas. The router itself is
deterministic given its inputs, stdlib-only, and lock-free (real-eval
serializes calls behind its policy lock).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

from routewise.core.beliefs import LatencyBeliefs
from routewise.core.cost import effective_cost
from routewise.core.hedging import (
    DISPATCH_OVERHEAD_MS,
    HEDGE_SUCCESS_TARGET,
    BackupCandidate,
    combined_success_probability,
    has_feasible_backup,
    hedge_checkpoints_for_slo,
    select_probability_backup,
)
from routewise.core.lp import (
    LP_EPS,
    BudgetLPCandidate,
    cost_tiebroken_objective,
    solve_budget_lp,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from routewise.core.provider_view import ProviderView

    Sampler = Callable[[Mapping[str, float], float], str]
    BackupSelector = Callable[[Sequence[BackupCandidate]], BackupCandidate | None]
    DispatchGate = Callable[[BackupCandidate, bool], bool]


class LPStatus(str, Enum):
    """Canonical body-selector outcome; adapters map to their legacy strings."""

    FEASIBLE = "feasible"
    SINGLE_CANDIDATE = "single_candidate"
    FALLBACK_MIN_COST = "fallback_min_cost"


@dataclass(frozen=True)
class RouteResult:
    """One body-routing decision; weight insertion order == input view order."""

    primary: str
    weights: dict[str, float]
    c_eff: dict[str, float]
    latency_objective_ms: dict[str, float]
    budget: float
    c_min: float
    c_max: float
    lp_status: LPStatus
    hedge_checkpoints_sec: tuple[float, ...]


@dataclass(frozen=True)
class CheckpointResult:
    """One in-flight hedge checkpoint evaluation."""

    backup: str | None
    success_probability: float | None
    future_feasible: bool
    feasible_count: int
    candidate_count: int


def argmax_weight_sampler(weights: Mapping[str, float], now: float) -> str:
    """Deterministic default sampler: highest weight, then lowest name."""
    del now
    return min(weights.items(), key=lambda item: (-item[1], item[0]))[0]


def latest_safe_dispatch_gate(selected: BackupCandidate, future_feasible: bool) -> bool:
    """Dispatch the backup only when no later checkpoint still has one."""
    return selected.feasible and not future_feasible


def _same_cost_shortcut_weights(
    names: list[str],
    *,
    objective: list[float],
    costs: list[float],
) -> dict[str, float]:
    """Return the one-hot LP optimum when every provider has the same cost.

    Replicates ``solve_budget_lp``'s tie-break key exactly (mixtures are
    skipped when the cost span is within eps, so pure vectors are the only
    candidates), avoiding the full enumeration.
    """
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


@dataclass
class RouteWiseRouter:
    """Single-source RouteWise algorithm over :class:`ProviderView` fleets."""

    alpha: float
    slo_ms: float
    beliefs: LatencyBeliefs = field(default_factory=LatencyBeliefs)
    cost_envelope: tuple[float, float] | None = None
    hedging: bool = True
    sampler: Sampler = argmax_weight_sampler
    hedge_dispatch_overhead_ms: float = DISPATCH_OVERHEAD_MS
    hedge_success_target: float = HEDGE_SUCCESS_TARGET
    # Hedge-backup latency tie-break: the simulator uses the oracle prior,
    # real-eval uses the penalized empirical belief.
    hedge_tiebreak_mean: Literal["prior", "belief"] = "belief"
    # Override points used by the hedging-ablation policies.
    backup_selector: BackupSelector = select_probability_backup
    dispatch_gate: DispatchGate = latest_safe_dispatch_gate

    def route(self, views: Sequence[ProviderView], now: float) -> RouteResult:
        """Solve the LP-TTFT-budget body and pick a primary provider.

        ``views`` must be non-empty and already availability-filtered by the
        adapter (each side keeps its own none-available behavior). The
        injected ``sampler`` is called exactly once per invocation, even for
        singleton weight maps, so harness RNG streams advance identically to
        the pre-router implementations.
        """
        if not views:
            raise ValueError("RouteWiseRouter.route requires at least one provider view")
        L, U = self._envelope_or_raise()
        c_eff = {view.name: self._effective_cost(view, now, L=L, U=U) for view in views}
        tbar = {view.name: self.beliefs.mean_ms(view, now) for view in views}

        names = [view.name for view in views]
        c_min = min(c_eff.values())
        c_max = max(c_eff.values())
        budget = (1.0 - self.alpha) * c_min + self.alpha * c_max
        objective = cost_tiebroken_objective(
            [tbar[name] for name in names],
            [c_eff[name] for name in names],
        )

        if c_max - c_min <= LP_EPS:
            weights = _same_cost_shortcut_weights(
                names,
                objective=objective,
                costs=[c_eff[name] for name in names],
            )
            lp_status = LPStatus.SINGLE_CANDIDATE if len(names) == 1 else LPStatus.FEASIBLE
        else:
            result = solve_budget_lp(
                [
                    BudgetLPCandidate(
                        name=name,
                        objective=objective[index],
                        effective_cost=c_eff[name],
                    )
                    for index, name in enumerate(names)
                ],
                budget=budget,
            )
            weights = result.weights if result.feasible else {}
            # budget = (1-alpha)*c_min + alpha*c_max with alpha in [0, 1], so
            # budget >= c_min and the min-cost provider is always within
            # budget. An empty solve is therefore a solver fallback, not an
            # over-budget infeasibility.
            lp_status = LPStatus.FEASIBLE
        if not weights:
            best = min(views, key=lambda view: (c_eff[view.name], tbar[view.name]))
            weights = {best.name: 1.0}
            lp_status = LPStatus.FALLBACK_MIN_COST

        primary = self.sampler(weights, now)
        hedge_checkpoints: tuple[float, ...] = ()
        if self.hedging:
            hedge_checkpoints = hedge_checkpoints_for_slo(self.slo_ms)

        return RouteResult(
            primary=primary,
            weights=weights,
            c_eff=c_eff,
            latency_objective_ms=tbar,
            budget=budget,
            c_min=c_min,
            c_max=c_max,
            lp_status=lp_status,
            hedge_checkpoints_sec=hedge_checkpoints,
        )

    def checkpoint_backup(
        self,
        primary: str,
        views: Sequence[ProviderView],
        now: float,
        *,
        elapsed_sec: float,
        future_checkpoints_sec: Sequence[float] = (),
    ) -> CheckpointResult:
        """Evaluate probability-target hedging at one in-flight checkpoint.

        ``views`` must include the primary (its belief CDF conditions the
        success probability); candidates are the available non-primary views
        in input order.
        """
        primary_view = self._view_by_name(primary, views)
        elapsed_ms = float(elapsed_sec) * 1000.0
        candidates = self._collect_backup_candidates(
            primary_view,
            views,
            now,
            elapsed_ms=elapsed_ms,
        )
        candidate_count = len(candidates)
        feasible_count = sum(1 for candidate in candidates if candidate.feasible)
        selected = self.backup_selector(candidates)
        if selected is None or not selected.feasible:
            return CheckpointResult(
                backup=None,
                success_probability=None,
                future_feasible=False,
                feasible_count=feasible_count,
                candidate_count=candidate_count,
            )

        future_feasible = False
        for future_elapsed in future_checkpoints_sec:
            if future_elapsed <= elapsed_sec + LP_EPS:
                continue
            future_candidates = self._collect_backup_candidates(
                primary_view,
                views,
                now,
                elapsed_ms=float(future_elapsed) * 1000.0,
            )
            if has_feasible_backup(future_candidates):
                future_feasible = True
                break

        backup: str | None = None
        if self.dispatch_gate(selected, future_feasible):
            backup = str(selected.provider.name)
        return CheckpointResult(
            backup=backup,
            success_probability=float(selected.success_probability),
            future_feasible=future_feasible,
            feasible_count=feasible_count,
            candidate_count=candidate_count,
        )

    def fallback_order(self, views: Sequence[ProviderView], now: float) -> list[str]:
        """Re-solve the body objective for provider-local failures (429s).

        Returns every view's name, ordered by the same cost budget and
        latency objective as :meth:`route`: LP-weighted providers first
        (heaviest weight, then latency, cost, name), then the remainder by
        (over-budget, latency, cost, name). The adapter excludes failed
        providers before calling.
        """
        if not views:
            return []
        L, U = self._envelope_or_raise()
        c_eff = {view.name: self._effective_cost(view, now, L=L, U=U) for view in views}
        tbar = {view.name: self.beliefs.mean_ms(view, now) for view in views}

        c_values = list(c_eff.values())
        c_min = min(c_values)
        c_max = max(c_values)
        budget = float((1.0 - self.alpha) * c_min + self.alpha * c_max)
        names = [view.name for view in views]
        objective = cost_tiebroken_objective(
            [tbar[name] for name in names],
            [c_eff[name] for name in names],
        )
        result = solve_budget_lp(
            [
                BudgetLPCandidate(
                    name=name,
                    objective=objective[index],
                    effective_cost=c_eff[name],
                )
                for index, name in enumerate(names)
            ],
            budget=budget,
        )

        lp_order: list[str] = []
        if result.feasible:
            lp_order = [
                name
                for name, weight in sorted(
                    result.weights.items(),
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

    def observe(self, provider_name: str, observed_at: float, ttft_ms: float) -> None:
        """Feed one observed TTFT sample into the latency beliefs."""
        self.beliefs.observe(provider_name, observed_at, ttft_ms)

    def observe_error(self, provider_name: str, observed_at: float, error_type: str) -> None:
        """Feed one failed attempt (429, timeout) into the latency beliefs."""
        self.beliefs.observe_error(provider_name, observed_at, error_type)

    def _envelope_or_raise(self) -> tuple[float, float]:
        if self.cost_envelope is None:
            raise ValueError("RouteWiseRouter requires cost_envelope before routing")
        return (float(self.cost_envelope[0]), float(self.cost_envelope[1]))

    def _effective_cost(
        self,
        view: ProviderView,
        now: float,
        *,
        L: float,
        U: float,
    ) -> float:
        """Paper-formula piecewise effective request cost for one view."""
        tier = view.tier
        if tier == "api":
            return effective_cost(
                "api",
                request_cost_usd=view.route_cost_usd(now),
                L=L,
                U=U,
            )
        if tier == "quota":
            return effective_cost(
                "quota",
                quota_fraction_used=view.quota_fraction_used(now),
                L=L,
                U=U,
            )
        if tier == "concurrency":
            return effective_cost(
                "concurrency",
                concurrency_utilization=None,
                L=L,
                U=U,
            )
        raise ValueError(f"Unsupported provider tier for RouteWise effective cost: {tier!r}")

    def _collect_backup_candidates(
        self,
        primary_view: ProviderView,
        views: Sequence[ProviderView],
        now: float,
        *,
        elapsed_ms: float,
    ) -> list[BackupCandidate[ProviderView]]:
        candidates: list[BackupCandidate[ProviderView]] = []
        for view in views:
            if view.name == primary_view.name or not view.is_available(now):
                continue
            candidates.append(
                BackupCandidate(
                    provider=view,
                    success_probability=combined_success_probability(
                        lambda value_ms: self.beliefs.cdf(primary_view, value_ms, now),
                        lambda value_ms, backup=view: self.beliefs.cdf(backup, value_ms, now),
                        elapsed_ms=elapsed_ms,
                        slo_ms=self.slo_ms,
                        dispatch_overhead_ms=self.hedge_dispatch_overhead_ms,
                    ),
                    marginal_cost=view.hedge_cost_usd(now),
                    true_mean_ms=self._hedge_tiebreak_mean_ms(view, now),
                    success_target=self.hedge_success_target,
                )
            )
        return candidates

    def _hedge_tiebreak_mean_ms(self, view: ProviderView, now: float) -> float:
        if self.hedge_tiebreak_mean == "prior":
            return self.beliefs.prior_mean_ms(view, now)
        return self.beliefs.mean_ms(view, now)

    def _view_by_name(
        self,
        name: str,
        views: Sequence[ProviderView],
    ) -> ProviderView:
        for view in views:
            if view.name == name:
                return view
        raise KeyError(f"Provider view not found for primary {name!r}")


__all__ = [
    "CheckpointResult",
    "LPStatus",
    "RouteResult",
    "RouteWiseRouter",
    "argmax_weight_sampler",
    "latest_safe_dispatch_gate",
]
