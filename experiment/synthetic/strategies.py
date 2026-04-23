"""Routing strategies for synthetic experiment.

Implements 10 strategies organized as 5 selection mechanisms x 2 architectures:

Selection mechanisms:
  - cheapest: argmin cost
  - fastest:  argmin P50 latency
  - v2:       P50 near-best band, then cheapest in band
  - lp:       LP with F(SLO) >= alpha constraint, cost objective
  - hedge:    base selection (cheapest), plus cross-provider hedging

Architectures:
  - two_layer:  Layer 1 picks tier (S_Q if quota available, else S_C if slots
                free, else S_A). Layer 2 picks provider inside that tier
                using the mechanism.
  - joint:     Single layer. Computes effective cost c_eff per provider across
                all tiers; applies mechanism globally.

Every strategy returns a RouteDecision with:
  - primary: chosen provider name
  - hedge_after_ms (optional): if set, a backup is dispatched after waiting
    this many ms for the primary
  - backup: backup provider name (for hedge variants)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from scipy.optimize import linprog

from experiment.synthetic.effective_cost import (
    DEFAULT_L,
    DEFAULT_U,
    effective_cost,
)
from experiment.synthetic.provider import (
    SyntheticProvider,
    TIER_S_A,
    TIER_S_C,
    TIER_S_Q,
)
from experiment.synthetic.state import SimState
from experiment.synthetic.workload import SyntheticRequest


@dataclass
class RouteDecision:
    """Routing decision for one request."""

    primary: str
    backup: str | None = None
    hedge_after_ms: float | None = None


# -----------------------------------------------------------------------------
# Base and helpers
# -----------------------------------------------------------------------------


class Strategy(ABC):
    """Abstract strategy interface."""

    def __init__(
        self,
        providers: list[SyntheticProvider],
        slo_ms: float = 2000.0,
        p50_band: float = 0.10,
        alpha: float = 0.99,
        hedge_after_frac: float = 0.3,
        L: float = DEFAULT_L,
        U: float = DEFAULT_U,
    ):
        self.providers = providers
        self.providers_by_name = {p.name: p for p in providers}
        self.slo_ms = slo_ms
        self.p50_band = p50_band
        self.alpha = alpha
        self.hedge_after_frac = hedge_after_frac
        self.L = L
        self.U = U

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def route(self, request: SyntheticRequest, state: SimState) -> RouteDecision:
        pass

    # --- Shared helpers ---

    def _available(
        self, providers: list[SyntheticProvider], state: SimState
    ) -> list[SyntheticProvider]:
        return [p for p in providers if state.can_accept(p)]

    def _p50_or_prior(
        self, provider: SyntheticProvider, state: SimState
    ) -> float:
        """P50 estimate with a prior for cold start.

        Before the rolling window has 3+ samples, use the provider's
        log-normal median e^mu as a sensible prior (not ground-truth but
        a model-based estimate, matching what probing would give in
        practice).
        """
        profile = state.profiles[provider.name]
        if profile.n_samples() >= 3:
            return profile.p50(state.current_time) or float("inf")
        # Cold-start prior: expected median from log-normal is e^mu.
        import math
        return math.exp(provider.ttft_mu)

    def _cdf_or_prior(
        self,
        provider: SyntheticProvider,
        state: SimState,
        threshold_ms: float,
    ) -> float:
        """Empirical F(threshold) or log-normal CDF if insufficient samples."""
        profile = state.profiles[provider.name]
        if profile.n_samples() >= 10:
            val = profile.cdf_at(state.current_time, threshold_ms)
            if val is not None:
                return val
        # Prior: log-normal CDF.
        import math
        if threshold_ms <= 0:
            return 0.0
        z = (math.log(threshold_ms) - provider.ttft_mu) / provider.ttft_sigma
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # --- Two-layer: tier selection ---

    def _pick_tier(
        self, request: SyntheticRequest, state: SimState
    ) -> list[SyntheticProvider]:
        """Layer 1 tier selection: choose the cheapest tier with capacity.

        Textbook two-layer: prefer S_Q, then S_C, then S_A. This is the
        greedy "fill subscription first" rule that the paper's cost router
        shadow-price mechanism is designed to replace. For a direct
        two-layer architecture comparison we use this greedy version as
        the reference (it's what most practitioners implement).
        """
        s_q = [p for p in self.providers if p.is_s_q]
        s_c = [p for p in self.providers if p.is_s_c]
        s_a = [p for p in self.providers if p.is_s_a]

        if any(state.can_accept(p) for p in s_q):
            return [p for p in s_q if state.can_accept(p)]
        if any(state.can_accept(p) for p in s_c):
            return [p for p in s_c if state.can_accept(p)]
        return s_a


# -----------------------------------------------------------------------------
# Mechanism 1: cheapest
# -----------------------------------------------------------------------------


class TwoLayerCheapest(Strategy):
    @property
    def name(self) -> str:
        return "two_layer_cheapest"

    def route(self, request, state):
        candidates = self._pick_tier(request, state)
        if not candidates:
            candidates = self._available(self.providers, state) or self.providers
        primary = min(
            candidates,
            key=lambda p: p.marginal_cost(request),
        )
        return RouteDecision(primary=primary.name)


class JointCheapest(Strategy):
    @property
    def name(self) -> str:
        return "joint_cheapest"

    def route(self, request, state):
        candidates = self._available(self.providers, state) or self.providers
        primary = min(
            candidates,
            key=lambda p: effective_cost(p, request, state, self.L, self.U),
        )
        return RouteDecision(primary=primary.name)


# -----------------------------------------------------------------------------
# Mechanism 2: fastest (argmin P50)
# -----------------------------------------------------------------------------


class TwoLayerFastest(Strategy):
    @property
    def name(self) -> str:
        return "two_layer_fastest"

    def route(self, request, state):
        candidates = self._pick_tier(request, state)
        if not candidates:
            candidates = self._available(self.providers, state) or self.providers
        primary = min(candidates, key=lambda p: self._p50_or_prior(p, state))
        return RouteDecision(primary=primary.name)


class JointFastest(Strategy):
    @property
    def name(self) -> str:
        return "joint_fastest"

    def route(self, request, state):
        candidates = self._available(self.providers, state) or self.providers
        primary = min(candidates, key=lambda p: self._p50_or_prior(p, state))
        return RouteDecision(primary=primary.name)


# -----------------------------------------------------------------------------
# Mechanism 3: V2 (P50 near-best band, then cheapest)
# -----------------------------------------------------------------------------


def _v2_select(
    candidates: list[SyntheticProvider],
    state: SimState,
    p50_band: float,
    cost_fn,
    p50_fn,
) -> SyntheticProvider:
    """V2 rule: Pareto filter + near-best P50 band + cheapest.

    cost_fn(p): USD cost (may be marginal or effective).
    p50_fn(p):  P50 in ms.
    """
    if not candidates:
        raise ValueError("No candidates for V2 selection")

    # Pareto prune on (P50, cost).
    non_dominated = []
    for p in candidates:
        p50_p = p50_fn(p)
        cost_p = cost_fn(p)
        dominated = False
        for q in candidates:
            if q is p:
                continue
            if p50_fn(q) <= p50_p and cost_fn(q) <= cost_p:
                if p50_fn(q) < p50_p or cost_fn(q) < cost_p:
                    dominated = True
                    break
        if not dominated:
            non_dominated.append(p)
    if not non_dominated:
        non_dominated = list(candidates)

    best_p50 = min(p50_fn(p) for p in non_dominated)
    band_threshold = best_p50 * (1.0 + p50_band)
    in_band = [p for p in non_dominated if p50_fn(p) <= band_threshold]
    if not in_band:
        in_band = non_dominated

    return min(in_band, key=cost_fn)


class TwoLayerV2(Strategy):
    @property
    def name(self) -> str:
        return "two_layer_v2"

    def route(self, request, state):
        candidates = self._pick_tier(request, state)
        if not candidates:
            candidates = self._available(self.providers, state) or self.providers
        primary = _v2_select(
            candidates,
            state,
            self.p50_band,
            cost_fn=lambda p: p.marginal_cost(request),
            p50_fn=lambda p: self._p50_or_prior(p, state),
        )
        return RouteDecision(primary=primary.name)


class JointV2(Strategy):
    @property
    def name(self) -> str:
        return "joint_v2"

    def route(self, request, state):
        candidates = self._available(self.providers, state) or self.providers
        primary = _v2_select(
            candidates,
            state,
            self.p50_band,
            cost_fn=lambda p: effective_cost(p, request, state, self.L, self.U),
            p50_fn=lambda p: self._p50_or_prior(p, state),
        )
        return RouteDecision(primary=primary.name)


# -----------------------------------------------------------------------------
# Mechanism 4: LP (F(SLO) >= alpha constraint)
# -----------------------------------------------------------------------------


def _solve_lp(
    candidates: list[SyntheticProvider],
    costs: list[float],
    cdf_at_slo: list[float],
    alpha: float,
) -> SyntheticProvider | None:
    """Solve LP: min sum(pi_j * cost_j) s.t. sum(pi_j * cdf_j) >= alpha, sum pi = 1.

    If feasible, sample from the solution (deterministically return argmax pi
    for stability in synthetic runs). If infeasible, fall back to argmax cdf.
    """
    n = len(candidates)
    if n == 0:
        return None

    c = costs
    # Constraint 1: -sum(cdf * pi) <= -alpha  (i.e., sum(cdf*pi) >= alpha)
    A_ub = [[-c for c in cdf_at_slo]]
    b_ub = [-alpha]
    # Constraint 2: sum(pi) == 1
    A_eq = [[1.0] * n]
    b_eq = [1.0]
    bounds = [(0.0, 1.0) for _ in range(n)]

    try:
        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
    except Exception:
        return None

    if res.success and res.x is not None:
        idx = int(max(range(n), key=lambda i: res.x[i]))
        return candidates[idx]

    # Infeasible: pick the one with highest cdf_at_slo.
    idx = int(max(range(n), key=lambda i: cdf_at_slo[i]))
    return candidates[idx]


class TwoLayerLP(Strategy):
    @property
    def name(self) -> str:
        return "two_layer_lp"

    def route(self, request, state):
        candidates = self._pick_tier(request, state)
        if not candidates:
            candidates = self._available(self.providers, state) or self.providers
        costs = [p.marginal_cost(request) for p in candidates]
        cdfs = [self._cdf_or_prior(p, state, self.slo_ms) for p in candidates]
        primary = _solve_lp(candidates, costs, cdfs, self.alpha)
        if primary is None:
            primary = min(candidates, key=lambda p: p.marginal_cost(request))
        return RouteDecision(primary=primary.name)


class JointLP(Strategy):
    @property
    def name(self) -> str:
        return "joint_lp"

    def route(self, request, state):
        candidates = self._available(self.providers, state) or self.providers
        costs = [
            effective_cost(p, request, state, self.L, self.U)
            for p in candidates
        ]
        cdfs = [self._cdf_or_prior(p, state, self.slo_ms) for p in candidates]
        primary = _solve_lp(candidates, costs, cdfs, self.alpha)
        if primary is None:
            primary = min(
                candidates,
                key=lambda p: effective_cost(p, request, state, self.L, self.U),
            )
        return RouteDecision(primary=primary.name)


# -----------------------------------------------------------------------------
# Mechanism 5: hedge (cheapest + hedging backup)
# -----------------------------------------------------------------------------


class TwoLayerHedge(Strategy):
    """Two-layer base (tier=S_Q if avail) + within-tier hedging.

    Primary: cheapest provider in chosen tier.
    Backup:  fastest provider in same tier (within-tier hedging only).
    Trigger: after hedge_after_frac * SLO milliseconds.
    """

    @property
    def name(self) -> str:
        return "two_layer_hedge"

    def route(self, request, state):
        candidates = self._pick_tier(request, state)
        if not candidates:
            candidates = self._available(self.providers, state) or self.providers
        primary = min(candidates, key=lambda p: p.marginal_cost(request))
        # Backup: fastest in same tier that isn't the primary.
        others = [p for p in candidates if p.name != primary.name]
        if others:
            backup = min(others, key=lambda p: self._p50_or_prior(p, state))
            hedge_after_ms = self.hedge_after_frac * self.slo_ms
            return RouteDecision(
                primary=primary.name,
                backup=backup.name,
                hedge_after_ms=hedge_after_ms,
            )
        return RouteDecision(primary=primary.name)


class JointHedge(Strategy):
    """Joint base (argmin c_eff globally) + cross-tier hedging.

    Primary: argmin c_eff across all tiers.
    Backup:  fastest provider (by P50) across all available providers,
             different from primary -- allowed to cross tiers.
    """

    @property
    def name(self) -> str:
        return "joint_hedge"

    def route(self, request, state):
        candidates = self._available(self.providers, state) or self.providers
        primary = min(
            candidates,
            key=lambda p: effective_cost(p, request, state, self.L, self.U),
        )
        others = [p for p in candidates if p.name != primary.name]
        if others:
            backup = min(others, key=lambda p: self._p50_or_prior(p, state))
            hedge_after_ms = self.hedge_after_frac * self.slo_ms
            return RouteDecision(
                primary=primary.name,
                backup=backup.name,
                hedge_after_ms=hedge_after_ms,
            )
        return RouteDecision(primary=primary.name)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


ALL_STRATEGY_CLASSES: list[type[Strategy]] = [
    TwoLayerCheapest,
    JointCheapest,
    TwoLayerFastest,
    JointFastest,
    TwoLayerV2,
    JointV2,
    TwoLayerLP,
    JointLP,
    TwoLayerHedge,
    JointHedge,
]


def build_all_strategies(
    providers: list[SyntheticProvider],
    slo_ms: float,
    **kwargs,
) -> list[Strategy]:
    return [
        cls(providers=providers, slo_ms=slo_ms, **kwargs)
        for cls in ALL_STRATEGY_CLASSES
    ]
