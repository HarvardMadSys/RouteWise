"""Tests for the unified RouteWise router over fake provider views."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from rwsim.core.beliefs import LatencyBeliefs
from rwsim.core.router import (
    LPStatus,
    RouteWiseRouter,
    argmax_weight_sampler,
)


@dataclass
class FakeView:
    """Configurable ProviderView stub."""

    name: str
    tier: str = "api"
    cost: float = 0.001
    hedge_cost: float | None = None
    quota_fraction: float | None = None
    available: bool = True
    prior_mean: float | None = None
    prior_cdf_value: float | None = None
    cost_calls: list[float] = field(default_factory=list)

    def is_available(self, now: float) -> bool:
        return self.available

    def quota_fraction_used(self, now: float) -> float | None:
        return self.quota_fraction

    def route_cost_usd(self, now: float) -> float:
        self.cost_calls.append(now)
        return self.cost

    def hedge_cost_usd(self, now: float) -> float:
        return self.cost if self.hedge_cost is None else self.hedge_cost

    def prior_ttft_mean_ms(self, now: float) -> float | None:
        return self.prior_mean

    def prior_ttft_cdf(self, value_ms: float, now: float) -> float | None:
        return self.prior_cdf_value


ENVELOPE = (0.0001, 0.01)


def make_router(**overrides) -> RouteWiseRouter:
    defaults = {
        "alpha": 0.75,
        "slo_ms": 2000.0,
        "beliefs": LatencyBeliefs(window_sec=900.0),
        "cost_envelope": ENVELOPE,
        "hedging": True,
    }
    defaults.update(overrides)
    return RouteWiseRouter(**defaults)


def seed_profile(router: RouteWiseRouter, name: str, ttfts: list[float], start: float = 0.0) -> None:
    for index, value in enumerate(ttfts):
        router.observe(name, start + float(index), value)


def test_route_prefers_fast_provider_within_budget() -> None:
    router = make_router(alpha=1.0)
    fast = FakeView("fast", cost=0.002)
    cheap = FakeView("cheap", cost=0.001)
    seed_profile(router, "fast", [100.0] * 5)
    seed_profile(router, "cheap", [900.0] * 5)
    result = router.route([fast, cheap], now=10.0)
    assert result.weights == {"fast": 1.0}
    assert result.lp_status is LPStatus.FEASIBLE
    assert result.primary == "fast"
    assert result.budget == pytest.approx(0.002)


def test_route_alpha_zero_forces_min_cost() -> None:
    router = make_router(alpha=0.0)
    fast = FakeView("fast", cost=0.002)
    cheap = FakeView("cheap", cost=0.001)
    seed_profile(router, "fast", [100.0] * 5)
    seed_profile(router, "cheap", [900.0] * 5)
    result = router.route([fast, cheap], now=10.0)
    assert result.weights == {"cheap": 1.0}


def test_route_same_cost_shortcut_matches_lp(monkeypatch) -> None:
    router = make_router()
    a = FakeView("a", cost=0.001)
    b = FakeView("b", cost=0.001)
    seed_profile(router, "a", [500.0] * 5)
    seed_profile(router, "b", [100.0] * 5)

    def _boom(*args, **kwargs):
        raise AssertionError("solve_budget_lp should be skipped for same-cost fleets")

    monkeypatch.setattr("rwsim.core.router.solve_budget_lp", _boom)
    result = router.route([a, b], now=10.0)
    assert result.weights == {"b": 1.0}
    assert result.lp_status is LPStatus.FEASIBLE


def test_route_single_candidate_status() -> None:
    router = make_router()
    only = FakeView("only", cost=0.001)
    seed_profile(router, "only", [100.0] * 5)
    result = router.route([only], now=10.0)
    assert result.lp_status is LPStatus.SINGLE_CANDIDATE
    assert result.weights == {"only": 1.0}


def test_route_sampler_called_exactly_once() -> None:
    calls: list[dict[str, float]] = []

    def counting_sampler(weights, now):
        calls.append(dict(weights))
        return argmax_weight_sampler(weights, now)

    router = make_router(sampler=counting_sampler)
    only = FakeView("only", cost=0.001)
    seed_profile(router, "only", [100.0] * 5)
    router.route([only], now=10.0)
    assert len(calls) == 1
    assert calls[0] == {"only": 1.0}


def test_route_requires_views_and_envelope() -> None:
    router = make_router()
    with pytest.raises(ValueError, match="at least one provider"):
        router.route([], now=0.0)
    router_no_envelope = make_router(cost_envelope=None)
    with pytest.raises(ValueError, match="cost_envelope"):
        router_no_envelope.route([FakeView("p")], now=0.0)


def test_route_quota_tier_uses_shadow_price_not_request_cost() -> None:
    router = make_router()
    quota = FakeView("quota", tier="quota", quota_fraction=0.5)
    api = FakeView("api", cost=0.005)
    seed_profile(router, "quota", [100.0] * 5)
    seed_profile(router, "api", [100.0] * 5)
    result = router.route([quota, api], now=10.0)
    assert quota.cost_calls == []  # request cost never queried for quota tier
    L, U = ENVELOPE
    assert result.c_eff["quota"] == pytest.approx(L * (U / L) ** 0.5)


def test_route_hedging_flag_controls_checkpoints() -> None:
    router = make_router(hedging=True, slo_ms=2000.0)
    no_hedge = make_router(hedging=False)
    view = FakeView("p", cost=0.001)
    seed_profile(router, "p", [100.0] * 5)
    seed_profile(no_hedge, "p", [100.0] * 5)
    assert router.route([view], now=10.0).hedge_checkpoints_sec != ()
    assert no_hedge.route([view], now=10.0).hedge_checkpoints_sec == ()


def _hedge_fixture(router: RouteWiseRouter) -> list[FakeView]:
    # Primary is slow past the SLO often; backup is fast and cheap.
    primary = FakeView("primary", cost=0.001)
    backup = FakeView("backup", cost=0.0005)
    slowpoke = FakeView("slowpoke", cost=0.0004)
    seed_profile(router, "primary", [500.0, 3000.0, 3500.0, 4000.0, 600.0])
    seed_profile(router, "backup", [100.0] * 10)
    seed_profile(router, "slowpoke", [5000.0] * 10)
    return [primary, backup, slowpoke]


def test_checkpoint_backup_selects_cheapest_feasible() -> None:
    router = make_router(slo_ms=2000.0, hedge_dispatch_overhead_ms=0.0)
    views = _hedge_fixture(router)
    result = router.checkpoint_backup(
        "primary",
        views,
        now=20.0,
        elapsed_sec=1.0,
        future_checkpoints_sec=(),
    )
    assert result.candidate_count == 2
    assert result.feasible_count >= 1
    assert result.backup == "backup"
    assert result.success_probability is not None


def test_checkpoint_backup_waits_for_future_feasible() -> None:
    router = make_router(slo_ms=2000.0, hedge_dispatch_overhead_ms=0.0)
    views = _hedge_fixture(router)
    result = router.checkpoint_backup(
        "primary",
        views,
        now=20.0,
        elapsed_sec=0.5,
        future_checkpoints_sec=(0.5, 1.0),
    )
    # A later checkpoint still has a feasible backup: defer.
    assert result.future_feasible is True
    assert result.backup is None
    assert result.success_probability is not None


def test_checkpoint_backup_none_when_no_candidates() -> None:
    router = make_router()
    primary = FakeView("primary")
    seed_profile(router, "primary", [100.0] * 3)
    result = router.checkpoint_backup("primary", [primary], now=10.0, elapsed_sec=1.0)
    assert result.backup is None
    assert result.candidate_count == 0
    assert result.feasible_count == 0


def test_checkpoint_backup_skips_unavailable_views() -> None:
    router = make_router(slo_ms=2000.0, hedge_dispatch_overhead_ms=0.0)
    views = _hedge_fixture(router)
    views[1].available = False  # the good backup goes away
    result = router.checkpoint_backup(
        "primary",
        views,
        now=20.0,
        elapsed_sec=1.0,
    )
    assert result.candidate_count == 1
    assert result.backup != "backup"


def test_checkpoint_backup_selector_and_gate_hooks() -> None:
    picked = []

    def pick_last_feasible(candidates):
        feasible = [c for c in candidates if c.feasible]
        picked.append(len(candidates))
        return feasible[-1] if feasible else None

    router = make_router(
        slo_ms=2000.0,
        hedge_dispatch_overhead_ms=0.0,
        backup_selector=pick_last_feasible,
        dispatch_gate=lambda selected, future: True,  # always dispatch
    )
    views = _hedge_fixture(router)
    result = router.checkpoint_backup(
        "primary",
        views,
        now=20.0,
        elapsed_sec=0.5,
        future_checkpoints_sec=(1.0,),
    )
    assert picked  # custom selector ran with the full candidate list
    assert result.backup is not None  # gate override dispatches despite future feasibility


def test_fallback_order_ranks_lp_weights_then_remainder() -> None:
    router = make_router(alpha=1.0)
    fast = FakeView("fast", cost=0.002)
    cheap = FakeView("cheap", cost=0.001)
    slow_pricy = FakeView("slow_pricy", cost=0.009)
    seed_profile(router, "fast", [100.0] * 5)
    seed_profile(router, "cheap", [400.0] * 5)
    seed_profile(router, "slow_pricy", [900.0] * 5)
    order = router.fallback_order([fast, cheap, slow_pricy], now=10.0)
    assert order[0] == "fast"
    assert set(order) == {"fast", "cheap", "slow_pricy"}
    assert len(order) == 3


def test_fallback_order_empty_views() -> None:
    router = make_router()
    assert router.fallback_order([], now=0.0) == []


def test_hedge_tiebreak_mean_modes() -> None:
    prior_router = make_router(hedge_tiebreak_mean="prior")
    belief_router = make_router(hedge_tiebreak_mean="belief")
    view = FakeView("p", prior_mean=42.0)
    seed_profile(prior_router, "p", [100.0] * 3)
    seed_profile(belief_router, "p", [100.0] * 3)
    assert prior_router._hedge_tiebreak_mean_ms(view, 5.0) == pytest.approx(42.0)
    assert belief_router._hedge_tiebreak_mean_ms(view, 5.0) == pytest.approx(100.0)
