"""Unit tests for offline oracle and simulator invariants.

These tests focus on Stage 1 offline optimal routing (daily quota knapsack with
unit weights) and basic simulator accounting invariants.

Goals:
1) Verify the Stage 1 offline oracle matches brute-force optimum on a toy trace.
2) Verify subscription eligibility constraints are respected by the oracle.
3) Sanity check: Offline-Optimal <= Online-Greedy <= All-API (total cost).
4) Verify subscription cost prorating and cost accounting are consistent.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from experiment.cost import CostCalculator
from experiment.data.schema import ProviderConfig, ProviderType, Request
from experiment.quota import QuotaManager
from experiment.simulator import OfflineSimulator
from experiment.strategies.all_api import AllAPIStrategy
from experiment.strategies.online.greedy import GreedyOnlineStrategy
from experiment.strategies.stage1_optimal import OptimalStrategy


def _make_request(
    req_id: int,
    day: int,
    request_tokens: int,
    response_tokens: int,
    model: str,
    offset_seconds: int = 0,
) -> Request:
    """Create a Request with deterministic timestamp."""
    timestamp = day * 86400 + offset_seconds
    return Request(
        id=req_id,
        timestamp=timestamp,
        request_tokens=request_tokens,
        response_tokens=response_tokens,
        total_tokens=request_tokens + response_tokens,
        model=model,
    )


def _make_stage1_config(
    *,
    api_provider_id: str = "api",
    sub_provider_id: str = "chutes-subscription",
    api_input_price_per_1k: float = 1.0,
    api_output_price_per_1k: float = 2.0,
    subscription_monthly_fee: float = 0.0,
    num_subscriptions: int = 1,
    supported_models: list[str] | None = None,
    model_pricing: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Create a minimal experiment config dict for the simulator/strategies."""
    providers = {
        api_provider_id: ProviderConfig(
            name="Test API",
            type=ProviderType.API,
            input_price_per_1k=api_input_price_per_1k,
            output_price_per_1k=api_output_price_per_1k,
        ),
        sub_provider_id: ProviderConfig(
            name="Test Subscription",
            type=ProviderType.SUBSCRIPTION,
            monthly_fee=subscription_monthly_fee,
            # Note: daily_quota here is informational for the simulator; strategies use QuotaManager.
            daily_quota=0,
        ),
    }

    return {
        "providers": providers,
        "simulation": {
            "num_subscriptions": num_subscriptions,
            "subscription_provider": sub_provider_id,
            "default_api_fallback": api_provider_id,
        },
        "subscriptions": {
            "chutes": {"supported_models": supported_models or []},
        },
        # Optional per-model pricing (per 1M tokens). When provided, it must cover all request.models.
        "model_pricing": model_pricing or {},
    }


def _api_cost_single_provider(
    *,
    request_tokens: int,
    response_tokens: int,
    input_price_per_1k: float,
    output_price_per_1k: float,
) -> float:
    """Compute API cost with a single provider (per-1K token pricing)."""
    return (
        request_tokens / 1000.0 * input_price_per_1k
        + response_tokens / 1000.0 * output_price_per_1k
    )


class TestOfflineStage1Oracle:
    """Tests for the Stage 1 offline optimal strategy."""

    def test_optimal_matches_bruteforce_single_day(self) -> None:
        """Optimal strategy should match brute-force selection (unit-weight knapsack)."""
        q = 2
        cfg = _make_stage1_config(
            api_input_price_per_1k=1.0,
            api_output_price_per_1k=2.0,
            subscription_monthly_fee=0.0,
            num_subscriptions=1,
            supported_models=[],  # all eligible
        )
        cost_calculator = CostCalculator(cfg["providers"], cfg.get("model_pricing"))
        quota_mgr = QuotaManager(daily_quota=q)

        # One-day toy trace with heterogeneous API costs (vary response tokens).
        outputs = [100, 200, 50, 400, 10, 300]
        requests = [
            _make_request(
                i, day=0, request_tokens=100, response_tokens=out, model="m", offset_seconds=i
            )
            for i, out in enumerate(outputs)
        ]

        # Brute force: choose min(q, n) requests to be "free"; pay API for the rest.
        costs = [
            _api_cost_single_provider(
                request_tokens=r.request_tokens,
                response_tokens=r.response_tokens,
                input_price_per_1k=1.0,
                output_price_per_1k=2.0,
            )
            for r in requests
        ]
        n = len(requests)
        choose_k = min(q, n)
        best_api_cost = float("inf")
        for chosen in combinations(range(n), choose_k):
            chosen_set = set(chosen)
            api_cost = sum(costs[i] for i in range(n) if i not in chosen_set)
            best_api_cost = min(best_api_cost, api_cost)

        strategy = OptimalStrategy(cost_calculator, quota_mgr, cfg)
        result = OfflineSimulator(requests, strategy, cfg).run()

        # subscription fee is 0 here, so total == api.
        assert result.to_dict()["costs"]["subscription"] == 0.0
        assert (
            pytest.approx(result.to_dict()["costs"]["api"], rel=1e-12, abs=1e-12) == best_api_cost
        )
        assert (
            pytest.approx(result.to_dict()["costs"]["total"], rel=1e-12, abs=1e-12) == best_api_cost
        )

    def test_optimal_respects_supported_models(self) -> None:
        """Oracle must not allocate subscription to ineligible models."""
        q = 1
        cfg = _make_stage1_config(
            api_input_price_per_1k=1.0,
            api_output_price_per_1k=10.0,
            subscription_monthly_fee=0.0,
            num_subscriptions=1,
            supported_models=["eligible"],
        )
        cost_calculator = CostCalculator(cfg["providers"], cfg.get("model_pricing"))
        quota_mgr = QuotaManager(daily_quota=q)

        # Ineligible request has huge cost, but should never get subscription.
        reqs = [
            _make_request(1, day=0, request_tokens=10, response_tokens=1000, model="ineligible"),
            _make_request(2, day=0, request_tokens=10, response_tokens=10, model="eligible"),
            _make_request(3, day=0, request_tokens=10, response_tokens=20, model="eligible"),
        ]

        strategy = OptimalStrategy(cost_calculator, quota_mgr, cfg)
        strategy.precompute(reqs)

        # The only subscription assignment must be among eligible requests.
        sub_assigned = [rid for rid, p in strategy.assignments.items() if p == "subscription"]
        assert len(sub_assigned) == 1
        assert sub_assigned[0] in {2, 3}
        assert strategy.assignments[1] == "api"


class TestOfflineSimulatorSanity:
    """Sanity tests comparing offline optimal and online baselines."""

    def test_cost_ordering_optimal_vs_greedy_vs_all_api(self) -> None:
        """Total cost should satisfy: Optimal <= Greedy <= All-API."""
        q = 2
        cfg = _make_stage1_config(
            api_input_price_per_1k=1.0,
            api_output_price_per_1k=2.0,
            subscription_monthly_fee=0.0,
            num_subscriptions=1,
            supported_models=[],
        )
        cost_calculator = CostCalculator(cfg["providers"], cfg.get("model_pricing"))

        # 2 days so quota reset logic is exercised.
        requests = [
            _make_request(1, day=0, request_tokens=50, response_tokens=500, model="m"),
            _make_request(2, day=0, request_tokens=50, response_tokens=10, model="m"),
            _make_request(3, day=0, request_tokens=50, response_tokens=100, model="m"),
            _make_request(4, day=1, request_tokens=50, response_tokens=300, model="m"),
            _make_request(5, day=1, request_tokens=50, response_tokens=20, model="m"),
            _make_request(6, day=1, request_tokens=50, response_tokens=200, model="m"),
        ]

        optimal = OfflineSimulator(
            requests,
            OptimalStrategy(cost_calculator, QuotaManager(daily_quota=q), cfg),
            cfg,
        ).run()
        greedy = OfflineSimulator(
            requests,
            GreedyOnlineStrategy(
                cost_calculator,
                QuotaManager(daily_quota=q),
                cfg,
                daily_quota=q,
                concurrency_limit=0,
            ),
            cfg,
        ).run()
        all_api = OfflineSimulator(
            requests,
            AllAPIStrategy(cost_calculator, QuotaManager(daily_quota=0), cfg),
            cfg,
        ).run()

        assert optimal.total_cost <= greedy.total_cost + 1e-12
        assert greedy.total_cost <= all_api.total_cost + 1e-12

    def test_subscription_cost_prorating_and_accounting(self) -> None:
        """Simulator should prorate subscription cost and keep total=subscription+api."""
        cfg = _make_stage1_config(
            api_input_price_per_1k=1.0,
            api_output_price_per_1k=1.0,
            subscription_monthly_fee=30.0,
            num_subscriptions=2,
            supported_models=[],
        )
        cost_calculator = CostCalculator(cfg["providers"], cfg.get("model_pricing"))

        # Requests across 3 days: day 0, 1, 2 -> num_days=3 in simulator.
        requests = [
            _make_request(1, day=0, request_tokens=10, response_tokens=10, model="m"),
            _make_request(2, day=2, request_tokens=10, response_tokens=10, model="m"),
        ]
        strategy = AllAPIStrategy(cost_calculator, QuotaManager(daily_quota=0), cfg)
        result = OfflineSimulator(requests, strategy, cfg).run()
        d = result.to_dict()

        expected_subscription = 30.0 * 2 * (3 / 30.0)  # monthly_fee * num_subscriptions * (days/30)
        assert (
            pytest.approx(d["costs"]["subscription"], rel=1e-12, abs=1e-12) == expected_subscription
        )
        assert (
            pytest.approx(d["costs"]["total"], rel=1e-12, abs=1e-12)
            == d["costs"]["subscription"] + d["costs"]["api"]
        )
