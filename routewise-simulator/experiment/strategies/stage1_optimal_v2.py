"""Stage 1 Optimal Strategy with Fixed-Window Subscriptions.

This implements the offline optimal routing strategy for Stage 1 as described
in docs/algorithm/offline_stage1_new.md.

Key features:
- Multiple subscription plans (OpenAI, Anthropic, Z.ai/GLM)
- Fixed time windows (e.g., 3h for OpenAI, 5h for Anthropic)
- Model-to-plan compatibility constraints
- Greedy selection within each (plan, window) group
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from experiment.data.schema import Request, RoutingDecision
from experiment.strategies.base import RoutingStrategy
from experiment.window_quota import WindowQuotaManager

logger = logging.getLogger(__name__)


@dataclass
class Stage1OptimalResult:
    """Result of Stage 1 optimal routing computation."""

    total_requests: int = 0
    subscription_requests: int = 0
    api_requests: int = 0
    total_api_cost: float = 0.0
    total_savings: float = 0.0
    by_plan: dict = field(default_factory=dict)


class Stage1OptimalStrategy(RoutingStrategy):
    """Offline optimal strategy for Stage 1 fixed-window subscriptions.

    Algorithm:
    1. Group requests by (plan, window) based on model compatibility and arrival time
    2. For each (plan, window) group:
       - Sort requests by API cost (descending)
       - Select top-Q requests for subscription (Q = quota_per_window)
       - Remaining requests go to API

    This is optimal because:
    - Windows are disjoint (no quota carry-over)
    - Plans are independent (each request maps to at most one plan)
    - Within each (plan, window), it's a cardinality-constrained selection problem
    """

    def __init__(self, cost_calculator, quota_manager, config, **kwargs):
        """Initialize strategy.

        Args:
            cost_calculator: CostCalculator instance
            quota_manager: QuotaManager instance (legacy, may be unused)
            config: Experiment configuration
            **kwargs: Additional arguments including:
                - active_plans: List of plan IDs to use (overrides config)
        """
        super().__init__(cost_calculator, quota_manager, config, **kwargs)

        # Initialize WindowQuotaManager from plans config
        plans_config = config.get("plans", {})
        if not plans_config:
            raise ValueError("No 'plans' configuration found. Check experiment.yaml.")

        # Get active plans from kwargs or config
        active_plans = kwargs.get("active_plans") or config.get("active_plans")

        # Use first request timestamp as reference (will be set in precompute)
        self.window_manager = WindowQuotaManager.from_config(
            plans_config, reference_time=0.0, active_plans=active_plans
        )

        # Precomputed assignments: request_id -> ("subscription", plan_id) or ("api", None)
        self.assignments: dict[int, tuple[str, str | None]] = {}

        # Statistics
        self.result: Stage1OptimalResult | None = None

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "Stage1-Optimal"

    def precompute(self, requests: list[Request]) -> None:
        """Precompute optimal routing using (plan, window) grouping.

        Args:
            requests: All requests in the dataset
        """
        if not requests:
            return

        logger.info(f"Precomputing Stage 1 optimal routing for {len(requests)} requests")

        # Set reference time to earliest request
        min_timestamp = min(r.timestamp for r in requests)
        self.window_manager.reference_time = min_timestamp
        logger.info(f"Reference time set to {min_timestamp}")

        # Calculate API costs for all requests
        model_pricing = self.config.get("model_pricing", {})
        request_costs = {}
        for r in requests:
            if r.model and r.model in model_pricing:
                pricing = model_pricing[r.model]
                cost = (
                    r.request_tokens / 1_000_000.0 * pricing["input"]
                    + r.response_tokens / 1_000_000.0 * pricing["output"]
                )
            else:
                # Use default API provider pricing
                provider = self.config["providers"].get(
                    self.config["simulation"].get("default_api_fallback", "openai-chatgpt")
                )
                if provider:
                    cost = (
                        r.request_tokens / 1000.0 * provider.input_price_per_1k
                        + r.response_tokens / 1000.0 * provider.output_price_per_1k
                    )
                else:
                    cost = 0.0
            request_costs[r.id] = cost

        # Group requests by (plan, window)
        # Key: (plan_id, window_index), Value: list of (request, cost)
        groups: dict[tuple[str, int], list[tuple[Request, float]]] = defaultdict(list)
        api_only_requests: list[tuple[Request, float]] = []

        for r in requests:
            plan_id = self.window_manager.get_plan_for_model(r.model) if r.model else None

            if plan_id is None:
                # No compatible plan, must use API
                api_only_requests.append((r, request_costs[r.id]))
            else:
                window_idx = self.window_manager.get_window_index(plan_id, r.timestamp)
                groups[(plan_id, window_idx)].append((r, request_costs[r.id]))

        logger.info(
            f"Grouped into {len(groups)} (plan, window) groups, "
            f"{len(api_only_requests)} API-only requests"
        )

        # Process each (plan, window) group
        total_subscription = 0
        total_api_from_groups = 0
        total_savings = 0.0
        by_plan_stats = defaultdict(lambda: {"subscription": 0, "api": 0, "savings": 0.0})

        for (plan_id, _window_idx), group_requests in groups.items():
            quota = self.window_manager.get_quota(plan_id)

            # Sort by cost descending
            sorted_requests = sorted(group_requests, key=lambda x: x[1], reverse=True)

            # Select top-Q for subscription
            for i, (r, cost) in enumerate(sorted_requests):
                if i < quota:
                    # Use subscription
                    self.assignments[r.id] = ("subscription", plan_id)
                    total_subscription += 1
                    total_savings += cost
                    by_plan_stats[plan_id]["subscription"] += 1
                    by_plan_stats[plan_id]["savings"] += cost
                else:
                    # Use API
                    self.assignments[r.id] = ("api", None)
                    total_api_from_groups += 1
                    by_plan_stats[plan_id]["api"] += 1

        # Assign API-only requests
        total_api_cost = 0.0
        for r, cost in api_only_requests:
            self.assignments[r.id] = ("api", None)
            total_api_cost += cost

        # Add API costs from overflow requests
        for r in requests:
            if self.assignments.get(r.id, ("api", None))[0] == "api":
                total_api_cost += request_costs[r.id]

        # Recalculate total API cost correctly
        total_api_cost = sum(
            request_costs[r.id]
            for r in requests
            if self.assignments.get(r.id, ("api", None))[0] == "api"
        )

        # Store result
        self.result = Stage1OptimalResult(
            total_requests=len(requests),
            subscription_requests=total_subscription,
            api_requests=len(api_only_requests) + total_api_from_groups,
            total_api_cost=total_api_cost,
            total_savings=total_savings,
            by_plan=dict(by_plan_stats),
        )

        logger.info(
            f"Precomputation complete: {total_subscription} subscription, "
            f"{len(api_only_requests) + total_api_from_groups} API"
        )
        logger.info(f"Total API cost: ${total_api_cost:.2f}, Total savings: ${total_savings:.2f}")

        # Log per-plan statistics
        for plan_id, stats in by_plan_stats.items():
            logger.info(
                f"  {plan_id}: {stats['subscription']} sub, {stats['api']} api, "
                f"${stats['savings']:.2f} saved"
            )

    def route(self, request: Request) -> RoutingDecision:
        """Route request using precomputed assignment.

        Args:
            request: Request to route

        Returns:
            Routing decision
        """
        assignment = self.assignments.get(request.id, ("api", None))
        provider_type, plan_id = assignment

        if provider_type == "subscription":
            self.subscription_used += 1
            return RoutingDecision(
                request=request,
                provider=f"subscription:{plan_id}" if plan_id else "subscription",
                cost=0.0,
                quota_used=1,
                timestamp=request.timestamp,
            )
        else:
            # Calculate API cost
            model_pricing = self.config.get("model_pricing", {})
            if request.model and request.model in model_pricing:
                cost = self.cost_calculator.calculate_cost_by_model(request)
            else:
                _, cost = self.cost_calculator.get_cheapest_api_provider(request)

            self.api_used += 1
            return RoutingDecision(
                request=request,
                provider="api",
                cost=cost,
                quota_used=0,
                timestamp=request.timestamp,
            )

    def get_result_summary(self) -> dict:
        """Get summary of optimal routing result.

        Returns:
            Dictionary with result summary
        """
        if self.result is None:
            return {}

        return {
            "total_requests": self.result.total_requests,
            "subscription_requests": self.result.subscription_requests,
            "api_requests": self.result.api_requests,
            "total_api_cost": self.result.total_api_cost,
            "total_savings": self.result.total_savings,
            "by_plan": self.result.by_plan,
        }
