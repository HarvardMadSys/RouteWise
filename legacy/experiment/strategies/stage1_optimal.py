"""Optimal routing strategy using NumPy acceleration."""

import logging

import numpy as np

from legacy.experiment.data.schema import Request, RoutingDecision
from legacy.experiment.strategies.base import RoutingStrategy

logger = logging.getLogger(__name__)


class OptimalStrategy(RoutingStrategy):
    """Offline optimal strategy using NumPy vectorization.

    This strategy has perfect future knowledge and computes the globally
    optimal routing that minimizes total cost. It serves as an upper bound
    for comparing online strategies.

    The algorithm:
    1. For each day, calculate the API cost for each request
    2. Select the top K eligible requests with highest API cost to use subscription
       (eligibility is determined by the subscription provider's supported_models list)
    3. Route remaining requests to the cheapest API

    This is essentially solving a daily knapsack problem where:
    - Items = requests
    - Capacity = daily quota
    - Value = API cost savings

    Attributes:
        assignments: Precomputed routing assignments (request_id -> provider)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assignments: dict[int, str] = {}

    @property
    def name(self) -> str:
        """Return strategy name.

        Returns:
            Strategy name
        """
        return "Optimal"

    def precompute(self, requests: list[Request]) -> None:
        """Precompute optimal routing using NumPy vectorization.

        This is the core of the Optimal strategy: we can see all future
        requests, so we can make globally optimal decisions.

        Args:
            requests: All requests in the dataset
        """
        if not requests:
            return

        logger.info(f"Precomputing optimal routing for {len(requests)} requests")

        # Extract data to NumPy arrays for vectorized operations
        request_ids = np.array([r.id for r in requests])
        days = np.array([r.day for r in requests])
        request_tokens = np.array([r.request_tokens for r in requests])
        response_tokens = np.array([r.response_tokens for r in requests])
        models = [r.model for r in requests]

        # Respect subscription model compatibility if configured.
        # If supported_models is empty, treat all models as eligible.
        subscriptions = self.config.get("subscriptions", {})
        chutes_config = subscriptions.get("chutes", {})
        supported_models_list = chutes_config.get("supported_models", [])
        supported_models = set(supported_models_list) if supported_models_list else None
        if supported_models:
            eligible_mask = np.array(
                [(m in supported_models) if m else False for m in models], dtype=bool
            )
        else:
            eligible_mask = np.ones(len(models), dtype=bool)

        # Calculate costs (supports multi-model pricing)
        costs = self._calculate_costs_vectorized(request_tokens, response_tokens, models)

        # Process each day independently
        unique_days = np.unique(days)
        logger.info(f"Processing {len(unique_days)} days")

        for day in unique_days:
            # Use boolean indexing to filter requests for this day
            day_mask = days == day
            day_request_ids = request_ids[day_mask]

            # Default: all requests route to API.
            for req_id in day_request_ids:
                self.assignments[int(req_id)] = "api"

            # Only eligible requests can be routed to subscription.
            eligible_day_mask = day_mask & eligible_mask
            eligible_costs = costs[eligible_day_mask]
            eligible_request_ids = request_ids[eligible_day_mask]
            if eligible_request_ids.size == 0:
                continue

            # Key optimization: use argpartition instead of full sort
            # argpartition is O(n), sort is O(n log n)
            quota = self.quota_manager.total_daily_quota

            if len(eligible_costs) <= quota:
                # All eligible requests can use subscription
                top_k_indices = np.arange(len(eligible_costs))
            else:
                # Find top K requests with highest cost
                # argpartition partitions array into two parts:
                # - First K elements are the largest (but not sorted)
                # - Remaining elements are smaller
                top_k_indices = np.argpartition(eligible_costs, -quota)[-quota:]

            # Assign subscription to top K requests
            for idx in top_k_indices:
                req_id = eligible_request_ids[idx]
                self.assignments[int(req_id)] = "subscription"

        logger.info(
            f"Precomputation complete: "
            f"{sum(1 for v in self.assignments.values() if v == 'subscription')} "
            f"subscription, "
            f"{sum(1 for v in self.assignments.values() if v == 'api')} API"
        )

    def _calculate_costs_vectorized(
        self,
        request_tokens: np.ndarray,
        response_tokens: np.ndarray,
        models: list[str | None] | None = None,
    ) -> np.ndarray:
        """Vectorized cost calculation using NumPy.

        Calculates API cost for all requests. Supports multi-model pricing
        when model_pricing is configured and models are provided.

        Args:
            request_tokens: Array of input token counts
            response_tokens: Array of output token counts
            models: Optional list of model names for per-model pricing

        Returns:
            Array of API costs
        """
        n = len(request_tokens)
        costs = np.zeros(n)

        # Check if multi-model pricing is available
        model_pricing = self.config.get("model_pricing", {})

        if model_pricing and models:
            # Multi-model pricing: calculate cost per model (no default fallback)
            for i in range(n):
                model = models[i]
                if not model:
                    raise ValueError(f"Request at index {i} has no model specified")

                pricing = model_pricing.get(model)
                if pricing is None:
                    raise ValueError(
                        f"Model '{model}' not found in model_pricing. "
                        f"Please add pricing in config/experiment.yaml. "
                        f"Available: {list(model_pricing.keys())}"
                    )

                # Pricing is per 1M tokens
                input_price = pricing.get("input")
                output_price = pricing.get("output")
                if input_price is None or output_price is None:
                    raise ValueError(f"Model '{model}' has incomplete pricing")

                costs[i] = (
                    request_tokens[i] / 1_000_000.0 * input_price
                    + response_tokens[i] / 1_000_000.0 * output_price
                )
        else:
            # Single-model pricing: use default API provider
            api_provider_id = self.config["simulation"].get(
                "default_api_fallback", "openai-chatgpt"
            )
            provider = self.config["providers"][api_provider_id]

            input_price = provider.input_price_per_1k
            output_price = provider.output_price_per_1k

            # Vectorized calculation (all requests at once)
            costs = request_tokens / 1000.0 * input_price + response_tokens / 1000.0 * output_price

        return costs

    def route(self, request: Request) -> RoutingDecision:
        """Route request using precomputed assignment.

        Args:
            request: Request to route

        Returns:
            Routing decision
        """
        # Reset quota if entering new day
        self.quota_manager.reset_if_new_day(request.day)

        # Use precomputed assignment (keyed by request.id for uniqueness)
        provider_type = self.assignments.get(request.id, "api")

        if provider_type == "subscription":
            # Use subscription
            self.quota_manager.use_quota()
            self.subscription_used += 1

            return RoutingDecision(
                request=request,
                provider="subscription",
                cost=0.0,  # Marginal cost is 0
                quota_used=1,
                timestamp=request.timestamp,
            )
        else:
            # Use API with model-specific pricing
            model_pricing = self.config.get("model_pricing", {})
            if model_pricing and request.model:
                cost = self.cost_calculator.calculate_cost_by_model(request)
                provider = "api"
            else:
                provider, cost = self.cost_calculator.get_cheapest_api_provider(request)
            self.api_used += 1

            return RoutingDecision(
                request=request,
                provider=provider,
                cost=cost,
                quota_used=0,
                timestamp=request.timestamp,
            )
