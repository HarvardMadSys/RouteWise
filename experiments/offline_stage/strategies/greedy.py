"""Greedy strategy for online routing.

This is a simple online baseline that uses subscription quota first,
then falls back to API when quota is exhausted.
"""

from rwsim.offline.schemas import Request, RoutingDecision
from rwsim.offline.strategy import RoutingStrategy


class GreedyStrategy(RoutingStrategy):
    """Greedy baseline: Use subscription quota first, then API.

    This is a simple online algorithm that makes greedy decisions:
    - If quota is available, use subscription (cost = $0)
    - If quota is exhausted, use API (pay per token)

    Characteristics:
    - No future knowledge required (online algorithm)
    - Simple to implement
    - May waste quota on cheap requests
    - Serves as a strong practical baseline

    Expected performance:
    - Better than All-API (uses some subscription)
    - Worse than Optimal (doesn't prioritize expensive requests)
    - Competitive ratio: typically 1.2-1.5x optimal
    """

    @property
    def name(self) -> str:
        """Return the name of this strategy."""
        return "Greedy"

    def route(self, request: Request) -> RoutingDecision:
        """Route to subscription if quota available, otherwise API.

        Args:
            request: Incoming request

        Returns:
            Routing decision
        """
        # Update quota for new day if needed
        self.quota_manager.reset_if_new_day(request.day)

        # Greedy decision: use subscription if available
        if self.quota_manager.has_quota():
            # Use subscription
            self.quota_manager.use_quota()
            self.subscription_used += 1

            return RoutingDecision(
                request=request,
                provider="subscription",
                cost=0.0,  # Subscription has zero marginal cost
                quota_used=1,
                timestamp=request.timestamp,
            )
        else:
            # Quota exhausted, use API
            # Check if multi-model pricing is enabled
            model_pricing = self.config.get("model_pricing", {})
            if model_pricing and request.model:
                cost = self.cost_calculator.calculate_cost_by_model(request)
                provider = "api"
            else:
                api_provider = self.config.get("simulation", {}).get("default_api_fallback")

                if not api_provider:
                    # Fallback: find first API provider
                    api_providers = [
                        name
                        for name, provider in self.cost_calculator.providers.items()
                        if provider.is_api()
                    ]
                    if not api_providers:
                        raise ValueError("No API provider configured")
                    api_provider = api_providers[0]

                cost = self.cost_calculator.calculate_api_cost(request, api_provider)
                provider = api_provider

            self.api_used += 1

            return RoutingDecision(
                request=request,
                provider=provider,
                cost=cost,
                quota_used=0,
                timestamp=request.timestamp,
            )
