"""All-API baseline strategy.

This strategy represents the scenario without any subscription service.
All requests are routed to the API provider (e.g., OpenAI).
Used as the primary baseline to measure cost savings from using subscriptions.
"""

from experiment.data.schema import Request, RoutingDecision
from experiment.strategies.base import RoutingStrategy


class AllAPIStrategy(RoutingStrategy):
    """Baseline: Route all requests to API provider.

    This represents the scenario without Chutes subscription.
    All requests use the pay-per-token API (e.g., OpenAI).

    For single model scenario:
    - No subscription cost ($0)
    - All requests pay API cost based on tokens

    This is the primary baseline to measure:
    - Cost savings from using subscription
    - ROI of subscription service
    """

    @property
    def name(self) -> str:
        """Return the name of this strategy."""
        return "All-API"

    def route(self, request: Request) -> RoutingDecision:
        """Route to API provider (never use subscription).

        Args:
            request: Incoming request

        Returns:
            Routing decision (always API, never subscription)
        """
        # Check if multi-model pricing is enabled
        model_pricing = self.config.get("model_pricing", {})
        if model_pricing and request.model:
            # Multi-model: calculate cost based on request's model
            cost = self.cost_calculator.calculate_cost_by_model(request)
            provider = "api"
        else:
            # Single model: use configured API provider
            api_provider = self.config.get("simulation", {}).get("default_api_fallback")

            if not api_provider:
                # Fallback: find first API provider
                api_providers = [
                    name for name, p in self.cost_calculator.providers.items() if p.is_api()
                ]
                if not api_providers:
                    raise ValueError("No API provider configured")
                api_provider = api_providers[0]

            cost = self.cost_calculator.calculate_api_cost(request, api_provider)
            provider = api_provider

        # Update statistics
        self.api_used += 1

        return RoutingDecision(
            request=request,
            provider=provider,
            cost=cost,
            quota_used=0,  # Never use subscription quota
            timestamp=request.timestamp,
        )
