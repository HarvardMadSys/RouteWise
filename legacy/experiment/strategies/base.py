"""Base class for routing strategies."""

from abc import ABC, abstractmethod

from legacy.experiment.cost import CostCalculator
from legacy.experiment.data.schema import Request, RoutingDecision
from legacy.experiment.quota import QuotaManager


class RoutingStrategy(ABC):
    """Abstract base class for routing strategies.

    Attributes:
        cost_calculator: Cost calculator instance
        quota_manager: Quota manager instance
        config: Configuration dictionary
        subscription_used: Count of requests routed to subscription
        api_used: Count of requests routed to API
    """

    def __init__(self, cost_calculator: CostCalculator, quota_manager: QuotaManager, config: dict):
        """Initialize routing strategy.

        Args:
            cost_calculator: Cost calculator instance
            quota_manager: Quota manager instance
            config: Configuration dictionary
        """
        self.cost_calculator = cost_calculator
        self.quota_manager = quota_manager
        self.config = config

        # Statistics
        self.subscription_used = 0
        self.api_used = 0

    @abstractmethod
    def name(self) -> str:
        """Return strategy name.

        Returns:
            Strategy name
        """
        pass

    def precompute(self, requests: list[Request]) -> None:  # noqa: B027
        """Precompute routing decisions (for offline strategies).

        This is optional and only used by offline strategies like Optimal.
        Online strategies can leave this as a no-op.

        Args:
            requests: All requests in the dataset
        """
        pass

    @abstractmethod
    def route(self, request: Request) -> RoutingDecision:
        """Make routing decision for a request.

        Args:
            request: Request to route

        Returns:
            Routing decision
        """
        pass

    def reset(self) -> None:
        """Reset strategy state (for multiple runs)."""
        self.subscription_used = 0
        self.api_used = 0
        self.quota_manager.current_day = 0
        self.quota_manager.used_today = 0
        self.quota_manager.total_used = 0
        self.quota_manager.daily_usage = {}
