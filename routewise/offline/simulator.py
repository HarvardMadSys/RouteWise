"""Offline simulator for routing strategies."""

import logging
import time
from dataclasses import dataclass

from tqdm import tqdm

from routewise.offline.schemas import Request, RoutingDecision
from routewise.offline.strategy import RoutingStrategy

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Results from a simulation run.

    Attributes:
        strategy_name: Name of the strategy used
        total_cost: Total cost (subscription + API)
        subscription_cost: Fixed subscription cost
        api_cost: Variable API cost
        num_requests: Total number of requests
        num_days: Number of days in simulation
        quota_utilization: Average quota utilization
        subscription_requests: Number of requests routed to subscription
        api_requests: Number of requests routed to API
        runtime_seconds: Simulation runtime
    """

    strategy_name: str
    total_cost: float
    subscription_cost: float
    api_cost: float
    num_requests: int
    num_days: int
    quota_utilization: float
    subscription_requests: int
    api_requests: int
    runtime_seconds: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy": self.strategy_name,
            "costs": {
                "total": round(self.total_cost, 4),
                "subscription": round(self.subscription_cost, 4),
                "api": round(self.api_cost, 4),
            },
            "requests": {
                "total": self.num_requests,
                "subscription": self.subscription_requests,
                "api": self.api_requests,
            },
            "quota_utilization": round(self.quota_utilization, 4),
            "num_days": self.num_days,
            "runtime_seconds": round(self.runtime_seconds, 2),
        }


class OfflineSimulator:
    """Simulate routing strategies on historical data.

    Attributes:
        requests: List of historical requests
        strategy: Routing strategy to evaluate
        config: Configuration dictionary
    """

    def __init__(self, requests: list[Request], strategy: RoutingStrategy, config: dict):
        """Initialize simulator.

        Args:
            requests: List of requests to simulate
            strategy: Routing strategy to use
            config: Configuration dictionary
        """
        self.requests = requests
        self.strategy = strategy
        self.config = config

    def run(self) -> SimulationResult:
        """Run simulation and collect results.

        Returns:
            Simulation results
        """
        logger.info(
            f"Starting simulation with {self.strategy.name} strategy "
            f"on {len(self.requests)} requests"
        )

        start_time = time.time()

        # Reset strategy state
        self.strategy.reset()

        # Precompute for offline strategies (e.g., Optimal)
        self.strategy.precompute(self.requests)

        # Track costs
        api_cost = 0.0
        decisions: list[RoutingDecision] = []

        # Simulate each request with progress bar
        for request in tqdm(
            self.requests,
            desc=f"Routing ({self.strategy.name})",
            unit="req",
            leave=True,
            ncols=100,
        ):
            decision = self.strategy.route(request)
            decisions.append(decision)
            api_cost += decision.cost

        runtime = time.time() - start_time

        # Calculate subscription cost based on relative days in dataset
        if self.requests:
            first_day = self.requests[0].day
            last_day = self.requests[-1].day
            num_days = last_day - first_day + 1
        else:
            num_days = 0

        num_subscriptions = self.config["simulation"]["num_subscriptions"]

        subscription_provider_id = self.config["simulation"].get(
            "subscription_provider", "chutes-subscription"
        )
        subscription_provider = self.config["providers"][subscription_provider_id]

        # Prorate subscription cost based on actual days
        subscription_cost = (
            subscription_provider.monthly_fee * num_subscriptions * (num_days / 30.0)
        )

        # Finalize quota tracking (record last day) and calculate utilization
        self.strategy.quota_manager.finalize()
        quota_stats = self.strategy.quota_manager.get_statistics()
        quota_utilization = quota_stats.get("avg_utilization", 0.0)

        result = SimulationResult(
            strategy_name=self.strategy.name,
            total_cost=subscription_cost + api_cost,
            subscription_cost=subscription_cost,
            api_cost=api_cost,
            num_requests=len(self.requests),
            num_days=num_days,
            quota_utilization=quota_utilization,
            subscription_requests=self.strategy.subscription_used,
            api_requests=self.strategy.api_used,
            runtime_seconds=runtime,
        )

        logger.info(
            f"Simulation complete: "
            f"Total cost ${result.total_cost:.2f} "
            f"(subscription ${result.subscription_cost:.2f}, "
            f"API ${result.api_cost:.2f}), "
            f"Quota utilization {result.quota_utilization:.1%}, "
            f"Runtime {result.runtime_seconds:.2f}s"
        )

        return result
