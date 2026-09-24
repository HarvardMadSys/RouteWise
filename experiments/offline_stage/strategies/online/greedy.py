"""Greedy online routing strategies.

This module implements greedy decision-making strategies that use subscription
resources first (FCFS - First Come First Serve) before falling back to API.

Supports both Stage1 (S_Q only) and Stage2 (S_Q + S_C) through configuration.
"""

import logging

from experiments.offline_stage.strategies.online.base import OnlineStrategy
from llm_routewise.offline.schemas import Request, RoutingDecision

logger = logging.getLogger(__name__)


class GreedyOnlineStrategy(OnlineStrategy):
    """Greedy online strategy: Use subscription resources first, then API.

    Decision priority (greedy FCFS):
    - Stage1: S_Q (if available) -> API
    - Stage2: S_Q (if available) -> S_C (if available) -> API

    This is a practical baseline showing what happens when we simply use
    subscription resources whenever available, without any optimization.

    Characteristics:
    - No future knowledge required (true online algorithm)
    - Simple to implement and understand
    - May waste quota on low-value requests
    - Competitive ratio can be unbounded in adversarial cases

    Expected performance:
    - Better than All-API (uses subscription when available)
    - Worse than Optimal (doesn't prioritize expensive requests)
    - Practical competitive ratio: typically 1.1-1.5x optimal
    """

    @property
    def name(self) -> str:
        """Return strategy name with stage indicator."""
        if self.sc_enabled:
            return "Greedy-Online-Stage2"
        return "Greedy-Online-Stage1"

    def route(self, request: Request) -> RoutingDecision:
        """Route request using greedy FCFS policy.

        Args:
            request: Incoming request

        Returns:
            Routing decision
        """
        # Update time-based state
        self._update_sq_state(request)

        if self.sc_enabled:
            self._update_sc_state(request.timestamp)

        # Greedy priority: S_Q > S_C > API

        # 1. Try S_Q (daily quota subscription)
        if self._sq_available(request):
            self._use_sq()
            return RoutingDecision(
                request=request,
                provider="daily-quota",
                cost=0.0,
                quota_used=1,
                timestamp=request.timestamp,
            )

        # 2. Try S_C (concurrency subscription) - Stage2 only
        # Strict gating: _sc_available() returns False if sc_enabled=False
        if self._sc_available(request):
            self._use_sc(request)
            return RoutingDecision(
                request=request,
                provider="concurrency",
                cost=0.0,
                quota_used=0,
                timestamp=request.timestamp,
            )

        # 3. Fallback to API
        cost = self._calculate_api_cost(request)
        self.api_used += 1

        return RoutingDecision(
            request=request,
            provider="api",
            cost=cost,
            quota_used=0,
            timestamp=request.timestamp,
        )


class GreedyCostAwareStrategy(OnlineStrategy):
    """Greedy strategy with simple cost-awareness.

    Similar to GreedyOnlineStrategy but with a twist:
    - Only uses subscription for requests above a minimum API cost threshold
    - This helps avoid wasting quota on very cheap requests

    Decision logic:
    - If api_cost >= min_cost_threshold AND subscription available -> use subscription
    - Else -> use API

    This is still greedy (no optimization), but slightly smarter about
    not wasting quota on trivially cheap requests.
    """

    def __init__(self, *args, min_cost_threshold: float = 0.0001, **kwargs):
        """Initialize cost-aware greedy strategy.

        Args:
            min_cost_threshold: Minimum API cost to consider using subscription
                               Default: $0.0001 (0.01 cents)
        """
        super().__init__(*args, **kwargs)
        self.min_cost_threshold = min_cost_threshold

    @property
    def name(self) -> str:
        """Return strategy name with stage indicator."""
        if self.sc_enabled:
            return "Greedy-CostAware-Stage2"
        return "Greedy-CostAware-Stage1"

    def route(self, request: Request) -> RoutingDecision:
        """Route request using cost-aware greedy policy.

        Args:
            request: Incoming request

        Returns:
            Routing decision
        """
        # Update time-based state
        self._update_sq_state(request)

        if self.sc_enabled:
            self._update_sc_state(request.timestamp)

        # Calculate API cost first
        api_cost = self._calculate_api_cost(request)

        # Only consider subscription if cost is above threshold
        if api_cost >= self.min_cost_threshold:
            # Greedy priority: S_Q > S_C

            # 1. Try S_Q
            if self._sq_available(request):
                self._use_sq()
                return RoutingDecision(
                    request=request,
                    provider="daily-quota",
                    cost=0.0,
                    quota_used=1,
                    timestamp=request.timestamp,
                )

            # 2. Try S_C (Stage2 only)
            if self._sc_available(request):
                self._use_sc(request)
                return RoutingDecision(
                    request=request,
                    provider="concurrency",
                    cost=0.0,
                    quota_used=0,
                    timestamp=request.timestamp,
                )

        # Use API (either cost too low or no subscription available)
        self.api_used += 1

        return RoutingDecision(
            request=request,
            provider="api",
            cost=api_cost,
            quota_used=0,
            timestamp=request.timestamp,
        )
