"""Base class for online routing strategies.

This module provides the foundation for online decision-making strategies
that must route requests without future knowledge. It supports both:
- Stage1: Daily quota subscription (S_Q) + API (S_A)
- Stage2: Daily quota (S_Q) + Concurrency subscription (S_C) + API (S_A)

The framework uses strict gating to ensure clean separation between stages.
"""

import logging
from abc import abstractmethod

from legacy.experiment.cost import CostCalculator
from legacy.experiment.data.schema import Request, RoutingDecision
from legacy.experiment.quota import QuotaManager
from legacy.experiment.strategies.base import RoutingStrategy

logger = logging.getLogger(__name__)


class OnlineStrategy(RoutingStrategy):
    """Base class for online routing strategies.

    Online strategies make irrevocable routing decisions without knowledge
    of future requests. This base class provides:
    - Strict gating between Stage1 (S_Q only) and Stage2 (S_Q + S_C)
    - Concurrency state simulation for offline trace replay
    - Interpretable threshold computation

    Attributes:
        sq_enabled: Whether daily quota subscription (S_Q) is enabled
        sc_enabled: Whether concurrency subscription (S_C) is enabled
        daily_quota: Daily quota limit for S_Q
        concurrency_limit: Concurrency limit for S_C
        thresholds: Computed base thresholds for subscription resources
    """

    def __init__(
        self,
        cost_calculator: CostCalculator,
        quota_manager: QuotaManager,
        config: dict,
        # Stage1 resource: Daily quota subscription (S_Q)
        daily_quota: int = 0,
        sq_monthly_fee: float = 20.0,
        # Stage2 resource: Concurrency subscription (S_C)
        # Set to 0 to disable (strict Stage1 mode)
        concurrency_limit: int = 0,
        sc_monthly_fee: float = 25.0,
    ):
        """Initialize online strategy.

        Args:
            cost_calculator: Cost calculator instance
            quota_manager: Quota manager instance
            config: Configuration dictionary
            daily_quota: Daily quota for S_Q (0 to disable)
            sq_monthly_fee: Monthly fee for S_Q subscription
            concurrency_limit: Concurrency limit for S_C (0 to disable)
            sc_monthly_fee: Monthly fee for S_C subscription
        """
        super().__init__(cost_calculator, quota_manager, config)

        # Stage1: S_Q configuration
        self.daily_quota = daily_quota
        self.sq_enabled = daily_quota > 0
        self.sq_monthly_fee = sq_monthly_fee

        # Stage2: S_C configuration (strict gating)
        self.concurrency_limit = concurrency_limit
        self.sc_enabled = concurrency_limit > 0
        self.sc_monthly_fee = sc_monthly_fee

        # Online state tracking
        self._current_day = -1
        self._daily_used = 0

        # Concurrency state (only initialized if S_C enabled - strict gating)
        if self.sc_enabled:
            self._active_requests: list[tuple[float, int, int]] = []
            # (finish_time, request_id, slots_used)
        else:
            self._active_requests = None  # type: ignore

        # Load model compatibility from config
        self._init_model_compatibility()

        # Compute interpretable thresholds
        self._init_thresholds()

        logger.info(
            f"OnlineStrategy initialized: S_Q={self.sq_enabled} (quota={daily_quota}), "
            f"S_C={self.sc_enabled} (C={concurrency_limit})"
        )

    def _init_model_compatibility(self) -> None:
        """Initialize model compatibility sets from config."""
        subscriptions = self.config.get("subscriptions", {})

        # S_Q (e.g., Chutes) compatibility
        chutes_config = subscriptions.get("chutes", {})
        self.sq_supported_models: set[str] = set(chutes_config.get("supported_models", []))

        # S_C compatibility (only if enabled)
        # Read from active_sc_subscription (set by experiment runner)
        if self.sc_enabled:
            sc_config = self.config.get("active_sc_subscription", {})
            featherless_models = sc_config.get("supported_models", {})

            self.sc_supported_models: set[str] = set()
            self.sc_multipliers: dict[str, int] = {}

            if isinstance(featherless_models, dict):
                for model, props in featherless_models.items():
                    self.sc_supported_models.add(model)
                    self.sc_multipliers[model] = props.get("multiplier", 1)
            elif isinstance(featherless_models, list):
                self.sc_supported_models = set(featherless_models)
                for model in featherless_models:
                    self.sc_multipliers[model] = 1
        else:
            # Strict gating: no S_C model tracking in Stage1
            self.sc_supported_models = set()
            self.sc_multipliers = {}

    def _init_thresholds(self) -> None:
        """Compute interpretable base thresholds.

        Thresholds represent the average cost per unit of subscription resource:
        - S_Q threshold = monthly_fee / (days_per_month * daily_quota)
          This is the cost per quota unit
        - S_C threshold = monthly_fee / (days_per_month * seconds_per_day * C)
          This is the cost per second of concurrency usage
        """
        self.thresholds: dict[str, float] = {}

        if self.sq_enabled:
            # Cost per quota unit
            self.thresholds["sq"] = self.sq_monthly_fee / (30.0 * self.daily_quota)
            logger.debug(f"S_Q threshold: ${self.thresholds['sq']:.6f}/request")

        if self.sc_enabled:
            # Cost per second of concurrency
            # monthly_fee / (30 days * 86400 seconds * C slots)
            self.thresholds["sc"] = self.sc_monthly_fee / (30.0 * 86400.0 * self.concurrency_limit)
            logger.debug(f"S_C threshold: ${self.thresholds['sc']:.9f}/second")

    # ========== S_Q State Management ==========

    def _update_sq_state(self, request: Request) -> None:
        """Update S_Q state for new day if needed."""
        if not self.sq_enabled:
            return

        if request.day != self._current_day:
            self._current_day = request.day
            self._daily_used = 0
            self.quota_manager.reset_if_new_day(request.day)

    def _sq_available(self, request: Request) -> bool:
        """Check if S_Q quota is available for the request.

        Args:
            request: The request to check

        Returns:
            True if S_Q can be used for this request
        """
        if not self.sq_enabled:
            return False

        # Check model compatibility
        if self.sq_supported_models and request.model not in self.sq_supported_models:
            return False

        # Check quota availability
        return self._daily_used < self.daily_quota

    def _use_sq(self) -> None:
        """Consume one unit of S_Q quota."""
        if not self.sq_enabled:
            return

        self._daily_used += 1
        self.quota_manager.use_quota()
        self.subscription_used += 1

    # ========== S_C State Management (Stage2 only) ==========

    def _update_sc_state(self, current_time: float) -> None:
        """Update S_C concurrency state by removing finished requests.

        This is a strict Stage2 operation - no-op if S_C is disabled.

        Args:
            current_time: Current timestamp in seconds
        """
        if not self.sc_enabled:
            return  # Strict gating: no-op for Stage1

        # Remove finished requests
        self._active_requests = [
            (ft, req_id, slots) for ft, req_id, slots in self._active_requests if ft > current_time
        ]

    def _sc_available(self, request: Request) -> bool:
        """Check if S_C has available slots for the request.

        This is a strict Stage2 operation - always returns False if S_C disabled.

        Args:
            request: The request to check

        Returns:
            True if S_C can be used for this request
        """
        if not self.sc_enabled:
            return False  # Strict gating

        # Check model compatibility
        if self.sc_supported_models and request.model not in self.sc_supported_models:
            return False

        # Get slots needed for this request
        slots_needed = self.sc_multipliers.get(request.model, 1)

        # Check concurrency availability
        current_usage = sum(slots for _, _, slots in self._active_requests)
        return current_usage + slots_needed <= self.concurrency_limit

    def _use_sc(self, request: Request) -> None:
        """Occupy S_C slot(s) for the request.

        This is a strict Stage2 operation - no-op if S_C is disabled.

        Args:
            request: The request being served
        """
        if not self.sc_enabled:
            return  # Strict gating

        slots_needed = self.sc_multipliers.get(request.model, 1)
        duration = request.latency_seconds
        finish_time = request.timestamp + duration

        self._active_requests.append((finish_time, request.id, slots_needed))
        self.subscription_used += 1

    def _get_sc_current_usage(self) -> int:
        """Get current S_C slot usage.

        Returns:
            Number of slots currently in use (0 if S_C disabled)
        """
        if not self.sc_enabled:
            return 0
        return sum(slots for _, _, slots in self._active_requests)

    # ========== Cost Calculation ==========

    def _calculate_api_cost(self, request: Request) -> float:
        """Calculate API cost for a request.

        Args:
            request: The request to calculate cost for

        Returns:
            API cost in dollars
        """
        model_pricing = self.config.get("model_pricing", {})

        if model_pricing and request.model:
            return self.cost_calculator.calculate_cost_by_model(request)
        else:
            _, cost = self.cost_calculator.get_cheapest_api_provider(request)
            return cost

    # ========== Abstract Methods ==========

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass

    @abstractmethod
    def route(self, request: Request) -> RoutingDecision:
        """Make routing decision for a request.

        Subclasses must implement the actual routing logic.

        Args:
            request: Request to route

        Returns:
            Routing decision
        """
        pass

    # ========== Lifecycle ==========

    def precompute(self, requests: list[Request]) -> None:
        """No precomputation for online strategies.

        Online strategies make decisions without future knowledge.
        This method resets the internal state for a fresh simulation run.

        Args:
            requests: List of requests (not used for precomputation)
        """
        # Reset state for fresh run
        self._current_day = -1
        self._daily_used = 0

        if self.sc_enabled:
            self._active_requests = []

        stage = "Stage2" if self.sc_enabled else "Stage1"
        logger.info(
            f"{self.name}: online strategy ready ({stage}, "
            f"S_Q={self.daily_quota}, S_C={self.concurrency_limit})"
        )

    def reset(self) -> None:
        """Reset strategy state for multiple runs."""
        super().reset()

        self._current_day = -1
        self._daily_used = 0

        if self.sc_enabled:
            self._active_requests = []
