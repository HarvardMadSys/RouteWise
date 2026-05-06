"""Primal-Dual online routing strategies.

This module implements the Primal-Dual threshold-based strategies for online
request routing, as described in docs/algorithm/online.md.

Components:
- PrimalDualQuotaManager: Manages S_Q (daily quota) with exponential threshold
- CAPQConcurrencyManager: Manages S_C (concurrency) with value-density based eviction
- PrimalDualOnlineStrategy: Unified router using both managers

IMPORTANT: All value/duration estimation must use PREDICTORS, not ground truth.
The route() method must NOT access request.response_tokens or request.latency_seconds
for decision making. Ground truth is only used for:
1. Post-decision predictor updates
2. Final cost calculation when actually routing to API
"""

import logging
import math
from dataclasses import dataclass, field

from experiments.offline_stage.strategies.online.base import OnlineStrategy
from experiments.offline_stage.value_estimators import EMAOutputPredictor, OutputTokenPredictor
from rwsim.offline.schemas import Request, RoutingDecision

logger = logging.getLogger(__name__)


def _calculate_predicted_api_cost(
    config: dict,
    request: Request,
    predicted_output_tokens: float,
) -> float:
    """Calculate predicted API cost using only arrival-time information.

    This helper MUST NOT access request.response_tokens. It supports:
    - model_pricing (per 1M tokens), if configured for request.model.
    - providers (per 1K tokens) as a fallback, choosing the cheapest API provider.

    Args:
        config: Experiment configuration dictionary.
        request: Incoming request.
        predicted_output_tokens: Predicted output token count.

    Returns:
        Predicted API cost in dollars.
    """
    predicted_output_tokens = max(float(predicted_output_tokens), 0.0)

    model_pricing = config.get("model_pricing", {})
    if model_pricing and request.model and request.model in model_pricing:
        pricing = model_pricing[request.model]
        input_price_per_1m = float(pricing.get("input") or 0.0)
        output_price_per_1m = float(pricing.get("output") or 0.0)
        return (
            request.request_tokens / 1_000_000.0 * input_price_per_1m
            + predicted_output_tokens / 1_000_000.0 * output_price_per_1m
        )

    # Fallback to providers config (ProviderConfig objects)
    providers = config.get("providers", {})
    api_providers = [p for p in providers.values() if hasattr(p, "is_api") and p.is_api()]
    if api_providers:
        return min(
            request.request_tokens / 1000.0 * p.input_price_per_1k
            + predicted_output_tokens / 1000.0 * p.output_price_per_1k
            for p in api_providers
        )

    return 0.0


class EMADurationEstimator:
    """Simple EMA-based duration estimator for Stage 2.

    Estimates request processing duration using exponential moving average
    of historical durations, stratified by model.
    """

    def __init__(self, alpha: float = 0.1, default_duration: float = 5.0):
        """Initialize duration estimator.

        Args:
            alpha: EMA smoothing factor (higher = more weight on recent)
            default_duration: Default duration before warmup
        """
        self.alpha = alpha
        self.default_duration = default_duration
        self._ema_by_model: dict[str, float] = {}
        self._global_ema: float = default_duration
        self._count: int = 0

    def reset(self) -> None:
        """Reset estimator state."""
        self._ema_by_model = {}
        self._global_ema = self.default_duration
        self._count = 0

    def estimate(self, request: Request) -> float:
        """Estimate duration for request (before seeing actual duration).

        Args:
            request: Incoming request

        Returns:
            Estimated duration in seconds
        """
        if request.model and request.model in self._ema_by_model:
            return self._ema_by_model[request.model]
        if self._count > 0:
            return self._global_ema
        return self.default_duration

    def update(self, request: Request) -> None:
        """Update estimator with actual duration (post-decision).

        Args:
            request: Completed request with actual latency
        """
        actual = request.latency_seconds
        if actual <= 0:
            return

        # Update global EMA
        if self._count == 0:
            self._global_ema = actual
        else:
            self._global_ema = self.alpha * actual + (1 - self.alpha) * self._global_ema
        self._count += 1

        # Update per-model EMA
        if request.model:
            if request.model in self._ema_by_model:
                self._ema_by_model[request.model] = (
                    self.alpha * actual + (1 - self.alpha) * self._ema_by_model[request.model]
                )
            else:
                self._ema_by_model[request.model] = actual


# =============================================================================
# Part 1: S_Q - Quota Manager with Primal-Dual Threshold
# =============================================================================


@dataclass
class QuotaState:
    """State for daily quota tracking."""

    current_day: int = -1
    used: int = 0
    # Statistics for L/U estimation
    daily_costs: list[float] = field(default_factory=list)


class PrimalDualQuotaManager:
    """Manages S_Q (daily quota subscription) with Primal-Dual threshold.

    The threshold function is:
        psi(z) = L * (U/L)^z
    where z = used/Q is the fraction of quota consumed.

    This achieves competitive ratio of ln(U/L) + 1 for online knapsack.
    """

    def __init__(
        self,
        daily_quota: int,
        min_value: float = 0.0001,
        max_value: float = 0.01,
        supported_models: set[str] | None = None,
    ):
        """Initialize quota manager.

        Args:
            daily_quota: Daily quota limit Q
            min_value: Lower bound L for request values (5th percentile)
            max_value: Upper bound U for request values (95th percentile)
            supported_models: Set of compatible models (None = all models)
        """
        self.Q = daily_quota
        self.L = max(min_value, 1e-9)  # Avoid division by zero
        self.U = max(max_value, self.L)  # Ensure U >= L
        self.supported_models = supported_models

        # State
        self.state = QuotaState()

        logger.debug(
            f"PrimalDualQuotaManager initialized: Q={daily_quota}, L={self.L:.6f}, U={self.U:.6f}"
        )

    def reset(self) -> None:
        """Reset state for new simulation run."""
        self.state = QuotaState()

    def check_daily_reset(self, day: int) -> None:
        """Reset quota if new day."""
        if day != self.state.current_day:
            # Note: Disabled adaptive L/U update because it creates a feedback loop
            # where only accepted (expensive) requests update bounds, causing L to
            # increase over time and reject more requests. Fixed bounds are better.
            self.state.current_day = day
            self.state.used = 0
            self.state.daily_costs = []

    def _update_bounds_from_history(self) -> None:
        """Update L and U from historical costs (optional enhancement)."""
        if len(self.state.daily_costs) < 10:
            return  # Not enough data

        sorted_costs = sorted(self.state.daily_costs)
        n = len(sorted_costs)
        # 5th and 95th percentile
        new_L = sorted_costs[max(0, int(n * 0.05))]
        new_U = sorted_costs[min(n - 1, int(n * 0.95))]

        if new_L > 0 and new_U > new_L:
            self.L = new_L
            self.U = new_U
            logger.debug(f"Updated bounds: L={self.L:.6f}, U={self.U:.6f}")

    def get_threshold(self) -> float:
        """Get current threshold (shadow price) for S_Q.

        Returns:
            Threshold value psi(z), or inf if quota exhausted
        """
        if self.state.used >= self.Q:
            return float("inf")

        z = self.state.used / self.Q
        # Exponential threshold: L * (U/L)^z
        threshold = self.L * math.pow(self.U / self.L, z)
        return threshold

    def is_compatible(self, request: Request) -> bool:
        """Check if request model is compatible with S_Q."""
        if self.supported_models is None:
            return True
        return request.model in self.supported_models

    def has_quota(self) -> bool:
        """Check if quota is available."""
        return self.state.used < self.Q

    def consume_quota(self, cost: float = 0.0) -> None:
        """Consume one unit of quota."""
        self.state.used += 1
        if cost > 0:
            self.state.daily_costs.append(cost)

    def get_opportunity_cost(self, request: Request) -> float:
        """Get opportunity cost for using S_Q on this request.

        Returns:
            Threshold if available and compatible, inf otherwise
        """
        if not self.is_compatible(request):
            return float("inf")
        return self.get_threshold()


# =============================================================================
# Part 2: S_C - Concurrency Manager with CAPQ
# =============================================================================


@dataclass
class QueuedRequest:
    """Request waiting in CAPQ queue."""

    request: Request
    value: float  # API cost (savings if served)
    concurrency_cost: int  # Concurrency cost units consumed
    duration: float  # Estimated processing time
    arrival_time: float  # When it entered the queue
    deadline: float | None = None  # Optional SLO deadline

    @property
    def value_density(self) -> float:
        """Value per unit resource (concurrency-cost-second)."""
        denominator = self.concurrency_cost * self.duration
        if denominator <= 0:
            return float("inf")
        return self.value / denominator


class CAPQConcurrencyManager:
    """Manages S_C (concurrency subscription) with Cost-Aware Priority Queue.

    Uses value-density based admission and eviction:
    - Shadow price lambda = min value_density in queue when saturated
    - Evicts lowest density request when higher density arrives
    """

    def __init__(
        self,
        total_capacity: int,
        queue_capacity: int = 0,
        supported_models: dict[str, int] | None = None,
        default_concurrency_cost: int = 1,
    ):
        """Initialize concurrency manager.

        Args:
            total_capacity: Total concurrency allotment C_total
            queue_capacity: Queue size K (0 = no queue, immediate decision)
            supported_models: Dict of model -> concurrency cost
                (None = all models, cost=1)
            default_concurrency_cost: Default concurrency cost for unknown models
        """
        self.C_total = total_capacity
        self.K = queue_capacity
        self.supported_models = supported_models or {}
        self.default_concurrency_cost = default_concurrency_cost

        # State
        self.active_requests: list[tuple[float, int, int]] = []
        # (finish_time, req_id, concurrency_cost)
        self.queue: list[QueuedRequest] = []

        logger.debug(
            f"CAPQConcurrencyManager initialized: C_total={total_capacity}, K={queue_capacity}"
        )

    def reset(self) -> None:
        """Reset state for new simulation run."""
        self.active_requests = []
        self.queue = []

    def get_concurrency_cost(self, model: str | None) -> int:
        """Get concurrency cost for a model."""
        if model is None:
            return self.default_concurrency_cost
        return self.supported_models.get(model, self.default_concurrency_cost)

    def is_compatible(self, request: Request) -> bool:
        """Check if request model is compatible with S_C."""
        if not self.supported_models:
            return True  # All models allowed
        return request.model in self.supported_models

    def update_state(self, current_time: float) -> None:
        """Update state: remove finished requests."""
        self.active_requests = [
            (ft, rid, cost) for ft, rid, cost in self.active_requests if ft > current_time
        ]

    def get_active_concurrency_cost(self) -> int:
        """Get total concurrency cost of active requests."""
        return sum(cost for _, _, cost in self.active_requests)

    def has_capacity(self, concurrency_cost: int) -> bool:
        """Check if there's capacity for a request with given concurrency cost."""
        return self.get_active_concurrency_cost() + concurrency_cost <= self.C_total

    def get_congestion_price(self) -> float:
        """Get current congestion price (shadow price lambda).

        Returns:
            0 if capacity available or queue not full,
            min value_density in queue if saturated,
            inf if at capacity with no queue
        """
        # If we have capacity, price is 0
        if self.get_active_concurrency_cost() < self.C_total:
            return 0.0

        # If queue not full, price is 0
        if len(self.queue) < self.K:
            return 0.0

        # At capacity with no queue space - S_C is unavailable
        if not self.queue:
            return float("inf")

        # Queue full - price is min value density (eviction threshold)
        return min(qr.value_density for qr in self.queue)

    def get_opportunity_cost(self, request: Request, value: float, duration: float) -> float:
        """Get opportunity cost for using S_C on this request.

        Args:
            request: The request
            value: Estimated API cost (value)
            duration: Estimated processing duration

        Returns:
            lambda * concurrency_cost * duration, or inf if not compatible
        """
        if not self.is_compatible(request):
            return float("inf")

        concurrency_cost = self.get_concurrency_cost(request.model)
        lambda_t = self.get_congestion_price()

        return lambda_t * concurrency_cost * duration

    def can_admit(self, request: Request, value: float, duration: float) -> bool:
        """Check if request can be admitted (has capacity or can evict)."""
        if not self.is_compatible(request):
            return False

        concurrency_cost = self.get_concurrency_cost(request.model)

        # Direct execution possible
        if self.has_capacity(concurrency_cost):
            return True

        # Queue has space
        if len(self.queue) < self.K:
            return True

        # Can evict if our density is higher
        if self.queue:
            denominator = concurrency_cost * duration
            our_density = value / denominator if denominator > 0 else float("inf")
            min_density = min(qr.value_density for qr in self.queue)
            return our_density > min_density

        return False

    def admit(
        self, request: Request, value: float, duration: float, current_time: float
    ) -> tuple[str, Request | None]:
        """Admit request to S_C.

        Args:
            request: The request to admit
            value: Estimated API cost
            duration: Estimated processing duration
            current_time: Current timestamp

        Returns:
            (status, evicted_request) where status is 'immediate', 'queued', or 'rejected'
        """
        concurrency_cost = self.get_concurrency_cost(request.model)

        # Try immediate execution
        if self.has_capacity(concurrency_cost):
            finish_time = current_time + duration
            self.active_requests.append((finish_time, request.id, concurrency_cost))
            return "immediate", None

        # Try queue
        queued_req = QueuedRequest(
            request=request,
            value=value,
            concurrency_cost=concurrency_cost,
            duration=duration,
            arrival_time=current_time,
        )

        if len(self.queue) < self.K:
            self.queue.append(queued_req)
            return "queued", None

        # Try eviction
        if self.queue:
            min_idx = min(range(len(self.queue)), key=lambda i: self.queue[i].value_density)
            min_density = self.queue[min_idx].value_density

            if queued_req.value_density > min_density:
                evicted = self.queue.pop(min_idx)
                self.queue.append(queued_req)
                return "queued", evicted.request

        return "rejected", None

    def on_slot_free(self, current_time: float) -> QueuedRequest | None:
        """Handle slot becoming free - schedule next request from queue.

        Returns:
            The scheduled request, or None if queue empty
        """
        if not self.queue:
            return None

        # Feasibility-First HVF: pick highest value among feasible
        feasible = []
        for i, qr in enumerate(self.queue):
            if qr.deadline is None or current_time + qr.duration <= qr.deadline:
                feasible.append((i, qr))

        if not feasible:
            # All expired - clear queue (they should be offloaded)
            self.queue = []
            return None

        # Pick highest value
        best_idx, best_qr = max(feasible, key=lambda x: x[1].value)
        self.queue.pop(best_idx)

        # Add to active
        finish_time = current_time + best_qr.duration
        self.active_requests.append(
            (finish_time, best_qr.request.id, best_qr.concurrency_cost)
        )

        return best_qr


# =============================================================================
# Part 3: Unified Primal-Dual Online Strategy
# =============================================================================


class PrimalDualOnlineStrategy(OnlineStrategy):
    """Unified Primal-Dual online routing strategy.

    Evaluates net gain for each service and selects the best:
    - G_Q = value - theta_Q (for S_Q)
    - G_C = value - theta_C * concurrency_cost * duration (for S_C)
    - G_A = 0 (for API)

    Supports both Stage 1 (S_Q only) and Stage 2 (S_Q + S_C) through configuration.
    """

    def __init__(
        self,
        cost_calculator,
        quota_manager,
        config: dict,
        # S_Q parameters
        daily_quota: int = 0,
        sq_min_value: float = 0.0001,
        sq_max_value: float = 0.01,
        # S_C parameters
        concurrency_limit: int = 0,
        queue_capacity: int = 0,
        # Subscription costs (for reference, not used in threshold calculation)
        sq_monthly_fee: float = 20.0,
        sc_monthly_fee: float = 25.0,
        # Optional custom predictor for ablation studies
        output_predictor: OutputTokenPredictor | None = None,
    ):
        """Initialize Primal-Dual strategy.

        Args:
            cost_calculator: Cost calculator instance
            quota_manager: Base quota manager (from parent class)
            config: Configuration dictionary
            daily_quota: Daily quota Q for S_Q (0 to disable)
            sq_min_value: Lower bound L for S_Q threshold
            sq_max_value: Upper bound U for S_Q threshold
            concurrency_limit: Concurrency limit C_total for S_C (0 to disable)
            queue_capacity: Queue size K for S_C CAPQ
            sq_monthly_fee: Monthly fee for S_Q (informational)
            sc_monthly_fee: Monthly fee for S_C (informational)
            output_predictor: Custom predictor (uses EMAOutputPredictor if None).
                When using a QuantilePrediction-compatible predictor (e.g.,
                HistogramOutputPredictor), q50 is used as the point estimate.
        """
        super().__init__(
            cost_calculator,
            quota_manager,
            config,
            daily_quota=daily_quota,
            concurrency_limit=concurrency_limit,
            sq_monthly_fee=sq_monthly_fee,
            sc_monthly_fee=sc_monthly_fee,
        )

        # Initialize S_Q manager
        self.sq_manager: PrimalDualQuotaManager | None = None
        if daily_quota > 0:
            self.sq_manager = PrimalDualQuotaManager(
                daily_quota=daily_quota,
                min_value=sq_min_value,
                max_value=sq_max_value,
                supported_models=self.sq_supported_models if self.sq_supported_models else None,
            )

        # Initialize S_C manager
        self.sc_manager: CAPQConcurrencyManager | None = None
        if concurrency_limit > 0:
            self.sc_manager = CAPQConcurrencyManager(
                total_capacity=concurrency_limit,
                queue_capacity=queue_capacity,
                supported_models=self.sc_concurrency_costs if self.sc_concurrency_costs else None,
            )

        # Initialize predictors for online estimation (no data leakage)
        # Custom predictor for ablation studies (e.g., HistogramOutputPredictor)
        self.output_predictor = output_predictor or EMAOutputPredictor()
        self._use_custom_predictor = output_predictor is not None
        # EMA for duration estimation (always needed)
        self.ema_duration_estimator = EMADurationEstimator()

        # Track evicted request costs
        self._evicted_cost = 0.0

        predictor_name = type(self.output_predictor).__name__
        logger.info(
            f"PrimalDualOnlineStrategy initialized: "
            f"S_Q={'enabled' if self.sq_manager else 'disabled'}, "
            f"S_C={'enabled' if self.sc_manager else 'disabled'}, "
            f"predictor={predictor_name}"
        )

    @property
    def name(self) -> str:
        """Return strategy name with stage indicator."""
        # Determine predictor suffix for ablation identification
        predictor_suffix = ""
        if self._use_custom_predictor:
            predictor_name = type(self.output_predictor).__name__
            if "Histogram" in predictor_name:
                predictor_suffix = "-Hist"
            elif "EMA" not in predictor_name:
                predictor_suffix = f"-{predictor_name}"

        if self.sc_manager:
            return f"PrimalDual-Online-Stage2{predictor_suffix}"
        elif self.sq_manager:
            return f"PrimalDual-Online-Stage1{predictor_suffix}"
        return f"PrimalDual-Online-APIOnly{predictor_suffix}"

    def precompute(self, requests: list[Request]) -> None:
        """Reset state for new simulation run.

        Note: L/U bounds are passed in at construction time from the runner,
        which computes them from historical data. We do NOT re-estimate L/U
        inside the strategy to avoid information leakage (using ground truth
        response_tokens before decisions are made).
        """
        super().precompute(requests)

        # Reset evicted cost tracking
        self._evicted_cost = 0.0

        # Reset predictors
        self.output_predictor.reset()
        self.ema_duration_estimator.reset()

        if self.sq_manager:
            self.sq_manager.reset()
        if self.sc_manager:
            self.sc_manager.reset()

        # L/U bounds are set at construction time, no re-estimation here
        # to avoid information leakage from ground truth response_tokens

    def route(self, request: Request) -> RoutingDecision:
        """Route request using Primal-Dual unified decision.

        IMPORTANT: Decision is made using PREDICTED values only.
        Ground truth (response_tokens, latency) is only used for:
        1. Post-decision predictor updates
        2. Final cost calculation when actually routing to API

        Args:
            request: Incoming request

        Returns:
            Routing decision
        """
        current_time = request.timestamp

        # Update time-based state
        self._update_sq_state(request)
        if self.sc_enabled:
            self._update_sc_state(current_time)
            if self.sc_manager:
                self.sc_manager.update_state(current_time)
                # Schedule queued requests when capacity becomes free
                self._schedule_from_queue(current_time)

        # 1. Estimate value using predictor (NO DATA LEAKAGE)
        # MUST NOT use request.response_tokens here
        # Use q50 as point estimate for all predictors:
        # - HistogramOutputPredictor: q50 from histogram bins
        # - EMAOutputPredictor: q50 = mean (normal approximation)
        # When not warmed up, predict() returns a reasonable default q50.
        prediction = self.output_predictor.predict(request)
        value = _calculate_predicted_api_cost(self.config, request, prediction.q50)

        # For S_C: estimate duration using EMA (NO DATA LEAKAGE)
        # MUST NOT use request.latency_seconds here
        duration = self.ema_duration_estimator.estimate(request)

        # 2. Calculate net gains for each option

        # Option S_Q
        gain_q = float("-inf")
        if self.sq_manager:
            self.sq_manager.check_daily_reset(request.day)
            theta_q = self.sq_manager.get_opportunity_cost(request)
            if theta_q < float("inf"):
                gain_q = value - theta_q

        # Option S_C (only consider if actually admissible)
        gain_c = float("-inf")
        if self.sc_manager and self.sc_manager.can_admit(request, value, duration):
            theta_c_cost = self.sc_manager.get_opportunity_cost(request, value, duration)
            if theta_c_cost < float("inf"):
                gain_c = value - theta_c_cost

        # Option API
        gain_a = 0.0

        # 3. Select best option
        best_gain = max(gain_q, gain_c, gain_a)

        # 4. Update predictors AFTER decision (post-decision update)
        # This is where we can use ground truth
        self.output_predictor.update(request)
        self.ema_duration_estimator.update(request)

        # 5. Execute decision
        if best_gain == gain_q and gain_q > float("-inf"):
            # Route to S_Q
            self.sq_manager.consume_quota(value)
            self._use_sq()
            return RoutingDecision(
                request=request,
                provider="daily-quota",
                cost=0.0,
                quota_used=1,
                timestamp=request.timestamp,
            )

        elif best_gain == gain_c and gain_c > float("-inf"):
            # Route to S_C
            status, evicted = self.sc_manager.admit(request, value, duration, current_time)

            if status in ("immediate", "queued"):
                # Fix finish_time to use actual latency for simulation accuracy.
                # The estimated duration was used for decision-making (opportunity cost),
                # but slot occupancy in trace replay should reflect real completion time.
                actual_duration = request.latency_seconds
                actual_finish = current_time + actual_duration
                self.sc_manager.active_requests = [
                    (ft, rid, w) if rid != request.id else (actual_finish, rid, w)
                    for ft, rid, w in self.sc_manager.active_requests
                ]

                self.subscription_used += 1

                # Handle evicted request - route to API and count ACTUAL cost
                evicted_cost = 0.0
                if evicted is not None:
                    # Here we use actual cost because this is a real API call
                    evicted_cost = self._calculate_api_cost(evicted)
                    self.api_used += 1
                    self._evicted_cost += evicted_cost
                    logger.debug(
                        f"Evicted request {evicted.id} (value_density too low), "
                        f"API cost: ${evicted_cost:.4f}"
                    )

                return RoutingDecision(
                    request=request,
                    provider="concurrency",
                    cost=evicted_cost,  # Include evicted request's API cost
                    quota_used=0,
                    timestamp=request.timestamp,
                )
            # S_C rejected (at capacity, no queue) - fall back to S_Q if worthwhile
            if gain_q > gain_a:
                self.sq_manager.consume_quota(value)
                self._use_sq()
                return RoutingDecision(
                    request=request,
                    provider="daily-quota",
                    cost=0.0,
                    quota_used=1,
                    timestamp=request.timestamp,
                )

        # Route to API - use ACTUAL cost for billing
        actual_cost = self._calculate_api_cost(request)
        self.api_used += 1
        return RoutingDecision(
            request=request,
            provider="api",
            cost=actual_cost,
            quota_used=0,
            timestamp=request.timestamp,
        )

    def _schedule_from_queue(self, current_time: float) -> None:
        """Schedule queued requests when capacity becomes free.

        This method should be called after update_state to check if
        any previously blocked requests can now be scheduled.
        """
        if not self.sc_manager:
            return

        # Keep scheduling while we have capacity and queued requests
        while True:
            # Check if we have any capacity
            active_usage = self.sc_manager.get_active_concurrency_cost()
            if active_usage >= self.sc_manager.C_total:
                break  # No capacity

            # Try to schedule from queue
            scheduled = self.sc_manager.on_slot_free(current_time)
            if scheduled is None:
                break  # Queue empty or no feasible requests

            logger.debug(f"Scheduled queued request {scheduled.request.id} at time {current_time}")

    def get_total_cost(self) -> float:
        """Get total cost including evicted requests.

        Override to include evicted request costs in the total.
        """
        # Get base cost from parent (API costs from direct routing)
        base_cost = super().get_total_cost() if hasattr(super(), "get_total_cost") else 0.0

        # Add evicted request costs
        return base_cost + self._evicted_cost
