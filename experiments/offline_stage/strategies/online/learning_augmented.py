"""Learning-Augmented Primal-Dual online routing strategies.

This module implements learning-augmented versions of the Primal-Dual threshold
strategy, using quantile predictors for conservative value estimation.

Key algorithms:
- LearningAugmentedPrimalDualStrategy: Stage 1 LA-PD with v_t^{LCB} + fallback
- LearningAugmentedUnifiedStrategy: Stage 2 unified router with duration UCB

These strategies are kept with the offline-stage research harness and are not
part of the public simulator policy surface.
"""

import logging
import math
from dataclasses import dataclass

from experiments.offline_stage.strategies.online.base import OnlineStrategy
from experiments.offline_stage.value_estimators import (
    EMAOutputPredictor,
    HistogramDurationPredictor,
    HistogramOutputPredictor,
    OutputTokenPredictor,
    QuantilePrediction,
)
from llm_routewise.offline.cost import CostCalculator
from llm_routewise.offline.quota import QuotaManager
from llm_routewise.offline.schemas import Request, RoutingDecision

logger = logging.getLogger(__name__)


def _require_quantile_prediction(prediction: object) -> QuantilePrediction:
    """Return a quantile prediction or fail fast for point predictors."""

    if not isinstance(prediction, QuantilePrediction):
        raise TypeError(
            "learning-augmented strategies require a quantile output predictor; "
            "point predictors such as BucketMeanOutputPredictor do not expose q10/q50/q90"
        )
    return prediction


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

    providers = config.get("providers", {})
    api_providers = [p for p in providers.values() if p.is_api()]
    if not api_providers:
        return 0.0

    return min(
        request.request_tokens / 1000.0 * p.input_price_per_1k
        + predicted_output_tokens / 1000.0 * p.output_price_per_1k
        for p in api_providers
    )


@dataclass
class LAPDConfig:
    """Configuration for Learning-Augmented Primal-Dual strategy.

    Attributes:
        quantile_for_lcb: Which quantile to use for LCB (default: 0.10)
        epsilon_fallback: Minimum LCB value before triggering fallback (None = auto)
        use_ema_fallback: Whether to use EMA as fallback when predictor not ready
        warmup_use_greedy: Whether to use greedy during warmup period
    """

    quantile_for_lcb: float = 0.10
    epsilon_fallback: float | None = None  # Auto-detect from data (P1 of costs)
    use_ema_fallback: bool = True
    warmup_use_greedy: bool = True


class LearningAugmentedPrimalDualStrategy(OnlineStrategy):
    """Learning-Augmented Primal-Dual strategy for Stage 1.

    Uses quantile prediction for conservative value estimation:
    - v_t^{LCB} = Cost_API(n_in, L10) where L10 is predicted 10th percentile
    - Route to subscription iff used < Q and v_t^{LCB} > threshold

    Includes safeguard/fallback:
    - If predictor not warmed up: use greedy or EMA-based estimation
    - If v_t^{LCB} < epsilon: fall back to EMA estimate

    This strategy achieves:
    - Better savings ratio than vanilla Primal-Dual (EMA) when predictions are good
    - Graceful degradation when predictions are poor (via fallback)
    """

    def __init__(
        self,
        cost_calculator: CostCalculator,
        quota_manager: QuotaManager,
        config: dict,
        # S_Q parameters
        daily_quota: int = 0,
        sq_min_value: float = 0.0001,
        sq_max_value: float = 0.01,
        # S_C parameters (Stage 2, set to 0 for Stage 1)
        concurrency_limit: int = 0,
        # LA-PD specific
        lapd_config: LAPDConfig | None = None,
        output_predictor: OutputTokenPredictor | None = None,
        # Subscription fees
        sq_monthly_fee: float = 20.0,
        sc_monthly_fee: float = 25.0,
    ):
        """Initialize LA-PD strategy.

        Args:
            cost_calculator: Cost calculator instance
            quota_manager: Base quota manager
            config: Configuration dictionary
            daily_quota: Daily quota Q for S_Q
            sq_min_value: Lower bound L for threshold
            sq_max_value: Upper bound U for threshold
            concurrency_limit: Concurrency limit for S_C (0 to disable)
            lapd_config: LA-PD specific configuration
            output_predictor: Custom predictor (uses HistogramOutputPredictor if None)
            sq_monthly_fee: Monthly fee for S_Q
            sc_monthly_fee: Monthly fee for S_C
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

        # LA-PD configuration
        self.lapd_config = lapd_config or LAPDConfig()

        # Threshold parameters
        self.L = max(sq_min_value, 1e-9)
        self.U = max(sq_max_value, self.L)

        # Initialize predictors
        self.output_predictor = output_predictor or HistogramOutputPredictor()
        self.ema_predictor = EMAOutputPredictor()  # Fallback

        # Initialize epsilon (will be auto-detected in precompute if None)
        self._epsilon = self.lapd_config.epsilon_fallback or 1e-9

        # Statistics tracking
        self._stats = {
            "total_decisions": 0,
            "used_predictor": 0,
            "used_fallback_warmup": 0,
            "used_fallback_epsilon": 0,
            "subscription_routed": 0,
            "api_routed": 0,
        }

        logger.info(
            f"LearningAugmentedPrimalDualStrategy initialized: "
            f"Q={daily_quota}, L={self.L:.6f}, U={self.U:.6f}, "
            f"quantile={self.lapd_config.quantile_for_lcb}"
        )

    @property
    def name(self) -> str:
        """Return strategy name."""
        q_pct = int(self.lapd_config.quantile_for_lcb * 100)
        if self.sc_enabled:
            return f"LA-PD-P{q_pct}-Stage2"
        return f"LA-PD-P{q_pct}-Stage1"

    def _get_threshold(self, z: float) -> float:
        """Compute Primal-Dual threshold.

        Args:
            z: Fraction of quota used (0 to 1)

        Returns:
            Threshold value psi(z) = L * (U/L)^z
        """
        if z >= 1.0:
            return float("inf")
        return self.L * math.pow(self.U / self.L, z)

    def _select_quantile_value(self, prediction: QuantilePrediction) -> float:
        """Select output token estimate based on configured quantile.

        Supports linear interpolation for quantiles between q10/q50/q90.

        Args:
            prediction: Quantile prediction with q10, q50, q90

        Returns:
            Interpolated output token estimate
        """
        q = self.lapd_config.quantile_for_lcb

        if q <= 0.10:
            return prediction.q10
        elif q >= 0.90:
            return prediction.q90
        elif q <= 0.50:
            # Linear interpolation between q10 and q50
            alpha = (q - 0.10) / (0.50 - 0.10)
            return prediction.q10 + alpha * (prediction.q50 - prediction.q10)
        else:
            # Linear interpolation between q50 and q90
            alpha = (q - 0.50) / (0.90 - 0.50)
            return prediction.q50 + alpha * (prediction.q90 - prediction.q50)

    def _calculate_value_lcb(self, request: Request, prediction: QuantilePrediction) -> float:
        """Calculate conservative value estimate using LCB.

        Args:
            request: The incoming request
            prediction: Quantile prediction for output tokens

        Returns:
            v_t^{LCB} = API cost using predicted quantile output tokens
        """
        # Select output tokens based on configured quantile
        output_tokens = self._select_quantile_value(prediction)
        return _calculate_predicted_api_cost(self.config, request, output_tokens)

    def _calculate_value_ema(self, request: Request) -> float:
        """Calculate value estimate using EMA predictor (fallback).

        This method avoids data leakage by only using predicted output tokens,
        never the actual response_tokens.

        Args:
            request: The incoming request

        Returns:
            API cost using EMA-estimated output tokens
        """
        ema_output = self.ema_predictor.get_mean_estimate(request)
        return _calculate_predicted_api_cost(self.config, request, ema_output)

    def precompute(self, requests: list[Request]) -> None:
        """Reset state for new simulation run.

        Note: L/U bounds are passed in at construction time from the runner,
        which computes them from historical data. We do NOT re-estimate L/U
        inside the strategy to avoid information leakage.
        """
        super().precompute(requests)

        # Reset predictors
        self.output_predictor.reset()
        self.ema_predictor.reset()

        # Reset stats
        self._stats = {
            "total_decisions": 0,
            "used_predictor": 0,
            "used_fallback_warmup": 0,
            "used_fallback_epsilon": 0,
            "subscription_routed": 0,
            "api_routed": 0,
        }

        # Set epsilon based on L (no information leakage)
        # epsilon is a fraction of L to detect suspiciously low predictions
        if self.lapd_config.epsilon_fallback is None:
            # Default: epsilon = L * 0.1 (10% of lower bound)
            self._epsilon = max(self.L * 0.1, 1e-9)
            logger.debug(f"Auto-set epsilon_fallback: {self._epsilon:.9f} (0.1 * L)")
        else:
            self._epsilon = self.lapd_config.epsilon_fallback

        logger.info(
            f"{self.name}: ready with L={self.L:.6f}, U={self.U:.6f}, epsilon={self._epsilon:.9f}"
        )

    def route(self, request: Request) -> RoutingDecision:
        """Route request using LA-PD decision rule.

        Decision rule:
        1. Predict output token quantiles
        2. Compute v_t^{LCB} using q10
        3. If not warmed up or v_t^{LCB} < epsilon: use fallback
        4. Route to subscription iff used < Q and v_t^{LCB} > threshold

        Args:
            request: Incoming request

        Returns:
            Routing decision
        """
        # Update time-based state
        self._update_sq_state(request)
        if self.sc_enabled:
            self._update_sc_state(request.timestamp)

        self._stats["total_decisions"] += 1

        # Step 1: Get prediction
        prediction = _require_quantile_prediction(self.output_predictor.predict(request))

        # Step 2: Compute value estimate with safeguard/fallback
        value_estimate: float

        if not prediction.is_warmed_up:
            # Fallback: predictor not warmed up
            self._stats["used_fallback_warmup"] += 1

            if self.lapd_config.warmup_use_greedy:
                # During warmup, use greedy (always use subscription if available)
                value_estimate = float("inf")
            else:
                # Always use EMA fallback to avoid data leakage
                value_estimate = self._calculate_value_ema(request)
        else:
            # Compute LCB value
            value_lcb = self._calculate_value_lcb(request, prediction)

            if value_lcb < self._epsilon:
                # Fallback: suspiciously low prediction
                # Use EMA fallback to avoid data leakage (never use _calculate_api_cost)
                self._stats["used_fallback_epsilon"] += 1
                value_estimate = self._calculate_value_ema(request)
            else:
                # Use predictor
                value_estimate = value_lcb
                self._stats["used_predictor"] += 1

        # Step 3: Compute threshold
        z = self._daily_used / self.daily_quota if self.daily_quota > 0 else 1.0
        threshold = self._get_threshold(z)

        # Step 4: Make routing decision
        route_to_subscription = False

        if self._sq_available(request) and value_estimate > threshold:
            route_to_subscription = True

        # Step 5: Execute decision and update predictors
        # IMPORTANT: Update predictors AFTER decision (post-decision update)
        self.output_predictor.update(request)
        self.ema_predictor.update(request)

        if route_to_subscription:
            self._use_sq()
            self._stats["subscription_routed"] += 1

            return RoutingDecision(
                request=request,
                provider="daily-quota",
                cost=0.0,
                quota_used=1,
                timestamp=request.timestamp,
            )
        else:
            cost = self._calculate_api_cost(request)
            self.api_used += 1
            self._stats["api_routed"] += 1

            return RoutingDecision(
                request=request,
                provider="api",
                cost=cost,
                quota_used=0,
                timestamp=request.timestamp,
            )

    def get_stats(self) -> dict:
        """Get strategy statistics.

        Returns:
            Dict with decision statistics and calibration metrics
        """
        stats = dict(self._stats)
        stats["predictor_calibration"] = self.output_predictor.get_calibration_stats()
        stats["ema_calibration"] = self.ema_predictor.get_calibration_stats()
        return stats


class LearningAugmentedUnifiedStrategy(OnlineStrategy):
    """Learning-Augmented Unified strategy for Stage 2.

    Extends LA-PD to handle both S_Q (quota) and S_C (concurrency) using:
    - v_t^{LCB} for conservative value estimation
    - p_t^{UCB} for conservative duration estimation

    Decision rule:
    - G_Q = v_t^{LCB} - theta_Q (net gain for S_Q)
    - G_C = v_t^{LCB} - lambda * w * p_t^{UCB} (net gain for S_C)
    - G_A = 0 (baseline for API)
    - Choose argmax{G_Q, G_C, G_A}
    """

    def __init__(
        self,
        cost_calculator: CostCalculator,
        quota_manager: QuotaManager,
        config: dict,
        # S_Q parameters
        daily_quota: int = 0,
        sq_min_value: float = 0.0001,
        sq_max_value: float = 0.01,
        # S_C parameters
        concurrency_limit: int = 0,
        queue_capacity: int = 0,
        # LA-PD specific
        lapd_config: LAPDConfig | None = None,
        # Subscription fees
        sq_monthly_fee: float = 20.0,
        sc_monthly_fee: float = 25.0,
    ):
        """Initialize LA-PD Unified strategy.

        Args:
            cost_calculator: Cost calculator instance
            quota_manager: Base quota manager
            config: Configuration dictionary
            daily_quota: Daily quota Q for S_Q
            sq_min_value: Lower bound L for S_Q threshold
            sq_max_value: Upper bound U for S_Q threshold
            concurrency_limit: Concurrency limit C for S_C
            queue_capacity: Queue capacity K for S_C
            lapd_config: LA-PD specific configuration
            sq_monthly_fee: Monthly fee for S_Q
            sc_monthly_fee: Monthly fee for S_C
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

        # LA-PD configuration
        self.lapd_config = lapd_config or LAPDConfig()

        # Threshold parameters
        self.L = max(sq_min_value, 1e-9)
        self.U = max(sq_max_value, self.L)

        # Initialize predictors
        self.output_predictor = HistogramOutputPredictor()
        self.duration_predictor = HistogramDurationPredictor()
        self.ema_predictor = EMAOutputPredictor()

        # S_C state (simple tracking without queue for now)
        self._sc_active_requests: list[tuple[float, int, int]] = []
        # (finish_time, req_id, concurrency_cost)

        # Initialize epsilon (will be auto-detected in precompute if None)
        self._epsilon = self.lapd_config.epsilon_fallback or 1e-9

        # Statistics
        self._stats = {
            "total_decisions": 0,
            "sq_routed": 0,
            "sc_routed": 0,
            "api_routed": 0,
        }

        logger.info(
            f"LearningAugmentedUnifiedStrategy initialized: "
            f"Q={daily_quota}, C={concurrency_limit}"
        )

    @property
    def name(self) -> str:
        """Return strategy name."""
        q_pct = int(self.lapd_config.quantile_for_lcb * 100)
        return f"LA-PD-Unified-P{q_pct}"

    def _get_sq_threshold(self, z: float) -> float:
        """Compute S_Q threshold."""
        if z >= 1.0:
            return float("inf")
        return self.L * math.pow(self.U / self.L, z)

    def _get_sc_congestion_price(self) -> float:
        """Get S_C congestion price (simplified: 0 if capacity is available)."""
        current_usage = sum(cost for _, _, cost in self._sc_active_requests)
        if current_usage < self.concurrency_limit:
            return 0.0
        # When saturated, return a base price
        return self.L

    def _update_sc_active(self, current_time: float) -> None:
        """Update S_C state by removing finished requests."""
        self._sc_active_requests = [
            (ft, rid, cost) for ft, rid, cost in self._sc_active_requests if ft > current_time
        ]

    def _sc_has_capacity(self, concurrency_cost: int) -> bool:
        """Check if S_C has capacity."""
        current_usage = sum(cost for _, _, cost in self._sc_active_requests)
        return current_usage + concurrency_cost <= self.concurrency_limit

    def _select_quantile_value(self, prediction: QuantilePrediction) -> float:
        """Select output token estimate based on configured quantile."""
        q = self.lapd_config.quantile_for_lcb

        if q <= 0.10:
            return prediction.q10
        elif q >= 0.90:
            return prediction.q90
        elif q <= 0.50:
            alpha = (q - 0.10) / (0.50 - 0.10)
            return prediction.q10 + alpha * (prediction.q50 - prediction.q10)
        else:
            alpha = (q - 0.50) / (0.90 - 0.50)
            return prediction.q50 + alpha * (prediction.q90 - prediction.q50)

    def _calculate_value_lcb(self, request: Request, prediction: QuantilePrediction) -> float:
        """Calculate conservative value using LCB."""
        output_tokens = self._select_quantile_value(prediction)
        return _calculate_predicted_api_cost(self.config, request, output_tokens)

    def _calculate_value_ema(self, request: Request) -> float:
        """Calculate value estimate using EMA predictor (fallback, no data leakage)."""
        ema_output = self.ema_predictor.get_mean_estimate(request)
        return _calculate_predicted_api_cost(self.config, request, ema_output)

    def precompute(self, requests: list[Request]) -> None:
        """Reset state for new simulation run.

        Note: L/U bounds are passed in at construction time from the runner.
        We do NOT re-estimate L/U inside the strategy to avoid information leakage.
        """
        super().precompute(requests)

        self.output_predictor.reset()
        self.duration_predictor.reset()
        self.ema_predictor.reset()
        self._sc_active_requests = []

        self._stats = {
            "total_decisions": 0,
            "sq_routed": 0,
            "sc_routed": 0,
            "api_routed": 0,
        }

        # Set epsilon based on L (no information leakage)
        if self.lapd_config.epsilon_fallback is None:
            self._epsilon = max(self.L * 0.1, 1e-9)
        else:
            self._epsilon = self.lapd_config.epsilon_fallback

    def route(self, request: Request) -> RoutingDecision:
        """Route request using unified LA-PD decision.

        Args:
            request: Incoming request

        Returns:
            Routing decision
        """
        current_time = request.timestamp

        # Update state
        self._update_sq_state(request)
        self._update_sc_active(current_time)

        self._stats["total_decisions"] += 1

        # Get predictions
        output_pred = _require_quantile_prediction(self.output_predictor.predict(request))
        duration_pred = self.duration_predictor.predict(request, output_pred.q90)

        # Calculate value LCB (using EMA fallback to avoid data leakage)
        if output_pred.is_warmed_up:
            value_lcb = self._calculate_value_lcb(request, output_pred)
            if value_lcb < self._epsilon:
                # Suspiciously low prediction, use EMA fallback
                value_lcb = self._calculate_value_ema(request)
        else:
            # Predictor not warmed up, use EMA fallback
            value_lcb = self._calculate_value_ema(request)

        # Calculate gains
        # G_Q: net gain for S_Q
        z = self._daily_used / self.daily_quota if self.daily_quota > 0 else 1.0
        theta_q = self._get_sq_threshold(z)
        gain_q = value_lcb - theta_q if self._sq_available(request) else float("-inf")

        # G_C: net gain for S_C
        gain_c = float("-inf")
        concurrency_cost = self.sc_concurrency_costs.get(request.model, 1) if self.sc_enabled else 1

        if (
            self.sc_enabled
            and self._sc_has_capacity(concurrency_cost)
            and (request.model in self.sc_supported_models or not self.sc_supported_models)
        ):
            lambda_c = self._get_sc_congestion_price()
            opportunity_cost = lambda_c * concurrency_cost * duration_pred.duration_ucb
            gain_c = value_lcb - opportunity_cost

        # G_A: baseline for API
        gain_a = 0.0

        # Choose best option
        best_gain = max(gain_q, gain_c, gain_a)

        # Update predictors (post-decision)
        self.output_predictor.update(request)
        self.duration_predictor.update(request)
        self.ema_predictor.update(request)

        # Execute decision
        if best_gain == gain_q and gain_q > float("-inf"):
            # Route to S_Q
            self._use_sq()
            self._stats["sq_routed"] += 1
            return RoutingDecision(
                request=request,
                provider="daily-quota",
                cost=0.0,
                quota_used=1,
                timestamp=request.timestamp,
            )

        elif best_gain == gain_c and gain_c > float("-inf"):
            # Route to S_C
            # Use actual latency for slot tracking (not UCB estimate).
            # UCB is used for decision-making, but slot occupancy in trace replay
            # should reflect real completion time.
            finish_time = current_time + request.latency_seconds
            self._sc_active_requests.append((finish_time, request.id, concurrency_cost))
            self.subscription_used += 1
            self._stats["sc_routed"] += 1
            return RoutingDecision(
                request=request,
                provider="concurrency",
                cost=0.0,
                quota_used=0,
                timestamp=request.timestamp,
            )

        else:
            # Route to API
            cost = self._calculate_api_cost(request)
            self.api_used += 1
            self._stats["api_routed"] += 1
            return RoutingDecision(
                request=request,
                provider="api",
                cost=cost,
                quota_used=0,
                timestamp=request.timestamp,
            )

    def get_stats(self) -> dict:
        """Get strategy statistics."""
        stats = dict(self._stats)
        stats["output_calibration"] = self.output_predictor.get_calibration_stats()
        return stats
