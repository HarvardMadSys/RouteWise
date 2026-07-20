"""Exponential Moving Average (EMA) baseline predictor.

This module implements a simple EMA-based predictor for output tokens.
It serves as a fallback when the histogram predictor is not warmed up
or produces suspiciously low predictions.

Key features:
- Simple and robust
- Fast warmup
- Per-model tracking with global fallback
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

from experiments.offline_stage.value_estimators.base import (
    OutputTokenPredictor,
    QuantilePrediction,
)
from llm_routewise.schemas import Request

logger = logging.getLogger(__name__)


@dataclass
class EMAState:
    """State for EMA tracking.

    Attributes:
        mean: Current EMA of output tokens
        variance: Current EMA of squared deviation (for std estimation)
        count: Number of samples seen
    """

    mean: float = 0.0
    variance: float = 0.0
    count: int = 0

    def update(self, value: float, alpha: float) -> None:
        """Update EMA with new value.

        Args:
            value: New observation
            alpha: Smoothing factor (0 < alpha <= 1)
        """
        if self.count == 0:
            self.mean = value
            self.variance = 0.0
        else:
            delta = value - self.mean
            self.mean = self.mean + alpha * delta
            # Welford's online algorithm for variance
            self.variance = (1 - alpha) * (self.variance + alpha * delta * delta)
        self.count += 1

    @property
    def std(self) -> float:
        """Get standard deviation estimate."""
        return math.sqrt(max(self.variance, 0.0))


class EMAOutputPredictor(OutputTokenPredictor):
    """EMA-based output token predictor.

    Uses exponential moving average with per-model tracking.
    Estimates quantiles using normal approximation:
    - q10 = mean - 1.28 * std
    - q50 = mean
    - q90 = mean + 1.28 * std
    """

    def __init__(
        self,
        alpha: float = 0.1,
        min_samples_warmup: int = 20,
        default_output_tokens: float = 500.0,
    ):
        """Initialize EMA predictor.

        Args:
            alpha: EMA smoothing factor (higher = more weight to recent)
            min_samples_warmup: Minimum samples for warmup
            default_output_tokens: Default prediction when not warmed up
        """
        self.alpha = alpha
        self.min_samples_warmup = min_samples_warmup
        self.default_output_tokens = default_output_tokens

        # Per-model EMA states
        self.model_states: dict[str, EMAState] = defaultdict(EMAState)
        self.global_state = EMAState()

        # Calibration tracking
        self._calibration_counts = {
            "q10_below": 0,
            "q50_below": 0,
            "q90_below": 0,
            "total": 0,
        }

    def predict(self, request: Request) -> QuantilePrediction:
        """Predict output token quantiles using EMA + normal approximation.

        Args:
            request: The incoming request

        Returns:
            QuantilePrediction with q10, q50, q90 estimates
        """
        model = request.model or "_unknown_"

        # Get mean and std from most specific available level
        if model in self.model_states and self.model_states[model].count >= 10:
            state = self.model_states[model]
        elif self.global_state.count >= self.min_samples_warmup:
            state = self.global_state
        else:
            # Not warmed up - return defaults
            return QuantilePrediction(
                q10=self.default_output_tokens * 0.3,
                q50=self.default_output_tokens,
                q90=self.default_output_tokens * 2.0,
                is_warmed_up=False,
            )

        mean = state.mean
        std = max(state.std, mean * 0.1)  # Ensure some minimum spread

        # Normal approximation for quantiles
        # z-scores: q10=-1.28, q50=0, q90=1.28
        q10 = max(1.0, mean - 1.28 * std)
        q50 = max(1.0, mean)
        q90 = max(q50, mean + 1.28 * std)

        return QuantilePrediction(
            q10=q10,
            q50=q50,
            q90=q90,
            is_warmed_up=self.is_warmed_up,
        )

    def update(self, request: Request) -> None:
        """Update predictor with realized output tokens.

        Args:
            request: The completed request
        """
        if request.response_tokens <= 0:
            return

        output_tokens = float(request.response_tokens)
        model = request.model or "_unknown_"

        # Update calibration before updating state
        if self.is_warmed_up:
            pred = self.predict(request)
            self._update_calibration(pred, output_tokens)

        # Update EMA states
        self.model_states[model].update(output_tokens, self.alpha)
        self.global_state.update(output_tokens, self.alpha)

    def _update_calibration(self, pred: QuantilePrediction, actual: float) -> None:
        """Update calibration statistics."""
        self._calibration_counts["total"] += 1
        if actual <= pred.q10:
            self._calibration_counts["q10_below"] += 1
        if actual <= pred.q50:
            self._calibration_counts["q50_below"] += 1
        if actual <= pred.q90:
            self._calibration_counts["q90_below"] += 1

    def reset(self) -> None:
        """Reset predictor state."""
        self.model_states.clear()
        self.global_state = EMAState()
        self._calibration_counts = {
            "q10_below": 0,
            "q50_below": 0,
            "q90_below": 0,
            "total": 0,
        }

    @property
    def is_warmed_up(self) -> bool:
        """Check if predictor has sufficient samples."""
        return self.global_state.count >= self.min_samples_warmup

    def get_calibration_stats(self) -> dict:
        """Get calibration statistics."""
        total = self._calibration_counts["total"]
        if total == 0:
            return {"q10_coverage": 0.0, "q50_coverage": 0.0, "q90_coverage": 0.0, "total": 0}

        return {
            "q10_coverage": self._calibration_counts["q10_below"] / total,
            "q50_coverage": self._calibration_counts["q50_below"] / total,
            "q90_coverage": self._calibration_counts["q90_below"] / total,
            "total": total,
        }

    def get_mean_estimate(self, request: Request) -> float:
        """Get simple mean estimate for fallback.

        Args:
            request: The request

        Returns:
            Mean output token estimate
        """
        model = request.model or "_unknown_"

        if model in self.model_states and self.model_states[model].count >= 10:
            return self.model_states[model].mean
        elif self.global_state.count >= self.min_samples_warmup:
            return self.global_state.mean
        else:
            return self.default_output_tokens
