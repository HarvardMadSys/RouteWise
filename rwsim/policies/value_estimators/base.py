"""Base interfaces and dataclasses for online predictors.

This module defines the interfaces for predictors used in learning-augmented
online routing strategies. Predictors must only use features observable at
request arrival time; ground-truth values are used only in post-decision updates.

Key interfaces:
- OutputTokenPredictor: Predicts output token quantiles (Stage 1)
- DurationPredictor: Predicts processing duration UCB (Stage 2)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from rwsim.schemas import Request


@dataclass
class QuantilePrediction:
    """Prediction result containing multiple quantiles.

    Attributes:
        q10: 10th percentile (conservative lower bound)
        q50: 50th percentile (median, point estimate)
        q90: 90th percentile (upper bound)
        is_warmed_up: Whether the predictor has sufficient samples
    """

    q10: float
    q50: float
    q90: float
    is_warmed_up: bool = True

    @property
    def lcb(self) -> float:
        """Lower confidence bound (conservative estimate)."""
        return self.q10

    @property
    def ucb(self) -> float:
        """Upper confidence bound (conservative high estimate)."""
        return self.q90

    @property
    def median(self) -> float:
        """Point estimate."""
        return self.q50


@dataclass
class DurationPrediction:
    """Prediction result for processing duration.

    Attributes:
        ttft_seconds: Time to first token (seconds)
        tps: Tokens per second (decode throughput)
        duration_ucb: Upper confidence bound on total duration
        is_warmed_up: Whether the predictor has sufficient samples
    """

    ttft_seconds: float
    tps: float
    duration_ucb: float
    is_warmed_up: bool = True


@dataclass
class PredictionContext:
    """Context for making predictions.

    Contains only features observable at request arrival time.
    Ground-truth response_tokens and latency_ms are NOT included.

    Attributes:
        request_tokens: Input token count
        model: Model name (categorical)
        timestamp: Request timestamp (seconds from epoch)
        hour_of_day: Hour of day (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        input_bin: Binned input token count (for histogram lookup)
    """

    request_tokens: int
    model: str | None
    timestamp: int
    hour_of_day: int
    day_of_week: int
    input_bin: int

    @classmethod
    def from_request(cls, request: Request) -> "PredictionContext":
        """Create context from a request using only arrival-time features.

        Args:
            request: The incoming request

        Returns:
            PredictionContext with observable features only
        """
        dt = datetime.fromtimestamp(request.timestamp, tz=timezone.utc)
        hour_of_day = dt.hour
        day_of_week = dt.weekday()

        # Log-spaced binning for input tokens
        input_bin = _compute_input_bin(request.request_tokens)

        return cls(
            request_tokens=request.request_tokens,
            model=request.model,
            timestamp=request.timestamp,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            input_bin=input_bin,
        )


def _compute_input_bin(tokens: int, num_bins: int = 20) -> int:
    """Compute log-spaced bin index for token count.

    Bins are: [0, 2), [2, 4), [4, 8), [8, 16), ..., [2^19, inf)

    Args:
        tokens: Token count
        num_bins: Number of bins

    Returns:
        Bin index (0 to num_bins-1)
    """
    if tokens <= 0:
        return 0
    # Log2-based binning
    import math

    bin_idx = int(math.log2(max(tokens, 1)))
    return min(bin_idx, num_bins - 1)


class OutputTokenPredictor(ABC):
    """Abstract base class for output token predictors.

    Predictors estimate the distribution of output tokens for a request
    using only features available at arrival time.
    """

    @abstractmethod
    def predict(self, request: Request) -> QuantilePrediction:
        """Predict output token quantiles.

        Args:
            request: The incoming request (only arrival-time features are used)

        Returns:
            QuantilePrediction with q10, q50, q90 estimates
        """
        pass

    @abstractmethod
    def update(self, request: Request) -> None:
        """Update predictor with realized output tokens.

        This method is called AFTER the routing decision is made,
        using the ground-truth response_tokens from the completed request.

        Args:
            request: The completed request with response_tokens available
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset predictor state for a new simulation run."""
        pass

    @property
    @abstractmethod
    def is_warmed_up(self) -> bool:
        """Check if predictor has sufficient samples for reliable predictions."""
        pass

    def get_calibration_stats(self) -> dict:
        """Get calibration statistics for evaluation.

        Returns:
            Dict with coverage rates and error metrics
        """
        return {}


class DurationPredictor(ABC):
    """Abstract base class for processing duration predictors (Stage 2).

    Predictors estimate the upper confidence bound on processing duration
    using TTFT and TPS statistics.
    """

    @abstractmethod
    def predict(self, request: Request, predicted_output_tokens: float) -> DurationPrediction:
        """Predict processing duration.

        Args:
            request: The incoming request
            predicted_output_tokens: Predicted output token count (from OutputTokenPredictor)

        Returns:
            DurationPrediction with duration_ucb estimate
        """
        pass

    @abstractmethod
    def update(self, request: Request) -> None:
        """Update predictor with realized latency stats.

        Uses ttft_ms and latency_ms from the completed request.

        Args:
            request: The completed request with latency data
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset predictor state for a new simulation run."""
        pass


class CombinedPredictor:
    """Combined predictor for both output tokens and duration.

    Convenience class that wraps both predictors for unified interface.
    """

    def __init__(
        self,
        output_predictor: OutputTokenPredictor,
        duration_predictor: DurationPredictor | None = None,
    ):
        """Initialize combined predictor.

        Args:
            output_predictor: Predictor for output tokens
            duration_predictor: Optional predictor for duration (Stage 2)
        """
        self.output_predictor = output_predictor
        self.duration_predictor = duration_predictor

    def predict_output_tokens(self, request: Request) -> QuantilePrediction:
        """Predict output token quantiles."""
        return self.output_predictor.predict(request)

    def predict_duration(self, request: Request, output_q90: float) -> DurationPrediction | None:
        """Predict duration UCB using q90 output tokens."""
        if self.duration_predictor is None:
            return None
        return self.duration_predictor.predict(request, output_q90)

    def update(self, request: Request) -> None:
        """Update both predictors after request completion."""
        self.output_predictor.update(request)
        if self.duration_predictor is not None:
            self.duration_predictor.update(request)

    def reset(self) -> None:
        """Reset both predictors."""
        self.output_predictor.reset()
        if self.duration_predictor is not None:
            self.duration_predictor.reset()
