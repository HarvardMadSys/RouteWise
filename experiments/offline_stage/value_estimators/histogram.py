"""Streaming histogram-based quantile predictor.

This module implements a dependency-free streaming quantile estimator using
histograms with a hierarchical backoff chain for handling sparse data.

Key features:
- No external dependencies (pure Python + standard library)
- Deterministic and fast
- Hierarchical backoff: (model, input_bin, hour_bin) -> (model, input_bin) -> (model) -> (global)
- Log-spaced bins for output tokens
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

from experiments.offline_stage.value_estimators.base import (
    DurationPrediction,
    DurationPredictor,
    OutputTokenPredictor,
    PredictionContext,
    QuantilePrediction,
)
from llm_routewise.schemas import Request

logger = logging.getLogger(__name__)


@dataclass
class HistogramBin:
    """A single histogram bin.

    Attributes:
        low: Lower bound (inclusive)
        high: Upper bound (exclusive)
        count: Number of samples in this bin
    """

    low: float
    high: float
    count: int = 0

    @property
    def midpoint(self) -> float:
        """Get bin midpoint."""
        return (self.low + self.high) / 2


class StreamingHistogram:
    """Streaming histogram for quantile estimation.

    Uses log-spaced bins to handle wide range of values efficiently.
    """

    def __init__(
        self,
        min_value: float = 1.0,
        max_value: float = 100000.0,
        num_bins: int = 50,
    ):
        """Initialize histogram with log-spaced bins.

        Args:
            min_value: Minimum value (values below are clipped)
            max_value: Maximum value (values above are clipped)
            num_bins: Number of bins
        """
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins

        # Create log-spaced bin edges
        log_min = math.log(min_value)
        log_max = math.log(max_value)
        log_edges = [log_min + i * (log_max - log_min) / num_bins for i in range(num_bins + 1)]
        self.edges = [math.exp(e) for e in log_edges]

        # Initialize bins
        self.bins = [
            HistogramBin(low=self.edges[i], high=self.edges[i + 1]) for i in range(num_bins)
        ]

        self.total_count = 0
        self._sum = 0.0  # For mean calculation

    def add(self, value: float) -> None:
        """Add a value to the histogram.

        Args:
            value: Value to add
        """
        # Clip to range
        value = max(self.min_value, min(value, self.max_value - 1e-9))

        # Find bin using binary search
        bin_idx = self._find_bin(value)
        self.bins[bin_idx].count += 1
        self.total_count += 1
        self._sum += value

    def _find_bin(self, value: float) -> int:
        """Find bin index for a value using binary search.

        Args:
            value: Value to find bin for

        Returns:
            Bin index
        """
        # Binary search for the correct bin
        low, high = 0, self.num_bins - 1
        while low < high:
            mid = (low + high) // 2
            if value < self.edges[mid + 1]:
                high = mid
            else:
                low = mid + 1
        return low

    def quantile(self, q: float, conservative: bool = True) -> float:
        """Get quantile value.

        Args:
            q: Quantile (0.0 to 1.0)
            conservative: If True, use bin lower bound for low quantiles (q < 0.5)
                         and upper bound for high quantiles (q > 0.5) to ensure
                         proper LCB/UCB semantics. If False, always use midpoint.

        Returns:
            Estimated quantile value
        """
        if self.total_count == 0:
            # Return geometric mean of range as default
            return math.sqrt(self.min_value * self.max_value)

        target_count = q * self.total_count
        cumulative = 0

        for bin in self.bins:
            cumulative += bin.count
            if cumulative >= target_count:
                if not conservative:
                    return bin.midpoint
                # Conservative estimation for proper LCB/UCB semantics:
                # - Low quantiles (e.g., P10): use lower bound for true LCB
                # - High quantiles (e.g., P90): use upper bound for true UCB
                # - Middle quantiles (around P50): use midpoint
                if q < 0.3:
                    return bin.low
                elif q > 0.7:
                    return bin.high
                else:
                    return bin.midpoint

        # Should not reach here, but return last bin value
        if conservative and q > 0.7:
            return self.bins[-1].high
        return self.bins[-1].midpoint

    def mean(self) -> float:
        """Get mean value."""
        if self.total_count == 0:
            return math.sqrt(self.min_value * self.max_value)
        return self._sum / self.total_count

    def merge(self, other: "StreamingHistogram") -> None:
        """Merge another histogram into this one.

        Args:
            other: Histogram to merge
        """
        for i, bin in enumerate(other.bins):
            self.bins[i].count += bin.count
        self.total_count += other.total_count
        self._sum += other._sum


@dataclass
class HierarchicalStats:
    """Hierarchical statistics with backoff chain.

    Maintains separate histograms at different granularity levels:
    - Level 0 (most specific): (model, input_bin, hour_bin)
    - Level 1: (model, input_bin)
    - Level 2: (model)
    - Level 3 (global): all data
    """

    # Minimum samples required at each level for reliable estimates
    min_samples_l0: int = 10
    min_samples_l1: int = 20
    min_samples_l2: int = 50
    min_samples_global: int = 100

    # Histograms at each level
    level0: dict[tuple[str, int, int], StreamingHistogram] = field(default_factory=dict)
    level1: dict[tuple[str, int], StreamingHistogram] = field(default_factory=dict)
    level2: dict[str, StreamingHistogram] = field(default_factory=dict)
    global_hist: StreamingHistogram = field(default_factory=StreamingHistogram)

    def add(self, model: str | None, input_bin: int, hour_bin: int, value: float) -> None:
        """Add a value to all applicable histograms.

        Args:
            model: Model name
            input_bin: Binned input token count
            hour_bin: Hour of day bin (0-23 or coarser)
            value: Output token count
        """
        model_key = model or "_unknown_"

        # Update all levels
        key0 = (model_key, input_bin, hour_bin)
        if key0 not in self.level0:
            self.level0[key0] = StreamingHistogram()
        self.level0[key0].add(value)

        key1 = (model_key, input_bin)
        if key1 not in self.level1:
            self.level1[key1] = StreamingHistogram()
        self.level1[key1].add(value)

        if model_key not in self.level2:
            self.level2[model_key] = StreamingHistogram()
        self.level2[model_key].add(value)

        self.global_hist.add(value)

    def quantile(
        self, model: str | None, input_bin: int, hour_bin: int, q: float, conservative: bool = True
    ) -> float:
        """Get quantile with backoff.

        Args:
            model: Model name
            input_bin: Binned input token count
            hour_bin: Hour of day bin
            q: Quantile (0.0 to 1.0)
            conservative: If True, use bin bounds for LCB/UCB semantics

        Returns:
            Quantile estimate from the most specific level with sufficient data
        """
        model_key = model or "_unknown_"

        # Try level 0 (most specific)
        key0 = (model_key, input_bin, hour_bin)
        if key0 in self.level0 and self.level0[key0].total_count >= self.min_samples_l0:
            return self.level0[key0].quantile(q, conservative)

        # Backoff to level 1
        key1 = (model_key, input_bin)
        if key1 in self.level1 and self.level1[key1].total_count >= self.min_samples_l1:
            return self.level1[key1].quantile(q, conservative)

        # Backoff to level 2
        if model_key in self.level2 and self.level2[model_key].total_count >= self.min_samples_l2:
            return self.level2[model_key].quantile(q, conservative)

        # Backoff to global
        return self.global_hist.quantile(q, conservative)

    def get_best_level(self, model: str | None, input_bin: int, hour_bin: int) -> int:
        """Get the most specific level with sufficient data.

        Returns:
            Level number (0-3) or -1 if no level has sufficient data
        """
        model_key = model or "_unknown_"

        key0 = (model_key, input_bin, hour_bin)
        if key0 in self.level0 and self.level0[key0].total_count >= self.min_samples_l0:
            return 0

        key1 = (model_key, input_bin)
        if key1 in self.level1 and self.level1[key1].total_count >= self.min_samples_l1:
            return 1

        if model_key in self.level2 and self.level2[model_key].total_count >= self.min_samples_l2:
            return 2

        if self.global_hist.total_count >= self.min_samples_global:
            return 3

        return -1  # Not warmed up


class HistogramOutputPredictor(OutputTokenPredictor):
    """Histogram-based output token predictor with hierarchical backoff.

    This is the default dependency-free predictor that:
    - Uses streaming histograms for online quantile estimation
    - Employs hierarchical backoff for sparse data
    - Tracks calibration statistics for evaluation
    """

    def __init__(
        self,
        min_samples_warmup: int = 100,
        hour_bin_size: int = 4,  # 4-hour bins (6 bins per day)
    ):
        """Initialize histogram predictor.

        Args:
            min_samples_warmup: Minimum global samples for warmup
            hour_bin_size: Hours per bin (1, 2, 4, 6, 8, 12, or 24)
        """
        self.min_samples_warmup = min_samples_warmup
        self.hour_bin_size = hour_bin_size

        # Main statistics
        self.stats = HierarchicalStats(min_samples_global=min_samples_warmup)

        # Calibration tracking
        self._calibration_counts = {
            "q10_below": 0,
            "q50_below": 0,
            "q90_below": 0,
            "total": 0,
        }

    def predict(self, request: Request) -> QuantilePrediction:
        """Predict output token quantiles.

        Args:
            request: The incoming request

        Returns:
            QuantilePrediction with q10, q50, q90 estimates.
            Uses conservative bounds: q10 returns bin lower bound (LCB),
            q90 returns bin upper bound (UCB), q50 returns midpoint.
        """
        ctx = PredictionContext.from_request(request)
        hour_bin = ctx.hour_of_day // self.hour_bin_size

        # Use conservative=True for proper LCB/UCB semantics:
        # - q10: bin lower bound (ensures LCB is truly conservative)
        # - q50: bin midpoint (balanced estimate)
        # - q90: bin upper bound (ensures UCB is truly conservative)
        q10 = self.stats.quantile(ctx.model, ctx.input_bin, hour_bin, 0.10, conservative=True)
        q50 = self.stats.quantile(ctx.model, ctx.input_bin, hour_bin, 0.50, conservative=True)
        q90 = self.stats.quantile(ctx.model, ctx.input_bin, hour_bin, 0.90, conservative=True)

        return QuantilePrediction(
            q10=q10,
            q50=q50,
            q90=q90,
            is_warmed_up=self.is_warmed_up,
        )

    def update(self, request: Request) -> None:
        """Update predictor with realized output tokens.

        Args:
            request: The completed request with response_tokens
        """
        if request.response_tokens <= 0:
            return

        ctx = PredictionContext.from_request(request)
        hour_bin = ctx.hour_of_day // self.hour_bin_size
        output_tokens = float(request.response_tokens)

        # Get prediction BEFORE updating (for calibration)
        if self.is_warmed_up:
            pred = self.predict(request)
            self._update_calibration(pred, output_tokens)

        # Update statistics
        self.stats.add(ctx.model, ctx.input_bin, hour_bin, output_tokens)

    def _update_calibration(self, pred: QuantilePrediction, actual: float) -> None:
        """Update calibration statistics.

        Args:
            pred: The prediction made
            actual: Actual output tokens
        """
        self._calibration_counts["total"] += 1
        if actual <= pred.q10:
            self._calibration_counts["q10_below"] += 1
        if actual <= pred.q50:
            self._calibration_counts["q50_below"] += 1
        if actual <= pred.q90:
            self._calibration_counts["q90_below"] += 1

    def reset(self) -> None:
        """Reset predictor state."""
        self.stats = HierarchicalStats(min_samples_global=self.min_samples_warmup)
        self._calibration_counts = {
            "q10_below": 0,
            "q50_below": 0,
            "q90_below": 0,
            "total": 0,
        }

    @property
    def is_warmed_up(self) -> bool:
        """Check if predictor has sufficient samples."""
        return self.stats.global_hist.total_count >= self.min_samples_warmup

    def get_calibration_stats(self) -> dict:
        """Get calibration statistics.

        Returns:
            Dict with coverage rates (should be close to quantile levels)
        """
        total = self._calibration_counts["total"]
        if total == 0:
            return {"q10_coverage": 0.0, "q50_coverage": 0.0, "q90_coverage": 0.0, "total": 0}

        return {
            "q10_coverage": self._calibration_counts["q10_below"] / total,
            "q50_coverage": self._calibration_counts["q50_below"] / total,
            "q90_coverage": self._calibration_counts["q90_below"] / total,
            "total": total,
        }


class HistogramDurationPredictor(DurationPredictor):
    """Histogram-based duration predictor for Stage 2.

    Predicts processing duration using:
    - TTFT (time to first token): Use 90th percentile for conservative estimate
    - TPS (tokens per second): Use 10th percentile for conservative estimate
    - Duration UCB = TTFT_p90 + output_tokens_p90 / TPS_p10
    """

    def __init__(
        self,
        min_samples_warmup: int = 50,
        default_ttft_seconds: float = 1.0,
        default_tps: float = 50.0,
    ):
        """Initialize duration predictor.

        Args:
            min_samples_warmup: Minimum samples for warmup
            default_ttft_seconds: Default TTFT when not warmed up
            default_tps: Default TPS when not warmed up
        """
        self.min_samples_warmup = min_samples_warmup
        self.default_ttft_seconds = default_ttft_seconds
        self.default_tps = default_tps

        # Per-model histograms for TTFT and TPS
        self.ttft_histograms: dict[str, StreamingHistogram] = defaultdict(
            lambda: StreamingHistogram(min_value=0.01, max_value=60.0, num_bins=30)
        )
        self.tps_histograms: dict[str, StreamingHistogram] = defaultdict(
            lambda: StreamingHistogram(min_value=1.0, max_value=500.0, num_bins=30)
        )
        self.global_ttft = StreamingHistogram(min_value=0.01, max_value=60.0, num_bins=30)
        self.global_tps = StreamingHistogram(min_value=1.0, max_value=500.0, num_bins=30)

    def predict(self, request: Request, predicted_output_tokens: float) -> DurationPrediction:
        """Predict processing duration.

        Args:
            request: The incoming request
            predicted_output_tokens: Predicted output tokens (q90)

        Returns:
            DurationPrediction with conservative UCB estimate
        """
        model = request.model or "_unknown_"

        # Get TTFT p90 (conservative high)
        if model in self.ttft_histograms and self.ttft_histograms[model].total_count >= 20:
            ttft_p90 = self.ttft_histograms[model].quantile(0.90)
        elif self.global_ttft.total_count >= self.min_samples_warmup:
            ttft_p90 = self.global_ttft.quantile(0.90)
        else:
            ttft_p90 = self.default_ttft_seconds

        # Get TPS p10 (conservative low)
        if model in self.tps_histograms and self.tps_histograms[model].total_count >= 20:
            tps_p10 = self.tps_histograms[model].quantile(0.10)
        elif self.global_tps.total_count >= self.min_samples_warmup:
            tps_p10 = self.global_tps.quantile(0.10)
        else:
            tps_p10 = self.default_tps

        # Ensure TPS is positive
        tps_p10 = max(tps_p10, 1.0)

        # Compute duration UCB
        duration_ucb = ttft_p90 + predicted_output_tokens / tps_p10

        return DurationPrediction(
            ttft_seconds=ttft_p90,
            tps=tps_p10,
            duration_ucb=duration_ucb,
            is_warmed_up=self.global_ttft.total_count >= self.min_samples_warmup,
        )

    def update(self, request: Request) -> None:
        """Update predictor with realized latency stats.

        Args:
            request: The completed request
        """
        model = request.model or "_unknown_"

        # Update TTFT if available
        if request.ttft_ms is not None and request.ttft_ms > 0:
            ttft_seconds = request.ttft_ms / 1000.0
            self.ttft_histograms[model].add(ttft_seconds)
            self.global_ttft.add(ttft_seconds)

        # Update TPS if we have latency and output tokens
        if (
            request.latency_ms is not None
            and request.response_tokens is not None
            and request.response_tokens > 0
        ):
            ttft_ms = request.ttft_ms or 0
            decode_time_ms = request.latency_ms - ttft_ms
            if decode_time_ms > 10:  # At least 10ms decode time
                tps = request.response_tokens / (decode_time_ms / 1000.0)
                if 1.0 <= tps <= 500.0:  # Reasonable TPS range
                    self.tps_histograms[model].add(tps)
                    self.global_tps.add(tps)

    def reset(self) -> None:
        """Reset predictor state."""
        self.ttft_histograms.clear()
        self.tps_histograms.clear()
        self.global_ttft = StreamingHistogram(min_value=0.01, max_value=60.0, num_bins=30)
        self.global_tps = StreamingHistogram(min_value=1.0, max_value=500.0, num_bins=30)
