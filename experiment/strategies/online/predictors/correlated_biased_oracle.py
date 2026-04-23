"""Input-correlated biased oracle predictor.

Models realistic predictor failure modes where the error is not a uniform
bias across all requests, but instead correlates with request features.
This tests robustness against the kinds of systematic failures that can
occur when a simple predictor (EMA) encounters workload shifts or
non-stationary request features.

Three modes:
    long_underestimate:
        Requests with `input_tokens > threshold_tokens` have their output
        length multiplied by `bias_factor < 1`. Requests with shorter
        input are unbiased. Models "EMA trained on short-context requests
        under-predicts output length for long-context requests".

    short_overestimate:
        Requests with `input_tokens <= threshold_tokens` are multiplied by
        `bias_factor`. Models the opposite correlation (rare but useful
        as a sanity-check direction).

    tail_underestimate:
        Requests whose true API cost falls in the top-(1 - threshold_pct)
        percentile are multiplied by `bias_factor`. Models the worst-case
        adversarial failure: the highest-value requests are precisely the
        ones being under-predicted, breaking ordering at the top of the
        value distribution.

Unlike BiasedOraclePredictor (uniform bias preserves relative ordering
and is trivially robust by theorem), this predictor breaks ordering in a
feature-correlated way and exercises the router's value-identification
mechanism.
"""

from __future__ import annotations

from experiment.data.schema import Request
from experiment.predictors.base import (
    OutputTokenPredictor,
    QuantilePrediction,
)


class InputCorrelatedBiasedPredictor(OutputTokenPredictor):
    """Oracle predictor with bias that depends on request features.

    Attributes:
        mode: Bias correlation mode -- one of
            "long_underestimate", "short_overestimate", "tail_underestimate".
        bias_factor: Multiplier applied to biased-class requests
            (0 < bias_factor). Unbiased-class requests pass through.
        threshold_tokens: Input-token threshold for input-length modes
            (ignored for tail_underestimate mode).
        tail_value_threshold: Per-request API cost threshold; used only
            in tail_underestimate mode to identify the top-value class.
    """

    VALID_MODES = ("long_underestimate", "short_overestimate", "tail_underestimate")

    def __init__(
        self,
        mode: str,
        bias_factor: float,
        threshold_tokens: int | None = None,
        tail_value_threshold: float | None = None,
    ) -> None:
        """Initialize input-correlated biased predictor.

        Args:
            mode: One of the supported VALID_MODES.
            bias_factor: Multiplier for the biased class (0 < bias_factor).
            threshold_tokens: Required for long_underestimate and
                short_overestimate modes. Input-token threshold.
            tail_value_threshold: Required for tail_underestimate mode.
                Requests with API cost above this threshold are considered
                "tail" and have their prediction biased.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode}")
        if bias_factor <= 0.0:
            raise ValueError(f"bias_factor must be positive, got {bias_factor}")

        if mode in ("long_underestimate", "short_overestimate"):
            if threshold_tokens is None:
                raise ValueError(f"mode {mode} requires threshold_tokens")
        elif mode == "tail_underestimate":
            if tail_value_threshold is None:
                raise ValueError(f"mode {mode} requires tail_value_threshold")

        self.mode = mode
        self.bias_factor = float(bias_factor)
        self.threshold_tokens = threshold_tokens
        self.tail_value_threshold = tail_value_threshold

        # Track how many predictions were biased vs pass-through, for diagnostics.
        self._biased_count = 0
        self._total_count = 0

    def _is_biased_class(self, request: Request, true_output: float) -> bool:
        """Decide whether this request falls in the biased class."""
        if self.mode == "long_underestimate":
            return request.request_tokens > self.threshold_tokens
        if self.mode == "short_overestimate":
            return request.request_tokens <= self.threshold_tokens
        if self.mode == "tail_underestimate":
            # Approximate per-request API cost using output tokens alone.
            # This is proportional to the completion-token component of
            # the full cost (input component is handled separately by
            # the router). Using output_tokens directly here ensures the
            # predictor is self-contained and does not depend on the
            # cost calculator.
            return true_output >= self.tail_value_threshold
        raise AssertionError(f"unreachable mode {self.mode}")

    def predict(self, request: Request) -> QuantilePrediction:
        """Return distorted ground-truth output length as prediction."""
        true_len = max(float(request.response_tokens), 1.0)
        self._total_count += 1

        if self._is_biased_class(request, true_len):
            predicted = max(1.0, true_len * self.bias_factor)
            self._biased_count += 1
        else:
            predicted = true_len

        return QuantilePrediction(
            q10=predicted,
            q50=predicted,
            q90=predicted,
            is_warmed_up=True,
        )

    def update(self, request: Request) -> None:
        """No-op: correlated biased oracle has no state to update."""
        pass

    def reset(self) -> None:
        """Reset diagnostic counters."""
        self._biased_count = 0
        self._total_count = 0

    @property
    def is_warmed_up(self) -> bool:
        """Correlated biased oracle is always warmed up."""
        return True

    def get_biased_fraction(self) -> float:
        """Fraction of predictions that fell in the biased class."""
        if self._total_count == 0:
            return 0.0
        return self._biased_count / self._total_count
