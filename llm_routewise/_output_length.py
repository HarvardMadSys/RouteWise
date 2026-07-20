"""Dependency-free online output-length estimation for the public facade."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class _MeanState:
    """Online arithmetic mean without retaining individual observations."""

    count: int = 0
    total: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.total += value

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class _OutputLengthEstimator:
    """Predict output tokens from a log-spaced input-length bucket.

    The estimator deliberately mirrors the research harness's simple online
    cascade without importing any experiment or simulator types:

    * use the matching bucket mean after five observations in that bucket;
    * otherwise use the global mean after twenty observations overall;
    * otherwise predict 500 output tokens.

    Only strictly positive, finite output lengths are observations. A caller
    can therefore pass a zero-token completion without warming the estimator.
    """

    def __init__(
        self,
        *,
        min_bucket_samples: int = 5,
        min_global_samples: int = 20,
        default_output_tokens: float = 500.0,
        num_buckets: int = 20,
    ) -> None:
        if isinstance(min_bucket_samples, bool) or min_bucket_samples <= 0:
            raise ValueError("min_bucket_samples must be a positive integer")
        if isinstance(min_global_samples, bool) or min_global_samples <= 0:
            raise ValueError("min_global_samples must be a positive integer")
        if isinstance(num_buckets, bool) or num_buckets <= 0:
            raise ValueError("num_buckets must be a positive integer")
        if not math.isfinite(default_output_tokens) or default_output_tokens <= 0:
            raise ValueError("default_output_tokens must be finite and positive")

        self._min_bucket_samples = int(min_bucket_samples)
        self._min_global_samples = int(min_global_samples)
        self._default_output_tokens = float(default_output_tokens)
        self._num_buckets = int(num_buckets)
        self._bucket_states: dict[int, _MeanState] = {}
        self._global_state = _MeanState()

    def predict(self, input_tokens: int) -> float:
        """Return the current point estimate for ``input_tokens``."""
        bucket = self._input_bucket(input_tokens)
        state = self._bucket_states.get(bucket)
        if state is not None and state.count >= self._min_bucket_samples:
            return state.mean
        if self._global_state.count >= self._min_global_samples:
            return self._global_state.mean
        return self._default_output_tokens

    def update(self, input_tokens: int, output_tokens: int | float) -> bool:
        """Observe a completed winner, returning whether it trained the model."""
        bucket = self._input_bucket(input_tokens)
        if isinstance(output_tokens, bool) or not isinstance(output_tokens, (int, float)):
            raise TypeError("output_tokens must be a real number")
        value = float(output_tokens)
        if not math.isfinite(value):
            raise ValueError("output_tokens must be finite")
        if value <= 0.0:
            return False

        self._bucket_states.setdefault(bucket, _MeanState()).update(value)
        self._global_state.update(value)
        return True

    def reset(self) -> None:
        """Forget every observation."""
        self._bucket_states.clear()
        self._global_state = _MeanState()

    @property
    def global_count(self) -> int:
        """Return the number of positive observations learned globally."""
        return self._global_state.count

    @property
    def is_warmed_up(self) -> bool:
        """Return whether the global fallback has reached its threshold."""
        return self._global_state.count >= self._min_global_samples

    def _input_bucket(self, input_tokens: int) -> int:
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int):
            raise TypeError("input_tokens must be an integer")
        if input_tokens < 0:
            raise ValueError("input_tokens must be non-negative")
        if input_tokens == 0:
            return 0
        return min(int(math.log2(max(input_tokens, 1))), self._num_buckets - 1)


__all__ = ["_OutputLengthEstimator"]
