"""Input-length bucket mean output-token predictor."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.offline_stage.value_estimators.base import (
    OutputTokenPredictor,
    PointPrediction,
    PredictionContext,
)
from rwsim.schemas import Request


@dataclass
class MeanState:
    """Online mean state for one output-token bucket."""

    count: int = 0
    total: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.total += value

    @property
    def mean(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total / self.count


class BucketMeanOutputPredictor(OutputTokenPredictor):
    """Predict output tokens using the mean of the matching input-length bucket.

    The bucket key is the log-spaced input-token bin from ``PredictionContext``.
    This intentionally ignores trace model labels so the predictor matches the
    single-target-model assumption used by the paper experiments.
    """

    def __init__(
        self,
        *,
        min_samples_bucket: int = 5,
        min_samples_global: int = 20,
        default_output_tokens: float = 500.0,
    ):
        if min_samples_bucket <= 0:
            raise ValueError("min_samples_bucket must be positive")
        if min_samples_global <= 0:
            raise ValueError("min_samples_global must be positive")
        if default_output_tokens <= 0:
            raise ValueError("default_output_tokens must be positive")
        self.min_samples_bucket = int(min_samples_bucket)
        self.min_samples_global = int(min_samples_global)
        self.default_output_tokens = float(default_output_tokens)
        self.bucket_states: dict[int, MeanState] = {}
        self.global_state = MeanState()

    def predict(self, request: Request) -> PointPrediction:
        ctx = PredictionContext.from_request(request)
        value, warmed_up = self._estimate(ctx.input_bin)
        return PointPrediction(tokens=value, is_warmed_up=warmed_up)

    def update(self, request: Request) -> None:
        output_tokens = float(request.response_tokens or 0)
        if output_tokens <= 0.0:
            return
        ctx = PredictionContext.from_request(request)
        state = self.bucket_states.setdefault(ctx.input_bin, MeanState())
        state.update(output_tokens)
        self.global_state.update(output_tokens)

    def reset(self) -> None:
        self.bucket_states.clear()
        self.global_state = MeanState()

    @property
    def is_warmed_up(self) -> bool:
        return self.global_state.count >= self.min_samples_global

    def _estimate(self, input_bin: int) -> tuple[float, bool]:
        state = self.bucket_states.get(input_bin)
        if state is not None and state.count >= self.min_samples_bucket:
            return max(0.0, state.mean), True
        if self.global_state.count >= self.min_samples_global:
            return max(0.0, self.global_state.mean), True
        return self.default_output_tokens, False


__all__ = ["BucketMeanOutputPredictor", "MeanState"]
