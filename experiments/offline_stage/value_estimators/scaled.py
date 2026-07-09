"""Scale another output-token predictor by a fixed multiplier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiments.offline_stage.value_estimators.base import (
    OutputTokenPredictor,
    PointPrediction,
    QuantilePrediction,
)

if TYPE_CHECKING:
    from routewise.schemas import Request


class ScaledOutputPredictor(OutputTokenPredictor):
    """Wrap an output-token predictor and multiply its predictions."""

    def __init__(self, base: OutputTokenPredictor, *, multiplier: float):
        if multiplier < 0:
            raise ValueError(f"prediction multiplier must be non-negative, got {multiplier}")
        self.base = base
        self.multiplier = float(multiplier)

    def predict(self, request: Request) -> PointPrediction | QuantilePrediction:
        prediction = self.base.predict(request)
        if isinstance(prediction, PointPrediction):
            return PointPrediction(
                tokens=self._scale(prediction.tokens),
                is_warmed_up=prediction.is_warmed_up,
            )
        return QuantilePrediction(
            q10=self._scale(prediction.q10),
            q50=self._scale(prediction.q50),
            q90=self._scale(prediction.q90),
            is_warmed_up=prediction.is_warmed_up,
        )

    def update(self, request: Request) -> None:
        self.base.update(request)

    def reset(self) -> None:
        self.base.reset()

    @property
    def is_warmed_up(self) -> bool:
        return self.base.is_warmed_up

    def get_calibration_stats(self) -> dict:
        stats = dict(self.base.get_calibration_stats())
        stats["prediction_multiplier"] = self.multiplier
        return stats

    def _scale(self, value: float) -> float:
        return max(0.0, float(value) * self.multiplier)


__all__ = ["ScaledOutputPredictor"]
