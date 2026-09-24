"""Constant output-token predictor and workload calibration helper.

Used by simulator runners' optional ``--predictor`` flag to support
``constant_<kind>`` and ``fixed:<value>`` baselines alongside streaming
predictors.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

import numpy as np

from experiments.offline_stage.value_estimators.base import (
    OutputTokenPredictor,
    QuantilePrediction,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rwsim.schemas import Request


class ConstantOutputPredictor(OutputTokenPredictor):
    """Always predicts the same number of output tokens for every request."""

    def __init__(self, value: float, *, label: str | None = None):
        if value <= 0:
            raise ValueError(f"ConstantOutputPredictor requires positive value, got {value}")
        self.value = float(value)
        self.label = label or f"constant_{int(round(value))}"

    def predict(self, request: Request) -> QuantilePrediction:
        del request
        return QuantilePrediction(
            q10=self.value,
            q50=self.value,
            q90=self.value,
            is_warmed_up=True,
        )

    def update(self, request: Request) -> None:
        del request

    def reset(self) -> None:
        pass

    @property
    def is_warmed_up(self) -> bool:
        return True


def _positive_response_tokens(requests: Iterable[Request]) -> list[float]:
    values = [
        float(getattr(request, "response_tokens", 0) or 0) for request in requests
    ]
    values = [value for value in values if value > 0.0]
    if not values:
        raise ValueError("workload has no requests with response_tokens > 0")
    return values


def workload_constant_value(requests: Iterable[Request], *, kind: str) -> float:
    """Calibrate a constant predictor against the workload."""
    if kind.startswith("fixed:"):
        value = float(kind.split(":", 1)[1])
        if value <= 0:
            raise ValueError(f"fixed predictor value must be > 0, got {value}")
        return value
    values = _positive_response_tokens(requests)
    if kind == "mean":
        return float(statistics.fmean(values))
    if kind in {"p1", "p10", "p25", "p50", "p75", "p90", "p99"}:
        return float(np.percentile(values, float(kind[1:])))
    raise ValueError(
        f"unknown constant predictor kind {kind!r}; expected one of "
        "mean, p1, p10, p25, p50, p75, p90, p99, fixed:<value>"
    )


__all__ = [
    "ConstantOutputPredictor",
    "workload_constant_value",
]
