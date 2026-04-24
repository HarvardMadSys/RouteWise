"""Compatibility wrapper for value-estimator base interfaces."""

from __future__ import annotations

from rwsim.policies.value_estimators.base import (
    CombinedPredictor,
    DurationPrediction,
    DurationPredictor,
    OutputTokenPredictor,
    PredictionContext,
    QuantilePrediction,
)

__all__ = [
    "CombinedPredictor",
    "DurationPrediction",
    "DurationPredictor",
    "OutputTokenPredictor",
    "PredictionContext",
    "QuantilePrediction",
]
