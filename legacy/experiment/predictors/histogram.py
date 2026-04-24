"""Compatibility wrapper for histogram value estimators."""

from __future__ import annotations

from rwsim.policies.value_estimators.histogram import (
    HierarchicalStats,
    HistogramBin,
    HistogramDurationPredictor,
    HistogramOutputPredictor,
    StreamingHistogram,
)

__all__ = [
    "HierarchicalStats",
    "HistogramBin",
    "HistogramDurationPredictor",
    "HistogramOutputPredictor",
    "StreamingHistogram",
]
