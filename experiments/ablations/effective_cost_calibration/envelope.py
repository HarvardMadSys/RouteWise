"""Envelope calibration helpers for effective-cost sensitivity sweeps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from routewise.capacity import ProviderTier

if TYPE_CHECKING:
    from routewise.schemas import Request
    from routewise.sim.world.providers import Provider

ApiReference = Literal["cheapest_api", "median_api", "mean_api", "max_api"]
PercentileEnvelope = Literal["p05_p95", "p10_p90", "p25_p75", "min_max"]

API_REFERENCES: tuple[ApiReference, ...] = (
    "cheapest_api",
    "median_api",
    "mean_api",
    "max_api",
)
PERCENTILE_ENVELOPES: tuple[PercentileEnvelope, ...] = (
    "p05_p95",
    "p10_p90",
    "p25_p75",
    "min_max",
)
DEFAULT_API_REFERENCE: ApiReference = "cheapest_api"
DEFAULT_PERCENTILE_ENVELOPE: PercentileEnvelope = "p10_p90"

_PERCENTILE_BOUNDS: dict[PercentileEnvelope, tuple[float, float]] = {
    "p05_p95": (5.0, 95.0),
    "p10_p90": (10.0, 90.0),
    "p25_p75": (25.0, 75.0),
    "min_max": (0.0, 100.0),
}


@dataclass(frozen=True, slots=True)
class EnvelopeSpec:
    """One L/U calibration choice."""

    api_reference: ApiReference = DEFAULT_API_REFERENCE
    percentile_envelope: PercentileEnvelope = DEFAULT_PERCENTILE_ENVELOPE

    @property
    def label(self) -> str:
        return f"ref={self.api_reference}__env={self.percentile_envelope}"


def workload_cost_envelope(
    providers: list[Provider],
    requests: list[Request],
    *,
    spec: EnvelopeSpec,
    floor_ratio: float = 1e-3,
) -> tuple[float, float]:
    """Calibrate L/U from workload request costs under one API reference."""
    values = workload_api_reference_costs(
        providers,
        requests,
        api_reference=spec.api_reference,
    )
    lower, upper = percentile_bounds(spec.percentile_envelope)
    return cost_envelope_from_values(
        values,
        lower_percentile=lower,
        upper_percentile=upper,
        floor_ratio=floor_ratio,
    )


def workload_api_reference_costs(
    providers: list[Provider],
    requests: list[Request],
    *,
    api_reference: ApiReference,
) -> list[float]:
    """Return one API-equivalent request cost per request."""
    api_providers = [
        provider
        for provider in providers
        if provider.tier == ProviderTier.S_A
        and (
            provider.effective_input_cost_per_token > 0.0
            or provider.effective_output_cost_per_token > 0.0
        )
    ]
    if not api_providers:
        raise ValueError("Cannot compute calibration envelope: no positive-cost S_A providers.")

    values: list[float] = []
    for request in requests:
        costs = [
            provider.marginal_cost_for_request(request, 0.0)
            for provider in api_providers
        ]
        costs = [cost for cost in costs if cost > 0.0 and math.isfinite(cost)]
        if costs:
            values.append(_reference_value(costs, api_reference))
    if not values:
        raise ValueError(
            "Cannot compute calibration envelope: no positive API-equivalent "
            "request costs were found in the workload."
        )
    return values


def cost_envelope_from_values(
    values: list[float],
    *,
    lower_percentile: float,
    upper_percentile: float,
    floor_ratio: float = 1e-3,
) -> tuple[float, float]:
    """Compute a positive L/U envelope from request-value samples."""
    if not values:
        raise ValueError("Cannot compute calibration envelope from an empty value set.")
    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError(
            "Envelope percentiles must satisfy 0 <= lower < upper <= 100; "
            f"got lower={lower_percentile}, upper={upper_percentile}"
        )

    costs = np.asarray(values, dtype=float)
    costs = costs[np.isfinite(costs) & (costs > 0.0)]
    if costs.size == 0:
        raise ValueError("Cannot compute calibration envelope from non-positive values.")
    L = float(np.percentile(costs, lower_percentile))
    U = float(np.percentile(costs, upper_percentile))
    if not (math.isfinite(L) and math.isfinite(U)) or U <= 0.0:
        raise ValueError(
            "Cannot compute calibration envelope from non-finite or non-positive "
            f"bounds: L={L}, U={U}"
        )
    if L <= 0.0 or L >= U:
        L = max(U * floor_ratio, 1e-9)
    return (L, U)


def percentile_bounds(percentile_envelope: PercentileEnvelope) -> tuple[float, float]:
    """Return the numeric percentile pair for a named envelope."""
    try:
        return _PERCENTILE_BOUNDS[percentile_envelope]
    except KeyError as exc:
        known = ", ".join(PERCENTILE_ENVELOPES)
        raise ValueError(
            f"unknown percentile envelope {percentile_envelope!r}; known: {known}"
        ) from exc


def _reference_value(costs: list[float], api_reference: ApiReference) -> float:
    if api_reference == "cheapest_api":
        return min(costs)
    if api_reference == "median_api":
        return float(np.median(np.asarray(costs, dtype=float)))
    if api_reference == "mean_api":
        return float(np.mean(np.asarray(costs, dtype=float)))
    if api_reference == "max_api":
        return max(costs)
    known = ", ".join(API_REFERENCES)
    raise ValueError(f"unknown API reference {api_reference!r}; known: {known}")


__all__ = [
    "API_REFERENCES",
    "DEFAULT_API_REFERENCE",
    "DEFAULT_PERCENTILE_ENVELOPE",
    "PERCENTILE_ENVELOPES",
    "ApiReference",
    "EnvelopeSpec",
    "PercentileEnvelope",
    "cost_envelope_from_values",
    "percentile_bounds",
    "workload_api_reference_costs",
    "workload_cost_envelope",
]
