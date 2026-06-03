"""Synthetic TTFT distribution factory for simulator experiments.

This module is the single source of truth for parametric synthetic latency
families. Callers must name the anchor semantics explicitly so values such as
``300 ms`` cannot silently mean P50 in one section and mean in another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rwsim.world.distributions import LogNormal, Normal, Uniform

if TYPE_CHECKING:
    from rwsim.world.distributions import LatencyDistribution

LatencyAnchorKind = Literal["mean", "p50"]

SYNTHETIC_LATENCY_VERSION = "synthetic_latency_v2"
DEFAULT_LATENCY_ANCHOR_KIND: LatencyAnchorKind = "p50"
DEFAULT_UNIFORM_HALF_WIDTH_RATIO = 0.5
DEFAULT_NORMAL_SIGMA_RATIO = 0.3
DEFAULT_LOGNORMAL_SIGMA = 0.5

_SYNTHETIC_FAMILIES = ("uniform", "normal", "heavy_tail", "lognormal")


@dataclass(frozen=True)
class SyntheticLatencySpec:
    """Specification for one synthetic TTFT distribution."""

    family: str
    anchor_ms: float
    anchor_kind: LatencyAnchorKind = DEFAULT_LATENCY_ANCHOR_KIND
    uniform_half_width_ratio: float = DEFAULT_UNIFORM_HALF_WIDTH_RATIO
    normal_sigma_ratio: float = DEFAULT_NORMAL_SIGMA_RATIO
    lognormal_sigma: float = DEFAULT_LOGNORMAL_SIGMA

    def __post_init__(self) -> None:
        normalize_synthetic_latency_family(self.family)
        if self.anchor_kind not in ("mean", "p50"):
            raise ValueError(
                f"anchor_kind must be 'mean' or 'p50', got {self.anchor_kind!r}"
            )
        if self.anchor_ms <= 0.0:
            raise ValueError(f"anchor_ms must be positive, got {self.anchor_ms}")
        if not 0.0 <= self.uniform_half_width_ratio < 1.0:
            raise ValueError(
                "uniform_half_width_ratio must be in [0, 1), got "
                f"{self.uniform_half_width_ratio}"
            )
        if self.normal_sigma_ratio <= 0.0:
            raise ValueError(
                f"normal_sigma_ratio must be positive, got {self.normal_sigma_ratio}"
            )
        if self.lognormal_sigma <= 0.0:
            raise ValueError(
                f"lognormal_sigma must be positive, got {self.lognormal_sigma}"
            )


def normalize_synthetic_latency_family(family: str) -> str:
    """Return the canonical family name used in metadata."""
    if family not in _SYNTHETIC_FAMILIES:
        known = ", ".join(_SYNTHETIC_FAMILIES)
        raise ValueError(f"unknown latency family {family!r}; known: {known}")
    return "heavy_tail" if family == "lognormal" else family


def make_synthetic_ttft(spec: SyntheticLatencySpec) -> LatencyDistribution:
    """Build a synthetic TTFT distribution from an explicit anchor spec."""
    family = normalize_synthetic_latency_family(spec.family)
    if family == "uniform":
        width = spec.uniform_half_width_ratio * spec.anchor_ms
        return Uniform(low=spec.anchor_ms - width, high=spec.anchor_ms + width)
    if family == "normal":
        return Normal(
            mean_ms=spec.anchor_ms,
            sigma=spec.normal_sigma_ratio * spec.anchor_ms,
        )
    if family == "heavy_tail":
        return LogNormal(
            mu=_lognormal_mu_for_anchor(
                anchor_ms=spec.anchor_ms,
                anchor_kind=spec.anchor_kind,
                sigma=spec.lognormal_sigma,
            ),
            sigma=spec.lognormal_sigma,
        )
    raise AssertionError(f"unreachable latency family {family!r}")


def synthetic_latency_metadata(spec: SyntheticLatencySpec) -> dict[str, object]:
    """Return common metadata for a synthetic TTFT distribution."""
    family = normalize_synthetic_latency_family(spec.family)
    dist = make_synthetic_ttft(spec)
    shape: dict[str, float] = {}
    if family == "uniform":
        shape["uniform_half_width_ratio"] = spec.uniform_half_width_ratio
    elif family == "normal":
        shape["normal_sigma_ratio"] = spec.normal_sigma_ratio
    elif family == "heavy_tail":
        shape["lognormal_sigma"] = spec.lognormal_sigma
    return {
        "latency_generation_version": SYNTHETIC_LATENCY_VERSION,
        "latency_family": family,
        "latency_anchor_kind": spec.anchor_kind,
        "latency_anchor_ms": spec.anchor_ms,
        "latency_distribution_mean_ms": dist.mean(),
        "latency_distribution_p50_ms": dist.p50(),
        "latency_shape": shape,
    }


def describe_synthetic_latency(spec: SyntheticLatencySpec) -> str:
    """Return a compact human-readable TTFT description for scenario text."""
    family = normalize_synthetic_latency_family(spec.family)
    dist = make_synthetic_ttft(spec)
    if family == "heavy_tail":
        if spec.anchor_kind == "p50":
            return f"{family}, P50={spec.anchor_ms:.0f}ms, sigma_log={spec.lognormal_sigma:g}"
        return (
            f"{family}, mean={spec.anchor_ms:.0f}ms, "
            f"P50={dist.p50():.0f}ms, sigma_log={spec.lognormal_sigma:g}"
        )
    return f"{family}, {spec.anchor_kind}={spec.anchor_ms:.0f}ms"


def _lognormal_mu_for_anchor(
    *,
    anchor_ms: float,
    anchor_kind: LatencyAnchorKind,
    sigma: float,
) -> float:
    if anchor_kind == "p50":
        return math.log(anchor_ms)
    if anchor_kind == "mean":
        return math.log(anchor_ms) - 0.5 * sigma * sigma
    raise ValueError(f"unknown anchor_kind {anchor_kind!r}")


__all__ = [
    "DEFAULT_LATENCY_ANCHOR_KIND",
    "DEFAULT_LOGNORMAL_SIGMA",
    "DEFAULT_NORMAL_SIGMA_RATIO",
    "DEFAULT_UNIFORM_HALF_WIDTH_RATIO",
    "SYNTHETIC_LATENCY_VERSION",
    "LatencyAnchorKind",
    "SyntheticLatencySpec",
    "describe_synthetic_latency",
    "make_synthetic_ttft",
    "normalize_synthetic_latency_family",
    "synthetic_latency_metadata",
]
