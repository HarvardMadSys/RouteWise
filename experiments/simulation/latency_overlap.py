"""§2.1 latency-layer band-overlap construction.

Builds three same-cost providers (fast / medium / slow) whose TTFT distributions
have a *target Q10-Q90 band coverage* on the (fast, medium) anchor pair:

    coverage(d_a -> d_b) = |B(d_a) ∩ B(d_b)| / |B(d_a)|

where B(d) = [q10(d), q90(d)] is the central 80% latency band of distribution d.
This is *asymmetric* (anchored on d_a's band length), as decided in the §2.1
design discussion (Slack 2026-05-07).

Construction (per parametric family, all three providers share one shape ratio
or one log-space sigma so the family stays self-similar):

- Uniform:  shared half-width ratio ρ = W / P50; closed-form solve.
- Normal:   shared sigma ratio ρ = σ / P50;   closed-form solve.
- LogNormal: shared σ_log;                     closed-form solve in log-space.
- real_world: empirical RW3 pool; no construction target, realised coverage reported.

Realised metrics (reported in summary csv):
- target_band_coverage_fast_medium     (declared, e.g. 0.0 / 0.5)
- realised_band_coverage_{fast_medium, medium_slow, fast_slow}
- normal_clip_fraction (Normal only; fraction of mass below MIN_LATENCY_MS)

Caveats (all to be reflected in paper caption):
1. half_overlap means "fast band 50% covered by medium band", not "full
   distribution overlap is 50%".
2. coverage is directional (fast -> medium); denominator is fast band length.
3. only the (fast, medium) pair is anchored to target; the other two pairs
   inherit whatever realised coverage falls out of the shared shape ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rwsim.world.distributions import LogNormal, Normal, Uniform

if TYPE_CHECKING:
    from rwsim.world.distributions import LatencyDistribution


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Three latency tiers — fast / medium / slow.
LATENCY_LAYER_P50_MS: tuple[float, float, float] = (100.0, 300.0, 1000.0)

# Shared one-sided z-score for q10/q90: P(Z <= z) = 0.90 → z ≈ 1.2816.
# Used for Normal and LogNormal band-edge computations.
_Z_P90: float = 1.2815515655446004

# Q10/Q90 band edges of Uniform[c-W, c+W] are c ± 0.8W (the 0.1/0.9 quantile
# offset is 0.4 of the full width = 0.8 of the half-width).
_UNIFORM_BAND_FRAC: float = 0.8

# Two named scenarios — the only construction targets we expose for §2.1.
OVERLAP_TARGETS: dict[str, float] = {
    "no_overlap": 0.0,
    "half_overlap": 0.5,
}

# Synthetic families that calibrate against a target coverage.
SYNTHETIC_FAMILIES: tuple[str, ...] = ("uniform", "normal", "heavy_tail")

# All families exposed by §2.1, including real_world (no calibration).
LATENCY_FAMILIES: tuple[str, ...] = (*SYNTHETIC_FAMILIES, "real_world")

# Real-world pool name (defined in latency_profiles/pools.yaml). RW3 is the
# small pool with three providers ordered roughly fast / medium / slow.
_RW3_POOL_NAME: str = "rw3"
_RW3_FAST_MEDIUM_SLOW: tuple[str, str, str] = ("WandB", "DeepInfra", "Novita")


# ----------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlapSpec:
    """High-level spec for one §2.1 latency scenario."""

    family: str
    overlap_label: str | None = None
    p50_anchors_ms: tuple[float, ...] = LATENCY_LAYER_P50_MS

    def __post_init__(self) -> None:
        if self.family not in LATENCY_FAMILIES:
            known = ", ".join(LATENCY_FAMILIES)
            raise ValueError(
                f"unknown latency family {self.family!r}; known: {known}"
            )
        if (
            self.family in SYNTHETIC_FAMILIES
            and self.overlap_label not in OVERLAP_TARGETS
        ):
            known = ", ".join(OVERLAP_TARGETS)
            raise ValueError(
                f"unknown overlap label {self.overlap_label!r}; known: {known}"
            )
        if self.family == "real_world" and self.overlap_label is not None:
            raise ValueError("real_world latency scenarios do not accept overlap_label")
        if len(self.p50_anchors_ms) != 3:
            raise ValueError(
                f"p50_anchors_ms must have exactly 3 entries (fast, medium, slow); "
                f"got {self.p50_anchors_ms!r}"
            )
        anchors = self.p50_anchors_ms
        if not (anchors[0] < anchors[1] < anchors[2]):
            raise ValueError(
                f"p50_anchors_ms must be strictly increasing (fast < medium < slow); "
                f"got {anchors!r}"
            )
        if anchors[0] <= 0.0:
            raise ValueError(
                f"p50_anchors_ms[0] (fast) must be positive; got {anchors[0]}"
            )

    @property
    def target_coverage(self) -> float | None:
        if self.overlap_label is None:
            return None
        return OVERLAP_TARGETS[self.overlap_label]


# ----------------------------------------------------------------------------
# Closed-form calibrators
# ----------------------------------------------------------------------------


def calibrate_uniform_ratio(
    p50_fast: float,
    p50_medium: float,
    target_cov: float,
) -> float:
    """Return shared half-width ratio ρ = W / P50 for Uniform.

    Derived from cov(fast, medium) = (0.8(W_a + W_b) - Δ) / (1.6 W_a) with
    W_a = ρ p50_fast, W_b = ρ p50_medium, Δ = p50_medium - p50_fast.

    Solving:
        ρ = (p50_medium - p50_fast) /
            (0.8 (p50_fast + p50_medium) - 1.6 p50_fast · target_cov)
    """
    if not 0.0 <= target_cov < 1.0:
        raise ValueError(
            f"target_cov must be in [0, 1); got {target_cov}"
        )
    delta = p50_medium - p50_fast
    denom = 0.8 * (p50_fast + p50_medium) - 1.6 * p50_fast * target_cov
    if denom <= 0.0:
        raise _infeasible(
            "uniform", p50_fast, p50_medium, target_cov, "denominator non-positive"
        )
    rho = delta / denom
    # Uniform requires low ≥ 0, i.e. ρ ≤ 1. Always true here for the
    # 100/300/1000 anchors with target ≤ 0.5, but check defensively.
    if rho >= 1.0:
        raise _infeasible(
            "uniform",
            p50_fast,
            p50_medium,
            target_cov,
            f"ρ={rho:.4f} would put low ≤ 0 on the fast provider",
        )
    return rho


def calibrate_normal_ratio(
    p50_fast: float,
    p50_medium: float,
    target_cov: float,
) -> float:
    """Return shared sigma ratio ρ = σ / P50 for clipped Normal.

    Same derivation as Uniform but with band half-width = z·σ where z = z(0.90).

    Solving:
        ρ = (p50_medium - p50_fast) /
            (z (p50_fast + p50_medium) - 2z p50_fast · target_cov)
    """
    if not 0.0 <= target_cov < 1.0:
        raise ValueError(f"target_cov must be in [0, 1); got {target_cov}")
    z = _Z_P90
    delta = p50_medium - p50_fast
    denom = z * (p50_fast + p50_medium) - 2.0 * z * p50_fast * target_cov
    if denom <= 0.0:
        raise _infeasible(
            "normal", p50_fast, p50_medium, target_cov, "denominator non-positive"
        )
    return delta / denom


def calibrate_lognormal_sigma(
    p50_fast: float,
    p50_medium: float,
    target_cov: float,
) -> float:
    """Return shared σ_log for LogNormal.

    LogNormal bands in real space:
        B(d) = [P50 · exp(-z σ),  P50 · exp(z σ)]
    where z = z(0.90). Setting y = exp(2 z σ),

        cov(fast, medium) = (p50_fast · y - p50_medium) /
                            (p50_fast · (y - 1))
    Solving:
        y = (p50_medium - target_cov · p50_fast) /
            (p50_fast · (1 - target_cov))
        σ_log = ln(y) / (2 z)
    """
    if not 0.0 <= target_cov < 1.0:
        raise ValueError(f"target_cov must be in [0, 1); got {target_cov}")
    z = _Z_P90
    if target_cov == 0.0:
        # Boundary: y = p50_medium / p50_fast (just-touching bands).
        y = p50_medium / p50_fast
    else:
        numerator = p50_medium - target_cov * p50_fast
        denom = p50_fast * (1.0 - target_cov)
        if numerator <= 0.0 or denom <= 0.0:
            raise _infeasible(
                "heavy_tail",
                p50_fast,
                p50_medium,
                target_cov,
                "log-space y term non-positive",
            )
        y = numerator / denom
    if y <= 1.0:
        raise _infeasible(
            "heavy_tail",
            p50_fast,
            p50_medium,
            target_cov,
            f"y={y:.4f} would yield σ_log ≤ 0",
        )
    return math.log(y) / (2.0 * z)


def _infeasible(
    family: str,
    p50_fast: float,
    p50_medium: float,
    target_cov: float,
    reason: str,
) -> ValueError:
    return ValueError(
        f"latency overlap calibration infeasible: family={family!r} "
        f"P50=({p50_fast}, {p50_medium}) target_cov={target_cov} — {reason}"
    )


# ----------------------------------------------------------------------------
# Distribution builders
# ----------------------------------------------------------------------------


def build_distributions(spec: OverlapSpec) -> tuple["LatencyDistribution", ...]:
    """Return three same-family distributions for fast/medium/slow."""
    if spec.family == "real_world":
        return _load_real_world_distributions()

    p50s = spec.p50_anchors_ms
    target_cov = spec.target_coverage
    if target_cov is None:
        raise ValueError(f"synthetic family {spec.family!r} requires overlap_label")
    p50_fast, p50_medium = p50s[0], p50s[1]

    if spec.family == "uniform":
        rho = calibrate_uniform_ratio(p50_fast, p50_medium, target_cov)
        return tuple(
            Uniform(low=p50 * (1.0 - rho), high=p50 * (1.0 + rho)) for p50 in p50s
        )
    if spec.family == "normal":
        rho = calibrate_normal_ratio(p50_fast, p50_medium, target_cov)
        return tuple(Normal(mean_ms=p50, sigma=rho * p50) for p50 in p50s)
    if spec.family == "heavy_tail":
        sigma_log = calibrate_lognormal_sigma(p50_fast, p50_medium, target_cov)
        return tuple(LogNormal(mu=math.log(p50), sigma=sigma_log) for p50 in p50s)
    raise ValueError(f"unknown family {spec.family!r}")


def _load_real_world_distributions() -> tuple["LatencyDistribution", ...]:
    """Load RW3 fast/medium/slow distributions from the committed pool."""
    # Lazy import to avoid pulling yaml at module-import time.
    from experiments.simulation.latency_profiles import load_pool

    pool = load_pool(_RW3_POOL_NAME)
    missing = [name for name in _RW3_FAST_MEDIUM_SLOW if name not in pool]
    if missing:
        known = ", ".join(sorted(pool))
        raise KeyError(
            f"real_world pool {_RW3_POOL_NAME!r} missing providers {missing}; "
            f"available: {known}"
        )
    return tuple(pool[name] for name in _RW3_FAST_MEDIUM_SLOW)


# ----------------------------------------------------------------------------
# Realised metric helpers
# ----------------------------------------------------------------------------


def measure_band_coverage(
    d_a: "LatencyDistribution",
    d_b: "LatencyDistribution",
) -> float:
    """Asymmetric Q10-Q90 band coverage: |B(d_a) ∩ B(d_b)| / |B(d_a)|."""
    a_lo = d_a.quantile(0.10)
    a_hi = d_a.quantile(0.90)
    b_lo = d_b.quantile(0.10)
    b_hi = d_b.quantile(0.90)
    a_band = a_hi - a_lo
    if a_band <= 0.0:
        raise ValueError(
            f"d_a has degenerate Q10-Q90 band (length={a_band}); cannot measure coverage"
        )
    overlap = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    return overlap / a_band


def measure_normal_clip_fraction(d: "LatencyDistribution") -> float:
    """Fraction of probability mass below MIN_LATENCY_MS (Normal only).

    Returns 0.0 for non-Normal distributions.
    """
    from rwsim.world.distributions import MIN_LATENCY_MS, Normal as _Normal

    if not isinstance(d, _Normal):
        return 0.0
    return float(d.cdf(MIN_LATENCY_MS))


# ----------------------------------------------------------------------------
# Pairwise summary (used by latency_layer.py to populate summary csv columns)
# ----------------------------------------------------------------------------

# Provider names used throughout §2.1.
PROVIDER_NAMES: tuple[str, str, str] = ("api_fast", "api_medium", "api_slow")

# Pair keys exposed in the summary csv (asymmetric — d_a is the divisor).
_PAIR_KEYS: tuple[tuple[str, int, int], ...] = (
    ("fast_medium", 0, 1),
    ("medium_slow", 1, 2),
    ("fast_slow", 0, 2),
)


@dataclass(frozen=True)
class RealisedOverlapSummary:
    """One pairwise-coverage block ready to flatten into csv columns."""

    target_band_coverage_fast_medium: float | None
    realised_band_coverage_fast_medium: float
    realised_band_coverage_medium_slow: float
    realised_band_coverage_fast_slow: float
    normal_clip_fraction_fast: float = 0.0
    normal_clip_fraction_medium: float = 0.0
    normal_clip_fraction_slow: float = 0.0

    def as_dict(self) -> dict[str, float | None]:
        return {
            "target_band_coverage_fast_medium": self.target_band_coverage_fast_medium,
            "realised_band_coverage_fast_medium": self.realised_band_coverage_fast_medium,
            "realised_band_coverage_medium_slow": self.realised_band_coverage_medium_slow,
            "realised_band_coverage_fast_slow": self.realised_band_coverage_fast_slow,
            "normal_clip_fraction_fast": self.normal_clip_fraction_fast,
            "normal_clip_fraction_medium": self.normal_clip_fraction_medium,
            "normal_clip_fraction_slow": self.normal_clip_fraction_slow,
        }


def summarise_realised_overlap(
    spec: OverlapSpec,
    distributions: tuple["LatencyDistribution", ...],
) -> RealisedOverlapSummary:
    """Compute the full realised-overlap block for one scenario."""
    if len(distributions) != 3:
        raise ValueError(
            f"summarise_realised_overlap expects 3 distributions, got {len(distributions)}"
        )
    coverages: dict[str, float] = {}
    for label, idx_a, idx_b in _PAIR_KEYS:
        coverages[label] = measure_band_coverage(
            distributions[idx_a], distributions[idx_b]
        )
    return RealisedOverlapSummary(
        target_band_coverage_fast_medium=spec.target_coverage,
        realised_band_coverage_fast_medium=coverages["fast_medium"],
        realised_band_coverage_medium_slow=coverages["medium_slow"],
        realised_band_coverage_fast_slow=coverages["fast_slow"],
        normal_clip_fraction_fast=measure_normal_clip_fraction(distributions[0]),
        normal_clip_fraction_medium=measure_normal_clip_fraction(distributions[1]),
        normal_clip_fraction_slow=measure_normal_clip_fraction(distributions[2]),
    )


# ----------------------------------------------------------------------------
# Sanity check — calibration round-trip
# ----------------------------------------------------------------------------


def verify_calibration(
    spec: OverlapSpec,
    *,
    tolerance: float = 1e-6,
) -> None:
    """Assert calibrated fast→medium realised coverage matches the target.

    Synthetic families only; real_world has no construction target.
    Raises if the realised coverage drifts more than `tolerance` from
    spec.target_coverage. Used by latency_layer.py to fail-loudly when a
    refactor breaks the closed-form derivation.
    """
    if spec.family == "real_world":
        return
    target = spec.target_coverage
    if target is None:
        return
    distributions = build_distributions(spec)
    realised = measure_band_coverage(distributions[0], distributions[1])
    drift = abs(realised - target)
    if drift > tolerance:
        raise AssertionError(
            f"latency overlap calibration drift {drift:.6e} exceeds tolerance "
            f"{tolerance} (target={target}, realised={realised}); "
            f"spec={spec!r}"
        )


__all__ = [
    "LATENCY_FAMILIES",
    "LATENCY_LAYER_P50_MS",
    "OVERLAP_TARGETS",
    "OverlapSpec",
    "PROVIDER_NAMES",
    "RealisedOverlapSummary",
    "SYNTHETIC_FAMILIES",
    "build_distributions",
    "calibrate_lognormal_sigma",
    "calibrate_normal_ratio",
    "calibrate_uniform_ratio",
    "measure_band_coverage",
    "measure_normal_clip_fraction",
    "summarise_realised_overlap",
    "verify_calibration",
]
