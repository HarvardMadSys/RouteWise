"""Preset helpers for the effective-cost ablation harness."""

from __future__ import annotations

from typing import Any

from experiments.simulation.common import WORKLOAD_COST_ENVELOPE, p_label
from rwsim.core.cost import SCARCITY_CURVES, ScarcityCurve

DEFAULT_P_VALUES = (0.5,)
DEFAULT_QUOTA_CURVES: tuple[ScarcityCurve, ...] = (
    "exp_lu",
    "linear_lu",
    "constant_l",
    "constant_u",
)
DEFAULT_CONCURRENCY_CURVES: tuple[ScarcityCurve, ...] = (
    "util_linear_u",
    "exp_lu",
    "linear_lu",
    "constant_l",
    "constant_u",
)
CONCURRENCY_ONLY_QUOTA_CURVE: ScarcityCurve = "exp_lu"
DEFAULT_CONCURRENCY_CURVE: ScarcityCurve = "constant_l"


def ablation_policy_name(
    quota_curve: ScarcityCurve,
    *,
    p: float,
    concurrency_curve: ScarcityCurve = DEFAULT_CONCURRENCY_CURVE,
) -> str:
    """Return a stable policy name for one curve/p combination."""
    return f"effective_cost__q={quota_curve}__c={concurrency_curve}__{p_label(p)}"


def parse_ablation_policy_name(policy: str) -> tuple[ScarcityCurve, ScarcityCurve, float]:
    """Parse a policy name emitted by :func:`ablation_policy_name`."""
    prefix = "effective_cost__"
    if not policy.startswith(prefix):
        raise ValueError(f"not an effective-cost ablation policy: {policy!r}")
    parts = policy.removeprefix(prefix).split("__")
    if len(parts) != 3 or not parts[0].startswith("q=") or not parts[1].startswith("c="):
        raise ValueError(f"invalid effective-cost ablation policy: {policy!r}")

    quota_curve = _validate_curve(parts[0].removeprefix("q="))
    concurrency_curve = _validate_curve(parts[1].removeprefix("c="))
    return quota_curve, concurrency_curve, _parse_p_label(parts[2])


def make_ablation_presets(
    *,
    curves: tuple[ScarcityCurve, ...] = DEFAULT_QUOTA_CURVES,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
    concurrency_curve: ScarcityCurve = DEFAULT_CONCURRENCY_CURVE,
    cost_envelope: tuple[float, float] | str = WORKLOAD_COST_ENVELOPE,
) -> dict[str, dict[str, Any]]:
    """Build ablation-local preset metadata for curve/p sweeps."""
    presets: dict[str, dict[str, Any]] = {}
    for p in p_values:
        for quota_curve in curves:
            name = ablation_policy_name(
                quota_curve,
                p=p,
                concurrency_curve=concurrency_curve,
            )
            presets[name] = {
                "policy": "LPOnlyAblationPolicy",
                "params": {
                    "quota_curve": quota_curve,
                    "concurrency_curve": concurrency_curve,
                    "p": float(p),
                    "cost_envelope": cost_envelope,
                },
            }
    return presets


def make_concurrency_ablation_presets(
    *,
    concurrency_curves: tuple[ScarcityCurve, ...] = DEFAULT_CONCURRENCY_CURVES,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
    quota_curve: ScarcityCurve = CONCURRENCY_ONLY_QUOTA_CURVE,
    cost_envelope: tuple[float, float] | str = WORKLOAD_COST_ENVELOPE,
) -> dict[str, dict[str, Any]]:
    """Build Phase B presets that sweep only the concurrency curve."""
    presets: dict[str, dict[str, Any]] = {}
    for p in p_values:
        for concurrency_curve in concurrency_curves:
            name = ablation_policy_name(
                quota_curve,
                p=p,
                concurrency_curve=concurrency_curve,
            )
            presets[name] = {
                "policy": "LPOnlyAblationPolicy",
                "params": {
                    "quota_curve": quota_curve,
                    "concurrency_curve": concurrency_curve,
                    "p": float(p),
                    "cost_envelope": cost_envelope,
                },
            }
    return presets


def _parse_p_label(label: str) -> float:
    if not label.startswith("p"):
        raise ValueError(f"invalid p label {label!r}")
    try:
        return int(label.removeprefix("p")) / 100.0
    except ValueError as exc:
        raise ValueError(f"invalid p label {label!r}") from exc


def _validate_curve(curve: str) -> ScarcityCurve:
    if curve not in SCARCITY_CURVES:
        known = ", ".join(SCARCITY_CURVES)
        raise ValueError(f"unknown scarcity curve {curve!r}; known curves: {known}")
    return curve  # type: ignore[return-value]


__all__ = [
    "CONCURRENCY_ONLY_QUOTA_CURVE",
    "DEFAULT_CONCURRENCY_CURVE",
    "DEFAULT_CONCURRENCY_CURVES",
    "DEFAULT_P_VALUES",
    "DEFAULT_QUOTA_CURVES",
    "ablation_policy_name",
    "make_ablation_presets",
    "make_concurrency_ablation_presets",
    "parse_ablation_policy_name",
]
