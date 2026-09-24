"""Policy presets and naming for the stale-telemetry robustness experiment."""

from __future__ import annotations

from typing import Any

from experiments.simulation import common, hedging
from llm_routewise.core.latency_profile import DEFAULT_PROFILE_WINDOW_SEC

BELIEF_MODES: tuple[str, ...] = ("stale", "observed", "fresh")
POLICY_FAMILIES: tuple[str, ...] = ("lp", "hedge")
# Providers share one price in the §2.2 scenarios, so the LP budget knob has
# no effect on routing; keep the §2.2 convention and record it in the rows.
ROUTEWISE_ALPHA: float = hedging.DEFAULT_ROUTEWISE_ALPHA
DEFAULT_PROFILE_WINDOW_MIN: float = DEFAULT_PROFILE_WINDOW_SEC / 60.0

_POLICY_PREFIX = "routewise_"
_OBSERVED_PREFIX = "observed"


def format_number(value: float) -> str:
    """Format a number for stable labels (``2.5`` -> ``2p5``, ``10.0`` -> ``10``)."""
    return f"{float(value):g}".replace(".", "p")


def parse_number(text: str) -> float:
    """Invert :func:`format_number`."""
    return float(text.replace("p", "."))


def belief_label(mode: str, window_min: float | None = None) -> str:
    """Return the belief-mode token used inside policy names."""
    if mode not in BELIEF_MODES:
        raise ValueError(f"unknown belief mode {mode!r}; known: {BELIEF_MODES}")
    if mode != "observed":
        return mode
    if window_min is None or window_min <= 0.0:
        raise ValueError(f"observed belief mode needs a positive window, got {window_min!r}")
    return f"{_OBSERVED_PREFIX}{format_number(window_min)}m"


def policy_name(family: str, mode: str, window_min: float | None = None) -> str:
    """Return the policy name for one (family, belief mode) cell."""
    if family not in POLICY_FAMILIES:
        raise ValueError(f"unknown policy family {family!r}; known: {POLICY_FAMILIES}")
    return f"{_POLICY_PREFIX}{family}_{belief_label(mode, window_min)}"


def parse_policy_name(name: str) -> dict[str, Any]:
    """Parse a preset name back into ``family``, ``belief_mode``, ``window_min``."""
    parts = name.split("_", 2)
    if len(parts) != 3 or parts[0] + "_" != _POLICY_PREFIX or parts[1] not in POLICY_FAMILIES:
        raise ValueError(f"unknown stale-telemetry policy name {name!r}")
    _, family, belief = parts
    if belief in ("stale", "fresh"):
        return {"family": family, "belief_mode": belief, "window_min": None}
    if belief.startswith(_OBSERVED_PREFIX) and belief.endswith("m"):
        window_min = parse_number(belief.removeprefix(_OBSERVED_PREFIX).removesuffix("m"))
        return {"family": family, "belief_mode": "observed", "window_min": window_min}
    raise ValueError(f"unknown stale-telemetry policy name {name!r}")


def make_presets(
    *,
    window_min: float = DEFAULT_PROFILE_WINDOW_MIN,
    output_predictor: str | dict[str, Any] | None = common.DEFAULT_OUTPUT_PREDICTOR,
) -> dict[str, dict[str, Any]]:
    """Build the LP-only / LP+hedging presets for the three telemetry regimes.

    ``stale`` runs :class:`StaleBeliefRouteWisePolicy` (priors frozen at the
    baseline distribution). ``fresh`` is plain ``configured``-mode RouteWise,
    whose priors track the true distribution instantly. ``observed`` is the
    production rolling profile with a ``window_min``-minute window; like the
    profile-window ablation it keeps ``explorer=True`` so hedge backups also
    feed the profile.
    """
    predictor_spec = common._normalize_predictor_arg(output_predictor)
    presets: dict[str, dict[str, Any]] = {}
    for mode in BELIEF_MODES:
        for family in POLICY_FAMILIES:
            params: dict[str, Any] = {
                "hedging": "probability_target" if family == "hedge" else False,
                "alpha": ROUTEWISE_ALPHA,
                "cost_envelope": common.WORKLOAD_COST_ENVELOPE,
            }
            if mode == "observed":
                params["latency_profile_mode"] = "observed"
                params["profile_window_sec"] = float(window_min) * 60.0
                params["explorer"] = True
            else:
                params["latency_profile_mode"] = "configured"
                params["explorer"] = False
            if predictor_spec is not None:
                params["output_predictor_spec"] = dict(predictor_spec)
            presets[policy_name(family, mode, window_min)] = {
                "policy": ("StaleBeliefRouteWisePolicy" if mode == "stale" else "RouteWisePolicy"),
                "params": params,
            }
    return presets


__all__ = [
    "BELIEF_MODES",
    "DEFAULT_PROFILE_WINDOW_MIN",
    "POLICY_FAMILIES",
    "ROUTEWISE_ALPHA",
    "belief_label",
    "format_number",
    "make_presets",
    "parse_number",
    "parse_policy_name",
    "policy_name",
]
