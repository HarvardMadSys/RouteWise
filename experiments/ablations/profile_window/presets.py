"""Policy presets and naming for the latency-profile window ablation."""

from __future__ import annotations

from typing import Any

from experiments.simulation import common
from rwsim.core.latency_profile import DEFAULT_PROFILE_WINDOW_SEC

DEFAULT_WINDOW_MINUTES: tuple[float, ...] = (1.0, 2.0, 5.0, 15.0, 30.0, 60.0)
DEFAULT_ALPHA_VALUES: tuple[float, ...] = (0.0,)
ORACLE_WINDOW_LABEL = "oracle"
POLICY_FAMILIES = ("lp", "hedge")

assert DEFAULT_PROFILE_WINDOW_SEC / 60.0 in DEFAULT_WINDOW_MINUTES, (
    "window sweep should include the shipped default window"
)


def window_label(window_min: float) -> str:
    """Return the stable window token used inside policy names."""
    return f"w{format_minutes(window_min)}m"


def format_minutes(value: float) -> str:
    """Format a minute value for labels (``1.5`` -> ``1p5``)."""
    return f"{float(value):g}".replace(".", "p")


def parse_minutes(text: str) -> float:
    """Invert :func:`format_minutes`."""
    return float(text.replace("p", "."))


def profile_window_policy_name(
    family: str,
    alpha_value: float,
    window_min: float | None,
) -> str:
    """Return the policy name for one (family, alpha, window) cell.

    ``window_min=None`` denotes the configured-mode oracle reference.
    """
    if family not in POLICY_FAMILIES:
        raise ValueError(f"unknown policy family {family!r}; known: {POLICY_FAMILIES}")
    window_token = (
        ORACLE_WINDOW_LABEL if window_min is None else window_label(window_min)
    )
    return f"routewise_{family}_{common.alpha_label(alpha_value)}_{window_token}"


def parse_profile_window_policy_name(name: str) -> dict[str, Any]:
    """Parse a preset name back into (family, alpha, window_min|None)."""
    parts = name.split("_")
    if len(parts) != 4 or parts[0] != "routewise" or parts[1] not in POLICY_FAMILIES:
        raise ValueError(f"unknown profile-window policy name {name!r}")
    _, family, alpha_token, window_token = parts
    if not alpha_token.startswith("alpha"):
        raise ValueError(f"unknown profile-window policy name {name!r}")
    alpha_value = float(alpha_token.removeprefix("alpha")) / 100.0
    window_min = (
        None
        if window_token == ORACLE_WINDOW_LABEL
        else parse_minutes(window_token.removeprefix("w").removesuffix("m"))
    )
    return {"family": family, "alpha": alpha_value, "window_min": window_min}


def make_profile_window_presets(
    *,
    window_minutes: tuple[float, ...] = DEFAULT_WINDOW_MINUTES,
    alpha_values: tuple[float, ...] = DEFAULT_ALPHA_VALUES,
    include_oracle: bool = True,
    output_predictor: str | dict[str, Any] | None = common.DEFAULT_OUTPUT_PREDICTOR,
) -> dict[str, dict[str, Any]]:
    """Build observed-mode window-sweep presets plus configured-mode oracles.

    The observed presets exercise the rolling profile under ablation
    (``latency_profile_mode="observed"``); ``explorer=True`` matches the
    production RouteWise default so hedge dispatches feed backup samples.
    The oracle presets read the true provider distributions directly and
    bound achievable performance under any window.
    """
    predictor_spec = common._normalize_predictor_arg(output_predictor)
    presets: dict[str, dict[str, Any]] = {}
    for alpha_value in alpha_values:
        for family in POLICY_FAMILIES:
            windows: tuple[float | None, ...] = tuple(window_minutes)
            if include_oracle:
                windows = (*windows, None)
            for window_min in windows:
                params: dict[str, Any] = {
                    "hedging": "probability_target" if family == "hedge" else False,
                    "alpha": float(alpha_value),
                    "cost_envelope": common.WORKLOAD_COST_ENVELOPE,
                }
                if window_min is None:
                    params["latency_profile_mode"] = "configured"
                    params["explorer"] = False
                else:
                    params["latency_profile_mode"] = "observed"
                    params["profile_window_sec"] = float(window_min) * 60.0
                    params["explorer"] = True
                if predictor_spec is not None:
                    params["output_predictor_spec"] = dict(predictor_spec)
                name = profile_window_policy_name(family, alpha_value, window_min)
                presets[name] = {"policy": "RouteWisePolicy", "params": params}
    return presets


__all__ = [
    "DEFAULT_ALPHA_VALUES",
    "DEFAULT_WINDOW_MINUTES",
    "ORACLE_WINDOW_LABEL",
    "POLICY_FAMILIES",
    "format_minutes",
    "make_profile_window_presets",
    "parse_minutes",
    "parse_profile_window_policy_name",
    "profile_window_policy_name",
    "window_label",
]
