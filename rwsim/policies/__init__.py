"""Flat policy exports and preset builder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rwsim.policies.base import Policy
from rwsim.policies.baselines import BaselinePolicy
from rwsim.policies.routewise import RouteWisePolicy

POLICY_CLASSES = {
    "BaselinePolicy": BaselinePolicy,
    "RouteWisePolicy": RouteWisePolicy,
}

DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    "greedy_cost": {
        "policy": "BaselinePolicy",
        "params": {"mode": "greedy_cost"},
    },
    "greedy_latency": {
        "policy": "BaselinePolicy",
        "params": {"mode": "greedy_latency"},
    },
    "random": {
        "policy": "BaselinePolicy",
        "params": {"mode": "random"},
    },
    "ablation_lp_only": {
        "policy": "RouteWisePolicy",
        "params": {"hedging": False, "explorer": False},
    },
    "ablation_lp_hedging": {
        "policy": "RouteWisePolicy",
        "params": {"hedging": "probability_target", "explorer": False},
    },
    "routewise": {
        "policy": "RouteWisePolicy",
        "params": {"hedging": "probability_target", "explorer": True},
    },
}


def available_policies(presets: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Return known policy preset names."""
    return tuple((presets or DEFAULT_PRESETS).keys())


def build_policy(
    name: str,
    presets: Mapping[str, Any] | None = None,
    *,
    seed: int | None = None,
) -> Policy:
    """Instantiate a policy from a preset table."""
    preset_table = presets or DEFAULT_PRESETS
    try:
        preset = preset_table[name]
    except KeyError as exc:
        known = ", ".join(available_policies(preset_table))
        raise KeyError(f"Unknown policy preset {name!r}. Known: {known}") from exc

    cls = POLICY_CLASSES[preset["policy"]]
    params = dict(preset.get("params", {}))
    if seed is not None:
        params.setdefault("seed", seed)
    return cls(**params)


__all__ = [
    "BaselinePolicy",
    "DEFAULT_PRESETS",
    "POLICY_CLASSES",
    "Policy",
    "RouteWisePolicy",
    "available_policies",
    "build_policy",
]
