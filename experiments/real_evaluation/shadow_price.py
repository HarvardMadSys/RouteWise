"""Shadow-price adapters for real-eval tiered providers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from experiments.real_evaluation.transports import compute_request_cost_usd
from rwsim.policies.effective_cost_kernel import scarcity_price

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from experiments.real_evaluation.inventory import ProviderSpec, ProviderState


def quota_shadow_price(
    state: ProviderState,
    now: float,
    *,
    U: float,
    L: float,
) -> float:
    """Exponential threshold ``L * (U/L)^z`` for a quota-limited provider."""
    if state.quota is None:
        return 0.0

    return scarcity_price("exp_lu", state.quota.fraction_used(now), L=L, U=U)


def concurrency_shadow_price(
    state: ProviderState,
    now: float,
    *,
    U: float,
    L: float,
    alpha: float = 1.0,
) -> float:
    """Constant price ``L`` for a concurrency-limited reusable-capacity provider."""
    del now, alpha
    if state.concurrency is None:
        return 0.0
    return scarcity_price("constant_l", 0.0, L=L, U=U)


def request_marginal_cost(
    spec: ProviderSpec, ctx_prompt_tokens: int, ctx_completion_tokens: int
) -> float:
    """Linear-pricing marginal cost for one request.

    For S_A providers this is the metered USD per request. For S_Q / S_C
    subscription providers this returns 0 (sunk subscription cost).

    Takes a ``ProviderSpec`` (static) rather than ``ProviderState``
    (dynamic) — pricing is purely a function of the spec.
    """
    if spec.tier != "api":
        return 0.0
    # TODO(routewise-cache): route-time API cost should account for predicted
    # prefix-cache hits. Per Juncheng's May notes, keep latency profiles
    # unchanged for now and discount only the cached input-token component:
    # (prompt_tokens - cached_tokens) * input_price
    # + cached_tokens * cache_hit_input_price
    # + completion_tokens * output_price.
    return compute_request_cost_usd(
        ctx_prompt_tokens,
        ctx_completion_tokens,
        spec.input_price_per_m,
        spec.output_price_per_m,
    )


def effective_cost(
    state: ProviderState,
    request_cost_usd: float,
    now: float,
    *,
    U: float,
    L: float,
    concurrency_alpha: float = 1.0,
) -> float:
    """Paper-formula piecewise effective cost used by the joint router.

    Matches :func:`rwsim.policies.routewise.effective_cost`:

    * ``S_A`` / ``tier="api"``: real API billing cost.
    * ``S_Q`` / ``tier="quota"``: quota opportunity cost ``psi(z)``.
    * ``S_C`` / ``tier="concurrency"``: concurrency opportunity cost
      ``lambda(u)``.

    Extra admission constraints on a provider still gate availability through
    :meth:`ProviderState.is_available`; they do not add another term to the
    routing effective cost.
    """
    if state.spec.tier == "api":
        return request_cost_usd
    if state.spec.tier == "quota":
        return quota_shadow_price(state, now, U=U, L=L)
    if state.spec.tier == "concurrency":
        return concurrency_shadow_price(state, now, U=U, L=L, alpha=concurrency_alpha)
    raise ValueError(f"Unsupported provider tier for real-eval effective cost: {state.spec.tier!r}")


def calibrate_envelopes(
    specs: Sequence[ProviderSpec],
    ctx_prompt_tokens: int,
    ctx_completion_tokens: int,
    *,
    floor_ratio: float = 1e-3,
    cost_fn: Callable[[ProviderSpec], float] | None = None,
) -> tuple[float, float]:
    """Compute ``(L, U)`` envelopes from API providers in the inventory.

    ``U`` is the most expensive API request cost; ``L = max(U * floor_ratio, 1e-9)``.
    Mirrors :func:`rwsim.policies.routewise.calibrate_envelopes` (the simulator
    version uses ``cost_per_token * typical_tokens`` since its providers expose
    a per-token rate; we use ``input + output`` linear pricing because that's
    what real provider APIs charge).
    """
    api_costs: list[float] = []
    for spec in specs:
        if spec.tier == "api" and (
            spec.input_price_per_m > 0 or spec.output_price_per_m > 0
        ):
            if cost_fn is not None:
                api_costs.append(cost_fn(spec))
            else:
                api_costs.append(
                    compute_request_cost_usd(
                        ctx_prompt_tokens,
                        ctx_completion_tokens,
                        spec.input_price_per_m,
                        spec.output_price_per_m,
                    )
                )
    if not api_costs:
        return (1e-6, 1e-3)
    U = float(max(api_costs))
    L = float(max(U * floor_ratio, 1e-9))
    return (L, U)


def workload_cost_envelope(
    specs: Sequence[ProviderSpec],
    prompt_tokens_list: Sequence[int],
    completion_tokens_list: Sequence[int],
    *,
    lower_percentile: float = 10.0,
    upper_percentile: float = 90.0,
    floor_ratio: float = 1e-3,
) -> tuple[float, float]:
    """Paper-formula ``(L, U)`` from the workload's cheapest-API cost distribution.

    Mirrors :func:`experiments.simulation.common.workload_cost_envelope`. For
    each request, the cheapest per-request cost across api-tier providers is
    taken; ``L`` and ``U`` are then the ``lower_percentile`` and
    ``upper_percentile`` of that distribution. Falls back to ``U * floor_ratio``
    when the lower percentile is degenerate or non-positive.

    The single-request shortcut in :func:`calibrate_envelopes` (which uses
    ``L = U * floor_ratio`` directly) is structurally wrong for the RouteWise
    shadow-price formula ``psi(z) = L * (U/L)^z`` because it makes ``L`` 1000x
    smaller than the paper intended, so subscription tiers stay near-zero
    in ``c_eff`` across most of the quota and the LP picks them blindly.
    """
    api_specs = [
        s
        for s in specs
        if s.tier == "api" and (s.input_price_per_m > 0 or s.output_price_per_m > 0)
    ]
    if not api_specs:
        return (1e-6, 1e-3)
    if len(prompt_tokens_list) != len(completion_tokens_list):
        raise ValueError(
            "workload_cost_envelope: prompt_tokens_list and "
            "completion_tokens_list must be the same length"
        )

    values: list[float] = []
    for prompt_tokens, completion_tokens in zip(
        prompt_tokens_list, completion_tokens_list, strict=False
    ):
        costs = [
            compute_request_cost_usd(
                int(prompt_tokens),
                int(completion_tokens),
                s.input_price_per_m,
                s.output_price_per_m,
            )
            for s in api_specs
        ]
        positive_costs = [c for c in costs if c > 0.0 and math.isfinite(c)]
        if positive_costs:
            values.append(min(positive_costs))

    if not values:
        return (1e-6, 1e-3)

    arr = np.asarray(values, dtype=float)
    L = float(np.percentile(arr, lower_percentile))
    U = float(np.percentile(arr, upper_percentile))
    if (
        not math.isfinite(L)
        or not math.isfinite(U)
        or U <= 0.0
        or L <= 0.0
        or L >= U
    ):
        L = float(max(U * floor_ratio, 1e-9))
    return (L, U)


__all__ = [
    "calibrate_envelopes",
    "concurrency_shadow_price",
    "effective_cost",
    "quota_shadow_price",
    "request_marginal_cost",
    "workload_cost_envelope",
]
