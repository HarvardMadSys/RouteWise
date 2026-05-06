"""Shadow-price functions for real-eval tiered providers.

This module **mirrors the formulas** in :mod:`rwsim.policies.routewise` but
operates on :class:`experiments.real_evaluation.inventory.ProviderState`
instead of ``rwsim.world.providers.TieredProvider``. The two implementations
must stay in lock-step:

- ``quota_shadow_price``        : ``L * (U/L)^z`` with ``z`` = quota fraction used
- ``concurrency_shadow_price``  : ``U * u^alpha`` with ``u`` = active fraction
- ``effective_cost``            : paper-formula piecewise cost by provider tier
- ``calibrate_envelopes``       : ``U = max api cost``, ``L = max(U*floor_ratio, 1e-9)``

If you change a formula here, change it in ``rwsim.policies.routewise`` too,
and update the real-eval policy tests.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from experiments.real_evaluation.transports import compute_request_cost_usd

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    z = state.quota.fraction_used(now)
    z = min(max(z, 0.0), 0.9999)

    if L <= 0 or U <= 0 or U <= L:
        raise ValueError(f"Require 0 < L < U; got L={L}, U={U}")

    return L * math.pow(U / L, z)


def concurrency_shadow_price(
    state: ProviderState,
    now: float,
    *,
    U: float,
    alpha: float = 1.0,
) -> float:
    """Congestion price ``U * u^alpha`` for a concurrency-limited provider."""
    if state.concurrency is None:
        return 0.0

    u = state.concurrency.utilization(now)
    return U * math.pow(u, alpha)


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
        return concurrency_shadow_price(state, now, U=U, alpha=concurrency_alpha)
    raise ValueError(f"Unsupported provider tier for real-eval effective cost: {state.spec.tier!r}")


def calibrate_envelopes(
    specs: Sequence[ProviderSpec],
    ctx_prompt_tokens: int,
    ctx_completion_tokens: int,
    *,
    floor_ratio: float = 1e-3,
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


__all__ = [
    "calibrate_envelopes",
    "concurrency_shadow_price",
    "effective_cost",
    "quota_shadow_price",
    "request_marginal_cost",
]
