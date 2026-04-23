"""Unified effective cost computation.

The central abstraction of the Joint architecture: every provider, regardless
of tier, has an effective cost that combines its marginal cost with the shadow
price of its scarce resource.

- S_A: c_eff = marginal cost of this request (real billed tokens, USD).
- S_Q: c_eff = psi(z) where psi is the per-request shadow price in USD.
       psi(z) = L * (U/L)^z is the classical primal-dual online knapsack
       threshold. z = quota_used / quota_limit in [0, 1].
- S_C: c_eff = 0 if slot free; otherwise congestion price U per request.

Design notes:
- L and U are per-request USD bounds, matching the primal-dual online
  knapsack framing in which the decision variable is per-request (use one
  quota slot vs pay API). This formulation makes the shadow price compare
  directly with the request's API cost: choose S_Q iff psi(z) < API cost.
  This is the key to value-density saving: short requests (low API cost)
  will prefer S_A once psi rises, while long requests (high API cost)
  continue using S_Q. Without this per-request framing, joint_cheapest
  degenerates to "fill subscription first" because both scale with tokens.
- For a workload with output tokens in [50, 4000] at $3/M, request costs
  range from ~$0.0002 to $0.012. We set L = 0.0001, U = 0.015 as default
  bounds that span the workload.
"""

from __future__ import annotations

from experiment.synthetic.provider import SyntheticProvider
from experiment.synthetic.state import SimState
from experiment.synthetic.workload import SyntheticRequest


DEFAULT_L = 0.0001  # USD per request: floor shadow price
DEFAULT_U = 0.015   # USD per request: ceiling shadow price (~= max request cost)


def _quota_shadow_price(
    z: float, L: float = DEFAULT_L, U: float = DEFAULT_U
) -> float:
    """Exponential threshold psi(z) = L * (U/L)^z. USD per request."""
    if U <= L:
        return L
    return L * (U / L) ** z


def _concurrency_shadow_price(u: float, U: float = DEFAULT_U) -> float:
    """Concurrency congestion price. USD per request.

    0 if u < 0.8. Ramps from 0 to U as u approaches 1. U when saturated.
    """
    if u >= 1.0:
        return U
    if u > 0.8:
        return U * (u - 0.8) / 0.2
    return 0.0


def effective_cost(
    provider: SyntheticProvider,
    request: SyntheticRequest,
    state: SimState,
    L: float = DEFAULT_L,
    U: float = DEFAULT_U,
) -> float:
    """Compute the effective cost of sending `request` to `provider`.

    Returns USD (actual cost of this one request under shadow pricing).
    """
    if provider.is_s_a:
        return provider.marginal_cost(request)
    if provider.is_s_q:
        z = state.quota_fraction(provider)
        return _quota_shadow_price(z, L=L, U=U)
    if provider.is_s_c:
        u = state.concurrency_fraction(provider)
        return _concurrency_shadow_price(u, U=U)
    return 0.0
