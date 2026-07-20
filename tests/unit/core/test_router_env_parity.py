"""Cross-environment parity: sim and real-eval share one RouteWise algorithm.

Builds one matched provider fleet twice — as simulator ``TieredProvider``s and
as real-eval ``ProviderState``s — warms both with identical TTFT samples,
neutralizes the documented calibration deltas (sampler, hedge overhead,
tie-break mean), and asserts both adapters produce the same routing and
checkpoint-hedging decisions through :class:`llm_routewise.core.router.RouteWiseRouter`.

A final test pins the intended divergences (the knobs that must NOT match).
"""

from __future__ import annotations

import pytest

from experiments.offline_stage.value_estimators import ConstantOutputPredictor
from experiments.real_evaluation.inventory import ProviderSpec
from experiments.real_evaluation.policies import (
    BudgetRangePolicy,
    RequestContext,
    select_checkpoint_backup,
)
from experiments.real_evaluation.transports import TransportConfig
from llm_routewise.capacity import ConcurrencyState, ProviderTier, QuotaState
from llm_routewise.core.hedging import DISPATCH_OVERHEAD_MS
from llm_routewise.core.router import argmax_weight_sampler
from llm_routewise.schemas import Request, RoutingDecision
from llm_routewise.sim.engine.state import SimulationState
from llm_routewise.sim.policies.routewise import RouteWisePolicy
from llm_routewise.sim.world.distributions import Uniform
from llm_routewise.sim.world.providers import TieredProvider

ENVELOPE = (1e-4, 1e-2)
ALPHA_PERCENT = 75
SLO_MS = 2000.0
NOW = 20.0
PROMPT_TOKENS = 1000
OUTPUT_TOKENS = 200

# name -> (input $/token, output $/token, tier, warmup ttft_ms)
FLEET = {
    "api_fast": (3e-6, 6e-6, "api", 110.0),
    "api_cheap": (1e-6, 2e-6, "api", 850.0),
    "sub_quota": (0.0, 0.0, "quota", 350.0),
    "sub_conc": (0.0, 0.0, "concurrency", 500.0),
}
# The hedging primary needs mass beyond both checkpoints (0.5s / 1.0s) and
# beyond the SLO, so "wait for the later checkpoint" stays feasible early.
WARMUP_TTFTS = {
    name: [ttft] * 10 for name, (_i, _o, _t, ttft) in FLEET.items()
} | {"api_cheap": [600.0, 2600.0] * 5}
QUOTA_SIZE = 10
QUOTA_CHARGED = 5


def _sim_policy_and_state() -> tuple[RouteWisePolicy, SimulationState, Request]:
    providers = []
    for name, (in_p, out_p, tier, _ttft) in FLEET.items():
        quota = QuotaState(size=QUOTA_SIZE, window_sec=86400.0) if tier == "quota" else None
        concurrency = ConcurrencyState(limit=4) if tier == "concurrency" else None
        providers.append(
            TieredProvider(
                name=name,
                cost_per_token=in_p,
                input_cost_per_token=in_p,
                output_cost_per_token=out_p,
                ttft_dist=Uniform(_ttft - 10.0, _ttft + 10.0),
                tps_dist=Uniform(100.0, 200.0),
                tier=ProviderTier(tier),
                quota=quota,
                concurrency=concurrency,
            )
        )
    state = SimulationState.from_providers({p.name: p for p in providers})
    state.now = NOW
    for _ in range(QUOTA_CHARGED):
        state.providers["sub_quota"].quota.charge(0.0)

    policy = RouteWisePolicy(
        hedging="probability_target",
        explorer=False,
        alpha=ALPHA_PERCENT / 100.0,
        cost_envelope=ENVELOPE,
        slo_ms=SLO_MS,
        seed=7,
    )
    policy.router.sampler = argmax_weight_sampler  # neutralize RNG delta
    for name, ttfts in WARMUP_TTFTS.items():
        for tick, ttft in enumerate(ttfts):
            policy.router.observe(name, float(tick), ttft)

    request = Request(
        id=1,
        timestamp=NOW,
        request_tokens=PROMPT_TOKENS,
        response_tokens=OUTPUT_TOKENS,
        total_tokens=PROMPT_TOKENS + OUTPUT_TOKENS,
    )
    return policy, state, request


def _real_policy() -> tuple[BudgetRangePolicy, RequestContext]:
    specs = []
    for name, (in_p, out_p, tier, _ttft) in FLEET.items():
        if tier == "api":
            transport_cfg = TransportConfig(
                name=name,
                transport="openrouter",
                model="x",
                input_price_per_m=in_p * 1e6,
                output_price_per_m=out_p * 1e6,
                stream_cancel_billing="stops",
            )
        else:
            transport = "chutes" if tier == "quota" else "featherless"
            transport_cfg = TransportConfig(name=name, transport=transport, model="x")
        specs.append(
            ProviderSpec(
                name=name,
                tier=tier,
                transport_cfg=transport_cfg,
                quota_window_sec=86400 if tier == "quota" else None,
                quota_requests=QUOTA_SIZE if tier == "quota" else None,
                concurrency_limit=4 if tier == "concurrency" else None,
            )
        )
    policy = BudgetRangePolicy(
        specs,
        slo_ms=SLO_MS,
        budget_alpha_percent=ALPHA_PERCENT,
        # The sim request carries realized response_tokens; the real side
        # must predict the same number for costs to match.
        output_predictor=ConstantOutputPredictor(value=float(OUTPUT_TOKENS)),
    )
    policy.set_cost_envelope(ENVELOPE)
    policy.router.sampler = argmax_weight_sampler  # neutralize RNG delta
    for _ in range(QUOTA_CHARGED):
        policy.states["sub_quota"].quota.charge(0.0)
    for name, ttfts in WARMUP_TTFTS.items():
        for tick, ttft in enumerate(ttfts):
            policy.add_sample(name, float(tick), ttft)

    ctx = RequestContext(prompt_tokens=PROMPT_TOKENS, completion_tokens_budget=1)
    return policy, ctx


def test_route_decisions_match_across_environments() -> None:
    sim_policy, sim_state, request = _sim_policy_and_state()
    real_policy, ctx = _real_policy()

    sim_decision = sim_policy.route(request, sim_state)
    real_decision = real_policy.route(NOW, ctx)

    assert sim_decision.primary_provider == real_decision.primary
    sim_weights = sim_decision.metadata["weights"]
    assert set(sim_weights) == set(real_decision.lp_weights)
    for name, weight in sim_weights.items():
        assert real_decision.lp_weights[name] == pytest.approx(weight)
    assert real_decision.budget_usd == pytest.approx(sim_decision.metadata["budget"])
    for name, c in sim_decision.metadata["c_eff"].items():
        assert real_decision.c_eff_map[name] == pytest.approx(c)
    # Same canonical LP outcome, each side's historical status string.
    assert sim_decision.metadata["lp_status"] == "feasible"
    assert real_decision.lp_status == "optimal"


def test_route_decisions_match_at_alpha_zero() -> None:
    sim_policy, sim_state, request = _sim_policy_and_state()
    real_policy, ctx = _real_policy()
    sim_policy.alpha = 0.0
    real_policy.router.alpha = 0.0

    sim_decision = sim_policy.route(request, sim_state)
    real_decision = real_policy.route(NOW, ctx)

    assert sim_decision.primary_provider == real_decision.primary
    assert real_decision.budget_usd == pytest.approx(sim_decision.metadata["budget"])


def test_checkpoint_backup_matches_with_deltas_neutralized() -> None:
    sim_policy, sim_state, request = _sim_policy_and_state()
    real_policy, ctx = _real_policy()
    # Neutralize the intended sim/real hedging deltas on the sim side.
    sim_policy.router.hedge_dispatch_overhead_ms = 0.0
    sim_policy.router.hedge_tiebreak_mean = "belief"

    primary = "api_cheap"
    checkpoints = (0.5, 1.0)
    sim_decision = RoutingDecision(primary_provider=primary, hedge_checkpoints=checkpoints)

    # Early checkpoint: a later checkpoint is still feasible -> both wait.
    early_sim = sim_policy.tick(request, sim_decision, 0.5, sim_state)
    early_real = select_checkpoint_backup(
        primary=primary,
        states=real_policy.states,
        ctx=ctx,
        slo_sec=SLO_MS / 1000.0,
        now=NOW,
        elapsed_sec=0.5,
        future_checkpoints_sec=checkpoints,
        cost_fn=lambda state: real_policy.request_cost_for_state(state, ctx),
    )
    assert early_sim is None
    assert early_real.backup is None
    assert early_real.future_feasible is True

    # Last checkpoint: no future opportunity -> both dispatch the same backup.
    late_sim = sim_policy.tick(request, sim_decision, 1.0, sim_state)
    late_real = select_checkpoint_backup(
        primary=primary,
        states=real_policy.states,
        ctx=ctx,
        slo_sec=SLO_MS / 1000.0,
        now=NOW,
        elapsed_sec=1.0,
        future_checkpoints_sec=checkpoints,
        cost_fn=lambda state: real_policy.request_cost_for_state(state, ctx),
    )
    assert late_sim is not None
    assert late_real.backup is not None
    assert late_sim.backup_provider == late_real.backup
    assert late_sim.metadata["combined_success"] == pytest.approx(
        late_real.success_probability
    )


def test_intended_divergences_are_pinned() -> None:
    """The calibration deltas between environments are explicit router knobs."""
    sim_policy, sim_state, request = _sim_policy_and_state()
    real_policy, _ctx = _real_policy()

    # Hedge calibration deltas.
    fresh_sim = RouteWisePolicy(cost_envelope=ENVELOPE, seed=0)
    assert fresh_sim.router.hedge_dispatch_overhead_ms == DISPATCH_OVERHEAD_MS
    assert fresh_sim.router.hedge_tiebreak_mean == "prior"
    assert real_policy.router.hedge_dispatch_overhead_ms == 0.0
    assert real_policy.router.hedge_tiebreak_mean == "belief"

    # Cold-start beliefs: sim views expose the oracle prior, real views none.
    sim_view = sim_policy._view(sim_state.providers["api_fast"], request, sim_state)
    assert sim_view.prior_ttft_mean_ms(NOW) == pytest.approx(
        sim_state.providers["api_fast"].true_mean_ms(NOW)
    )
    from experiments.real_evaluation.policies import _RealProviderView

    real_view = _RealProviderView(
        real_policy.states["api_fast"], lambda state: 0.0
    )
    assert real_view.prior_ttft_mean_ms(NOW) is None
    assert real_view.prior_ttft_cdf(100.0, NOW) is None
