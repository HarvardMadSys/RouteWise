# RouteWise Simulator Architecture and Algorithms

This document describes the current simulator architecture and policy
contracts after the flat policy refactor and the sim/real router unification.

## Core Boundary

The simulator has three runtime layers:

1. **World** (`llm_routewise/sim/world/`): provider metadata, latency distributions, quota
   state, and concurrency state.
2. **Engine** (`llm_routewise/sim/engine/`): request loop, admission/accounting, primary
   and backup execution, and in-flight policy callbacks.
3. **Policy** (`llm_routewise/sim/policies/`): routing decisions and policy-owned learning
   state.

Metrics are output types, not execution logic, and live under `llm_routewise/metrics/`.

## One Algorithm, Two Worlds

The RouteWise algorithm is implemented once, in `llm_routewise/core/router.py`
(`RouteWiseRouter`), against the `ProviderView` protocol
(`llm_routewise/core/provider_view.py`): a read-only, request-bound snapshot exposing
only the signals the algorithm consumes (tier, availability, quota fraction,
route/hedge marginal cost, prior latency beliefs). Rolling-profile learning
plus prior/penalty fallbacks live in `llm_routewise/core/beliefs.py`
(`LatencyBeliefs`). All of it is stdlib-only and exported through
`llm_routewise.core`.

Each environment supplies a thin adapter:

- **Simulator** (`llm_routewise/sim/policies/routewise.py`): `_SimProviderView` binds
  `(Provider, Request, SimulationState)`; priors are the provider's true
  distribution (oracle fallback / `configured` mode); sampling uses the
  policy's persistent seeded numpy RNG.
- **Real eval** (`experiments/real_evaluation/policies.py`):
  `_RealProviderView` binds `(ProviderState, RequestContext)`; there are no
  priors (empty windows fall back to a large finite penalty), failed attempts
  enter the beliefs as 60s error penalties, and sampling uses a per-call
  time-seeded RNG. Locking, capacity charging, transports, and the recorder
  boundary stay in the harness.

Environment differences are injected router knobs (sampler, hedge dispatch
overhead, tie-break mean, penalties), never branches inside the algorithm.
Quota/concurrency ledger implementations are intentionally NOT unified; they
stay behind `ProviderView.is_available` / `quota_fraction_used`.
`tests/unit/core/test_router_env_parity.py` pins both the parity and the
intended divergences.

## Target Tree

```text
llm_routewise/
  capacity.py          # quota/concurrency primitives shared by sim + real
  schemas.py           # Request / RoutingOutcome / scenario config contracts
  const.py             # protocol constants (SLO, hedge schedule fractions)
  core/                # the RouteWise algorithm (stdlib-only)
    router.py
    provider_view.py
    beliefs.py
    lp.py  hedging.py  cost.py  pricing.py  latency_profile.py  types.py
  metrics/
    run.py
  offline/             # offline-stage primitives
  sim/                 # the simulated world
    scenarios.py
    runner.py
    data/
      loader.py
    engine/
      simulator.py
      state.py
    policies/
      base.py
      baselines.py
      routewise.py
      __init__.py
    world/
      distributions.py
      empirical.py
      providers.py
      scenarios.py
```

The following old implementation surfaces are intentionally absent:

- `llm_routewise/sim/strategies/`
- `llm_routewise/sim/policies/composer.py`
- `llm_routewise/sim/policies/{value_estimators,cost_routers,latency_routers,hedgers}/`
- `llm_routewise/sim/world/shadow_price.py`
- `llm_routewise/sim/world/workload.py`
- `llm_routewise/sim/registry.py`

## Policy Contract

Policies implement:

```python
class Policy(Protocol):
    def route(self, request, state) -> RoutingDecision: ...
    def tick(self, request, decision, elapsed, state) -> HedgeDispatch | None: ...
    def observe(self, request, decision, outcome) -> None: ...
```

`route()` chooses the primary provider and declares any hedge checkpoints in
`RoutingDecision.hedge_checkpoints`.

`tick()` is called while a request is still in flight. This is required for
RouteWise hedging because success probability must be re-evaluated at multiple
checkpoints as queue depth, capacity, and observed profiles change.

`observe()` updates policy-owned learning state after completion.

## State Ownership

`SimulationState` contains world state visible to every policy:

- current simulated time
- provider mapping
- capacity view
- prefix-cache user/provider memory

Policy-specific quantities stay inside the policy implementation. The
RouteWise algorithm state (latency beliefs, LP weights, hedge decisions) lives
in the policy-owned `RouteWiseRouter` (`llm_routewise/core/router.py`); the
environment binding lives in `llm_routewise/sim/policies/routewise.py`.

## Presets

The public simulator policy presets are:

- `greedy_cost`
- `greedy_latency`
- `random`
- `ablation_lp_only`
- `ablation_lp_hedging`
- `routewise`

`llm_routewise.sim.policies.build_policy()` is the only preset builder. There is no runtime
compatibility layer for historical strategy names.

## Workloads

Main simulator experiments are trace-driven. `llm_routewise/sim/world/` does not generate
request streams. Experiment runners load trace data through
`experiments.simulation.common.load_workload()` and `llm_routewise/sim/data/loader.py`.

## Offline Stage

The older offline/stage experiment remains separate:

- reusable offline primitives live under `llm_routewise/offline/`
- offline experiment strategies live under `experiments/offline_stage/`
- offline-only predictors live under `experiments/offline_stage/value_estimators/`

This keeps offline research code from reintroducing policy-stage directories
under the online simulator package.

## Baselines

`greedy_cost`

Routes each request to the currently available provider with the lowest real
marginal request cost. Subscription quota and concurrency providers have zero
real marginal cost, but they are still constrained by their capacity state.
When multiple feasible providers share the same marginal cost (typically an
`S_C` and an `S_Q` provider both at 0), ties are broken deterministically by
tier priority: `S_C` first, then `S_Q`, then `S_A`. This order reflects that
concurrency slots are perishable while quota is bankable. Lower expected TTFT
and then provider name break any remaining ties.

`greedy_latency`

Routes each request to the currently available provider with the lowest current
median TTFT.

`random`

Routes each request uniformly over currently available providers. This is a
control baseline, not a RouteWise component.

## RouteWise Ablations

`ablation_lp_only`

Runs the RouteWise LP body without hedging and without explorer feedback.

`ablation_lp_hedging`

Runs the RouteWise LP body plus probability-target hedging. It does not feed
backup observations into the latency profile.

`routewise`

Full RouteWise: LP body, multi-checkpoint hedging, and hedge-as-probe explorer
feedback.

## RouteWise Body

For each request, RouteWise:

1. Computes effective cost for currently feasible providers.
2. Solves a cost-budgeted latency objective.
3. Samples the primary provider from the resulting mixture.

Effective cost is policy-owned logic. API providers use real marginal request
cost. Quota and concurrency providers add RouteWise shadow prices based on
current capacity usage. These values are not world facts and do not live in the
engine.

## Hedging

RouteWise does not decide hedging once at dispatch time. `route()` declares
checkpoint times, and the simulator calls `tick()` while the request is in
flight. At each checkpoint RouteWise re-evaluates whether dispatching a backup
improves SLO success enough to justify the added cost.

The simulator owns execution mechanics. The policy owns the decision of whether
to hedge and which backup provider to request.

## Explorer

Explorer is implemented as hedge-as-probe. When a backup is actually dispatched,
RouteWise can feed the backup TTFT observation into its latency profile. There
is no separate synthetic probing path in the simulator.

## Learning State

RouteWise-owned learning state includes:

- rolling latency profiles
- LP mixture bookkeeping
- shadow-price helper calculations
- hedge/explorer observations

The engine only exposes world state through `SimulationState` and completed
outcomes through `observe()`.
