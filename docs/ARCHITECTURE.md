# RouteWise Simulator Architecture

This document describes the current simulator architecture after the flat
policy refactor and the sim/real router unification.

## Core Boundary

The simulator has three runtime layers:

1. **World** (`routewise/sim/world/`): provider metadata, latency distributions, quota
   state, and concurrency state.
2. **Engine** (`routewise/sim/engine/`): request loop, admission/accounting, primary
   and backup execution, and in-flight policy callbacks.
3. **Policy** (`routewise/sim/policies/`): routing decisions and policy-owned learning
   state.

Metrics are output types, not execution logic, and live under `routewise/metrics/`.

## One Algorithm, Two Worlds

The RouteWise algorithm is implemented once, in `routewise/core/router.py`
(`RouteWiseRouter`), against the `ProviderView` protocol
(`routewise/core/provider_view.py`): a read-only, request-bound snapshot exposing
only the signals the algorithm consumes (tier, availability, quota fraction,
route/hedge marginal cost, prior latency beliefs). Rolling-profile learning
plus prior/penalty fallbacks live in `routewise/core/beliefs.py`
(`LatencyBeliefs`). All of it is stdlib-only and exported through
`routewise.core`.

Each environment supplies a thin adapter:

- **Simulator** (`routewise/sim/policies/routewise.py`): `_SimProviderView` binds
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
routewise/
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

- `routewise/sim/strategies/`
- `routewise/sim/policies/composer.py`
- `routewise/sim/policies/{value_estimators,cost_routers,latency_routers,hedgers}/`
- `routewise/sim/world/shadow_price.py`
- `routewise/sim/world/workload.py`
- `routewise/sim/registry.py`

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
in the policy-owned `RouteWiseRouter` (`routewise/core/router.py`); the
environment binding lives in `routewise/sim/policies/routewise.py`.

## Presets

The public simulator policy presets are:

- `greedy_cost`
- `greedy_latency`
- `random`
- `ablation_lp_only`
- `ablation_lp_hedging`
- `routewise`

`routewise.sim.policies.build_policy()` is the only preset builder. There is no runtime
compatibility layer for historical strategy names.

## Workloads

Main simulator experiments are trace-driven. `routewise/sim/world/` does not generate
request streams. Experiment runners load trace data through
`experiments.simulation.common.load_workload()` and `routewise/sim/data/loader.py`.

## Offline Stage

The older offline/stage experiment remains separate:

- reusable offline primitives live under `routewise/offline/`
- offline experiment strategies live under `experiments/offline_stage/`
- offline-only predictors live under `experiments/offline_stage/value_estimators/`

This keeps offline research code from reintroducing policy-stage directories
under the online simulator package.
