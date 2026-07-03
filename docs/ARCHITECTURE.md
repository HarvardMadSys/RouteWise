# RouteWise Simulator Architecture

This document describes the current simulator architecture after the flat
policy refactor and the sim/real router unification.

## Core Boundary

The simulator has three runtime layers:

1. **World** (`rwsim/world/`): provider metadata, latency distributions, quota
   state, and concurrency state.
2. **Engine** (`rwsim/engine/`): request loop, admission/accounting, primary
   and backup execution, and in-flight policy callbacks.
3. **Policy** (`rwsim/policies/`): routing decisions and policy-owned learning
   state.

Metrics are output types, not execution logic, and live under `rwsim/metrics/`.

## One Algorithm, Two Worlds

The RouteWise algorithm is implemented once, in `rwsim/core/router.py`
(`RouteWiseRouter`), against the `ProviderView` protocol
(`rwsim/core/provider_view.py`): a read-only, request-bound snapshot exposing
only the signals the algorithm consumes (tier, availability, quota fraction,
route/hedge marginal cost, prior latency beliefs). Rolling-profile learning
plus prior/penalty fallbacks live in `rwsim/core/beliefs.py`
(`LatencyBeliefs`). All of it is stdlib-only and exported through
`routewise.core`.

Each environment supplies a thin adapter:

- **Simulator** (`rwsim/policies/routewise.py`): `_SimProviderView` binds
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
rwsim/
  schemas.py
  scenarios.py
  runner.py
  data/
    loader.py
  engine/
    simulator.py
    state.py
  metrics/
    run.py
  policies/
    base.py
    baselines.py
    routewise.py
    __init__.py
  world/
    capacity.py
    distributions.py
    empirical.py
    providers.py
    scenarios.py
```

The following old implementation surfaces are intentionally absent:

- `rwsim/strategies/`
- `rwsim/policies/composer.py`
- `rwsim/policies/{value_estimators,cost_routers,latency_routers,hedgers}/`
- `rwsim/world/shadow_price.py`
- `rwsim/world/workload.py`
- `rwsim/registry.py`

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
in the policy-owned `RouteWiseRouter` (`rwsim/core/router.py`); the
environment binding lives in `rwsim/policies/routewise.py`.

## Presets

The public simulator policy presets are:

- `greedy_cost`
- `greedy_latency`
- `random`
- `ablation_lp_only`
- `ablation_lp_hedging`
- `routewise`

`rwsim.policies.build_policy()` is the only preset builder. There is no runtime
compatibility layer for historical strategy names.

## Workloads

Main simulator experiments are trace-driven. `rwsim/world/` does not generate
request streams. Experiment runners load trace data through
`experiments.simulation.common.load_workload()` and `rwsim/data/loader.py`.

## Offline Stage

The older offline/stage experiment remains separate:

- reusable offline primitives live under `rwsim/offline/`
- offline experiment strategies live under `experiments/offline_stage/`
- offline-only predictors live under `experiments/offline_stage/value_estimators/`

This keeps offline research code from reintroducing policy-stage directories
under the online simulator package.
