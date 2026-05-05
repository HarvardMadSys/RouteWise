# RouteWise Simulator Architecture

This document describes the current simulator architecture after the flat
policy refactor.

## Core Boundary

The simulator has three runtime layers:

1. **World** (`rwsim/world/`): provider metadata, latency distributions, quota
   state, and concurrency state.
2. **Engine** (`rwsim/engine/`): request loop, admission/accounting, primary
   and backup execution, and in-flight policy callbacks.
3. **Policy** (`rwsim/policies/`): routing decisions and policy-owned learning
   state.

Metrics are output types, not execution logic, and live under `rwsim/metrics/`.

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

Policy-specific quantities stay inside the policy implementation. RouteWise
shadow prices, latency profiles, LP weights, and hedge/explorer decisions live
inside `rwsim/policies/routewise.py`.

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
`experiments.simulation.lp_budget_eval.generate_scenario_workload()` and
`rwsim/data/loader.py`.

## Offline Stage

The older offline/stage experiment remains separate:

- reusable offline primitives live under `rwsim/offline/`
- offline experiment strategies live under `experiments/offline_stage/`
- offline-only predictors live under `experiments/offline_stage/value_estimators/`

This keeps offline research code from reintroducing policy-stage directories
under the online simulator package.
