# RouteWise Algorithm Structure

This document defines the target algorithm decomposition for RouteWise routing
policies. Its purpose is to prevent the codebase from growing one separate
strategy implementation per experiment condition.

The target design is pipeline-based:

```text
value estimation + cost routing + latency control + hedging
```

Most named strategies should be expressible as different pipeline
configurations, not separate simulation loops.

## Core Interfaces

### Request

A `Request` describes one simulated inference request.

Expected fields:

- Arrival time.
- Input length.
- Estimated output length.
- Optional true output length.
- Optional request class or tenant.
- Optional SLO target.

The exact schema should live in `rwsim/schemas.py` or
`rwsim/world/workload.py`, depending on whether it is shared across modules.

### RoutingDecision

A `RoutingDecision` is the policy output for one request.

Expected fields:

- Primary provider.
- Optional hedge provider.
- Optional hedge trigger time or hedge probability.
- Optional debug metadata such as shadow prices, value estimates, and filtered
  provider sets.

The schema should live in `rwsim/schemas.py` or `rwsim/policies/base.py`.

### Policy

A policy should choose a routing decision. It should not own the simulation
loop.

Target shape:

```python
class Policy(Protocol):
    def route(self, request: Request, state: SimulationState) -> RoutingDecision:
        ...
```

The engine owns execution, state updates, and metric recording.

## Pipeline Stages

### Value Estimators

Value estimators predict request value needed by downstream routing stages.
The first use case is output length or expected token cost estimation.

Candidate modules:

```text
rwsim/policies/value_estimators/
  fixed.py
  oracle.py
  ema.py
  histogram.py
```

Research questions:

- How much does output length estimation affect routing quality?
- When is a naive estimator enough?
- Which estimator failure modes materially change cost, latency, or SLO
  violation rate?

This stage should support the estimator ablation experiment directly.

### Cost Routers

Cost routers choose providers using price, quota, concurrency, and shadow
price information.

Candidate modules:

```text
rwsim/policies/cost_routers/
  fixed.py
  round_robin.py
  primal_dual.py
  two_layer.py
  cheapest_effective.py
```

Expected responsibilities:

- Base provider selection.
- Quota-aware effective cost.
- Concurrency-aware effective cost.
- Optional exploration over low-cost providers.

Cost routers should not implement hedging.

### Latency Routers

Latency routers filter or re-rank provider choices using latency constraints.

Candidate modules:

```text
rwsim/policies/latency_routers/
  none.py
  p95_filter.py
  p50_band.py
  lp_cost_budget.py
```

Expected responsibilities:

- SLO feasibility filters.
- P50 or P95 band constraints.
- LP-based latency-cost tradeoff decisions.

Latency routers should expose enough debug metadata to explain why a provider
was accepted or rejected.

### Hedgers

Hedgers decide whether to issue a backup request.

Candidate modules:

```text
rwsim/policies/hedgers/
  none.py
  probability_targeted.py
  smart_economic.py
```

Expected responsibilities:

- Hedge probability or trigger threshold.
- Hedge provider selection.
- Incremental cost and SLO tradeoff calculation.

Hedgers should not own primary provider selection.

### Composer

The composer builds a policy pipeline from config.

Candidate module:

```text
rwsim/policies/composer.py
```

Target shape:

```yaml
policy:
  value_estimator:
    name: ema
  cost_router:
    name: primal_dual
  latency_router:
    name: p95_filter
  hedger:
    name: smart_economic
```

The composer should make ablations cheap. For example, changing from
`joint_nohedge` to `joint_hedge` should be a config difference, not a new
simulation loop.

## Mapping Current Strategies To Pipeline Configs

This mapping is provisional. It is a migration alias table, not a behavior
switch. When a current strategy has a legacy implementation detail, the alias
should preserve that behavior until a separate research change intentionally
changes the algorithm.

| Current strategy | Value estimator | Cost router | Latency router | Hedger |
| --- | --- | --- | --- | --- |
| `cheapest_fixed` | fixed or none | fixed cheapest | none | none |
| `fastest_fixed` | fixed or none | fixed fastest | none | none |
| `round_robin` | fixed or none | round robin | none | none |
| `oracle_per_window` | oracle | oracle window optimizer | optional | none |
| `lp_mix` | fixed or estimator | LP mix | LP cost budget | none |
| `lp_hedge` | fixed or estimator | LP mix | LP cost budget | smart economic |
| `lp_explorer` | estimator plus probes | LP explorer | LP cost budget | smart economic |
| `lp_explorer_no_probe` | estimator without probes | LP explorer | LP cost budget | smart economic |
| `v2_only` | estimator | V2 cost router | V2 latency control | none |
| `v2_p50_hedge` | estimator | V2 cost router | P50 band | smart economic |
| `v2_explorer` | estimator plus probes | V2 explorer | P50 band | smart economic |
| `v2_explorer_no_probe` | estimator without probes | V2 explorer | P50 band | smart economic |
| `two_layer` | fixed or estimator | two layer | tier gate | none |
| `joint_nohedge` | estimator | cheapest effective | P95 SLO filter | none |
| `joint_hedge` | estimator | cheapest effective | P95 SLO filter | smart economic |
| `joint_p50band_nohedge` | estimator | cheapest effective | P50 band | none |
| `joint_p50band_hedge` | estimator | cheapest effective | P50 band | smart economic |

The sidecar LP-budget experiment already contains a newer probability-targeted
hedge variant. That should migrate as a distinct hedger stage, not be folded
silently into legacy `lp_hedge` or `v2_p50_hedge` aliases.

## Engine Boundary

The engine should be the only component that iterates over requests and
updates simulation state.

Target shape:

```python
for request in workload:
    decision = policy.route(request, state)
    outcome = executor.execute(decision, request, state)
    state.update(outcome)
    metrics.record(request, decision, outcome)
```

Policies may maintain internal estimator state, but they should expose that
state through a narrow interface and should not directly mutate provider quota
or concurrency state unless the engine explicitly passes a mutable state view.

## Metrics Stream

The engine should emit a uniform metrics stream for every strategy.

Expected per-request fields:

- Request id and timestamp.
- Primary provider.
- Optional hedge provider.
- Realized TTFT.
- Realized cost.
- SLO violation indicator.
- Rejection indicator.
- Quota state snapshot.
- Concurrency state snapshot.
- Shadow price snapshot.
- Policy debug metadata.

Aggregated metrics should be derived from this stream rather than manually
assembled inside each strategy runner.

This should replace the current pattern where `StrategyRun` has many optional
fields that only some runners populate.

## Experiment Implications

The policy decomposition should directly support:

- Output length estimator ablations.
- Hedging ablations.
- Latency router ablations.
- Quota and concurrency stress tests.
- Config-driven paper scenarios.

When adding a new paper experiment, prefer adding a config and, if needed, a
small reusable pipeline stage. Avoid adding a new monolithic strategy unless
the algorithm is genuinely new.

## Migration Checklist

For each strategy migration:

1. Identify the current runner and all state it mutates.
2. Split provider selection, latency filtering, and hedging into stages.
3. Move the request loop into `rwsim/engine/`.
4. Emit a uniform metrics stream.
5. Reproduce the old strategy result under golden comparison.
6. Convert the old strategy name into a pipeline config alias.

The strategy is not considered migrated until its legacy runner can be
removed or converted into a thin wrapper around the pipeline composer.
