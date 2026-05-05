# RouteWise Simulator Algorithms

The simulator now exposes paper-name policies instead of implementation-stage
strategies.

## Baselines

`greedy_cost`

Routes each request to the currently available provider with the lowest real
marginal request cost. Subscription quota and concurrency providers have zero
real marginal cost, but they are still constrained by their capacity state.

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
