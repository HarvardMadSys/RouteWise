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

### Value Estimator

#### Canonical Definition (authoritative)

For request `i`, the value-estimator stage predicts output length from only
arrival-time features:

    x_i = (n_in(i), model_i, timestamp_i, tenant_i, class_i, ...)          [eq. 1]
    L_hat_i = E[Y_i | x_i, H_{i-1}]                                       [eq. 2]
    (L10_i, L50_i, L90_i) = Quantiles(Y_i | x_i, H_{i-1})                  [eq. 3]

where `Y_i` is true `response_tokens` and `H_{i-1}` is the completed-request
history before request `i`. Ground-truth `Y_i` is only available after the
routing decision and execution outcome:

    H_i = H_{i-1} union {(x_i, Y_i, ttft_i, latency_i, provider_i)}         [eq. 4]

Current canonical estimators:

    fixed:      L_hat_i = L0                                               [eq. 5]
    oracle:     L_hat_i = Y_i                                              [eq. 6]
    ema:        m_i = (1-alpha) m_{i-1} + alpha Y_i                         [eq. 7]
                L50_i = max(1, m_i)
                L10_i = max(1, m_i - 1.28 sigma_i)
                L90_i = max(L50_i, m_i + 1.28 sigma_i)                     [eq. 8]
    histogram:  Lq_i = empirical_quantile_q(H_{i-1}, backoff(x_i))          [eq. 9]

For duration-aware stages, use conservative decode-time upper confidence:

    D_ucb_i = TTFT90_i + L90_i / TPS10_i                                  [eq. 10]

#### Pipeline Role

Value estimator = request-value prediction. It does not choose a provider, does
not apply quota/concurrency pricing, and does not hedge. It writes estimates
such as `L_hat`, `L10/L50/L90`, and optional `D_ucb` into pipeline context.
Updates must happen after the engine observes the completed request.

#### Three-Column Reconciliation

| Aspect | Paper / experiment intent | Current code | Canonical decision | Reason |
| --- | --- | --- | --- | --- |
| Causality | Prediction uses only arrival-time information | `PredictionContext.from_request()` uses request tokens, model, timestamp, input bin | Same | Keeps estimator ablations honest |
| Oracle | Analytical upper bound | `OracleOutputPredictor.predict()` reads `response_tokens` | Keep only for oracle experiments | It intentionally violates causality to quantify estimation error |
| EMA | Simple online baseline | `EMAOutputPredictor` maintains per-model and global EMA with normal-approx quantiles | Same | Cheap, deterministic, useful warmup baseline |
| Histogram | Non-parametric online predictor | `HistogramOutputPredictor` uses hierarchical log histograms | Same | Handles skewed token lengths without external models |
| Duration UCB | Conservative service-time bound | `HistogramDurationPredictor` uses TTFT P90 and TPS P10 | Same | Matches Stage 2 learning-augmented intent |
| Missing true output | Not all requests have labels before completion | EMA/histogram/oracle still assume non-null `response_tokens` in some paths | Skip update on unlabeled requests; oracle must raise a clear error | Optional schema fields must not crash normal online routing |

#### Stage Interface

```python
class ValueEstimator(Protocol):
    def estimate(
        self,
        request: Request,
        state: SimulationState,
    ) -> ValueEstimate:
        """Return predicted output length and optional uncertainty bounds."""

    def update(
        self,
        request: Request,
        outcome: RoutingOutcome,
    ) -> None:
        """Update only after ground truth is observed."""
```

Pure `estimate()` is deterministic given `(request, estimator_state)`. `update()`
is the only state-mutating method.

#### Reference Implementations

| Module | Role | Notes |
| --- | --- | --- |
| `rwsim/policies/value_estimators/oracle.py` | Oracle length predictor | Analytical upper bound only |
| `rwsim/policies/value_estimators/ema.py` | EMA output-length predictor | Per-model plus global fallback |
| `rwsim/policies/value_estimators/histogram.py` | Histogram output and duration predictors | Hierarchical backoff and conservative quantiles |
| `experiments/estimator_ablation/` | Estimator sensitivity experiment | Should consume this interface directly |

#### Unit-Test Contract

Each estimator must pass the following in `tests/unit/policies/value_estimators/`:

1. **Causality**: `estimate()` does not require `response_tokens`.
2. **Oracle guardrail**: oracle either returns exact truth when present or raises
   a clear error when truth is missing.
3. **EMA warmup**: cold predictions use configured defaults; after warmup, EMA
   tracks a constant stream within tolerance.
4. **Histogram monotonicity**: `q10 <= q50 <= q90` for every backoff level.
5. **Update discipline**: unlabeled requests do not mutate token histograms or
   EMA state.

#### Known Divergence to Resolve

`rwsim/schemas.Request.response_tokens` is optional, but
`EMAOutputPredictor.update()`, `HistogramOutputPredictor.update()`, and
`OracleOutputPredictor.predict()` still assume it is present in some paths.
Fix this as a behavior-change commit with unit tests; do not hide it in a
structural refactor.

### Cost Router

#### Canonical Definition (authoritative)

For each provider `j` and request with predicted output length `L_hat`:

    c_eff(j) = { c_A(j)        if j in S_A                                [eq. 1]
               { psi(z_j)      if j in S_Q                                [eq. 2]
               { lambda(u_j)   if j in S_C                                [eq. 3]

where

    c_A(j)    = c_in(j) * n_in + c_out(j) * L_hat                          [eq. 4]
    psi(z)    = L * (U/L)^z,         z = k_j / Q_j                         [eq. 5]
    lambda(u) = U * u,               u = a_j / C_j                         [eq. 6]

and the per-scenario envelope is calibrated from the `S_A` price range:

    U = max_{j in S_A} c_A(j) at a typical request size                    [eq. 7]
    L = U * floor_ratio,           floor_ratio = 1e-3                      [eq. 8]

Infeasible providers get `c_eff(j) = +inf`:

    j infeasible if j in S_Q and k_j >= Q_j                                [eq. 9]
    j infeasible if j in S_C and a_j >= C_j                                [eq. 10]

A per-request cost anchor for downstream stages:

    v_hat = min_{j in S_A and SLO-safe} c_A(j)                             [eq. 11]

Fallback when no SLO-safe `S_A` exists: cheapest `S_A`, then cheapest
positive-cost provider, then `0`.

References: Buchbinder-Naor 2009 Thm 4.2, competitive ratio `ln(U/L) + 1` for
the online knapsack interpretation of `psi`; paper §3.2.3.

#### Pipeline Role

Cost router = per-provider effective-cost computation. It does not own latency
optimization, hedging, request execution, or the simulation loop. In the
canonical pipeline it emits `{c_eff(j)}` and `v_hat`; provider selection is
performed by the latency router or by a trivial selector such as
`argmin c_eff`.

#### Three-Column Reconciliation

| Aspect | Paper §3.2.3 | Current code | Canonical decision | Reason |
| --- | --- | --- | --- | --- |
| `c_eff` form | Category switch, eq. 1-3 above | `marginal + q_sp + c_sp + lat_term` in `rwsim/world/shadow_price.py` | Category switch, as above | Mathematically equivalent when inapplicable terms are zero; switch form is clearer and easier to test |
| `psi(z)` | `L (U/L)^z` | Same in `quota_shadow_price()` | Same | Already aligned |
| `lambda(u)` | `U * u` linear | `U * u^alpha`, default `alpha=1.0`, in `concurrency_shadow_price()` | `U * u` linear by default | Paper specifies linear; keep `alpha` only as an ablation knob |
| `lat_term` in `c_eff` | Not present | Optional `latency_alpha * true_p50_ms(now)` | Remove from cost router | Latency belongs in the latency-router objective |
| Envelope `(L, U)` | Assumed given | `U = max api_cost_at(typical_tokens=200)`, `L = U * 1e-3` | Code calibration | Paper does not specify calibration; current rule is reproducible |
| `v_hat` anchor | Minimum API request cost | Sidecar LP-budget uses cheapest SLO-safe `S_A` anchor | SLO-safe `S_A` anchor | Prevents a pathologically slow cheap API from starving latency budget |
| Infeasibility | Exhausted capacity excluded | Selectors filter `is_available`; `two_layer` filters within the selected tier before P50 ranking | Infeasible means `+inf` and cannot be selected | Selection safety should be uniform across routers |

#### Stage Interface

```python
class CostRouter(Protocol):
    def effective_costs(
        self,
        request: Request,
        providers: Sequence[Provider],
        state: SimulationState,
    ) -> CostEstimate:
        """Return c_eff, v_hat, and infeasibility reasons."""
```

`effective_costs()` is a pure function of `(request, providers,
state.{quota_used, concurrency_active})`. It does not hedge, iterate the
workload, sample latency, or mutate provider state.

#### Reference Implementations

| Module | Role | Notes |
| --- | --- | --- |
| `rwsim/policies/cost_routers/fixed.py` | Constant-provider selectors | Cheapest / fastest baselines |
| `rwsim/policies/cost_routers/round_robin.py` | Deterministic round-robin selector | Baseline only |
| `rwsim/policies/cost_routers/tiered.py` | Tier priority and cheapest-effective helpers | Current home for two-layer and joint cost helpers |
| `rwsim/world/shadow_price.py` | Shadow-price primitives | Should become an implementation detail of cost routers |
| `experiments/tiered_capacity/lp_budget_eval.py` | Sidecar budget variants | Source for `v_hat` budget semantics |

#### Unit-Test Contract

Each cost router must pass the following in `tests/unit/policies/cost_routers/`:

1. **Shadow-price endpoints**: `psi(z=0) == L`, `psi(z -> 1) -> U`,
   `lambda(u=0) == 0`, `lambda(u=1) == U`.
2. **Monotonicity**: `psi` is non-decreasing in `z`; `lambda` is
   non-decreasing in `u`.
3. **Scale invariance**: Scaling all `S_A` costs by constant `k` scales
   `(L, U, c_A, v_hat)` by `k` and leaves the argmin unchanged when all
   providers are scaled together.
4. **Infeasibility**: Exhausted quota or saturated concurrency yields
   `c_eff = +inf`, and downstream selectors never choose that provider.
5. **Competitive ratio on quota-only**: On a random quota-only workload with
   values in `[L, U]`, online PD cost satisfies
   `online_cost <= (ln(U/L) + 1) * offline_optimal_cost` across at least 10
   seeds. This belongs in `tests/integration/`.

#### Known Divergence to Resolve

The current `lat_term` parameter in `rwsim/world/shadow_price.py` is not
canonical. The latency router should own that trade-off. Track its removal as a
behavior-change commit with golden updates or explicit golden-preservation notes.

### Latency Router

#### Canonical Definition (authoritative)

For each provider `j`, keep a causal rolling latency profile:

    W_j(t) = {ttft_r : t - window <= timestamp_r < t, provider_r = j}       [eq. 1]
    e_j(t) = failures_j(t) / attempts_j(t)                                 [eq. 2]

Successful-sample CDF:

    F_success_j(l, t) = (1 / |W_j(t)|) * sum_{x in W_j(t)} 1[x <= l]        [eq. 3]

Failure-aware CDF:

    F_j(l, t) = F_success_j(l, t)                         SEPARATE mode    [eq. 4]
    F_j(l, t) = (1 - e_j(t)) * F_success_j(l, t)           INFINITY mode    [eq. 5]

Hard pre-filter:

    eligible_j(t) = (e_j(t) <= e_max) and (F_j(L_min, t) >= beta)           [eq. 6]

LP latency router:

    minimize    sum_j pi_j * (c_eff(j) * (1 + kappa e_j(t)) + alpha P50_j) [eq. 7]
    subject to  sum_j pi_j * F_j(SLO, t) >= rho                            [eq. 8]
                sum_j pi_j = 1,  pi_j >= 0                                 [eq. 9]

Budget-body latency router:

    minimize    sum_j pi_j * Tbar_j(t)                                     [eq. 10]
    subject to  sum_j pi_j * c_eff(j) <= tau * v_hat                       [eq. 11]
                sum_j pi_j = 1,  pi_j >= 0                                 [eq. 12]

P50-band latency router:

    P = Pareto({(P50_j, c_eff(j))})                                        [eq. 13]
    B = {j in P : P50_j <= min_k P50_k * (1 + band)}                       [eq. 14]
    primary = argmin_{j in B} c_eff(j)                                     [eq. 15]

P95 SLO filter:

    S = {j : P95_j(t) <= safety_margin * SLO}                              [eq. 16]
    primary = argmin_{j in S} c_eff(j)                                     [eq. 17]

#### Pipeline Role

Latency router = provider selection under SLO and cost-budget constraints. It
consumes `{c_eff(j)}` and `v_hat` from the cost router, plus latency profiles.
It may output a single primary provider or a mixing distribution. It does not
compute shadow prices, launch hedges, or update provider capacity.

#### Three-Column Reconciliation

| Aspect | Paper / experiment intent | Current code | Canonical decision | Reason |
| --- | --- | --- | --- | --- |
| Rolling profile | Causal online profiling | `ProviderProfile.get_samples_before()` filters by cutoff and window | Same | Prevents lookahead |
| Failure handling | Treat failure as missed deadline by default | `FailureMode.INFINITY` and `SEPARATE` | INFINITY default, SEPARATE ablation | Reliability should affect SLO feasibility |
| LP objective | Minimize cost under tail target | `solve_lp()` uses cost, optional `kappa`, optional `alpha * P50` | Cost objective with optional latency-alpha ablation | Keeps paper LP intact while allowing documented ablation |
| LP fallback | Avoid empty routing on infeasible LP | `solve_lp_with_fallback()` relaxes SLO then best-effort | Same | Operationally necessary and deterministic |
| SWRR | Implement LP weights as routing decisions | `SWRRSampler` smooths weights | Same | Reduces short-run variance without changing target weights |
| V2 / P50-band | P50 rank, Pareto prune, near-best band | `V2Router` and `tiered_filters.select_p50_band()` | Keep as latency-router variant | It is provider selection, not cost routing |
| P95 filter drift | Analytical P95 must follow active distribution | `provider_p95_at()` misses `TieredProvider._active_ttft_dist()` | Use active TTFT distribution for all providers | Drift must affect both sampling and filtering |
| Budget body | Minimize latency inside cost envelope | Sidecar `_select_budget_body()` uses `Tbar` under `budget_tau` | Promote as `lp_cost_budget` latency router | This is a latency selection rule consuming cost-router output |

#### Stage Interface

```python
class LatencyRouter(Protocol):
    def select(
        self,
        request: Request,
        providers: Sequence[Provider],
        cost: CostEstimate,
        profiles: Mapping[str, LatencyProfile],
        state: SimulationState,
    ) -> LatencyDecision:
        """Return primary provider or provider weights plus debug metadata."""
```

The selected provider must be feasible under the cost-router infeasibility map.
If no provider is feasible, the fallback rule must be explicit in metadata.

#### Reference Implementations

| Module | Role | Notes |
| --- | --- | --- |
| `rwsim/policies/latency_routers/online_lp.py` | LP tail-CDF router and SWRR | Current `lp_mix` / `lp_hedge` base |
| `rwsim/policies/latency_routers/v2.py` | P50 Pareto-band router | Current `v2_*` base |
| `rwsim/policies/latency_routers/tiered_filters.py` | Tiered P95 and P50-band filters | Current `joint_*` selectors |
| `experiments/tiered_capacity/lp_budget_eval.py` | LP cost-budget sidecar | Migration source for `lp_cost_budget` |

#### Unit-Test Contract

Each latency router must pass the following in `tests/unit/policies/latency_routers/`:

1. **Causal profile**: samples at or after decision time are ignored.
2. **CDF bounds**: every CDF is in `[0, 1]`; INFINITY mode is never greater
   than SEPARATE mode for the same samples.
3. **LP feasibility**: if any provider satisfies the target CDF alone, LP
   returns weights summing to one and satisfying eq. 8 within tolerance.
4. **Budget feasibility**: budget-body expected cost is `<= tau * v_hat` when
   the LP status is optimal.
5. **P50-band determinism**: ties are broken by effective cost and then stable
   provider ordering.
6. **Drift awareness**: P95/P50 filters use the provider distribution active at
   the decision time.

#### Known Divergence to Resolve

`provider_p95_at()` currently only recognizes `_active_dist`; it must use
`_active_ttft_dist()` for `TieredProvider` drift. Also, `alpha * P50` inside
`solve_lp()` is an ablation knob, not the default canonical LP objective.

### Hedger

#### Canonical Definition (authoritative)

Hedging is evaluated after a primary provider has been selected. At elapsed
time `t`, for primary `p`, backup `b`, SLO `L`, and dispatch overhead `delta`:

    S_p(x) = P(T_p > x)                                                    [eq. 1]
    F_b(x) = P(T_b <= x)                                                   [eq. 2]
    P_viol(t) = P(T_p > L | T_p > t) = S_p(L) / S_p(t)                    [eq. 3]
    R_b(t) = L - t - delta                                                 [eq. 4]

Economic hedge rule:

    hedge at t iff P_viol(t) * F_b(R_b(t)) > C_b / V                      [eq. 5]

where `C_b` is incremental backup cost and `V` is the penalty value of an SLO
violation. The canonical hedge time is the earliest grid time satisfying eq. 5:

    h* = min {t in grid : P_viol(t) * F_b(R_b(t)) > C_b / V}               [eq. 6]

Probability-target hedge rule:

    P_success_if_hedge(t) =
        P(T_p <= L | T_p > t) +
        P(T_p > L | T_p > t) * F_b(R_b(t))                                [eq. 7]

    h_latest = max {t in grid : P_success_if_hedge(t) >= rho}              [eq. 8]

The backup dispatches only if the primary has not completed by `h*` or
`h_latest`.

#### Pipeline Role

Hedger = optional backup decision after primary selection. It does not choose
the primary provider, does not compute `c_eff`, and does not solve the
latency-router LP. It may select a backup provider and attach hedge trigger
metadata. The engine owns actual dispatch, billing, first-completion logic, and
capacity accounting.

#### Three-Column Reconciliation

| Aspect | Paper / experiment intent | Current code | Canonical decision | Reason |
| --- | --- | --- | --- | --- |
| Economic rule | Hedge when expected violation reduction exceeds backup cost | `smart_hedge_economic()` implements `P_viol * F_backup > cost_ratio` | Keep as canonical `smart_economic` | It has a clear cost-benefit interpretation |
| Hedge time | Earliest time when economic condition becomes true | `find_optimal_hedge_time_economic()` grid-searches earliest trigger | Same | Earliest trigger protects tail without unconditional replication |
| Backup choice | Backup should be non-primary and viable | `select_backup()` supports fastest, LP-other, cheapest-viable; sidecar probability-target hedge supports `backup_scope` | Canonical probability-target hedge considers any available non-primary provider; cross-tier is an ablation/legacy constraint | Backup selection is separate from trigger math, and tier restrictions are not part of the hedge formula |
| Old tiered hedge threshold | Heuristic trigger before economic check | `max(1.5 * P50, 0.5 * SLO)` with fixed `P_viol = 0.5` in `tiered_impl.py` and old sidecar hedge | Mark as non-canonical historical behavior | It mixes a heuristic delay with a hard-coded violation probability |
| Probability target | Maintain target success probability | Sidecar `_apply_probability_target_hedge()` uses latest safe time with target `rho=0.99` | Keep as separate `probability_targeted` hedger | It answers a different objective than economic hedging |
| `max(1.5*P50, 0.5*SLO)` vs eq. 5 | Not in paper formula | Used by current `joint_hedge` and `*_oldhedge` variants | Eq. 5 is canonical; threshold heuristic remains migration alias only | The formula exposes the cost/SLO trade-off; the heuristic hides it |
| Billing | Both requests billed if hedged | `HedgingResult.total_cost` bills primary plus backup | Same | Matches provider billing semantics |

#### Stage Interface

```python
class Hedger(Protocol):
    def plan(
        self,
        request: Request,
        primary: Provider,
        candidates: Sequence[Provider],
        cost: CostEstimate,
        profiles: Mapping[str, LatencyProfile],
        state: SimulationState,
    ) -> HedgePlan:
        """Return backup provider and trigger time, or no hedge."""
```

`plan()` must be deterministic given profiles and state. It must not sample
primary or backup latency; the engine samples and executes.

#### Reference Implementations

| Module | Role | Notes |
| --- | --- | --- |
| `rwsim/policies/hedgers/smart_economic.py` | Economic, survival, residual, percentile hedge helpers | `SMART_ECONOMIC` is canonical for cost-benefit hedging |
| `experiments/tiered_capacity/lp_budget_eval.py` | Probability-target and old tiered hedge variants | Migration source for `probability_targeted` and old alias behavior |
| `rwsim/strategies/tiered_impl.py` | Current `joint_hedge` behavior | Preserved by golden until behavior-change migration |

#### Unit-Test Contract

Each hedger must pass the following in `tests/unit/policies/hedgers/`:

1. **Economic monotonicity**: increasing `C_b / V` never increases hedge rate.
2. **Backup usefulness**: if `F_b(R_b(t)) = 0`, economic hedger does not hedge.
3. **No-time-left**: if `R_b(t) <= 0`, no backup is dispatched.
4. **Earliest trigger**: `smart_economic` returns the first grid time
   satisfying eq. 5 or `inf`.
5. **Probability target**: probability-target hedger returns the latest safe
   trigger satisfying eq. 8 or no hedge.
6. **Execution separation**: hedger planning never samples request latency or
   mutates quota/concurrency state.

#### Known Divergence to Resolve

There are two smart-hedging families today. Canonical `smart_economic` is eq.
5, implemented in `rwsim/policies/hedgers/smart_economic.py`. The
`max(1.5 * P50, 0.5 * SLO)` trigger with fixed `P_viol = 0.5` is not
canonical; keep it only as a migration alias for existing `joint_hedge` golden
reproduction until the behavior-change commit replaces it or removes the alias.

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
switch. When a current strategy has an existing implementation detail, the
alias should preserve that behavior until a separate research change
intentionally changes the algorithm.

The code-level alias table lives in `rwsim/policies/composer.py`.

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
silently into existing `lp_hedge` or `v2_p50_hedge` aliases.

The paper offline/stage strategy implementations now live in
`experiments/offline_stage/strategies/`. They are canonical for reproducing
the existing paper artifacts, but they are still migration targets: when each
strategy is decomposed into reusable stages, keep the old strategy name as a
pipeline alias and remove the monolithic implementation.

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

The strategy is not considered migrated until its monolithic runner can be
removed or converted into a thin wrapper around the pipeline composer.
