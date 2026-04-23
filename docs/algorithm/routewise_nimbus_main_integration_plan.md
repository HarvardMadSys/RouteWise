# RouteWise + Nimbus Unified Integration Plan (v4)

## 1. Executive Summary

This plan is updated to reflect the branch workflow that has already happened:

- Keep **main as the protected production branch**.
- Use **dev / integration branches** for heavy merge and conflict resolution work.
- Reuse Nimbus's existing **`BaseRouter` architecture** as the unified execution facade.
- Integrate simulator assets into a dedicated RouteWise development branch.
- Implement RouteWise incrementally (`Stage 1` first, then `Stage 2`).

Expected outcome:

- A single production codebase that can be promoted to `main` via small reviewed PRs.
- Runtime selection of routing behavior via `routing_strategy` parameter.
- Independent failure domains: Nimbus bugs do not block RouteWise progress.


## 2. Branch Reality (Validated, Updated)

Branches involved now:

- `origin/main`: newest product features and infra changes.
- `origin/dev`: integration baseline that now contains Nimbus integration work.
- `origin/murphy/dev/req_queue` (Nimbus source branch): original Nimbus router architecture + outsourcing stack.
- `murphy/dev/simulator`: experiment framework, paper, slides, algorithm docs (source branch).
- `origin/murphy/dev/routewise-online`: current RouteWise integration branch.

Execution status (already done):

1. `main` was merged into `dev`.
2. Nimbus (`req_queue`) was merged into `dev` with conflict resolution.
3. A dedicated branch `murphy/dev/routewise-online` was created from `origin/dev`.
4. `murphy/dev/simulator` was merged into `murphy/dev/routewise-online` (conflict resolved in `pyproject.toml`).

Implication:

- The old "how to merge Nimbus first" decision is no longer pending; it is completed on integration branches.
- The remaining work is RouteWise online serving implementation and staged promotion back to `main`.


## 3. Design Decision: Reuse Nimbus BaseRouter

### 3.1 What we keep

Use Nimbus architecture as the unified routing execution core:

- `BaseRouter`: shared execution, fallback, circuit breaker, stream hooks.
- `FixedRouter`: baseline weighted routing.
- `NimbusRouter`: SLO-aware outsourcing decision path.

RouteWise is added as a new peer router:

- `RouteWiseRouter` (new): cost-aware online decision path.

Conceptual structure:

```text
BaseRouter
├── FixedRouter
├── NimbusRouter
└── RouteWiseRouter   (new)
```

### 3.2 What we avoid

- Do not introduce an extra new facade that duplicates `BaseRouter`.
- Do not rewrite Nimbus router internals from scratch.


## 4. Branch Integration Strategy (Merge Which to Which)

## Step A (Completed): Main baseline into dev

Flow:

1. Merge latest `origin/main` into `dev`.
2. Use `dev` as integration buffer so `main` remains untouched.

## Step B (Completed): Nimbus into dev

Flow:

1. Merge `origin/murphy/dev/req_queue` into `dev`.
2. Resolve routing/serving conflicts in integration branch.
3. Validate that fixed and nimbus paths remain healthy.

## Step C (Completed): Simulator assets into routewise-online

Flow:

1. Create `murphy/dev/routewise-online` from `origin/dev`.
2. Merge `murphy/dev/simulator` into `murphy/dev/routewise-online`.
3. Resolve remaining conflict(s) and publish branch.

Scope note:

- This branch intentionally contains broad research assets (experiment/paper/slide/docs) for fast RouteWise iteration.
- Runtime promotion to `main` will still be sliced into focused PRs.

## Step D (In Progress): RouteWise implementation on routewise-online

Implement online RouteWise serving on top of existing `BaseRouter` architecture in `murphy/dev/routewise-online`.

## Step E (Planned): Promote stable increments to main

Promotion policy:

1. `routewise-online` -> `dev` via small validated PRs.
2. `dev` -> `main` only for reviewed, low-blast-radius slices.
3. Keep feature flags and rollback-by-config for production safety.


## 5. Runtime Configuration Model

Top-level routing selection stays coarse-grained:

```yaml
routing_strategy: fixed | nimbus | routewise
```

RouteWise variants are internal sub-configuration:

```yaml
routewise:
  decision_rule: pd | lapd
  predictor: ema | histogram
  risk_quantile: 0.10
  quota:
    daily_quota: 5000
    monthly_fee: 20.0
  concurrency:
    enabled: true
    limit: 8
    monthly_fee: 25.0
```

Why:

- Prevents enum explosion at top-level.
- Keeps future algorithm variants extensible.

### 5.1 Subscription type mapping

RouteWiseRouter must know which adapter corresponds to which subscription type
(`quota` / `concurrency` / `api`). This mapping is defined in `models.yaml`
alongside existing adapter declarations, keeping a single source of truth:

```yaml
# In models.yaml, each adapter gains an optional subscription_type field
adapters:
  chutes-llama-70b:
    provider: openai_compat
    base_url: https://chutes.ai/v1
    subscription_type: quota        # S_Q

  featherless-llama-70b:
    provider: openai_compat
    base_url: https://api.featherless.ai/v1
    subscription_type: concurrency  # S_C

  together-llama-70b:
    provider: openai_compat
    base_url: https://api.together.xyz/v1
    subscription_type: api          # S_A (default if omitted)
```

At init time, `RouteWiseRouter` reads the `subscription_type` tag from each
adapter in the model's route config. Adapters without a tag default to `api`.
No duplication between `models.yaml` and `routewise` config: `models.yaml`
owns the provider/model/subscription mapping; `routewise` config owns only
policy parameters (decision rule, predictor, quota limits, pricing).


## 6. RouteWise Decision Flow

RouteWise and Nimbus solve fundamentally different problems:
- **Nimbus**: "Should this request stay on our GPU or go to a remote API?" (latency-driven, binary)
- **RouteWise**: "Across multiple subscription providers, which one minimizes total cost?" (cost-driven, multi-choice)

### 6.1 RouteWiseRouter._select_adapter() decision flow

```text
Request(model_id, messages, params) arrives
|
|-- 1. Predict output length
|     pred = predictor.predict(model_id, input_tokens)
|     -> QuantilePrediction(q10, q50, q90)
|
|-- 2. Estimate API value (cost saved by using a subscription)
|     if decision_rule == pd:   v_t = api_price_in * n_in + api_price_out * pred.q50
|     if decision_rule == lapd: v_t = api_price_in * n_in + api_price_out * pred.q10  (LCB)
|
|-- 3. Compute shadow prices for available subscription adapters
|     theta_Q = L * (U/L)^(used_today / daily_quota)    [S_Q, exponential in usage ratio]
|     theta_C = lambda * weight * est_duration           [S_C, congestion price]
|
|-- 4. Compute net gain for each candidate adapter
|     For each adapter in model's route config:
|       QUOTA adapter:       G = v_t - theta_Q  (if quota remaining)
|       CONCURRENCY adapter: G = v_t - theta_C  (if slots available)
|       API adapter:         G = 0              (baseline, always available)
|
|-- 5. Select adapter = argmax(G)
|     Ties broken in favor of subscription (cost savings > 0 means subscription is worthwhile)
|
|-- 6. Return selected adapter to BaseRouter for execution
```

After request completes, predictor receives feedback (see Section 7.1).

### 6.2 Key differences from simulation code

The algorithm logic (threshold function, shadow price, value estimation) is identical
to `experiment/strategies/online/primal_dual.py`. The production differences are:

| Aspect | Simulation (`experiment/`) | Production (`routing/routewise/`) |
|--------|---------------------------|-----------------------------------|
| Time model | Simulated day counter | Wall-clock with timezone-aware daily reset |
| Batch vs per-request | Processes full trace in a loop | One request at a time, async |
| Output tokens | Known after simulation step | Unknown; predicted by EMA, revealed after response |
| Concurrency (S_C) | Simulated finish_time | Real in-flight tracking via async lifecycle |
| Thread safety | Single-threaded | Must be thread/async-safe |

The two codebases are **intentionally independent**. Simulation code stays in `experiment/`
for offline evaluation and paper experiments. Production code in `routing/routewise/` is
purpose-built for real-time serving. Correctness is validated by replay tests: feeding the
same request trace to both implementations and asserting identical decision sequences.


## 7. Implementation Phases

### Phase 0: Interface freeze and ADR (0.5 day)

Deliverables:

- Confirm `BaseRouter` extension strategy for RouteWise.
- Confirm config schema (`routing_strategy=routewise` + sub-config).
- Define routing metadata contract (`_routing` fields).
- Resolve open questions (see Section 11).

Exit criteria:

- ADR merged, config schema approved.

### Phase 1: Nimbus integration hardening on dev (Completed)

Deliverables:

- Nimbus merge into integration branch complete.
- Regression tests pass for fixed + nimbus behavior.

Exit criteria:

- Nimbus path healthy on integration branch.
- Fixed path unchanged from user perspective.

### Phase 1.5: Research asset sync into routewise-online (Completed)

Deliverables:

- Research directories and experiment framework merged into `murphy/dev/routewise-online`.

Exit criteria:

- Experiment scripts runnable from integration branch.

### Phase 2: RouteWise Stage 1 (`S_Q + S_A`) (4-6 days)

Deliverables:

- `RouteWiseRouter` with PD and LA-PD decision rules (single PR, switchable via config).
- Predictor module (EMA first, optional histogram).
- Quota state management (wall-clock reset semantics).
- Replay tests: feed same trace to simulation and production code, assert identical decisions.

Exit criteria:

- Deterministic decision trace matching simulation output on replay.
- Stable API behavior under load and failures.

### Phase 3: RouteWise Stage 2 (`S_Q + S_C + S_A`) (5-8 days)

Deliverables:

- Concurrency-aware admission and shadow-price logic.
- Streaming-aware lifecycle handling for in-flight state.
- Extended metrics (decision reason, quota usage, estimated gain).

Exit criteria:

- No queue/accounting leaks under streaming and cancellation.
- Clear metrics for online evaluation.

### Phase 4: Rollout and guardrails (2-3 days + canary window)

Deliverables:

- Feature flags and model-scoped enablement.
- Canary rollout plan and rollback runbook.
- A/B evaluation hooks (Nimbus vs RouteWise for selected models).

Exit criteria:

- Rollback by config only.
- Canary success criteria defined and measurable.


## 8. Critical Engineering Notes

### 8.1 Predictor feedback mechanism

Do not rely on a router-level `_on_completion` hook for predictor updates.

**Problem**: Streaming providers often return incomplete or missing `usage` data.
Long responses (the most valuable for prediction) are also the most likely to
timeout or drop the final usage chunk. A naive router-level hook creates
systematic bias toward underestimating output length.

**Solution**: The `completions` endpoint already normalizes usage across all
adapters (streaming and non-streaming). The predictor update is triggered there,
after normalization:

```text
completions.py (after response completes)
  |-- extract normalized completion_tokens from usage
  |-- if missing: estimate from chunk count * avg_tokens_per_chunk, flag as estimated
  |-- call routewise_router.predictor.update(model_id, tokens, is_estimated)
```

The `routewise_router` is accessible via FastAPI dependency injection (same pattern
as `get_nimbus_router()`). The predictor exposes an `update()` method. No event bus
or callback registration needed; it is a direct method call at a well-defined point
in the request lifecycle.

### 8.2 Source-of-truth for provider capabilities

Avoid dual source drift between `models.yaml` and routewise config:

- `models.yaml` owns: provider endpoints, model support, `subscription_type` tag.
- `routewise` config owns: decision rule, predictor settings, quota limits, pricing.

### 8.3 Blast radius isolation

- RouteWise should not depend on Nimbus outsourcing internals.
- Nimbus-only failures must not affect fixed/routewise paths.
- Each router class is instantiated independently in `bootstrap.py`.


## 9. PR Slicing (Updated for Current Branch Route)

1. **PR-1**: RouteWise config surface (`routing_strategy=routewise` + sub-config validation).
2. **PR-2**: `RouteWiseRouter` scaffold integrated with `BaseRouter` execution lifecycle.
3. **PR-3**: RouteWise Stage 1 (PD + LA-PD, switchable by `decision_rule` config).
4. **PR-4**: RouteWise Stage 2 concurrency support.
5. **PR-5**: Observability, canary controls, replay tests, and rollout docs.
6. **PR-6**: Promotion PR from integration branch to `main` (only after gates pass).

PD and LA-PD share 90%+ code (LA-PD is PD with a conservative quantile). They
belong in one PR, switchable by config. Splitting them creates unnecessary churn.

Each PR must include:

- test updates,
- clear rollback mechanism,
- explicit non-goals.


## 10. Risks and Mitigations

Risk 1: Large integration branch drifts too far from main  
Mitigation: frequent small promotion PRs and periodic main sync into integration branch.

Risk 2: Simulation-to-production mismatch in time/accounting semantics  
Mitigation: Stage 1 first; add replay harness before Stage 2.

Risk 3: Streaming lifecycle causes stale concurrency state  
Mitigation: explicit streaming tests for cancel/error/no-usage paths.

Risk 4: Config complexity leads to operator mistakes  
Mitigation: top-level strategy remains simple; validate sub-config on startup.

Risk 5: Timeline optimism  
Mitigation: ship by milestone gates, not calendar-only promises.


## 11. Practical Timeline (Realistic Range)

Assuming one primary owner plus code review support:

| Phase | Work | Estimate |
|:-----:|------|:--------:|
| 0 | ADR + interface freeze | 0.5 day |
| 1 | Nimbus integration hardening on dev | Completed |
| 1.5 | Research asset sync into routewise-online | Completed |
| 2 | RouteWise Stage 1 (PD + LA-PD) | 4-6 days |
| 3 | RouteWise Stage 2 (concurrency) | 5-8 days |
| 4 | Rollout hardening + canary | 2-3 days + observation |
| **Remaining** | | **~2 to 3 weeks** |


## 12. Open Questions (Resolve in Phase 0)

1. **Provider accounts**: Are Chutes (S_Q) and Featherless (S_C) accounts set up
   for the prototype, or do we start with simulated subscription constraints?

2. **Quota tracking**: Does Chutes expose a "remaining quota" API, or do we track
   quota purely client-side (as in the simulation)?

3. **Initial L/U bounds**: How do we bootstrap the primal-dual value bounds
   (`L`, `U`) at startup? Options: hardcode from simulation results, estimate
   from first N requests, or load from a config seed.

4. **Timezone for daily reset**: Which timezone defines the "day" boundary for
   quota reset? UTC, or provider-specific?

5. **A/B scope**: Do we want to run Nimbus and RouteWise simultaneously on
   different model groups, or strictly one strategy at a time system-wide?


## 13. Final Recommendation

Adopt this strategy:

1. **Main remains protected production branch**.
2. **Nimbus architecture is reused** (no duplicate facade design).
3. **Integration work happens on `dev` / `routewise-online` first**, then promoted in slices.
4. **Simulator assets are already integrated on `routewise-online`**.
5. **RouteWise should now be implemented incrementally on `routewise-online`**, Stage 1 first.
6. **Simulation and production code are intentionally independent**, validated by replay tests.

