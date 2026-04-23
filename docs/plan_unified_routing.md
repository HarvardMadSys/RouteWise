# Plan: Unified Routing Architecture for RouteWise + Nimbus

## 1. Current Branch Situation

### Three branches

| Branch | HEAD | Key Content |
|--------|------|-------------|
| **origin/main** (a4b5838) | `fix: remove streaming bottlenecks` | Latest infra: embeddings, Qdrant, auth, streaming fixes, GLM-5 |
| **origin/murphy/dev/req_queue** (nimbus) | `feat: dry_run_outsource` | New routing arch: `BaseRouter`/`NimbusRouter`, outsourcing engine, tree cache |
| **murphy/dev/simulator** (current) | `feat: oracle sensitivity` | Research: `experiment/`, ICML paper, slides, docs |

### Divergence analysis (relative to origin/main)

```
origin/main
├── +58 commits ahead of nimbus,  nimbus has 29 unique commits
└── +94 commits ahead of simulator, simulator has 52 unique commits
```

### What each branch uniquely contributes

**Nimbus branch** (29 unique commits):
- `routing/routers.py` — BaseRouter + FixedRouter + NimbusRouter (1264 lines)
- `routing/outsourcing/` — decision engine, knapsack, violation detection, adapters
- `routing/outsourcing_integration.py` — OutsourcingRouter with TreeCache
- `routing/tree_cache.py` — prefix cache approximation
- `serving/config/settings.py` — typed settings with RoutingStrategy enum
- Tests for all of the above
- **DELETES**: `routing/executor.py`, `routing/manager.py`, `routing/strategies.py`, `routing/waiting_queue_api.py`

**Simulator branch** (52 unique commits):
- `experiment/` — full RouteWise simulation framework (strategies, predictors, datasets, results)
- `ICML2026_HybridInference/` — paper LaTeX
- `slide/` — presentation
- `docs/algorithm/` — design docs
- `config/experiment.yaml` — experiment config
- Minor modifications to `routing/executor.py`, `routing/manager.py` (files that nimbus DELETES)

### Conflict zones

| File | Nimbus | Simulator | Conflict? |
|------|--------|-----------|-----------|
| `routing/executor.py` | DELETED | Modified | Yes — simulator changes lost, but they're minor |
| `routing/manager.py` | DELETED | Modified | Yes — same situation |
| `routing/waiting_queue_api.py` | DELETED | Exists (old) | Yes — nimbus replaces entirely |
| `config/models.yaml` | Modified | Modified | Possible — need manual resolve |
| `experiment/**` | Untouched | All new | No conflict |
| `ICML2026_HybridInference/**` | Untouched | All new | No conflict |
| `slide/**` | Untouched | All new | No conflict |
| `docs/algorithm/**` | Untouched | All new | No conflict |

**Key insight**: The simulator branch's core routing changes (`routing/executor.py`, `routing/manager.py`)
are moot because nimbus replaces the entire routing architecture. The only truly valuable and
conflict-free content from the simulator branch is the research artifacts.

## 2. Recommended Merge Strategy

### Step 1: Merge nimbus → main (PR)

Nimbus rewrites the routing layer. This is the structural foundation that both Nimbus and RouteWise
will build on. It needs to go in first.

```
origin/main  ←──  nimbus (rebase onto origin/main first, resolve conflicts)
```

- Rebase nimbus onto latest origin/main to pick up new features (embeddings, qdrant, etc.)
- Resolve conflicts (nimbus's serving/config/settings.py vs origin/main's existing patterns)
- PR review, CI pass, merge

**Risk**: Medium. Nimbus deletes and rewrites routing/. Need careful conflict resolution.
But nimbus was designed to replace the old routing layer, so the deletions are intentional.

### Step 2: Merge research artifacts from simulator → main (PR)

Cherry-pick or merge ONLY the research content. The core routing changes in simulator are
irrelevant post-nimbus.

Content to merge:
- `experiment/` (entire directory)
- `ICML2026_HybridInference/` (entire directory)
- `slide/` (entire directory)
- `docs/algorithm/` (design docs)
- `config/experiment.yaml`

```
main (post-nimbus)  ←──  simulator research artifacts only
```

**Risk**: Low. These are purely additive directories with no overlap.

### Step 3: Implement RouteWiseRouter on main

With both nimbus infrastructure and research code in main, implement RouteWiseRouter
as a new BaseRouter subclass.

```
main (post-merge)  →  new branch: murphy/dev/routewise-router
```

## 3. RouteWiseRouter Implementation Plan

### 3.1 Architecture

RouteWiseRouter is a `BaseRouter` subclass, parallel to NimbusRouter.
Both share the same execution infrastructure (circuit breaker, health, fallback).

```
BaseRouter (abstract)
│   _select_adapter()    ← subclass decision point
│   _execute_adapter()   ← shared execution
│   _on_first_token()    ← optional hook
│   _on_completion()     ← NEW hook for predictor feedback
│
├── FixedRouter          (weighted random)
├── NimbusRouter         (SLO-aware local/remote outsourcing)
└── RouteWiseRouter      (cost-optimal multi-subscription routing)
```

### 3.2 New module: routing/routewise/

```
routing/routewise/
├── __init__.py
├── router.py              ← RouteWiseRouter (BaseRouter subclass)
├── quota_manager.py       ← PrimalDualQuotaManager (from experiment/)
├── concurrency_manager.py ← CAPQConcurrencyManager (from experiment/)
├── cost.py                ← CostCalculator adapted for real adapters
└── predictors/
    ├── __init__.py
    ├── base.py            ← OutputTokenPredictor interface
    └── ema.py             ← EMAOutputPredictor (from experiment/)
```

### 3.3 Adapter subscription tagging

Each adapter needs a subscription type. Injected via config, not code change to adapter.

```python
class SubscriptionType(Enum):
    QUOTA = "quota"           # S_Q: daily quota (e.g., Chutes)
    CONCURRENCY = "concurrency"  # S_C: fixed concurrency (e.g., Featherless, local GPU)
    API = "api"               # S_A: pay-per-token (e.g., Together)
```

RouteWiseRouter maps adapter → SubscriptionType at init time based on config.

### 3.4 Config extension

```yaml
# routing_strategy selects the active router
routing_strategy: routewise  # "fixed" | "nimbus" | "routewise"

# RouteWise-specific config
routewise:
  subscriptions:
    - name: chutes
      type: quota
      daily_quota: 5000
      monthly_fee: 20.0
      models: ["llama-3.3-70b-instruct", "qwen-2.5-72b-instruct"]

    - name: featherless
      type: concurrency
      concurrency_limit: 8
      monthly_fee: 25.0
      models: ["llama-3.3-70b-instruct"]

    - name: together
      type: api
      models: ["*"]

  primal_dual:
    value_lower_percentile: 5
    value_upper_percentile: 95

  predictor:
    type: ema
    alpha: 0.1
    warmup_requests: 50
```

### 3.5 Settings extension

```python
class RoutingStrategy(Enum):
    FIXED = "fixed"
    NIMBUS = "nimbus"
    ROUTEWISE = "routewise"
```

### 3.6 Bootstrap integration

In `serving/servers/bootstrap.py`, the router instantiation logic:

```python
strategy = settings.get_routing_strategy()

if strategy == RoutingStrategy.FIXED:
    router = FixedRouter(...)
elif strategy == RoutingStrategy.NIMBUS:
    router = NimbusRouter(fixed_router=fixed_router, settings=settings)
elif strategy == RoutingStrategy.ROUTEWISE:
    router = RouteWiseRouter(fixed_router=fixed_router, settings=settings)
```

### 3.7 RouteWiseRouter._select_adapter() decision flow

```
Request(model_id, messages, params) arrives
│
├── 1. Predict output length
│     pred = predictor.predict(model_id, input_tokens)
│     → QuantilePrediction(q10, q50, q90)
│
├── 2. Estimate API value (savings if we use subscription)
│     v_t = api_price_in * n_in + api_price_out * pred.q50
│
├── 3. Get shadow prices
│     θ_Q = L * (U/L)^(used_today / daily_quota)
│     θ_C = λ * weight * estimated_duration    (if S_C enabled)
│
├── 4. Compute net gains for each available adapter
│     For each adapter with subscription_type:
│       if QUOTA and quota_available:  G = v_t - θ_Q
│       if CONCURRENCY and slots_available: G = v_t - θ_C
│       if API: G = 0 (baseline)
│
├── 5. Select adapter = argmax(G) across all options
│
└── 6. Return selected adapter (BaseRouter handles execution)

Post-completion (via _on_completion hook):
  predictor.update(model_id, actual_output_tokens)
```

### 3.8 Key implementation detail: _on_completion hook

BaseRouter currently has `_on_first_token()`. We add `_on_completion()`:

```python
# In BaseRouter (add alongside existing _on_first_token)
def _on_completion(self, adapter, model_id, usage_metadata):
    """Called after request completes. Override for feedback loops."""
    pass

# In RouteWiseRouter
def _on_completion(self, adapter, model_id, usage_metadata):
    output_tokens = usage_metadata.get("completion_tokens", 0)
    self.predictor.update(model_id, output_tokens)
```

This is the same pattern as NimbusRouter's `_on_first_token` / `_remove_from_shadow_queue`.

### 3.9 Productionization delta from experiment/ code

| Component | Experiment source | Production changes needed |
|-----------|------------------|--------------------------|
| PrimalDualQuotaManager | `experiment/strategies/online/primal_dual.py` | Remove simulation coupling, add thread-safety |
| CAPQConcurrencyManager | Same file | Track real concurrent requests (vs simulated finish_time) |
| EMAOutputPredictor | `experiment/strategies/online/predictors/ema.py` | Per-model tracking, thread-safe update |
| CostCalculator | `experiment/cost.py` | Map adapter configs to pricing |
| QuotaManager | `experiment/quota.py` | Add time-based quota reset (wall clock vs simulation day) |

Core algorithm logic (threshold function, shadow price, value estimation) stays the same.
Main changes are: simulation artifacts → real-time state, numpy batch ops → per-request ops,
simulated time → wall clock time.

## 4. Completions Endpoint: Zero Change Needed

The beauty of the BaseRouter abstraction is that `serving/servers/routers/completions.py`
does NOT need to change. It currently does:

```python
strategy = settings.get_routing_strategy()
if strategy == RoutingStrategy.NIMBUS and nimbus_router:
    use nimbus_router
else:
    use fixed_router
```

This becomes:

```python
strategy = settings.get_routing_strategy()
if strategy == RoutingStrategy.NIMBUS and nimbus_router:
    use nimbus_router
elif strategy == RoutingStrategy.ROUTEWISE and routewise_router:
    use routewise_router
else:
    use fixed_router
```

Or better: unify all routers behind a single `router` dependency that is already
the correct type based on config. Then the completions endpoint just calls
`router.chat_completion()` / `router.stream_chat_completion()` with zero branching.

## 5. Observability & Experiment Data Collection

For research validation, RouteWiseRouter should log:

- Per-request: `{model, input_tokens, predicted_output, actual_output, subscription_used, api_cost_saved, shadow_price, quota_utilization}`
- Per-day: `{total_cost, optimal_cost (offline), competitive_ratio, quota_used, quota_wasted}`
- Predictor state: `{ema_mean, ema_std, warmup_progress}`

These feed into the ICML evaluation. The same metrics collected in simulation
(`experiment/scripts/run_online_experiments.py`) should be collected in production.

## 6. Implementation Timeline

| Phase | Task | Estimated |
|:-----:|------|:---------:|
| 0 | Merge nimbus → main (rebase + PR) | 1-2 days |
| 1 | Merge research artifacts (experiment/, paper, slides) → main | 0.5 day |
| 2 | Implement `routing/routewise/` module (port from experiment/) | 2-3 days |
| 3 | Config + Settings + Bootstrap integration | 1 day |
| 4 | Tests (unit + integration) | 1-2 days |
| 5 | Deploy to prototype, collect real traffic data | ongoing |
| **Total** | | **~6-8 days** |

## 7. Open Questions

1. **Real provider accounts**: Do we already have Chutes (S_Q) and Featherless (S_C) accounts
   set up for the prototype? Or do we start with simulated subscription constraints?

2. **Quota API**: Does Chutes expose a "remaining quota" API, or do we track quota
   purely client-side (as in the simulation)?

3. **Multi-model pricing**: The experiment code has `model_pricing` config. Do we want
   to hardcode OpenRouter pricing, or make it configurable?

4. **A/B testing**: Do we want to run Nimbus and RouteWise simultaneously on different
   model groups, or strictly one-at-a-time?

5. **Oracle experiment**: Should we run the oracle experiment (output length sensitivity
   analysis from `oracle_experiment_plan.md`) before or in parallel with the
   productionization? It only needs the simulation code.
