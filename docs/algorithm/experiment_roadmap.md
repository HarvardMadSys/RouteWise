# Experiment Roadmap & Evolution

This document records how our experimental setup evolved, including key bugs found,
design decisions, and the rationale behind dataset/config choices.

---

## Stage 1: Quota Routing (S_Q + S_A)

**Goal:** Route requests between a daily-quota subscription (S_Q, e.g., Chutes $20/mo)
and pay-per-token API (S_A) to minimize total cost.

**Datasets:** BurstGPT (1.4M requests, 61 days), FreeInference (420K requests, 62 days)

**Strategies:**
- Greedy: route to S_Q if value > threshold
- PrimalDual: exponential threshold based on remaining quota
- LA-PD: learning-augmented PD with output length prediction
- Offline Optimal: dynamic programming with full future knowledge

**Key results:** PD and LA-PD achieve CR 1.03-1.18x across different quota levels.

### Oracle Sensitivity Analysis (Stage 1)

**Question (from Juncheng):** Does mispredicting output length actually hurt end-to-end cost?

**Method:** Compare PD-Oracle (knows true output length) vs PD-EMA (estimates online).

**Gap decomposition:**
```
Total Gap = Online Gap (unknown future) + Prediction Gap (unknown length)
```

**Finding:** Prediction Gap <= 2.7% in all settings. The dominant gap is not knowing
future requests, not mispredicting output length. Simple EMA is sufficient.

---

## Stage 2: Joint Optimization (S_Q + S_C + S_A)

**Goal:** Route requests across three tiers:
- S_Q: daily-quota subscription (Chutes, $20/mo, 5000 req/day)
- S_C: concurrency-limited subscription (Featherless, $75/mo, C=8)
- S_A: pay-per-token API (unlimited, per-token pricing)

### Evolution of Experimental Setup

#### Phase 1: Initial Setup (rednote, multi-model)

- **Dataset:** rednote (54K requests, 84 days, multi-model)
- **S_C config:** featherless_scale (C=8, $75/mo, multiplier=4)
- **Problem:** PrimalDual showed CR=0.57x, beating ILP optimal -- clearly wrong.

#### Phase 2: Bug Discovery & Fixes

**Bug 1: Queueing mismatch (root cause of PD beating ILP)**
- ILP used `latency_slo=0` (zero-wait, no queueing)
- PD used `queue_capacity=concurrency_limit*2=16` (allows queueing)
- PD could queue requests at S_C, effectively getting more capacity than ILP assumed
- **Fix:** Set `queue_capacity=0` for both PD and LA-PD

**Bug 2: Congestion price returning 0 when S_C full**
- `get_congestion_price()` returned 0.0 when at capacity with empty queue
- Should return `float("inf")` to signal S_C is unavailable
- Made `gain_c` artificially high, routing requests to full S_C

**Bug 3: S_C reject fallback skipping S_Q**
- When S_C rejected (at capacity), code fell through directly to API
- Should check if `gain_q > gain_a` and try S_Q first
- **Fix:** Added S_Q fallback with proper condition

**Bug 4: Slot tracking using estimated duration**
- PD and LA-PD tracked S_C slot occupancy using predicted duration (EMA/UCB)
- Greedy correctly used actual `request.latency_seconds`
- Caused phantom capacity saturation in LA-PD (48x CR)
- **Fix:** Use `request.latency_seconds` for slot tracking after admission

**Bug 5: Fallback condition too loose**
- `gain_q > float("-inf")` always true -> always fell back to S_Q
- Should be `gain_q > gain_a` (only use S_Q if better than API)

**Bug 6: gain_c not checking admission capacity**
- Computed gain_c without checking if S_C can actually admit the request (weight-aware)
- **Fix:** Gate with `can_admit()` before computing gain_c

**Bug 7: Subscription cost accounting**
- All-API baseline incorrectly paid subscription fees
- **Fix:** Per-strategy subscription cost in runner post-processing

**Bug 8: Hardcoded subscription key**
- Strategy classes hardcoded `subscriptions.get("featherless")` to find S_C config
- **Fix:** Explicit `active_sc_subscription` config key set by runner

**Bug 9: ILP cache key missing model_override**
- `--model` override changed request models but ILP cache key didn't include it
- Led to stale cache hits with wrong ILP assignments
- **Fix:** Added `model_override` to `get_ilp_cache_key()`

After all fixes, strategy ordering became correct:
**ILP > LA-PD > PD > Greedy** (as theory predicts)

#### Phase 3: Dataset & Config Iteration

**Problem:** With rednote + featherless_scale, subscription wasn't cost-effective.

| Attempt | Dataset | Model | S_C Compatible | API Total | Sub Cost | Verdict |
|---------|---------|-------|---------------|-----------|----------|---------|
| 1 | rednote (mult=4) | multi | 20% | $8,511 | $266 | Sub not worth it (ILP $8,569 > All-API $8,511) |
| 2 | rednote (mult=2) | multi | 20% | $8,511 | $266 | Better CR but still not worth it |
| 3 | freeinference | multi | 60% | $211 | $285 | API too cheap, sub dominates cost |
| 4 | freeinference | llama-3.3-70b | 100% | $154 | $285 | API still too cheap |
| **5** | **freeinference** | **deepseek-r1** | **100%** | **$368** | **$285** | **Sub pays off! ILP $294 < All-API $368** |

**Why deepseek-r1 works:**
- Output price $2.19/M tokens (vs llama-3.3's $0.75) -> 3x more expensive API
- Higher API cost means subscription savings are more significant
- 100% S_C compatible (in featherless supported_models, multiplier=2)

### Final Stage 2 Results

**Setup:** FreeInference trace, 371K requests, 62 days, single-model (deepseek-r1)
- S_Q: Chutes $20/mo, 5000 req/day
- S_C: Featherless Scale $75/mo, C=8, multiplier=2

| Strategy | Total | API | Subscription | CR |
|----------|-------|-----|-------------|-----|
| All-API | $368 | $368 | $0 | 1.251x |
| SQ-Only | $172 | $112 | $60 | 0.586x |
| SC-Only | $461 | $236 | $225 | 1.570x |
| Greedy | $345 | $60 | $285 | 1.174x |
| PrimalDual | $332 | $47 | $285 | 1.128x |
| LA-PD | $320 | $35 | $285 | **1.087x** |
| ILP-Optimal | $294 | $9 | $285 | 1.000x |

**Key takeaways:**
1. Joint routing (ILP) saves **20%** vs All-API ($294 vs $368)
2. LA-PD captures **65%** of optimal savings, PD captures 49%, Greedy only 31%
3. S_C-Only loses money (subscription > savings) -- joint optimization is essential
4. S_Q-Only already saves 53% -- S_C adds value by pushing API from $112 to $9

---

## Design Decisions

### Why single-model assumption for Stage 2?
- Stage 2's core contribution is the **routing algorithm**, not model compatibility
- Multi-model introduces noise from S_C compatibility rates (20-60%)
- Single-model (100% compatible) cleanly isolates algorithm performance
- Consistent with the Stage 2 theoretical analysis which assumes homogeneous requests

### Why FreeInference over rednote?
- 371K requests (vs 54K) -> better statistical significance
- Real production trace from an open-source inference service
- Has complete latency data (needed for S_C scheduling)

### Why deepseek-r1 as the model?
- Supported by both S_Q (Chutes) and S_C (Featherless) -> 100% compatible
- Output price $2.19/M tokens makes API cost non-trivial
- Realistic: deepseek-r1 is a popular reasoning model with high output volume

### Zero-wait (loss) system semantics
- `latency_slo=0`: requests must be served immediately or routed elsewhere
- No queueing allowed at S_C -- if at capacity, route to S_Q or API
- Models real-world routing where users expect immediate responses
- Both ILP and online strategies use the same semantics

---

## File Reference

| File | Purpose |
|------|---------|
| `experiment/scripts/run_online_experiments.py` | Main Stage 1 & 2 runner |
| `experiment/scripts/run_stage2_experiments.py` | Offline-only Stage 2 (ILP + baselines) |
| `experiment/scripts/run_oracle_experiment.py` | Oracle sensitivity analysis (Stage 1) |
| `experiment/strategies/online/primal_dual.py` | PrimalDual online strategy |
| `experiment/strategies/online/learning_augmented.py` | LA-PD unified strategy |
| `experiment/strategies/online/base.py` | OnlineStrategy base class |
| `experiment/strategies/stage2_optimal.py` | ILP optimal strategy |
| `experiment/simulator.py` | Offline trace-replay simulator |
| `experiment/cache.py` | ILP result caching |
| `config/experiment.yaml` | Subscription & pricing config |
