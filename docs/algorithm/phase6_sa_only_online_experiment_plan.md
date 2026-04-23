# Phase 6 S_A-Only Online Experiment Plan

## 1. Objective

Run a simple online validation of the new latency-layer method in the existing
OpenRouter `S_A` setting before attempting a full cross-tier joint deployment.

This plan intentionally does not introduce real `S_Q` token-plan providers.
It asks one focused question:

> If we keep the provider universe flat and API-only, does the new body LP plus
> the new hedge outperform the old online method on the same real OpenRouter
> workloads we already used?

This is a pre-joint online validation, not the final joint paper experiment.

This first-pass online study must also avoid the methodology pitfall from the
old invalid Phase 5 replay:

- no closed-loop "wait for all policies to finish, then advance" replay
- no shared mutable router state across compared policies
- no arrival-only replay without real prompt or frozen token context
- no paper-grade claim based on harness modes that distort the trace arrival
  process

## 2. Scope

We will reuse the model families that already anchor our prior online story:

- `deepseek/deepseek-v3.2`
- `qwen/qwen3-235b-a22b-2507`
- `meta-llama/llama-3.3-70b-instruct`

These are the right first-pass models because:

- they were already part of our earlier online reasoning
- they cover different provider ecosystems and tail profiles
- keeping the model set unchanged lets us compare old vs new methods cleanly

## 3. What Must Be Re-run

Because the body selector and hedge semantics changed, the main online
evaluation must be re-run on the old model set.

But we do **not** need to re-run every historical artifact.

### 3.1 Re-run Required

For each of the three models above, we need a fresh online run for:

- `openrouter_auto`
- `sort_price`
- `sort_throughput`
- `sort_latency`
- `cheapest_fixed`
- `lp_mix`
- `smart_hedge`
- `budget_vhat_t75`
- `budget_vhat_t75_hedge`

This is the minimum set required to answer:

1. Does the new body LP beat the old body LP?
2. Does the new full method beat the old full method?
3. How far are both from the OpenRouter default?
4. Where is the cheapest external anchor?
5. Where are the OpenRouter sort-based anchors?

### 3.2 Re-run Not Required in the First Pass

We do not need to immediately re-run:

- every old ablation
- provider-percentile budget variants
- `τ = 0.25`
- random-explorer backup mixing
- adaptive backup mixing
- full joint `S_Q + S_A` transport experiments

Those belong to follow-up rounds, not the first S_A-only refresh.

## 4. Mainline Methods for This S_A-Only Study

### 4.1 Old Baselines

- `lp_mix`
  - old body selector
- `smart_hedge`
  - old full method

### 4.2 New Methods

- `budget_vhat_t75`
  - new body LP
- `budget_vhat_t75_hedge`
  - new body LP + new hedge

### 4.3 External Baseline

- `openrouter_auto`
- `sort_price`
- `sort_throughput`
- `sort_latency`
- `cheapest_fixed`

These stay as external system baselines and anchors, not as the main
algorithmic baseline.

## 5. New Algorithm Definition in S_A-Only Setting

In the API-only setting:

- `c_eff_j = c_j`
- there is no quota shadow price
- there is no concurrency shadow price

So the new LP becomes:

```text
min Σ_j π_j * mean_TTFT_j
s.t. Σ_j π_j * c_j <= τ * v_hat_i
```

with:

- `τ = 0.75`
- `v_hat_i` = request-level API-cost anchor derived from model reference price

Recommended anchor:

```text
v_hat_i =
  ref_input_price_per_m  * prompt_tokens_i / 1e6 +
  ref_output_price_per_m * max_tokens_i    / 1e6
```

The reference price should be the model-level nominal API price, not a
provider-specific price.

## 6. New Hedge Definition in S_A-Only Setting

To keep the method simple, the first S_A-only online pass should use:

### 6.1 Trigger

```text
P(not violate | t) + P(violate | t) * P(backup succeeds) >= 0.99
```

### 6.2 Backup Selection

Use deterministic `safe_cheapest` only:

- among available non-primary providers, keep those that can satisfy the SLO
  on their own
- choose the cheapest one
- if none is individually SLO-safe, fall back to the fastest available backup

### 6.3 Explorer / Probing

For this first pass:

- keep `hedge-as-probe` feedback
- keep low-rate dedicated probing
- do **not** add random explorer mixing
- do **not** add adaptive backup mixing

The goal is to minimize new variables.

## 7. Ground Truth and Baselines

In this online experiment there is no request-level provider label ground truth.

The ground truth is the measured outcome of each independent run:

- mean / P50 / P99 TTFT
- SLO violation rate
- incremental API spend
- provider mix
- hedge rate

The main baseline is:

- `smart_hedge`

because it is the old full method.

The external baseline is:

- `openrouter_auto`

## 8. Run Structure

For the paper-grade first pass, we should **not** reuse the old
all-policies-per-request harness mode as the main evaluation mode, even though
this study is `S_A`-only.

The reason is methodological conservatism. In `S_A`-only there is no shared
real `S_Q` quota, so simultaneous multi-policy replay is less problematic than
in the full joint setting. However, we already know from the old invalid 7-day
replay that online harness semantics can quietly dominate the result. Since
cost is not the limiting factor here, the main paper-grade result should use
the cleanest replay structure:

- one policy per replay run
- one run directory per `(model, policy)` pair
- the same trace window for all compared policies
- the same replay speed and stop conditions for all compared policies
- open-loop arrival-driven replay only
- no closed-loop round gating
- policy-local mutable router state only

This means:

- Run 1: model `M`, policy `openrouter_auto`
- Run 2: model `M`, policy `sort_price`
- Run 3: model `M`, policy `sort_throughput`
- ...
- Run 9: model `M`, policy `budget_vhat_t75_hedge`

All policies still see the same workload, but they do so in independent runs.

The old simultaneous multi-policy mode may still be useful for quick debugging
or paired sanity checks, but it should not be the primary evidence path for
this S_A-only refresh.

## 9. Required Inputs

For each model we need:

1. pricing / provider file
2. real prompt trace or frozen token trace
3. model-specific OpenRouter provider availability

Current status:

- existing:
  - `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/data/openrouter_qwen3_235b.json`
- still needed:
  - `openrouter_deepseek_v32.json`
  - `openrouter_llama33_70b.json`

If an old pricing file already exists elsewhere in the repo history, we should
reuse it instead of regenerating it by hand.

## 10. Metrics

For each model and policy, report:

- `mean TTFT`
- `P50`
- `P90`
- `P99`
- `SLO violation rate`
- `incremental API spend`
- `hedge rate`
- `provider mix`

Optional diagnostics:

- LP chosen weights
- budget utilization
- backup provider distribution

## 11. Success Criteria

This S_A-only pre-joint study is successful if, across the three models:

1. `budget_vhat_t75` improves body metrics relative to `lp_mix`
2. `budget_vhat_t75_hedge` improves tail metrics relative to `smart_hedge`
3. the cost increase remains explainable and bounded
4. the method is not dominated by `openrouter_auto`
5. the conclusions remain stable under independent per-policy replay runs

We do not require the new method to dominate every baseline on every metric.
We require it to form a cleaner and more defensible cost-latency tradeoff.

## 12. Why This Comes Before Full Joint

This step answers a simpler question first:

> Does the new LP + new hedge work online at all on the same real model families
> where we previously validated the old method?

If the answer is yes, then we move to the harder cross-tier joint transport
problem with much higher confidence.

If the answer is no, we stop early before spending effort on full `S_Q + S_A`
integration.

## 13. Implementation Recommendation

Do not build a full Phase 6 joint transport system first.

Instead:

1. fork the existing Phase 5 harness into a small sidecar
2. preserve only the corrected open-loop replay semantics from the post-fix
   Phase 5 harness
3. make the sidecar execute exactly one policy per run
4. replace only:
   - old body LP -> `budget_vhat_t75`
   - old hedge trigger -> new probability hedge
   - old backup policy -> deterministic `safe_cheapest`
5. keep real prompt / token context, not arrival-only replay
3. keep everything else as stable as possible

Recommended script:

- `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/scripts/phase6_sa_online_evaluation.py`

## 14. Bottom Line

Yes, we need to re-run the main online evaluation on the old model set because
the method changed.

But we do **not** need to re-run every old experiment or every old ablation.

The right first step is a narrow S_A-only refresh on:

- DeepSeek
- Qwen
- Llama 3.3

with only:

- `openrouter_auto`
- `lp_mix`
- `smart_hedge`
- `budget_vhat_t75`
- `budget_vhat_t75_hedge`
