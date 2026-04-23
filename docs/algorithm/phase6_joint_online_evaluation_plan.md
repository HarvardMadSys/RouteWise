# Phase 6 Joint Online Evaluation Plan

## 1. Goal

Upgrade the current real online harness from an OpenRouter provider selector into
a joint router that can make one unified decision across tiers and then evaluate
the new LP and new hedge design on a real workload for at least one hour.

The first online validation target is:

- model family: `MiniMax M2.7`
- duration: `1 hour`
- mode: `trace replay`
- objective: validate transport integration, the new body LP, and the new hedge
  trigger on real latency drift, not just in the synthetic simulator

## 2. Why Phase 5 Is Not Enough

The current online harness is in:

- `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/scripts/phase5_online_evaluation.py`

Phase 5 is useful, but it is still not a true joint evaluation for the current
paper story.

Current limitations:

1. It is OpenRouter-only.
   All providers come from one pricing file and are treated as flat API choices.
   There is no tier abstraction, no quota shadow price, and no concurrency price.

2. Its body selector is still the old policy family.
   The main policies are `lp_mix`, `smart_hedge`, and `v2_p50_hedge`.
   None of them implement the new request-driven budget LP.

3. Its hedge logic is still Phase 4 logic.
   The existing `SmartHedger` uses the old economic trigger and fastest-backup
   selection. It does not implement the new probability-only trigger or the new
   explorer backup policy.

Because of these gaps, running Phase 5 with `MiniMax M2.7` today would be a
useful smoke test, but it would not be a clean evaluation of the new joint
algorithm.

## 3. Main Design Decision

Do not keep extending `phase5_online_evaluation.py` in-place.

Instead, create a new sidecar online harness:

- script:
  `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/scripts/phase6_joint_online_evaluation.py`
- strategy helpers:
  `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/strategies/joint_online/`

This keeps the old Phase 5 results reproducible and avoids mixing two different
algorithm generations in one file.

## 3.1 Experimental Isolation Rule

The first joint online pilot must not reuse the Phase 5
`each request -> all policies simultaneously` execution model.

Reason:

- once `S_Q` becomes a real token-plan endpoint, multiple policies would share
  the same external quota, rate limit, and hedge/probe side effects
- that would turn a policy comparison into a multi-policy interference test

Therefore, the Phase 6 first-pass rule is:

- run one policy at a time
- each policy gets its own isolated run directory
- do not compare policies inside one shared live quota session

Shadow-only counterfactual policies can be added later, but they are not part of
the first implementation.

## 4. What "Joint" Means Online

For the online paper story, "joint" must mean a unified choice across provider
tiers, not just a better policy over one flat OpenRouter pool.

The online provider universe should therefore become:

- `S_A`: pay-per-token API providers
- `S_Q`: subscription / token-plan providers with marginal billed cost near zero
  but finite request budget over a window
- `S_C`: optional concurrency-limited providers if we have a stable live endpoint

For the first 1-hour M2.7 pilot, the minimum acceptable joint setup is:

- at least one `S_Q` provider
- multiple `S_A` providers

That gives us a real `S_Q + S_A` joint evaluation, which is already sufficient
for the main LP-budget and hedge story. `S_C` can be added later.

## 5. First-Pass M2.7 Inventory

The first online pilot should be based on `MiniMax M2.7`.

Two facts are already clear:

1. OpenRouter exposes `MiniMax M2.7` and its model pricing is currently
   `$0.30/M input` and `$1.20/M output`.
2. MiniMax native API also exposes `MiniMax-M2.7`, including token-plan style
   access.

Therefore, the first practical online inventory should be:

- `S_Q`:
  - MiniMax native token-plan endpoint for `MiniMax-M2.7`
- `S_A`:
  - OpenRouter-routed `MiniMax M2.7` providers

This is enough to test the new joint body selector and new hedging strategy.

## 6. Required New Metadata Layer

Phase 5 loads a simple pricing file:

- `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/data/openrouter_minimax_m25.json`

That format is not rich enough for joint routing. Phase 6 needs a provider
inventory file instead of a flat pricing map.

Recommended new file:

- `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/data/joint_minimax_m27_online.json`

Recommended schema:

```json
{
  "providers": [
    {
      "name": "MiniMax_Native_SQ",
      "tier": "quota",
      "transport": "minimax_native",
      "model": "MiniMax-M2.7",
      "input_price_per_m": 0.0,
      "output_price_per_m": 0.0,
      "quota_window_sec": 18000,
      "quota_requests": 1500,
      "billing_mode": "subscription",
      "plan_fee_usd": null,
      "weight": 1.0
    },
    {
      "name": "Chutes_OR",
      "tier": "api",
      "transport": "openrouter_provider",
      "model": "minimax/minimax-m2.7",
      "provider_hint": "Chutes",
      "input_price_per_m": 0.30,
      "output_price_per_m": 1.20,
      "weight": 1.0
    }
  ]
}
```

Key point:

- billing cost should be request-accurate when possible
- effective cost should be derived online from tier metadata plus shadow prices

### 6.1 Cost Metric Contract

To avoid mixing internal decision cost with reported experiment cost, Phase 6
must keep two cost metrics separate:

1. `incremental_api_spend_usd`
   - primary reported online cost metric
   - includes only pay-per-request / pay-per-token spend actually incurred
   - token-plan subscription fee is not amortized into this number

2. `effective_cost`
   - internal routing score used by the selector
   - includes shadow-price penalties for scarce `S_Q` capacity

The first online pilot should report `incremental_api_spend_usd` as the main
table metric and clearly state that subscription sunk cost is excluded. If we
later want an amortized economic-cost table, it should be reported separately.

## 7. New Module Layout

Recommended new modules:

- `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/strategies/joint_online/types.py`
  - `JointProviderSpec`
  - `JointRequestContext`
  - `JointDecision`

- `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/strategies/joint_online/transports.py`
  - `OpenRouterProviderTransport`
  - `MiniMaxNativeTransport`
  - optional future `OpenAICompatTransport`

- `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/strategies/joint_online/shadow_price.py`
  - online `c_eff` calculation
  - quota shadow price
  - optional concurrency shadow price

- `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/strategies/joint_online/profile.py`
  - rolling TTFT profile
  - rolling miss-rate profile
  - explorer feedback updates

- `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/strategies/joint_online/body_lp.py`
  - old body baseline
  - new budgeted body LP

- `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/strategies/joint_online/hedge.py`
  - new probability-only hedge trigger
  - deterministic safe-cheapest backup selection for the first pass
  - optional explorer/random extensions in a later pass

- `/Users/realtmxi/Desktop/NSDI2027_RouteWise/experiment/scripts/phase6_joint_online_evaluation.py`
  - orchestration, replay, warmup, CSV logging, summaries

## 8. Body Selector to Implement

### 8.1 Old Body Baseline

Keep one clean old-body baseline online:

```text
min Σ_j π_j * c_eff_j
s.t. Σ_j π_j * F_j(SLO) >= 0.99
```

This is needed to isolate the value of the new LP online, the same way we did
in the synthetic study.

### 8.2 New Mainline LP

Mainline online selector:

```text
min Σ_j π_j * mean_TTFT_j
s.t. Σ_j π_j * c_eff_j <= τ * v_hat_i
```

Definitions:

- `mean_TTFT_j`: rolling body-latency estimate from online profile
- `c_eff_j`: current effective cost from tier-aware shadow pricing
- `v_hat_i`: request-side estimated API spend anchor for request `i`
- `τ`: Pareto knob

For the first online pilot, use:

- default mainline: `τ = 0.75`
- optional ablation: `τ = 0.50`

Do not launch the first online pilot with `τ = 0.25`.
Synthetic already shows that this point is mostly useful as a Pareto/negative
case, not as a default operating point.

## 9. Hedge Strategy to Implement

The first online hedge should be simpler than the final synthetic exploration
stack. The goal of the first pass is to validate the new hedge trigger itself,
not every optional backup-selection variant at once.

### 9.1 Trigger

Use the conditional-probability rule:

```text
P(not violate | t) + P(violate | t) * P(backup succeeds) >= 0.99
```

Operationalization:

- compute the latest safe dispatch time `t*`
- dispatch backup if the primary has still not produced TTFT by `t*`

### 9.2 Backup Selection

For the first online pilot, use a deterministic backup rule:

- `safe_cheapest` only

Definition:

- among currently available non-primary providers, keep those that can satisfy
  the SLO on their own
- choose the cheapest one
- if none is individually SLO-safe, fall back to the fastest available backup

Do not include `random_explorer` or adaptive random shrinking in the first
online pilot. Those branches should be deferred to a later follow-up.

### 9.3 Explorer Feedback

When hedge fires:

- primary TTFT sample always updates the profile
- backup TTFT sample also updates the profile

This is `hedge-as-probe`.

### 9.4 Dedicated Probing

Keep a dedicated probing path in the first online mainline.

Reason:

- synthetic already showed that pure explorer is not enough in low-hedge-rate
  scenarios
- if the online hedge rate is low, non-primary providers will otherwise drift
  stale silently

Therefore:

- mainline default: keep low-rate probing
- optional ablation: `probe_rate = 0`

## 10. Transport Abstraction

This is the largest structural change from Phase 5.

Today, Phase 5 assumes every provider can be called through the same OpenRouter
request path with an optional provider hint. That assumption breaks as soon as
we add native MiniMax token-plan traffic.

Phase 6 should therefore move request execution behind a transport interface:

```text
send(provider_spec, prompt, max_tokens) -> SingleRequestResult
```

At minimum we need:

- OpenRouter transport:
  - model = `minimax/minimax-m2.7`
  - optional provider hint
- MiniMax native transport:
  - model = `MiniMax-M2.7`
  - token-plan auth

The rest of the evaluator should stay transport-agnostic.

## 11. Logging and Diagnostics

The online harness should log more than Phase 5.

Add the following fields to the per-request log:

- `tier`
- `transport`
- `effective_cost`
- `budget_limit`
- `budget_utilization`
- `selected_weights`
- `primary_provider`
- `backup_provider`
- `backup_selection_mode`
- `backup_random_prob`
- `explorer_feedback_applied`
- `quota_fraction_used`
- `quota_shadow_price`

This is necessary if we want to explain online behavior rather than only report
aggregate latency.

## 12. Mainline Policy Set for the First Online Pilot

Do not overload the first run with too many policies.

Recommended first 1-hour set:

- `openrouter_auto`
- `old_body`
- `old_body_hedge`
- `joint_budget_vhat_t75`
- `joint_budget_vhat_t75_hedge`

Optional additional comparators:

- `sort_latency`
- `cheapest_fixed`

Avoid bringing every old policy into the first M2.7 pilot. The goal is to
validate the new joint mainline, not to repeat the entire Phase 5 policy zoo.

Also note:

- these policies should be run as separate live runs, not simultaneously inside
  one shared quota session

## 13. One-Hour M2.7 Pilot Plan

### 13.1 Preconditions

Before running:

1. Build `joint_minimax_m27_online.json`
2. Verify all listed endpoints are callable
3. Run a short warmup and probing-only connectivity pass

### 13.2 First Run Configuration

Recommended first run:

- model family: `MiniMax M2.7`
- duration: `3600s`
- warmup: `300s`
- replay mode: real prompts, or at minimum a frozen token trace with request-level
  `prompt_tokens` and `max_tokens`
- cost cap: conservative, then increase only if the run is stable

Arrival-only replay is not valid for the first online LP-budget validation,
because the selector budget is request-specific through `τ * v_hat_i`.

### 13.3 First Success Criteria

The first 1-hour pilot is successful if:

1. `joint_budget_vhat_t75` forms a sensible cost/body point online
2. `joint_budget_vhat_t75_hedge` lowers tail risk without exploding cost
3. dedicated probing cost remains modest
4. no transport family silently starves from stale profile data

The first 1-hour pilot is not required to prove the full quota shadow-price
story. Its primary purpose is to validate:

- transport correctness
- online body-LP behavior
- online probability-hedge behavior
- profile freshness under real drift

## 14. What We Should Not Do

Do not:

1. pretend Phase 5 with `M2.7` is already a joint experiment
2. directly mutate `phase5_online_evaluation.py` until it becomes a second,
   incompatible framework
3. launch a 24-hour run before a clean 1-hour M2.7 pilot passes
4. start with `τ = 0.25` as the online default
5. interpret a 1-hour run as a quota-shadow-price validation if the live quota
   barely moves during the run

## 15. Recommended Next Steps

The implementation order should be:

1. Create `joint_minimax_m27_online.json`
2. Add provider transport abstraction
3. Port the new body LP into `joint_online/body_lp.py`
4. Port the new probability hedge with deterministic `safe_cheapest` backup into
   `joint_online/hedge.py`
5. Build `phase6_joint_online_evaluation.py`
6. Run one 1-hour M2.7 pilot with real prompts and low-rate dedicated probing
7. Only then decide whether to add `random_explorer`, adaptive backup mixing,
   or pure-explorer ablations
8. Only then decide whether to fold parts of Phase 6 back into the main online
   experimentation stack

## 16. Bottom Line

We are ready to move from synthetic to online validation.

But the correct next step is not "run Phase 5 on M2.7 and call it joint".

The correct next step is:

- build a small Phase 6 sidecar,
- make the provider universe truly joint,
- port the new LP and a simple deterministic version of the new hedge faithfully,
- then run a 1-hour MiniMax M2.7 pilot.
