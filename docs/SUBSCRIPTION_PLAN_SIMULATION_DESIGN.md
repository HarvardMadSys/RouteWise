# Subscription Plan Simulation Design

> Design doc for RouteWise simulator cost-layer §1.2 and §1.3:
> quota-style and concurrency-style subscription plans. This document
> defines how to simulate Chutes, MiniMax subscription tiers, Featherless,
> and similar subscription plans without confusing routing marginal cost
> with reported invoice cost.

Last updated: 2026-05-06.

---

## 1. TL;DR

Cost-layer §1.2 and §1.3 should simulate **specific subscription plans**,
not anonymous `quota_q1..q4` or `concurrency_c1..c4` resources.

The target shape is:

```text
experiments/
  subscription_plans.yaml          # shared quota / concurrency plan facts
  subscriptions.py                 # shared loader/dataclass for experiment code

experiments/simulation/
  cost_layer.py                    # emits parameterized plan runs
  common.py                        # provider builder + cost summary helpers
```

The public CLI should keep one quota scenario and one concurrency scenario,
with the product plan and subscription count as explicit parameters:

```bash
routewise simulator cost-layer \
  --scenario quota \
  --subscription-plan chutes \
  --subscription-count 2

routewise simulator cost-layer \
  --scenario concurrency \
  --concurrency-plan featherless_premium \
  --concurrency-count 1
```

Artifact labels may still include the expanded parameter values, e.g.
`quota__plan=chutes__n=2`, but that is output metadata, not a public
scenario name.

The key design rule:

```text
Routing cost for S_Q:
  marginal request cost = 0 inside quota
  effective cost = quota shadow price ψ(z)

Routing cost for S_C:
  marginal request cost = 0 while capacity is available
  effective cost = concurrency shadow price λ(u_weighted) * concurrency_cost(model)

Reported experiment cost:
  total_cost_usd = api_cost_usd + subscription_fixed_cost_usd
```

So Chutes and Featherless are still zero marginal cost **for routing** once
purchased, but the final cost bar/table includes the prorated fixed
subscription fee.

---

## 2. Problem

The current cost-layer quota and concurrency scenarios are generic:

```text
cost_layer_quota_q1
cost_layer_quota_q2
cost_layer_quota_q3
cost_layer_quota_q4
cost_layer_concurrency_c1
cost_layer_concurrency_c2
cost_layer_concurrency_c3
cost_layer_concurrency_c4
```

and are built from hard-coded capacity knobs in
`experiments/simulation/cost_layer.py`. This is now too ambiguous.

The opposite extreme, expanding every product/count combination into public
scenario names like `cost_layer_quota_chutes_q1`, is also not the right final
shape. That moves the ambiguity into a long scenario catalogue and makes
`plan × n` look like separate experiment types. They are parameters of the
same plan-backed experiment.

We need to answer a paper question:

> Given a real subscription plan, how many subscriptions/accounts should
> RouteWise buy, and how should it allocate the quota across requests?

For §1.3, the analogous question is:

> Given a real concurrency subscription plan, how many subscriptions/accounts
> should RouteWise buy, and which in-flight requests should consume the scarce
> concurrency capacity?

That question depends on plan facts:

- quota size and reset window
- concurrency allotment and per-model concurrency cost
- monthly fee
- model/provider identity
- whether the fee is confirmed enough to make dollar-cost claims

Those facts should be declared once in a small experiment config file, then
consumed by `cost_layer.py`.

---

## 3. Conceptual Boundary

### 3.1 Routing marginal cost is not invoice cost

For a subscription plan already bought for the experiment window, serving
one more request inside quota has no additional API invoice. Therefore the
provider's marginal token price remains zero:

```python
TieredProvider(
    tier=ProviderTier.S_Q,
    input_cost_per_token=0.0,
    output_cost_per_token=0.0,
    quota=QuotaState(...),
)
```

That is correct for dispatch.

What RouteWise should compare during routing is the **opportunity cost of
quota**, represented by the shadow price:

```text
c_eff(S_Q, request, z) = ψ(z; L, U)
```

where `z` is quota fraction used and `L/U` are workload-level cost-envelope
calibration values.

### 3.2 Reported cost includes the fixed fee

The paper cost bar/table should not pretend subscriptions are free. It
should report:

```text
api_cost_usd
subscription_fixed_cost_usd
total_cost_usd = api_cost_usd + subscription_fixed_cost_usd
```

For a monthly subscription:

```text
billing_period_days = 30
subscription_fixed_cost_usd =
  monthly_fee_usd * num_subscriptions * (trace_days / billing_period_days)
```

This is paid whether or not RouteWise uses every quota slot.

### 3.3 Why not amortize fee into per-request routing cost?

Do not add `$20 / (30 * 5000)` to every Chutes request during routing.

That would mix two different decisions:

1. **Purchase decision:** should we buy `q` subscriptions?
2. **Dispatch decision after purchase:** which requests deserve scarce
   zero-marginal-cost quota?

In this simulator, `q1..q4` scenarios answer the purchase decision by
running separate experiments. Inside each fixed `q` scenario, RouteWise's
dispatch decision should be governed by quota scarcity, not by charging a
second fake per-request fee for an already-paid plan.

---

## 4. Plan Configuration

Add:

```text
experiments/subscription_plans.yaml
```

This file belongs under `experiments/`, not under `experiments/simulation/`,
because simulator, real-evaluation, and the older offline-stage experiments
all need the same subscription facts. It should not live in `rwsim/` because
`rwsim` is the generic engine and should not know Chutes/MiniMax product
facts. It should not live under `experiments/simulation/latency_profiles/`
because these plans are billing/capacity facts, not latency samples.

Existing subscription facts in `experiments/offline_stage/configs/experiment.yaml`
must not remain an independent source of truth. The migration target is:

```text
experiments/subscription_plans.yaml        # canonical plan facts
experiments/subscriptions.py               # canonical loader/dataclass
experiments/offline_stage/...              # imports or validates against canonical facts
experiments/real_evaluation/...            # imports or validates against canonical facts
```

During migration, old YAML files may keep their local experiment settings,
but duplicated plan facts such as Chutes `$20/month` and `5000/day` must be
removed in the same PR that introduces the canonical file. Validators are
not a substitute for a single source of truth; they are only acceptable as
temporary guardrails while removing duplicated fields.

### 4.1 Schema

```yaml
plans:
  chutes:
    display_name: "Chutes"
    tier: quota
    billing_mode: subscription
    monthly_fee_usd: 20.0
    quota_windows:
      - {name: daily, quota_requests: 5000, quota_window_sec: 86400}
    subscription_counts: [1, 2, 3, 4, 5, 6, 8]
    eligible_sections: [cost_layer_quota, end_to_end]
    cost_claim_allowed: true
    source: "experiments/offline_stage/configs/experiment.yaml subscriptions.chutes"
    notes: "Chutes public plan: $20/mo for 5000 requests/day."

  minimax_subscription_starter:
    display_name: "MiniMax Starter"
    tier: quota
    billing_mode: subscription
    monthly_fee_usd: 10.0
    quota_windows:
      - {name: five_hour, quota_requests: 1500, quota_window_sec: 18000}
      - {name: weekly_allowance, quota_requests: 15000, quota_window_sec: 604800}
    subscription_counts: [1, 2, 3, 4]
    eligible_sections: [cost_layer_quota]
    cost_claim_allowed: true
    source: "User-provided MiniMax pricing screenshot, 2026-05-06"
    notes: "$10/mo; 1500 model requests / 5h; weekly allowance is 10x the 5-hour quota."

  minimax_subscription_plus:
    display_name: "MiniMax Plus"
    tier: quota
    billing_mode: subscription
    monthly_fee_usd: 20.0
    quota_windows:
      - {name: five_hour, quota_requests: 4500, quota_window_sec: 18000}
      - {name: weekly_allowance, quota_requests: 45000, quota_window_sec: 604800}
    subscription_counts: [1, 2, 3, 4]
    eligible_sections: [cost_layer_quota]
    cost_claim_allowed: true
    source: "User-provided MiniMax pricing screenshot, 2026-05-06"
    notes: "$20/mo; 4500 model requests / 5h; weekly allowance is 10x the 5-hour quota."

  minimax_subscription_max:
    display_name: "MiniMax Max"
    tier: quota
    billing_mode: subscription
    monthly_fee_usd: 50.0
    quota_windows:
      - {name: five_hour, quota_requests: 15000, quota_window_sec: 18000}
      - {name: weekly_allowance, quota_requests: 150000, quota_window_sec: 604800}
    subscription_counts: [1, 2]
    eligible_sections: [cost_layer_quota]
    cost_claim_allowed: true
    source: "User-provided MiniMax pricing screenshot, 2026-05-06"
    notes: "$50/mo; 15000 model requests / 5h; weekly allowance is 10x the 5-hour quota."

  featherless_premium:
    display_name: "Featherless Premium"
    tier: concurrency
    billing_mode: subscription
    monthly_fee_usd: 25.0
    concurrency_allotment: 4
    model_concurrency_costs_by_class:
      le_15b: 1
      24_34b: 2
      ge_70b: 4
    default_model_class: ge_70b
    model_class_overrides:
      llama-3.3-70b-instruct: ge_70b
      qwen3-coder-30b: 24_34b
      llama-4-scout: le_15b
    subscription_counts: [1, 2, 3, 4]
    eligible_sections: [cost_layer_concurrency, end_to_end]
    cost_claim_allowed: true
    source: "Featherless docs: Plans and Concurrency Limits, checked 2026-05-06"
    notes: "Concurrency is weighted capacity, not request count. Premium allotment=4, so one 70B request with cost=4 fills the plan."

```

### 4.2 Field semantics

| Field | Meaning |
|---|---|
| `monthly_fee_usd` | Fixed subscription fee. `null` means we can simulate routing/utilization but not make total-cost claims. |
| `quota_windows` | One or more quota constraints. A request can use the plan only if every window has remaining quota. Chutes has one daily window; MiniMax subscription tiers have both a 5-hour quota and a weekly allowance. |
| `concurrency_allotment` | Total weighted concurrency capacity for one subscription/account. Featherless Premium has allotment `4`; this is not four arbitrary requests. |
| `model_concurrency_costs_by_class` | Weighted capacity cost by model-size class. Featherless documents cost classes such as `le_15b=1`, `24_34b=2`, and `ge_70b=4`. |
| `default_model_class` | Fallback class for model IDs that do not match a known override. Use a conservative default such as `ge_70b` for §1.3 paper smoke rather than silently dropping requests from S_C. |
| `model_class_overrides` | Optional workload/provider-specific mapping from trace model IDs to model-size classes. This keeps OpenRouter-style IDs and trace aliases out of the core capacity state. |
| `subscription_counts` | Paper sweep counts for this plan. Avoid running counts that saturate the whole workload and stop exercising quota scarcity. |
| `eligible_sections` | Which experiment sections may use this plan. §1.2 main figures should use plans tagged `cost_layer_quota`; §1.3 should use plans tagged `cost_layer_concurrency`. |
| `cost_claim_allowed` | Whether paper figures may include `total_cost_usd` for this plan. |
| `source` | Human-auditable provenance. |

---

## 5. Scenario Design

### 5.1 One scenario, explicit parameters

Keep the public scenario shape small:

```text
--scenario quota
--subscription-plan <plan_id>
--subscription-count <n>
```

Examples:

```bash
routewise simulator cost-layer --scenario quota --subscription-plan chutes --subscription-count 1
routewise simulator cost-layer --scenario quota --subscription-plan chutes --subscription-count 4
routewise simulator cost-layer --scenario quota --subscription-plan minimax_subscription_plus --subscription-count 2
```

This is better than both old options:

- `cost_layer_quota_q1` hides which product plan is being tested.
- `cost_layer_quota_chutes_q1` explodes a parameter sweep into many scenario
  names and recreates the grid-style API we have been deleting.

The run output should still write the resolved parameter values into
metadata and filenames/directories, for example:

```text
scenario = "quota"
subscription_plan = "chutes"
subscription_count = 2
artifact_label = "quota__plan=chutes__n=2"
```

### 5.2 Subscription count sweep

For each plan:

```text
q1 = 1 subscription/account
q2 = 2 subscriptions/accounts
q3 = 3 subscriptions/accounts
q4 = 4 subscriptions/accounts
```

The quota capacity per reset window is:

```text
quota_requests_per_window = quota_window.quota_requests * q
quota_window_sec = quota_window.quota_window_sec
```

This is more realistic than the current generic `_QUOTA_SIZE_PER_PROVIDER`
because Chutes `q1` means `5000/day`, while MiniMax subscription tiers have
both 5-hour and weekly limits:

- MiniMax Starter `q1` = `1500/5h` and `15000/week`
- MiniMax Plus `q1` = `4500/5h` and `45000/week`
- MiniMax Max `q1` = `15000/5h` and `150000/week`

Quota resets on each configured quota window. For example, a 30-day Chutes
run with `q=1` presents roughly 30 sequential daily windows of `5000`
requests each. MiniMax subscription tiers have two constraints, and both
must have remaining quota. The simulator's quota state should handle these
resets deterministically.

Before running a paper figure, the section should bucket requests by quota
window and compute whether quota is saturated in every window. Do not use
only aggregate monthly capacity; a bursty trace can still exceed daily quota
inside one window even if total capacity across the month covers all
requests.

```text
window_id(request, quota_window) =
  floor((request.timestamp - trace_start) / quota_window.quota_window_sec)
request_count_by_window = count requests per (quota_window, window_id)
quota_capacity_per_window = quota_window.quota_requests * q
```

Set:

```text
quota_saturated_in_trace =
  all(
    request_count_by_window[quota_window, w] <= quota_capacity_per_window(quota_window)
    for every quota_window and every window w
  )
```

When `quota_saturated_in_trace=true`, the run does not exercise scarcity:
RouteWise can send essentially every request in every reset window to quota.
It should not be used as a main paper q-sweep result. This is why each plan
has its own `subscription_counts`; Chutes can use `[1, 2, 3, 4, 5, 6, 8]`, while
some MiniMax tiers may saturate after fewer counts.

The CLI should support either a single value:

```bash
--subscription-count 2
```

or a comma-separated sweep:

```bash
--subscription-counts 1,2,3,4,5,6,8
```

Similarly, the full paper run can accept:

```bash
--subscription-plans chutes,minimax_subscription_plus
```

These are section-specific flags. They do not need to become generic
`rwsim` engine concepts.

### 5.3 Provider set per scenario

All §1.2 quota scenarios use `latency_family = heavy_tail`, which is the
simulator's LogNormal latency family. §1.2 is a cost-layer quota-scarcity
experiment, so distribution robustness is a follow-up check after the main
subscription-count sweep rather than a dimension of the main grid.

Each quota run should contain:

```text
1 aggregate S_Q provider representing q subscriptions/accounts of the selected plan
a fixed S_A fallback provider set shared by every q in the sweep
```

Default provider shape:

```text
all q values: 1 aggregate S_Q provider + the same S_A fallback provider set
```

Do **not** model q identical subscription accounts as q separate providers by
default. Equivalent providers create noisy, arbitrary sub-provider fractions
and make goldens less stable. For §1.2, q means "aggregate purchased quota":

```text
QuotaState(size = q * quota_window.quota_requests, window_sec = quota_window.quota_window_sec)
```

For plans with multiple quota windows, use a composite quota state with one
aggregate counter per window. A request consumes one unit from every
configured window. If the simulator only supports single-window `QuotaState`,
MiniMax subscription tiers must be rejected until composite quota support lands.

If a later ablation truly needs per-account behavior, add it explicitly.

The aggregate model assumes synchronous reset across accounts. For Chutes,
that matches the intended simulator setup: buy the same plan at the same
time, with the same daily reset boundary. Staggered per-account resets are
out of scope; if a future ablation requires them, model accounts as separate
`QuotaState` instances.

The §1.2 aggregate-q abstraction is routing-equivalent for cost analysis.
Real evaluation realizes the selected `n=k` as `k` actual provider accounts
or API keys. That is a deployment detail, not a routing-model change.

Fallback API providers use the cost-layer API price ladder already defined
for §1.1 unless the section intentionally tests a real-world API price set.

### 5.4 No concurrency in §1.2

§1.2 is quota-only. It treats only the documented request quota windows as
binding.

This keeps the section focused on the paper question:

```text
Given a subscription quota and a monthly fee, how many subscriptions should
RouteWise buy, and which requests should consume the scarce quota?
```

Concurrency is handled by §1.3 below, not as a hidden extra constraint
inside the §1.2 quota sweep.

### 5.5 Concurrency subscription §1.3

§1.3 is the concurrency-plan counterpart to §1.2. It should not reuse the
old public scenario names `cost_layer_concurrency_c1..c4`. The public shape is:

```text
--scenario concurrency
--concurrency-plan <plan_id>
--concurrency-count <n>
```

Examples:

```bash
routewise simulator cost-layer --scenario concurrency --concurrency-plan featherless_premium --concurrency-count 1
routewise simulator cost-layer --scenario concurrency --concurrency-plan featherless_premium --concurrency-count 4
```

The run output should still write resolved parameter values into metadata:

```text
scenario = "concurrency"
concurrency_plan = "featherless_premium"
concurrency_count = 1
artifact_label = "concurrency__plan=featherless_premium__n=1"
```

Each §1.3 run contains:

```text
1 aggregate S_C provider representing n subscriptions/accounts of the selected plan
the same S_A fallback provider set shared by every n in the sweep
```

The aggregate S_C capacity is weighted capacity:

```text
concurrency_capacity = plan.concurrency_allotment * n
request_model_class = resolve_model_class(request.model, plan.model_class_overrides, plan.default_model_class)
request_concurrency_cost = plan.model_concurrency_costs_by_class[request_model_class]
used_concurrency_cost = sum(active_request.concurrency_cost)
available iff used_concurrency_cost + request_concurrency_cost <= concurrency_capacity
```

This is the main modeling point. Featherless `concurrency_allotment=4` does
not mean four simultaneous requests for every model. A `70B` request with
`concurrency_cost=4` fills one Premium subscription by itself. Four
simultaneous cost-1 small-model requests are also feasible. Mixed requests
are feasible only while their summed `concurrency_cost` stays within the
allotment.

Routing should use the same piecewise effective-cost semantics as the paper:

```text
c_eff(S_A, r) = API token cost
c_eff(S_Q, r, z) = ψ(z)                    if quota is available, else ∞
c_eff(S_C, r, u_weighted) = λ(u_weighted) * concurrency_cost(r.model)
                  if weighted capacity is available, else ∞

u_weighted = used_concurrency_cost / concurrency_capacity
```

These terms are alternatives across providers, not additive penalties on one
provider. In a concurrency-only §1.3 run, RouteWise compares the S_C
effective cost against the S_A API cost. In later joint end-to-end runs, S_Q
and S_C are still separate candidate providers; do not compute
`API cost + quota shadow price + concurrency shadow price`.

Do not multiply the online S_C effective cost by a predicted request
duration. The simulator still needs an observed or sampled service time to
release capacity at the correct finish time, and offline scheduling still has
processing times. Those are state-evolution facts, not an online
effective-cost factor.

The implementation belongs in `rwsim`, not in `experiments/simulation`.
`rwsim/world/capacity.py` should own a weighted concurrency state with:

```text
capacity_units
used_concurrency_cost
active requests keyed by finish time
admit(request_model, finish_time) -> bool
release_finished(current_time)
```

`experiments/simulation` should only resolve the plan, construct a
`TieredProvider(tier=S_C, concurrency=...)`, and launch the sweep. The
section runner should reject a concurrency plan if it lacks
`concurrency_allotment` or if the resolved class has no `concurrency_cost`.
Unknown trace model IDs should resolve to `default_model_class` with warning
metadata; genuinely incompatible model classes should be rejected from S_C.

For the first §1.3 implementation, keep the cost-layer S_C provider
zero-queue / immediate-admission only. Queueing policy belongs to later
end-to-end experiments where SLO and latency behavior are part of the paper
question.


---

## 6. Metrics and Artifacts

### 6.0 Metric ownership

Keep the layers separate:

```text
PerRequestRecord  -> per-request engine facts, no fixed subscription fee
Run               -> engine aggregate over records, no fixed subscription fee
SectionSummary    -> paper-facing row, includes fixed subscription fee
```

`subscription_fixed_cost_usd` is not a property of one request; it is a
paper/section aggregate for a purchased plan over a trace span. Therefore it
should live in the section summary/artifact layer, not inside every
`PerRequestRecord`.

### 6.1 Required cost fields

The section output should expose separate cost components:

```text
api_cost_usd
subscription_fixed_cost_usd
total_cost_usd
subscription_cost_known
trace_paper_grade
quota_saturated_in_trace
```

Rules:

- `api_cost_usd` is the sum of on-demand S_A API request costs.
- `subscription_fixed_cost_usd` is the prorated fixed fee.
- `total_cost_usd` is only paper-claimable when every active plan has
  `cost_claim_allowed: true`.
- `subscription_cost_known=false` for future plans whose monthly fee is not
  confirmed.
- `trace_paper_grade=false` for short smoke runs whose trace span is too
  small to make fixed-fee conclusions.
- `quota_saturated_in_trace=true` means the run should be excluded from main
  paper q-sweep plots because quota scarcity was not exercised.

For concurrency-plan runs, add:

```text
concurrency_capacity_units       # integer weighted capacity units
peak_used_concurrency_cost       # integer weighted capacity units
mean_concurrency_utilization     # ratio in [0, 1]
concurrency_saturated_in_trace   # boolean
```

All concurrency metrics are based on weighted capacity units, not raw
in-flight request count.

### 6.2 Existing `cost_usd` field

The engine-level `Run` may keep its current internal `cost_usd` field. The
paper-facing section summary should not expose an ambiguous `cost_usd` column
for subscription-plan outputs.

```text
engine Run.cost_usd:
  internal simulator API cost

section summary:
  api_cost_usd
  subscription_fixed_cost_usd
  total_cost_usd
```

Do not silently redefine `cost_usd` to include fixed fees. If an existing
section summary currently writes `cost_usd`, the subscription-plan migration
should be schema-breaking for that artifact: rename it to `api_cost_usd`
and regenerate the relevant goldens.

### 6.3 Trace duration

The fixed fee must be prorated by the workload span:

```text
trace_days = (last_request_timestamp - first_request_timestamp) / 86400
```

For smoke runs using `--max-requests`, report the actual smoke trace span.
Do not pretend a 100k laptop smoke is a full one-month experiment.

Full paper runs should run the full BurstGPT month on the larger machines
(`gpu1` / `gpu2`) and report the full trace span.

The full monthly fee is paid in the real world. The simulator prorates by
`trace_days / billing_period_days` only to make different trace spans
comparable. A full 30-day BurstGPT run pays the full monthly fee; a 5-day
smoke reports a 5/30 prorated cost and should set `trace_paper_grade=false`.

Suggested paper-grade check:

```text
trace_paper_grade =
  trace_days >= 5 * min(quota_window.quota_window_sec for quota_window in plan.quota_windows) / 86400
```

The factor `5` is a heuristic: at least five reset windows lets the shadow
price traverse multiple capacity cycles instead of reporting a single-window
startup artifact. This is a minimum threshold only. Main paper numbers
should still use the full BurstGPT month.

Smoke runs shorter than a plan's quota window see that window's full quota as
immediately available. The simulator does not model within-window rate
pacing. Smoke is for routing-decision sanity, not for capacity-pacing claims.

---

## 7. Offline Baseline

Offline is still a routing baseline, not a magical no-subscription world.

For scenarios that include subscription plans:

- Offline may know the full workload and select which requests use quota.
- Offline must pay the same fixed subscription fee as the online policies
  for the same `q` scenario.
- Offline should therefore be compared on:
  - quota allocation quality
  - API fallback cost
  - total cost including the fixed plan fee

This keeps the comparison fair:

```text
same purchased capacity, different routing/allocation policy
```

The fixed fee is paid regardless of utilization. Do not multiply the
subscription fee by "fraction of quota used"; under-using a subscription is
exactly what the purchase-count sweep is supposed to penalize.

Future joint scenarios may compose multiple subscription plans. In that
case, compute each plan's fixed fee independently and sum:

```text
subscription_fixed_cost_usd =
  Σ_plan monthly_fee_usd(plan) * count(plan) * (trace_days / billing_period_days)
```

---

## 8. Implementation Plan

### Phase 0 — Prerequisite: workload-level cost envelope

Do not implement §1.2 on top of per-request `L/U` calibration.

RouteWise must receive a workload-level cost envelope:

```text
L, U = P10/P90 of cheapest-API request cost over the workload
```

and use the same `L/U` for all requests in the same run. This is required
for the quota shadow price `ψ(z)` to mean "scarcity of quota" rather than
"cost scale of the current request." This is the effective-cost fix: replace
per-request `L/U` with workload-level `(P10, P90)` of cheapest-API request
cost over the entire workload; every request in one run sees the same
envelope.

### Phase 1 — Plan config and provider construction

1. Add `experiments/subscription_plans.yaml`.
2. Add a shared loader, for example:

   ```python
   def load_subscription_plans(path: Path | None = None) -> dict[str, SubscriptionPlan]:
       ...
   ```

3. Add a shared dataclass in `experiments/subscriptions.py`:

   ```python
   @dataclass(frozen=True)
   class SubscriptionPlan:
       plan_id: str
       display_name: str
       tier: Literal["quota", "concurrency"]
       monthly_fee_usd: float | None
       quota_windows: tuple[QuotaWindow, ...]
       concurrency_allotment: int | None
       model_concurrency_costs_by_class: Mapping[str, int]
       default_model_class: str | None
       model_class_overrides: Mapping[str, str]
       subscription_counts: tuple[int, ...]
       eligible_sections: tuple[str, ...]
       cost_claim_allowed: bool
       source: str
       notes: str = ""
   ```

4. Update `make_quota_provider()` to optionally accept a plan:

   ```python
   def make_quota_provider(
       name: str,
       *,
       quota_size: int | None = None,
       plan: SubscriptionPlan | None = None,
       subscription_count: int = 1,
       ...
   ) -> TieredProvider:
       ...
   ```

If both `plan` and `quota_size` are passed, raise `ValueError`. `quota_size`
is a legacy/manual path for non-plan experiments; plan-backed §1.2 runs must
derive quota size from the selected `SubscriptionPlan`.

At this phase, `TieredProvider` can still remain generic. Plan facts may
live in provider metadata or in section-level scenario metadata. Do not add
Chutes-specific fields to `rwsim` unless there is a generic need.

`subscription_count` should aggregate capacity into one provider:

```text
single-window plan:
  QuotaState(size = subscription_count * quota_window.quota_requests)

multi-window plan:
  one aggregate quota counter per quota window
```

It is not a provider index.

### Phase 2 — Parameterized quota scenario

Update `cost_layer.py`:

```python
def _make_quota_scenario_for_plan(plan_id: str, subscription_count: int) -> ScenarioConfig:
    plan = load_subscription_plans()[plan_id]
    providers = [
        make_quota_provider(
            f"{plan_id}_quota",
            plan=plan,
            subscription_count=subscription_count,
        )
    ]
    ...
```

Add section CLI flags:

```text
--scenario quota
--subscription-plan chutes
--subscription-plans chutes,minimax_subscription_plus
--subscription-count 2
--subscription-counts 1,2,3,4,5,6,8
```

The CLI expands these parameter lists internally into run cells. It should
not expose a generated scenario catalogue. Requested counts must be members
of `plan.subscription_counts`; out-of-range values raise an error that shows
the allowed set.

When this path lands, delete the generic public `cost_layer_quota_q1..q4`
scenarios in the same PR. Do not keep a legacy public alias unless a golden
migration forces a short-lived internal test-only path.

### Phase 3 — Cost summary fields

Add fixed-fee accounting in the section summary path.

The preferred implementation point is `experiments/simulation/common.py`
because every section runner already flows through it. The function should
take:

```text
scenario
run parameters (subscription_plan, subscription_count)
policy
seed
trace_start_ts
trace_end_ts
Run summary
```

and append:

```text
api_cost_usd
subscription_fixed_cost_usd
total_cost_usd
subscription_cost_known
trace_paper_grade
quota_saturated_in_trace
```

This should be done once in the shared section runner, not separately in
every policy.

### Phase 4 — Smoke and full runs

Laptop smoke:

```bash
uv run routewise simulator cost-layer \
  --scenario quota \
  --subscription-plan chutes \
  --subscription-counts 1,2,4 \
  --policy greedy_cost \
  --policy offline \
  --policy ablation_lp_only_p0 \
  --policy ablation_lp_only_p25 \
  --policy ablation_lp_only_p50 \
  --workload burstgpt \
  --max-requests 100000 \
  --seed 42 \
  --jobs 8 \
  --output-dir outputs/simulation/cost_layer_1_2_chutes_smoke
```

Full paper run:

```text
Run on gpu1/gpu2, full BurstGPT month, no --max-requests.
```

The full run should be launched only after the smoke shows:

- quota allocation favors high-value requests
- `api_cost_usd` decreases as subscription count increases
- `subscription_fixed_cost_usd` increases as subscription count increases
- `total_cost_usd = api_cost_usd + subscription_fixed_cost_usd`
- offline pays the same fixed fee as RouteWise for the same `q`

### Phase 5 — Parameterized concurrency scenario (§1.3)

Add §1.3 after §1.2 has landed:

1. Extend `experiments/subscription_plans.yaml` and `experiments/subscriptions.py`
   with a narrow `featherless_premium` concurrency plan: `concurrency_allotment`,
   `model_concurrency_costs_by_class`, `default_model_class`, and optional
   `model_class_overrides`.
2. Add weighted concurrency state to `rwsim/world/capacity.py`; do not
   implement Featherless-specific logic in experiment scripts.
3. Add a provider builder in `experiments/simulation/common.py` that turns a
   concurrency plan/count into one aggregate S_C provider.
4. Add `--scenario concurrency`, `--concurrency-plan`, and
   `--concurrency-count(s)` to `experiments/simulation/cost_layer.py`.
5. Delete the old public `cost_layer_concurrency_c1..c4` scenarios in the
   same PR. If a golden migration needs an alias, keep it internal and
   test-only.
6. Keep fixed-fee accounting at the section-summary layer, exactly as for
   quota plans.

For the first paper-grade smoke, it is acceptable to map every §1.3 workload
model to the conservative `ge_70b` class. Extending the class resolver to a
large provider-specific model catalogue is a follow-up, not a blocker for the
initial Featherless Premium result.

---

## 9. Expected Results

For Chutes:

- Increasing `q` should reduce API fallback cost.
- Increasing `q` also increases fixed subscription cost.
- Therefore total cost should have an optimum, not necessarily at `q4`.
- RouteWise should allocate scarce quota to larger/high-value requests more
  strongly than greedy quota-first.
- Offline should be a lower bound under the same purchased capacity.

For MiniMax Starter / Plus / Max:

- The monthly fees and quotas are known from the pricing screenshots, so they
  can enter the same cost-layer sweep as Chutes.
- Starter and Plus can use `[1, 2, 3, 4]`; Max starts with `[1, 2]` because its
  per-window quota is much larger and may saturate the workload at higher
  counts.
- Any tier/count with `quota_saturated_in_trace=true` should be excluded from
  the main q-sweep plot.

For Featherless-style concurrency:

- Increasing `n` should reduce API fallback caused by saturated S_C capacity.
- Increasing `n` also increases fixed subscription cost.
- Total cost should therefore have an optimum, not necessarily at the
  largest `n`.
- Utilization must be reported in weighted capacity units. For example, a
  single in-flight 70B request at cost 4 is 100% utilization of one Premium
  subscription, not 25%.

---

## 10. Selection Rule for Later Experiments

The May-4 discussion establishes an important workflow:

```text
cost-layer §1.2 finds the subscription plan/count;
cost-layer §1.3 finds the concurrency plan/count;
later end-to-end experiments reuse that selected setting.
```

So §1.2 has two jobs:

1. report the cost-layer subscription-count sweep itself; and
2. choose a canonical `subscription_plan`, `subscription_count`, and workload
   window for later sections.

§1.3 follows the same workflow for concurrency plans.

### 10.1 What gets selected

The selected setting should be recorded as metadata, not kept in someone's
head:

```yaml
selected_subscription_setting:
  source_experiment: cost_layer_quota
  workload: burstgpt
  workload_window: full_month
  subscription_plan: chutes
  subscription_count: 2
  selection_metric: total_cost_usd
  tie_breaks: [smaller_subscription_count, higher_quota_utilization]

selected_concurrency_setting:
  source_experiment: cost_layer_concurrency
  workload: burstgpt
  workload_window: full_month
  concurrency_plan: featherless_premium
  concurrency_count: 1
  selection_metric: total_cost_usd
  tie_breaks: [smaller_concurrency_count, higher_weighted_utilization]
```

This can live in the cost-layer output metadata first. If it becomes stable,
copy it into a small checked-in config for end-to-end runs.

### 10.2 Selection metric

For plans with known monthly fees, choose:

```text
argmin_n total_cost_usd(plan, n)
```

subject to:

- `subscription_cost_known = true`
- `trace_paper_grade = true`
- the run does not saturate quota for the whole workload
- RouteWise actually allocates quota toward higher-value requests

For §1.3, replace the quota-specific constraints with:

- weighted concurrency capacity is exercised
- RouteWise spills to API when S_C is saturated or incompatible
- utilization is computed from `used_concurrency_cost / concurrency_capacity`

Tie-breaks:

1. choose the smaller `n`
2. choose the setting with higher quota utilization for §1.2 or higher
   weighted concurrency utilization for §1.3
3. choose the provider whose latency distribution is usable for end-to-end

For any future plan with unknown monthly fee, do not select it as the main
dollar-cost setting. It can still be reported as an allocation/utilization
robustness run.

For `cost_claim_allowed=false` quota plans, use:

```text
selection metric = argmax_n quota_utilization
```

subject to:

- `quota_saturated_in_trace = false`
- the run still reports `api_cost_usd`
- no `total_cost_usd` claim appears in paper text

For `cost_claim_allowed=false` concurrency plans, use:

```text
selection metric = argmax_n mean_concurrency_utilization
```

subject to the same no-dollar-claim rule.

### 10.3 Workload window

The default paper-grade setting is the full one-month workload. That is the
cleanest answer for §1.2 because subscription economics depend on enough
requests.

The May-4 discussion also suggested looking for a period where one
subscription is enough, to keep the later end-to-end story simple. This is
allowed only if the selection rule is explicit. Do not hand-pick a convenient
day after seeing the result.

Acceptable window rules:

```text
full_month
representative_7d_window = median-request-volume contiguous 7-day window
representative_1d_window = median-request-volume day
```

Tie-break for representative windows: if multiple windows have the same
median-volume distance, pick the earliest start timestamp.

For the main cost-layer claim, prefer `full_month`. For end-to-end, a
representative smaller window is acceptable if full-month simulation is too
large or if the one-subscription story would otherwise be buried by scale.
The chosen window must be written into metadata.

### 10.4 How later sections use it

End-to-end should not reopen the q-sweep unless the paper question is
explicitly about purchase count. It should import the selected setting:

```text
selected quota plan/count from §1.2
selected concurrency plan/count from §1.3
fixed RW3/RW8 on-demand pools
```

This keeps the end-to-end experiment focused on the full RouteWise routing
stack rather than re-running the subscription purchase search.

---

## 11. Tests

### 11.1 Unit tests

Add tests for:

- YAML loader rejects missing quota size.
- YAML loader rejects `cost_claim_allowed=true` with `monthly_fee_usd=null`.
- Chutes plan loads as `$20/month`, `5000/day`.
- `make_quota_provider(plan=chutes, subscription_count=2)` produces
  `QuotaState(size=10000, window_sec=86400)`.
- A saturated count emits a warning / metadata flag rather than silently
  entering the paper q-sweep.
- Featherless Premium loads as `$25/month`, `concurrency_allotment=4`.
- `make_concurrency_provider(plan=featherless_premium, concurrency_count=2)`
  produces weighted capacity `8`.
- A model with `concurrency_cost=4` consumes all capacity of one Premium
  subscription.
- A model with `concurrency_cost=1` can admit four simultaneous requests on
  one Premium subscription.
- An unknown trace model ID resolves to `default_model_class` and emits
  warning metadata instead of silently falling out of S_C.
- A request with a genuinely incompatible resolved class is not admitted to S_C.

### 11.2 Cost accounting tests

Use a tiny synthetic workload with known timestamps:

```text
trace span = 1 day
monthly_fee_usd = $30
q = 2
billing_period_days = 30
expected fixed fee =
  monthly_fee_usd * q * (trace_days / billing_period_days)
  = 30 * 2 * (1 / 30) = $2
```

Assert:

```text
subscription_fixed_cost_usd == 2.0
total_cost_usd == api_cost_usd + 2.0
```

Also assert offline pays the same fixed fee in the same scenario.

Repeat the same fixed-fee accounting test for a concurrency plan:

```text
monthly_fee_usd = $30
concurrency_count = 2
trace span = 1 day
expected fixed fee = 30 * 2 * (1 / 30) = $2
```

### 11.3 Behavior smoke

For Chutes `q1`, 100k BurstGPT smoke:

```text
mean cheapest-API cost of quota-routed requests >
mean cheapest-API cost of API-routed requests
```

This protects the main paper claim: scarce quota is reserved for higher
value requests.

For Featherless Premium `n=1`, synthetic behavior smoke:

```text
concurrency_allotment = 4
one active request with concurrency_cost=4 blocks another cost-1 request
four active requests with concurrency_cost=1 block the fifth cost-1 request
after finish_time passes, capacity is released
```

This protects the main §1.3 claim: concurrency is weighted capacity, not raw
request count.

---

## 12. Out of Scope

Do not include these in the first implementation:

| Item | Reason |
|---|---|
| Per-token subscription quotas | Current paper plan uses request quotas. Different unit requires a new formula. |
| Buying decision optimizer | `q1..q4` sweep is the optimizer for now. |
| Chutes live calls | This is simulator-only. |
| Featherless live calls | §1.3 is simulator-only. Real evaluation can use the selected setting later. |
| S_C queueing policy | First §1.3 is immediate-admission only; queueing belongs to end-to-end SLO experiments. |
| MiniMax high-speed / Ultra plans | Excluded from the first §1.2 sweep. They likely require separate latency profiles and would mix pricing with model-speed changes. |
| Full end-to-end joint setup | Separate section after cost-layer §1.2 and §1.3 are stable. |
| Joint S_Q + S_C purchase optimizer | §1.2 and §1.3 select quota and concurrency settings independently first. |
| Tier-upgrade pricing | `q`/`n` means multiple independent accounts/subscriptions, not upgrading to a higher plan tier. |

---

## 13. Open Questions

1. For later end-to-end, do we use the full-month selected setting or a
   representative smaller window selected by a predeclared rule?

2. The old offline-stage config contains `featherless_scale` with
   `monthly_fee_usd=$75`, capacity `8`, and `concurrency_cost=2` for 70B-class
   models. Current Featherless docs expose Premium plus newer business/agentic
   plans, while the concurrency docs still describe an 8-unit scale-style
   allotment. The concurrency docs also keep 70B-class model cost at `4`; the
   "2 simultaneous 70B requests" result comes from `capacity=8 / cost=4`, not
   from changing the model cost to `2`. Before making §1.3 paper dollar claims
   for an 8-unit plan, reconcile the plan id, monthly fee, and compatibility
   against current official pricing or a saved dashboard screenshot.

The only blocker for Chutes and MiniMax Starter / Plus / Max implementation
is composite quota support for MiniMax's 5-hour + weekly allowance. Chutes can
run with the existing single-window quota state.
